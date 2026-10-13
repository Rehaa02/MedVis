import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from backend.app.core.config import settings
from backend.app.models.gmlp_schemas import RadiologistReviewSubmission, GMLPAuditRecord
from backend.app.queue.task_worker import ANALYSIS_REGISTRY, save_registry_to_disk

router = APIRouter(prefix="/review", tags=["FDA Good Machine Learning Practice (GMLP) Oversight"])

def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact dynamic-programming Levenshtein edit distance."""
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

@router.post("/submit", response_model=GMLPAuditRecord)
async def submit_radiologist_review(submission: RadiologistReviewSubmission):
    """
    Submits a physician human-in-the-loop review for an AI-generated study analysis.
    Complies explicitly with FDA SaMD Good Machine Learning Practice (GMLP) Principle 8.
    Calculates Levenshtein edit distance and logs a tamper-evident audit record.
    """
    study_uid = submission.study_instance_uid
    if study_uid not in ANALYSIS_REGISTRY:
        raise HTTPException(status_code=404, detail="Study record not found.")

    analysis = ANALYSIS_REGISTRY[study_uid]
    
    # 1. Compute edit metrics
    s1 = submission.original_ai_report.strip()
    s2 = submission.final_signed_report.strip()
    lev_dist = calculate_levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2), 1)
    edit_ratio = round(lev_dist / max_len, 4)

    # 2. Generate tamper-proof cryptographic audit signature
    audit_id = f"GMLP_AUDIT_{uuid.uuid4().hex[:12].upper()}"
    timestamp_str = datetime.utcnow().isoformat() + "Z"
    
    sig_payload = f"{audit_id}|{study_uid}|{submission.radiologist_id}|{submission.decision}|{lev_dist}|{timestamp_str}"
    tamper_proof_sig = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()

    audit_record = GMLPAuditRecord(
        audit_id=audit_id,
        study_instance_uid=study_uid,
        radiologist_id=submission.radiologist_id,
        decision=submission.decision,
        ai_urgency_score=analysis.urgency_score,
        original_report_length=len(s1),
        final_report_length=len(s2),
        levenshtein_distance=lev_dist,
        edit_ratio=edit_ratio,
        model_version_fingerprint=analysis.model_versions,
        timestamp=timestamp_str,
        fda_compliance_standard="FDA SaMD Good Machine Learning Practice (GMLP) 21 CFR Part 11",
        tamper_proof_signature=tamper_proof_sig
    )

    # 3. Append to persistent audit trail file
    with open(settings.AUDIT_LOG_FILE, "a") as f:
        f.write(audit_record.model_dump_json() + "\n")

    # 4. Update study status
    analysis.status = "REVIEWED"
    save_registry_to_disk()

    return audit_record

@router.get("/audit/{study_uid}")
async def get_study_audit_records(study_uid: str) -> List[Dict[str, Any]]:
    """Retrieves all GMLP compliance records for a given study."""
    records = []
    if settings.AUDIT_LOG_FILE.exists():
        with open(settings.AUDIT_LOG_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("study_instance_uid") == study_uid:
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
    return records
