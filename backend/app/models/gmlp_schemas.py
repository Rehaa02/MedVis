from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class RadiologistReviewSubmission(BaseModel):
    study_instance_uid: str
    radiologist_id: str = "RAD_DR_SMITH_MD"
    decision: Literal["accepted", "modified", "rejected"]
    original_ai_report: str
    final_signed_report: str
    feedback_notes: Optional[str] = None
    agree_with_urgency_tier: bool = True
    suggested_urgency_tier: Optional[Literal["STAT", "URGENT", "ROUTINE"]] = None

class GMLPAuditRecord(BaseModel):
    audit_id: str
    study_instance_uid: str
    radiologist_id: str
    decision: str
    ai_urgency_score: float
    original_report_length: int
    final_report_length: int
    levenshtein_distance: int
    edit_ratio: float  # Levenshtein / max(len1, len2)
    model_version_fingerprint: Dict[str, str]
    timestamp: str
    fda_compliance_standard: str = "FDA SaMD Good Machine Learning Practice (GMLP) 21 CFR Part 11"
    tamper_proof_signature: str
