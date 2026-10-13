from pathlib import Path
from typing import List, Dict, Any, Optional
import pydicom
from fastapi import APIRouter, HTTPException, Query

from backend.app.core.config import settings
from backend.app.queue.task_worker import ANALYSIS_REGISTRY, save_registry_to_disk
from backend.app.models.triage_schemas import StudyAnalysisResult
from backend.app.inference.triage_2d import triage_2d_engine
from backend.app.inference.adversarial_test import AdversarialAttackTester

router = APIRouter(prefix="/triage", tags=["Clinical Worklist Triage & Adversarial Evaluation"])

@router.get("/worklist")
async def get_prioritized_worklist():
    """
    Returns clinical radiology worklist dynamically organized by AI Urgency Score.
    STAT acute cases (urgency >= 0.70) are prioritized at the top (< 30s turnaround).
    """
    worklist_items = []
    
    for study_uid, analysis in ANALYSIS_REGISTRY.items():
        # Get slice count
        study_dir = settings.DICOM_CLEAN_DIR / study_uid
        slice_count = len(list(study_dir.glob("*.dcm"))) if study_dir.exists() else 1

        top_findings = [p.name for p in analysis.acute_pathologies if p.probability > 0.40]
        
        worklist_items.append({
            "study_instance_uid": study_uid,
            "pseudo_patient_id": analysis.pseudo_patient_id,
            "modality": analysis.modality,
            "urgency_score": analysis.urgency_score,
            "triage_category": analysis.triage_category,
            "critical_alert_triggered": analysis.critical_alert_triggered,
            "time_to_triage_seconds": analysis.time_to_triage_seconds,
            "slice_count": slice_count,
            "top_findings": top_findings or ["No acute pathology"],
            "analyzed_at": analysis.analyzed_at,
            "status": analysis.status
        })

    # Sort descending by urgency score: STAT (>= 0.70) -> URGENT -> ROUTINE
    sorted_worklist = sorted(worklist_items, key=lambda x: x["urgency_score"], reverse=True)
    return {
        "total_studies": len(sorted_worklist),
        "stat_critical_count": sum(1 for x in sorted_worklist if x["triage_category"] == "STAT"),
        "urgent_count": sum(1 for x in sorted_worklist if x["triage_category"] == "URGENT"),
        "routine_count": sum(1 for x in sorted_worklist if x["triage_category"] == "ROUTINE"),
        "worklist": sorted_worklist
    }

@router.get("/studies/{study_uid}", response_model=StudyAnalysisResult)
async def get_study_analysis(study_uid: str):
    """
    Retrieves the complete diagnostic analysis package for a specific study.
    """
    if study_uid not in ANALYSIS_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Study analysis for {study_uid} not found. Please upload or triage first.")
    return ANALYSIS_REGISTRY[study_uid]

@router.post("/studies/{study_uid}/adversarial-eval")
async def evaluate_adversarial_security(study_uid: str, epsilon: float = Query(0.05, ge=0.01, le=0.20)):
    """
    Executes on-demand white-box adversarial gradient attack (FGSM) and
    evaluates defensive Feature Squeezing restoration.
    """
    clean_dir = settings.DICOM_CLEAN_DIR / study_uid
    if not clean_dir.exists():
        raise HTTPException(status_code=404, detail="Study DICOM files not found.")

    dcm_files = sorted(list(clean_dir.glob("*.dcm")), key=lambda p: p.name)
    if not dcm_files:
        raise HTTPException(status_code=404, detail="No DICOM instances found.")

    # Read mid-slice
    mid_ds = pydicom.dcmread(str(dcm_files[len(dcm_files) // 2]))
    pixel_arr = mid_ds.pixel_array

    # Preprocess into tensor
    input_tensor = triage_2d_engine.preprocess_image(pixel_arr)

    # Run adversarial test on ResNet model
    eval_metrics = AdversarialAttackTester.evaluate_defense(
        model=triage_2d_engine.model,
        image=input_tensor,
        target_class=0,
        epsilon=epsilon
    )

    return {
        "study_instance_uid": study_uid,
        "attack_method": "Fast Gradient Sign Method (FGSM)",
        "perturbation_epsilon": epsilon,
        "defensive_mechanisms": ["Bit-Depth Reduction (16-bit to 8-bit)", "Spatial Gaussian Smoothing"],
        "metrics": eval_metrics
    }
