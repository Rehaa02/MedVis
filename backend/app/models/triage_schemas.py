from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class PathologyFinding(BaseModel):
    name: str
    probability: float
    is_acute: bool
    severity: Literal["none", "mild", "moderate", "severe", "critical"] = "none"
    recommended_action: Optional[str] = None

class BoundingBox(BaseModel):
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    slice_index: int = 0
    confidence: float

class OrganVolume(BaseModel):
    organ_name: str
    volume_cm3: float
    normal_range_cm3: str
    is_abnormal: bool
    status: Literal["normal", "atrophy", "hypertrophy", "mass_detected"] = "normal"

class TokenHallucinationDetail(BaseModel):
    token: str
    entropy: float
    risk_level: Literal["grounded", "moderate_uncertainty", "high_risk_hallucination"]
    reason: Optional[str] = None

class HallucinationAudit(BaseModel):
    overall_hallucination_score: float  # 0.0 (fully grounded) to 1.0 (hallucinated)
    rextrust_mean_entropy: float
    high_entropy_token_count: int
    radflag_consistency_score: float
    dear_anatomical_grounding_passed: bool
    tokens_audit: List[TokenHallucinationDetail] = []

class RadiologyReportDraft(BaseModel):
    clinical_indication: str
    technique: str
    comparison: str = "None available for this study"
    findings: str
    impression: str
    key_findings_list: List[str] = []
    generated_by_model: str
    hallucination_audit: HallucinationAudit

class StudyAnalysisResult(BaseModel):
    study_instance_uid: str
    pseudo_patient_id: str
    modality: str
    urgency_score: float  # 0.0 to 1.0
    triage_category: Literal["STAT", "URGENT", "ROUTINE"]
    time_to_triage_seconds: float
    critical_alert_triggered: bool
    acute_pathologies: List[PathologyFinding] = []
    bounding_boxes: List[BoundingBox] = []
    organ_volumes: List[OrganVolume] = []
    report_draft: RadiologyReportDraft
    defensive_squeezing_applied: bool = True
    model_versions: Dict[str, str]
    analyzed_at: str
    status: Literal["PENDING_TRIAGE", "TRIAGED", "PENDING_REVIEW", "REVIEWED"] = "TRIAGED"
