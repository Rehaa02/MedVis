import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pydicom

from backend.app.core.config import settings
from backend.app.storage.minio_client import storage_manager
from backend.app.models.triage_schemas import StudyAnalysisResult, PathologyFinding, BoundingBox, OrganVolume
from backend.app.models.fhir_schemas import ClinicalContextPayload
from backend.app.inference.triage_2d import triage_2d_engine
from backend.app.inference.segmentation_3d import segmentation_3d_engine
from backend.app.reporting.llm_engine import clinical_reporting_llm

# In-memory and persisted analysis registry
ANALYSIS_REGISTRY: Dict[str, StudyAnalysisResult] = {}
REGISTRY_FILE = settings.DATA_DIR / "analysis_results.json"

def save_registry_to_disk():
    data = {k: v.model_dump() for k, v in ANALYSIS_REGISTRY.items()}
    with open(REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_registry_from_disk():
    global ANALYSIS_REGISTRY
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                raw = json.load(f)
                for k, v in raw.items():
                    ANALYSIS_REGISTRY[k] = StudyAnalysisResult(**v)
        except Exception as e:
            print(f"[TaskWorker] Error loading cached registry: {e}")

load_registry_from_disk()

class StudyInferencePipeline:
    """
    Asynchronous Inference Pipeline Orchestrator.
    Ingests de-identified DICOM files, coordinates 2D triage, 3D SwinUNETR segmentation,
    and LLM reporting with hallucination guards.
    """

    @classmethod
    async def process_study_async(
        cls,
        study_instance_uid: str,
        pseudo_patient_id: str,
        dicom_file_paths: list[str],
        clinical_context: Optional[ClinicalContextPayload] = None
    ) -> StudyAnalysisResult:
        """
        Processes single-image or multi-slice DICOM studies asynchronously.
        """
        print(f"[TaskWorker] Processing study {study_instance_uid} with {len(dicom_file_paths)} instance(s)...")
        start_time = time.time()

        # 1. Load DICOM datasets
        datasets = []
        for fp in dicom_file_paths:
            try:
                ds = pydicom.dcmread(fp)
                datasets.append(ds)
            except Exception as e:
                print(f"[TaskWorker] Error reading DICOM {fp}: {e}")

        if not datasets:
            raise ValueError(f"No valid DICOM datasets found for study {study_instance_uid}")

        # Determine modality
        modality = str(datasets[0].get("Modality", "CT"))
        is_3d = len(datasets) > 1 or (hasattr(datasets[0], "pixel_array") and datasets[0].pixel_array.ndim == 3)

        pixel_spacing = list(datasets[0].get("PixelSpacing", [1.0, 1.0]))
        slice_thickness = float(datasets[0].get("SliceThickness", 2.5))

        # 2. Run 2D Triage (Sub-30s triage path)
        if is_3d:
            # Sort slices by InstanceNumber or SliceLocation
            sorted_ds = sorted(datasets, key=lambda x: float(x.get("SliceLocation", x.get("InstanceNumber", 0))))
            slices = [d.pixel_array for d in sorted_ds if hasattr(d, "pixel_array")]
            volume_3d = np.stack(slices, axis=0)  # (D, H, W)
            
            # Use mid-volume slice for rapid 2D triage
            mid_slice = slices[len(slices) // 2]
            triage_res = triage_2d_engine.classify_and_triage(mid_slice, {"modality": modality})
            
            # 3. Run MONAI SwinUNETR 3D Volumetric Segmentation
            seg_res = segmentation_3d_engine.segment_and_quantify(
                volume_array=volume_3d,
                pixel_spacing=pixel_spacing,
                slice_thickness=slice_thickness
            )
            organ_volumes = seg_res["organ_volumes"]
        else:
            # 2D Radiograph
            ds = datasets[0]
            pixel_arr = ds.pixel_array
            triage_res = triage_2d_engine.classify_and_triage(pixel_arr, {"modality": modality})
            organ_volumes = []

        # 4. Multimodal LLM Report Generation with Hallucination Mitigation
        report_draft = clinical_reporting_llm.generate_draft_report(
            modality=modality,
            pathologies=triage_res["pathology_findings"],
            organ_volumes=organ_volumes,
            clinical_context=clinical_context
        )

        total_time = round(time.time() - start_time, 3)

        # 5. Assemble Comprehensive Result
        result = StudyAnalysisResult(
            study_instance_uid=study_instance_uid,
            pseudo_patient_id=pseudo_patient_id,
            modality=modality,
            urgency_score=triage_res["urgency_score"],
            triage_category=triage_res["triage_category"],
            time_to_triage_seconds=triage_res["time_to_triage_seconds"],
            critical_alert_triggered=triage_res["critical_alert_triggered"],
            acute_pathologies=triage_res["pathology_findings"],
            bounding_boxes=triage_res["bounding_boxes"],
            organ_volumes=organ_volumes,
            report_draft=report_draft,
            defensive_squeezing_applied=True,
            model_versions={
                "triage_2d": triage_res["model_version"],
                "segmentation_3d": seg_res["model_version"] if is_3d else "N/A",
                "reporting_llm": report_draft.generated_by_model
            },
            analyzed_at=datetime.utcnow().isoformat() + "Z",
            status="TRIAGED"
        )

        ANALYSIS_REGISTRY[study_instance_uid] = result
        save_registry_to_disk()
        print(f"[TaskWorker] Finished study {study_instance_uid} in {total_time}s. Category: {result.triage_category}, Urgency: {result.urgency_score}")
        return result

# Celery Task Definition (for distributed production worker setup)
if settings.USE_CELERY:
    try:
        from celery import Celery
        celery_app = Celery("medical_tasks", broker=settings.REDIS_URL)
        
        @celery_app.task(name="process_study_task")
        def process_study_celery(study_instance_uid: str, pseudo_patient_id: str, dicom_file_paths: list[str]):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                StudyInferencePipeline.process_study_async(study_instance_uid, pseudo_patient_id, dicom_file_paths)
            )
    except Exception as e:
        print(f"[TaskWorker] Celery initialization warning: {e}")
