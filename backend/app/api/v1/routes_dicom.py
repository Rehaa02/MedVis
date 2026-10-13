import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
import pydicom
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Form

from backend.app.core.config import settings
from backend.app.core.dicom_deid import DICOMDeidentifier
from backend.app.storage.minio_client import storage_manager
from backend.app.models.dicom_schemas import DICOMUploadResponse
from backend.app.models.fhir_schemas import ClinicalContextPayload
from backend.app.queue.task_worker import StudyInferencePipeline

router = APIRouter(prefix="/dicom", tags=["DICOM Ingestion & De-identification"])

@router.post("/upload", response_model=DICOMUploadResponse)
async def upload_dicom_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    referring_indication: Optional[str] = Form("STAT Emergency Evaluation"),
    clinical_history: Optional[str] = Form("Acute onset symptoms")
):
    """
    Accepts one or more DICOM files, applies strict DICOM PS 3.15 Annex E de-identification,
    persists sanitized binaries to storage, and dispatches asynchronous AI inference.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No DICOM files provided.")

    temp_batch_id = str(uuid.uuid4())[:8]
    raw_dir = settings.DICOM_RAW_DIR / temp_batch_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file_paths = []
    study_instance_uid = None
    pseudo_patient_id = None

    try:
        for file in files:
            # Save raw upload temporarily
            raw_path = raw_dir / file.filename
            with open(raw_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Read DICOM dataset
            try:
                ds = pydicom.dcmread(raw_path)
            except Exception as e:
                continue

            study_uid = str(ds.get("StudyInstanceUID", uuid.uuid4().hex))
            if not study_instance_uid:
                study_instance_uid = study_uid

            # Apply DICOM PS 3.15 Annex E De-identification
            clean_ds, audit_record = DICOMDeidentifier.deidentify(ds)
            pseudo_patient_id = audit_record["pseudo_patient_id"]

            # Save sanitized DICOM
            clean_study_dir = settings.DICOM_CLEAN_DIR / study_instance_uid
            clean_study_dir.mkdir(parents=True, exist_ok=True)
            clean_path = clean_study_dir / f"{ds.SOPInstanceUID}.dcm"
            clean_ds.save_as(str(clean_path))
            cleaned_file_paths.append(str(clean_path))

            # Store to MinIO / S3 object store
            storage_manager.put_file(f"studies/{study_instance_uid}/{ds.SOPInstanceUID}.dcm", str(clean_path))

        # Cleanup raw directory
        shutil.rmtree(raw_dir, ignore_errors=True)

        if not cleaned_file_paths or not study_instance_uid:
            raise HTTPException(status_code=400, detail="Failed to parse valid DICOM files from upload.")

        clinical_ctx = ClinicalContextPayload(
            patient_id=pseudo_patient_id or "ANON_PATIENT",
            clinical_history=clinical_history,
            referring_indication=referring_indication
        )

        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Trigger asynchronous inference pipeline
        background_tasks.add_task(
            StudyInferencePipeline.process_study_async,
            study_instance_uid=study_instance_uid,
            pseudo_patient_id=pseudo_patient_id,
            dicom_file_paths=cleaned_file_paths,
            clinical_context=clinical_ctx
        )

        return DICOMUploadResponse(
            success=True,
            message="DICOM study uploaded, de-identified according to DICOM PS 3.15 Annex E, and queued for AI triage.",
            study_instance_uid=study_instance_uid,
            series_count=1,
            instance_count=len(cleaned_file_paths),
            pseudo_patient_id=pseudo_patient_id,
            task_id=task_id
        )

    except Exception as e:
        shutil.rmtree(raw_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"DICOM Ingestion Error: {str(e)}")

@router.post("/c-store")
async def simulate_c_store_push(
    background_tasks: BackgroundTasks,
    study_instance_uid: str = Form(...),
    pseudo_patient_id: str = Form(...)
):
    """
    Simulates a hospital PACS network C-STORE event pushing a new imaging series to the AI listener.
    """
    clean_study_dir = settings.DICOM_CLEAN_DIR / study_instance_uid
    if not clean_study_dir.exists():
        raise HTTPException(status_code=404, detail="Study directory not found.")

    dicom_files = [str(p) for p in clean_study_dir.glob("*.dcm")]
    if not dicom_files:
        raise HTTPException(status_code=404, detail="No DICOM files in study directory.")

    background_tasks.add_task(
        StudyInferencePipeline.process_study_async,
        study_instance_uid=study_instance_uid,
        pseudo_patient_id=pseudo_patient_id,
        dicom_file_paths=dicom_files
    )

    return {"status": "SUCCESS", "message": "C-STORE event ingested into Redis/Celery queue."}

@router.post("/seed-demo")
async def seed_demo_studies(background_tasks: BackgroundTasks):
    """
    Synthesizes and ingests 3 clinical demo studies:
    1. STAT Tension Pneumothorax (2D CXR)
    2. Multi-slice 3D Abdominal CT Volume (MONAI SwinUNETR)
    3. Routine Normal CXR
    """
    from scripts.generate_demo_data import generate_sample_cases
    demo_raw_dir = settings.DATA_DIR / "demo_dicom"
    generate_sample_cases(demo_raw_dir)

    ingested_count = 0
    # Ingest each case directory
    for case_dir in sorted(demo_raw_dir.iterdir()):
        if case_dir.is_dir():
            dcm_files = list(case_dir.glob("*.dcm"))
            if not dcm_files:
                continue
            
            # Read first to get study UID
            sample_ds = pydicom.dcmread(str(dcm_files[0]))
            orig_study_uid = str(sample_ds.StudyInstanceUID)
            
            cleaned_paths = []
            clean_study_dir = settings.DICOM_CLEAN_DIR / orig_study_uid
            clean_study_dir.mkdir(parents=True, exist_ok=True)
            
            last_pseudo_id = "ANON_PATIENT"
            for f in dcm_files:
                ds = pydicom.dcmread(str(f))
                clean_ds, audit = DICOMDeidentifier.deidentify(ds)
                last_pseudo_id = audit["pseudo_patient_id"]
                dest = clean_study_dir / f"{clean_ds.SOPInstanceUID}.dcm"
                clean_ds.save_as(str(dest))
                cleaned_paths.append(str(dest))
                storage_manager.put_file(f"studies/{orig_study_uid}/{clean_ds.SOPInstanceUID}.dcm", str(dest))

            # Synchronously or in background process study
            await StudyInferencePipeline.process_study_async(
                study_instance_uid=orig_study_uid,
                pseudo_patient_id=last_pseudo_id,
                dicom_file_paths=cleaned_paths
            )
            ingested_count += 1

    return {"status": "SUCCESS", "message": f"Successfully synthesized and triaged {ingested_count} clinical demo studies!"}

