import io
from pathlib import Path
from typing import List, Optional
import pydicom
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, Response

from backend.app.core.config import settings

router = APIRouter(prefix="/dicomweb", tags=["DICOMweb (WADO-RS, QIDO-RS)"])

def render_dicom_slice_to_png(ds: pydicom.Dataset, window_center: Optional[float] = None, window_width: Optional[float] = None) -> bytes:
    """
    Renders a pydicom Dataset slice into a displayable 8-bit PNG buffer
    using linear VOI LUT / Window-Level transformations.
    """
    pixels = ds.pixel_array.astype(np.float32)
    
    # Apply Rescale Slope & Intercept for Hounsfield Units (HU) if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu_pixels = pixels * slope + intercept

    # Determine Window Center (WC) and Window Width (WW)
    wc = window_center if window_center is not None else float(getattr(ds, "WindowCenter", 40.0) if hasattr(ds, "WindowCenter") else (hu_pixels.max() + hu_pixels.min()) / 2.0)
    ww = window_width if window_width is not None else float(getattr(ds, "WindowWidth", 400.0) if hasattr(ds, "WindowWidth") else max(hu_pixels.max() - hu_pixels.min(), 1.0))

    if isinstance(wc, (list, pydicom.multival.MultiValue)):
        wc = float(wc[0])
    if isinstance(ww, (list, pydicom.multival.MultiValue)):
        ww = float(ww[0])

    # Linear VOI LUT mapping
    min_val = wc - 0.5 - (ww - 1.0) / 2.0
    max_val = wc - 0.5 + (ww - 1.0) / 2.0
    
    clipped = np.clip(hu_pixels, min_val, max_val)
    normalized = ((clipped - min_val) / max(max_val - min_val, 1e-6) * 255.0).astype(np.uint8)

    # Invert if PhotometricInterpretation is MONOCHROME1 (air is white, bone is black)
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        normalized = 255 - normalized

    img = Image.fromarray(normalized)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@router.get("/studies")
async def qido_search_studies():
    """
    QIDO-RS (Query based on ID for DICOM Objects):
    Lists available studies and series metadata.
    """
    studies_dir = settings.DICOM_CLEAN_DIR
    study_list = []
    
    for study_folder in studies_dir.iterdir():
        if study_folder.is_dir():
            dcm_files = list(study_folder.glob("*.dcm"))
            if dcm_files:
                sample_ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                study_list.append({
                    "0020000D": {"vr": "UI", "Value": [study_folder.name]},
                    "00100010": {"vr": "PN", "Value": [str(getattr(sample_ds, "PatientName", "ANON"))]},
                    "00100020": {"vr": "LO", "Value": [str(getattr(sample_ds, "PatientID", "ANON_ID"))]},
                    "00080060": {"vr": "CS", "Value": [str(getattr(sample_ds, "Modality", "CT"))]},
                    "00200010": {"vr": "SH", "Value": [str(getattr(sample_ds, "StudyID", "STUDY_1"))]},
                    "numberOfInstances": len(dcm_files)
                })
                
    return study_list

@router.get("/studies/{study_uid}/instances/{instance_uid}/frames/{frame_id}")
async def wado_retrieve_frame(study_uid: str, instance_uid: str, frame_id: int = 1):
    """
    WADO-RS (Web Access to DICOM Objects by RESTful Services):
    Retrieves rendered frame image as image/png.
    """
    clean_dir = settings.DICOM_CLEAN_DIR / study_uid
    dcm_path = clean_dir / f"{instance_uid}.dcm"
    if not dcm_path.exists():
        # Search by file match
        matches = list(clean_dir.glob("*.dcm"))
        if matches:
            dcm_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail="DICOM instance not found.")

    ds = pydicom.dcmread(str(dcm_path))
    png_bytes = render_dicom_slice_to_png(ds)
    return Response(content=png_bytes, media_type="image/png")

@router.get("/studies/{study_uid}/slice/{slice_index}")
async def retrieve_study_slice(study_uid: str, slice_index: int = 0, wc: Optional[float] = None, ww: Optional[float] = None):
    """
    High-performance endpoint rendering slice index for PACS web viewer.
    """
    clean_dir = settings.DICOM_CLEAN_DIR / study_uid
    if not clean_dir.exists():
        raise HTTPException(status_code=404, detail="Study not found.")

    dcm_files = sorted(list(clean_dir.glob("*.dcm")), key=lambda p: p.name)
    if not dcm_files:
        raise HTTPException(status_code=404, detail="No slices in study.")

    idx = max(0, min(slice_index, len(dcm_files) - 1))
    ds = pydicom.dcmread(str(dcm_files[idx]))
    png_bytes = render_dicom_slice_to_png(ds, window_center=wc, window_width=ww)
    return Response(content=png_bytes, media_type="image/png")
