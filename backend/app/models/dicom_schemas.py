from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DICOMInstanceMetadata(BaseModel):
    sop_instance_uid: str
    series_instance_uid: str
    study_instance_uid: str
    instance_number: Optional[int] = 1
    rows: int
    columns: int
    modality: str
    pixel_spacing: Optional[List[float]] = None
    slice_thickness: Optional[float] = None
    slice_location: Optional[float] = None
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_intercept: Optional[float] = 0.0
    rescale_slope: Optional[float] = 1.0
    photometric_interpretation: Optional[str] = "MONOCHROME2"
    file_path: str

class DICOMSeriesMetadata(BaseModel):
    series_instance_uid: str
    modality: str
    series_description: Optional[str] = "Standard Series"
    series_number: Optional[int] = 1
    number_of_instances: int
    instances: List[DICOMInstanceMetadata] = []

class DICOMStudyMetadata(BaseModel):
    study_instance_uid: str
    pseudo_patient_id: str
    pseudo_patient_name: str
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    accession_number: Optional[str] = None
    modality: str
    study_description: Optional[str] = "Diagnostic Imaging Study"
    series: List[DICOMSeriesMetadata] = []
    created_at: str
    deidentified: bool = True
    phi_removed: str = "YES"

class DICOMUploadResponse(BaseModel):
    success: bool
    message: str
    study_instance_uid: str
    series_count: int
    instance_count: int
    pseudo_patient_id: str
    task_id: str
