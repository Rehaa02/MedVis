from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class FHIRCoding(BaseModel):
    system: str = "http://dicom.nema.org/resources/ontology/DCM"
    code: str
    display: str

class FHIRIdentifier(BaseModel):
    use: str = "official"
    system: str = "urn:dicom:uid"
    value: str

class FHIRReference(BaseModel):
    reference: str
    display: Optional[str] = None

class FHIRImagingStudySeriesInstance(BaseModel):
    uid: str
    sopClass: FHIRCoding
    number: Optional[int] = 1
    title: Optional[str] = None

class FHIRImagingStudySeries(BaseModel):
    uid: str
    number: Optional[int] = 1
    modality: FHIRCoding
    description: Optional[str] = None
    numberOfInstances: int
    endpoint: List[FHIRReference] = []
    instance: List[FHIRImagingStudySeriesInstance] = []

class FHIRImagingStudy(BaseModel):
    resourceType: str = "ImagingStudy"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    identifier: List[FHIRIdentifier] = []
    status: str = "available"
    subject: FHIRReference
    started: Optional[str] = None
    numberOfSeries: int = 1
    numberOfInstances: int = 1
    procedureCode: List[Dict[str, Any]] = []
    series: List[FHIRImagingStudySeries] = []
    description: Optional[str] = None

class ClinicalContextPayload(BaseModel):
    patient_id: str
    age: Optional[str] = "54Y"
    gender: Optional[str] = "M"
    clinical_history: str = "Acute chest discomfort, shortness of breath, evaluating for acute cardiopulmonary pathology."
    prior_interventions: Optional[str] = "None"
    referring_indication: str = "STAT Rule out Pneumothorax / Intracranial Hemorrhage"
    imaging_study_resource: Optional[FHIRImagingStudy] = None
