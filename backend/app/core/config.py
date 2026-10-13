import os
from pathlib import Path
from typing import Literal
import torch
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Based Medical Image Diagnosis System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DICOM_RAW_DIR: Path = DATA_DIR / "dicom_raw"
    DICOM_CLEAN_DIR: Path = DATA_DIR / "dicom_clean"
    STORAGE_DIR: Path = DATA_DIR / "storage"
    MODELS_DIR: Path = BASE_DIR / "models_weights"
    
    # Storage settings (MinIO / S3 or Local S3-compatible fallback)
    USE_LOCAL_STORAGE: bool = True
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "medical-imaging-studies"
    MINIO_SECURE: bool = False
    
    # Task Queue (Celery + Redis or Local Async In-Process Worker)
    REDIS_URL: str = "redis://localhost:6379/0"
    USE_CELERY: bool = False  # Set to True when Redis/Celery daemon is active
    
    # Deep Learning Hardware Acceleration
    DEVICE: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model configuration
    RESNET_MODEL_TYPE: str = "resnet50"  # or 'resnet18'
    SWIN_FEATURE_SIZE: int = 12  # Compact for fast edge inference (48 for full server)
    SWIN_IN_CHANNELS: int = 1
    SWIN_OUT_CHANNELS: int = 14  # 14 anatomical classes / multi-organ
    
    # Reporting LLM configuration
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"  # or 'BioMistral/BioMistral-7B'
    USE_LOCAL_HF_LLM: bool = False  # If True, attempts loading weights from HF; if False, uses internal deterministic clinical engine
    
    # Hallucination Guard thresholds (ReXTrust & RadFlag)
    REXTRUST_ENTROPY_THRESHOLD: float = 1.25  # High-uncertainty token threshold
    RADFLAG_ENTAILMENT_THRESHOLD: float = 0.75
    
    # FDA Good Machine Learning Practice (GMLP) Settings
    AUDIT_LOG_FILE: Path = DATA_DIR / "gmlp_audit_trail.jsonl"
    HUMAN_IN_THE_LOOP_REQUIRED: bool = True
    CRITICAL_URGENCY_THRESHOLD: float = 0.70  # Studies with urgency >= 0.70 are flagged STAT (<30s triage)

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# Ensure required directories exist
for path in [settings.DATA_DIR, settings.DICOM_RAW_DIR, settings.DICOM_CLEAN_DIR, settings.STORAGE_DIR, settings.MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
