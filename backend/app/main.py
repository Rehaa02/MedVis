from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.api.v1.routes_dicom import router as dicom_router
from backend.app.api.v1.routes_dicomweb import router as dicomweb_router
from backend.app.api.v1.routes_triage import router as triage_router
from backend.app.api.v1.routes_review import router as review_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup initialization
    print(f"============================================================")
    print(f"   {settings.PROJECT_NAME} (v{settings.VERSION})")
    print(f"   Hardware Acceleration Device: {settings.DEVICE.upper()}")
    print(f"   Storage Root: {settings.STORAGE_DIR}")
    print(f"   FDA GMLP Human-in-Loop: {'ENABLED' if settings.HUMAN_IN_THE_LOOP_REQUIRED else 'DISABLED'}")
    print(f"============================================================")
    yield
    print("[Shutdown] Closing clinical imaging background processes.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Clinical AI Diagnostic & Triage Platform with MONAI SwinUNETR, 2D ResNet Triage, Adversarial Defenses, Hallucination Mitigation, and FDA GMLP Compliance.",
    lifespan=lifespan
)

# CORS Middleware for PACS Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Core API Routes
app.include_router(dicom_router, prefix=settings.API_V1_STR)
app.include_router(dicomweb_router)
app.include_router(triage_router, prefix=settings.API_V1_STR)
app.include_router(review_router, prefix=settings.API_V1_STR)

# Healthcheck & System Specs Endpoint
@app.get(f"{settings.API_V1_STR}/health", tags=["System Health"])
async def system_health():
    import torch
    import monai
    import pydicom
    return {
        "status": "HEALTHY",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "device": settings.DEVICE,
        "pytorch_version": torch.__version__,
        "monai_version": monai.__version__,
        "pydicom_version": pydicom.__version__,
        "accelerator_mps_available": torch.backends.mps.is_available(),
        "accelerator_cuda_available": torch.cuda.is_available(),
        "storage_mode": "Local S3-Compatible" if settings.USE_LOCAL_STORAGE else "MinIO S3",
        "compliance": [
            "DICOM PS 3.15 Annex E Application Level Confidentiality Profile",
            "HL7 FHIR R4 ImagingStudy Resource Standard",
            "FDA SaMD Good Machine Learning Practice (GMLP) Principle 8"
        ]
    }

# Mount Web Client UI
web_dir = Path(__file__).resolve().parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def serve_index():
        return FileResponse(str(web_dir / "index.html"))
