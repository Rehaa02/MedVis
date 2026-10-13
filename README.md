# ClinAI PACS: AI-Based Medical Image Diagnosis & Triage Orchestrator

An enterprise-grade, clinical AI diagnostic platform combining **Convolutional Neural Networks (CNNs)**, **Vision Transformers (SwinUNETR)**, and **Multimodal Clinical LLMs** embedded directly into a PACS / DICOMweb orchestration pipeline.

---

## Key Features & Clinical Architecture

1. **DICOM Ingestion & PS 3.15 Annex E De-identification**:
   - Asynchronous DICOM C-STORE and multipart upload gateway (`/api/v1/dicom/upload`, `/api/v1/dicom/c-store`).
   - Cleanses, strips, or pseudonymizes all Protected Health Information (PHI tags: `PatientName`, `PatientID`, `AccessionNumber`, `ReferringPhysicianName`) while preserving spatial/geometric acquisition metadata (`PixelSpacing`, `SliceThickness`, `WindowCenter`, `WindowWidth`).

2. **Sub-30s Acute Worklist Triage (2D ResNet-50 / ResNet-18)**:
   - Evaluates acute chest and brain pathologies (Pneumothorax, Intracranial Hemorrhage, Pulmonary Embolism, Pleural Effusion, Cardiomegaly).
   - Formulates dynamic Urgency Scores ($0.0 - 1.0$) and automatically pushes STAT cases ($\ge 0.70$) to the top of the radiologist's queue in $< 0.3$ seconds.

3. **3D Volumetric Multi-Organ Segmentation (MONAI SwinUNETR)**:
   - Uses hierarchical local-window Swin Transformer with residual 3D U-Net decoder.
   - Segments 14 anatomical organs (Liver, Kidneys, Spleen, Aorta, Pancreas, etc.) and computes physical volume ($\text{cm}^3$) using voxel dimensions and DICOM slice thickness.

4. **Multimodal LLM Reporting with Hallucination Mitigation**:
   - Synthesizes ACR-standard structured **Findings** (by anatomical system) and **Impression** sections.
   - **ReXTrust White-box Entropy**: Calculates Shannon entropy on token logit distributions ($H(V) = -\sum p \log_2 p$), color-coding tokens in the UI (Green = grounded, Amber = moderate uncertainty, Red = high risk).
   - **RadFlag Semantic Consistency**: Checks for cross-sample logical contradictions.
   - **DEAR Dual-Stream Alignment**: Validates organ laterality and anatomical consistency against verified segmentation masks.

5. **Adversarial Cybersecurity Defenses (FGSM & PGD Defense)**:
   - Protects against adversarial gradient exploitation via **Feature Squeezing** (bit-depth reduction from 16-bit to 8-bit) and **Spatial Gaussian Smoothing**.
   - Includes interactive security testbed to run live FGSM attacks and measure defense recovery.

6. **FDA Good Machine Learning Practice (GMLP) Compliance**:
   - Implements Principle 8 (Human-in-the-Loop oversight): Radiologists can Accept, Modify, or Reject AI drafts.
   - Computes real-time **Levenshtein edit distance** and signs audit logs with cryptographic SHA-256 signatures for 21 CFR Part 11 traceability.

---

## Required Models Catalog

| Tier | Model Architecture | Recommended Weights / Source | Footprint | Hardware Target |
| :--- | :--- | :--- | :--- | :--- |
| **2D Triage** | ResNet-50 / ResNet-18 | `torchvision.models.resnet50` / `torchxrayvision` | ~98 MB | Apple Silicon MPS / CUDA / CPU |
| **3D Segmentation** | MONAI SwinUNETR (3D) | MONAI Model Zoo (`swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt` / BTCV) | ~240 MB | MPS / CUDA / CPU |
| **Clinical LLM (Edge)** | Qwen2.5-1.5B-Instruct | Hugging Face (`Qwen/Qwen2.5-1.5B-Instruct`) | ~3.2 GB | Apple Silicon Mac (M-series) |
| **Clinical LLM (Full)** | BioMistral-7B | Hugging Face (`BioMistral/BioMistral-7B` / GGUF Q4_K_M) | ~4.2 GB - 14 GB | 16 GB+ RAM / GPU |

---

## Quick Start Guide

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Inspect Active Hardware and Models
```bash
python3 scripts/download_models.py --inspect
```

### 3. Start the System
```bash
./run.sh
# or
python3 scripts/run_system.py
```
Access the application:
* **Interactive Radiologist PACS Cockpit**: [http://localhost:8000/](http://localhost:8000/)
* **Interactive OpenAPI Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Seed Clinical Demo Studies
In the web interface, click the **"Seed Demo Studies"** button in the top header, or run:
```bash
python3 scripts/generate_demo_data.py
```

### 5. Run Automated Test Suites
```bash
PYTHONPATH=. python3 tests/test_dicom_deid.py
PYTHONPATH=. python3 tests/test_adversarial_defenses.py
PYTHONPATH=. python3 tests/test_monai_segmentation_triage.py
PYTHONPATH=. python3 tests/test_hallucination_and_gmlp.py
```
