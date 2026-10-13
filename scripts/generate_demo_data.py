import os
import uuid
import datetime
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def create_dicom_dataset(
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    patient_name: str,
    patient_id: str,
    accession_no: str,
    modality: str,
    instance_number: int = 1,
    slice_location: float = 0.0,
    pixel_array: np.ndarray = None,
    pixel_spacing: list = [0.75, 0.75],
    slice_thickness: float = 2.5
) -> FileDataset:
    """Creates a fully valid DICOM dataset with pixel data and standard headers."""
    # Create File Meta Information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" if modality == "CT" else "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("temp.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient & Encounter Info (Simulated PHI to test PS 3.15 Annex E de-identification)
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19680520"
    ds.PatientSex = "M"
    ds.AccessionNumber = accession_no
    ds.ReferringPhysicianName = "DR_SARAH_CONNOR_MD"
    ds.InstitutionName = "METROPOLITAN_MEMORIAL_HOSPITAL"

    # Study / Series / Instance Info
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = sop_uid
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = modality
    ds.StudyID = "DEMO_STUDY_1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.StudyDate = datetime.date.today().strftime("%Y%m%d")
    ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    ds.StudyDescription = f"AI Triage Demo - {modality} Protocol"
    ds.SeriesDescription = f"Recon 1 - {modality} Axial"

    # Geometric & Spatial Metadata
    ds.SliceLocation = slice_location
    ds.SliceThickness = slice_thickness
    ds.PixelSpacing = pixel_spacing
    ds.ImagePositionPatient = [0.0, 0.0, slice_location]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Display / VOI LUT
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.WindowCenter = 40.0 if modality == "CT" else 2048.0
    ds.WindowWidth = 400.0 if modality == "CT" else 4096.0
    ds.RescaleIntercept = -1024.0 if modality == "CT" else 0.0
    ds.RescaleSlope = 1.0

    # Ensure 16-bit uint pixel data
    pixel_16 = np.clip(pixel_array, 0, 65535).astype(np.uint16)
    ds.PixelData = pixel_16.tobytes()

    return ds

def generate_cxr_phantom(size: int = 512, with_pneumothorax: bool = False, with_cardiomegaly: bool = False) -> np.ndarray:
    """Generates synthetic chest radiograph phantom."""
    arr = np.full((size, size), 500, dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    
    # Outer chest wall
    chest_mask = ((x - size/2)**2 / (size*0.42)**2 + (y - size/2)**2 / (size*0.46)**2) <= 1
    arr[chest_mask] = 1800
    
    # Left and Right Lungs (Dark air attenuation)
    r_lung = ((x - size*0.32)**2 / (size*0.14)**2 + (y - size*0.48)**2 / (size*0.30)**2) <= 1
    l_lung = ((x - size*0.68)**2 / (size*0.14)**2 + (y - size*0.48)**2 / (size*0.30)**2) <= 1
    arr[r_lung] = 350
    arr[l_lung] = 350

    # Mediastinum and Heart (Dense white/gray attenuation)
    c_radius = size*0.18 if with_cardiomegaly else size*0.12
    heart = ((x - size*0.52)**2 / (c_radius)**2 + (y - size*0.60)**2 / (size*0.14)**2) <= 1
    arr[heart] = 3200

    # Spine column
    spine = (abs(x - size/2) <= size*0.04) & (y >= size*0.15) & (y <= size*0.85)
    arr[spine] = 3600

    # Pneumothorax: Hyperlucent pleural air collection with absent lung markings
    if with_pneumothorax:
        ptx_region = ((x - size*0.75)**2 / (size*0.09)**2 + (y - size*0.32)**2 / (size*0.16)**2) <= 1
        arr[ptx_region] = 100  # Extreme air lucency
        
    # Add subtle organic noise
    noise = np.random.normal(0, 25, (size, size)).astype(np.float32)
    return np.clip(arr + noise, 0, 4095)

def generate_ct_volume_phantom(num_slices: int = 16, size: int = 256) -> list:
    """Generates synthetic multi-slice abdominal CT series."""
    slices = []
    y, x = np.ogrid[:size, :size]

    for i in range(num_slices):
        z_ratio = i / max(num_slices - 1, 1)
        arr = np.full((size, size), 0, dtype=np.float32)  # Air ~ 0 HU mapped (Rescale -1024)

        # Abdominal wall
        body = ((x - size/2)**2 / (size*0.42)**2 + (y - size/2)**2 / (size*0.36)**2) <= 1
        arr[body] = 1060  # Soft tissue (+36 HU)

        # Spine & Vertebra
        spine = ((x - size/2)**2 / (size*0.06)**2 + (y - size*0.76)**2 / (size*0.06)**2) <= 1
        arr[spine] = 1800  # Dense bone (+776 HU)

        # Aorta
        aorta = ((x - size*0.45)**2 / (size*0.03)**2 + (y - size*0.70)**2 / (size*0.03)**2) <= 1
        arr[aorta] = 1140  # Contrast blood pool (+116 HU)

        # Liver (Right upper quadrant, slices 0.2 to 0.8)
        if 0.15 <= z_ratio <= 0.85:
            liver = ((x - size*0.32)**2 / (size*0.18)**2 + (y - size*0.42)**2 / (size*0.16)**2) <= 1
            arr[liver] = 1085  # Parenchyma (+61 HU)

        # Spleen (Left upper quadrant, slices 0.3 to 0.7)
        if 0.30 <= z_ratio <= 0.70:
            spleen = ((x - size*0.72)**2 / (size*0.10)**2 + (y - size*0.48)**2 / (size*0.09)**2) <= 1
            arr[spleen] = 1075  # (+51 HU)

        # Kidneys (Bilateral, slices 0.45 to 0.90)
        if 0.45 <= z_ratio <= 0.90:
            r_kidney = ((x - size*0.36)**2 / (size*0.06)**2 + (y - size*0.64)**2 / (size*0.08)**2) <= 1
            l_kidney = ((x - size*0.64)**2 / (size*0.06)**2 + (y - size*0.64)**2 / (size*0.08)**2) <= 1
            arr[r_kidney] = 1090
            arr[l_kidney] = 1090

        noise = np.random.normal(0, 15, (size, size)).astype(np.float32)
        slices.append(np.clip(arr + noise, 0, 4095))

    return slices

def generate_sample_cases(output_dir: Path):
    """Generates 3 clinical demo cases in raw DICOM format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Case 1: STAT Critical Acute Pneumothorax 2D CXR
    case1_dir = output_dir / "case_01_stat_pneumothorax"
    case1_dir.mkdir(exist_ok=True)
    study_uid1 = generate_uid(prefix="1.2.826.0.1.3680043.9.")[:48]
    series_uid1 = f"{study_uid1}.1"
    sop_uid1 = f"{series_uid1}.1"
    
    px1 = generate_cxr_phantom(size=512, with_pneumothorax=True)
    ds1 = create_dicom_dataset(
        study_uid=study_uid1,
        series_uid=series_uid1,
        sop_uid=sop_uid1,
        patient_name="HARRISON^MICHAEL^J",
        patient_id="PAT-908234",
        accession_no="ACC-STAT-101",
        modality="CR",
        pixel_array=px1
    )
    ds1.save_as(str(case1_dir / "cxr_stat_ptx.dcm"))
    print(f"Created Case 1 (STAT Pneumothorax CXR): {case1_dir}")

    # Case 2: 3D Volumetric Abdominal CT (16 slices) for MONAI SwinUNETR
    case2_dir = output_dir / "case_02_3d_ct_multiorgan"
    case2_dir.mkdir(exist_ok=True)
    study_uid2 = generate_uid(prefix="1.2.826.0.1.3680043.9.")[:48]
    series_uid2 = f"{study_uid2}.1"
    
    ct_slices = generate_ct_volume_phantom(num_slices=16, size=256)
    for idx, sl_arr in enumerate(ct_slices):
        sop_uid = f"{series_uid2}.{idx+1}"
        ds_slice = create_dicom_dataset(
            study_uid=study_uid2,
            series_uid=series_uid2,
            sop_uid=sop_uid,
            patient_name="REYNOLDS^ELIZABETH^M",
            patient_id="PAT-448192",
            accession_no="ACC-CT-202",
            modality="CT",
            instance_number=idx + 1,
            slice_location=float(idx * 5.0),
            pixel_array=sl_arr,
            slice_thickness=5.0
        )
        ds_slice.save_as(str(case2_dir / f"slice_{idx+1:03d}.dcm"))
    print(f"Created Case 2 (3D Volumetric CT, 16 slices): {case2_dir}")

    # Case 3: Routine Normal CXR
    case3_dir = output_dir / "case_03_routine_normal"
    case3_dir.mkdir(exist_ok=True)
    study_uid3 = generate_uid(prefix="1.2.826.0.1.3680043.9.")[:48]
    series_uid3 = f"{study_uid3}.1"
    sop_uid3 = f"{series_uid3}.1"
    
    px3 = generate_cxr_phantom(size=512, with_pneumothorax=False, with_cardiomegaly=False)
    ds3 = create_dicom_dataset(
        study_uid=study_uid3,
        series_uid=series_uid3,
        sop_uid=sop_uid3,
        patient_name="THOMPSON^DAVID^K",
        patient_id="PAT-112390",
        accession_no="ACC-ROUTINE-303",
        modality="CR",
        pixel_array=px3
    )
    ds3.save_as(str(case3_dir / "cxr_routine_normal.dcm"))
    print(f"Created Case 3 (Routine Normal CXR): {case3_dir}")

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "demo_dicom"
    generate_sample_cases(out)
