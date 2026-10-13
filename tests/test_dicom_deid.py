import pydicom
from pathlib import Path
from backend.app.core.dicom_deid import DICOMDeidentifier

def test_dicom_deidentification():
    # Load Case 1 demo DICOM
    case1_path = Path("data/demo_dicom/case_01_stat_pneumothorax/cxr_stat_ptx.dcm")
    assert case1_path.exists(), "Case 1 DICOM should exist from generator"
    
    ds = pydicom.dcmread(str(case1_path))
    original_patient_name = str(ds.PatientName)
    assert original_patient_name == "HARRISON^MICHAEL^J"
    
    # De-identify according to DICOM PS 3.15 Annex E
    clean_ds, audit = DICOMDeidentifier.deidentify(ds)
    
    # Verify PHI is removed or pseudonymized
    assert str(clean_ds.PatientName) != "HARRISON^MICHAEL^J"
    assert str(clean_ds.PatientName).startswith("PATIENT^ANON_")
    assert clean_ds.PatientID.startswith("ANON_")
    assert clean_ds.PatientIdentityRemoved == "YES"
    assert "DICOM PS 3.15 Annex E" in clean_ds.DeidentificationMethod
    assert (0x0008, 0x0090) not in clean_ds  # Referring Physician removed
    
    # Verify critical imaging physics and spatial metadata remain intact
    assert hasattr(clean_ds, "PixelSpacing")
    assert hasattr(clean_ds, "Rows")
    assert hasattr(clean_ds, "Columns")
    assert hasattr(clean_ds, "PixelData")
    print("\n[TEST PASSED] DICOM PS 3.15 Annex E De-identification verified.")

if __name__ == "__main__":
    test_dicom_deidentification()
