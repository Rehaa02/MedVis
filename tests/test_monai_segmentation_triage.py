import numpy as np
from backend.app.inference.triage_2d import triage_2d_engine
from backend.app.inference.segmentation_3d import segmentation_3d_engine

def test_triage_and_segmentation():
    # 1. Test 2D Triage
    dummy_pixels_2d = np.random.randint(0, 4095, (512, 512), dtype=np.uint16)
    triage_res = triage_2d_engine.classify_and_triage(dummy_pixels_2d, {"modality": "CR"})
    
    assert "urgency_score" in triage_res
    assert triage_res["triage_category"] in ["STAT", "URGENT", "ROUTINE"]
    assert len(triage_res["pathology_findings"]) > 0
    assert triage_res["time_to_triage_seconds"] < 10.0  # Fast sub-30s triage constraint

    # 2. Test MONAI 3D SwinUNETR Volumetric Segmentation
    dummy_volume_3d = np.random.randint(0, 4095, (16, 128, 128), dtype=np.uint16)
    seg_res = segmentation_3d_engine.segment_and_quantify(
        volume_array=dummy_volume_3d,
        pixel_spacing=[1.0, 1.0],
        slice_thickness=2.5
    )
    
    assert "organ_volumes" in seg_res
    assert len(seg_res["organ_volumes"]) == 13  # 13 BTCV organs
    assert any(o.organ_name == "Liver" for o in seg_res["organ_volumes"])
    print(f"\n[TEST PASSED] 2D Triage & MONAI 3D SwinUNETR executed successfully in {seg_res['inference_time_seconds']}s.")

if __name__ == "__main__":
    test_triage_and_segmentation()
