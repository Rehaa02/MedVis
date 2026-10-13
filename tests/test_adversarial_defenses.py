import torch
from backend.app.inference.defenses import AdversarialDefenses
from backend.app.inference.adversarial_test import AdversarialAttackTester
from backend.app.inference.triage_2d import triage_2d_engine

def test_adversarial_defense_pipeline():
    # Generate test image tensor [1, 3, 224, 224]
    dummy_input = torch.rand(1, 3, 224, 224, device=triage_2d_engine.device)
    
    # 1. Test Feature Squeezing bit-depth reduction
    squeezed = AdversarialDefenses.bit_depth_reduction(dummy_input, target_bits=8)
    assert squeezed.shape == dummy_input.shape
    
    # 2. Test Spatial Smoothing
    smoothed = AdversarialDefenses.spatial_smoothing(squeezed, kernel_size=3, sigma=0.8)
    assert smoothed.shape == dummy_input.shape

    # 3. Test FGSM Attack and Defense Recovery
    metrics = AdversarialAttackTester.evaluate_defense(
        model=triage_2d_engine.model,
        image=dummy_input,
        target_class=0,
        epsilon=0.05
    )
    
    assert "clean_confidence" in metrics
    assert "adversarial_confidence" in metrics
    assert "defended_confidence" in metrics
    assert "defense_recovery_pct" in metrics
    print(f"\n[TEST PASSED] Adversarial defense evaluated: Clean={metrics['clean_confidence']}, Adv={metrics['adversarial_confidence']}, Defended={metrics['defended_confidence']}")

if __name__ == "__main__":
    test_adversarial_defense_pipeline()
