from backend.app.reporting.hallucination_guard import hallucination_guard
from backend.app.api.v1.routes_review import calculate_levenshtein_distance

def test_hallucination_mitigation_and_gmlp():
    # 1. Test ReXTrust Token Entropy Auditing
    test_tokens = ["Lungs", "are", "clear", "atypical", "borderline"]
    test_entropies = [0.25, 0.15, 0.30, 1.45, 1.55]  # High entropy on ambiguous terms
    
    token_audits, mean_ent, high_risk_count = hallucination_guard.audit_tokens(test_tokens, test_entropies)
    assert len(token_audits) == 5
    assert high_risk_count == 2
    assert token_audits[3].risk_level == "high_risk_hallucination"
    assert token_audits[0].risk_level == "grounded"

    # 2. Test DEAR Anatomical Grounding
    text_normal = "Lungs are clear bilaterally. Spleen is normal."
    assert hallucination_guard.verify_dear_alignment(text_normal, ["Lungs", "Spleen"], []) is True

    text_mismatch = "Intracranial hemorrhage noted with midline shift."
    # Mismatch when modality is abdominal/chest with no brain structures
    assert hallucination_guard.verify_dear_alignment(text_mismatch, ["Liver", "Spleen"], ["Pleural Effusion"]) is False

    # 3. Test Levenshtein Distance for FDA GMLP
    s1 = "There is an acute right apical pneumothorax."
    s2 = "There is a small right apical pneumothorax."
    dist = calculate_levenshtein_distance(s1, s2)
    assert dist > 0
    assert calculate_levenshtein_distance(s1, s1) == 0

    print(f"\n[TEST PASSED] Hallucination mitigation and FDA GMLP Levenshtein distance verified (dist={dist}).")

if __name__ == "__main__":
    test_hallucination_mitigation_and_gmlp()
