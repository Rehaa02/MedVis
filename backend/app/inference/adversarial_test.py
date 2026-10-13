import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from backend.app.inference.defenses import AdversarialDefenses

class AdversarialAttackTester:
    """
    Simulates white-box gradient attacks (FGSM and PGD) on clinical deep learning models
    to test the effectiveness of defensive Feature Squeezing.
    """

    @staticmethod
    def fgsm_attack(model: nn.Module, image: torch.Tensor, target_class: int, epsilon: float = 0.05) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM):
        x_adv = x + epsilon * sign(grad_x J(theta, x, y))
        """
        image_var = image.clone().detach().requires_grad_(True)
        outputs = model(image_var)
        criterion = nn.CrossEntropyLoss()
        target = torch.tensor([target_class], dtype=torch.long, device=image.device)
        loss = criterion(outputs, target)
        
        model.zero_grad()
        loss.backward()
        
        # Extract gradient sign
        data_grad = image_var.grad.data
        perturbed_image = image + epsilon * data_grad.sign()
        
        # Maintain within reasonable pixel bounds
        perturbed_image = torch.clamp(perturbed_image, image.min(), image.max())
        return perturbed_image.detach()

    @staticmethod
    def pgd_attack(model: nn.Module, image: torch.Tensor, target_class: int, epsilon: float = 0.05, alpha: float = 0.01, iters: int = 5) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD):
        Multi-step iterative perturbation projected back into L-infinity epsilon ball.
        """
        perturbed_image = image.clone().detach()
        criterion = nn.CrossEntropyLoss()
        target = torch.tensor([target_class], dtype=torch.long, device=image.device)
        
        for _ in range(iters):
            perturbed_image.requires_grad = True
            outputs = model(perturbed_image)
            loss = criterion(outputs, target)
            
            model.zero_grad()
            loss.backward()
            
            # Step in direction of gradient
            adv = perturbed_image + alpha * perturbed_image.grad.sign()
            
            # Projection onto L-infinity ball [image - eps, image + eps]
            eta = torch.clamp(adv - image, min=-epsilon, max=epsilon)
            perturbed_image = torch.clamp(image + eta, min=image.min(), max=image.max()).detach()
            
        return perturbed_image

    @classmethod
    def evaluate_defense(cls, model: nn.Module, image: torch.Tensor, target_class: int, epsilon: float = 0.05) -> Dict[str, Any]:
        """
        Evaluates model predictions across three states:
        1. Baseline Clean Image
        2. Adversarially Perturbed Image (FGSM)
        3. Defended Image (Feature Squeezing + Spatial Smoothing)
        """
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else "cpu"
        image = image.to(device)

        # 1. Clean Baseline Prediction
        with torch.no_grad():
            clean_logits = model(image)
            clean_probs = torch.softmax(clean_logits, dim=-1)
            clean_conf = clean_probs[0, target_class].item() if clean_probs.shape[-1] > target_class else float(clean_probs.max())
            clean_pred = int(torch.argmax(clean_probs, dim=-1)[0])

        # 2. Adversarial Attack
        adv_image = cls.fgsm_attack(model, image, target_class=clean_pred, epsilon=epsilon)
        with torch.no_grad():
            adv_logits = model(adv_image)
            adv_probs = torch.softmax(adv_logits, dim=-1)
            adv_conf = adv_probs[0, target_class].item() if adv_probs.shape[-1] > target_class else float(adv_probs.max())
            adv_pred = int(torch.argmax(adv_probs, dim=-1)[0])

        # 3. Defended Inference
        defended_image = AdversarialDefenses.apply_defensive_filter(adv_image, target_bits=8, smooth_sigma=0.8)
        with torch.no_grad():
            defended_logits = model(defended_image)
            defended_probs = torch.softmax(defended_logits, dim=-1)
            defended_conf = defended_probs[0, target_class].item() if defended_probs.shape[-1] > target_class else float(defended_probs.max())
            defended_pred = int(torch.argmax(defended_probs, dim=-1)[0])

        return {
            "clean_prediction_class": clean_pred,
            "clean_confidence": round(clean_conf, 4),
            "adversarial_prediction_class": adv_pred,
            "adversarial_confidence": round(adv_conf, 4),
            "attack_degradation_pct": round(max(0.0, (clean_conf - adv_conf) / max(clean_conf, 1e-6) * 100), 2),
            "defended_prediction_class": defended_pred,
            "defended_confidence": round(defended_conf, 4),
            "defense_recovery_pct": round(min(100.0, (defended_conf / max(clean_conf, 1e-6)) * 100), 2),
            "defense_successful": (defended_pred == clean_pred)
        }
