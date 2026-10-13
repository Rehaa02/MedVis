import torch
import torch.nn.functional as F
import numpy as np
from typing import Union

class AdversarialDefenses:
    """
    Adversarial Cybersecurity Defenses for Medical Imaging Neural Networks.
    Implements Feature Squeezing, Bit-Depth Reduction, and Spatial Smoothing
    to neutralize imperceptible gradient perturbation attacks (FGSM, PGD).
    """

    @staticmethod
    def bit_depth_reduction(tensor: torch.Tensor, target_bits: int = 8) -> torch.Tensor:
        """
        Feature Squeezing via Bit-Depth Reduction.
        Compresses high-bit-depth medical image representations (e.g. 16-bit) to target_bits,
        coarsening the pixel precision and stripping low-amplitude adversarial noise.
        """
        levels = 2 ** target_bits - 1
        # Normalize to [0, 1] range if not already
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            normalized = (tensor - min_val) / (max_val - min_val)
        else:
            normalized = tensor
            
        # Quantize to discrete levels and map back
        quantized = torch.round(normalized * levels) / levels
        
        # Rescale back to original dynamic range
        squeezed = quantized * (max_val - min_val) + min_val
        return squeezed

    @staticmethod
    def spatial_smoothing(tensor: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
        """
        Spatial Gaussian Smoothing.
        Applies a low-pass Gaussian filter to eliminate high-frequency spatial gradients
        typically introduced by white-box adversarial attacks.
        Works across 2D (B, C, H, W) and 3D (B, C, D, H, W) tensors.
        """
        dims = tensor.dim()
        if dims == 4:  # 2D image (B, C, H, W)
            channels = tensor.shape[1]
            radius = kernel_size // 2
            x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=tensor.device)
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
            
            return F.conv2d(tensor, kernel_2d, padding=radius, groups=channels)
            
        elif dims == 5:  # 3D volume (B, C, D, H, W)
            channels = tensor.shape[1]
            radius = kernel_size // 2
            x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=tensor.device)
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
            kernel_3d = kernel_3d.expand(channels, 1, kernel_size, kernel_size, kernel_size)
            
            return F.conv3d(tensor, kernel_3d, padding=radius, groups=channels)
        else:
            return tensor

    @classmethod
    def apply_defensive_filter(cls, tensor: torch.Tensor, target_bits: int = 8, smooth_sigma: float = 0.8) -> torch.Tensor:
        """
        Executes the full defensive squeezing pipeline:
        1. Bit-depth reduction (neutralizes micro-epsilon gradients)
        2. Spatial Gaussian smoothing (attenuates high-frequency pixel anomalies)
        """
        squeezed = cls.bit_depth_reduction(tensor, target_bits=target_bits)
        filtered = cls.spatial_smoothing(squeezed, kernel_size=3, sigma=smooth_sigma)
        return filtered
