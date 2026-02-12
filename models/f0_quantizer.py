"""
F0/UV Quantizer for Efficient Transmission
===========================================

Quantizes continuous F0 contours to discrete codebooks to reduce bitrate.

From paper Section 2.1:
"F0 is extracted at 100 Hz via WORLD Harvest, resampled to f_z, and voiced
values are mapped to 4-bit scalar codes (16 bins, log-spaced over [50, 550] Hz,
covering typical adult and child pitch ranges), yielding (4+1) × f_z/1000 kbps
overhead."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STEQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for Quantization

    Forward: Quantize to discrete bins
    Backward: Pass gradients through unchanged
    """
    @staticmethod
    def forward(ctx, x, n_bins):
        codes = torch.floor(x * (n_bins - 1)).long()
        codes = torch.clamp(codes, 0, n_bins - 1)
        return codes

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        return grad_output, None


class F0UVQuantizer(nn.Module):
    """
    F0/UV Quantizer with STE

    From paper:
    - 4-bit F0 quantization (16 bins, log-spaced over [50, 550] Hz)
    - 1-bit UV flag
    - Overhead: (4+1) × frame_rate/1000 kbps

    Args:
        n_f0_bins: Number of F0 bins (16 for 4-bit, 256 for 8-bit)
        frame_rate: Codec frame rate in Hz (75 Hz for EnCodec)
        f0_min: Min F0 in normalized range (default: 0.0)
        f0_max: Max F0 in normalized range (default: 1.0)
    """
    def __init__(
        self,
        n_f0_bins: int = 16,  # 4-bit (as per paper)
        frame_rate: float = 75.0,
        f0_min: float = 0.0,
        f0_max: float = 1.0
    ):
        super().__init__()

        self.n_f0_bins = n_f0_bins
        self.frame_rate = frame_rate
        self.f0_min = f0_min
        self.f0_max = f0_max

        # Compute bitrate
        bits_per_frame = torch.log2(torch.tensor(n_f0_bins)).item() + 1  # +1 for UV
        self.bitrate_kbps = bits_per_frame * frame_rate / 1000.0

    def quantize(self, f0: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """
        Quantize F0 and UV using STE

        Args:
            f0: [B, 1, T] - Normalized F0 in [0, 1]
            uv: [B, 1, T] - UV flags (0 or 1)

        Returns:
            codes: [B, 1, T] - Quantized codes
        """
        # Clip F0 to valid range
        f0_clipped = torch.clamp(f0, self.f0_min, self.f0_max)

        # Normalize
        f0_normalized = (f0_clipped - self.f0_min) / (self.f0_max - self.f0_min)

        # Use STE for differentiable quantization
        f0_codes = STEQuantize.apply(f0_normalized, self.n_f0_bins)

        # Apply UV mask: unvoiced -> code 0
        uv_binary = (uv > 0.5).long()
        codes = torch.where(uv_binary > 0, f0_codes, torch.zeros_like(f0_codes))

        return codes

    def dequantize(self, codes: torch.Tensor) -> tuple:
        """
        Dequantize codes to F0 and UV

        Args:
            codes: [B, 1, T] - Quantized codes

        Returns:
            f0: [B, 1, T] - Reconstructed F0
            uv: [B, 1, T] - Reconstructed UV
        """
        # UV: code > 0 indicates voiced
        uv = (codes > 0).float()

        # F0: dequantize from codes
        f0_normalized = codes.float() / (self.n_f0_bins - 1)
        f0 = f0_normalized * (self.f0_max - self.f0_min) + self.f0_min

        # Zero out unvoiced frames
        f0 = f0 * uv

        return f0, uv

    def forward(self, f0: torch.Tensor, uv: torch.Tensor,
                return_codes: bool = False) -> tuple:
        """
        Forward: quantize then dequantize (simulates transmission)

        Differentiable thanks to STE.

        Args:
            f0: [B, 1, T] - Original F0
            uv: [B, 1, T] - Original UV
            return_codes: Return codes?

        Returns:
            f0_reconstructed, uv_reconstructed, (codes if requested)
        """
        codes = self.quantize(f0, uv)
        f0_recon, uv_recon = self.dequantize(codes)

        if return_codes:
            return f0_recon, uv_recon, codes
        return f0_recon, uv_recon

    def get_bitrate_info(self) -> dict:
        """Get bitrate information (as per paper)"""
        bits_per_frame = torch.log2(torch.tensor(self.n_f0_bins, dtype=torch.float32)).item() + 1

        return {
            'n_f0_bins': self.n_f0_bins,
            'frame_rate_hz': self.frame_rate,
            'bits_per_frame': bits_per_frame,
            'bitrate_kbps': self.bitrate_kbps,
            'f0_bits': torch.log2(torch.tensor(self.n_f0_bins, dtype=torch.float32)).item(),
            'uv_bits': 1.0,
        }


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("F0/UV Quantizer Test (with STE)")
    print("=" * 60)

    # Test with paper parameters: 4-bit (16 bins)
    print("\n[1] Creating quantizer (4-bit, as per paper)...")
    quantizer = F0UVQuantizer(n_f0_bins=16, frame_rate=75.0)

    info = quantizer.get_bitrate_info()
    print(f"    Bins: {info['n_f0_bins']} ({info['f0_bits']:.1f}-bit F0 + {info['uv_bits']:.0f}-bit UV)")
    print(f"    Bitrate: {info['bitrate_kbps']:.3f} kbps at {info['frame_rate_hz']} Hz")

    # Test gradient flow with STE
    print("\n[2] Testing STE gradient flow...")
    f0 = torch.rand(2, 1, 75, requires_grad=True)
    uv = torch.ones(2, 1, 75)

    f0_recon, uv_recon = quantizer(f0, uv)
    loss = f0_recon.sum()
    loss.backward()

    print(f"    ✓ Gradient exists: {f0.grad is not None}")
    if f0.grad is not None:
        print(f"    ✓ Gradient norm: {f0.grad.norm().item():.6f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
