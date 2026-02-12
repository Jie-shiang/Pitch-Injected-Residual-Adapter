"""
Test F0/UV Quantizer with STE
"""

import pytest
import torch
from models.f0_quantizer import F0UVQuantizer, STEQuantize


class TestSTEQuantize:
    """Test Straight-Through Estimator"""

    def test_forward_quantization(self):
        """Test forward pass quantizes correctly"""
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        n_bins = 4  # 2-bit

        codes = STEQuantize.apply(x, n_bins)

        # Expected: [0, 0, 1, 2, 3] (quantized to 4 bins)
        expected = torch.tensor([0, 0, 1, 2, 3])
        assert torch.equal(codes, expected)

    def test_gradient_flow(self):
        """Test gradients flow through STE"""
        x = torch.tensor([0.3, 0.6, 0.9], requires_grad=True)
        n_bins = 16

        codes = STEQuantize.apply(x, n_bins)
        loss = codes.float().sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestF0UVQuantizer:
    """Test F0/UV Quantizer"""

    @pytest.fixture
    def quantizer(self):
        return F0UVQuantizer(
            n_f0_bins=16,  # 4-bit as per paper
            frame_rate=75.0
        )

    def test_quantize_shape(self, quantizer):
        """Test quantize output shape"""
        f0 = torch.rand(4, 1, 75)
        uv = torch.ones(4, 1, 75)

        codes = quantizer.quantize(f0, uv)
        assert codes.shape == (4, 1, 75)
        assert codes.dtype == torch.long

    def test_dequantize_shape(self, quantizer):
        """Test dequantize output shape"""
        codes = torch.randint(0, 16, (4, 1, 75))

        f0_recon, uv_recon = quantizer.dequantize(codes)
        assert f0_recon.shape == (4, 1, 75)
        assert uv_recon.shape == (4, 1, 75)
        assert f0_recon.dtype == torch.float32
        assert uv_recon.dtype == torch.float32

    def test_unvoiced_masking(self, quantizer):
        """Test unvoiced frames are set to code 0"""
        f0 = torch.rand(2, 1, 50)
        uv = torch.zeros(2, 1, 50)  # All unvoiced

        codes = quantizer.quantize(f0, uv)
        # All codes should be 0 for unvoiced
        assert (codes == 0).all()

    def test_round_trip(self, quantizer):
        """Test quantize -> dequantize round trip"""
        f0 = torch.rand(2, 1, 75)
        uv = (torch.rand(2, 1, 75) > 0.5).float()

        f0_recon, uv_recon = quantizer(f0, uv)

        # UV should be preserved
        assert torch.equal(uv, uv_recon)

        # F0 should be approximately preserved (with quantization error)
        # Only check voiced frames
        voiced_mask = uv > 0.5
        if voiced_mask.any():
            f0_error = (f0[voiced_mask] - f0_recon[voiced_mask]).abs()
            # Error should be < 1/n_bins (quantization step)
            assert (f0_error < 1.0 / quantizer.n_f0_bins).all()

    def test_gradient_flow(self, quantizer):
        """Test gradients flow through STE"""
        f0 = torch.rand(2, 1, 75, requires_grad=True)
        uv = torch.ones(2, 1, 75)

        f0_recon, uv_recon = quantizer(f0, uv)
        loss = f0_recon.sum()
        loss.backward()

        assert f0.grad is not None
        assert f0.grad.norm().item() > 0

    def test_bitrate_calculation(self, quantizer):
        """Test bitrate calculation (from paper)"""
        info = quantizer.get_bitrate_info()

        # Paper: (4+1) × 75/1000 = 0.375 kbps
        expected_bitrate = 5.0 * 75.0 / 1000.0
        assert abs(info['bitrate_kbps'] - expected_bitrate) < 0.01

        assert info['n_f0_bins'] == 16
        assert info['frame_rate_hz'] == 75.0
        assert info['f0_bits'] == 4.0  # log2(16) = 4
        assert info['uv_bits'] == 1.0

    def test_different_bin_counts(self):
        """Test quantizer with different bit depths"""
        for n_bins in [16, 32, 64, 128, 256]:
            quantizer = F0UVQuantizer(n_f0_bins=n_bins)
            f0 = torch.rand(2, 1, 50)
            uv = torch.ones(2, 1, 50)

            codes = quantizer.quantize(f0, uv)
            # Codes should be in valid range
            assert codes.min() >= 0
            assert codes.max() < n_bins
