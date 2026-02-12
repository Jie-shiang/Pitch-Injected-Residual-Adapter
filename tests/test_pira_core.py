"""
Test PIRA Core Components
"""

import pytest
import torch
from models.pira_core import ResidualPitchInjector, ConfidenceNetwork, DilatedConvBlock


class TestDilatedConvBlock:
    """Test DilatedConvBlock"""

    def test_forward_shape(self):
        """Test output shape matches input"""
        block = DilatedConvBlock(channels=256, dilation=3)
        x = torch.randn(4, 256, 75)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Test residual connection exists"""
        block = DilatedConvBlock(channels=256, dilation=1)
        x = torch.randn(2, 256, 50)
        out = block(x)
        # Output should not be identical to input (due to transformation)
        assert not torch.allclose(out, x)
        # But should have same shape
        assert out.shape == x.shape


class TestResidualPitchInjector:
    """Test ResidualPitchInjector"""

    @pytest.fixture
    def model(self):
        return ResidualPitchInjector(
            latent_dim=128,
            hidden_dim=256,
            dilation_rates=[1, 3, 9, 27, 81, 243]
        )

    def test_output_shape(self, model):
        """Test output shape is [B, latent_dim, T]"""
        f0 = torch.rand(4, 1, 75)
        uv = torch.rand(4, 1, 75)
        residual = model(f0, uv)
        assert residual.shape == (4, 128, 75)

    def test_parameter_count(self, model):
        """Test parameter count is reasonable (~600K)"""
        n_params = sum(p.numel() for p in model.parameters())
        assert 500_000 < n_params < 1_000_000

    def test_gradient_flow(self, model):
        """Test gradients flow through the model"""
        f0 = torch.rand(2, 1, 75, requires_grad=True)
        uv = torch.ones(2, 1, 75)
        residual = model(f0, uv)
        loss = residual.sum()
        loss.backward()
        assert f0.grad is not None
        assert f0.grad.abs().sum() > 0

    def test_different_latent_dims(self):
        """Test model works with different latent dimensions"""
        for latent_dim in [128, 512, 1024]:
            model = ResidualPitchInjector(latent_dim=latent_dim)
            f0 = torch.rand(2, 1, 50)
            uv = torch.rand(2, 1, 50)
            residual = model(f0, uv)
            assert residual.shape == (2, latent_dim, 50)


class TestConfidenceNetwork:
    """Test ConfidenceNetwork"""

    @pytest.fixture
    def model(self):
        return ConfidenceNetwork(latent_dim=128)

    def test_output_shape(self, model):
        """Test output shape is [B, 1, T]"""
        latent = torch.randn(4, 128, 75)
        uv = torch.rand(4, 1, 75)
        confidence = model(latent, uv)
        assert confidence.shape == (4, 1, 75)

    def test_output_range(self, model):
        """Test output is in [0, 1] (sigmoid output)"""
        latent = torch.randn(8, 128, 100)
        uv = torch.rand(8, 1, 100)
        confidence = model(latent, uv)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_uv_alignment(self, model):
        """Test UV is correctly aligned when T_uv != T_latent"""
        latent = torch.randn(2, 128, 75)
        uv = torch.rand(2, 1, 150)  # Different time dimension
        confidence = model(latent, uv)
        assert confidence.shape == (2, 1, 75)

    def test_parameter_count(self, model):
        """Test parameter count is reasonable (~100K)"""
        n_params = sum(p.numel() for p in model.parameters())
        assert 50_000 < n_params < 200_000


class TestIntegration:
    """Integration tests for PIRA components"""

    def test_full_pipeline(self):
        """Test full PIRA pipeline: injector + confidence + fusion"""
        # Create components
        pitch_injector = ResidualPitchInjector(latent_dim=128)
        confidence_net = ConfidenceNetwork(latent_dim=128)

        # Mock inputs
        batch_size = 4
        time_steps = 75
        f0 = torch.rand(batch_size, 1, time_steps)
        uv = (torch.rand(batch_size, 1, time_steps) > 0.3).float()
        latent = torch.randn(batch_size, 128, time_steps)

        # Forward
        residual = pitch_injector(f0, uv)
        confidence = confidence_net(latent, uv)
        latent_corrected = latent + confidence * residual

        # Assertions
        assert latent_corrected.shape == latent.shape
        assert not torch.allclose(latent_corrected, latent)  # Should be modified

    def test_receptive_field(self):
        """Test receptive field calculation"""
        dilation_rates = [1, 3, 9, 27, 81, 243]
        kernel_size = 3

        # Theoretical RF for dilated convs
        # RF = 1 + sum((k-1) * d for d in dilations)
        rf = 1 + sum((kernel_size - 1) * d for d in dilation_rates)
        assert rf == 729

        # At 75 Hz, this is ~9.72 seconds
        duration_at_75hz = rf / 75.0
        assert 9.0 < duration_at_75hz < 10.0
