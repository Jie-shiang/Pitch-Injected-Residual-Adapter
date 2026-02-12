"""
PIRA: Pitch-Injected Residual Adapter
======================================

Core architecture for tonal language preservation in neural audio codecs.

This module contains codec-agnostic components:
- ResidualPitchInjector: Converts F0/UV to latent-space corrections
- ConfidenceNetwork: Learns per-frame gating for residual injection
- DilatedConvBlock: Captures tone sandhi dependencies

Usage:
    from models.pira_core import ResidualPitchInjector, ConfidenceNetwork

    # Initialize components
    pitch_injector = ResidualPitchInjector(
        latent_dim=128,      # Codec-specific: match your codec's latent dimension
        hidden_dim=256,      # Universal: 256 works well for all codecs
        dilation_rates=[1,3,9,27,81,243]  # Universal: captures tone sandhi context
    )

    confidence_net = ConfidenceNetwork(latent_dim=128)

    # Forward pass
    residual = pitch_injector(f0, uv)  # [B, latent_dim, T]
    confidence = confidence_net(latent, uv)  # [B, 1, T]
    latent_corrected = latent + confidence * residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    """
    Dilated Convolution Block with residual connection

    Captures long-range dependencies for tone sandhi modeling.
    Uses depthwise-separable convolution for efficiency.

    Args:
        channels: Number of channels
        dilation: Dilation rate (higher values capture longer context)
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()

        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels  # Depthwise convolution for efficiency
        )
        self.norm = nn.GroupNorm(8, channels)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: [B, channels, T]
        Returns:
            out: [B, channels, T] with residual connection
        """
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class ResidualPitchInjector(nn.Module):
    """
    Residual Pitch Injector

    Transforms F0/UV information into latent-space corrections using
    dilated convolutions to capture tone sandhi context.

    Architecture:
        Input [F0, UV] → Conv1d → Dilated Conv Blocks → Output Residual

    Args:
        latent_dim: Codec latent dimension (codec-specific)
                    - EnCodec: 128, DAC/BigCodec: 1024, Mimi/WavTokenizer: 512
        hidden_dim: Hidden layer dimension (D_h in paper, default=256)
        dilation_rates: Base-3 dilation [1,3,9,27,81,243] (RF=729 frames, 9.7s at 75Hz)
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        dilation_rates: list = [1, 3, 9, 27, 81, 243]
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection: [F0, UV] -> hidden_dim
        self.input_proj = nn.Conv1d(2, hidden_dim, kernel_size=1)

        # Dilated conv stack (captures tone sandhi context)
        self.conv_blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, d) for d in dilation_rates
        ])

        # Output projection: hidden_dim -> latent_dim
        self.output_proj = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, f0: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            f0: [B, 1, T] - Normalized F0 contour (0-1 range)
            uv: [B, 1, T] - Voicing flags (0=unvoiced, 1=voiced)

        Returns:
            residual: [B, latent_dim, T] - Latent space correction
        """
        # Concatenate F0 and UV
        x = torch.cat([f0, uv], dim=1)  # [B, 2, T]

        # Input projection
        x = self.input_proj(x)  # [B, hidden_dim, T]

        # Dilated conv blocks with residual connections
        for block in self.conv_blocks:
            x = block(x)  # [B, hidden_dim, T]

        # Output projection to latent space
        residual = self.output_proj(x)  # [B, latent_dim, T]

        return residual


class ConfidenceNetwork(nn.Module):
    """
    Confidence Network

    Learns per-frame gating to modulate residual injection strength.

    Key idea:
        - Voiced regions with clear pitch → high confidence (α ≈ 1)
        - Unvoiced/checked tone regions → low confidence (α ≈ 0)
        - Prevents corrupting spectral cues where F0 isn't primary

    Args:
        latent_dim: Codec latent dimension (codec-specific)
    """
    def __init__(self, latent_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            # Input: [latent, uv] concatenated
            nn.Conv1d(latent_dim + 1, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),

            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output confidence in [0, 1]
        )

    def forward(self, latent: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            latent: [B, latent_dim, T_latent] - Codec latent features
            uv: [B, 1, T_uv] - Voicing flags (may need alignment)

        Returns:
            confidence: [B, 1, T_latent] - Confidence weights in [0, 1]
        """
        # Align UV to latent time dimension if needed
        if uv.shape[2] != latent.shape[2]:
            uv_aligned = F.interpolate(
                uv,
                size=latent.shape[2],
                mode='nearest'  # UV is discrete (0 or 1)
            )
        else:
            uv_aligned = uv

        # Concatenate latent and UV
        x = torch.cat([latent, uv_aligned], dim=1)  # [B, latent_dim+1, T]

        # Predict confidence
        confidence = self.net(x)  # [B, 1, T]

        return confidence


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("PIRA Core Components Test")
    print("=" * 60)

    # Test parameters (using EnCodec as example)
    batch_size = 4
    latent_dim = 128  # EnCodec latent dimension
    time_steps = 75   # 1 second at 75 Hz

    # Create components
    print("\n[1] Creating ResidualPitchInjector...")
    pitch_injector = ResidualPitchInjector(
        latent_dim=latent_dim,
        hidden_dim=256,
        dilation_rates=[1, 3, 9, 27, 81, 243]
    )
    print(f"    Trainable parameters: {sum(p.numel() for p in pitch_injector.parameters()):,}")

    print("\n[2] Creating ConfidenceNetwork...")
    confidence_net = ConfidenceNetwork(latent_dim=latent_dim)
    print(f"    Trainable parameters: {sum(p.numel() for p in confidence_net.parameters()):,}")

    # Test forward pass
    print("\n[3] Testing forward pass...")

    # Mock inputs
    f0 = torch.rand(batch_size, 1, time_steps)  # Normalized F0
    uv = (torch.rand(batch_size, 1, time_steps) > 0.3).float()  # 70% voiced
    latent = torch.randn(batch_size, latent_dim, time_steps)  # Mock latent

    # Forward through pitch injector
    residual = pitch_injector(f0, uv)
    print(f"    Residual shape: {residual.shape}")
    assert residual.shape == (batch_size, latent_dim, time_steps)

    # Forward through confidence network
    confidence = confidence_net(latent, uv)
    print(f"    Confidence shape: {confidence.shape}")
    print(f"    Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    assert confidence.shape == (batch_size, 1, time_steps)
    assert confidence.min() >= 0.0 and confidence.max() <= 1.0

    # Test confidence-weighted fusion
    latent_corrected = latent + confidence * residual
    print(f"    Corrected latent shape: {latent_corrected.shape}")
    assert latent_corrected.shape == latent.shape

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    # Print receptive field info
    print("\nReceptive Field Analysis:")
    print(f"  Dilation rates: {[1, 3, 9, 27, 81, 243]}")
    print(f"  Theoretical RF: 729 frames")
    print(f"  At 75 Hz: 9.72 seconds")
    print(f"  At 50 Hz (DAC): 14.58 seconds")
    print(f"  At 80 Hz (BigCodec): 9.11 seconds")
    print(f"  At 12.5 Hz (Mimi): 58.32 seconds")
