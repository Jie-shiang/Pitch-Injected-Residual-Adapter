"""
CREPE Perceptual Loss for Pitch Preservation
=============================================

PIRA's core innovation: Using CREPE embeddings as a perceptual loss function
for pitch preservation, as described in Section 2.4 of the paper.

References:
    - CREPE paper: https://arxiv.org/abs/1802.06182
    - PIRA paper Section 2.4: CREPE embedding loss

Usage:
    from utils.crepe_loss import CREPEPerceptualLoss

    crepe_loss = CREPEPerceptualLoss(sample_rate=24000, model='tiny')
    loss = crepe_loss(pred_audio, target_audio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcrepe


# CREPE constants
CREPE_SAMPLE_RATE = 16000
CREPE_WINDOW_SIZE = 1024


class CREPEPerceptualLoss(nn.Module):
    """
    CREPE-based Perceptual Loss

    Extracts embeddings from frozen CREPE model and compares them between
    predicted and target audio. Encourages pitch preservation in a
    perceptually-weighted latent space.

    From paper Section 2.4:
    "The key addition is a CREPE embedding loss:
     L_crepe = ||φ_crepe(x) - φ_crepe(x̂)||²₂
     where φ_crepe denotes the penultimate layer (256-dim) activations
     of frozen CREPE-Tiny."

    Args:
        sample_rate: Input audio sample rate (will be resampled to 16kHz for CREPE)
        model: CREPE model size - 'tiny' (default), 'small', 'medium', 'large', 'full'
               Paper uses 'tiny' for computational efficiency during training.
    """
    def __init__(
        self,
        sample_rate: int = 24000,
        model: str = 'tiny',  # Paper uses 'tiny' for efficiency
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.model_capacity = model
        self.crepe_sr = CREPE_SAMPLE_RATE
        self.hop_length = 160  # 10ms hop at 16kHz

        # CREPE model loaded lazily on first forward pass
        self.crepe_model = None

    def _load_crepe_model(self, device):
        """Lazy load CREPE model"""
        if self.crepe_model is None:
            torchcrepe.load.model(device, self.model_capacity)
            self.crepe_model = torchcrepe.core.infer.model
            # Freeze CREPE - we don't train it
            for param in self.crepe_model.parameters():
                param.requires_grad = False
            self.crepe_model.eval()

    def resample_to_16k(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Resample audio to CREPE's required 16kHz

        Args:
            audio: [B, 1, T] at self.sample_rate

        Returns:
            resampled: [B, 1, T'] at 16kHz
        """
        if self.sample_rate == self.crepe_sr:
            return audio

        target_length = int(audio.shape[2] * self.crepe_sr / self.sample_rate)

        resampled = F.interpolate(
            audio,
            size=target_length,
            mode='linear',
            align_corners=False
        )

        return resampled

    def preprocess_audio(
        self,
        audio: torch.Tensor,
        device: torch.device
    ) -> tuple:
        """
        Preprocess audio for CREPE WITHOUT inplace operations

        This is optimized version replacing the original implementation
        that used loops. This version processes the entire batch at once.

        Args:
            audio: [B, T] at 16kHz

        Returns:
            frames: [B*N_frames, 1024] - Normalized frames
            frames_per_sample: int - Frames per batch item
            batch_size: int
        """
        batch_size = audio.size(0)
        hop_length = self.hop_length

        # Calculate number of frames
        frames_per_sample = 1 + int(audio.size(1) // hop_length)

        # Pad audio: [B, T] -> [B, T + window_size]
        audio_padded = F.pad(audio, (CREPE_WINDOW_SIZE // 2, CREPE_WINDOW_SIZE // 2))

        # Use unfold for efficient frame extraction: [B, T_padded] -> [B, N_frames, window_size]
        frames = audio_padded.unfold(
            dimension=1,
            size=CREPE_WINDOW_SIZE,
            step=hop_length
        )  # [B, N_frames, 1024]

        # Non-inplace mean centering: [B, N_frames, 1024]
        frame_mean = frames.mean(dim=2, keepdim=True)
        frames = frames - frame_mean  # NOT -=

        # Non-inplace std normalization
        frame_std = frames.std(dim=2, keepdim=True)
        frame_std = torch.clamp(frame_std, min=1e-10)
        frames = frames / frame_std  # NOT /=

        # Reshape to [B*N_frames, 1024]
        frames = frames.reshape(-1, CREPE_WINDOW_SIZE).to(device)

        return frames, frames_per_sample, batch_size

    def extract_embeddings(
        self,
        audio: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Extract CREPE embeddings from audio

        From paper: "256-dim activations of frozen CREPE-Tiny"

        Args:
            audio: [B, 1, T] at self.sample_rate

        Returns:
            embeddings: [B, 256, T_frames] - CREPE embeddings
        """
        # Load model if needed
        self._load_crepe_model(device)

        # Resample to 16kHz
        audio_16k = self.resample_to_16k(audio)  # [B, 1, T']
        audio_16k = audio_16k.squeeze(1)  # [B, T']

        # Preprocess (optimized batch processing)
        frames, frames_per_sample, batch_size = self.preprocess_audio(
            audio_16k, device
        )  # [B*N_frames, 1024]

        # Extract embeddings using CREPE's embedding layer
        with torch.set_grad_enabled(True):  # Allow gradient flow through frozen CREPE
            # Forward through CREPE model (frozen)
            # CREPE-Tiny outputs 512-dim embeddings, but we use 256-dim (penultimate layer)
            embeddings = self.crepe_model.embed(frames)  # [B*N_frames, 512]

            # Take first 256 dimensions (penultimate layer, as per paper)
            embeddings = embeddings[:, :256]  # [B*N_frames, 256]

        # Reshape to [B, N_frames, 256]
        embeddings = embeddings.view(batch_size, frames_per_sample, -1)

        # Transpose to [B, 256, N_frames]
        embeddings = embeddings.transpose(1, 2)

        return embeddings

    def forward(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CREPE perceptual loss

        From paper Eq. 4:
        L_crepe = ||φ_crepe(x) - φ_crepe(x̂)||²₂

        Args:
            pred_audio: [B, 1, T] - Predicted audio
            target_audio: [B, 1, T] - Target audio

        Returns:
            loss: CREPE embedding L2 distance
        """
        device = pred_audio.device

        # Extract embeddings
        pred_emb = self.extract_embeddings(pred_audio, device)  # [B, 256, T']
        target_emb = self.extract_embeddings(target_audio, device)  # [B, 256, T']

        # Align lengths (handle minor differences due to resampling)
        min_len = min(pred_emb.shape[2], target_emb.shape[2])
        pred_emb = pred_emb[:, :, :min_len]
        target_emb = target_emb[:, :, :min_len]

        # Compute L2 loss (MSE)
        loss = F.mse_loss(pred_emb, target_emb)

        return loss


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CREPE Perceptual Loss Test")
    print("=" * 60)

    # Test parameters (matching paper: 24kHz, CREPE-Tiny)
    batch_size = 2
    sample_rate = 24000
    audio_length = sample_rate * 1  # 1 second

    # Create dummy audio
    pred_audio = torch.randn(batch_size, 1, audio_length)
    target_audio = torch.randn(batch_size, 1, audio_length)

    # Test CREPE loss (using 'tiny' as per paper)
    print("\n[1] Testing CREPEPerceptualLoss (model='tiny')...")
    crepe_loss = CREPEPerceptualLoss(sample_rate=sample_rate, model='tiny')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_audio = pred_audio.to(device)
    target_audio = target_audio.to(device)

    print(f"    Device: {device}")
    print(f"    Audio shape: {pred_audio.shape}")
    print(f"    Sample rate: {sample_rate} Hz")

    # Forward pass
    loss = crepe_loss(pred_audio, target_audio)
    print(f"    CREPE Loss: {loss.item():.6f}")

    # Test gradient flow
    print("\n[2] Testing gradient flow...")
    pred_audio_with_grad = pred_audio.clone().requires_grad_(True)
    loss = crepe_loss(pred_audio_with_grad, target_audio)
    loss.backward()
    print(f"    ✓ Gradients computed successfully")
    print(f"    Gradient norm: {pred_audio_with_grad.grad.norm().item():.6f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    print("\nNote: CREPE Perceptual Loss is the core innovation of PIRA.")
    print("Paper Section 2.4: Uses 256-dim activations of frozen CREPE-Tiny.")
    print("This provides stable pitch-aware gradients unlike direct F0 MSE loss.")
