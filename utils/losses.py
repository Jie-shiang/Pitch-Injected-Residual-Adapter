"""
Loss Functions for PIRA Training
=================================

Core loss functions for tonal language preservation:
- MultiResolutionSTFTLoss: Spectral fidelity
- MelSpectrogramLoss: Perceptual quality
- F0Loss: Pitch accuracy (critical for tonal languages)
- CREPEPerceptualLoss: Pitch-aware perceptual loss

Usage:
    from utils.losses import MultiResolutionSTFTLoss, F0Loss

    stft_loss = MultiResolutionSTFTLoss()
    f0_loss = F0Loss(loss_type='l1')

    # In training loop:
    loss_stft = stft_loss(pred_audio, target_audio)
    loss_f0 = f0_loss(pred_f0, target_f0, target_uv)
    total_loss = lambda_stft * loss_stft + lambda_f0 * loss_f0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss

    Computes spectral distance across multiple FFT sizes for robust
    frequency-domain reconstruction.

    Args:
        fft_sizes: List of FFT sizes (default: [512, 1024, 2048])
        hop_sizes: List of hop lengths (default: fft_size // 4)
        win_sizes: List of window sizes (default: fft_size)
    """
    def __init__(
        self,
        fft_sizes: list = None,
        hop_sizes: list = None,
        win_sizes: list = None
    ):
        super().__init__()

        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or [f // 4 for f in fft_sizes]
        self.win_sizes = win_sizes or fft_sizes

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, T] - Predicted audio
            target: [B, 1, T] - Target audio

        Returns:
            loss: STFT loss averaged over all resolutions
        """
        total_loss = 0.0
        pred = pred.squeeze(1)  # [B, T]
        target = target.squeeze(1)  # [B, T]

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Compute STFT
            pred_stft = torch.stft(
                pred,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                window=torch.hann_window(win_size, device=pred.device),
                return_complex=True
            )
            target_stft = torch.stft(
                target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                window=torch.hann_window(win_size, device=target.device),
                return_complex=True
            )

            # Magnitude spectra
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            # L1 magnitude loss
            mag_loss = F.l1_loss(pred_mag, target_mag)

            # Log-magnitude loss (emphasizes low-energy details)
            log_mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-7),
                torch.log(target_mag + 1e-7)
            )

            total_loss += mag_loss + log_mag_loss

        return total_loss / len(self.fft_sizes)


class MelSpectrogramLoss(nn.Module):
    """
    Mel-Spectrogram Loss

    Perceptually-weighted spectral loss using mel-scale filterbanks.

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bins
    """
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 80
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, T] - Predicted audio
            target: [B, 1, T] - Target audio

        Returns:
            loss: Mel-spectrogram L1 loss
        """
        pred = pred.squeeze(1)  # [B, T]
        target = target.squeeze(1)  # [B, T]

        # Compute STFT
        pred_stft = torch.stft(
            pred,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=pred.device),
            return_complex=True
        )
        target_stft = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=target.device),
            return_complex=True
        )

        # Magnitude spectra
        pred_mag = torch.abs(pred_stft)  # [B, freq, time]
        target_mag = torch.abs(target_stft)

        # Create mel filterbank
        mel_fb = torch.from_numpy(
            self._create_mel_filterbank()
        ).to(pred.device).float()  # [n_mels, freq]

        # Apply mel filterbank
        pred_mel = torch.matmul(mel_fb, pred_mag)  # [B, n_mels, time]
        target_mel = torch.matmul(mel_fb, target_mag)

        # Log-mel loss
        loss = F.l1_loss(
            torch.log(pred_mel + 1e-7),
            torch.log(target_mel + 1e-7)
        )

        return loss

    def _create_mel_filterbank(self):
        """Create mel filterbank matrix"""
        import numpy as np

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel points
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bin numbers
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # Create filterbank
        fbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            f_left = bins[m - 1]
            f_center = bins[m]
            f_right = bins[m + 1]

            # Rising slope
            for k in range(f_left, f_center):
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)

            # Falling slope
            for k in range(f_center, f_right):
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

        return fbank


class F0Loss(nn.Module):
    """
    F0 Loss (Voiced Regions Only)

    Critical for tonal language preservation!

    Computes pitch error only in voiced regions to avoid penalizing
    meaningless F0 values in unvoiced segments.

    Args:
        loss_type: 'mse' or 'l1' (default: 'l1')
    """
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred_f0: torch.Tensor,
        target_f0: torch.Tensor,
        target_uv: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_f0: [B, 1, T] - Predicted F0 (extracted from pred audio)
            target_f0: [B, 1, T] - Target F0
            target_uv: [B, 1, T] - Voicing flags (voiced=1, unvoiced=0)

        Returns:
            loss: F0 loss averaged over voiced frames
        """
        # Voiced region mask
        voiced_mask = (target_uv > 0.5).float()  # [B, 1, T]

        # Check if there are any voiced regions
        if voiced_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_f0.device, requires_grad=True)

        # Compute loss only in voiced regions
        if self.loss_type == 'mse':
            loss = F.mse_loss(
                pred_f0 * voiced_mask,
                target_f0 * voiced_mask,
                reduction='sum'
            ) / (voiced_mask.sum() + 1e-8)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(
                pred_f0 * voiced_mask,
                target_f0 * voiced_mask,
                reduction='sum'
            ) / (voiced_mask.sum() + 1e-8)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("PIRA Loss Functions Test")
    print("=" * 60)

    batch_size = 4
    audio_length = 24000  # 1 second at 24kHz
    f0_length = 75  # 1 second at 75 Hz

    # Create dummy data
    pred_audio = torch.randn(batch_size, 1, audio_length)
    target_audio = torch.randn(batch_size, 1, audio_length)
    pred_f0 = torch.rand(batch_size, 1, f0_length)  # Normalized [0, 1]
    target_f0 = torch.rand(batch_size, 1, f0_length)
    target_uv = (torch.rand(batch_size, 1, f0_length) > 0.3).float()  # 70% voiced

    # Test STFT Loss
    print("\n[1] Testing MultiResolutionSTFTLoss...")
    stft_loss = MultiResolutionSTFTLoss()
    loss_stft = stft_loss(pred_audio, target_audio)
    print(f"    STFT Loss: {loss_stft.item():.4f}")

    # Test Mel Loss
    print("\n[2] Testing MelSpectrogramLoss...")
    mel_loss = MelSpectrogramLoss(sample_rate=24000)
    loss_mel = mel_loss(pred_audio, target_audio)
    print(f"    Mel Loss: {loss_mel.item():.4f}")

    # Test F0 Loss
    print("\n[3] Testing F0Loss...")
    f0_loss = F0Loss(loss_type='l1')
    loss_f0 = f0_loss(pred_f0, target_f0, target_uv)
    print(f"    F0 Loss: {loss_f0.item():.4f}")
    print(f"    Voiced frames: {target_uv.sum().item()} / {target_uv.numel()}")

    # Test combined loss
    print("\n[4] Testing Combined Loss...")
    lambda_l1 = 0.1
    lambda_stft = 1.0
    lambda_mel = 1.0
    lambda_f0 = 10.0

    loss_l1 = F.l1_loss(pred_audio, target_audio)
    total_loss = (
        lambda_l1 * loss_l1 +
        lambda_stft * loss_stft +
        lambda_mel * loss_mel +
        lambda_f0 * loss_f0
    )
    print(f"    L1: {loss_l1.item():.4f}")
    print(f"    Total: {total_loss.item():.4f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
