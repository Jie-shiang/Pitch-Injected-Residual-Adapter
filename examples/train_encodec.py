"""
Complete Training Example for EnCodec with PIRA

This script shows how to:
1. Load pre-trained EnCodec
2. Create PIRA components
3. Train with CREPE perceptual loss
4. Save checkpoints

Based on the paper's experiments with EnCodec @ 24kHz, 1.5 kbps.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.pira_core import ResidualPitchInjector, ConfidenceNetwork
from models.f0_quantizer import F0UVQuantizer
from utils.crepe_loss import CREPEPerceptualLoss


class PIRAEnCodec(nn.Module):
    """
    PIRA-enhanced EnCodec

    Wraps EnCodec with PIRA components for pitch-preserving neural audio coding.
    """
    def __init__(
        self,
        encodec_bandwidth=1.5,
        use_confidence=True,
        use_f0_quantizer=True,
        f0_n_bins=16,  # 4-bit as per paper
    ):
        super().__init__()

        # Load pre-trained EnCodec
        from encodec import EncodecModel
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(encodec_bandwidth)

        # Freeze EnCodec
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.encodec.eval()

        # PIRA components
        self.pitch_injector = ResidualPitchInjector(
            latent_dim=128,  # EnCodec latent dimension
            hidden_dim=256,
            dilation_rates=[1, 3, 9, 27, 81, 243]
        )

        self.use_confidence = use_confidence
        if use_confidence:
            self.confidence_net = ConfidenceNetwork(latent_dim=128)

        self.use_f0_quantizer = use_f0_quantizer
        if use_f0_quantizer:
            self.f0_quantizer = F0UVQuantizer(
                n_f0_bins=f0_n_bins,
                frame_rate=75.0  # EnCodec frame rate
            )

    def get_trainable_parameters(self):
        """Return only trainable parameters (PIRA components)"""
        params = list(self.pitch_injector.parameters())
        if self.use_confidence:
            params += list(self.confidence_net.parameters())
        return params

    def forward(self, audio, f0, uv):
        """
        Forward pass

        Args:
            audio: [B, 1, T] - Input audio @ 24kHz
            f0: [B, 1, T_f0] - Normalized F0 (0-1 range)
            uv: [B, 1, T_f0] - Voicing flags (0 or 1)

        Returns:
            reconstructed: [B, 1, T] - Reconstructed audio
        """
        # Encode with frozen EnCodec
        with torch.no_grad():
            # EnCodec encode
            encoded_frames = self.encodec.encode(audio)
            codes = encoded_frames[0][0]  # [B, K, T_latent]

            # Get latent from codes
            codes_t = codes.transpose(0, 1)  # [K, B, T_latent]
            latent = self.encodec.quantizer.decode(codes_t)  # [B, 128, T_latent]

        # Align F0/UV to latent time dimension
        if f0.shape[2] != latent.shape[2]:
            f0_aligned = F.interpolate(f0, size=latent.shape[2], mode='linear', align_corners=False)
            uv_aligned = F.interpolate(uv, size=latent.shape[2], mode='nearest')
        else:
            f0_aligned, uv_aligned = f0, uv

        # Optional: F0 quantization (reduces bitrate)
        if self.use_f0_quantizer:
            f0_aligned, uv_aligned = self.f0_quantizer(f0_aligned, uv_aligned)

        # Generate pitch residual
        residual = self.pitch_injector(f0_aligned, uv_aligned)  # [B, 128, T_latent]

        # Confidence-weighted fusion
        if self.use_confidence:
            confidence = self.confidence_net(latent, uv_aligned)  # [B, 1, T_latent]
            latent_corrected = latent + confidence * residual
        else:
            latent_corrected = latent + residual

        # Decode with frozen decoder
        with torch.no_grad():
            reconstructed = self.encodec.decoder(latent_corrected)  # [B, 1, T]

        return reconstructed


def create_losses(device):
    """Create loss functions (as per paper)"""
    losses = {}

    # L1 Loss
    losses['l1'] = nn.L1Loss()

    # Multi-Resolution STFT Loss
    from utils.losses import MultiResolutionSTFTLoss
    losses['stft'] = MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512, 256],
        hop_sizes=[512, 256, 128, 64],
        win_sizes=[2048, 1024, 512, 256]
    ).to(device)

    # Mel-Spectrogram Loss
    from utils.losses import MelSpectrogramLoss
    losses['mel'] = MelSpectrogramLoss(
        sample_rate=24000,
        n_fft_list=[2048, 1024, 512],
        hop_length_list=[512, 256, 128],
        n_mels=80
    ).to(device)

    # CREPE Perceptual Loss (Core innovation!)
    losses['crepe'] = CREPEPerceptualLoss(
        sample_rate=24000,
        model='tiny'  # Paper uses CREPE-Tiny
    )

    return losses


def compute_total_loss(pred_audio, target_audio, losses, loss_weights):
    """
    Compute total loss with paper's weights

    Paper uses: λ_l1=0.1, λ_stft=1.0, λ_mel=1.0, λ_crepe=30
    """
    loss_dict = {}

    loss_dict['l1'] = losses['l1'](pred_audio, target_audio)
    loss_dict['stft'] = losses['stft'](pred_audio, target_audio)
    loss_dict['mel'] = losses['mel'](pred_audio, target_audio)
    loss_dict['crepe'] = losses['crepe'](pred_audio, target_audio)

    total_loss = (
        loss_weights['lambda_l1'] * loss_dict['l1'] +
        loss_weights['lambda_stft'] * loss_dict['stft'] +
        loss_weights['lambda_mel'] * loss_dict['mel'] +
        loss_weights['lambda_crepe'] * loss_dict['crepe']
    )

    loss_dict['total'] = total_loss
    return total_loss, loss_dict


def train_epoch(model, dataloader, optimizer, losses, loss_weights, device, epoch):
    """Train one epoch"""
    model.train()
    total_losses = {'total': 0, 'l1': 0, 'stft': 0, 'mel': 0, 'crepe': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        audio = batch['audio'].to(device)
        f0 = batch['f0'].to(device)
        uv = batch['uv'].to(device)

        # Forward
        pred_audio = model(audio, f0, uv)

        # Compute losses
        total_loss, loss_dict = compute_total_loss(
            pred_audio, audio, losses, loss_weights
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
        optimizer.step()

        # Accumulate
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total'].item():.4f}",
            'crepe': f"{loss_dict['crepe'].item():.4f}"
        })

    # Average
    for key in total_losses.keys():
        total_losses[key] /= num_batches

    return total_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--audio_root', type=str, required=True, help='Path to audio directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints_encodec', help='Save directory')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("PIRA Training for EnCodec")
    print("=" * 60)
    print(f"Dataset: {args.data_csv}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Create model
    print("\nCreating PIRA-EnCodec model...")
    model = PIRAEnCodec(
        encodec_bandwidth=1.5,  # Paper uses 1.5 kbps
        use_confidence=True,
        use_f0_quantizer=True,
        f0_n_bins=16  # 4-bit as per paper
    ).to(device)

    n_trainable = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"✓ Trainable parameters: {n_trainable:,}")

    # Create dataset (you need to implement this based on your data)
    # Example:
    # from your_dataset import YourDataset
    # train_dataset = YourDataset(args.data_csv, args.audio_root)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print("\n[WARNING] You need to implement your dataset class!")
    print("Dataset should return: {'audio': [B,1,T], 'f0': [B,1,T'], 'uv': [B,1,T']}")
    print("See paper Section 3.1 for F0 extraction using WORLD Harvest")
    return

    # Create losses (paper weights)
    print("\nCreating losses...")
    losses = create_losses(device)
    loss_weights = {
        'lambda_l1': 0.1,
        'lambda_stft': 1.0,
        'lambda_mel': 1.0,
        'lambda_crepe': 30  # Core innovation weight
    }
    print("✓ Loss weights (from paper):")
    for k, v in loss_weights.items():
        print(f"  {k}: {v}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=args.lr,
        betas=(0.8, 0.99),
        weight_decay=0.01
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        train_losses = train_epoch(
            model, train_loader, optimizer, losses, loss_weights, device, epoch
        )

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Loss: {train_losses['total']:.4f}")
        print(f"  L1: {train_losses['l1']:.4f}, STFT: {train_losses['stft']:.4f}")
        print(f"  Mel: {train_losses['mel']:.2f}, CREPE: {train_losses['crepe']:.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'pitch_injector_state_dict': model.pitch_injector.state_dict(),
                'confidence_net_state_dict': model.confidence_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': train_losses['total']
            }
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, save_path)
            print(f"✓ Saved checkpoint to {save_path}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
