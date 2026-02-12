"""
PIRA Training Script
====================

Universal training script for PIRA (Pitch-Injected Residual Adapter).

This script is codec-agnostic - it works with any neural audio codec by
adapting to codec-specific parameters via the configuration file.

Usage:
    python train.py --config configs/encodec_example.yaml

Required:
    - Your neural audio codec model (e.g., encodec, dac, mimi, etc.)
    - Configuration file specifying codec parameters
    - Dataset with F0 features

The script will:
    1. Load your codec model (frozen)
    2. Create PIRA adapter components
    3. Train on your dataset
    4. Save checkpoints
    5. Validate and early stop

For details, see README.md
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# PIRA components
from models.pira_core import ResidualPitchInjector, ConfidenceNetwork
from models.f0_quantizer import F0UVQuantizer
from utils.losses import MultiResolutionSTFTLoss, MelSpectrogramLoss, F0Loss
from utils.crepe_loss import CREPEPerceptualLoss


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_pira_adapter(config, codec_model):
    """
    Create PIRA adapter components

    Args:
        config: Configuration dict
        codec_model: Your pretrained codec (will be frozen)

    Returns:
        pitch_injector: Trainable ResidualPitchInjector
        confidence_net: Trainable ConfidenceNetwork (if enabled)
        f0_quantizer: F0UVQuantizer (frozen, if enabled)
    """
    model_cfg = config['model']
    codec_cfg = config['codec']

    # Create Pitch Injector
    pitch_injector = ResidualPitchInjector(
        latent_dim=model_cfg['latent_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        dilation_rates=model_cfg['dilation_rates']
    )

    # Create Confidence Network (if enabled)
    confidence_net = None
    if model_cfg.get('use_confidence', True):
        confidence_net = ConfidenceNetwork(latent_dim=model_cfg['latent_dim'])

    # Create F0 Quantizer (if enabled)
    f0_quantizer = None
    if model_cfg.get('use_f0_quantizer', True):
        f0_quantizer = F0UVQuantizer(
            n_f0_bins=model_cfg.get('f0_n_bins', 256),
            frame_rate=codec_cfg['frame_rate']
        )
        # Freeze quantizer (it's non-trainable)
        for param in f0_quantizer.parameters():
            param.requires_grad = False

    # Freeze codec model
    for param in codec_model.parameters():
        param.requires_grad = False

    # Print model info
    pitch_params = sum(p.numel() for p in pitch_injector.parameters())
    conf_params = sum(p.numel() for p in confidence_net.parameters()) if confidence_net else 0
    total_params = pitch_params + conf_params

    print(f"\n{'='*60}")
    print("PIRA Adapter Created:")
    print(f"  ResidualPitchInjector: {pitch_params:,} parameters")
    if confidence_net:
        print(f"  ConfidenceNetwork: {conf_params:,} parameters")
    print(f"  Total trainable: {total_params:,} parameters")
    if f0_quantizer:
        bitrate_info = f0_quantizer.get_bitrate_info()
        print(f"  F0/UV overhead: {bitrate_info['bitrate_kbps']:.3f} kbps")
    print(f"{'='*60}\n")

    return pitch_injector, confidence_net, f0_quantizer


def create_loss_functions(config, device):
    """Create loss functions based on config"""
    loss_cfg = config['loss']
    codec_cfg = config['codec']

    losses = {}

    # STFT Loss
    if loss_cfg.get('lambda_stft', 0) > 0:
        losses['stft'] = MultiResolutionSTFTLoss().to(device)

    # Mel Loss
    if loss_cfg.get('lambda_mel', 0) > 0:
        losses['mel'] = MelSpectrogramLoss(
            sample_rate=codec_cfg['sample_rate']
        ).to(device)

    # F0 Loss
    if loss_cfg.get('lambda_f0', 0) > 0:
        losses['f0'] = F0Loss(loss_type='l1').to(device)

    # CREPE Perceptual Loss (PIRA's core innovation!)
    if loss_cfg.get('lambda_crepe', 0) > 0:
        crepe_model = loss_cfg.get('crepe_model', 'tiny')
        losses['crepe'] = CREPEPerceptualLoss(
            sample_rate=codec_cfg['sample_rate'],
            model=crepe_model
        ).to(device)
        print(f"✓ CREPE Perceptual Loss initialized (model={crepe_model})")

    return losses


def compute_total_loss(losses_dict, loss_weights, pred_audio, target_audio,
                       pred_f0=None, target_f0=None, target_uv=None):
    """
    Compute weighted sum of all losses

    Args:
        losses_dict: Dict of loss functions
        loss_weights: Dict of loss weights (lambdas)
        pred_audio: Predicted audio [B, 1, T]
        target_audio: Target audio [B, 1, T]
        pred_f0: Predicted F0 [B, 1, T'] (optional, for F0 loss)
        target_f0: Target F0 [B, 1, T'] (optional, for F0 loss)
        target_uv: Target UV [B, 1, T'] (optional, for F0 loss)

    Returns:
        total_loss: Weighted sum of losses
        loss_components: Dict of individual loss values (for logging)
    """
    total_loss = 0.0
    loss_components = {}

    # L1 Loss
    if loss_weights.get('lambda_l1', 0) > 0:
        loss_l1 = nn.functional.l1_loss(pred_audio, target_audio)
        total_loss += loss_weights['lambda_l1'] * loss_l1
        loss_components['l1'] = loss_l1.item()

    # STFT Loss
    if 'stft' in losses_dict and loss_weights.get('lambda_stft', 0) > 0:
        loss_stft = losses_dict['stft'](pred_audio, target_audio)
        total_loss += loss_weights['lambda_stft'] * loss_stft
        loss_components['stft'] = loss_stft.item()

    # Mel Loss
    if 'mel' in losses_dict and loss_weights.get('lambda_mel', 0) > 0:
        loss_mel = losses_dict['mel'](pred_audio, target_audio)
        total_loss += loss_weights['lambda_mel'] * loss_mel
        loss_components['mel'] = loss_mel.item()

    # F0 Loss
    if 'f0' in losses_dict and loss_weights.get('lambda_f0', 0) > 0:
        if pred_f0 is not None and target_f0 is not None and target_uv is not None:
            loss_f0 = losses_dict['f0'](pred_f0, target_f0, target_uv)
            total_loss += loss_weights['lambda_f0'] * loss_f0
            loss_components['f0'] = loss_f0.item()

    # CREPE Loss (PIRA's core innovation!)
    if 'crepe' in losses_dict and loss_weights.get('lambda_crepe', 0) > 0:
        loss_crepe = losses_dict['crepe'](pred_audio, target_audio)
        total_loss += loss_weights['lambda_crepe'] * loss_crepe
        loss_components['crepe'] = loss_crepe.item()

    return total_loss, loss_components


def train_epoch(model, dataloader, optimizer, losses, loss_weights, device, config):
    """
    Train for one epoch

    NOTE: This function requires you to implement the forward pass logic
    specific to your codec. See the example in the main training loop below.
    """
    model.train()
    total_loss = 0.0
    loss_accum = {key: 0.0 for key in ['l1', 'stft', 'mel', 'f0', 'crepe']}

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        audio = batch['audio'].to(device)
        f0 = batch['f0'].to(device)
        uv = batch['uv'].to(device)

        # TODO: Implement forward pass with your codec
        # This is codec-specific and must be customized!
        #
        # Example structure:
        # 1. Encode audio with frozen codec -> latent
        # 2. Generate pitch residual with PIRA
        # 3. Fuse: latent_corrected = latent + confidence * residual
        # 4. Decode with frozen codec decoder
        # 5. Compute losses
        #
        # See README.md for codec-specific examples.

        raise NotImplementedError(
            "You must implement the forward pass for your specific codec. "
            "See README.md section 'Training: Codec-Specific Forward Pass' for examples."
        )

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        if config['training'].get('gradient_clip_norm'):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip_norm']
            )
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        for key, val in loss_components.items():
            loss_accum[key] += val

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_accum.items()}

    return avg_loss, avg_components


def validate(model, dataloader, losses, loss_weights, device, config):
    """Validation loop (similar to train_epoch but without gradients)"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # TODO: Same as train_epoch - implement codec-specific forward pass
            raise NotImplementedError("Implement validation forward pass")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train PIRA adapter")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"✓ Configuration loaded from {args.config}")

    # Device
    device = torch.device('cuda' if config['device']['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # TODO: Load your codec model
    # This is codec-specific!
    #
    # Example for EnCodec:
    #   from encodec import EncodecModel
    #   codec_model = EncodecModel.encodec_model_24khz()
    #   codec_model = codec_model.to(device)
    #
    # Example for DAC:
    #   import dac
    #   codec_model = dac.DAC.load(dac.utils.download(model_type="16khz"))
    #   codec_model = codec_model.to(device)
    #
    raise NotImplementedError(
        "You must load your codec model here. "
        "See README.md section 'Setup: Loading Your Codec' for examples."
    )

    # Create PIRA components
    pitch_injector, confidence_net, f0_quantizer = create_pira_adapter(config, codec_model)
    pitch_injector = pitch_injector.to(device)
    if confidence_net:
        confidence_net = confidence_net.to(device)

    # TODO: Load dataset
    # You need to implement a dataset that provides:
    #   - audio: [1, T] waveform
    #   - f0: [1, T'] F0 contour (normalized 0-1)
    #   - uv: [1, T'] voicing flags (0 or 1)
    #
    # See README.md section 'Dataset Preparation' for guidance.
    raise NotImplementedError(
        "You must implement dataset loading. "
        "See README.md section 'Dataset Preparation' for examples."
    )

    # Create loss functions
    losses = create_loss_functions(config, device)
    loss_weights = {
        'lambda_l1': config['loss'].get('lambda_l1', 0),
        'lambda_stft': config['loss'].get('lambda_stft', 0),
        'lambda_mel': config['loss'].get('lambda_mel', 0),
        'lambda_f0': config['loss'].get('lambda_f0', 0),
        'lambda_crepe': config['loss'].get('lambda_crepe', 0),
    }

    # Optimizer
    trainable_params = list(pitch_injector.parameters())
    if confidence_net:
        trainable_params += list(confidence_net.parameters())

    optimizer = optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )

    # Scheduler (optional)
    scheduler = None
    if config['training'].get('scheduler_type') == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['training'].get('scheduler_patience', 5),
            factor=config['training'].get('scheduler_factor', 0.5)
        )

    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")

        # Train
        train_loss, train_components = train_epoch(
            model=(pitch_injector, confidence_net, f0_quantizer, codec_model),
            dataloader=train_loader,
            optimizer=optimizer,
            losses=losses,
            loss_weights=loss_weights,
            device=device,
            config=config
        )

        # Validate
        val_loss = validate(
            model=(pitch_injector, confidence_net, f0_quantizer, codec_model),
            dataloader=val_loader,
            losses=losses,
            loss_weights=loss_weights,
            device=device,
            config=config
        )

        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Scheduler step
        if scheduler:
            scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_dir = config['training']['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'pitch_injector': pitch_injector.state_dict(),
                'confidence_net': confidence_net.state_dict() if confidence_net else None,
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }

            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
