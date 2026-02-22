# PIRA: Pitch-Injected Residual Adapter

**Lightweight, plug-and-play adapter for preserving tonal information in neural audio codecs**

---

## Overview

PIRA is a **codec-agnostic adapter** that preserves pitch and tonal information in neural audio codecs (EnCodec, DAC, Mimi, etc.). It addresses a critical limitation: existing codecs degrade fundamental frequency (F0), causing severe quality issues for **tonal languages** (Mandarin, Cantonese, Vietnamese, Hokkien).

### Key Features

- **Plug-and-Play**: Works with any pretrained codec
- **Lightweight**: Only 1.25M trainable parameters
- **Low Bitrate**: F0/UV adds ~0.4 kbps overhead
- **Zero Forgetting**: Fully removable - original codec unchanged when disabled
- **Real-Time**: RTF < 0.01 (190× real-time on A100)

### Core Innovation

**CREPE Perceptual Loss**: Instead of unstable frame-level F0 regression, PIRA uses embeddings from a frozen CREPE (pitch estimation) model as a perceptual loss, providing stable pitch-aware gradients.

### Results (from paper)

| Dataset | dTER Baseline | dTER + PIRA | Improvement |
|---------|---------------|-------------|-------------|
| Hokkien | 0.267         | 0.171       | **-36.0%**  |
| Cantonese | 0.236       | 0.098       | **-58.5%**  |
| Vietnamese | 0.490      | 0.120       | **-75.5%**  |

*dTER = differential Tone Error Rate (codec-induced tone degradation)*

---

## Architecture

```
Input Audio → [Frozen Codec Encoder] → Latent [B, D, T]
                                           ↓
                        ┌──────────────────┴───────────────────┐
                        │  F0/UV → Pitch Injector (trainable)  │
                        │  Latent + UV → Confidence (trainable)│
                        │  Residual Fusion: z' = z + α ⊙ R    │
                        └──────────────────┬───────────────────┘
                                           ↓
              [Frozen Codec Decoder] → Reconstructed Audio
                                           ↓
                                    CREPE Embedding Loss
```

## Installation

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- CUDA ≥ 11.8 (recommended)

### Install

```bash
git clone https://github.com/your-username/PIRA.git
cd PIRA
pip install -r requirements.txt

# Install PyTorch with CUDA (if needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install your codec (e.g., EnCodec)
pip install encodec
```

---

## Quick Start

### 1. Prepare Data

Create a CSV file:

```csv
filename,text
audio/sample_001.wav,这是一个测试
audio/sample_002.wav,粤语测试音频
```

Audio requirements:
- Format: WAV, mono
- Sample rate: Match codec (16kHz or 24kHz)
- F0 extraction: Use WORLD Harvest (see `examples/extract_f0.py`)

### 2. Configure

Edit `configs/encodec_hokkien.yaml`:

```yaml
codec:
  type: "encodec"
  sample_rate: 24000
  bandwidth: 1.5
  latent_dim: 128  # CRITICAL: Must match your codec!

loss:
  lambda_crepe: 30  # Core innovation weight
```

**Codec Parameters**:

| Codec | latent_dim | sample_rate | frame_rate |
|-------|------------|-------------|------------|
| EnCodec | 128 | 24000 | 75 |
| DAC | 1024 | 16000 | 50 |
| Mimi | 512 | 24000 | 12.5 |

### 3. Train

```bash
# Option A: Use provided example (need to implement dataset)
python examples/train_encodec.py \
  --data_csv data/train.csv \
  --audio_root data/audio \
  --num_epochs 50

# Option B: Use template script
python train.py --config configs/encodec_hokkien.yaml
```

**Important**: The training script requires you to implement a dataset class that returns:
```python
{
    'audio': torch.Tensor,  # [B, 1, T] - Waveform
    'f0': torch.Tensor,     # [B, 1, T'] - Normalized F0 (0-1)
    'uv': torch.Tensor      # [B, 1, T'] - Voicing flags (0/1)
}
```

See `examples/train_encodec.py` for a complete working example.

---

## Usage

### Training

Key steps (see `examples/train_encodec.py` for full code):

```python
from models.pira_core import ResidualPitchInjector, ConfidenceNetwork
from utils.crepe_loss import CREPEPerceptualLoss

# 1. Load frozen codec
from encodec import EncodecModel
codec = EncodecModel.encodec_model_24khz()
codec.set_target_bandwidth(1.5)
for param in codec.parameters():
    param.requires_grad = False

# 2. Create PIRA components
pitch_injector = ResidualPitchInjector(latent_dim=128)
confidence_net = ConfidenceNetwork(latent_dim=128)

# 3. CREPE perceptual loss (core innovation!)
crepe_loss = CREPEPerceptualLoss(sample_rate=24000, model='tiny')

# 4. Training loop
latent = codec.encoder(audio)
residual = pitch_injector(f0, uv)
confidence = confidence_net(latent, uv)
latent_corrected = latent + confidence * residual
reconstructed = codec.decoder(latent_corrected)

loss = crepe_loss(reconstructed, audio)  # + other losses
```

### Inference

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
pitch_injector.load_state_dict(checkpoint['pitch_injector_state_dict'])
confidence_net.load_state_dict(checkpoint['confidence_net_state_dict'])

# Inference
with torch.no_grad():
    latent = codec.encoder(audio)
    residual = pitch_injector(f0, uv)
    confidence = confidence_net(latent, uv)
    latent_corrected = latent + confidence * residual
    output = codec.decoder(latent_corrected)
```

---

## Project Structure

```
PIRA/
├── models/
│   ├── pira_core.py       # Core components (ResidualPitchInjector, ConfidenceNetwork)
│   └── f0_quantizer.py    # F0 quantization with STE
├── utils/
│   ├── crepe_loss.py      # CREPE perceptual loss (core innovation!)
│   ├── losses.py          # Multi-resolution STFT loss
│   └── mel_loss.py        # Mel-spectrogram loss
├── examples/
│   └── train_encodec.py   # Complete training example for EnCodec
├── configs/
│   └── encodec_hokkien.yaml  # Example configuration
├── requirements.txt       # Dependencies
└── README.md
```

---

## Configuration Guide

### Critical Parameters (Codec-Specific)

```yaml
codec:
  latent_dim: 128      # MUST match your codec!
  sample_rate: 24000   # MUST match your codec!
  frame_rate: 75       # MUST match your codec!
```

How to find these:
- `latent_dim`: Check codec's encoder output channels
- `frame_rate = sample_rate / hop_length`

### Universal Parameters

```yaml
model:
  hidden_dim: 256  # Works well for all codecs
  dilation_rates: [1, 3, 9, 27, 81, 243]  # Tone sandhi context
  use_confidence: true
  use_f0_quantizer: true
```

### Loss Weights

```yaml
loss:
  lambda_l1: 0.1       # Time-domain
  lambda_stft: 1.0     # Spectral fidelity
  lambda_mel: 1.0      # Perceptual quality
  lambda_crepe: 30     # CREPE loss (MOST IMPORTANT!)
```

**Tuning tips**:
- Increase `lambda_crepe` (→ 50) if pitch quality is poor
- Decrease `lambda_crepe` (→ 10) if audio quality degrades
