# Quick GPU Setup Guide

## Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (or AMD/Intel/Apple GPU)
- NVIDIA drivers installed (for NVIDIA GPUs)

## Installation Steps

### 1. Install Python Dependencies

```bash
# Navigate to project root
cd /path/to/Nexlify

# Install all requirements
pip install -r requirements.txt
```

### 2. Verify GPU Setup

```bash
# Quick verification
python scripts/verify_gpu_training.py --quick

# Full verification (includes training test)
python scripts/verify_gpu_training.py
```

### 3. Run Example

```bash
# Run the GPU training example
python examples/gpu_training_example.py
```

## Platform-Specific Instructions

### NVIDIA (CUDA)

Requirements already included in `requirements.txt`:
- torch==2.1.0 (CUDA support)
- pynvml==11.5.0 (GPU monitoring)

If CUDA is not available:

```bash
# For CUDA 11.8
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### AMD (ROCm)

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/rocm5.6
```

### Intel (Arc/Xe)

```bash
pip install intel-extension-for-pytorch
```

### Apple Silicon (M1/M2/M3)

Metal Performance Shaders (MPS) support is included in PyTorch. No additional setup needed.

## Troubleshooting

### Check GPU Detection

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### Check NVIDIA Drivers

```bash
nvidia-smi
```

## Running Training

All existing training scripts work automatically with GPU:

```bash
# 1000-round training
python scripts/train_ml_rl_1000_rounds.py --agent-type ultra

# Perfect ML training
python scripts/train_perfect_ml.py

# Adaptive RL training
python scripts/train_adaptive_rl_agent.py
```

## CPU Fallback

If no GPU is detected, training automatically falls back to CPU. No code changes needed.

## Getting Help

1. Run verification: `python scripts/verify_gpu_training.py`
2. Check logs: `logs/ml_rl_*.log`
3. See full guide: `docs/GPU_TRAINING_GUIDE.md`
