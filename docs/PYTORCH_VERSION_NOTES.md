# PyTorch Version Notes

## Current Version

The Nexlify codebase is designed to work with **PyTorch 2.1.0** as specified in `requirements.txt`.

## Dependency Conflicts

If you encounter conflicts with `torchaudio` or `torchvision`:

### Option 1: Remove Unused Packages (Recommended)

Nexlify does not use `torchaudio` or `torchvision`. You can safely uninstall them:

```bash
pip uninstall torchaudio torchvision -y
```

### Option 2: Upgrade PyTorch

If you need these packages for other projects, upgrade to PyTorch 2.5.1+:

```bash
pip uninstall torch torchaudio torchvision -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Then update `requirements.txt`:
```
torch==2.5.1
```

## Compatibility

Nexlify's GPU training features are compatible with:
- PyTorch 2.1.0+ (tested)
- PyTorch 2.5.1+ (should work, minimal API changes)

The core GPU optimization code (`nexlify_gpu_optimizations.py`) uses stable PyTorch APIs that haven't changed significantly between versions.

## CUDA Version

Make sure your PyTorch CUDA version matches your system's CUDA installation:

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch:
# CUDA 11.8:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## Verification

After resolving conflicts, verify GPU training works:

```bash
python scripts/verify_gpu_training.py --quick
```
