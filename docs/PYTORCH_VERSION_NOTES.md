# PyTorch Version Notes

## Current Version

The Nexlify codebase is designed to work with **PyTorch 2.6.0** as specified in `requirements.txt`.

## Dependency Conflicts

If you encounter conflicts with `torchaudio` or `torchvision`:

### Option 1: Remove Unused Packages (Recommended)

If your workflow does not require audio/image APIs, you can uninstall optional packages:

```bash
pip uninstall torchaudio torchvision -y
```

### Option 2: Reinstall a Matched PyTorch Stack

Use matching versions for torch/torchvision/torchaudio:

```bash
pip uninstall torch torchaudio torchvision -y
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

Then ensure `requirements.txt` remains aligned:

```text
torch==2.6.0
torchvision==0.21.0
```

## Compatibility

Nexlify GPU features target:
- PyTorch 2.6.0+ (project baseline)
- CUDA 12.1 wheels by default for NVIDIA setups

The GPU optimization code uses stable PyTorch APIs and should remain compatible across minor 2.6.x updates.

## CUDA Version

Make sure the installed PyTorch CUDA wheel matches your driver/runtime capabilities:

```bash
# Check driver/CUDA runtime
nvidia-smi

# Install PyTorch with CUDA 12.1 wheels
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
```

## Verification

After resolving conflicts, verify GPU training works:

```bash
python scripts/verify_gpu_training.py --quick
```
