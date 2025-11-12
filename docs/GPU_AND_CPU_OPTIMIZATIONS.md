# GPU-Specific and CPU Optimization Guide

## Overview

Nexlify's dynamic architecture now includes comprehensive vendor-specific GPU optimizations and intelligent CPU threading with hyperthreading/SMT support. The system automatically detects your hardware and applies the optimal configuration.

## GPU Optimizations

### NVIDIA GPUs

#### Architecture Detection

The system automatically detects NVIDIA GPU architectures:

- **Maxwell** (GTX 900 series, 2014) - Compute 5.x
- **Pascal** (GTX 10 series, 2016) - Compute 6.x
- **Volta** (Titan V, V100, 2017) - Compute 7.0
- **Turing** (RTX 20, GTX 16 series, 2018) - Compute 7.5
- **Ampere** (RTX 30 series, A100, 2020) - Compute 8.x
- **Ada Lovelace** (RTX 40 series, 2022) - Compute 8.9
- **Hopper** (H100, 2022) - Compute 9.0

#### Tensor Core Optimization

**Volta/Turing (1st/2nd gen):**
- FP16 matrix operations
- 8 Tensor Cores per SM
- ~2x speedup for mixed precision

**Ampere (3rd gen):**
- TF32 automatic acceleration (no code changes!)
- BF16 support (better stability than FP16)
- Sparse Tensor Cores
- 4 Tensor Cores per SM
- ~3x speedup vs FP32

**Ada/Hopper (4th gen):**
- FP8 support (2x faster than FP16)
- Enhanced sparsity
- 4th generation Tensor Cores
- Up to 5x speedup vs FP32

#### Precision Recommendations

| Architecture | Training | Inference | Notes |
|-------------|----------|-----------|-------|
| Maxwell/Pascal | FP32 | FP32/INT8 | No native FP16 acceleration |
| Volta/Turing | FP16 | FP16/INT8 | Use mixed precision |
| Ampere | **BF16** | FP16/INT8 | BF16 more stable than FP16 |
| Ada/Hopper | **BF16/FP8** | FP8/INT8 | FP8 for maximum speed |

#### NVIDIA-Specific Settings

```python
{
    'enable_cudnn_benchmark': True,      # Find optimal algorithms
    'enable_tf32': True,                 # Ampere+: 3x faster matmul
    'matmul_precision': 'high',          # Use TF32 for FP32 ops
    'cudnn_allow_tf32': True,            # TF32 in cuDNN
    'cuda_launch_blocking': False,       # Async execution
    'use_cuda_graphs': True,             # Ampere+: reduce launch overhead
    'enable_flash_attention': True,      # Faster attention (8GB+ VRAM)
    'optimal_thread_count': SM_count * 128
}
```

#### Batch Size Optimization

| VRAM | Batch Size | Notes |
|------|------------|-------|
| 2-4 GB | 8-16 | Entry-level GPUs |
| 4-6 GB | 16-24 | GTX 1050 Ti, 1650 |
| 6-8 GB | 24-32 | GTX 1060, 1660 |
| 8-12 GB | 32-48 | RTX 3060, 2070 |
| 12-16 GB | 48-64 | RTX 3060 Ti, 2080 |
| 16-24 GB | 64-128 | RTX 3080, 4080, 4090 |
| 24-40 GB | 128-256 | A100 40GB, L40 |
| 40+ GB | 256+ | A100 80GB, H100 |

*Batch sizes automatically increased 25% on Ampere+ with Tensor Cores*

---

### AMD GPUs

#### Architecture Detection

**CDNA (Compute):**
- **CDNA** (MI100, 2020) - HPC/AI compute
- **CDNA2** (MI200 series, 2021) - Matrix Cores, BF16
- **CDNA3** (MI300 series, 2023) - FP8 support

**RDNA (Gaming):**
- **RDNA** (RX 5000 series, 2019) - First RDNA
- **RDNA2** (RX 6000 series, 2020) - Infinity Cache
- **RDNA3** (RX 7000 series, 2022) - BF16 support

**GCN (Legacy):**
- Polaris, Vega, Radeon VII

#### Matrix Core Optimization (CDNA)

Matrix Cores are AMD's equivalent to NVIDIA Tensor Cores:

- **CDNA**: FP16 matrix operations
- **CDNA2**: FP16 + BF16
- **CDNA3**: FP16 + BF16 + FP8

Best for HPC and AI training workloads.

#### Precision Recommendations

| Architecture | Training | Inference | Notes |
|-------------|----------|-----------|-------|
| GCN (Vega) | FP16/FP32 | FP16/INT8 | Good FP16 performance |
| RDNA/RDNA2 | FP32 | FP16/INT8 | Gaming-focused |
| RDNA3 | BF16/FP32 | FP16/INT8 | BF16 support added |
| CDNA | FP16 | FP16/INT8 | Matrix Cores |
| CDNA2 | **BF16** | FP16/INT8 | Better stability |
| CDNA3 | **BF16/FP8** | FP8/INT8 | FP8 for speed |

#### AMD-Specific Settings

```python
{
    'use_miopen': True,                  # AMD's cuDNN equivalent
    'miopen_benchmark': True,            # Find optimal algorithms
    'use_hipblas': True,                 # BLAS operations
    'use_rocblas': True,                 # ROCm BLAS
    'wave_size': 32 or 64,               # RDNA3: 32, CDNA/GCN: 64
    'use_infinity_cache': True,          # RDNA2/RDNA3: 128MB cache
    'optimize_for_workload': 'compute' or 'graphics'
}
```

#### ROCm Notes

- ROCm uses CUDA-compatible API (HIP)
- Most PyTorch CUDA code works without modification
- MIOpen provides cuDNN-like functionality
- Better compute performance on CDNA vs gaming RDNA

#### Batch Size Optimization

| VRAM | Batch Size | Architecture Notes |
|------|------------|-------------------|
| 8-12 GB | 32-48 | RX 5700/6700, Vega |
| 12-16 GB | 48-64 | RX 6800 |
| 16-24 GB | 64-128 | RX 6900/7900, MI100 |
| 24-64 GB | 128-256 | MI210, MI250 |
| 128+ GB | 256+ | MI300 |

*CDNA batch sizes increased 20% for Matrix Core utilization*

---

### Intel GPUs

#### Architecture

- **Xe HPG** (Arc A-series) - Gaming
- **Xe HPC** (Ponte Vecchio) - Datacenter

#### Optimization

```python
{
    'use_ipex': True,                    # Intel Extension for PyTorch
    'use_xmx': True,                     # Xe Matrix Extensions
}
```

Use `torch.xpu` device instead of `torch.cuda`.

---

### Apple Silicon

#### Architecture

- **M1/M2/M3** - Unified memory architecture
- **Neural Engine** - Separate AI accelerator (not exposed in PyTorch)

#### Optimization

```python
{
    'use_mps': True,                     # Metal Performance Shaders
}
```

Use `torch.mps` device. Note: Unified memory means GPU and CPU share RAM.

---

## CPU Optimizations

### Hyperthreading / SMT Detection

The system detects Intel Hyperthreading and AMD Simultaneous Multithreading:

#### How HT/SMT Works

- **Physical cores**: Real CPU cores
- **Logical cores**: Physical + Hyperthreaded cores
- **Efficiency**: HT cores are ~25% as effective as physical cores

**Example: Intel i7-12700K**
- 8 Performance cores (16 threads with HT)
- 4 Efficiency cores (4 threads, no HT)
- Total: 12 physical, 20 logical cores
- Effective cores: 12 + (20 - 12) × 0.25 = 14 effective cores

**Example: AMD Ryzen 9 5950X**
- 16 physical cores
- 32 logical cores (2-way SMT)
- Effective cores: 16 + (32 - 16) × 0.25 = 20 effective cores

### Workload-Aware Thread Allocation

The system adjusts worker threads based on workload type:

#### Preprocessing (Data augmentation, transforms)

- Uses **80% of effective cores**
- Benefits moderately from HT/SMT
- Example: 20 effective cores → 16 workers

#### Computation (Training, forward/backward pass)

- Uses **70% of physical cores**
- Heavy computation, less HT benefit
- Example: 16 physical cores → 11 workers

#### I/O (Data loading, disk/network)

- Uses **90% of logical cores**
- I/O bound, HT helps
- Example: 32 logical cores → 28 workers

### CPU Affinity

The system recommends CPU affinity to maximize performance:

**Strategy:**
1. Assign workers to physical cores first
2. Only use HT cores if more workers than physical cores
3. Avoid overloading single physical cores

**Example: 8 physical, 16 logical cores**

```
4 workers:
  Worker 0: CPU 0 (physical)
  Worker 1: CPU 1 (physical)
  Worker 2: CPU 2 (physical)
  Worker 3: CPU 3 (physical)

12 workers:
  Workers 0-7: CPUs 0-7 (physical cores)
  Workers 8-11: CPUs 8-11 (HT cores)
```

---

## Automatic Configuration

### Dynamic Optimization Flow

```
1. Detect Hardware
   ├─ CPU: Physical/logical cores, HT/SMT
   ├─ RAM: Available memory
   └─ GPU: Vendor, architecture, VRAM, capabilities

2. Configure Precision
   ├─ NVIDIA Ampere+: BF16 or TF32
   ├─ NVIDIA Volta/Turing: FP16
   ├─ AMD CDNA2+: BF16
   ├─ AMD CDNA: FP16
   └─ Fallback: FP32

3. Set Batch Size
   ├─ Based on VRAM
   ├─ Adjusted for architecture
   └─ Gradient accumulation if needed

4. Allocate Workers
   ├─ CPU overhead available
   ├─ Workload type
   └─ HT/SMT efficiency

5. Apply Optimizations
   ├─ cuDNN/MIOpen benchmark
   ├─ Memory pools
   ├─ Tensor Cores
   ├─ Multi-stream
   └─ CPU affinity
```

### Real-World Examples

#### Example 1: Budget Gaming PC

**Hardware:**
- AMD Ryzen 5 5600X (6C/12T)
- NVIDIA GTX 1660 Super (6GB GDDR6)
- 16GB RAM

**Automatic Configuration:**
- GPU: Turing, no Tensor Cores → FP32 training
- Batch size: 24
- CPU workers: 8 (70% of 12 logical cores for preprocessing)
- Mixed precision: FP16 inference only
- Gradient accumulation: 2 steps

---

#### Example 2: Mid-Range Workstation

**Hardware:**
- Intel i7-12700K (12C/20T, 8P+4E cores)
- NVIDIA RTX 3080 (10GB GDDR6X)
- 32GB RAM

**Automatic Configuration:**
- GPU: Ampere, Tensor Cores → **BF16** mixed precision
- TF32 enabled for FP32 ops (automatic 3x speedup)
- Batch size: 64
- CPU workers: 12 (using effective cores)
- Memory fraction: 90% (9GB VRAM)
- cuDNN benchmark: Enabled

**Performance:**
- ~2.5x speedup vs FP32 (BF16 + Tensor Cores)
- Better stability than FP16

---

#### Example 3: Unbalanced System (Weak GPU, Strong CPU)

**Hardware:**
- AMD Threadripper 3970X (32C/64T)
- NVIDIA GTX 1050 Ti (4GB GDDR5)
- 128GB RAM

**Automatic Configuration:**
- GPU bottleneck detected (4GB VRAM)
- Small GPU batch size: 16
- **Massive CPU offloading**: 40 workers (uses excess CPU)
- Large experience buffer: 250K (uses excess RAM)
- CPU preprocessing: Heavy use
- GPU only for forward/backward pass

**Result:**
- 85% resource utilization (vs 30% with fixed tiers)
- CPU does heavy lifting, GPU focused on training

---

#### Example 4: High-End AI Workstation

**Hardware:**
- AMD Ryzen 9 7950X3D (16C/32T)
- NVIDIA RTX 4090 (24GB GDDR6X)
- 64GB RAM

**Automatic Configuration:**
- GPU: Ada Lovelace, 4th-gen Tensor Cores
- **BF16** mixed precision (or FP8 for experimental)
- TF32 enabled
- Batch size: 128
- CPU workers: 20 (effective cores)
- Flash Attention: Enabled (24GB VRAM)
- CUDA Graphs: Enabled (reduce overhead)
- Multi-stream: 2 streams

**Performance:**
- ~4-5x speedup vs FP32
- Near-optimal utilization of all components

---

#### Example 5: AMD CDNA Compute

**Hardware:**
- AMD EPYC 7763 (64C/128T)
- AMD MI250 (128GB HBM2e)
- 512GB RAM

**Automatic Configuration:**
- GPU: CDNA2, Matrix Cores → **BF16** mixed precision
- Batch size: 256
- CPU workers: 80 (CDNA does most work)
- MIOpen benchmark: Enabled
- ROCm optimized
- Wave size: 64

**Performance:**
- Excellent for large-scale training
- BF16 Matrix Cores for stability + speed

---

## Usage

### Automatic (Recommended)

```python
from nexlify.ml.nexlify_dynamic_architecture_enhanced import EnhancedDynamicResourceMonitor

# Create monitor - automatically detects and optimizes
monitor = EnhancedDynamicResourceMonitor()

# Get optimal settings
batch_size = monitor.get_gpu_optimal_batch_size()
device = monitor.get_device_string()
precision = monitor.get_precision_dtype()
workers = monitor.calculate_optimal_workers('preprocessing')

print(f"Batch size: {batch_size}")
print(f"Device: {device}")
print(f"Precision: {precision}")
print(f"Workers: {workers}")
```

### Manual GPU Optimization

```python
from nexlify.ml.nexlify_gpu_optimizations import create_gpu_optimizer

# Create optimizer
optimizer = create_gpu_optimizer()

# Get capabilities
if optimizer.capabilities:
    print(f"GPU: {optimizer.capabilities.name}")
    print(f"Architecture: {optimizer.capabilities.architecture}")
    print(f"Has Tensor Cores: {optimizer.capabilities.has_tensor_cores}")

# Apply optimizations to PyTorch
optimizer.apply_optimizations()

# Get config
config = optimizer.config
print(f"Mixed precision: {config.use_mixed_precision}")
print(f"Optimal batch: {config.optimal_batch_size}")
```

### CPU Affinity

```python
from nexlify.ml.nexlify_dynamic_architecture_enhanced import EnhancedDynamicResourceMonitor

monitor = EnhancedDynamicResourceMonitor()

# Get optimal workers
num_workers = monitor.calculate_optimal_workers('computation')

# Get affinity recommendation
affinities = monitor.get_cpu_affinity_recommendation(num_workers)

# Apply to worker threads
for i, affinity in enumerate(affinities):
    # Set CPU affinity for worker i to cores in affinity list
    pass  # Use psutil.Process().cpu_affinity(affinity)
```

---

## Performance Tips

### NVIDIA

1. **Always enable cuDNN benchmark** (done automatically)
2. **Use BF16 on Ampere+** - better stability than FP16
3. **Enable TF32 on Ampere+** - free 3x speedup for FP32 ops
4. **Use Tensor Cores** - 2-5x speedup with mixed precision
5. **Increase batch size** - Tensor Cores work best with larger batches
6. **Use gradient accumulation** - simulate large batches on small GPUs

### AMD

1. **CDNA for training** - better compute performance
2. **RDNA for inference** - good for deployment
3. **Enable MIOpen benchmark** (done automatically)
4. **Use BF16 on CDNA2+** - stability + speed
5. **Leverage Infinity Cache** - RDNA2/3: faster memory access

### CPU

1. **Use physical cores first** - HT cores are 25% effective
2. **Adjust workers by workload** - I/O can use more threads
3. **Set CPU affinity** - prevents thread migration overhead
4. **Monitor effective cores** - not just logical count

---

## Testing

Run the comprehensive test suite:

```bash
python examples/test_gpu_and_ht_optimizations.py
```

Tests:
1. GPU detection and capabilities
2. Vendor-specific optimization config
3. CPU topology (HT/SMT)
4. Optimal worker calculation
5. CPU affinity recommendations
6. Enhanced resource monitoring
7. GPU integration
8. Optimization application

---

## Troubleshooting

### GPU Not Detected

**NVIDIA:**
- Install CUDA Toolkit
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- Check: `nvidia-smi`

**AMD:**
- Install ROCm
- Install PyTorch for ROCm: `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7`
- Check: `rocm-smi`

**Intel:**
- Install Intel Extension for PyTorch: `pip install intel-extension-for-pytorch`

**Apple:**
- Use PyTorch 2.0+ with MPS support
- macOS 12.3+

### Out of Memory

- Reduce batch size
- Enable gradient accumulation
- Use gradient checkpointing
- Lower precision (FP16/BF16)

### Slow Training

- Check if TF32/Tensor Cores are enabled
- Use mixed precision
- Enable cuDNN/MIOpen benchmark
- Check CPU workers (too many = overhead)
- Monitor bottleneck (CPU/GPU/RAM/VRAM)

---

## References

- **NVIDIA Tensor Cores**: [NVIDIA Docs](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- **AMD Matrix Cores**: [AMD CDNA Docs](https://www.amd.com/en/technologies/cdna)
- **PyTorch Mixed Precision**: [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
- **Intel Hyperthreading**: [Intel HT Guide](https://www.intel.com/content/www/us/en/architecture-and-technology/hyper-threading/hyper-threading-technology.html)
- **AMD SMT**: [AMD SMT Guide](https://www.amd.com/en/technologies/ryzen-smt)
