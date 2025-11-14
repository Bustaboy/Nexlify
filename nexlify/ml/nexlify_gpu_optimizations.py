#!/usr/bin/env python3
"""
Nexlify GPU-Specific Optimizations

Vendor-specific optimizations for:
- NVIDIA GPUs (CUDA, Tensor Cores, cuDNN, etc.)
- AMD GPUs (ROCm, MIOpen, RDNA/CDNA, etc.)
- Architecture-specific tuning
- Dynamic optimization selection
"""

import logging
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUVendor(Enum):
    """GPU vendor types"""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class NVIDIAArchitecture(Enum):
    """NVIDIA GPU architectures"""

    MAXWELL = "maxwell"  # GTX 900 series (2014)
    PASCAL = "pascal"  # GTX 10 series (2016)
    VOLTA = "volta"  # Titan V, V100 (2017)
    TURING = "turing"  # RTX 20 series, GTX 16 series (2018)
    AMPERE = "ampere"  # RTX 30 series, A100 (2020)
    ADA = "ada"  # RTX 40 series (2022)
    HOPPER = "hopper"  # H100 (2022)
    UNKNOWN = "unknown"


class AMDArchitecture(Enum):
    """AMD GPU architectures"""

    GCN = "gcn"  # Polaris, Vega (2016-2017)
    RDNA = "rdna"  # RX 5000 series (2019)
    RDNA2 = "rdna2"  # RX 6000 series (2020)
    RDNA3 = "rdna3"  # RX 7000 series (2022)
    CDNA = "cdna"  # MI100 (2020)
    CDNA2 = "cdna2"  # MI200 series (2021)
    CDNA3 = "cdna3"  # MI300 series (2023)
    UNKNOWN = "unknown"


@dataclass
class GPUCapabilities:
    """Detailed GPU capabilities"""

    vendor: GPUVendor
    name: str
    compute_capability: Optional[str]  # e.g., "8.6" for NVIDIA
    architecture: str
    vram_gb: float
    cuda_cores: Optional[int]
    tensor_cores: Optional[int]
    has_tensor_cores: bool
    has_fp16: bool
    has_bf16: bool
    has_tf32: bool
    has_fp8: bool
    has_int8: bool
    memory_bandwidth_gbps: Optional[float]
    memory_bus_width: Optional[int]
    clock_mhz: Optional[int]
    sm_count: Optional[int]  # Streaming Multiprocessors (NVIDIA) or Compute Units (AMD)
    max_threads_per_sm: Optional[int]
    optimal_batch_size: int
    supports_concurrent_kernels: bool
    supports_multi_stream: bool


@dataclass
class GPUOptimizationConfig:
    """GPU-specific optimization configuration"""

    vendor: GPUVendor

    # Precision settings
    use_mixed_precision: bool
    use_tf32: bool
    use_fp16: bool
    use_bf16: bool
    use_fp8: bool

    # Memory settings
    use_memory_pool: bool
    memory_fraction: float
    allow_growth: bool

    # Execution settings
    use_cudnn_benchmark: bool
    use_tensor_cores: bool
    num_streams: int
    persistent_workers: bool

    # Batch settings
    optimal_batch_size: int
    gradient_accumulation_steps: int

    # Vendor-specific settings
    vendor_specific: Dict[str, Any]


class GPUOptimizer:
    """
    GPU-specific optimizer that detects and configures
    vendor-specific optimizations
    """

    def __init__(self):
        self.capabilities = None
        self.config = None

        # Detect GPU
        self.capabilities = self._detect_gpu_capabilities()

        if self.capabilities:
            self.config = self._create_optimization_config(self.capabilities)
            logger.info(f"🎮 GPU Optimizer initialized: {self.capabilities.name}")
            logger.info(f"   Vendor: {self.capabilities.vendor.value}")
            logger.info(f"   Architecture: {self.capabilities.architecture}")
            logger.info(f"   VRAM: {self.capabilities.vram_gb:.1f} GB")
            if self.capabilities.has_tensor_cores:
                logger.info(
                    f"   ✓ Tensor Cores available ({self.capabilities.tensor_cores})"
                )
            logger.info(
                f"   Precision: FP16={self.capabilities.has_fp16}, "
                f"BF16={self.capabilities.has_bf16}, "
                f"TF32={self.capabilities.has_tf32}"
            )
        else:
            logger.warning("⚠️  No GPU detected or GPU not supported")

    def _detect_gpu_capabilities(self) -> Optional[GPUCapabilities]:
        """Detect GPU vendor and capabilities"""

        # Try NVIDIA first
        nvidia_caps = self._detect_nvidia_gpu()
        if nvidia_caps:
            return nvidia_caps

        # Try AMD
        amd_caps = self._detect_amd_gpu()
        if amd_caps:
            return amd_caps

        # Try Intel
        intel_caps = self._detect_intel_gpu()
        if intel_caps:
            return intel_caps

        # Try Apple Metal
        if platform.system() == "Darwin":
            apple_caps = self._detect_apple_gpu()
            if apple_caps:
                return apple_caps

        return None

    def _detect_nvidia_gpu(self) -> Optional[GPUCapabilities]:
        """Detect NVIDIA GPU and capabilities"""
        try:
            import torch

            if not torch.cuda.is_available():
                return None

            device_id = 0
            props = torch.cuda.get_device_properties(device_id)

            name = props.name
            vram_gb = props.total_memory / (1024**3)

            # Compute capability (e.g., 8.6 for RTX 3090)
            compute_capability = f"{props.major}.{props.minor}"

            # Detect architecture from compute capability
            architecture = self._nvidia_cc_to_arch(props.major, props.minor)

            # Detect Tensor Cores
            has_tensor_cores = props.major >= 7  # Volta and newer
            tensor_cores = None

            if has_tensor_cores:
                # Estimate tensor core count (varies by model)
                sm_count = props.multi_processor_count

                if props.major == 7:  # Volta, Turing
                    tensor_cores = sm_count * 8
                elif props.major == 8:  # Ampere
                    if props.minor == 0:  # A100
                        tensor_cores = sm_count * 4  # 3rd gen
                    else:
                        tensor_cores = sm_count * 4  # 3rd gen
                elif props.major >= 9:  # Ada, Hopper
                    tensor_cores = sm_count * 4  # 4th gen

            # Precision support
            has_fp16 = True  # All CUDA GPUs
            has_bf16 = props.major >= 8  # Ampere and newer
            has_tf32 = props.major >= 8  # Ampere and newer
            has_fp8 = props.major >= 9  # Ada and newer
            has_int8 = True

            # Memory bandwidth (estimate from common models)
            memory_bandwidth = self._estimate_nvidia_bandwidth(name, vram_gb)

            # Optimal batch size (based on VRAM and architecture)
            optimal_batch_size = self._calculate_nvidia_batch_size(
                vram_gb, architecture, has_tensor_cores
            )

            return GPUCapabilities(
                vendor=GPUVendor.NVIDIA,
                name=name,
                compute_capability=compute_capability,
                architecture=architecture.value,
                vram_gb=vram_gb,
                cuda_cores=props.multi_processor_count * 64,  # Rough estimate
                tensor_cores=tensor_cores,
                has_tensor_cores=has_tensor_cores,
                has_fp16=has_fp16,
                has_bf16=has_bf16,
                has_tf32=has_tf32,
                has_fp8=has_fp8,
                has_int8=has_int8,
                memory_bandwidth_gbps=memory_bandwidth,
                memory_bus_width=None,  # Not exposed by PyTorch
                clock_mhz=None,
                sm_count=props.multi_processor_count,
                max_threads_per_sm=props.max_threads_per_multi_processor,
                optimal_batch_size=optimal_batch_size,
                supports_concurrent_kernels=True,
                supports_multi_stream=True,
            )

        except Exception as e:
            logger.debug(f"NVIDIA GPU detection failed: {e}")
            return None

    def _detect_amd_gpu(self) -> Optional[GPUCapabilities]:
        """Detect AMD GPU and capabilities"""
        try:
            import torch

            # Check for ROCm
            if not hasattr(torch.version, "hip") or torch.version.hip is None:
                return None

            if not torch.cuda.is_available():  # ROCm uses CUDA API
                return None

            device_id = 0
            props = torch.cuda.get_device_properties(device_id)

            name = props.name
            vram_gb = props.total_memory / (1024**3)

            # Detect architecture from name
            architecture = self._detect_amd_architecture(name)

            # AMD precision support
            has_fp16 = True
            has_bf16 = architecture in [
                AMDArchitecture.RDNA3,
                AMDArchitecture.CDNA2,
                AMDArchitecture.CDNA3,
            ]
            has_tf32 = False  # AMD doesn't have TF32
            has_fp8 = architecture == AMDArchitecture.CDNA3
            has_int8 = True

            # Matrix cores (AMD's version of Tensor Cores)
            has_matrix_cores = architecture in [
                AMDArchitecture.CDNA,
                AMDArchitecture.CDNA2,
                AMDArchitecture.CDNA3,
            ]

            # Compute Units (AMD's SM equivalent)
            cu_count = props.multi_processor_count

            # Memory bandwidth
            memory_bandwidth = self._estimate_amd_bandwidth(name, vram_gb, architecture)

            # Optimal batch size
            optimal_batch_size = self._calculate_amd_batch_size(
                vram_gb, architecture, has_matrix_cores
            )

            return GPUCapabilities(
                vendor=GPUVendor.AMD,
                name=name,
                compute_capability=None,  # AMD doesn't use compute capability
                architecture=architecture.value,
                vram_gb=vram_gb,
                cuda_cores=None,  # AMD has Stream Processors instead
                tensor_cores=None,  # AMD has Matrix Cores
                has_tensor_cores=has_matrix_cores,
                has_fp16=has_fp16,
                has_bf16=has_bf16,
                has_tf32=has_tf32,
                has_fp8=has_fp8,
                has_int8=has_int8,
                memory_bandwidth_gbps=memory_bandwidth,
                memory_bus_width=None,
                clock_mhz=None,
                sm_count=cu_count,
                max_threads_per_sm=props.max_threads_per_multi_processor,
                optimal_batch_size=optimal_batch_size,
                supports_concurrent_kernels=True,
                supports_multi_stream=True,
            )

        except Exception as e:
            logger.debug(f"AMD GPU detection failed: {e}")
            return None

    def _detect_intel_gpu(self) -> Optional[GPUCapabilities]:
        """Detect Intel GPU (Arc, Xe)"""
        try:
            import torch

            # Intel extension for PyTorch
            try:
                import intel_extension_for_pytorch as ipex

                if torch.xpu.is_available():
                    device_id = 0
                    name = torch.xpu.get_device_name(device_id)
                    vram_gb = torch.xpu.get_device_properties(
                        device_id
                    ).total_memory / (1024**3)

                    return GPUCapabilities(
                        vendor=GPUVendor.INTEL,
                        name=name,
                        compute_capability=None,
                        architecture="xe",
                        vram_gb=vram_gb,
                        cuda_cores=None,
                        tensor_cores=None,
                        has_tensor_cores=True,  # Xe Matrix Extensions
                        has_fp16=True,
                        has_bf16=True,
                        has_tf32=False,
                        has_fp8=False,
                        has_int8=True,
                        memory_bandwidth_gbps=None,
                        memory_bus_width=None,
                        clock_mhz=None,
                        sm_count=None,
                        max_threads_per_sm=None,
                        optimal_batch_size=16,
                        supports_concurrent_kernels=True,
                        supports_multi_stream=True,
                    )
            except ImportError:
                pass

        except Exception as e:
            logger.debug(f"Intel GPU detection failed: {e}")

        return None

    def _detect_apple_gpu(self) -> Optional[GPUCapabilities]:
        """Detect Apple Silicon GPU (M1, M2, M3)"""
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon detected
                chip_name = platform.processor()

                # Estimate based on chip
                if "M1" in chip_name or "M2" in chip_name or "M3" in chip_name:
                    return GPUCapabilities(
                        vendor=GPUVendor.APPLE,
                        name=chip_name,
                        compute_capability=None,
                        architecture="apple_silicon",
                        vram_gb=8.0,  # Unified memory - estimate
                        cuda_cores=None,
                        tensor_cores=None,
                        has_tensor_cores=True,  # Neural Engine
                        has_fp16=True,
                        has_bf16=False,
                        has_tf32=False,
                        has_fp8=False,
                        has_int8=True,
                        memory_bandwidth_gbps=None,
                        memory_bus_width=None,
                        clock_mhz=None,
                        sm_count=None,
                        max_threads_per_sm=None,
                        optimal_batch_size=16,
                        supports_concurrent_kernels=False,
                        supports_multi_stream=False,
                    )

        except Exception as e:
            logger.debug(f"Apple GPU detection failed: {e}")

        return None

    def _nvidia_cc_to_arch(self, major: int, minor: int) -> NVIDIAArchitecture:
        """Convert NVIDIA compute capability to architecture"""
        if major == 5:
            return NVIDIAArchitecture.MAXWELL
        elif major == 6:
            return NVIDIAArchitecture.PASCAL
        elif major == 7:
            if minor == 0:
                return NVIDIAArchitecture.VOLTA
            else:
                return NVIDIAArchitecture.TURING
        elif major == 8:
            return NVIDIAArchitecture.AMPERE
        elif major == 9:
            if minor == 0:
                return NVIDIAArchitecture.HOPPER
            else:
                return NVIDIAArchitecture.ADA
        else:
            return NVIDIAArchitecture.UNKNOWN

    def _detect_amd_architecture(self, name: str) -> AMDArchitecture:
        """Detect AMD architecture from GPU name"""
        name_lower = name.lower()

        # CDNA (compute)
        if "mi300" in name_lower:
            return AMDArchitecture.CDNA3
        elif "mi200" in name_lower or "mi250" in name_lower:
            return AMDArchitecture.CDNA2
        elif "mi100" in name_lower:
            return AMDArchitecture.CDNA

        # RDNA (gaming)
        elif "rx 7" in name_lower or "7900" in name_lower or "7800" in name_lower:
            return AMDArchitecture.RDNA3
        elif "rx 6" in name_lower or "6900" in name_lower or "6800" in name_lower:
            return AMDArchitecture.RDNA2
        elif "rx 5" in name_lower or "5700" in name_lower:
            return AMDArchitecture.RDNA

        # GCN (older)
        elif "vega" in name_lower or "radeon vii" in name_lower:
            return AMDArchitecture.GCN
        elif (
            "polaris" in name_lower or "rx 580" in name_lower or "rx 480" in name_lower
        ):
            return AMDArchitecture.GCN

        return AMDArchitecture.UNKNOWN

    def _estimate_nvidia_bandwidth(self, name: str, vram_gb: float) -> Optional[float]:
        """Estimate NVIDIA memory bandwidth from model name"""
        name_lower = name.lower()

        # High-end datacenter
        if "h100" in name_lower:
            return 3350.0  # HBM3
        elif "a100" in name_lower:
            return 2000.0 if vram_gb >= 80 else 1555.0  # HBM2e
        elif "v100" in name_lower:
            return 900.0

        # RTX 40 series (Ada)
        elif "4090" in name_lower:
            return 1008.0
        elif "4080" in name_lower:
            return 716.8
        elif "4070" in name_lower:
            return 504.2

        # RTX 30 series (Ampere)
        elif "3090" in name_lower:
            return 936.0
        elif "3080" in name_lower:
            return 760.0
        elif "3070" in name_lower:
            return 448.0
        elif "3060" in name_lower:
            return 360.0

        # RTX 20 series (Turing)
        elif "2080" in name_lower:
            return 616.0
        elif "2070" in name_lower:
            return 448.0
        elif "2060" in name_lower:
            return 336.0

        # GTX 10 series (Pascal)
        elif "1080" in name_lower:
            return 320.0
        elif "1070" in name_lower:
            return 256.0
        elif "1060" in name_lower:
            return 192.0
        elif "1050" in name_lower:
            return 112.0

        return None

    def _estimate_amd_bandwidth(
        self, name: str, vram_gb: float, arch: AMDArchitecture
    ) -> Optional[float]:
        """Estimate AMD memory bandwidth"""
        name_lower = name.lower()

        # CDNA datacenter
        if "mi300" in name_lower:
            return 5300.0  # HBM3
        elif "mi250" in name_lower:
            return 3277.0  # HBM2e
        elif "mi200" in name_lower:
            return 1638.0
        elif "mi100" in name_lower:
            return 1228.0

        # RDNA3
        elif "7900 xtx" in name_lower:
            return 960.0
        elif "7900 xt" in name_lower:
            return 800.0
        elif "7800" in name_lower:
            return 624.0

        # RDNA2
        elif "6900" in name_lower:
            return 512.0
        elif "6800" in name_lower:
            return 512.0
        elif "6700" in name_lower:
            return 384.0

        # RDNA
        elif "5700" in name_lower:
            return 448.0

        return None

    def _calculate_nvidia_batch_size(
        self, vram_gb: float, arch: NVIDIAArchitecture, has_tensor_cores: bool
    ) -> int:
        """Calculate optimal batch size for NVIDIA GPU"""

        # Base batch size on VRAM
        if vram_gb >= 40:
            base_batch = 256
        elif vram_gb >= 24:
            base_batch = 128
        elif vram_gb >= 16:
            base_batch = 64
        elif vram_gb >= 12:
            base_batch = 48
        elif vram_gb >= 8:
            base_batch = 32
        elif vram_gb >= 6:
            base_batch = 24
        elif vram_gb >= 4:
            base_batch = 16
        else:
            base_batch = 8

        # Adjust for architecture
        if arch in [
            NVIDIAArchitecture.AMPERE,
            NVIDIAArchitecture.ADA,
            NVIDIAArchitecture.HOPPER,
        ]:
            # Newer architectures benefit from larger batches with Tensor Cores
            if has_tensor_cores:
                base_batch = int(base_batch * 1.25)

        return base_batch

    def _calculate_amd_batch_size(
        self, vram_gb: float, arch: AMDArchitecture, has_matrix_cores: bool
    ) -> int:
        """Calculate optimal batch size for AMD GPU"""

        # Base batch size on VRAM
        if vram_gb >= 40:
            base_batch = 256
        elif vram_gb >= 24:
            base_batch = 128
        elif vram_gb >= 16:
            base_batch = 64
        elif vram_gb >= 12:
            base_batch = 48
        elif vram_gb >= 8:
            base_batch = 32
        elif vram_gb >= 6:
            base_batch = 24
        else:
            base_batch = 16

        # CDNA benefits from larger batches
        if arch in [AMDArchitecture.CDNA2, AMDArchitecture.CDNA3] and has_matrix_cores:
            base_batch = int(base_batch * 1.2)

        return base_batch

    def _create_optimization_config(
        self, caps: GPUCapabilities
    ) -> GPUOptimizationConfig:
        """Create vendor-specific optimization configuration"""

        if caps.vendor == GPUVendor.NVIDIA:
            return self._create_nvidia_config(caps)
        elif caps.vendor == GPUVendor.AMD:
            return self._create_amd_config(caps)
        elif caps.vendor == GPUVendor.INTEL:
            return self._create_intel_config(caps)
        elif caps.vendor == GPUVendor.APPLE:
            return self._create_apple_config(caps)
        else:
            return self._create_default_config(caps)

    def _create_nvidia_config(self, caps: GPUCapabilities) -> GPUOptimizationConfig:
        """Create NVIDIA-specific optimization config"""

        arch = (
            NVIDIAArchitecture(caps.architecture)
            if caps.architecture != "unknown"
            else NVIDIAArchitecture.UNKNOWN
        )

        # Mixed precision settings
        use_mixed_precision = caps.has_fp16 or caps.has_bf16
        use_tf32 = caps.has_tf32  # Ampere and newer
        use_fp16 = caps.has_fp16
        use_bf16 = caps.has_bf16  # Ampere and newer (more stable)
        use_fp8 = caps.has_fp8  # Ada and newer

        # Prefer BF16 over FP16 on Ampere+ for better stability
        if use_bf16:
            use_fp16 = False  # Use BF16 instead

        # Tensor Cores
        use_tensor_cores = caps.has_tensor_cores

        # cuDNN benchmark (finds optimal algorithms)
        use_cudnn_benchmark = True

        # Memory settings
        use_memory_pool = True
        memory_fraction = 0.9  # Use 90% of VRAM
        allow_growth = True

        # Multi-stream for concurrent execution
        num_streams = 2 if caps.supports_multi_stream else 1

        # Persistent workers
        persistent_workers = caps.vram_gb >= 8  # Only on larger GPUs

        # Gradient accumulation for small GPUs
        if caps.vram_gb < 6:
            gradient_accumulation_steps = 4
        elif caps.vram_gb < 12:
            gradient_accumulation_steps = 2
        else:
            gradient_accumulation_steps = 1

        # NVIDIA-specific settings
        vendor_specific = {
            "compute_capability": caps.compute_capability,
            "architecture": arch.value,
            "enable_cudnn_benchmark": True,
            "enable_tf32": use_tf32,
            "matmul_precision": "high" if use_tf32 else "highest",
            "cudnn_deterministic": False,  # False for performance
            "cudnn_allow_tf32": use_tf32,
            "cuda_launch_blocking": False,  # False for async
            "use_cuda_graphs": arch
            in [
                NVIDIAArchitecture.AMPERE,
                NVIDIAArchitecture.ADA,
                NVIDIAArchitecture.HOPPER,
            ],
            "enable_flash_attention": caps.has_tensor_cores and caps.vram_gb >= 8,
            "optimal_thread_count": caps.sm_count * 128 if caps.sm_count else 1024,
        }

        return GPUOptimizationConfig(
            vendor=GPUVendor.NVIDIA,
            use_mixed_precision=use_mixed_precision,
            use_tf32=use_tf32,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            use_fp8=use_fp8,
            use_memory_pool=use_memory_pool,
            memory_fraction=memory_fraction,
            allow_growth=allow_growth,
            use_cudnn_benchmark=use_cudnn_benchmark,
            use_tensor_cores=use_tensor_cores,
            num_streams=num_streams,
            persistent_workers=persistent_workers,
            optimal_batch_size=caps.optimal_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            vendor_specific=vendor_specific,
        )

    def _create_amd_config(self, caps: GPUCapabilities) -> GPUOptimizationConfig:
        """Create AMD-specific optimization config"""

        arch = (
            AMDArchitecture(caps.architecture)
            if caps.architecture != "unknown"
            else AMDArchitecture.UNKNOWN
        )

        # Mixed precision settings
        use_mixed_precision = caps.has_fp16 or caps.has_bf16
        use_fp16 = caps.has_fp16
        use_bf16 = caps.has_bf16  # CDNA2+, RDNA3+
        use_fp8 = caps.has_fp8  # CDNA3

        # AMD doesn't have TF32
        use_tf32 = False

        # Matrix cores (AMD's Tensor Core equivalent)
        use_tensor_cores = caps.has_tensor_cores

        # MIOpen (AMD's cuDNN)
        use_cudnn_benchmark = True  # ROCm uses same API

        # Memory settings
        use_memory_pool = True
        memory_fraction = 0.9
        allow_growth = True

        # Multi-stream
        num_streams = 2 if caps.supports_multi_stream else 1

        # Persistent workers
        persistent_workers = caps.vram_gb >= 8

        # Gradient accumulation
        if caps.vram_gb < 6:
            gradient_accumulation_steps = 4
        elif caps.vram_gb < 12:
            gradient_accumulation_steps = 2
        else:
            gradient_accumulation_steps = 1

        # AMD-specific settings
        vendor_specific = {
            "architecture": arch.value,
            "use_miopen": True,
            "miopen_benchmark": True,
            "rocm_version": None,  # Could detect from torch.version.hip
            "use_hipblas": True,
            "use_rocblas": True,
            "wave_size": (
                64
                if arch
                in [AMDArchitecture.GCN, AMDArchitecture.CDNA, AMDArchitecture.CDNA2]
                else 32
            ),
            "use_infinity_cache": arch
            in [AMDArchitecture.RDNA2, AMDArchitecture.RDNA3],
            "optimize_for_workload": "compute" if "CDNA" in arch.value else "graphics",
        }

        return GPUOptimizationConfig(
            vendor=GPUVendor.AMD,
            use_mixed_precision=use_mixed_precision,
            use_tf32=use_tf32,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            use_fp8=use_fp8,
            use_memory_pool=use_memory_pool,
            memory_fraction=memory_fraction,
            allow_growth=allow_growth,
            use_cudnn_benchmark=use_cudnn_benchmark,
            use_tensor_cores=use_tensor_cores,
            num_streams=num_streams,
            persistent_workers=persistent_workers,
            optimal_batch_size=caps.optimal_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            vendor_specific=vendor_specific,
        )

    def _create_intel_config(self, caps: GPUCapabilities) -> GPUOptimizationConfig:
        """Create Intel-specific optimization config"""
        return GPUOptimizationConfig(
            vendor=GPUVendor.INTEL,
            use_mixed_precision=True,
            use_tf32=False,
            use_fp16=True,
            use_bf16=True,
            use_fp8=False,
            use_memory_pool=True,
            memory_fraction=0.9,
            allow_growth=True,
            use_cudnn_benchmark=False,
            use_tensor_cores=True,
            num_streams=1,
            persistent_workers=False,
            optimal_batch_size=caps.optimal_batch_size,
            gradient_accumulation_steps=2,
            vendor_specific={
                "use_ipex": True,
                "use_xmx": True,  # Xe Matrix Extensions
            },
        )

    def _create_apple_config(self, caps: GPUCapabilities) -> GPUOptimizationConfig:
        """Create Apple Silicon-specific optimization config"""
        return GPUOptimizationConfig(
            vendor=GPUVendor.APPLE,
            use_mixed_precision=True,
            use_tf32=False,
            use_fp16=True,
            use_bf16=False,
            use_fp8=False,
            use_memory_pool=False,  # Unified memory
            memory_fraction=0.8,  # Leave more for system
            allow_growth=True,
            use_cudnn_benchmark=False,
            use_tensor_cores=False,  # Neural Engine is separate
            num_streams=1,
            persistent_workers=False,
            optimal_batch_size=caps.optimal_batch_size,
            gradient_accumulation_steps=2,
            vendor_specific={
                "use_mps": True,
                "use_neural_engine": False,  # Not exposed via PyTorch
            },
        )

    def _create_default_config(self, caps: GPUCapabilities) -> GPUOptimizationConfig:
        """Create default/fallback optimization config"""
        return GPUOptimizationConfig(
            vendor=caps.vendor,
            use_mixed_precision=False,
            use_tf32=False,
            use_fp16=False,
            use_bf16=False,
            use_fp8=False,
            use_memory_pool=False,
            memory_fraction=0.9,
            allow_growth=True,
            use_cudnn_benchmark=False,
            use_tensor_cores=False,
            num_streams=1,
            persistent_workers=False,
            optimal_batch_size=16,
            gradient_accumulation_steps=1,
            vendor_specific={},
        )

    def apply_optimizations(self):
        """Apply GPU-specific optimizations to PyTorch"""
        if not self.config:
            logger.warning("No GPU optimization config available")
            return

        try:
            import torch

            logger.info("🚀 Applying GPU optimizations...")

            if self.config.vendor == GPUVendor.NVIDIA:
                self._apply_nvidia_optimizations(torch)
            elif self.config.vendor == GPUVendor.AMD:
                self._apply_amd_optimizations(torch)
            elif self.config.vendor == GPUVendor.INTEL:
                self._apply_intel_optimizations(torch)
            elif self.config.vendor == GPUVendor.APPLE:
                self._apply_apple_optimizations(torch)

            logger.info("✅ GPU optimizations applied")

        except Exception as e:
            logger.error(f"Failed to apply GPU optimizations: {e}")

    def _apply_nvidia_optimizations(self, torch):
        """Apply NVIDIA-specific optimizations"""
        config = self.config.vendor_specific

        # cuDNN benchmark
        if self.config.use_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("   ✓ cuDNN benchmark enabled")

        # TF32
        if self.config.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            logger.info("   ✓ TF32 enabled for matmul")

        # Memory management
        if self.config.use_memory_pool:
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            logger.info(
                f"   ✓ Memory fraction set to {self.config.memory_fraction:.0%}"
            )

        # Disable blocking for async execution
        if not config.get("cuda_launch_blocking", False):
            import os

            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

        logger.info(f"   ✓ Optimal batch size: {self.config.optimal_batch_size}")
        if self.config.use_bf16:
            logger.info("   ✓ BF16 mixed precision recommended")
        elif self.config.use_fp16:
            logger.info("   ✓ FP16 mixed precision recommended")

    def _apply_amd_optimizations(self, torch):
        """Apply AMD-specific optimizations"""
        config = self.config.vendor_specific

        # MIOpen benchmark
        if self.config.use_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("   ✓ MIOpen benchmark enabled")

        # Memory management
        if self.config.use_memory_pool:
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            logger.info(
                f"   ✓ Memory fraction set to {self.config.memory_fraction:.0%}"
            )

        logger.info(f"   ✓ Optimal batch size: {self.config.optimal_batch_size}")
        logger.info(f"   ✓ Wave size: {config.get('wave_size', 64)}")

        if self.config.use_bf16:
            logger.info("   ✓ BF16 mixed precision recommended (CDNA2+/RDNA3+)")
        elif self.config.use_fp16:
            logger.info("   ✓ FP16 mixed precision recommended")

    def _apply_intel_optimizations(self, torch):
        """Apply Intel-specific optimizations"""
        try:
            import intel_extension_for_pytorch as ipex

            logger.info("   ✓ Intel Extension for PyTorch available")
        except ImportError:
            logger.warning("   ⚠️  Intel Extension for PyTorch not found")

    def _apply_apple_optimizations(self, torch):
        """Apply Apple Silicon-specific optimizations"""
        if hasattr(torch.backends, "mps"):
            logger.info("   ✓ Metal Performance Shaders available")
            logger.info(
                "   ⚠️  Note: Unified memory - adjust batch sizes conservatively"
            )

    def get_device_string(self) -> str:
        """Get the appropriate device string for PyTorch"""
        if not self.capabilities:
            return "cpu"

        if self.capabilities.vendor == GPUVendor.NVIDIA:
            return "cuda"
        elif self.capabilities.vendor == GPUVendor.AMD:
            return "cuda"  # ROCm uses CUDA API
        elif self.capabilities.vendor == GPUVendor.INTEL:
            return "xpu"
        elif self.capabilities.vendor == GPUVendor.APPLE:
            return "mps"
        else:
            return "cpu"


# Convenience functions
def create_gpu_optimizer() -> GPUOptimizer:
    """Create and initialize GPU optimizer"""
    return GPUOptimizer()


def get_optimal_batch_size() -> int:
    """Get optimal batch size for current GPU"""
    optimizer = create_gpu_optimizer()
    if optimizer.config:
        return optimizer.config.optimal_batch_size
    return 16  # Default fallback


# Export
__all__ = [
    "GPUVendor",
    "NVIDIAArchitecture",
    "AMDArchitecture",
    "GPUCapabilities",
    "GPUOptimizationConfig",
    "GPUOptimizer",
    "create_gpu_optimizer",
    "get_optimal_batch_size",
]
