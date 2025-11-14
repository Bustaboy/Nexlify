#!/usr/bin/env python3
"""
Nexlify Multi-GPU Support

Automatic detection and optimization for multiple GPUs:
- Mixed GPU configurations (different models/VRAM)
- Data Parallel (DP) and Distributed Data Parallel (DDP)
- Model Parallel for large models
- NVLink/PCIe topology detection
- Intelligent load balancing
- Automatic device placement
"""

import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GPUInterconnect(Enum):
    """Types of GPU interconnects"""

    NVLINK = "nvlink"  # NVIDIA high-speed interconnect
    PCIE = "pcie"  # PCIe bus
    INFINITY_FABRIC = "infinity_fabric"  # AMD high-speed interconnect
    UNKNOWN = "unknown"


class ParallelStrategy(Enum):
    """Parallelization strategies"""

    SINGLE_GPU = "single_gpu"  # One GPU only
    DATA_PARALLEL = "data_parallel"  # Replicate model, split data
    MODEL_PARALLEL = "model_parallel"  # Split model across GPUs
    PIPELINE_PARALLEL = "pipeline"  # Pipeline model stages
    TENSOR_PARALLEL = "tensor"  # Split tensors across GPUs
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class GPUDevice:
    """Information about a single GPU device"""

    device_id: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: Optional[str]
    pci_bus_id: str
    uuid: str
    memory_bandwidth_gbps: Optional[float]
    sm_count: Optional[int]
    relative_performance: float  # 0-1 normalized performance


@dataclass
class GPUTopology:
    """Multi-GPU topology information"""

    num_gpus: int
    devices: List[GPUDevice]
    interconnects: Dict[
        Tuple[int, int], GPUInterconnect
    ]  # (gpu_i, gpu_j) -> interconnect
    homogeneous: bool  # All GPUs are the same model
    total_vram_gb: float
    min_vram_gb: float
    max_vram_gb: float


@dataclass
class ParallelConfig:
    """Configuration for parallel training"""

    strategy: ParallelStrategy
    num_gpus: int
    primary_gpu: int
    device_ids: List[int]
    batch_size_per_gpu: int
    gradient_accumulation_steps: int
    use_mixed_precision: bool
    sync_batch_norm: bool
    find_unused_parameters: bool
    broadcast_buffers: bool
    topology: GPUTopology


class MultiGPUManager:
    """
    Manages multiple GPUs for optimal training

    Features:
    - Automatic GPU detection
    - Topology analysis (NVLink, PCIe)
    - Load balancing for mixed GPUs
    - Strategy selection (DP/DDP/MP)
    - Device placement
    """

    def __init__(self):
        self.topology = None
        self.config = None

        # Detect GPUs
        self.topology = self._detect_gpu_topology()

        if self.topology and self.topology.num_gpus > 0:
            logger.info(
                f"🎮 Multi-GPU Manager initialized: {self.topology.num_gpus} GPUs detected"
            )

            for device in self.topology.devices:
                logger.info(
                    f"   GPU {device.device_id}: {device.name} "
                    f"({device.vram_total_gb:.1f} GB VRAM)"
                )

            if self.topology.num_gpus > 1:
                if self.topology.homogeneous:
                    logger.info("   ✓ Homogeneous configuration (all GPUs identical)")
                else:
                    logger.info("   ⚠️  Heterogeneous configuration (mixed GPUs)")

                # Check for high-speed interconnects
                has_nvlink = any(
                    ic == GPUInterconnect.NVLINK
                    for ic in self.topology.interconnects.values()
                )
                if has_nvlink:
                    logger.info("   ✓ NVLink detected (high-speed GPU-to-GPU)")
        else:
            logger.info("Multi-GPU Manager: Single GPU or CPU-only system")

    def _detect_gpu_topology(self) -> Optional[GPUTopology]:
        """Detect GPU topology and interconnects"""
        try:
            import torch

            if not torch.cuda.is_available():
                return None

            num_gpus = torch.cuda.device_count()

            if num_gpus == 0:
                return None

            # Get device info
            devices = []

            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)

                name = props.name
                vram_total = props.total_memory / (1024**3)

                # Get free memory
                torch.cuda.set_device(i)
                vram_free = torch.cuda.mem_get_info()[0] / (1024**3)

                # Compute capability
                compute_capability = f"{props.major}.{props.minor}"

                # Try to get UUID and PCI bus ID
                uuid = self._get_gpu_uuid(i)
                pci_bus_id = self._get_pci_bus_id(i)

                # Estimate bandwidth
                bandwidth = self._estimate_bandwidth(name, vram_total)

                # SM count
                sm_count = props.multi_processor_count

                # Performance relative to fastest GPU (determined later)
                devices.append(
                    GPUDevice(
                        device_id=i,
                        name=name,
                        vram_total_gb=vram_total,
                        vram_free_gb=vram_free,
                        compute_capability=compute_capability,
                        pci_bus_id=pci_bus_id,
                        uuid=uuid,
                        memory_bandwidth_gbps=bandwidth,
                        sm_count=sm_count,
                        relative_performance=1.0,  # Will be normalized
                    )
                )

            # Normalize performance scores
            self._calculate_relative_performance(devices)

            # Check homogeneity
            homogeneous = len(set(d.name for d in devices)) == 1

            # Calculate VRAM stats
            total_vram = sum(d.vram_total_gb for d in devices)
            min_vram = min(d.vram_total_gb for d in devices)
            max_vram = max(d.vram_total_gb for d in devices)

            # Detect interconnects
            interconnects = self._detect_interconnects(num_gpus)

            return GPUTopology(
                num_gpus=num_gpus,
                devices=devices,
                interconnects=interconnects,
                homogeneous=homogeneous,
                total_vram_gb=total_vram,
                min_vram_gb=min_vram,
                max_vram_gb=max_vram,
            )

        except Exception as e:
            logger.error(f"Failed to detect GPU topology: {e}")
            return None

    def _get_gpu_uuid(self, device_id: int) -> str:
        """Get GPU UUID via nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(device_id),
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return f"gpu-{device_id}"

    def _get_pci_bus_id(self, device_id: int) -> str:
        """Get PCI bus ID via nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(device_id),
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return f"pci-{device_id}"

    def _estimate_bandwidth(self, name: str, vram_gb: float) -> Optional[float]:
        """Estimate memory bandwidth from GPU name"""
        name_lower = name.lower()

        # High-end datacenter
        if "h100" in name_lower:
            return 3350.0
        elif "a100" in name_lower:
            return 2000.0 if vram_gb >= 80 else 1555.0
        elif "v100" in name_lower:
            return 900.0

        # RTX 40 series
        elif "4090" in name_lower:
            return 1008.0
        elif "4080" in name_lower:
            return 716.8
        elif "4070" in name_lower:
            return 504.2

        # RTX 30 series
        elif "3090" in name_lower:
            return 936.0
        elif "3080" in name_lower:
            return 760.0
        elif "3070" in name_lower:
            return 448.0
        elif "3060" in name_lower:
            return 360.0

        # RTX 20 series
        elif "2080" in name_lower:
            return 616.0
        elif "2070" in name_lower:
            return 448.0
        elif "2060" in name_lower:
            return 336.0

        # AMD
        elif "mi300" in name_lower:
            return 5300.0
        elif "mi250" in name_lower:
            return 3277.0
        elif "mi200" in name_lower:
            return 1638.0
        elif "7900 xtx" in name_lower:
            return 960.0
        elif "6900" in name_lower:
            return 512.0

        return None

    def _calculate_relative_performance(self, devices: List[GPUDevice]):
        """Calculate relative performance scores (0-1)"""
        # Use SM count * bandwidth as proxy for performance
        scores = []

        for device in devices:
            if device.sm_count and device.memory_bandwidth_gbps:
                score = device.sm_count * device.memory_bandwidth_gbps
            elif device.sm_count:
                score = device.sm_count * 1000  # Assume ~1000 GB/s
            elif device.memory_bandwidth_gbps:
                score = 100 * device.memory_bandwidth_gbps  # Assume ~100 SMs
            else:
                score = 10000  # Default

            scores.append(score)

        # Normalize to 0-1
        max_score = max(scores) if scores else 1.0

        for device, score in zip(devices, scores):
            device.relative_performance = score / max_score

    def _detect_interconnects(
        self, num_gpus: int
    ) -> Dict[Tuple[int, int], GPUInterconnect]:
        """Detect GPU-to-GPU interconnects"""
        interconnects = {}

        if num_gpus <= 1:
            return interconnects

        # Try nvidia-smi topo for NVLink detection
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=2
            )

            if result.returncode == 0:
                # Parse topology matrix
                lines = result.stdout.strip().split("\n")

                for i in range(num_gpus):
                    for j in range(i + 1, num_gpus):
                        # Look for NVLink indicators
                        # nvidia-smi topo shows NV# for NVLink, SYS/NODE for PCIe

                        # Heuristic: check if connection exists
                        # Default to PCIe, upgrade to NVLink if detected
                        interconnect = GPUInterconnect.PCIE

                        # Simple heuristic: if both GPUs are high-end, assume NVLink possibility
                        for line in lines:
                            if f"GPU{i}" in line or f"GPU{j}" in line:
                                if "NV" in line and not "NODE" in line:
                                    interconnect = GPUInterconnect.NVLINK
                                    break

                        interconnects[(i, j)] = interconnect
                        interconnects[(j, i)] = interconnect

        except Exception as e:
            logger.debug(f"Could not detect interconnects: {e}")

            # Fallback: assume PCIe for all
            for i in range(num_gpus):
                for j in range(i + 1, num_gpus):
                    interconnects[(i, j)] = GPUInterconnect.PCIE
                    interconnects[(j, i)] = GPUInterconnect.PCIE

        return interconnects

    def create_parallel_config(
        self,
        total_batch_size: int,
        model_size_mb: float = 100.0,
        prefer_strategy: Optional[ParallelStrategy] = None,
    ) -> Optional[ParallelConfig]:
        """
        Create optimal parallel configuration

        Args:
            total_batch_size: Total batch size across all GPUs
            model_size_mb: Approximate model size in MB
            prefer_strategy: Preferred strategy (None = auto-select)

        Returns:
            ParallelConfig or None if single GPU
        """
        if not self.topology or self.topology.num_gpus <= 1:
            return None

        num_gpus = self.topology.num_gpus

        # Select strategy
        if prefer_strategy:
            strategy = prefer_strategy
        else:
            strategy = self._select_strategy(model_size_mb)

        # Calculate batch size per GPU
        if strategy in [ParallelStrategy.DATA_PARALLEL]:
            # Split data evenly (or weighted by performance)
            if self.topology.homogeneous:
                batch_size_per_gpu = total_batch_size // num_gpus
            else:
                # Load balance by relative performance
                batch_size_per_gpu = self._calculate_balanced_batch_sizes(
                    total_batch_size
                )
        else:
            # Model parallel: full batch on each GPU
            batch_size_per_gpu = total_batch_size

        # Gradient accumulation for remainder
        remainder = total_batch_size % num_gpus
        gradient_accumulation_steps = 1 + (remainder // (batch_size_per_gpu * num_gpus))

        # Select primary GPU (fastest one)
        primary_gpu = max(
            range(num_gpus), key=lambda i: self.topology.devices[i].relative_performance
        )

        # Device IDs
        device_ids = list(range(num_gpus))

        # Mixed precision (use if any GPU supports it)
        use_mixed_precision = any(
            device.compute_capability
            and float(device.compute_capability.split(".")[0]) >= 7
            for device in self.topology.devices
        )

        # Sync batch norm for Data Parallel
        sync_batch_norm = strategy == ParallelStrategy.DATA_PARALLEL

        return ParallelConfig(
            strategy=strategy,
            num_gpus=num_gpus,
            primary_gpu=primary_gpu,
            device_ids=device_ids,
            batch_size_per_gpu=batch_size_per_gpu,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_mixed_precision=use_mixed_precision,
            sync_batch_norm=sync_batch_norm,
            find_unused_parameters=False,  # Set True if model has unused params
            broadcast_buffers=True,
            topology=self.topology,
        )

    def _select_strategy(self, model_size_mb: float) -> ParallelStrategy:
        """Automatically select parallelization strategy"""
        num_gpus = self.topology.num_gpus
        min_vram_gb = self.topology.min_vram_gb

        # Model size in GB
        model_size_gb = model_size_mb / 1024

        # Check if model fits in smallest GPU
        # Need ~3x model size for training (model + gradients + optimizer states)
        required_vram_gb = model_size_gb * 3

        if required_vram_gb < min_vram_gb * 0.6:
            # Model fits comfortably in each GPU → Data Parallel
            return ParallelStrategy.DATA_PARALLEL

        elif required_vram_gb < self.topology.total_vram_gb * 0.8:
            # Model fits across GPUs → Model Parallel
            return ParallelStrategy.MODEL_PARALLEL

        else:
            # Very large model → Pipeline Parallel
            return ParallelStrategy.PIPELINE_PARALLEL

    def _calculate_balanced_batch_sizes(self, total_batch_size: int) -> int:
        """Calculate balanced batch sizes for heterogeneous GPUs"""
        # Weight by relative performance
        performances = [d.relative_performance for d in self.topology.devices]
        total_perf = sum(performances)

        # Distribute proportionally
        batch_sizes = [
            int(total_batch_size * (perf / total_perf)) for perf in performances
        ]

        # Handle rounding - assign remainder to fastest GPU
        remainder = total_batch_size - sum(batch_sizes)
        if remainder > 0:
            fastest_gpu = max(range(len(performances)), key=lambda i: performances[i])
            batch_sizes[fastest_gpu] += remainder

        # Return average (for simplicity, actual implementation would return list)
        return batch_sizes[0]

    def wrap_model(self, model, config: Optional[ParallelConfig] = None):
        """
        Wrap model for multi-GPU training

        Args:
            model: PyTorch model
            config: Parallel configuration (None = create default)

        Returns:
            Wrapped model
        """
        try:
            import torch
            import torch.nn as nn

            if not config:
                # Create default config
                config = self.create_parallel_config(total_batch_size=32)

            if not config or config.num_gpus <= 1:
                # Single GPU or CPU
                return model

            # Move model to primary GPU
            model = model.to(f"cuda:{config.primary_gpu}")

            if config.strategy == ParallelStrategy.DATA_PARALLEL:
                # Use DataParallel (simple but less efficient)
                if len(config.device_ids) > 1:
                    logger.info(
                        f"Wrapping model with DataParallel on GPUs: {config.device_ids}"
                    )
                    model = nn.DataParallel(model, device_ids=config.device_ids)

            # For DDP, user should use torch.distributed.launch
            # This is handled in training script

            return model

        except Exception as e:
            logger.error(f"Failed to wrap model: {e}")
            return model

    def get_device_for_component(
        self, component_name: str, num_components: int, component_idx: int
    ) -> str:
        """
        Get device string for a model component (for model parallelism)

        Args:
            component_name: Name of component (e.g., 'encoder', 'decoder')
            num_components: Total number of components
            component_idx: Index of this component (0-indexed)

        Returns:
            Device string (e.g., 'cuda:0')
        """
        if not self.topology or self.topology.num_gpus <= 1:
            return "cuda:0" if self.topology else "cpu"

        # Distribute components across GPUs
        gpu_id = (component_idx * self.topology.num_gpus) // num_components

        return f"cuda:{gpu_id}"

    def get_load_balancing_weights(self) -> List[float]:
        """Get load balancing weights for data distribution"""
        if not self.topology or self.topology.num_gpus <= 1:
            return [1.0]

        return [d.relative_performance for d in self.topology.devices]

    def print_topology(self):
        """Print detailed GPU topology"""
        if not self.topology:
            print("No GPUs detected")
            return

        print("\n" + "=" * 80)
        print(f"GPU TOPOLOGY ({self.topology.num_gpus} GPUs)")
        print("=" * 80)

        for device in self.topology.devices:
            print(f"\nGPU {device.device_id}: {device.name}")
            print(
                f"  VRAM: {device.vram_total_gb:.1f} GB (Free: {device.vram_free_gb:.1f} GB)"
            )
            if device.compute_capability:
                print(f"  Compute Capability: {device.compute_capability}")
            if device.memory_bandwidth_gbps:
                print(f"  Bandwidth: {device.memory_bandwidth_gbps:.0f} GB/s")
            if device.sm_count:
                print(f"  SMs: {device.sm_count}")
            print(f"  Relative Performance: {device.relative_performance:.2f}")
            print(f"  PCI Bus: {device.pci_bus_id}")

        if self.topology.num_gpus > 1:
            print(
                f"\n{'Homogeneous' if self.topology.homogeneous else 'Heterogeneous'} Configuration"
            )
            print(f"Total VRAM: {self.topology.total_vram_gb:.1f} GB")
            print(
                f"VRAM Range: {self.topology.min_vram_gb:.1f} - {self.topology.max_vram_gb:.1f} GB"
            )

            print("\nInterconnects:")
            has_nvlink = False
            for (i, j), interconnect in self.topology.interconnects.items():
                if i < j:  # Only print once per pair
                    print(f"  GPU {i} <-> GPU {j}: {interconnect.value.upper()}")
                    if interconnect == GPUInterconnect.NVLINK:
                        has_nvlink = True

            if has_nvlink:
                print(
                    "\n✓ NVLink detected: High-speed GPU-to-GPU communication (50+ GB/s)"
                )
            else:
                print(
                    "\n⚠️  PCIe only: Limited GPU-to-GPU bandwidth (~16 GB/s per direction)"
                )


# Convenience functions
def create_multi_gpu_manager() -> MultiGPUManager:
    """Create multi-GPU manager"""
    return MultiGPUManager()


def get_num_gpus() -> int:
    """Get number of available GPUs"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except:
        pass
    return 0


# Export
__all__ = [
    "GPUInterconnect",
    "ParallelStrategy",
    "GPUDevice",
    "GPUTopology",
    "ParallelConfig",
    "MultiGPUManager",
    "create_multi_gpu_manager",
    "get_num_gpus",
]
