#!/usr/bin/env python3
"""
Nexlify Dynamic Architecture - Enhanced with Hyperthreading/SMT Support

Fully supports:
- Intel Hyperthreading (2 logical cores per physical)
- AMD SMT (2 logical cores per physical)
- Intelligent logical core utilization
- Workload-aware thread scheduling
- Per-core performance accounting
"""

import logging
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Import GPU optimizer
try:
    from nexlify.ml.nexlify_gpu_optimizations import (GPUCapabilities,
                                                      GPUOptimizer)

    GPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)


class Bottleneck(Enum):
    """Types of system bottlenecks"""

    CPU = "cpu"
    GPU = "gpu"
    RAM = "ram"
    VRAM = "vram"
    BANDWIDTH = "bandwidth"
    DISK_IO = "disk_io"
    NONE = "none"


@dataclass
class CPUTopology:
    """Detailed CPU topology information"""

    physical_cores: int
    logical_cores: int
    has_ht_smt: bool  # Hyperthreading or SMT enabled
    ht_efficiency: float  # Expected efficiency of hyperthreading (0.2-0.3 typically)
    effective_cores: float  # Physical + (Logical - Physical) * ht_efficiency
    architecture: str  # x86_64, ARM, etc.
    vendor: str  # Intel, AMD, ARM, etc.
    per_core_usage: List[float]  # Usage per logical core


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time"""

    timestamp: float
    cpu_percent: float
    cpu_cores_used: float
    cpu_topology: CPUTopology
    ram_used_gb: float
    ram_percent: float
    gpu_percent: float
    gpu_memory_used_gb: float
    gpu_memory_percent: float
    gpu_capabilities: Optional[Any]  # GPUCapabilities if available
    disk_io_mb_per_sec: float
    bottleneck: Bottleneck
    overhead_capacity: Dict[str, float]


class EnhancedDynamicResourceMonitor:
    """
    Enhanced resource monitoring with hyperthreading/SMT support

    Improvements:
    - Detects hyperthreading/SMT capability
    - Per-core usage monitoring
    - SMT efficiency calculation
    - Intelligent core allocation
    """

    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.history = deque(maxlen=100)

        # Running statistics
        self.running_stats = {
            "cpu": deque(maxlen=50),
            "ram": deque(maxlen=50),
            "gpu": deque(maxlen=50),
            "vram": deque(maxlen=50),
        }

        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None

        # Cache
        self.prev_disk_io = None
        self.prev_net_io = None
        self.prev_timestamp = None

        # Detect CPU topology once
        self.cpu_topology = self._detect_cpu_topology()

        # Initialize GPU optimizer
        self.gpu_optimizer = None
        if GPU_OPTIMIZER_AVAILABLE:
            try:
                self.gpu_optimizer = GPUOptimizer()
                if self.gpu_optimizer.capabilities:
                    self.gpu_optimizer.apply_optimizations()
            except Exception as e:
                logger.warning(f"Failed to initialize GPU optimizer: {e}")
                self.gpu_optimizer = None

        logger.info("🔍 Enhanced Dynamic Resource Monitor initialized")
        logger.info(
            f"   CPU Topology: {self.cpu_topology.physical_cores}P / "
            f"{self.cpu_topology.logical_cores}L cores "
            f"(HT/SMT: {'Yes' if self.cpu_topology.has_ht_smt else 'No'})"
        )
        if self.cpu_topology.has_ht_smt:
            logger.info(
                f"   Effective cores: {self.cpu_topology.effective_cores:.1f} "
                f"(HT efficiency: {self.cpu_topology.ht_efficiency*100:.0f}%)"
            )

    def _detect_cpu_topology(self) -> CPUTopology:
        """
        Detect detailed CPU topology including hyperthreading/SMT

        Intel Hyperthreading: 2 logical cores per physical core
        AMD SMT: 2 logical cores per physical core (some Threadripper: 1 logical = 1 physical)
        """
        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or 1

        # Detect hyperthreading/SMT
        has_ht_smt = logical > physical

        # Estimate hyperthreading efficiency
        # Hyperthreading typically provides 20-30% performance boost
        # So logical cores are ~20-30% as effective as physical cores
        if has_ht_smt:
            ht_ratio = logical / physical

            if ht_ratio == 2:
                # Standard 2-way SMT (most common)
                # Each logical core beyond physical is ~25% as effective
                ht_efficiency = 0.25
            elif ht_ratio > 2:
                # High SMT ratio (rare, some ARM chips)
                ht_efficiency = 0.20
            else:
                # Partial SMT
                ht_efficiency = 0.30
        else:
            ht_efficiency = 0.0

        # Calculate effective cores
        # Example: 8 physical, 16 logical, 25% efficiency
        # Effective = 8 + (16 - 8) * 0.25 = 8 + 2 = 10 effective cores
        effective_cores = physical + (logical - physical) * ht_efficiency

        # Detect vendor and architecture
        cpu_info = platform.processor()
        architecture = platform.machine()

        vendor = "Unknown"
        if "Intel" in cpu_info or "intel" in cpu_info.lower():
            vendor = "Intel"
        elif "AMD" in cpu_info or "amd" in cpu_info.lower():
            vendor = "AMD"
        elif "ARM" in architecture.upper() or "AARCH64" in architecture.upper():
            vendor = "ARM"

        # Get per-core usage
        try:
            per_core_usage = psutil.cpu_percent(interval=0, percpu=True)
        except:
            per_core_usage = [0.0] * logical

        return CPUTopology(
            physical_cores=physical,
            logical_cores=logical,
            has_ht_smt=has_ht_smt,
            ht_efficiency=ht_efficiency,
            effective_cores=effective_cores,
            architecture=architecture,
            vendor=vendor,
            per_core_usage=per_core_usage,
        )

    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()
            logger.info("📊 Enhanced resource monitoring started")

    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("⏹️  Enhanced resource monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            snapshot = self.take_snapshot()
            self.history.append(snapshot)

            # Update running stats
            self.running_stats["cpu"].append(snapshot.cpu_percent)
            self.running_stats["ram"].append(snapshot.ram_percent)
            self.running_stats["gpu"].append(snapshot.gpu_percent)
            self.running_stats["vram"].append(snapshot.gpu_memory_percent)

            time.sleep(self.sample_interval)

    def take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage"""
        timestamp = time.time()

        # CPU - enhanced with per-core monitoring
        cpu_percent = psutil.cpu_percent(interval=0)

        # Get per-core usage
        try:
            per_core_usage = psutil.cpu_percent(interval=0, percpu=True)
        except:
            per_core_usage = [cpu_percent] * self.cpu_topology.logical_cores

        # Update topology with current usage
        current_topology = CPUTopology(
            physical_cores=self.cpu_topology.physical_cores,
            logical_cores=self.cpu_topology.logical_cores,
            has_ht_smt=self.cpu_topology.has_ht_smt,
            ht_efficiency=self.cpu_topology.ht_efficiency,
            effective_cores=self.cpu_topology.effective_cores,
            architecture=self.cpu_topology.architecture,
            vendor=self.cpu_topology.vendor,
            per_core_usage=per_core_usage,
        )

        # Calculate effective cores in use
        # Account for SMT efficiency
        if current_topology.has_ht_smt:
            # Split into physical and hyperthreaded cores
            physical_usage = per_core_usage[: current_topology.physical_cores]
            ht_usage = per_core_usage[current_topology.physical_cores :]

            # Effective cores = physical usage + HT usage * efficiency
            physical_cores_used = sum(physical_usage) / 100
            ht_cores_used = sum(ht_usage) / 100 * current_topology.ht_efficiency

            cpu_cores_used = physical_cores_used + ht_cores_used
        else:
            cpu_cores_used = (cpu_percent / 100) * current_topology.physical_cores

        # RAM
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024**3)
        ram_percent = mem.percent

        # GPU
        gpu_percent, gpu_memory_used_gb, gpu_memory_percent = self._get_gpu_stats()

        # Disk I/O
        disk_io_mb_per_sec = self._get_disk_io_rate()

        # Detect bottleneck
        bottleneck = self._detect_bottleneck(
            cpu_percent, ram_percent, gpu_percent, gpu_memory_percent
        )

        # Calculate overhead capacity
        overhead = self._calculate_overhead(
            cpu_percent, ram_percent, gpu_percent, gpu_memory_percent
        )

        # GPU capabilities
        gpu_capabilities = None
        if self.gpu_optimizer and self.gpu_optimizer.capabilities:
            gpu_capabilities = self.gpu_optimizer.capabilities

        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_cores_used=cpu_cores_used,
            cpu_topology=current_topology,
            ram_used_gb=ram_used_gb,
            ram_percent=ram_percent,
            gpu_percent=gpu_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_percent=gpu_memory_percent,
            gpu_capabilities=gpu_capabilities,
            disk_io_mb_per_sec=disk_io_mb_per_sec,
            bottleneck=bottleneck,
            overhead_capacity=overhead,
        )

    def _get_gpu_stats(self) -> Tuple[float, float, float]:
        """Get GPU utilization and memory stats"""
        try:
            import torch

            if torch.cuda.is_available():
                # Try nvidia-smi first
                try:
                    import subprocess

                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=1,
                    )

                    if result.returncode == 0:
                        parts = result.stdout.strip().split(",")
                        gpu_util = float(parts[0])
                        mem_used_mb = float(parts[1])
                        mem_total_mb = float(parts[2])

                        mem_used_gb = mem_used_mb / 1024
                        mem_percent = (mem_used_mb / mem_total_mb) * 100

                        return gpu_util, mem_used_gb, mem_percent
                except:
                    pass

                # Fallback: PyTorch memory only
                mem_allocated = torch.cuda.memory_allocated(0)
                mem_total = torch.cuda.get_device_properties(0).total_memory

                mem_used_gb = mem_allocated / (1024**3)
                mem_percent = (mem_allocated / mem_total) * 100
                gpu_util = min(mem_percent * 1.2, 100)

                return gpu_util, mem_used_gb, mem_percent

        except:
            pass

        return 0.0, 0.0, 0.0

    def _get_disk_io_rate(self) -> float:
        """Get disk I/O rate in MB/s"""
        try:
            current_io = psutil.disk_io_counters()
            current_time = time.time()

            if self.prev_disk_io and self.prev_timestamp:
                read_bytes = current_io.read_bytes - self.prev_disk_io.read_bytes
                write_bytes = current_io.write_bytes - self.prev_disk_io.write_bytes
                time_delta = current_time - self.prev_timestamp

                if time_delta > 0:
                    mb_per_sec = (read_bytes + write_bytes) / time_delta / (1024**2)
                else:
                    mb_per_sec = 0.0
            else:
                mb_per_sec = 0.0

            self.prev_disk_io = current_io
            self.prev_timestamp = current_time

            return mb_per_sec

        except:
            return 0.0

    def _detect_bottleneck(
        self,
        cpu_percent: float,
        ram_percent: float,
        gpu_percent: float,
        vram_percent: float,
    ) -> Bottleneck:
        """Detect the primary bottleneck"""
        bottlenecks = []

        if cpu_percent > 85:
            bottlenecks.append((Bottleneck.CPU, cpu_percent))
        if ram_percent > 85:
            bottlenecks.append((Bottleneck.RAM, ram_percent))
        if gpu_percent > 85:
            bottlenecks.append((Bottleneck.GPU, gpu_percent))
        if vram_percent > 85:
            bottlenecks.append((Bottleneck.VRAM, vram_percent))

        if bottlenecks:
            bottlenecks.sort(key=lambda x: x[1], reverse=True)
            return bottlenecks[0][0]

        return Bottleneck.NONE

    def _calculate_overhead(
        self,
        cpu_percent: float,
        ram_percent: float,
        gpu_percent: float,
        vram_percent: float,
    ) -> Dict[str, float]:
        """Calculate available overhead capacity for each component"""
        return {
            "cpu": max(0, 100 - cpu_percent),
            "ram": max(0, 100 - ram_percent),
            "gpu": max(0, 100 - gpu_percent),
            "vram": max(0, 100 - vram_percent),
        }

    def calculate_optimal_workers(self, workload_type: str = "preprocessing") -> int:
        """
        Calculate optimal number of worker threads considering hyperthreading/SMT

        Args:
            workload_type: Type of work ('preprocessing', 'computation', 'io')

        Returns:
            Optimal number of workers
        """
        topology = self.cpu_topology
        snapshot = self.take_snapshot()

        # Get available CPU capacity
        cpu_overhead = snapshot.overhead_capacity["cpu"]

        if cpu_overhead < 10:
            # CPU saturated - minimal workers
            return 0

        # Calculate available effective cores
        available_effective_cores = topology.effective_cores * (cpu_overhead / 100)

        # Workload-specific adjustments
        if workload_type == "preprocessing":
            # Data preprocessing benefits moderately from HT/SMT
            # Use up to 80% of available effective cores
            optimal_workers = int(available_effective_cores * 0.8)

        elif workload_type == "computation":
            # Heavy computation may not benefit much from HT/SMT
            # Prefer physical cores
            available_physical = topology.physical_cores * (cpu_overhead / 100)
            optimal_workers = int(available_physical * 0.7)

        elif workload_type == "io":
            # I/O bound work benefits greatly from HT/SMT
            # Can use all logical cores
            available_logical = topology.logical_cores * (cpu_overhead / 100)
            optimal_workers = int(available_logical * 0.9)

        else:
            # Default: balanced
            optimal_workers = int(available_effective_cores * 0.75)

        # Clamp to reasonable range
        optimal_workers = max(0, min(optimal_workers, topology.logical_cores - 1))

        logger.debug(
            f"Optimal workers for {workload_type}: {optimal_workers} "
            f"(from {available_effective_cores:.1f} available effective cores)"
        )

        return optimal_workers

    def get_cpu_affinity_recommendation(self, num_workers: int) -> List[List[int]]:
        """
        Recommend CPU affinity for workers to maximize performance

        Strategy:
        - Spread workers across physical cores first
        - Use hyperthreaded cores only if needed
        - Avoid overloading single physical cores

        Returns:
            List of CPU core lists for each worker
        """
        topology = self.cpu_topology

        if not topology.has_ht_smt:
            # No hyperthreading - simple assignment
            return [[i] for i in range(num_workers)]

        # With hyperthreading: be smart about assignment
        physical_cores = topology.physical_cores
        logical_cores = topology.logical_cores

        # Assume first N logical cores are physical, rest are HT
        # (This is typical but not always true - best effort)

        affinities = []

        if num_workers <= physical_cores:
            # Fewer workers than physical cores - use physical cores only
            for i in range(num_workers):
                affinities.append([i])

        else:
            # More workers than physical cores - need to use HT cores

            # First, assign one worker per physical core
            for i in range(physical_cores):
                affinities.append([i])

            # Then, assign remaining workers to HT cores
            remaining = num_workers - physical_cores
            for i in range(remaining):
                ht_core = physical_cores + i
                if ht_core < logical_cores:
                    affinities.append([ht_core])

        return affinities

    def get_current_bottleneck(self) -> Bottleneck:
        """Get the current primary bottleneck"""
        if self.history:
            return self.history[-1].bottleneck
        return Bottleneck.NONE

    def get_available_overhead(self) -> Dict[str, float]:
        """Get current available overhead for each component"""
        if self.history:
            return self.history[-1].overhead_capacity
        return {"cpu": 0, "ram": 0, "gpu": 0, "vram": 0}

    def get_average_usage(self, window: int = 10) -> Dict[str, float]:
        """Get average resource usage over last N samples"""
        if len(self.history) < window:
            window = len(self.history)

        if window == 0:
            return {"cpu": 0, "ram": 0, "gpu": 0, "vram": 0}

        recent = list(self.history)[-window:]

        return {
            "cpu": np.mean([s.cpu_percent for s in recent]),
            "ram": np.mean([s.ram_percent for s in recent]),
            "gpu": np.mean([s.gpu_percent for s in recent]),
            "vram": np.mean([s.gpu_memory_percent for s in recent]),
        }

    def get_gpu_optimal_batch_size(self) -> int:
        """Get GPU-optimized batch size"""
        if self.gpu_optimizer and self.gpu_optimizer.config:
            return self.gpu_optimizer.config.optimal_batch_size
        return 32  # Default fallback

    def should_use_mixed_precision(self) -> bool:
        """Check if mixed precision should be used"""
        if self.gpu_optimizer and self.gpu_optimizer.config:
            return self.gpu_optimizer.config.use_mixed_precision
        return False

    def get_precision_dtype(self) -> str:
        """Get recommended precision dtype"""
        if not self.gpu_optimizer or not self.gpu_optimizer.config:
            return "float32"

        config = self.gpu_optimizer.config

        if config.use_fp8:
            return "float8"
        elif config.use_bf16:
            return "bfloat16"
        elif config.use_fp16:
            return "float16"
        elif config.use_tf32:
            return "tf32"
        else:
            return "float32"

    def get_device_string(self) -> str:
        """Get appropriate device string for PyTorch"""
        if self.gpu_optimizer:
            return self.gpu_optimizer.get_device_string()
        return "cpu"

    def get_gpu_info_summary(self) -> Dict[str, Any]:
        """Get summary of GPU information"""
        if not self.gpu_optimizer or not self.gpu_optimizer.capabilities:
            return {"available": False}

        caps = self.gpu_optimizer.capabilities
        config = self.gpu_optimizer.config

        return {
            "available": True,
            "vendor": caps.vendor.value,
            "name": caps.name,
            "architecture": caps.architecture,
            "vram_gb": caps.vram_gb,
            "has_tensor_cores": caps.has_tensor_cores,
            "optimal_batch_size": caps.optimal_batch_size,
            "use_mixed_precision": config.use_mixed_precision,
            "precision": self.get_precision_dtype(),
            "num_streams": config.num_streams,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        }


# Export
__all__ = [
    "Bottleneck",
    "CPUTopology",
    "ResourceSnapshot",
    "EnhancedDynamicResourceMonitor",
]
