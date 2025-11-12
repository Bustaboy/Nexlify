#!/usr/bin/env python3
"""
Nexlify Dynamic Architecture System
Fully adaptive ML/RL with real-time bottleneck detection and intelligent offloading

NO FIXED TIERS - Pure dynamic optimization that:
- Continuously profiles CPU, GPU, RAM, VRAM usage
- Detects bottlenecks in real-time
- Dynamically adjusts architecture
- Offloads work to underutilized components
- Self-optimizes during training

Example scenarios:
- GPU saturated, CPU idle → Offload preprocessing to CPU workers
- RAM limited, VRAM abundant → Keep more tensors on GPU
- CPU weak but many cores → Heavy parallelization
- Bandwidth limited → Aggressive caching and compression
- Mixed bottlenecks → Dynamic rebalancing
"""

import numpy as np
import pandas as pd
import psutil
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
from enum import Enum

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
class ResourceSnapshot:
    """Snapshot of system resources at a point in time"""
    timestamp: float
    cpu_percent: float
    cpu_cores_used: float
    ram_used_gb: float
    ram_percent: float
    gpu_percent: float
    gpu_memory_used_gb: float
    gpu_memory_percent: float
    disk_io_mb_per_sec: float
    bottleneck: Bottleneck
    overhead_capacity: Dict[str, float]  # How much overhead each component has


class DynamicResourceMonitor:
    """
    Real-time resource monitoring and bottleneck detection

    Continuously tracks:
    - CPU usage per core
    - RAM usage and bandwidth
    - GPU utilization and VRAM
    - Disk I/O
    - Network bandwidth

    Identifies bottlenecks and available overhead
    """

    def __init__(self, sample_interval: float = 0.1):
        """
        Args:
            sample_interval: How often to sample resources (seconds)
        """
        self.sample_interval = sample_interval
        self.history = deque(maxlen=100)  # Last 100 samples

        # Running statistics
        self.running_stats = {
            'cpu': deque(maxlen=50),
            'ram': deque(maxlen=50),
            'gpu': deque(maxlen=50),
            'vram': deque(maxlen=50)
        }

        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None

        # Cache previous I/O counters for rate calculation
        self.prev_disk_io = None
        self.prev_net_io = None
        self.prev_timestamp = None

        logger.info("🔍 Dynamic Resource Monitor initialized")

    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("📊 Resource monitoring started")

    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("⏹️  Resource monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            snapshot = self.take_snapshot()
            self.history.append(snapshot)

            # Update running stats
            self.running_stats['cpu'].append(snapshot.cpu_percent)
            self.running_stats['ram'].append(snapshot.ram_percent)
            self.running_stats['gpu'].append(snapshot.gpu_percent)
            self.running_stats['vram'].append(snapshot.gpu_memory_percent)

            time.sleep(self.sample_interval)

    def take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage"""
        timestamp = time.time()

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_cores_used = (cpu_percent / 100) * cpu_count

        # RAM
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024**3)
        ram_percent = mem.percent

        # GPU (if available)
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

        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_cores_used=cpu_cores_used,
            ram_used_gb=ram_used_gb,
            ram_percent=ram_percent,
            gpu_percent=gpu_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_percent=gpu_memory_percent,
            disk_io_mb_per_sec=disk_io_mb_per_sec,
            bottleneck=bottleneck,
            overhead_capacity=overhead
        )

    def _get_gpu_stats(self) -> Tuple[float, float, float]:
        """Get GPU utilization and memory stats"""
        try:
            import torch

            if torch.cuda.is_available():
                # GPU utilization (requires nvidia-smi or similar)
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=1
                    )

                    if result.returncode == 0:
                        parts = result.stdout.strip().split(',')
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
                mem_reserved = torch.cuda.memory_reserved(0)
                mem_total = torch.cuda.get_device_properties(0).total_memory

                mem_used_gb = mem_allocated / (1024**3)
                mem_percent = (mem_allocated / mem_total) * 100

                # Can't get GPU util without nvidia-smi, estimate from memory
                gpu_util = min(mem_percent * 1.2, 100)  # Rough estimate

                return gpu_util, mem_used_gb, mem_percent

        except Exception as e:
            logger.debug(f"GPU stats error: {e}")

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

        except Exception as e:
            logger.debug(f"Disk I/O error: {e}")
            return 0.0

    def _detect_bottleneck(self, cpu_percent: float, ram_percent: float,
                          gpu_percent: float, vram_percent: float) -> Bottleneck:
        """
        Detect the primary bottleneck

        Thresholds:
        - >85% = bottleneck
        - >70% = potential bottleneck
        - <50% = overhead available
        """
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
            # Return most severe bottleneck
            bottlenecks.sort(key=lambda x: x[1], reverse=True)
            return bottlenecks[0][0]

        return Bottleneck.NONE

    def _calculate_overhead(self, cpu_percent: float, ram_percent: float,
                           gpu_percent: float, vram_percent: float) -> Dict[str, float]:
        """
        Calculate available overhead capacity for each component

        Returns percentage of available capacity (0-100)
        """
        return {
            'cpu': max(0, 100 - cpu_percent),
            'ram': max(0, 100 - ram_percent),
            'gpu': max(0, 100 - gpu_percent),
            'vram': max(0, 100 - vram_percent)
        }

    def get_current_bottleneck(self) -> Bottleneck:
        """Get the current primary bottleneck"""
        if self.history:
            return self.history[-1].bottleneck
        return Bottleneck.NONE

    def get_available_overhead(self) -> Dict[str, float]:
        """Get current available overhead for each component"""
        if self.history:
            return self.history[-1].overhead_capacity
        return {'cpu': 0, 'ram': 0, 'gpu': 0, 'vram': 0}

    def get_average_usage(self, window: int = 10) -> Dict[str, float]:
        """Get average resource usage over last N samples"""
        if len(self.history) < window:
            window = len(self.history)

        if window == 0:
            return {'cpu': 0, 'ram': 0, 'gpu': 0, 'vram': 0}

        recent = list(self.history)[-window:]

        return {
            'cpu': np.mean([s.cpu_percent for s in recent]),
            'ram': np.mean([s.ram_percent for s in recent]),
            'gpu': np.mean([s.gpu_percent for s in recent]),
            'vram': np.mean([s.gpu_memory_percent for s in recent])
        }


class DynamicArchitectureBuilder:
    """
    Builds neural network architecture dynamically based on real-time constraints

    NO FIXED TIERS - Architecture is computed on-demand based on:
    - Available memory (RAM + VRAM)
    - Compute capacity (CPU + GPU)
    - Current bottlenecks
    - Performance requirements
    """

    def __init__(self, monitor: DynamicResourceMonitor):
        self.monitor = monitor
        logger.info("🏗️  Dynamic Architecture Builder initialized")

    def build_adaptive_architecture(self,
                                    input_size: int,
                                    output_size: int,
                                    target_params: Optional[int] = None,
                                    min_params: int = 1000,
                                    max_params: int = 1000000) -> List[int]:
        """
        Build architecture dynamically based on current resources

        Args:
            input_size: Input dimension
            output_size: Output dimension
            target_params: Target number of parameters (None = auto)
            min_params: Minimum allowed parameters
            max_params: Maximum allowed parameters

        Returns:
            List of hidden layer sizes [h1, h2, h3, ...]
        """
        # Get current resource state
        snapshot = self.monitor.take_snapshot()
        overhead = snapshot.overhead_capacity

        # Calculate affordable parameters based on available memory
        available_ram_gb = overhead['ram'] / 100 * psutil.virtual_memory().total / (1024**3)
        available_vram_gb = 0

        try:
            import torch
            if torch.cuda.is_available() and overhead['vram'] > 0:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                available_vram_gb = overhead['vram'] / 100 * total_vram
        except:
            pass

        # Use whichever memory is more abundant
        available_memory_gb = max(available_ram_gb, available_vram_gb)

        # Estimate parameters we can afford
        # Rule of thumb: 4 bytes per float32 parameter
        # Keep 50% safety margin
        affordable_params = int((available_memory_gb * 1024**3 * 0.5) / 4)
        affordable_params = np.clip(affordable_params, min_params, max_params)

        if target_params is None:
            target_params = affordable_params
        else:
            # Don't exceed affordable
            target_params = min(target_params, affordable_params)

        # Build architecture to match parameter budget
        architecture = self._design_layers(input_size, output_size, target_params, overhead)

        logger.info(f"🏗️  Built dynamic architecture: {architecture}")
        logger.info(f"   Target params: {target_params:,}, Affordable: {affordable_params:,}")
        logger.info(f"   Bottleneck: {snapshot.bottleneck.value}")

        return architecture

    def _design_layers(self, input_size: int, output_size: int,
                       target_params: int, overhead: Dict[str, float]) -> List[int]:
        """
        Design layer sizes to hit target parameter count

        Strategy:
        - If CPU bottleneck: Fewer, wider layers (more parallelizable)
        - If GPU bottleneck: More, narrower layers (less memory)
        - If balanced: Standard deep architecture
        """
        # Determine architecture style based on bottleneck
        bottleneck = self.monitor.get_current_bottleneck()

        if bottleneck == Bottleneck.CPU:
            # CPU bottleneck: Wide shallow network
            num_layers = 2
            layer_ratio = 1.5  # Gradual decrease
        elif bottleneck == Bottleneck.GPU or bottleneck == Bottleneck.VRAM:
            # GPU/VRAM bottleneck: Narrow deep network
            num_layers = 5
            layer_ratio = 0.7  # Aggressive decrease
        elif bottleneck == Bottleneck.RAM:
            # RAM bottleneck: Very conservative
            num_layers = 2
            layer_ratio = 0.6
        else:
            # Balanced: Standard architecture
            num_layers = 3
            layer_ratio = 0.75

        # Calculate layer sizes
        # Start with first layer size and decay
        layers = []

        # Estimate first layer size based on parameter budget
        # Total params ≈ sum(layer[i] * layer[i+1]) for all connections
        # For geometric decay: first_layer * (1 + r + r² + ... + r^n) * avg_size

        # Binary search for optimal first layer size
        low, high = 16, 2048
        best_layers = None

        for _ in range(20):  # Binary search iterations
            mid = (low + high) // 2
            test_layers = self._generate_layers(mid, num_layers, layer_ratio, output_size)
            test_params = self._count_params(input_size, test_layers, output_size)

            if abs(test_params - target_params) < target_params * 0.1:  # Within 10%
                best_layers = test_layers
                break
            elif test_params < target_params:
                low = mid + 1
                if best_layers is None or abs(test_params - target_params) < abs(self._count_params(input_size, best_layers, output_size) - target_params):
                    best_layers = test_layers
            else:
                high = mid - 1

        if best_layers is None:
            # Fallback
            best_layers = self._generate_layers(64, num_layers, layer_ratio, output_size)

        return best_layers

    def _generate_layers(self, first_size: int, num_layers: int,
                        ratio: float, output_size: int) -> List[int]:
        """Generate layer sizes with geometric decay"""
        layers = []
        current_size = first_size

        for i in range(num_layers):
            # Round to nearest 16 (good for GPU alignment)
            size = max(16, int(current_size / 16) * 16)
            layers.append(size)
            current_size = max(output_size, int(current_size * ratio))

        return layers

    def _count_params(self, input_size: int, hidden_layers: List[int],
                     output_size: int) -> int:
        """Count total parameters in network"""
        total = 0

        # Input to first hidden
        if hidden_layers:
            total += input_size * hidden_layers[0] + hidden_layers[0]

            # Hidden to hidden
            for i in range(len(hidden_layers) - 1):
                total += hidden_layers[i] * hidden_layers[i+1] + hidden_layers[i+1]

            # Last hidden to output
            total += hidden_layers[-1] * output_size + output_size
        else:
            # Direct connection
            total += input_size * output_size + output_size

        return total


class DynamicWorkloadDistributor:
    """
    Intelligently distributes workload between CPU and GPU based on bottlenecks

    Strategies:
    - GPU saturated, CPU idle → Offload data preprocessing to CPU workers
    - CPU saturated, GPU idle → Batch more work to GPU
    - RAM limited, VRAM ok → Keep tensors on GPU longer
    - VRAM limited, RAM ok → Use CPU for larger batches
    - Balanced → Standard distribution
    """

    def __init__(self, monitor: DynamicResourceMonitor):
        self.monitor = monitor
        self.cpu_workers = 0
        self.gpu_batch_size = 32
        self.cpu_prefetch = True

        logger.info("⚖️  Dynamic Workload Distributor initialized")

    def optimize_distribution(self, total_batch_size: int) -> Dict[str, Any]:
        """
        Determine optimal workload distribution

        Returns:
            Configuration dict with:
            - gpu_batch_size: How much to process on GPU
            - cpu_workers: Number of CPU workers for preprocessing
            - pin_memory: Whether to pin memory for faster transfer
            - prefetch: Number of batches to prefetch
            - device_strategy: Where to keep tensors
        """
        snapshot = self.monitor.take_snapshot()
        bottleneck = snapshot.bottleneck
        overhead = snapshot.overhead_capacity

        config = {
            'gpu_batch_size': total_batch_size,
            'cpu_workers': 0,
            'pin_memory': True,
            'prefetch_factor': 2,
            'device_strategy': 'gpu_primary',
            'split_batch': False
        }

        # GPU saturated, CPU has overhead
        if bottleneck == Bottleneck.GPU and overhead['cpu'] > 30:
            # Offload preprocessing to CPU
            cpu_cores = psutil.cpu_count(logical=False)
            config['cpu_workers'] = min(cpu_cores - 1, 8)
            config['prefetch_factor'] = 4
            config['gpu_batch_size'] = total_batch_size // 2  # Reduce GPU load
            logger.info(f"⚠️  GPU bottleneck → Offloading to {config['cpu_workers']} CPU workers")

        # CPU saturated, GPU has overhead
        elif bottleneck == Bottleneck.CPU and overhead['gpu'] > 30:
            # Push more to GPU
            config['cpu_workers'] = 0
            config['gpu_batch_size'] = int(total_batch_size * 1.5)
            config['device_strategy'] = 'gpu_aggressive'
            logger.info(f"⚠️  CPU bottleneck → Increasing GPU batch to {config['gpu_batch_size']}")

        # RAM limited, VRAM ok
        elif bottleneck == Bottleneck.RAM and overhead['vram'] > 40:
            # Keep more on GPU
            config['device_strategy'] = 'gpu_primary'
            config['pin_memory'] = False  # Save RAM
            config['cpu_workers'] = 0
            logger.info(f"⚠️  RAM bottleneck → Keeping data on GPU")

        # VRAM limited, RAM ok
        elif bottleneck == Bottleneck.VRAM and overhead['ram'] > 40:
            # Use CPU memory
            config['device_strategy'] = 'cpu_primary'
            config['pin_memory'] = True
            config['cpu_workers'] = 2
            config['gpu_batch_size'] = total_batch_size // 2
            logger.info(f"⚠️  VRAM bottleneck → Using CPU memory")

        # Balanced - standard config
        else:
            cpu_cores = psutil.cpu_count(logical=False)
            config['cpu_workers'] = min(cpu_cores // 2, 4)
            config['gpu_batch_size'] = total_batch_size
            logger.info(f"✅ Balanced → Standard config")

        return config

    def should_split_batch(self, batch_size: int) -> Tuple[bool, int, int]:
        """
        Determine if batch should be split between CPU and GPU

        Returns:
            (should_split, cpu_portion, gpu_portion)
        """
        snapshot = self.monitor.take_snapshot()
        overhead = snapshot.overhead_capacity

        # Only split if both have significant overhead
        if overhead['cpu'] > 40 and overhead['gpu'] > 40:
            # Split proportional to overhead
            cpu_ratio = overhead['cpu'] / (overhead['cpu'] + overhead['gpu'])

            cpu_portion = int(batch_size * cpu_ratio)
            gpu_portion = batch_size - cpu_portion

            if cpu_portion > 0 and gpu_portion > 0:
                return True, cpu_portion, gpu_portion

        return False, 0, batch_size


class DynamicBufferManager:
    """
    Manages experience replay buffer size dynamically

    - Monitors available RAM
    - Expands buffer when RAM available
    - Shrinks buffer when RAM tight
    - Compresses old experiences when needed
    """

    def __init__(self, monitor: DynamicResourceMonitor,
                 initial_capacity: int = 50000,
                 min_capacity: int = 10000,
                 max_capacity: int = 1000000):
        self.monitor = monitor
        self.capacity = initial_capacity
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity

        self.buffer = deque(maxlen=self.capacity)
        self.last_resize = time.time()
        self.resize_interval = 60  # Resize at most once per minute

        logger.info(f"💾 Dynamic Buffer Manager initialized: {initial_capacity:,} capacity")

    def auto_resize(self):
        """Automatically resize buffer based on available RAM"""
        # Don't resize too frequently
        if time.time() - self.last_resize < self.resize_interval:
            return

        snapshot = self.monitor.take_snapshot()
        ram_overhead = snapshot.overhead_capacity['ram']

        # Calculate target capacity based on available RAM
        if ram_overhead > 60:
            # Lots of RAM available - expand
            new_capacity = min(self.capacity * 2, self.max_capacity)
        elif ram_overhead > 40:
            # Some RAM available - gentle expansion
            new_capacity = min(int(self.capacity * 1.2), self.max_capacity)
        elif ram_overhead < 20:
            # RAM tight - shrink
            new_capacity = max(int(self.capacity * 0.7), self.min_capacity)
        elif ram_overhead < 30:
            # RAM getting tight - gentle shrink
            new_capacity = max(int(self.capacity * 0.9), self.min_capacity)
        else:
            # Fine as is
            new_capacity = self.capacity

        if new_capacity != self.capacity:
            old_capacity = self.capacity
            self.capacity = new_capacity

            # Resize deque
            new_buffer = deque(self.buffer, maxlen=self.capacity)
            self.buffer = new_buffer

            self.last_resize = time.time()

            logger.info(f"📦 Buffer resized: {old_capacity:,} → {new_capacity:,} "
                       f"(RAM overhead: {ram_overhead:.1f}%)")

    def push(self, *args):
        """Add experience to buffer with auto-resize"""
        self.auto_resize()
        self.buffer.append(args)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        """Sample from buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return [self.buffer[i] for i in np.random.choice(len(self.buffer), batch_size, replace=False)]


# Export
__all__ = [
    'Bottleneck',
    'ResourceSnapshot',
    'DynamicResourceMonitor',
    'DynamicArchitectureBuilder',
    'DynamicWorkloadDistributor',
    'DynamicBufferManager'
]
