#!/usr/bin/env python3
"""
Nexlify Adaptive Reinforcement Learning Agent
Optimized for consumer hardware with dynamic model scaling and resource adaptation

This module provides an intelligent RL agent that automatically adapts to:
- Varying VRAM (1GB - 24GB+)
- Different RAM configurations (4GB - 128GB+)
- CPU capabilities (dual-core to threadripper)
- GPU compute power (GTX 1050 to RTX 4090)

The agent dynamically selects model architecture, batch sizes, buffer sizes,
and training strategies based on detected hardware capabilities.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
from datetime import datetime
import json
from pathlib import Path
import psutil
import threading
import time

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class HardwareProfiler:
    """
    Enhanced hardware detection with granular GPU/CPU/RAM profiling
    Detects VRAM, compute capability, memory bandwidth, etc.
    """

    def __init__(self):
        self.profile = {}
        self.capabilities = {}
        self.optimal_config = {}

        logger.info("🔍 Hardware Profiler initializing...")
        self.detect_hardware()
        self.benchmark_performance()
        self.calculate_optimal_config()

    def detect_hardware(self):
        """Detect all hardware with granular details"""
        self.profile = {
            'cpu': self._detect_cpu(),
            'ram': self._detect_ram(),
            'gpu': self._detect_gpu(),
            'storage': self._detect_storage()
        }

    def _detect_cpu(self) -> Dict:
        """Detect CPU capabilities"""
        import platform

        cpu_info = {
            'cores_physical': psutil.cpu_count(logical=False) or 1,
            'cores_logical': psutil.cpu_count(logical=True) or 1,
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 2000,
            'frequency_max_mhz': psutil.cpu_freq().max if psutil.cpu_freq() else 3000,
            'architecture': platform.machine(),
            'usage_percent': psutil.cpu_percent(interval=0.1),
            'available_cores': psutil.cpu_count(logical=False) or 1
        }

        # Estimate CPU tier based on cores and frequency
        cores = cpu_info['cores_physical']
        freq = cpu_info['frequency_max_mhz']

        if cores >= 16 or (cores >= 8 and freq >= 4000):
            cpu_info['tier'] = 'high_end'  # Threadripper, i9, Ryzen 9
        elif cores >= 8 or (cores >= 6 and freq >= 3500):
            cpu_info['tier'] = 'mid_high'  # i7, Ryzen 7
        elif cores >= 6 or (cores >= 4 and freq >= 3000):
            cpu_info['tier'] = 'mid'  # i5, Ryzen 5
        elif cores >= 4:
            cpu_info['tier'] = 'low_mid'  # i3, older i5
        else:
            cpu_info['tier'] = 'low'  # Dual core

        logger.info(f"CPU: {cores} cores @ {freq:.0f} MHz ({cpu_info['tier']})")
        return cpu_info

    def _detect_ram(self) -> Dict:
        """Detect RAM capabilities"""
        mem = psutil.virtual_memory()

        ram_info = {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent_used': mem.percent,
            'free_for_ml_gb': max(0, mem.available / (1024**3) - 2)  # Reserve 2GB for OS
        }

        # RAM tier
        total_gb = ram_info['total_gb']
        if total_gb >= 64:
            ram_info['tier'] = 'ultra'  # 64GB+
        elif total_gb >= 32:
            ram_info['tier'] = 'high'  # 32-64GB
        elif total_gb >= 16:
            ram_info['tier'] = 'mid_high'  # 16-32GB
        elif total_gb >= 8:
            ram_info['tier'] = 'mid'  # 8-16GB
        elif total_gb >= 4:
            ram_info['tier'] = 'low'  # 4-8GB
        else:
            ram_info['tier'] = 'minimal'  # <4GB

        logger.info(f"RAM: {total_gb:.1f} GB total, {ram_info['available_gb']:.1f} GB available ({ram_info['tier']})")
        return ram_info

    def _detect_gpu(self) -> Dict:
        """Detect GPU capabilities with VRAM and compute info"""
        gpu_info = {
            'available': False,
            'vendor': 'none',
            'name': 'CPU Only',
            'vram_gb': 0,
            'compute_capability': None,
            'cuda_cores': 0,
            'tensor_cores': False,
            'supports_fp16': False,
            'tier': 'none'
        }

        # Try PyTorch CUDA detection first (most reliable)
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['vendor'] = 'nvidia'
                gpu_info['name'] = torch.cuda.get_device_name(0)

                # Get VRAM
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_info['vram_gb'] = vram_bytes / (1024**3)

                # Get compute capability
                compute_cap = torch.cuda.get_device_properties(0)
                gpu_info['compute_capability'] = f"{compute_cap.major}.{compute_cap.minor}"

                # Tensor cores available on compute capability >= 7.0 (Volta+)
                if compute_cap.major >= 7:
                    gpu_info['tensor_cores'] = True
                    gpu_info['supports_fp16'] = True

                # Estimate CUDA cores (approximate)
                gpu_info['cuda_cores'] = compute_cap.multi_processor_count * 128  # Rough estimate

                # Determine GPU tier based on name and VRAM
                gpu_name_lower = gpu_info['name'].lower()
                vram = gpu_info['vram_gb']

                if 'rtx 4090' in gpu_name_lower or 'rtx 4080' in gpu_name_lower or vram >= 20:
                    gpu_info['tier'] = 'ultra'  # RTX 4090, 4080, A100
                elif 'rtx 40' in gpu_name_lower or 'rtx 30' in gpu_name_lower or vram >= 10:
                    gpu_info['tier'] = 'high'  # RTX 4070, 3080, 3090
                elif 'rtx 20' in gpu_name_lower or 'gtx 1660' in gpu_name_lower or vram >= 6:
                    gpu_info['tier'] = 'mid'  # RTX 2060, 2070, GTX 1660
                elif 'gtx 16' in gpu_name_lower or 'gtx 10' in gpu_name_lower or vram >= 3:
                    gpu_info['tier'] = 'low_mid'  # GTX 1050, 1060
                else:
                    gpu_info['tier'] = 'low'  # Older GPUs

                logger.info(f"GPU: {gpu_info['name']} with {vram:.1f} GB VRAM ({gpu_info['tier']})")

        except ImportError:
            logger.warning("PyTorch not available - GPU detection limited")
        except Exception as e:
            logger.error(f"GPU detection error: {e}")

        # Fallback: Try nvidia-smi
        if not gpu_info['available']:
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        parts = lines[0].split(',')
                        gpu_info['available'] = True
                        gpu_info['vendor'] = 'nvidia'
                        gpu_info['name'] = parts[0].strip()

                        # Parse VRAM (format: "8192 MiB")
                        vram_str = parts[1].strip().split()[0]
                        gpu_info['vram_gb'] = int(vram_str) / 1024

                        # Infer capabilities from GPU name (nvidia-smi fallback)
                        gpu_name_lower = gpu_info['name'].lower()

                        # RTX 40 series (Ada Lovelace) - Compute 8.9
                        if 'rtx 40' in gpu_name_lower:
                            gpu_info['compute_capability'] = '8.9'
                            gpu_info['tensor_cores'] = True
                            gpu_info['supports_fp16'] = True
                            gpu_info['has_tensor_cores'] = True
                        # RTX 30 series (Ampere) - Compute 8.6
                        elif 'rtx 30' in gpu_name_lower:
                            gpu_info['compute_capability'] = '8.6'
                            gpu_info['tensor_cores'] = True
                            gpu_info['supports_fp16'] = True
                            gpu_info['has_tensor_cores'] = True
                        # RTX 20 series (Turing) - Compute 7.5
                        elif 'rtx 20' in gpu_name_lower:
                            gpu_info['compute_capability'] = '7.5'
                            gpu_info['tensor_cores'] = True
                            gpu_info['supports_fp16'] = True
                            gpu_info['has_tensor_cores'] = True
                        # GTX 16 series (Turing, no tensor cores) - Compute 7.5
                        elif 'gtx 16' in gpu_name_lower:
                            gpu_info['compute_capability'] = '7.5'
                            gpu_info['tensor_cores'] = False
                            gpu_info['supports_fp16'] = True
                            gpu_info['has_tensor_cores'] = False
                        # GTX 10 series (Pascal) - Compute 6.1
                        elif 'gtx 10' in gpu_name_lower:
                            gpu_info['compute_capability'] = '6.1'
                            gpu_info['tensor_cores'] = False
                            gpu_info['supports_fp16'] = True
                            gpu_info['has_tensor_cores'] = False
                        # Volta (V100, Titan V) - Compute 7.0
                        elif 'v100' in gpu_name_lower or 'titan v' in gpu_name_lower:
                            gpu_info['compute_capability'] = '7.0'
                            gpu_info['tensor_cores'] = True
                            gpu_info['supports_fp16'] = True
                            gpu_info['has_tensor_cores'] = True
                        # Default for modern NVIDIA GPUs
                        else:
                            gpu_info['compute_capability'] = '7.0'  # Conservative estimate
                            gpu_info['supports_fp16'] = True
                            # Assume tensor cores for modern GPUs with 6GB+ VRAM
                            if gpu_info['vram_gb'] >= 6:
                                gpu_info['tensor_cores'] = True
                                gpu_info['has_tensor_cores'] = True

                        # Determine GPU tier based on name and VRAM
                        vram = gpu_info['vram_gb']
                        if 'rtx 4090' in gpu_name_lower or 'rtx 4080' in gpu_name_lower or vram >= 20:
                            gpu_info['tier'] = 'ultra'  # RTX 4090, 4080, A100
                        elif 'rtx 40' in gpu_name_lower or 'rtx 30' in gpu_name_lower or vram >= 10:
                            gpu_info['tier'] = 'high'  # RTX 4070, 3080, 3090
                        elif 'rtx 20' in gpu_name_lower or 'gtx 1660' in gpu_name_lower or vram >= 6:
                            gpu_info['tier'] = 'mid'  # RTX 2060, 2070, GTX 1660
                        elif 'gtx 16' in gpu_name_lower or 'gtx 10' in gpu_name_lower or vram >= 3:
                            gpu_info['tier'] = 'low_mid'  # GTX 1050, 1060
                        else:
                            gpu_info['tier'] = 'low'  # Older GPUs

                        logger.info(f"GPU: {gpu_info['name']} with {gpu_info['vram_gb']:.1f} GB VRAM (nvidia-smi)")

            except Exception:
                pass

        if not gpu_info['available']:
            logger.info("No GPU detected - CPU-only mode will be used")

        return gpu_info

    def _detect_storage(self) -> Dict:
        """Detect storage capabilities"""
        import os

        usage = psutil.disk_usage(os.getcwd())

        storage_info = {
            'free_gb': usage.free / (1024**3),
            'total_gb': usage.total / (1024**3),
            'type': 'unknown'
        }

        # Try to detect SSD vs HDD
        try:
            import platform
            if platform.system() == 'Linux':
                with open('/sys/block/sda/queue/rotational', 'r') as f:
                    is_rotational = f.read().strip() == '1'
                    storage_info['type'] = 'HDD' if is_rotational else 'SSD'
        except:
            pass

        return storage_info

    def benchmark_performance(self):
        """Quick performance benchmark to measure actual compute speed"""
        logger.info("⚡ Running performance benchmark...")

        benchmark_results = {
            'cpu_gflops': self._benchmark_cpu(),
            'gpu_gflops': self._benchmark_gpu() if self.profile['gpu']['available'] else 0,
            'memory_bandwidth_gbps': self._benchmark_memory()
        }

        self.profile['benchmark'] = benchmark_results
        logger.info(f"Benchmark: CPU={benchmark_results['cpu_gflops']:.1f} GFLOPS, "
                   f"GPU={benchmark_results['gpu_gflops']:.1f} GFLOPS")

    def _benchmark_cpu(self) -> float:
        """Benchmark CPU performance (GFLOPS)"""
        try:
            import numpy as np
            size = 512
            iterations = 10

            # Matrix multiplication benchmark
            start = time.time()
            for _ in range(iterations):
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                c = np.dot(a, b)
            end = time.time()

            # Calculate GFLOPS (2*n^3 operations per matmul)
            ops = iterations * 2 * (size ** 3)
            duration = end - start
            gflops = (ops / duration) / 1e9

            return gflops

        except Exception as e:
            logger.error(f"CPU benchmark error: {e}")
            return 10.0  # Conservative default

    def _benchmark_gpu(self) -> float:
        """Benchmark GPU performance (GFLOPS)"""
        try:
            import torch

            if not torch.cuda.is_available():
                return 0

            device = torch.device('cuda')
            size = 2048
            iterations = 20

            # Warm up
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            torch.matmul(a, b)
            torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            for _ in range(iterations):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.time()

            # Calculate GFLOPS
            ops = iterations * 2 * (size ** 3)
            duration = end - start
            gflops = (ops / duration) / 1e9

            return gflops

        except Exception as e:
            logger.error(f"GPU benchmark error: {e}")
            return 0

    def _benchmark_memory(self) -> float:
        """Benchmark memory bandwidth (GB/s)"""
        try:
            import numpy as np
            size = 100_000_000  # 100M floats = 400MB
            iterations = 5

            start = time.time()
            for _ in range(iterations):
                a = np.random.rand(size).astype(np.float32)
                b = a.copy()
            end = time.time()

            # Calculate bandwidth (2 operations: read + write)
            bytes_transferred = iterations * size * 4 * 2  # 4 bytes per float32
            duration = end - start
            gbps = (bytes_transferred / duration) / 1e9

            return gbps

        except Exception as e:
            logger.error(f"Memory benchmark error: {e}")
            return 10.0

    def calculate_optimal_config(self):
        """Calculate optimal model and training configuration"""
        cpu = self.profile['cpu']
        ram = self.profile['ram']
        gpu = self.profile['gpu']

        config = {
            'model_size': 'medium',
            'use_gpu': gpu['available'],
            'use_mixed_precision': False,
            'batch_size': 64,
            'buffer_size': 100000,
            'num_workers': 0,
            'gradient_accumulation_steps': 1,
            'checkpoint_compression': False,
            'parallel_envs': 1,
            'optimization_level': 'balanced'
        }

        # Determine model size based on available resources
        vram = gpu['vram_gb'] if gpu['available'] else 0
        available_ram = ram['free_for_ml_gb']

        # Model size selection
        if gpu['available'] and vram >= 12 and available_ram >= 16:
            config['model_size'] = 'xlarge'
        elif gpu['available'] and vram >= 8 and available_ram >= 8:
            config['model_size'] = 'large'
        elif gpu['available'] and vram >= 6 and available_ram >= 4:
            config['model_size'] = 'medium'
        elif (gpu['available'] and vram >= 3) or available_ram >= 8:
            config['model_size'] = 'small'
        else:
            config['model_size'] = 'tiny'

        # Mixed precision (if GPU supports it)
        if gpu['available'] and gpu.get('supports_fp16', False):
            config['use_mixed_precision'] = True

        # Batch size based on VRAM/RAM
        if gpu['available']:
            # GPU memory limited
            if vram >= 12:
                config['batch_size'] = 512
            elif vram >= 8:
                config['batch_size'] = 256
            elif vram >= 6:
                config['batch_size'] = 128
            elif vram >= 4:
                config['batch_size'] = 64
            else:
                config['batch_size'] = 32
        else:
            # CPU/RAM limited
            if available_ram >= 16:
                config['batch_size'] = 128
            elif available_ram >= 8:
                config['batch_size'] = 64
            elif available_ram >= 4:
                config['batch_size'] = 32
            else:
                config['batch_size'] = 16

        # Buffer size based on available RAM
        if available_ram >= 32:
            config['buffer_size'] = 500000
        elif available_ram >= 16:
            config['buffer_size'] = 250000
        elif available_ram >= 8:
            config['buffer_size'] = 100000
        elif available_ram >= 4:
            config['buffer_size'] = 50000
        else:
            config['buffer_size'] = 25000

        # Gradient accumulation for low memory
        if (gpu['available'] and vram < 4) or (not gpu['available'] and available_ram < 4):
            config['gradient_accumulation_steps'] = 4
            config['batch_size'] = config['batch_size'] // 4

        # CPU parallelization
        if not gpu['available'] or gpu['tier'] in ['low', 'low_mid']:
            # Leverage CPU when GPU is weak
            config['num_workers'] = max(1, cpu['cores_physical'] - 2)
            config['parallel_envs'] = min(4, cpu['cores_physical'])

        # Storage optimization
        if self.profile['storage']['free_gb'] < 10:
            config['checkpoint_compression'] = True

        # Overall optimization level
        if gpu['tier'] in ['ultra', 'high'] and ram['tier'] in ['ultra', 'high']:
            config['optimization_level'] = 'max_performance'
        elif gpu['tier'] in ['mid', 'mid_high'] or ram['tier'] in ['mid_high', 'high']:
            config['optimization_level'] = 'balanced'
        elif gpu['available'] and gpu['tier'] in ['low', 'low_mid']:
            config['optimization_level'] = 'gpu_conserve'
        else:
            config['optimization_level'] = 'cpu_optimize'

        self.optimal_config = config

        logger.info(f"✅ Optimal config: {config['model_size']} model, "
                   f"batch={config['batch_size']}, buffer={config['buffer_size']}")
        logger.info(f"   Optimization: {config['optimization_level']}, "
                   f"Mixed Precision: {config['use_mixed_precision']}")

        return config

    def get_hardware_summary(self) -> Dict:
        """Get complete hardware summary"""
        return {
            'profile': self.profile,
            'optimal_config': self.optimal_config,
            'timestamp': datetime.now().isoformat()
        }


class AdaptiveReplayBuffer:
    """
    Memory-efficient experience replay buffer with dynamic sizing
    """

    def __init__(self, capacity: int = 100000, compression: bool = False):
        self.capacity = capacity
        self.compression = compression
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity) if capacity < 200000 else None

        logger.info(f"📦 Replay buffer initialized: capacity={capacity}, compression={compression}")

    def push(self, state, action, reward, next_state, done, priority: float = 1.0):
        """Add experience to buffer"""
        if self.compression:
            # Compress states to float16 for memory savings
            state = state.astype(np.float16)
            next_state = next_state.astype(np.float16)

        self.buffer.append((state, action, reward, next_state, done))

        if self.priorities is not None:
            self.priorities.append(priority)

    def sample(self, batch_size: int, prioritized: bool = False) -> List:
        """Sample batch from buffer"""
        if prioritized and self.priorities is not None:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
            return [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear buffer to free memory"""
        self.buffer.clear()
        if self.priorities is not None:
            self.priorities.clear()


class AdaptiveDQNAgent:
    """
    Hardware-adaptive Deep Q-Network agent

    Automatically adapts to hardware with:
    - Dynamic model sizing (tiny to xlarge)
    - Adaptive batch sizes and buffer management
    - CPU parallelization for weak GPUs
    - Mixed precision training when supported
    - Gradient accumulation for low memory
    - Real-time performance monitoring
    """

    def __init__(self, state_size: int, action_size: int,
                 hardware_config: Optional[Dict] = None,
                 custom_config: Optional[Dict] = None):
        """
        Initialize adaptive DQN agent

        Args:
            state_size: Size of state space
            action_size: Size of action space
            hardware_config: Optional pre-detected hardware config
            custom_config: Optional manual configuration overrides
        """
        self.state_size = state_size
        self.action_size = action_size

        # Detect hardware if not provided
        if hardware_config is None:
            profiler = HardwareProfiler()
            self.hardware_profile = profiler.profile
            self.config = profiler.optimal_config
        else:
            self.hardware_profile = hardware_config
            self.config = hardware_config.get('optimal_config', {})

        # Apply custom overrides
        if custom_config:
            self.config.update(custom_config)

        # RL Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Adaptive parameters from hardware config
        self.batch_size = self.config.get('batch_size', 64)
        self.buffer_capacity = self.config.get('buffer_size', 100000)
        self.use_mixed_precision = self.config.get('use_mixed_precision', False)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.num_workers = self.config.get('num_workers', 0)

        # Experience replay
        self.memory = AdaptiveReplayBuffer(
            capacity=self.buffer_capacity,
            compression=self.config.get('checkpoint_compression', False)
        )

        # Device selection
        self.device = self._get_device()

        # Build adaptive model
        self.model = None
        self.target_model = None
        self._build_adaptive_model()

        # Training stats
        self.training_history = []
        self.performance_monitor = {
            'batch_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'loss_values': deque(maxlen=1000)
        }

        logger.info(f"🤖 Adaptive DQN Agent initialized")
        logger.info(f"   Device: {self.device}, Model: {self.config.get('model_size', 'medium')}")
        logger.info(f"   Batch: {self.batch_size}, Buffer: {self.buffer_capacity}")

    def _get_device(self) -> str:
        """Detect and select compute device"""
        if self.config.get('use_gpu', False):
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"

    def _build_adaptive_model(self):
        """Build neural network with adaptive architecture"""
        try:
            import torch
            import torch.nn as nn

            model_size = self.config.get('model_size', 'medium')

            # Architecture definitions
            architectures = {
                'tiny': [64, 32],  # 2-layer, small width
                'small': [128, 64],  # 2-layer, medium width
                'medium': [128, 128, 64],  # 3-layer (original)
                'large': [256, 256, 128, 64],  # 4-layer, wide
                'xlarge': [512, 512, 256, 128, 64]  # 5-layer, very wide
            }

            hidden_layers = architectures.get(model_size, architectures['medium'])

            class AdaptiveDQNNetwork(nn.Module):
                def __init__(self, state_size, action_size, hidden_layers):
                    super(AdaptiveDQNNetwork, self).__init__()

                    layers = []
                    input_size = state_size

                    # Build hidden layers
                    for hidden_size in hidden_layers:
                        layers.append(nn.Linear(input_size, hidden_size))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.1))  # Light dropout for regularization
                        input_size = hidden_size

                    # Output layer
                    layers.append(nn.Linear(input_size, action_size))

                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            self.model = AdaptiveDQNNetwork(self.state_size, self.action_size, hidden_layers)
            self.target_model = AdaptiveDQNNetwork(self.state_size, self.action_size, hidden_layers)

            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
                self.target_model = self.target_model.cuda()

            self.update_target_model()

            # Optimizer with gradient clipping
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()

            # Mixed precision training
            self.scaler = None
            if self.use_mixed_precision and self.device == "cuda":
                try:
                    self.scaler = torch.cuda.amp.GradScaler()
                    logger.info("✅ Mixed precision (FP16) enabled")
                except:
                    logger.warning("Mixed precision not available, using FP32")

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ {model_size.upper()} model built: {len(hidden_layers)} layers, {total_params:,} parameters")

        except ImportError:
            logger.error("PyTorch not available - cannot build model")
            raise
        except Exception as e:
            logger.error(f"Model building error: {e}")
            raise

    def update_target_model(self):
        """Copy weights from model to target model"""
        if self.model is not None and self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        try:
            import torch

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if self.device == "cuda":
                    state_tensor = state_tensor.cuda()

                q_values = self.model(state_tensor)
                action = q_values.argmax().item()

            return action

        except Exception as e:
            logger.error(f"Action selection error: {e}")
            return random.randrange(self.action_size)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, iteration: int = 0):
        """
        Train on batch from replay buffer with adaptive optimizations
        """
        if len(self.memory) < self.batch_size:
            return None

        try:
            import torch

            batch_start = time.time()

            # Sample batch
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            if self.device == "cuda":
                states = states.cuda()
                actions = actions.cuda()
                rewards = rewards.cuda()
                next_states = next_states.cuda()
                dones = dones.cuda()

            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    current_q = self.model(states).gather(1, actions.unsqueeze(1))

                    with torch.no_grad():
                        next_q = self.target_model(next_states).max(1)[0]
                        target_q = rewards + (1 - dones) * self.gamma * next_q

                    loss = self.criterion(current_q.squeeze(), target_q)

                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                if (iteration + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training
                current_q = self.model(states).gather(1, actions.unsqueeze(1))

                with torch.no_grad():
                    next_q = self.target_model(next_states).max(1)[0]
                    target_q = rewards + (1 - dones) * self.gamma * next_q

                loss = self.criterion(current_q.squeeze(), target_q)

                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (iteration + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Performance monitoring
            batch_time = time.time() - batch_start
            self.performance_monitor['batch_times'].append(batch_time)
            self.performance_monitor['loss_values'].append(loss.item())

            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                self.performance_monitor['memory_usage'].append(memory_used)

            return loss.item()

        except Exception as e:
            logger.error(f"Replay error: {e}")
            return None

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_performance_stats(self) -> Dict:
        """Get real-time performance statistics"""
        stats = {
            'avg_batch_time_ms': 0,
            'avg_memory_usage_gb': 0,
            'recent_loss': 0,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory)
        }

        if self.performance_monitor['batch_times']:
            stats['avg_batch_time_ms'] = np.mean(self.performance_monitor['batch_times']) * 1000

        if self.performance_monitor['memory_usage']:
            stats['avg_memory_usage_gb'] = np.mean(self.performance_monitor['memory_usage'])

        if self.performance_monitor['loss_values']:
            stats['recent_loss'] = np.mean(list(self.performance_monitor['loss_values'])[-100:])

        return stats

    def save(self, filepath: str):
        """Save model with optional compression"""
        try:
            import torch

            save_data = {
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_history': self.training_history,
                'hardware_config': self.config,
                'model_size': self.config.get('model_size', 'medium')
            }

            # Compression for limited storage
            if self.config.get('checkpoint_compression', False):
                torch.save(save_data, filepath, _use_new_zipfile_serialization=True)
            else:
                torch.save(save_data, filepath)

            logger.info(f"✅ Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Save error: {e}")

    def load(self, filepath: str):
        """Load model from checkpoint"""
        try:
            import torch

            checkpoint = torch.load(filepath, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.training_history = checkpoint.get('training_history', [])

            logger.info(f"✅ Model loaded from {filepath}")
            logger.info(f"   Model size: {checkpoint.get('model_size', 'unknown')}, Epsilon: {self.epsilon:.3f}")

        except Exception as e:
            logger.error(f"Load error: {e}")


def create_optimized_agent(state_size: int, action_size: int,
                           auto_detect: bool = True,
                           config_override: Optional[Dict] = None) -> AdaptiveDQNAgent:
    """
    Factory function to create hardware-optimized agent

    Args:
        state_size: Size of state space
        action_size: Size of action space
        auto_detect: Whether to auto-detect hardware
        config_override: Optional manual configuration

    Returns:
        Optimized AdaptiveDQNAgent instance
    """
    if auto_detect:
        logger.info("🔍 Auto-detecting hardware configuration...")
        profiler = HardwareProfiler()
        hardware_config = profiler.get_hardware_summary()

        agent = AdaptiveDQNAgent(
            state_size=state_size,
            action_size=action_size,
            hardware_config=hardware_config,
            custom_config=config_override
        )
    else:
        agent = AdaptiveDQNAgent(
            state_size=state_size,
            action_size=action_size,
            custom_config=config_override
        )

    return agent


# Export main classes
__all__ = [
    'HardwareProfiler',
    'AdaptiveReplayBuffer',
    'AdaptiveDQNAgent',
    'create_optimized_agent'
]
