#!/usr/bin/env python3
"""
Nexlify Fully Dynamic RL Agent
NO FIXED TIERS - Pure adaptive optimization with intelligent bottleneck offloading

This agent:
- Continuously monitors system resources in real-time
- Detects bottlenecks (CPU, GPU, RAM, VRAM)
- Dynamically adjusts architecture during training
- Intelligently offloads work to underutilized components
- Self-optimizes without manual configuration

Example: GTX 1050 (2GB VRAM) + Threadripper (32 cores) + 64GB RAM
- Detects: VRAM bottleneck, CPU overhead, RAM overhead
- Response: Small GPU model, 16 CPU workers for preprocessing, large replay buffer
- Result: Maximizes throughput by using all available resources
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random
from datetime import datetime
import time

from nexlify.ml.nexlify_dynamic_architecture import (
    DynamicResourceMonitor,
    DynamicArchitectureBuilder,
    DynamicWorkloadDistributor,
    DynamicBufferManager,
    Bottleneck
)

logger = logging.getLogger(__name__)


class FullyDynamicDQNAgent:
    """
    Fully dynamic DQN agent with NO fixed tiers

    Architecture, batch sizes, buffer sizes, and workload distribution
    are continuously adjusted based on real-time resource monitoring
    """

    def __init__(self, state_size: int, action_size: int,
                 auto_optimize: bool = True,
                 min_params: int = 1000,
                 max_params: int = 2000000):
        """
        Initialize fully dynamic agent

        Args:
            state_size: Size of state space
            action_size: Size of action space
            auto_optimize: Enable continuous auto-optimization
            min_params: Minimum model parameters
            max_params: Maximum model parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.auto_optimize = auto_optimize
        self.min_params = min_params
        self.max_params = max_params

        # Dynamic resource monitoring
        self.monitor = DynamicResourceMonitor(sample_interval=0.1)

        # Start monitoring in background
        if auto_optimize:
            self.monitor.start_monitoring()

        # Dynamic components
        self.arch_builder = DynamicArchitectureBuilder(self.monitor)
        self.workload_dist = DynamicWorkloadDistributor(self.monitor)
        self.buffer = DynamicBufferManager(
            self.monitor,
            initial_capacity=50000,
            min_capacity=10000,
            max_capacity=1000000
        )

        # RL hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Current configuration (dynamic)
        self.current_config = {
            'architecture': None,
            'batch_size': 64,
            'device': 'cpu',
            'gpu_batch_size': 64,
            'cpu_workers': 0,
            'pin_memory': True
        }

        # Device selection
        self.device = self._select_device()

        # Build initial model
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.criterion = None

        self._build_dynamic_model()

        # Training statistics
        self.training_stats = {
            'architecture_changes': 0,
            'bottleneck_history': deque(maxlen=100),
            'batch_times': deque(maxlen=100),
            'optimization_history': []
        }

        # Optimization thread (optional)
        self.last_optimization = time.time()
        self.optimization_interval = 30  # Reoptimize every 30 seconds

        logger.info(f"🚀 Fully Dynamic DQN Agent initialized")
        logger.info(f"   Auto-optimization: {auto_optimize}")
        logger.info(f"   Initial device: {self.device}")

    def _select_device(self) -> str:
        """Dynamically select best device"""
        try:
            import torch

            if torch.cuda.is_available():
                # Check if GPU has enough memory
                snapshot = self.monitor.take_snapshot()

                if snapshot.gpu_memory_percent < 80:
                    return "cuda"
                else:
                    logger.warning(f"⚠️  GPU VRAM usage high ({snapshot.gpu_memory_percent:.1f}%), using CPU")
                    return "cpu"
        except ImportError:
            pass

        return "cpu"

    def _build_dynamic_model(self):
        """Build model with dynamically determined architecture"""
        try:
            import torch
            import torch.nn as nn

            # Get current resource state
            snapshot = self.monitor.take_snapshot()

            # Dynamically determine architecture
            architecture = self.arch_builder.build_adaptive_architecture(
                input_size=self.state_size,
                output_size=self.action_size,
                min_params=self.min_params,
                max_params=self.max_params
            )

            # Store configuration
            self.current_config['architecture'] = architecture

            # Build network
            class DynamicDQNNetwork(nn.Module):
                def __init__(self, input_size, hidden_layers, output_size):
                    super(DynamicDQNNetwork, self).__init__()

                    layers = []
                    prev_size = input_size

                    # Build hidden layers
                    for hidden_size in hidden_layers:
                        layers.append(nn.Linear(prev_size, hidden_size))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.1))
                        prev_size = hidden_size

                    # Output layer
                    layers.append(nn.Linear(prev_size, output_size))

                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            # Create models
            self.model = DynamicDQNNetwork(self.state_size, architecture, self.action_size)
            self.target_model = DynamicDQNNetwork(self.state_size, architecture, self.action_size)

            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
                self.target_model = self.target_model.cuda()

            # Update target
            self.update_target_model()

            # Optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss()

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())

            logger.info(f"✅ Dynamic model built:")
            logger.info(f"   Architecture: {architecture}")
            logger.info(f"   Parameters: {total_params:,}")
            logger.info(f"   Bottleneck: {snapshot.bottleneck.value}")
            logger.info(f"   Overhead: CPU={snapshot.overhead_capacity['cpu']:.1f}%, "
                       f"RAM={snapshot.overhead_capacity['ram']:.1f}%, "
                       f"GPU={snapshot.overhead_capacity['gpu']:.1f}%")

        except ImportError:
            logger.error("PyTorch not available")
            raise

    def maybe_reoptimize(self):
        """
        Check if reoptimization is needed based on bottlenecks

        Reoptimizes if:
        - Bottleneck has changed
        - Performance degraded
        - Enough time has passed
        """
        if not self.auto_optimize:
            return

        current_time = time.time()

        # Don't reoptimize too frequently
        if current_time - self.last_optimization < self.optimization_interval:
            return

        # Check if bottleneck has changed
        current_bottleneck = self.monitor.get_current_bottleneck()

        if self.training_stats['bottleneck_history']:
            recent_bottlenecks = list(self.training_stats['bottleneck_history'])[-10:]
            prev_bottleneck = max(set(recent_bottlenecks), key=recent_bottlenecks.count)

            # Bottleneck changed - reoptimize
            if current_bottleneck != prev_bottleneck and current_bottleneck != Bottleneck.NONE:
                logger.info(f"🔄 Bottleneck changed: {prev_bottleneck.value} → {current_bottleneck.value}")
                self._reoptimize_configuration()
                self.last_optimization = current_time

        # Record bottleneck
        self.training_stats['bottleneck_history'].append(current_bottleneck)

    def _reoptimize_configuration(self):
        """Reoptimize entire configuration based on current bottlenecks"""
        logger.info("⚙️  Reoptimizing configuration...")

        snapshot = self.monitor.take_snapshot()
        bottleneck = snapshot.bottleneck

        # Rebuild architecture if needed
        current_arch = self.current_config['architecture']
        new_arch = self.arch_builder.build_adaptive_architecture(
            input_size=self.state_size,
            output_size=self.action_size,
            min_params=self.min_params,
            max_params=self.max_params
        )

        if new_arch != current_arch:
            logger.info(f"🏗️  Architecture change: {current_arch} → {new_arch}")
            self._rebuild_model(new_arch)
            self.training_stats['architecture_changes'] += 1

        # Reoptimize workload distribution
        workload_config = self.workload_dist.optimize_distribution(
            self.current_config['batch_size']
        )

        self.current_config.update(workload_config)

        # Optimization complete
        self.training_stats['optimization_history'].append({
            'timestamp': time.time(),
            'bottleneck': bottleneck.value,
            'architecture': new_arch,
            'config': workload_config.copy()
        })

        logger.info(f"✅ Reoptimization complete")
        logger.info(f"   CPU workers: {self.current_config['cpu_workers']}")
        logger.info(f"   GPU batch: {self.current_config['gpu_batch_size']}")
        logger.info(f"   Buffer capacity: {len(self.buffer.buffer):,}/{self.buffer.capacity:,}")

    def _rebuild_model(self, new_architecture: List[int]):
        """Rebuild model with new architecture (preserves training state)"""
        try:
            import torch
            import torch.nn as nn

            # Save current model state
            old_state = None
            if self.model is not None:
                try:
                    old_state = self.model.state_dict()
                except:
                    pass

            # Build new model
            class DynamicDQNNetwork(nn.Module):
                def __init__(self, input_size, hidden_layers, output_size):
                    super(DynamicDQNNetwork, self).__init__()

                    layers = []
                    prev_size = input_size

                    for hidden_size in hidden_layers:
                        layers.append(nn.Linear(prev_size, hidden_size))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.1))
                        prev_size = hidden_size

                    layers.append(nn.Linear(prev_size, output_size))
                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            # Create new models
            new_model = DynamicDQNNetwork(self.state_size, new_architecture, self.action_size)
            new_target = DynamicDQNNetwork(self.state_size, new_architecture, self.action_size)

            # Move to device
            if self.device == "cuda":
                new_model = new_model.cuda()
                new_target = new_target.cuda()

            # Try to transfer weights from compatible layers
            if old_state is not None:
                try:
                    # Partial weight transfer (best effort)
                    new_state = new_model.state_dict()
                    for key in old_state.keys():
                        if key in new_state and old_state[key].shape == new_state[key].shape:
                            new_state[key] = old_state[key]
                    new_model.load_state_dict(new_state)
                    logger.info("   Transferred compatible weights")
                except Exception as e:
                    logger.warning(f"   Could not transfer weights: {e}")

            # Replace models
            self.model = new_model
            self.target_model = new_target
            self.update_target_model()

            # Update optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Update config
            self.current_config['architecture'] = new_architecture

        except Exception as e:
            logger.error(f"Model rebuild failed: {e}")

    def update_target_model(self):
        """Copy weights from model to target model"""
        if self.model is not None and self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        # Maybe reoptimize before acting
        if training:
            self.maybe_reoptimize()

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
        """Store experience in dynamic buffer"""
        self.buffer.push(state, action, reward, next_state, done)

    def replay(self, iteration: int = 0):
        """Train with dynamically optimized batch processing"""
        # Get dynamic batch size
        batch_size = self.current_config['batch_size']

        if len(self.buffer) < batch_size:
            return None

        try:
            import torch

            batch_start = time.time()

            # Sample batch
            experiences = self.buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)

            # Convert to tensors
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            # Dynamic device placement
            if self.device == "cuda":
                states = states.cuda()
                actions = actions.cuda()
                rewards = rewards.cuda()
                next_states = next_states.cuda()
                dones = dones.cuda()

            # Forward pass
            current_q = self.model(states).gather(1, actions.unsqueeze(1))

            with torch.no_grad():
                next_q = self.target_model(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q

            # Loss
            loss = self.criterion(current_q.squeeze(), target_q)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track batch time
            batch_time = time.time() - batch_start
            self.training_stats['batch_times'].append(batch_time)

            return loss.item()

        except Exception as e:
            logger.error(f"Replay error: {e}")
            return None

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        snapshot = self.monitor.take_snapshot()
        avg_usage = self.monitor.get_average_usage(window=20)

        stats = {
            'current_bottleneck': snapshot.bottleneck.value,
            'architecture': self.current_config['architecture'],
            'model_params': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'batch_size': self.current_config['batch_size'],
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.buffer.capacity,
            'cpu_workers': self.current_config['cpu_workers'],
            'device': self.device,
            'epsilon': self.epsilon,
            'architecture_changes': self.training_stats['architecture_changes'],
            'avg_batch_time_ms': np.mean(self.training_stats['batch_times']) * 1000 if self.training_stats['batch_times'] else 0,
            'resource_usage': {
                'cpu': avg_usage['cpu'],
                'ram': avg_usage['ram'],
                'gpu': avg_usage['gpu'],
                'vram': avg_usage['vram']
            },
            'overhead_capacity': snapshot.overhead_capacity
        }

        return stats

    def save(self, filepath: str):
        """Save model and configuration"""
        try:
            import torch

            save_data = {
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'current_config': self.current_config,
                'training_stats': {
                    'architecture_changes': self.training_stats['architecture_changes'],
                    'optimization_history': self.training_stats['optimization_history']
                }
            }

            torch.save(save_data, filepath)
            logger.info(f"✅ Dynamic model saved to {filepath}")

        except Exception as e:
            logger.error(f"Save error: {e}")

    def load(self, filepath: str):
        """Load model and configuration"""
        try:
            import torch

            checkpoint = torch.load(filepath, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.current_config = checkpoint.get('current_config', self.current_config)

            if 'training_stats' in checkpoint:
                self.training_stats.update(checkpoint['training_stats'])

            logger.info(f"✅ Dynamic model loaded from {filepath}")
            logger.info(f"   Architecture: {self.current_config['architecture']}")

        except Exception as e:
            logger.error(f"Load error: {e}")

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()


def create_fully_dynamic_agent(state_size: int, action_size: int,
                               auto_optimize: bool = True) -> FullyDynamicDQNAgent:
    """
    Factory function to create fully dynamic agent

    Args:
        state_size: Size of state space
        action_size: Size of action space
        auto_optimize: Enable continuous auto-optimization

    Returns:
        FullyDynamicDQNAgent instance
    """
    logger.info("🚀 Creating fully dynamic agent...")

    agent = FullyDynamicDQNAgent(
        state_size=state_size,
        action_size=action_size,
        auto_optimize=auto_optimize
    )

    return agent


# Export
__all__ = ['FullyDynamicDQNAgent', 'create_fully_dynamic_agent']
