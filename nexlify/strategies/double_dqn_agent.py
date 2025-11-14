#!/usr/bin/env python3
"""
Double DQN and Dueling DQN Agent
Advanced DQN variants for reduced overestimation bias and better value estimation

Double DQN:
- Uses online network to SELECT best action
- Uses target network to EVALUATE that action
- Reduces Q-value overestimation bias

Dueling DQN:
- Separates value V(s) and advantage A(s,a) estimation
- Better learning when actions don't significantly affect outcomes
- More stable value estimates

Can combine both for best results!
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from nexlify.utils.error_handler import get_error_handler
from nexlify.strategies.nexlify_rl_agent import DQNAgent, ReplayBuffer
from nexlify.models.dueling_network import create_network
from nexlify.strategies.epsilon_decay import EpsilonDecayFactory
from nexlify.strategies.gamma_selector import GammaSelector
from nexlify.config.crypto_trading_config import CRYPTO_24_7_CONFIG

# Prioritized Experience Replay
try:
    from nexlify.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
    PER_AVAILABLE = True
except ImportError:
    PER_AVAILABLE = False

# Training optimization utilities
try:
    from nexlify.training.training_optimizers import GradientClipper, LRSchedulerManager
    TRAINING_OPTIMIZERS_AVAILABLE = True
except ImportError:
    TRAINING_OPTIMIZERS_AVAILABLE = False

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent with optional Dueling architecture

    Features:
    - Double DQN: Reduces overestimation by decoupling action selection and evaluation
    - Dueling DQN: Separates value and advantage streams for better learning
    - Configurable: Can toggle both features independently
    - Q-value tracking: Monitors overestimation reduction
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict = None,
        **kwargs
    ):
        """
        Initialize Double DQN agent

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config: Configuration dictionary
            **kwargs: Additional parameters (backward compatibility)

        Config options:
            use_double_dqn (bool): Enable Double DQN (default: True)
            use_dueling_dqn (bool): Enable Dueling architecture (default: True)
            dueling_aggregation (str): 'mean' or 'max' for advantage aggregation
            track_q_values (bool): Track Q-value statistics (default: True)
        """
        # Don't call super().__init__() yet - we need to set flags first
        # Validate inputs
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}")
        if action_size <= 0:
            raise ValueError(f"action_size must be positive, got {action_size}")

        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}
        self.config.update(kwargs)

        # Double DQN and Dueling DQN flags
        self.use_double_dqn = self.config.get("use_double_dqn", True)
        self.use_dueling_dqn = self.config.get("use_dueling_dqn", True)
        self.track_q_values = self.config.get("track_q_values", True)

        # Q-value tracking for overestimation analysis
        self.q_value_history = []
        self.target_q_history = []
        self.online_q_history = []

        # Initialize parent class components
        self._init_parent_components()

        # Build custom models (override parent's models)
        self.model = None
        self.target_model = None
        self.device = self._get_device()
        self._build_model()

        # Training stats
        self.training_history = []
        self._training_step_count = 0

        # Training optimizations
        self.gradient_clipper = None
        self.lr_scheduler = None

        logger.info(
            f"🤖 Double DQN Agent initialized:\n"
            f"   Double DQN: {'✅ Enabled' if self.use_double_dqn else '❌ Disabled'}\n"
            f"   Dueling DQN: {'✅ Enabled' if self.use_dueling_dqn else '❌ Disabled'}\n"
            f"   Device: {self.device}\n"
            f"   Architecture: {self._get_architecture_name()}"
        )

    def _get_architecture_name(self) -> str:
        """Get human-readable architecture name"""
        if self.use_double_dqn and self.use_dueling_dqn:
            return "Double Dueling DQN"
        elif self.use_double_dqn:
            return "Double DQN"
        elif self.use_dueling_dqn:
            return "Dueling DQN"
        else:
            return "Standard DQN"

    def _init_parent_components(self):
        """Initialize components from parent DQNAgent"""
        # Hyperparameters
        self.gamma_selector = self._create_gamma_selector()
        self.gamma = self.gamma_selector.get_gamma()

        self.learning_rate = self.config.get(
            "learning_rate", CRYPTO_24_7_CONFIG.learning_rate
        )
        self.learning_rate_decay = self.config.get(
            "learning_rate_decay", CRYPTO_24_7_CONFIG.learning_rate_decay
        )
        self.batch_size = self.config.get("batch_size", CRYPTO_24_7_CONFIG.batch_size)
        self.target_update_freq = self.config.get(
            "target_update_freq", CRYPTO_24_7_CONFIG.target_update_freq
        )

        # Epsilon decay
        self.epsilon_decay_strategy = self._create_epsilon_strategy()
        self.epsilon = self.epsilon_decay_strategy.current_epsilon

        # Experience replay
        replay_buffer_size = self.config.get(
            "replay_buffer_size", CRYPTO_24_7_CONFIG.replay_buffer_size
        )
        use_per = self.config.get(
            "use_prioritized_replay", CRYPTO_24_7_CONFIG.use_prioritized_replay
        )

        if use_per and PER_AVAILABLE:
            per_alpha = self.config.get("per_alpha", CRYPTO_24_7_CONFIG.per_alpha)
            per_beta_start = self.config.get(
                "per_beta_start", CRYPTO_24_7_CONFIG.per_beta_start
            )
            per_beta_end = self.config.get(
                "per_beta_end", CRYPTO_24_7_CONFIG.per_beta_end
            )
            per_beta_annealing_steps = self.config.get(
                "per_beta_annealing_steps", CRYPTO_24_7_CONFIG.per_beta_annealing_steps
            )
            per_epsilon = self.config.get(
                "per_epsilon", CRYPTO_24_7_CONFIG.per_epsilon
            )
            per_priority_clip = self.config.get(
                "per_priority_clip", CRYPTO_24_7_CONFIG.per_priority_clip
            )

            self.memory = PrioritizedReplayBuffer(
                capacity=replay_buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_end=per_beta_end,
                beta_annealing_steps=per_beta_annealing_steps,
                epsilon=per_epsilon,
                priority_clip=per_priority_clip,
            )
            self.use_per = True
            logger.info(
                f"✨ Using Prioritized Experience Replay (alpha={per_alpha}, beta={per_beta_start}→{per_beta_end})"
            )
        else:
            self.memory = ReplayBuffer(capacity=replay_buffer_size)
            self.use_per = False

    def _build_model(self):
        """Build neural network models using configurable architecture"""
        try:
            # Create networks using factory function
            self.model = create_network(
                state_size=self.state_size,
                action_size=self.action_size,
                use_dueling=self.use_dueling_dqn,
                config=self.config,
            )

            self.target_model = create_network(
                state_size=self.state_size,
                action_size=self.action_size,
                use_dueling=self.use_dueling_dqn,
                config=self.config,
            )

            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
                self.target_model = self.target_model.cuda()

            # Copy weights to target network
            self.update_target_model()

            # Optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
            self.criterion = nn.MSELoss()

            # Initialize training optimizations
            if TRAINING_OPTIMIZERS_AVAILABLE:
                self._init_training_optimizers()

            logger.info(
                f"✅ {self._get_architecture_name()} model built successfully"
            )

        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise

    def replay(self, batch_size: int = None):
        """
        Train on batch from replay buffer using Double DQN

        Double DQN improvement:
        - Standard DQN: target = r + γ * max_a' Q_target(s', a')
        - Double DQN: target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))

        This decouples action selection (online) from evaluation (target),
        reducing overestimation bias.
        """
        batch_size = batch_size or self.batch_size
        if len(self.memory) < batch_size:
            return 0.0

        try:
            # Sample batch
            if self.use_per:
                batch, indices, is_weights = self.memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                is_weights = torch.FloatTensor(is_weights)
                if self.device == "cuda":
                    is_weights = is_weights.cuda()
            else:
                batch = self.memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                indices = None
                is_weights = None

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

            # Current Q values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

            # Compute target Q values
            with torch.no_grad():
                if self.use_double_dqn:
                    # DOUBLE DQN: Use online network to select action
                    online_next_q = self.model(next_states)
                    best_actions = online_next_q.argmax(1)  # argmax_a' Q_online(s', a')

                    # Use target network to evaluate selected action
                    target_next_q = self.target_model(next_states)
                    next_q_values = target_next_q.gather(
                        1, best_actions.unsqueeze(1)
                    ).squeeze(1)

                    # Track Q-values for analysis
                    if self.track_q_values:
                        self.online_q_history.append(online_next_q.max(1)[0].mean().item())
                        self.target_q_history.append(next_q_values.mean().item())
                else:
                    # STANDARD DQN: Use target network for both selection and evaluation
                    next_q_values = self.target_model(next_states).max(1)[0]

                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Track overall Q-values
            if self.track_q_values and len(self.q_value_history) < 10000:
                self.q_value_history.append(current_q_values.mean().item())

            # Compute TD errors (for PER)
            current_q = current_q_values.squeeze(1)
            td_errors = (target_q_values - current_q).detach()

            # Compute loss
            if self.use_per and is_weights is not None:
                element_wise_loss = torch.nn.functional.mse_loss(
                    current_q, target_q_values, reduction="none"
                )
                loss = (is_weights * element_wise_loss).mean()
            else:
                loss = self.criterion(current_q, target_q_values)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            self._training_step_count += 1
            gradient_norm_before = 0.0
            gradient_norm_after = 0.0
            gradient_clipped = False

            if self.gradient_clipper is not None:
                log_step = self._training_step_count % 100 == 0
                gradient_norm_before, gradient_norm_after, gradient_clipped = (
                    self.gradient_clipper.clip_gradients(self.model, log_step=log_step)
                )

            # Optimizer step
            self.optimizer.step()

            # Update PER priorities
            if self.use_per and indices is not None:
                td_errors_np = (
                    td_errors.cpu().numpy()
                    if self.device == "cuda"
                    else td_errors.numpy()
                )
                self.memory.update_priorities(indices, td_errors_np)

            # LR scheduling
            current_lr = self.learning_rate
            if self.lr_scheduler is not None:
                log_step = self._training_step_count % 100 == 0
                self.lr_scheduler.step(loss=loss.item(), log_step=log_step)
                current_lr = self.lr_scheduler.get_current_lr()

            # Track training metrics
            if hasattr(self, "training_history"):
                training_info = {
                    "step": self._training_step_count,
                    "loss": loss.item(),
                    "learning_rate": current_lr,
                    "mean_q_value": current_q_values.mean().item(),
                }

                if self.gradient_clipper is not None:
                    training_info.update({
                        "gradient_norm_before": gradient_norm_before,
                        "gradient_norm_after": gradient_norm_after,
                        "gradient_clipped": gradient_clipped,
                    })

                if self.use_per:
                    per_stats = self.memory.get_stats()
                    training_info.update({
                        "per_beta": per_stats["beta"],
                        "per_mean_priority": per_stats["mean_priority"],
                    })

                if self._training_step_count % 10 == 0:
                    self.training_history.append(training_info)

            return loss.item()

        except Exception as e:
            logger.error(f"Replay error: {e}")
            return 0.0

    def get_q_value_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get Q-value statistics to analyze overestimation reduction

        Returns:
            Dictionary with Q-value statistics if tracking enabled
        """
        if not self.track_q_values or not self.q_value_history:
            return None

        stats = {
            "mean_q_value": np.mean(self.q_value_history[-1000:]),
            "std_q_value": np.std(self.q_value_history[-1000:]),
            "total_samples": len(self.q_value_history),
        }

        if self.use_double_dqn and self.online_q_history and self.target_q_history:
            # Measure overestimation: difference between online and target Q-values
            online_q = np.array(self.online_q_history[-1000:])
            target_q = np.array(self.target_q_history[-1000:])
            overestimation = online_q - target_q

            stats.update({
                "mean_online_q": np.mean(online_q),
                "mean_target_q": np.mean(target_q),
                "mean_overestimation": np.mean(overestimation),
                "overestimation_reduction": (
                    np.mean(overestimation) / (np.mean(online_q) + 1e-8) * 100
                ),
            })

        return stats

    def get_model_summary(self) -> str:
        """Get detailed model summary including architecture info"""
        try:
            if self.model is None:
                return "Model not initialized"

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            gamma_info = self.get_gamma_info()
            summary = (
                f"{self._get_architecture_name()} Model Summary\n"
                f"{'='*60}\n"
                f"Architecture: {self._get_architecture_name()}\n"
                f"State Size: {self.state_size}\n"
                f"Action Size: {self.action_size}\n"
                f"Total Parameters: {total_params:,}\n"
                f"Trainable Parameters: {trainable_params:,}\n"
                f"Device: {self.device}\n"
                f"\nHyperparameters:\n"
                f"  Epsilon: {self.epsilon:.4f}\n"
                f"  Learning Rate: {self.learning_rate}\n"
                f"  Gamma: {self.gamma:.3f} ({gamma_info['style']}, {gamma_info['timeframe']})\n"
                f"  Batch Size: {self.batch_size}\n"
            )

            # Add PER info
            if self.use_per:
                per_stats = self.get_per_stats()
                summary += (
                    f"\nReplay Buffer: Prioritized (PER)\n"
                    f"  Size: {per_stats['size']}/{per_stats['capacity']}\n"
                    f"  Alpha: {per_stats['alpha']:.2f}\n"
                    f"  Beta: {per_stats['beta']:.3f}\n"
                )
            else:
                summary += f"\nReplay Buffer: Standard (size: {len(self.memory)})\n"

            # Add Q-value stats
            if self.track_q_values:
                q_stats = self.get_q_value_stats()
                if q_stats:
                    summary += f"\nQ-Value Statistics (last 1000 steps):\n"
                    summary += f"  Mean Q: {q_stats['mean_q_value']:.4f}\n"
                    summary += f"  Std Q: {q_stats['std_q_value']:.4f}\n"

                    if self.use_double_dqn and "mean_overestimation" in q_stats:
                        summary += (
                            f"\nDouble DQN Overestimation Analysis:\n"
                            f"  Online Q: {q_stats['mean_online_q']:.4f}\n"
                            f"  Target Q: {q_stats['mean_target_q']:.4f}\n"
                            f"  Overestimation: {q_stats['mean_overestimation']:.4f}\n"
                            f"  Reduction: {q_stats['overestimation_reduction']:.2f}%\n"
                        )

            return summary

        except Exception as e:
            logger.error(f"Model summary error: {e}")
            return f"Model summary error: {str(e)}"


# Export
__all__ = ["DoubleDQNAgent"]
