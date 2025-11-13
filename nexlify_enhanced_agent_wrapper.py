#!/usr/bin/env python3
"""
Enhanced Agent Wrapper with ML Best Practices

Adds critical improvements to the base agent:
✅ Gradient clipping (prevents exploding gradients)
✅ Learning rate scheduling (better convergence)
✅ L2 regularization (prevents overfitting)
✅ Early stopping support
✅ Training metrics tracking

This wrapper can be used with any DQN agent (base, adaptive, ultra-optimized, etc.)
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from typing import Optional, Dict, Any, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EnhancedAgentWrapper:
    """
    Wraps any DQN agent to add ML best practices

    Features:
    - Gradient clipping: Prevents exploding gradients
    - LR scheduling: ReduceLROnPlateau or CosineAnnealing
    - L2 regularization: Weight decay in optimizer
    - Early stopping: Tracks validation performance
    - Metrics tracking: Loss, gradients, LR history
    """

    def __init__(
        self,
        base_agent,
        gradient_clip_norm: float = 1.0,
        lr_scheduler_type: str = 'plateau',  # 'plateau', 'cosine', or 'none'
        lr_scheduler_patience: int = 5,
        lr_scheduler_factor: float = 0.5,
        lr_min: float = 1e-6,
        weight_decay: float = 1e-5,  # L2 regularization
        early_stop_patience: int = 10,
        early_stop_threshold: float = 0.01,  # 1% degradation threshold
        track_metrics: bool = True
    ):
        """
        Initialize enhanced agent wrapper

        Args:
            base_agent: Base DQN agent to wrap
            gradient_clip_norm: Maximum gradient norm (default: 1.0)
            lr_scheduler_type: 'plateau', 'cosine', or 'none'
            lr_scheduler_patience: Patience for ReduceLROnPlateau
            lr_scheduler_factor: Factor for LR reduction
            lr_min: Minimum learning rate
            weight_decay: L2 regularization strength
            early_stop_patience: Patience for early stopping
            early_stop_threshold: Threshold for performance degradation (0.01 = 1%)
            track_metrics: Whether to track training metrics
        """
        self.agent = base_agent
        self.gradient_clip_norm = gradient_clip_norm
        self.lr_scheduler_type = lr_scheduler_type
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        self.track_metrics = track_metrics

        # Add L2 regularization if not already present
        if hasattr(self.agent, 'optimizer_nn'):
            # Update weight decay
            for param_group in self.agent.optimizer_nn.param_groups:
                param_group['weight_decay'] = weight_decay
            logger.info(f"✅ L2 regularization: {weight_decay}")

        # Create learning rate scheduler
        self.lr_scheduler = None
        if lr_scheduler_type != 'none' and hasattr(self.agent, 'optimizer_nn'):
            if lr_scheduler_type == 'plateau':
                self.lr_scheduler = ReduceLROnPlateau(
                    self.agent.optimizer_nn,
                    mode='max',  # Maximize validation score
                    factor=lr_scheduler_factor,
                    patience=lr_scheduler_patience,
                    min_lr=lr_min,
                    verbose=True
                )
                logger.info(f"✅ LR Scheduler: ReduceLROnPlateau (patience={lr_scheduler_patience}, factor={lr_scheduler_factor})")

            elif lr_scheduler_type == 'cosine':
                self.lr_scheduler = CosineAnnealingWarmRestarts(
                    self.agent.optimizer_nn,
                    T_0=10,  # Initial restart period
                    T_mult=2,  # Period multiplier
                    eta_min=lr_min
                )
                logger.info(f"✅ LR Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")

        # Early stopping tracking
        self.best_val_score = float('-inf')
        self.val_score_history = []
        self.no_improvement_count = 0
        self.should_stop = False

        # Metrics tracking
        if track_metrics:
            self.metrics = {
                'loss_history': [],
                'grad_norm_history': [],
                'lr_history': [],
                'val_score_history': []
            }
        else:
            self.metrics = None

        logger.info(f"✅ Gradient clipping: {gradient_clip_norm}")
        logger.info(f"✅ Early stopping: patience={early_stop_patience}, threshold={early_stop_threshold*100:.1f}%")

    def replay(self):
        """
        Enhanced replay with gradient clipping and metrics tracking
        """
        # Call base agent's replay
        loss = self.agent.replay()

        # Apply gradient clipping
        if hasattr(self.agent, 'model') and self.gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.model.parameters(),
                self.gradient_clip_norm
            )

            # Track metrics
            if self.track_metrics and self.metrics is not None:
                self.metrics['loss_history'].append(float(loss) if isinstance(loss, (int, float, torch.Tensor)) else 0.0)
                self.metrics['grad_norm_history'].append(float(grad_norm))

                if hasattr(self.agent, 'optimizer_nn'):
                    current_lr = self.agent.optimizer_nn.param_groups[0]['lr']
                    self.metrics['lr_history'].append(current_lr)

        # Update cosine scheduler (step-based)
        if self.lr_scheduler and self.lr_scheduler_type == 'cosine':
            self.lr_scheduler.step()

        return loss

    def update_validation_score(self, val_score: float) -> bool:
        """
        Update validation score and check early stopping

        Args:
            val_score: Current validation score (higher is better)

        Returns:
            True if training should stop, False otherwise
        """
        self.val_score_history.append(val_score)

        # Track metrics
        if self.track_metrics and self.metrics is not None:
            self.metrics['val_score_history'].append(val_score)

        # Update plateau scheduler (epoch-based)
        if self.lr_scheduler and self.lr_scheduler_type == 'plateau':
            self.lr_scheduler.step(val_score)

        # Check for improvement
        if val_score > self.best_val_score:
            improvement_pct = ((val_score - self.best_val_score) / abs(self.best_val_score)) * 100 if self.best_val_score != float('-inf') else 100
            self.best_val_score = val_score
            self.no_improvement_count = 0
            logger.info(f"🏆 Validation improved: {val_score:.2f} (+{improvement_pct:.2f}%)")
            return False

        # Check for degradation
        degradation = (self.best_val_score - val_score) / abs(self.best_val_score)

        if degradation > self.early_stop_threshold:
            self.no_improvement_count += 1
            logger.info(f"⚠️  Validation degraded: {val_score:.2f} (best: {self.best_val_score:.2f}, patience: {self.no_improvement_count}/{self.early_stop_patience})")

            if self.no_improvement_count >= self.early_stop_patience:
                logger.info(f"🛑 EARLY STOPPING: No improvement for {self.early_stop_patience} validations")
                self.should_stop = True
                return True
        else:
            # Small degradation is okay, just track it
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.early_stop_patience:
                logger.info(f"🛑 EARLY STOPPING: No significant improvement for {self.early_stop_patience} validations")
                self.should_stop = True
                return True

        return False

    def get_current_lr(self) -> float:
        """Get current learning rate"""
        if hasattr(self.agent, 'optimizer_nn'):
            return self.agent.optimizer_nn.param_groups[0]['lr']
        return 0.0

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics"""
        if not self.track_metrics or self.metrics is None:
            return {}

        summary = {
            'avg_loss': np.mean(self.metrics['loss_history'][-100:]) if self.metrics['loss_history'] else 0.0,
            'avg_grad_norm': np.mean(self.metrics['grad_norm_history'][-100:]) if self.metrics['grad_norm_history'] else 0.0,
            'current_lr': self.get_current_lr(),
            'best_val_score': self.best_val_score,
            'val_score_trend': 'improving' if len(self.val_score_history) >= 2 and self.val_score_history[-1] > self.val_score_history[-2] else 'declining'
        }

        return summary

    def reset_early_stopping(self):
        """Reset early stopping counters (useful between retraining iterations)"""
        self.no_improvement_count = 0
        self.should_stop = False
        logger.info("✅ Early stopping counters reset")

    # Forward all other attributes to base agent
    def __getattr__(self, name):
        """Forward attribute access to base agent"""
        return getattr(self.agent, name)

    def act(self, state, training: bool = True):
        """Forward to base agent"""
        return self.agent.act(state, training)

    def remember(self, state, action, reward, next_state, done):
        """Forward to base agent"""
        return self.agent.remember(state, action, reward, next_state, done)


class EnsembleAgent:
    """
    Ensemble of multiple agents for more robust predictions

    Uses majority voting for discrete actions or averaging for Q-values
    """

    def __init__(self, agents: List[Any], voting_method: str = 'average'):
        """
        Initialize ensemble

        Args:
            agents: List of trained agents
            voting_method: 'average' (average Q-values) or 'majority' (majority vote)
        """
        self.agents = agents
        self.voting_method = voting_method
        self.num_agents = len(agents)

        logger.info(f"✅ Ensemble created with {self.num_agents} agents (method: {voting_method})")

    def act(self, state, training: bool = False):
        """
        Get action from ensemble

        Args:
            state: Current state
            training: If True, average epsilon-greedy; if False, greedy ensemble

        Returns:
            Action chosen by ensemble
        """
        if self.voting_method == 'average':
            # Average Q-values from all agents
            import torch

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                # Get Q-values from each agent
                q_values_list = []
                for agent in self.agents:
                    if hasattr(agent, 'model'):
                        if hasattr(agent, 'device'):
                            state_tensor = state_tensor.to(agent.device)
                        q_vals = agent.model(state_tensor)
                        q_values_list.append(q_vals.cpu().numpy())

                # Average Q-values
                if q_values_list:
                    avg_q_values = np.mean(q_values_list, axis=0)
                    action = np.argmax(avg_q_values)
                else:
                    # Fallback to first agent
                    action = self.agents[0].act(state, training=False)

            return action

        else:  # majority voting
            # Get action from each agent
            actions = [agent.act(state, training=False) for agent in self.agents]

            # Majority vote
            action_counts = {}
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1

            # Return most common action
            majority_action = max(action_counts, key=action_counts.get)
            return majority_action

    def get_ensemble_confidence(self, state) -> float:
        """
        Get confidence of ensemble prediction (agreement between agents)

        Returns:
            Confidence score (0-1, higher = more agreement)
        """
        actions = [agent.act(state, training=False) for agent in self.agents]

        # Calculate agreement
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        max_votes = max(action_counts.values())
        confidence = max_votes / self.num_agents

        return confidence


def create_enhanced_agent(base_agent_class, state_size: int, action_size: int,
                          use_enhancements: bool = True, **kwargs) -> Any:
    """
    Factory function to create an enhanced agent

    Args:
        base_agent_class: Base agent class (e.g., UltraOptimizedDQNAgent)
        state_size: State space size
        action_size: Action space size
        use_enhancements: Whether to wrap with enhancements
        **kwargs: Additional arguments for base agent or wrapper

    Returns:
        Enhanced or base agent
    """
    # Separate wrapper kwargs from base agent kwargs
    wrapper_kwargs = {
        'gradient_clip_norm': kwargs.pop('gradient_clip_norm', 1.0),
        'lr_scheduler_type': kwargs.pop('lr_scheduler_type', 'plateau'),
        'lr_scheduler_patience': kwargs.pop('lr_scheduler_patience', 5),
        'lr_scheduler_factor': kwargs.pop('lr_scheduler_factor', 0.5),
        'lr_min': kwargs.pop('lr_min', 1e-6),
        'weight_decay': kwargs.pop('weight_decay', 1e-5),
        'early_stop_patience': kwargs.pop('early_stop_patience', 10),
        'early_stop_threshold': kwargs.pop('early_stop_threshold', 0.01),
        'track_metrics': kwargs.pop('track_metrics', True)
    }

    # Create base agent
    base_agent = base_agent_class(state_size=state_size, action_size=action_size, **kwargs)

    # Wrap with enhancements if requested
    if use_enhancements:
        enhanced_agent = EnhancedAgentWrapper(base_agent, **wrapper_kwargs)
        logger.info("✅ Agent created with ML best practices enhancements")
        return enhanced_agent
    else:
        logger.info("✅ Agent created without enhancements")
        return base_agent
