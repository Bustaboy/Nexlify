#!/usr/bin/env python3
"""
Nexlify Training Optimization Utilities

Comprehensive training optimization including:
- Gradient clipping with monitoring
- Learning rate scheduling (Cosine, Plateau, Step)
- LR warmup
- Training stability tracking

Prevents common training issues:
- Gradient explosion (NaN losses)
- Gradient vanishing
- Learning rate instability
- Sudden loss spikes
"""

import json
import logging
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)

from nexlify.utils.error_handler import get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class SchedulerType(Enum):
    """Available LR scheduler types"""
    COSINE = "cosine"
    PLATEAU = "plateau"
    STEP = "step"
    AUTO = "auto"


class TrainingPhase(Enum):
    """Training phases for auto scheduler selection"""
    EXPLORATION = "exploration"  # Early training, high exploration
    EXPLOITATION = "exploitation"  # Later training, exploit learned patterns
    REFINEMENT = "refinement"  # Fine-tuning phase


class GradientClipper:
    """
    Gradient clipping with monitoring and statistics

    Features:
    - Configurable max_norm for clipping
    - Track gradient norms before/after clipping
    - Alert on gradient explosion/vanishing
    - Save gradient history for analysis
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        explosion_threshold: float = 10.0,
        vanishing_threshold: float = 1e-6,
        history_size: int = 1000,
        alert_frequency: int = 100,
    ):
        """
        Initialize gradient clipper

        Args:
            max_norm: Maximum gradient norm for clipping
            norm_type: Type of norm (2.0 = L2 norm)
            explosion_threshold: Threshold for gradient explosion alert
            vanishing_threshold: Threshold for gradient vanishing alert
            history_size: Number of gradient norms to track
            alert_frequency: How often to check for gradient issues
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.alert_frequency = alert_frequency

        # Tracking
        self.gradient_norms_before = deque(maxlen=history_size)
        self.gradient_norms_after = deque(maxlen=history_size)
        self.clip_count = 0
        self.explosion_count = 0
        self.vanishing_count = 0
        self.step_count = 0

        logger.info(
            f"🔧 GradientClipper initialized (max_norm={max_norm}, "
            f"explosion_threshold={explosion_threshold}, "
            f"vanishing_threshold={vanishing_threshold})"
        )

    def clip_gradients(
        self,
        model: nn.Module,
        log_step: bool = False,
    ) -> Tuple[float, float, bool]:
        """
        Clip gradients and track statistics

        Args:
            model: PyTorch model with gradients
            log_step: Whether to log this step's gradient info

        Returns:
            Tuple of (norm_before, norm_after, was_clipped)
        """
        # Calculate norm before clipping
        parameters = [p for p in model.parameters() if p.grad is not None]

        if not parameters:
            return 0.0, 0.0, False

        # Calculate total gradient norm
        total_norm_before = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), self.norm_type) for p in parameters]),
            self.norm_type
        ).item()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )

        # Calculate norm after clipping
        total_norm_after = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), self.norm_type) for p in parameters]),
            self.norm_type
        ).item()

        # Track statistics
        self.gradient_norms_before.append(total_norm_before)
        self.gradient_norms_after.append(total_norm_after)
        self.step_count += 1

        was_clipped = total_norm_before > self.max_norm
        if was_clipped:
            self.clip_count += 1

        # Check for gradient issues
        if total_norm_before > self.explosion_threshold:
            self.explosion_count += 1
            logger.warning(
                f"⚠️  GRADIENT EXPLOSION detected! "
                f"Norm: {total_norm_before:.4f} > {self.explosion_threshold}"
            )

        if total_norm_before < self.vanishing_threshold and total_norm_before > 0:
            self.vanishing_count += 1
            if self.vanishing_count % 10 == 0:  # Don't spam warnings
                logger.warning(
                    f"⚠️  GRADIENT VANISHING detected! "
                    f"Norm: {total_norm_before:.2e} < {self.vanishing_threshold}"
                )

        # Periodic logging
        if log_step or (self.step_count % self.alert_frequency == 0):
            self._log_statistics()

        return total_norm_before, total_norm_after, was_clipped

    def _log_statistics(self):
        """Log gradient statistics"""
        if not self.gradient_norms_before:
            return

        norms_before = list(self.gradient_norms_before)
        norms_after = list(self.gradient_norms_after)

        stats = {
            "steps": self.step_count,
            "clip_rate": self.clip_count / max(self.step_count, 1),
            "explosion_count": self.explosion_count,
            "vanishing_count": self.vanishing_count,
            "norm_before_mean": np.mean(norms_before),
            "norm_before_std": np.std(norms_before),
            "norm_before_max": np.max(norms_before),
            "norm_before_min": np.min(norms_before),
            "norm_after_mean": np.mean(norms_after),
            "norm_after_std": np.std(norms_after),
        }

        logger.info(
            f"📊 Gradient Stats (last {len(norms_before)} steps): "
            f"mean_norm={stats['norm_before_mean']:.4f}±{stats['norm_before_std']:.4f}, "
            f"clip_rate={stats['clip_rate']:.2%}, "
            f"explosions={stats['explosion_count']}, "
            f"vanishing={stats['vanishing_count']}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get gradient statistics"""
        if not self.gradient_norms_before:
            return {}

        norms_before = list(self.gradient_norms_before)
        norms_after = list(self.gradient_norms_after)

        return {
            "steps": self.step_count,
            "clip_count": self.clip_count,
            "clip_rate": self.clip_count / max(self.step_count, 1),
            "explosion_count": self.explosion_count,
            "vanishing_count": self.vanishing_count,
            "norm_before": {
                "mean": float(np.mean(norms_before)),
                "std": float(np.std(norms_before)),
                "max": float(np.max(norms_before)),
                "min": float(np.min(norms_before)),
                "median": float(np.median(norms_before)),
            },
            "norm_after": {
                "mean": float(np.mean(norms_after)),
                "std": float(np.std(norms_after)),
                "max": float(np.max(norms_after)),
                "min": float(np.min(norms_after)),
                "median": float(np.median(norms_after)),
            },
        }

    def save_history(self, filepath: str):
        """Save gradient history to file"""
        history = {
            "config": {
                "max_norm": self.max_norm,
                "norm_type": self.norm_type,
                "explosion_threshold": self.explosion_threshold,
                "vanishing_threshold": self.vanishing_threshold,
            },
            "statistics": self.get_statistics(),
            "history": {
                "norms_before": list(self.gradient_norms_before),
                "norms_after": list(self.gradient_norms_after),
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"✅ Gradient history saved to {filepath}")


class LRWarmup:
    """
    Learning rate warmup

    Linearly increases learning rate from lr_start to lr_target over warmup_steps.
    Prevents early training instability.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int = 1000,
        lr_start: float = 1e-5,
        lr_target: Optional[float] = None,
    ):
        """
        Initialize LR warmup

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            lr_start: Starting learning rate
            lr_target: Target learning rate (defaults to optimizer's LR)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr_start = lr_start
        self.lr_target = lr_target or optimizer.param_groups[0]['lr']
        self.current_step = 0
        self.warmup_complete = False

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_start

        logger.info(
            f"🔥 LR Warmup initialized: {lr_start:.2e} → {self.lr_target:.2e} "
            f"over {warmup_steps} steps"
        )

    def step(self) -> float:
        """
        Step the warmup scheduler

        Returns:
            Current learning rate
        """
        if self.warmup_complete:
            return self.lr_target

        self.current_step += 1

        if self.current_step >= self.warmup_steps:
            self.warmup_complete = True
            current_lr = self.lr_target
        else:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            current_lr = self.lr_start + (self.lr_target - self.lr_start) * progress

        # Update optimizer LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        if self.current_step == self.warmup_steps:
            logger.info(f"🔥 LR Warmup complete! Target LR: {current_lr:.2e}")

        return current_lr

    def is_complete(self) -> bool:
        """Check if warmup is complete"""
        return self.warmup_complete


class LRSchedulerManager:
    """
    Learning rate scheduler manager with auto-selection

    Features:
    - Multiple scheduler types (Cosine, Plateau, Step)
    - Auto-select best scheduler based on training phase and performance
    - Track learning rate history
    - Adaptive scheduler switching
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: str = "cosine",
        config: Optional[Dict[str, Any]] = None,
        enable_warmup: bool = True,
        warmup_steps: int = 1000,
    ):
        """
        Initialize LR scheduler manager

        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('cosine', 'plateau', 'step', 'auto')
            config: Configuration dict
            enable_warmup: Whether to enable LR warmup
            warmup_steps: Number of warmup steps
        """
        self.optimizer = optimizer
        self.config = config or {}
        self.base_lr = optimizer.param_groups[0]['lr']

        # Warmup
        self.enable_warmup = enable_warmup
        self.warmup = None
        if enable_warmup:
            lr_start = self.config.get('lr_warmup_start', 1e-5)
            self.warmup = LRWarmup(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                lr_start=lr_start,
                lr_target=self.base_lr,
            )

        # Scheduler type
        self.scheduler_type = SchedulerType(scheduler_type.lower())
        self.scheduler = self._create_scheduler()

        # Tracking
        self.lr_history = []
        self.step_count = 0
        self.training_phase = TrainingPhase.EXPLORATION
        self.loss_history = deque(maxlen=100)
        self.performance_trend = deque(maxlen=50)

        # Auto-scheduler selection
        self.auto_mode = self.scheduler_type == SchedulerType.AUTO
        self.last_scheduler_switch = 0

        logger.info(
            f"📈 LRSchedulerManager initialized (type={scheduler_type}, "
            f"warmup={enable_warmup}, base_lr={self.base_lr:.2e})"
        )

    def _create_scheduler(self) -> Optional[Any]:
        """Create LR scheduler based on type"""
        if self.scheduler_type == SchedulerType.COSINE:
            return self._create_cosine_scheduler()
        elif self.scheduler_type == SchedulerType.PLATEAU:
            return self._create_plateau_scheduler()
        elif self.scheduler_type == SchedulerType.STEP:
            return self._create_step_scheduler()
        elif self.scheduler_type == SchedulerType.AUTO:
            # Start with cosine for exploration phase
            return self._create_cosine_scheduler()

        return None

    def _create_cosine_scheduler(self):
        """Create CosineAnnealingWarmRestarts scheduler"""
        T_0 = self.config.get('lr_cosine_T_0', 50)
        T_mult = self.config.get('lr_cosine_T_mult', 2)
        eta_min = self.config.get('lr_min', 1e-6)

        logger.info(
            f"📈 Creating CosineAnnealingWarmRestarts: "
            f"T_0={T_0}, T_mult={T_mult}, eta_min={eta_min:.2e}"
        )

        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )

    def _create_plateau_scheduler(self):
        """Create ReduceLROnPlateau scheduler"""
        factor = self.config.get('lr_plateau_factor', 0.5)
        patience = self.config.get('lr_plateau_patience', 10)
        min_lr = self.config.get('lr_min', 1e-6)

        logger.info(
            f"📈 Creating ReduceLROnPlateau: "
            f"factor={factor}, patience={patience}, min_lr={min_lr:.2e}"
        )

        return ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True,
        )

    def _create_step_scheduler(self):
        """Create StepLR scheduler"""
        step_size = self.config.get('lr_step_size', 500)
        gamma = self.config.get('lr_step_gamma', 0.5)

        logger.info(
            f"📈 Creating StepLR: "
            f"step_size={step_size}, gamma={gamma}"
        )

        return StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    def step(self, loss: Optional[float] = None, log_step: bool = False):
        """
        Step the scheduler

        Args:
            loss: Current loss (required for ReduceLROnPlateau)
            log_step: Whether to log this step
        """
        self.step_count += 1

        # Warmup phase
        if self.warmup and not self.warmup.is_complete():
            current_lr = self.warmup.step()
            self.lr_history.append(current_lr)
            return

        # Track loss for auto-scheduler
        if loss is not None:
            self.loss_history.append(loss)

        # Auto-scheduler selection
        if self.auto_mode and self.step_count % 100 == 0:
            self._update_training_phase()
            self._maybe_switch_scheduler()

        # Step the scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if loss is not None:
                    self.scheduler.step(loss)
            else:
                self.scheduler.step()

        # Track LR
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)

        # Periodic logging
        if log_step or (self.step_count % 100 == 0):
            logger.info(
                f"📈 LR Step {self.step_count}: lr={current_lr:.2e}, "
                f"phase={self.training_phase.value}, "
                f"scheduler={self.scheduler_type.value}"
            )

    def _update_training_phase(self):
        """Update training phase based on performance"""
        if len(self.loss_history) < 50:
            self.training_phase = TrainingPhase.EXPLORATION
            return

        # Calculate loss trend
        recent_losses = list(self.loss_history)[-50:]
        loss_trend = np.mean(np.diff(recent_losses))

        # Calculate loss stability
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        stability = loss_std / (loss_mean + 1e-8)

        # Determine phase
        if self.step_count < 500:
            self.training_phase = TrainingPhase.EXPLORATION
        elif stability < 0.1 and abs(loss_trend) < 0.001:
            self.training_phase = TrainingPhase.REFINEMENT
        elif stability < 0.3:
            self.training_phase = TrainingPhase.EXPLOITATION
        else:
            self.training_phase = TrainingPhase.EXPLORATION

    def _maybe_switch_scheduler(self):
        """Switch scheduler if beneficial"""
        if self.step_count - self.last_scheduler_switch < 500:
            return  # Don't switch too frequently

        # Switch based on training phase
        if self.training_phase == TrainingPhase.EXPLORATION:
            if not isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                logger.info("🔄 Switching to CosineAnnealingWarmRestarts for exploration")
                self.scheduler = self._create_cosine_scheduler()
                self.last_scheduler_switch = self.step_count

        elif self.training_phase == TrainingPhase.EXPLOITATION:
            if not isinstance(self.scheduler, ReduceLROnPlateau):
                logger.info("🔄 Switching to ReduceLROnPlateau for exploitation")
                self.scheduler = self._create_plateau_scheduler()
                self.last_scheduler_switch = self.step_count

        elif self.training_phase == TrainingPhase.REFINEMENT:
            if not isinstance(self.scheduler, StepLR):
                logger.info("🔄 Switching to StepLR for refinement")
                self.scheduler = self._create_step_scheduler()
                self.last_scheduler_switch = self.step_count

    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def get_statistics(self) -> Dict[str, Any]:
        """Get LR scheduler statistics"""
        if not self.lr_history:
            return {}

        return {
            "steps": self.step_count,
            "current_lr": self.get_current_lr(),
            "base_lr": self.base_lr,
            "scheduler_type": self.scheduler_type.value,
            "training_phase": self.training_phase.value,
            "warmup_complete": self.warmup.is_complete() if self.warmup else True,
            "lr_history": {
                "mean": float(np.mean(self.lr_history)),
                "min": float(np.min(self.lr_history)),
                "max": float(np.max(self.lr_history)),
                "current": self.get_current_lr(),
            },
        }

    def save_history(self, filepath: str):
        """Save LR history to file"""
        history = {
            "config": {
                "scheduler_type": self.scheduler_type.value,
                "base_lr": self.base_lr,
                "enable_warmup": self.enable_warmup,
            },
            "statistics": self.get_statistics(),
            "history": {
                "learning_rates": self.lr_history,
                "losses": list(self.loss_history) if self.loss_history else [],
            },
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"✅ LR history saved to {filepath}")


class TrainingOptimizer:
    """
    Unified training optimizer combining gradient clipping and LR scheduling

    This is a convenience class that integrates all training optimizations.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize training optimizer

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            config: Configuration dict
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or {}

        # Gradient clipping
        gradient_clip_norm = self.config.get('gradient_clip_norm', 1.0)
        self.gradient_clipper = GradientClipper(
            max_norm=gradient_clip_norm,
            explosion_threshold=self.config.get('gradient_explosion_threshold', 10.0),
            vanishing_threshold=self.config.get('gradient_vanishing_threshold', 1e-6),
        )

        # LR scheduling
        scheduler_type = self.config.get('lr_scheduler_type', 'cosine')
        enable_warmup = self.config.get('lr_warmup_enabled', True)
        warmup_steps = self.config.get('lr_warmup_steps', 1000)

        self.lr_scheduler = LRSchedulerManager(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            config=config,
            enable_warmup=enable_warmup,
            warmup_steps=warmup_steps,
        )

        # Tracking
        self.step_count = 0

        logger.info("🚀 TrainingOptimizer initialized (gradient clipping + LR scheduling)")

    def step(
        self,
        loss: Optional[float] = None,
        log_step: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform optimization step with gradient clipping and LR scheduling

        Args:
            loss: Current loss (for plateau scheduler)
            log_step: Whether to log this step

        Returns:
            Dict with step info (gradient norms, LR, etc.)
        """
        self.step_count += 1

        # Clip gradients
        norm_before, norm_after, was_clipped = self.gradient_clipper.clip_gradients(
            self.model,
            log_step=log_step,
        )

        # Step optimizer
        self.optimizer.step()

        # Step LR scheduler
        self.lr_scheduler.step(loss=loss, log_step=log_step)

        # Return step info
        return {
            "step": self.step_count,
            "gradient_norm_before": norm_before,
            "gradient_norm_after": norm_after,
            "gradient_clipped": was_clipped,
            "learning_rate": self.lr_scheduler.get_current_lr(),
            "loss": loss,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        return {
            "steps": self.step_count,
            "gradient_clipper": self.gradient_clipper.get_statistics(),
            "lr_scheduler": self.lr_scheduler.get_statistics(),
        }

    def save_histories(self, output_dir: str):
        """Save all training histories"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save gradient history
        gradient_path = output_path / "gradient_history.json"
        self.gradient_clipper.save_history(str(gradient_path))

        # Save LR history
        lr_path = output_path / "lr_history.json"
        self.lr_scheduler.save_history(str(lr_path))

        logger.info(f"✅ Training histories saved to {output_dir}")


# Export main classes
__all__ = [
    "GradientClipper",
    "LRSchedulerManager",
    "LRWarmup",
    "TrainingOptimizer",
    "SchedulerType",
    "TrainingPhase",
]
