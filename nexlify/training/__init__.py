"""
Nexlify Training Optimization Module

Comprehensive training utilities for gradient clipping, learning rate scheduling,
training stability optimization, validation monitoring, and early stopping.
"""

from nexlify.training.training_optimizers import (
    GradientClipper,
    LRSchedulerManager,
    LRWarmup,
    TrainingOptimizer,
)

from nexlify.training.validation_monitor import (
    ValidationMonitor,
    ValidationDataSplitter,
    ValidationResult,
    DataSplit,
)

from nexlify.training.early_stopping import (
    EarlyStopping,
    EarlyStoppingConfig,
    TrainingPhaseDetector,
    TrainingPhase,
    OverfittingDetector,
)

__all__ = [
    # Training optimizers
    "GradientClipper",
    "LRSchedulerManager",
    "LRWarmup",
    "TrainingOptimizer",
    # Validation monitoring
    "ValidationMonitor",
    "ValidationDataSplitter",
    "ValidationResult",
    "DataSplit",
    # Early stopping
    "EarlyStopping",
    "EarlyStoppingConfig",
    "TrainingPhaseDetector",
    "TrainingPhase",
    "OverfittingDetector",
]
