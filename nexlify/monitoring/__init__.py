"""
Nexlify Training Monitoring System
Real-time dashboard and metrics tracking for RL agent training

This package provides comprehensive monitoring capabilities:
- Real-time training dashboard with live plots
- Metrics logging with minimal overhead
- Alert system for critical events
- Episode comparison tools
- Model diagnostics tracking
- Experiment tracking and comparison
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Lazy imports to avoid heavy dependencies at package level
if TYPE_CHECKING:
    from nexlify.monitoring.metrics_logger import MetricsLogger
    from nexlify.monitoring.training_dashboard import TrainingDashboard
    from nexlify.monitoring.alert_system import AlertSystem
    from nexlify.monitoring.experiment_tracker import ExperimentTracker

__all__ = [
    'MetricsLogger',
    'TrainingDashboard',
    'AlertSystem',
    'ExperimentTracker',
]

__version__ = '1.0.0'
