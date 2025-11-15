"""
Nexlify Hyperparameter Optimization Package
Automated hyperparameter tuning using Optuna for RL trading agents
"""

from nexlify.optimization.hyperparameter_tuner import HyperparameterTuner
from nexlify.optimization.hyperparameter_space import (
    HyperparameterSpace,
    DEFAULT_SEARCH_SPACE,
    create_custom_search_space,
    validate_hyperparameters
)
from nexlify.optimization.objective_functions import (
    ObjectiveFunction,
    SharpeObjective,
    ReturnObjective,
    DrawdownObjective,
    MultiObjective,
    create_objective,
    create_balanced_objective,
    create_aggressive_objective,
    create_conservative_objective
)
from nexlify.optimization.integration import (
    OptimizationIntegration,
    create_optimized_agent,
    run_training_with_optimization
)

__all__ = [
    # Core optimization
    'HyperparameterTuner',
    'HyperparameterSpace',
    'DEFAULT_SEARCH_SPACE',
    'create_custom_search_space',
    'validate_hyperparameters',

    # Objectives
    'ObjectiveFunction',
    'SharpeObjective',
    'ReturnObjective',
    'DrawdownObjective',
    'MultiObjective',
    'create_objective',
    'create_balanced_objective',
    'create_aggressive_objective',
    'create_conservative_objective',

    # Integration
    'OptimizationIntegration',
    'create_optimized_agent',
    'run_training_with_optimization',
]
