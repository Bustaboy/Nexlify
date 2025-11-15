"""
Hyperparameter Space Definitions
Defines search spaces for RL agent hyperparameters with validation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class HyperparameterSpace:
    """
    Defines hyperparameter search spaces for optimization

    Supports different parameter types:
    - 'float': Continuous float values with uniform sampling
    - 'loguniform': Log-uniform sampling (for learning rates, etc.)
    - 'int': Integer values
    - 'categorical': Discrete choices from a list
    """

    def __init__(self, search_space: Optional[Dict[str, Tuple]] = None):
        """
        Initialize hyperparameter search space

        Args:
            search_space: Dict mapping parameter names to (type, *args) tuples
                Examples:
                    'gamma': ('float', 0.90, 0.99)
                    'learning_rate': ('loguniform', 1e-5, 1e-2)
                    'batch_size': ('categorical', [32, 64, 128, 256])
        """
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self._validate_search_space()
        logger.info(f"Initialized hyperparameter space with {len(self.search_space)} parameters")

    def _validate_search_space(self) -> None:
        """Validate search space definition"""
        valid_types = {'float', 'loguniform', 'int', 'categorical'}

        for param_name, param_def in self.search_space.items():
            if not isinstance(param_def, tuple) or len(param_def) < 2:
                raise ValueError(
                    f"Invalid definition for '{param_name}': "
                    f"Expected tuple (type, *args), got {param_def}"
                )

            param_type = param_def[0]
            if param_type not in valid_types:
                raise ValueError(
                    f"Invalid type '{param_type}' for '{param_name}'. "
                    f"Must be one of {valid_types}"
                )

            # Type-specific validation
            if param_type in {'float', 'loguniform'}:
                if len(param_def) != 3:
                    raise ValueError(
                        f"'{param_name}' with type '{param_type}' requires "
                        f"(type, min, max), got {param_def}"
                    )
                if param_def[1] >= param_def[2]:
                    raise ValueError(
                        f"'{param_name}': min ({param_def[1]}) must be < max ({param_def[2]})"
                    )
                if param_type == 'loguniform' and param_def[1] <= 0:
                    raise ValueError(
                        f"'{param_name}' with loguniform requires min > 0, got {param_def[1]}"
                    )

            elif param_type == 'int':
                if len(param_def) != 3:
                    raise ValueError(
                        f"'{param_name}' with type 'int' requires "
                        f"(type, min, max), got {param_def}"
                    )
                if param_def[1] >= param_def[2]:
                    raise ValueError(
                        f"'{param_name}': min ({param_def[1]}) must be < max ({param_def[2]})"
                    )

            elif param_type == 'categorical':
                if len(param_def) != 2 or not isinstance(param_def[1], list):
                    raise ValueError(
                        f"'{param_name}' with type 'categorical' requires "
                        f"(type, [choices]), got {param_def}"
                    )
                if len(param_def[1]) == 0:
                    raise ValueError(f"'{param_name}' categorical choices cannot be empty")

    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters using Optuna trial

        Args:
            trial: Optuna trial object

        Returns:
            Dict of suggested hyperparameters
        """
        params = {}

        for param_name, param_def in self.search_space.items():
            param_type = param_def[0]

            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_def[1], param_def[2]
                )

            elif param_type == 'loguniform':
                params[param_name] = trial.suggest_float(
                    param_name, param_def[1], param_def[2], log=True
                )

            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_def[1], param_def[2]
                )

            elif param_type == 'categorical':
                # Handle nested lists (like hidden_layers)
                choices = param_def[1]
                if all(isinstance(c, list) for c in choices):
                    # For list choices, serialize for Optuna
                    choice_idx = trial.suggest_categorical(
                        f"{param_name}_idx", list(range(len(choices)))
                    )
                    params[param_name] = choices[choice_idx]
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, choices
                    )

        return params

    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names"""
        return list(self.search_space.keys())

    def get_parameter_bounds(self, param_name: str) -> Optional[Tuple]:
        """Get bounds for a specific parameter"""
        if param_name not in self.search_space:
            return None

        param_def = self.search_space[param_name]
        param_type = param_def[0]

        if param_type in {'float', 'loguniform', 'int'}:
            return (param_def[1], param_def[2])
        elif param_type == 'categorical':
            return param_def[1]

        return None


# Default search space for RL agent hyperparameters
DEFAULT_SEARCH_SPACE = {
    # Discount factor (gamma)
    'gamma': ('float', 0.90, 0.99),

    # Learning rate
    'learning_rate': ('loguniform', 1e-5, 1e-2),

    # Epsilon decay for exploration
    'epsilon_decay_steps': ('int', 500, 5000),
    'epsilon_start': ('float', 0.8, 1.0),
    'epsilon_end': ('float', 0.01, 0.1),

    # Batch size for training
    'batch_size': ('categorical', [32, 64, 128, 256]),

    # Neural network architecture
    'hidden_layers': ('categorical', [
        [64, 64],
        [128, 128],
        [128, 128, 64],
        [256, 256, 128],
        [256, 256, 256],
        [512, 256, 128]
    ]),

    # Replay buffer size
    'buffer_size': ('int', 10000, 200000),

    # N-step returns
    'n_step': ('int', 1, 10),

    # Target network update frequency
    'target_update_frequency': ('int', 50, 500),

    # Tau for soft updates
    'tau': ('float', 0.001, 0.1),

    # Gradient clipping
    'gradient_clip': ('float', 0.5, 10.0),

    # Dropout rate
    'dropout_rate': ('float', 0.0, 0.5),

    # L2 regularization
    'l2_reg': ('loguniform', 1e-6, 1e-3),

    # Prioritized experience replay
    'per_alpha': ('float', 0.4, 0.8),
    'per_beta_start': ('float', 0.4, 0.6),
    'per_beta_frames': ('int', 10000, 100000),

    # Optimizer choice
    'optimizer': ('categorical', ['adam', 'adamw', 'rmsprop']),
}


# Compact search space for quick optimization
COMPACT_SEARCH_SPACE = {
    'gamma': ('float', 0.95, 0.99),
    'learning_rate': ('loguniform', 1e-4, 1e-2),
    'batch_size': ('categorical', [64, 128, 256]),
    'hidden_layers': ('categorical', [
        [128, 128],
        [256, 256, 128],
    ]),
    'buffer_size': ('int', 50000, 150000),
    'n_step': ('int', 1, 5),
}


# Advanced search space with more parameters
ADVANCED_SEARCH_SPACE = {
    **DEFAULT_SEARCH_SPACE,

    # Additional advanced parameters
    'double_dqn': ('categorical', [True, False]),
    'dueling_dqn': ('categorical', [True, False]),
    'noisy_net': ('categorical', [True, False]),

    # Learning rate scheduling
    'lr_scheduler': ('categorical', ['none', 'step', 'cosine', 'exponential']),
    'lr_decay_rate': ('float', 0.9, 0.99),

    # Multi-step returns mixing
    'nstep_mix_ratio': ('float', 0.0, 1.0),

    # Reward scaling
    'reward_scale': ('loguniform', 0.1, 10.0),
}


def create_custom_search_space(
    base_space: str = 'default',
    override_params: Optional[Dict[str, Tuple]] = None,
    additional_params: Optional[Dict[str, Tuple]] = None
) -> HyperparameterSpace:
    """
    Create custom search space based on predefined template

    Args:
        base_space: Base search space ('default', 'compact', 'advanced')
        override_params: Parameters to override in base space
        additional_params: Additional parameters to add

    Returns:
        HyperparameterSpace instance

    Example:
        >>> space = create_custom_search_space(
        ...     base_space='compact',
        ...     override_params={'learning_rate': ('loguniform', 1e-5, 1e-3)},
        ...     additional_params={'custom_param': ('float', 0.0, 1.0)}
        ... )
    """
    # Select base space
    if base_space == 'default':
        search_space = DEFAULT_SEARCH_SPACE.copy()
    elif base_space == 'compact':
        search_space = COMPACT_SEARCH_SPACE.copy()
    elif base_space == 'advanced':
        search_space = ADVANCED_SEARCH_SPACE.copy()
    else:
        raise ValueError(
            f"Unknown base_space '{base_space}'. "
            f"Choose from: 'default', 'compact', 'advanced'"
        )

    # Apply overrides
    if override_params:
        for param_name, param_def in override_params.items():
            if param_name in search_space:
                logger.info(f"Overriding parameter '{param_name}'")
            search_space[param_name] = param_def

    # Add additional parameters
    if additional_params:
        for param_name, param_def in additional_params.items():
            if param_name in search_space:
                logger.warning(
                    f"Parameter '{param_name}' already exists, "
                    f"use override_params to modify it"
                )
            else:
                search_space[param_name] = param_def

    return HyperparameterSpace(search_space)


def validate_hyperparameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate hyperparameter values

    Args:
        params: Dictionary of hyperparameters to validate

    Returns:
        Tuple of (is_valid, list of error messages)

    Example:
        >>> params = {'gamma': 0.95, 'learning_rate': 0.001}
        >>> is_valid, errors = validate_hyperparameters(params)
        >>> if not is_valid:
        ...     print("Errors:", errors)
    """
    errors = []

    # Check required parameters
    required_params = {
        'gamma', 'learning_rate', 'batch_size', 'hidden_layers'
    }
    missing = required_params - set(params.keys())
    if missing:
        errors.append(f"Missing required parameters: {missing}")

    # Validate parameter values
    if 'gamma' in params:
        if not 0 < params['gamma'] < 1:
            errors.append(f"gamma must be in (0, 1), got {params['gamma']}")

    if 'learning_rate' in params:
        if params['learning_rate'] <= 0:
            errors.append(
                f"learning_rate must be positive, got {params['learning_rate']}"
            )

    if 'batch_size' in params:
        if params['batch_size'] <= 0:
            errors.append(
                f"batch_size must be positive, got {params['batch_size']}"
            )

    if 'buffer_size' in params:
        if params['buffer_size'] < params.get('batch_size', 0):
            errors.append(
                f"buffer_size ({params['buffer_size']}) must be >= "
                f"batch_size ({params.get('batch_size', 0)})"
            )

    if 'epsilon_start' in params and 'epsilon_end' in params:
        if params['epsilon_start'] < params['epsilon_end']:
            errors.append(
                f"epsilon_start ({params['epsilon_start']}) must be >= "
                f"epsilon_end ({params['epsilon_end']})"
            )

    if 'hidden_layers' in params:
        layers = params['hidden_layers']
        if not isinstance(layers, list) or len(layers) == 0:
            errors.append(f"hidden_layers must be non-empty list, got {layers}")
        elif not all(isinstance(x, int) and x > 0 for x in layers):
            errors.append(f"hidden_layers must contain positive integers, got {layers}")

    is_valid = len(errors) == 0
    return is_valid, errors
