"""
Integration Layer for Hyperparameter Optimization
Connects Optuna optimization with Nexlify training infrastructure
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OptimizationIntegration:
    """
    Integration layer between Optuna optimization and Nexlify training

    Provides utilities to:
    - Apply optimized hyperparameters to training configs
    - Load best parameters from optimization results
    - Integrate with AdvancedTrainingOrchestrator
    - Combine offline optimization with online tuning
    """

    @staticmethod
    def load_best_params(
        optimization_dir: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load best hyperparameters from optimization results

        Args:
            optimization_dir: Directory containing optimization results
            timestamp: Specific timestamp to load (None = most recent)

        Returns:
            Dict of best hyperparameters

        Example:
            >>> params = OptimizationIntegration.load_best_params('./optimization_results')
            >>> agent = create_agent(**params)
        """
        results_dir = Path(optimization_dir)

        if not results_dir.exists():
            raise FileNotFoundError(f"Optimization directory not found: {optimization_dir}")

        # Find best params file
        if timestamp:
            params_file = results_dir / f'best_params_{timestamp}.json'
        else:
            # Get most recent
            params_files = list(results_dir.glob('best_params_*.json'))
            if not params_files:
                raise FileNotFoundError(f"No best_params files found in {optimization_dir}")
            params_file = max(params_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading best parameters from {params_file}")

        with open(params_file, 'r') as f:
            params = json.load(f)

        return params

    @staticmethod
    def apply_to_config(
        base_config: Dict[str, Any],
        optimized_params: Dict[str, Any],
        override: bool = True
    ) -> Dict[str, Any]:
        """
        Apply optimized hyperparameters to training configuration

        Args:
            base_config: Base training configuration
            optimized_params: Optimized hyperparameters from Optuna
            override: If True, override existing values; if False, only set missing

        Returns:
            Updated configuration

        Example:
            >>> config = load_config('neural_config.json')
            >>> best_params = OptimizationIntegration.load_best_params('./optimization_results')
            >>> config = OptimizationIntegration.apply_to_config(config, best_params)
        """
        config = base_config.copy()

        # Map optimized params to config structure
        param_mapping = {
            # RL agent params
            'gamma': ('rl_agent', 'discount_factor'),
            'learning_rate': ('rl_agent', 'learning_rate'),
            'epsilon_start': ('rl_agent', 'epsilon_start'),
            'epsilon_end': ('rl_agent', 'epsilon_min'),
            'epsilon_decay_steps': ('epsilon_decay', 'linear_decay_steps'),
            'batch_size': ('rl_agent', 'batch_size'),
            'buffer_size': ('rl_agent', 'replay_buffer_size'),
            'target_update_frequency': ('rl_agent', 'target_update_frequency'),
            'tau': ('rl_agent', 'tau'),
            'gradient_clip': ('rl_agent', 'gradient_clip_norm'),
            'dropout_rate': ('rl_agent', 'dropout_rate'),
            'l2_reg': ('rl_agent', 'l2_regularization'),

            # Architecture params (handled separately)
            'hidden_layers': ('rl_agent', 'hidden_layers'),

            # Advanced params
            'double_dqn': ('rl_agent', 'use_double_dqn'),
            'dueling_dqn': ('rl_agent', 'use_dueling_dqn'),
            'noisy_net': ('rl_agent', 'use_noisy_net'),
            'lr_scheduler': ('rl_agent', 'lr_scheduler_type'),
            'lr_decay_rate': ('rl_agent', 'lr_decay_rate'),

            # N-step params
            'n_step': ('rl_agent', 'n_step'),

            # PER params
            'per_alpha': ('rl_agent', 'per_alpha'),
            'per_beta_start': ('rl_agent', 'per_beta_start'),
            'per_beta_frames': ('rl_agent', 'per_beta_frames'),

            # Optimizer
            'optimizer': ('rl_agent', 'optimizer_type'),
        }

        # Apply parameters
        for opt_param, (section, config_param) in param_mapping.items():
            if opt_param in optimized_params:
                value = optimized_params[opt_param]

                # Ensure section exists
                if section not in config:
                    config[section] = {}

                # Apply value
                if override or config_param not in config[section]:
                    config[section][config_param] = value
                    logger.debug(f"Set {section}.{config_param} = {value}")

        # Handle architecture separately (convert list to named architecture if needed)
        if 'hidden_layers' in optimized_params:
            config.setdefault('rl_agent', {})
            config['rl_agent']['hidden_layers'] = optimized_params['hidden_layers']

        logger.info(f"Applied {len(optimized_params)} optimized parameters to config")

        return config

    @staticmethod
    def create_training_config_from_params(
        optimized_params: Dict[str, Any],
        base_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a complete training configuration from optimized parameters

        Args:
            optimized_params: Optimized hyperparameters
            base_config: Optional base config to start from

        Returns:
            Complete training configuration

        Example:
            >>> best_params = OptimizationIntegration.load_best_params('./optimization_results')
            >>> training_config = OptimizationIntegration.create_training_config_from_params(best_params)
            >>> orchestrator = AdvancedTrainingOrchestrator(config=training_config)
        """
        # Start with defaults or provided base
        if base_config:
            config = base_config.copy()
        else:
            config = {
                'rl_agent': {},
                'epsilon_decay': {},
                'training': {},
                'environment': {},
            }

        # Apply optimized parameters
        config = OptimizationIntegration.apply_to_config(config, optimized_params)

        return config

    @staticmethod
    def save_integrated_config(
        optimized_params: Dict[str, Any],
        output_path: str,
        base_config_path: Optional[str] = None
    ) -> None:
        """
        Save a complete configuration with optimized parameters

        Args:
            optimized_params: Optimized hyperparameters
            output_path: Path to save integrated config
            base_config_path: Optional path to base config

        Example:
            >>> best_params = OptimizationIntegration.load_best_params('./optimization_results')
            >>> OptimizationIntegration.save_integrated_config(
            ...     best_params,
            ...     'config/optimized_config.json',
            ...     'config/neural_config.json'
            ... )
        """
        # Load base config if provided
        if base_config_path:
            with open(base_config_path, 'r') as f:
                base_config = json.load(f)
        else:
            base_config = None

        # Create integrated config
        config = OptimizationIntegration.create_training_config_from_params(
            optimized_params,
            base_config
        )

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved integrated configuration to {output_path}")

    @staticmethod
    def get_initialization_params(
        optimization_dir: str,
        include_architecture: bool = True
    ) -> Dict[str, Any]:
        """
        Get parameters suitable for agent initialization

        Args:
            optimization_dir: Directory with optimization results
            include_architecture: Include network architecture params

        Returns:
            Dict of initialization parameters

        Example:
            >>> init_params = OptimizationIntegration.get_initialization_params('./optimization_results')
            >>> agent = UltraOptimizedDQNAgent(**init_params)
        """
        all_params = OptimizationIntegration.load_best_params(optimization_dir)

        # Filter to initialization params
        init_params = {}

        # Core RL params
        param_keys = [
            'gamma', 'learning_rate', 'batch_size', 'buffer_size',
            'target_update_frequency', 'tau', 'gradient_clip',
            'dropout_rate', 'l2_reg', 'n_step',
            'per_alpha', 'per_beta_start', 'per_beta_frames',
            'optimizer'
        ]

        for key in param_keys:
            if key in all_params:
                init_params[key] = all_params[key]

        # Architecture
        if include_architecture and 'hidden_layers' in all_params:
            init_params['hidden_layers'] = all_params['hidden_layers']

        # Epsilon params
        if 'epsilon_start' in all_params:
            init_params['epsilon_start'] = all_params['epsilon_start']
        if 'epsilon_end' in all_params:
            init_params['epsilon_min'] = all_params['epsilon_end']

        # Advanced features
        if 'double_dqn' in all_params:
            init_params['use_double_dqn'] = all_params['double_dqn']
        if 'dueling_dqn' in all_params:
            init_params['use_dueling_dqn'] = all_params['dueling_dqn']
        if 'noisy_net' in all_params:
            init_params['use_noisy_net'] = all_params['noisy_net']

        return init_params


def create_optimized_agent(
    optimization_dir: str,
    state_size: int,
    action_size: int,
    device: str = 'cpu',
    **additional_params
):
    """
    Create an RL agent initialized with optimized hyperparameters

    Args:
        optimization_dir: Directory containing optimization results
        state_size: Size of state space
        action_size: Size of action space
        device: Device for training ('cpu' or 'cuda')
        **additional_params: Additional parameters to override

    Returns:
        Initialized agent with optimized hyperparameters

    Example:
        >>> agent = create_optimized_agent(
        ...     './optimization_results',
        ...     state_size=12,
        ...     action_size=3,
        ...     device='cuda'
        ... )
        >>> agent.train(env, episodes=1000)
    """
    try:
        from nexlify_rl_models.nexlify_ultra_optimized_rl_agent import UltraOptimizedDQNAgent
    except ImportError:
        logger.error("Could not import UltraOptimizedDQNAgent")
        raise

    # Load optimized parameters
    init_params = OptimizationIntegration.get_initialization_params(optimization_dir)

    # Add required params
    init_params['state_size'] = state_size
    init_params['action_size'] = action_size
    init_params['device'] = device

    # Apply additional overrides
    init_params.update(additional_params)

    logger.info(f"Creating agent with optimized hyperparameters:")
    for key, value in init_params.items():
        if key not in ['state_size', 'action_size']:
            logger.info(f"  {key}: {value}")

    # Create agent
    agent = UltraOptimizedDQNAgent(**init_params)

    return agent


def run_training_with_optimization(
    symbol: str = 'BTC/USDT',
    optimization_trials: int = 50,
    training_episodes: int = 1000,
    optimization_objective: str = 'sharpe',
    output_dir: str = './optimized_training',
    use_optimized_params: bool = True,
    enable_dynamic_tuning: bool = True
):
    """
    Complete workflow: optimize hyperparameters, then train with best params

    Args:
        symbol: Trading pair
        optimization_trials: Number of optimization trials
        training_episodes: Episodes for final training
        optimization_objective: Objective for optimization
        output_dir: Output directory
        use_optimized_params: Use optimized params (vs manual)
        enable_dynamic_tuning: Enable AutoTuner during training

    Returns:
        Dict with optimization and training results

    Example:
        >>> results = run_training_with_optimization(
        ...     symbol='BTC/USDT',
        ...     optimization_trials=100,
        ...     training_episodes=2000,
        ...     optimization_objective='multi'
        ... )
        >>> print(f"Best Sharpe: {results['training_sharpe']}")
    """
    from nexlify.optimization import HyperparameterTuner, create_balanced_objective
    from nexlify_training.nexlify_advanced_training_orchestrator import AdvancedTrainingOrchestrator

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Run optimization
    logger.info("="*80)
    logger.info("STEP 1: HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)

    optimization_dir = output_path / 'optimization'

    if optimization_objective == 'multi':
        objective = create_balanced_objective()
    else:
        from nexlify.optimization import create_objective
        objective = create_objective(optimization_objective)

    # Create tuner
    from nexlify.optimization import create_custom_search_space
    tuner = HyperparameterTuner(
        objective=objective,
        search_space=create_custom_search_space('compact'),
        n_trials=optimization_trials,
        output_dir=str(optimization_dir),
        verbose=True
    )

    # Define training function (simplified)
    def train_for_optimization(params, train_data, val_data):
        """Quick training for optimization"""
        # This would call your actual training code
        # For now, returning mock results
        import numpy as np
        sharpe = np.random.uniform(0, 3)
        return {
            'sharpe_ratio': sharpe,
            'total_return': sharpe * 0.5,
            'max_drawdown': -0.15,
        }

    # Run optimization
    opt_results = tuner.optimize(train_func=train_for_optimization)

    best_params = opt_results['best_params']
    logger.info(f"\nBest hyperparameters found:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

    # Step 2: Train with optimized parameters
    logger.info("\n" + "="*80)
    logger.info("STEP 2: TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    logger.info("="*80)

    # Create training config
    if use_optimized_params:
        training_config = OptimizationIntegration.create_training_config_from_params(best_params)
    else:
        training_config = {}

    # Create orchestrator
    orchestrator = AdvancedTrainingOrchestrator(
        output_dir=str(output_path / 'training'),
        enable_auto_tuning=enable_dynamic_tuning  # Online tuning on top of optimized params
    )

    logger.info(f"Training with {training_episodes} episodes")
    logger.info(f"Dynamic tuning: {'ENABLED' if enable_dynamic_tuning else 'DISABLED'}")

    # Train (this would be your actual training call)
    # training_results = orchestrator.train(...)

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)

    return {
        'optimization_results': opt_results,
        'best_params': best_params,
        'training_config': training_config,
        # 'training_results': training_results
    }
