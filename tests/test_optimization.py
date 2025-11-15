"""
Tests for Hyperparameter Optimization System
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

from nexlify.optimization import (
    HyperparameterSpace,
    HyperparameterTuner,
    DEFAULT_SEARCH_SPACE,
    COMPACT_SEARCH_SPACE,
    create_custom_search_space,
    validate_hyperparameters,
    ObjectiveFunction,
    SharpeObjective,
    ReturnObjective,
    DrawdownObjective,
    MultiObjective,
    create_objective,
    create_balanced_objective,
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# Fixtures

@pytest.fixture
def sample_search_space():
    """Sample search space for testing"""
    return {
        'gamma': ('float', 0.90, 0.99),
        'learning_rate': ('loguniform', 1e-4, 1e-2),
        'batch_size': ('categorical', [32, 64, 128]),
        'hidden_layers': ('categorical', [[64, 64], [128, 128]]),
        'n_step': ('int', 1, 5),
    }


@pytest.fixture
def sample_training_results():
    """Sample training results for objective testing"""
    returns = np.random.randn(100) * 0.02 + 0.001
    balance_history = [10000]
    for ret in returns:
        balance_history.append(balance_history[-1] * (1 + ret))

    return {
        'returns': returns,
        'final_balance': balance_history[-1],
        'initial_balance': 10000,
        'balance_history': balance_history,
        'trades': returns,
        'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
        'total_return': (balance_history[-1] - 10000) / 10000,
        'max_drawdown': -0.15,
        'win_rate': 0.55
    }


@pytest.fixture
def mock_train_function():
    """Mock training function for testing"""
    def train_func(params: Dict[str, Any], train_data=None, val_data=None) -> Dict[str, Any]:
        """Mock training that returns random results"""
        # Simulate that better learning rates give better results
        lr_score = -np.log10(params.get('learning_rate', 0.001))
        gamma_score = params.get('gamma', 0.95)
        base_score = (lr_score + gamma_score) / 2

        sharpe = base_score + np.random.randn() * 0.1

        returns = np.random.randn(50) * 0.02
        balance_history = [10000]
        for ret in returns:
            balance_history.append(balance_history[-1] * (1 + ret))

        return {
            'sharpe_ratio': float(sharpe),
            'total_return': float(np.random.uniform(-0.1, 0.3)),
            'max_drawdown': float(np.random.uniform(-0.3, -0.05)),
            'final_balance': balance_history[-1],
            'initial_balance': 10000,
            'returns': returns.tolist(),
            'balance_history': balance_history,
        }

    return train_func


# HyperparameterSpace Tests

@pytest.mark.unit
def test_hyperparameter_space_init(sample_search_space):
    """Test HyperparameterSpace initialization"""
    space = HyperparameterSpace(sample_search_space)
    assert space.search_space == sample_search_space
    assert len(space.get_parameter_names()) == 5


@pytest.mark.unit
def test_hyperparameter_space_validation():
    """Test search space validation"""
    # Valid space
    valid_space = {
        'gamma': ('float', 0.90, 0.99),
        'lr': ('loguniform', 1e-5, 1e-2),
    }
    space = HyperparameterSpace(valid_space)
    assert space is not None

    # Invalid: min >= max
    with pytest.raises(ValueError):
        HyperparameterSpace({
            'gamma': ('float', 0.99, 0.90)  # min > max
        })

    # Invalid: loguniform with negative min
    with pytest.raises(ValueError):
        HyperparameterSpace({
            'lr': ('loguniform', -0.01, 0.01)  # min <= 0
        })

    # Invalid: empty categorical choices
    with pytest.raises(ValueError):
        HyperparameterSpace({
            'choice': ('categorical', [])  # empty list
        })

    # Invalid: unknown type
    with pytest.raises(ValueError):
        HyperparameterSpace({
            'param': ('unknown_type', 0, 1)
        })


@pytest.mark.unit
@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
def test_hyperparameter_space_suggest(sample_search_space):
    """Test parameter suggestion"""
    space = HyperparameterSpace(sample_search_space)

    # Create mock trial
    study = optuna.create_study()
    trial = study.ask()

    params = space.suggest_hyperparameters(trial)

    # Check all parameters are suggested
    assert 'gamma' in params
    assert 'learning_rate' in params
    assert 'batch_size' in params
    assert 'hidden_layers' in params
    assert 'n_step' in params

    # Check bounds
    assert 0.90 <= params['gamma'] <= 0.99
    assert 1e-4 <= params['learning_rate'] <= 1e-2
    assert params['batch_size'] in [32, 64, 128]
    assert params['hidden_layers'] in [[64, 64], [128, 128]]
    assert 1 <= params['n_step'] <= 5


@pytest.mark.unit
def test_create_custom_search_space():
    """Test custom search space creation"""
    # Default base
    space = create_custom_search_space(base_space='default')
    assert 'gamma' in space.search_space
    assert 'learning_rate' in space.search_space

    # Compact base
    space = create_custom_search_space(base_space='compact')
    assert len(space.search_space) < len(DEFAULT_SEARCH_SPACE)

    # With overrides
    space = create_custom_search_space(
        base_space='compact',
        override_params={'gamma': ('float', 0.95, 0.99)}
    )
    assert space.search_space['gamma'] == ('float', 0.95, 0.99)

    # With additional params
    space = create_custom_search_space(
        base_space='compact',
        additional_params={'custom_param': ('float', 0.0, 1.0)}
    )
    assert 'custom_param' in space.search_space


@pytest.mark.unit
def test_validate_hyperparameters():
    """Test hyperparameter validation"""
    # Valid parameters
    valid_params = {
        'gamma': 0.95,
        'learning_rate': 0.001,
        'batch_size': 64,
        'hidden_layers': [128, 128]
    }
    is_valid, errors = validate_hyperparameters(valid_params)
    assert is_valid
    assert len(errors) == 0

    # Missing required parameter
    invalid_params = {
        'gamma': 0.95,
        # missing learning_rate, batch_size, hidden_layers
    }
    is_valid, errors = validate_hyperparameters(invalid_params)
    assert not is_valid
    assert len(errors) > 0

    # Invalid gamma
    invalid_params = {
        'gamma': 1.5,  # > 1
        'learning_rate': 0.001,
        'batch_size': 64,
        'hidden_layers': [128, 128]
    }
    is_valid, errors = validate_hyperparameters(invalid_params)
    assert not is_valid

    # Invalid learning rate
    invalid_params = {
        'gamma': 0.95,
        'learning_rate': -0.001,  # negative
        'batch_size': 64,
        'hidden_layers': [128, 128]
    }
    is_valid, errors = validate_hyperparameters(invalid_params)
    assert not is_valid


# Objective Function Tests

@pytest.mark.unit
def test_sharpe_objective(sample_training_results):
    """Test Sharpe ratio objective"""
    objective = SharpeObjective()

    assert objective.name == 'sharpe_ratio'
    assert objective.direction == 'maximize'

    # Calculate from results
    score = objective.calculate(sample_training_results)
    assert isinstance(score, float)
    assert not np.isnan(score)

    # Should match pre-calculated value roughly
    expected = sample_training_results['sharpe_ratio']
    assert abs(score - expected) < 1.0  # Allow some difference due to annualization


@pytest.mark.unit
def test_return_objective(sample_training_results):
    """Test return objective"""
    objective = ReturnObjective()

    assert objective.name == 'total_return'
    assert objective.direction == 'maximize'

    score = objective.calculate(sample_training_results)
    assert isinstance(score, float)

    # Check with volatility penalty
    objective_with_penalty = ReturnObjective(volatility_penalty=0.1)
    score_penalized = objective_with_penalty.calculate(sample_training_results)
    assert score_penalized < score  # Should be lower due to penalty


@pytest.mark.unit
def test_drawdown_objective(sample_training_results):
    """Test drawdown objective"""
    objective = DrawdownObjective()

    assert objective.name == 'max_drawdown'
    assert objective.direction == 'minimize'

    score = objective.calculate(sample_training_results)
    assert isinstance(score, float)
    assert score <= 0  # Drawdown is negative


@pytest.mark.unit
def test_multi_objective(sample_training_results):
    """Test multi-objective"""
    objectives = [
        (SharpeObjective(), 0.5),
        (ReturnObjective(), 0.3),
        (DrawdownObjective(), 0.2)
    ]
    multi_obj = MultiObjective(objectives)

    assert multi_obj.name == 'multi_objective'
    assert multi_obj.direction == 'maximize'

    score = multi_obj.calculate(sample_training_results)
    assert isinstance(score, float)
    assert not np.isnan(score)


@pytest.mark.unit
def test_create_objective():
    """Test objective factory function"""
    # Sharpe
    obj = create_objective('sharpe')
    assert isinstance(obj, SharpeObjective)

    # Return
    obj = create_objective('return')
    assert isinstance(obj, ReturnObjective)

    # Drawdown
    obj = create_objective('drawdown')
    assert isinstance(obj, DrawdownObjective)

    # Invalid type
    with pytest.raises(ValueError):
        create_objective('invalid_type')


@pytest.mark.unit
def test_balanced_objective():
    """Test balanced multi-objective"""
    obj = create_balanced_objective()
    assert isinstance(obj, MultiObjective)
    assert len(obj.objectives) == 3


# HyperparameterTuner Tests

@pytest.mark.integration
@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
def test_hyperparameter_tuner_init(sample_search_space):
    """Test HyperparameterTuner initialization"""
    space = HyperparameterSpace(sample_search_space)
    objective = SharpeObjective()

    tuner = HyperparameterTuner(
        objective=objective,
        search_space=space,
        n_trials=10,
        sampler='tpe',
        pruner='median'
    )

    assert tuner.objective == objective
    assert tuner.search_space == space
    assert tuner.n_trials == 10


@pytest.mark.integration
@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
def test_hyperparameter_tuner_optimize(sample_search_space, mock_train_function, tmp_path):
    """Test hyperparameter optimization"""
    space = HyperparameterSpace(sample_search_space)
    objective = SharpeObjective()

    tuner = HyperparameterTuner(
        objective=objective,
        search_space=space,
        n_trials=5,  # Small number for testing
        sampler='random',  # Random is faster
        pruner='none',
        output_dir=str(tmp_path),
        verbose=False
    )

    # Run optimization
    results = tuner.optimize(
        train_func=mock_train_function,
        train_data=None,
        validation_data=None
    )

    # Check results
    assert 'best_params' in results
    assert 'best_value' in results
    assert 'study' in results
    assert 'optimization_history' in results

    assert len(results['optimization_history']) == 5
    assert tuner.best_params is not None
    assert tuner.best_value is not None

    # Check that files were saved
    assert (tmp_path / 'best_params_*.json').exists() or len(list(tmp_path.glob('best_params_*.json'))) > 0


@pytest.mark.integration
@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
def test_hyperparameter_tuner_with_timeout(sample_search_space, tmp_path):
    """Test optimization with timeout"""
    import time

    # Create a slower mock function to properly test timeout
    def slow_train_func(params, train_data=None, val_data=None):
        """Mock training with small delay to test timeout"""
        time.sleep(0.01)  # 10ms per trial
        sharpe = np.random.uniform(-1, 3)
        returns = np.random.randn(50) * 0.02
        balance_history = [10000]
        for ret in returns:
            balance_history.append(balance_history[-1] * (1 + ret))

        return {
            'sharpe_ratio': sharpe,
            'total_return': np.random.uniform(-0.1, 0.3),
            'max_drawdown': np.random.uniform(-0.3, -0.05),
            'final_balance': balance_history[-1],
            'initial_balance': 10000,
            'returns': returns.tolist(),
            'balance_history': balance_history,
        }

    space = HyperparameterSpace(sample_search_space)
    objective = SharpeObjective()

    tuner = HyperparameterTuner(
        objective=objective,
        search_space=space,
        n_trials=1000,  # Large number
        timeout=2,  # But short timeout (with 10ms/trial, should get ~200 trials max)
        sampler='random',
        pruner='none',
        output_dir=str(tmp_path),
        verbose=False
    )

    results = tuner.optimize(
        train_func=slow_train_func
    )

    # With 10ms per trial and 2s timeout, should complete 100-200 trials
    # (accounting for Optuna overhead)
    assert len(results['optimization_history']) < 500  # Much less than 1000
    assert results['elapsed_time'] < 5  # Timeout respected (2s + margin)


@pytest.mark.integration
@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
def test_hyperparameter_tuner_report(sample_search_space, mock_train_function, tmp_path):
    """Test optimization report generation"""
    space = HyperparameterSpace(sample_search_space)
    objective = SharpeObjective()

    tuner = HyperparameterTuner(
        objective=objective,
        search_space=space,
        n_trials=5,
        sampler='random',
        pruner='none',
        output_dir=str(tmp_path),
        verbose=False
    )

    tuner.optimize(train_func=mock_train_function)

    # Generate report
    report = tuner.generate_report()
    assert isinstance(report, str)
    assert 'BEST TRIAL' in report
    assert 'TRIAL STATISTICS' in report


@pytest.mark.integration
@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
def test_different_samplers(sample_search_space, mock_train_function, tmp_path):
    """Test different sampling algorithms"""
    space = HyperparameterSpace(sample_search_space)
    objective = SharpeObjective()

    samplers = ['tpe', 'random']  # CMA-ES requires continuous space

    for sampler_name in samplers:
        tuner = HyperparameterTuner(
            objective=objective,
            search_space=space,
            n_trials=3,
            sampler=sampler_name,
            pruner='none',
            output_dir=str(tmp_path / sampler_name),
            verbose=False
        )

        results = tuner.optimize(train_func=mock_train_function)
        assert results['best_params'] is not None


@pytest.mark.unit
def test_edge_cases():
    """Test edge cases"""
    # Empty returns
    objective = SharpeObjective()
    result = objective.calculate({'returns': []})
    assert result == float('-inf')

    # Zero std returns
    objective = SharpeObjective()
    result = objective.calculate({'returns': [0.01] * 100})
    assert result == float('-inf')

    # Division by zero in return calculation
    objective = ReturnObjective()
    with pytest.raises(ValueError):
        objective.calculate({
            'final_balance': 11000,
            'initial_balance': 0  # Invalid
        })


@pytest.mark.unit
def test_parameter_importance(sample_search_space, mock_train_function, tmp_path):
    """Test parameter importance calculation"""
    if not OPTUNA_AVAILABLE:
        pytest.skip("Optuna not available")

    space = HyperparameterSpace(sample_search_space)
    objective = SharpeObjective()

    tuner = HyperparameterTuner(
        objective=objective,
        search_space=space,
        n_trials=10,
        sampler='random',
        pruner='none',
        output_dir=str(tmp_path),
        verbose=False
    )

    tuner.optimize(train_func=mock_train_function)

    # Get parameter importance
    importance = tuner.get_parameter_importance(n_top=3)
    assert isinstance(importance, dict)
    assert len(importance) <= 3


# Integration test with full pipeline

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not available")
def test_full_optimization_pipeline(tmp_path):
    """Test complete optimization pipeline"""
    # Create search space
    search_space = create_custom_search_space(base_space='compact')

    # Create multi-objective
    objective = create_balanced_objective()

    # Create tuner
    tuner = HyperparameterTuner(
        objective=objective,
        search_space=search_space,
        n_trials=10,
        sampler='tpe',
        pruner='median',
        output_dir=str(tmp_path),
        verbose=False
    )

    # Mock training function
    def train_func(params, train_data=None, val_data=None):
        returns = np.random.randn(100) * 0.02
        balance_history = [10000]
        for ret in returns:
            balance_history.append(balance_history[-1] * (1 + ret))

        return {
            'sharpe_ratio': np.random.uniform(-1, 3),
            'total_return': (balance_history[-1] - 10000) / 10000,
            'max_drawdown': np.random.uniform(-0.3, -0.05),
            'final_balance': balance_history[-1],
            'initial_balance': 10000,
            'returns': returns.tolist(),
            'balance_history': balance_history,
        }

    # Run optimization
    results = tuner.optimize(train_func=train_func)

    # Verify results
    assert results['best_params'] is not None
    assert results['best_value'] is not None
    assert len(results['optimization_history']) == 10

    # Generate report
    report = tuner.generate_report(output_path=str(tmp_path / 'report.txt'))
    assert (tmp_path / 'report.txt').exists()

    # Check parameter validation
    is_valid, errors = validate_hyperparameters(results['best_params'])
    assert is_valid, f"Best parameters are invalid: {errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
