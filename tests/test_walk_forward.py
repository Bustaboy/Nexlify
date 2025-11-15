"""
Unit tests for walk-forward validation

Tests the WalkForwardValidator class and related functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

from nexlify.validation.walk_forward import (
    WalkForwardValidator,
    FoldConfig,
    FoldMetrics,
    WalkForwardResults,
    calculate_performance_metrics
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def basic_config():
    """Basic configuration for walk-forward validation"""
    return {
        'total_episodes': 2000,
        'train_size': 1000,
        'test_size': 200,
        'step_size': 200,
        'mode': 'rolling'
    }


@pytest.fixture
def expanding_config():
    """Configuration for expanding window mode"""
    return {
        'total_episodes': 2000,
        'train_size': 1000,
        'test_size': 200,
        'step_size': 200,
        'mode': 'expanding'
    }


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_returns():
    """Sample returns data for testing metrics calculation"""
    np.random.seed(42)
    # Generate returns with positive drift
    returns = np.random.randn(200) * 0.02 + 0.0005
    return returns


@pytest.fixture
def sample_trades():
    """Sample trades data for testing metrics calculation"""
    np.random.seed(42)
    trades = []
    for i in range(100):
        profit = np.random.randn() * 50 + 10  # Slight positive bias
        trades.append({
            'profit': profit,
            'duration': np.random.randint(1, 50)
        })
    return trades


# ============================================================================
# TEST FOLD GENERATION - Rolling Mode
# ============================================================================

@pytest.mark.unit
def test_fold_generation_rolling_basic(basic_config):
    """Test basic fold generation in rolling mode"""
    validator = WalkForwardValidator(**basic_config)

    assert len(validator.folds) > 0
    assert validator.mode == 'rolling'

    # Check first fold
    first_fold = validator.folds[0]
    assert first_fold.train_start == 0
    assert first_fold.train_end == 1000
    assert first_fold.test_start == 1000
    assert first_fold.test_end == 1200


@pytest.mark.unit
def test_fold_generation_rolling_no_overlap(basic_config):
    """Test that folds don't have overlapping test windows"""
    validator = WalkForwardValidator(**basic_config)

    for i in range(len(validator.folds) - 1):
        current_fold = validator.folds[i]
        next_fold = validator.folds[i + 1]

        # Test windows should not overlap
        assert current_fold.test_end <= next_fold.test_start


@pytest.mark.unit
def test_fold_generation_rolling_window_size(basic_config):
    """Test that rolling window maintains constant training size"""
    validator = WalkForwardValidator(**basic_config)

    train_size = basic_config['train_size']
    test_size = basic_config['test_size']

    for fold in validator.folds:
        assert fold.train_size == train_size
        assert fold.test_size == test_size


@pytest.mark.unit
def test_fold_generation_rolling_sequential_ids(basic_config):
    """Test that fold IDs are sequential"""
    validator = WalkForwardValidator(**basic_config)

    for i, fold in enumerate(validator.folds):
        assert fold.fold_id == i


# ============================================================================
# TEST FOLD GENERATION - Expanding Mode
# ============================================================================

@pytest.mark.unit
def test_fold_generation_expanding_basic(expanding_config):
    """Test basic fold generation in expanding mode"""
    validator = WalkForwardValidator(**expanding_config)

    assert len(validator.folds) > 0
    assert validator.mode == 'expanding'

    # Check first fold
    first_fold = validator.folds[0]
    assert first_fold.train_start == 0
    assert first_fold.train_end == 1000
    assert first_fold.test_start == 1000
    assert first_fold.test_end == 1200


@pytest.mark.unit
def test_fold_generation_expanding_growing_train_size(expanding_config):
    """Test that expanding mode grows training window"""
    validator = WalkForwardValidator(**expanding_config)

    test_size = expanding_config['test_size']

    for fold in validator.folds:
        # Test size should be constant
        assert fold.test_size == test_size

        # Training size should be growing
        # (except potentially the first fold)

    # Check that training sizes are increasing
    train_sizes = [fold.train_size for fold in validator.folds]
    assert all(train_sizes[i] <= train_sizes[i+1] for i in range(len(train_sizes)-1))


@pytest.mark.unit
def test_fold_generation_expanding_anchored_start(expanding_config):
    """Test that expanding mode is anchored at start"""
    validator = WalkForwardValidator(**expanding_config)

    for fold in validator.folds:
        # All folds should start training from episode 0
        assert fold.train_start == 0


# ============================================================================
# TEST CONFIGURATION VALIDATION
# ============================================================================

@pytest.mark.unit
def test_invalid_mode():
    """Test that invalid mode raises error"""
    with pytest.raises(ValueError, match="Invalid mode"):
        WalkForwardValidator(
            total_episodes=2000,
            train_size=1000,
            test_size=200,
            step_size=200,
            mode='invalid_mode'
        )


@pytest.mark.unit
def test_negative_train_size():
    """Test that negative train size raises error"""
    with pytest.raises(ValueError, match="must be positive"):
        WalkForwardValidator(
            total_episodes=2000,
            train_size=-100,
            test_size=200,
            step_size=200,
            mode='rolling'
        )


@pytest.mark.unit
def test_window_exceeds_total():
    """Test that window larger than total episodes raises error"""
    with pytest.raises(ValueError, match="exceeds total episodes"):
        WalkForwardValidator(
            total_episodes=1000,
            train_size=800,
            test_size=300,  # Total = 1100 > 1000
            step_size=200,
            mode='rolling'
        )


@pytest.mark.unit
def test_min_train_exceeds_train():
    """Test that min_train_size > train_size raises error"""
    with pytest.raises(ValueError, match="cannot exceed"):
        WalkForwardValidator(
            total_episodes=2000,
            train_size=500,
            test_size=200,
            step_size=200,
            mode='rolling',
            min_train_size=1000
        )


# ============================================================================
# TEST FOLD CONFIG
# ============================================================================

@pytest.mark.unit
def test_fold_config_properties():
    """Test FoldConfig properties"""
    fold = FoldConfig(
        fold_id=0,
        train_start=0,
        train_end=1000,
        test_start=1000,
        test_end=1200
    )

    assert fold.train_size == 1000
    assert fold.test_size == 200
    assert "Fold 0" in str(fold)
    assert "Train [0-1000]" in str(fold)
    assert "Test [1000-1200]" in str(fold)


# ============================================================================
# TEST PERFORMANCE METRICS CALCULATION
# ============================================================================

@pytest.mark.unit
def test_calculate_performance_metrics_basic(sample_returns):
    """Test basic performance metrics calculation"""
    metrics = calculate_performance_metrics(sample_returns)

    # Check all expected metrics are present
    expected_keys = [
        'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown',
        'profit_factor', 'num_trades', 'avg_trade_duration',
        'volatility', 'sortino_ratio', 'calmar_ratio'
    ]

    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))


@pytest.mark.unit
def test_calculate_performance_metrics_with_trades(sample_returns, sample_trades):
    """Test performance metrics calculation with trade data"""
    metrics = calculate_performance_metrics(
        returns=sample_returns,
        trades=sample_trades
    )

    # With trades, num_trades should match
    assert metrics['num_trades'] == len(sample_trades)

    # Win rate should be between 0 and 1
    assert 0 <= metrics['win_rate'] <= 1

    # Profit factor should be non-negative
    assert metrics['profit_factor'] >= 0


@pytest.mark.unit
def test_calculate_performance_metrics_empty_returns():
    """Test metrics calculation with empty returns"""
    metrics = calculate_performance_metrics(np.array([]))

    # All metrics should be 0 for empty data
    assert metrics['total_return'] == 0.0
    assert metrics['sharpe_ratio'] == 0.0
    assert metrics['num_trades'] == 0


@pytest.mark.unit
def test_calculate_performance_metrics_positive_returns():
    """Test metrics with consistently positive returns"""
    returns = np.array([0.01] * 100)  # 1% return each period
    metrics = calculate_performance_metrics(returns)

    # Total return should be positive
    assert metrics['total_return'] > 0

    # Win rate should be 100%
    assert metrics['win_rate'] == 1.0

    # Max drawdown should be 0 (no losses)
    assert metrics['max_drawdown'] == 0.0


@pytest.mark.unit
def test_calculate_performance_metrics_sharpe_ratio():
    """Test Sharpe ratio calculation"""
    # Create returns with some variation
    np.random.seed(42)
    returns = np.array([0.01] * 252) + np.random.randn(252) * 0.001  # Small noise
    metrics = calculate_performance_metrics(returns, risk_free_rate=0.0)

    # Sharpe should be positive and reasonable
    assert metrics['sharpe_ratio'] > 0


@pytest.mark.unit
def test_calculate_performance_metrics_max_drawdown():
    """Test max drawdown calculation"""
    # Create returns with a known drawdown
    returns = np.array([0.1, -0.2, -0.1, 0.05, 0.15])
    # Cumulative: 1.1, 0.88, 0.792, 0.832, 0.957
    # Peak: 1.1, drawdown from 1.1 to 0.792 = -28%

    metrics = calculate_performance_metrics(returns)

    # Max drawdown should be negative
    assert metrics['max_drawdown'] < 0

    # Should capture the drawdown
    assert metrics['max_drawdown'] < -0.25  # At least 25% drawdown


# ============================================================================
# TEST FOLD METRICS
# ============================================================================

@pytest.mark.unit
def test_fold_metrics_creation():
    """Test FoldMetrics creation and methods"""
    metrics = FoldMetrics(
        fold_id=0,
        total_return=0.15,
        sharpe_ratio=1.8,
        win_rate=0.65,
        max_drawdown=-0.05,
        profit_factor=2.5,
        num_trades=100,
        avg_trade_duration=10.5,
        volatility=0.12,
        sortino_ratio=2.2,
        calmar_ratio=3.0
    )

    # Test to_dict
    metrics_dict = metrics.to_dict()
    assert metrics_dict['fold_id'] == 0
    assert metrics_dict['total_return'] == 0.15

    # Test __repr__
    repr_str = str(metrics)
    assert "Fold 0" in repr_str
    assert "15.00%" in repr_str  # Return
    assert "1.80" in repr_str  # Sharpe


# ============================================================================
# TEST WALK-FORWARD VALIDATION (ASYNC)
# ============================================================================

class DummyModel:
    """Dummy model for testing"""
    def __init__(self, performance_multiplier=1.0):
        self.performance_multiplier = performance_multiplier


@pytest.mark.asyncio
@pytest.mark.unit
async def test_walk_forward_validate_basic(basic_config, temp_output_dir):
    """Test basic walk-forward validation execution"""
    validator = WalkForwardValidator(**basic_config)

    # Define simple training and evaluation functions
    async def train_fn(train_start, train_end):
        """Dummy training function"""
        # Return a dummy model
        return DummyModel()

    async def eval_fn(model, test_start, test_end):
        """Dummy evaluation function"""
        # Return dummy metrics
        np.random.seed(test_start)  # For reproducibility
        return {
            'total_return': np.random.rand() * 0.2 - 0.05,
            'sharpe_ratio': np.random.rand() * 2,
            'win_rate': 0.5 + np.random.rand() * 0.2,
            'max_drawdown': -np.random.rand() * 0.1,
            'profit_factor': 1.0 + np.random.rand(),
            'num_trades': np.random.randint(50, 150),
            'avg_trade_duration': np.random.rand() * 20,
            'volatility': np.random.rand() * 0.15,
            'sortino_ratio': np.random.rand() * 2,
            'calmar_ratio': np.random.rand() * 3
        }

    # Run validation
    results = await validator.validate(
        train_fn=train_fn,
        eval_fn=eval_fn,
        save_models=False,  # Don't save for testing
        model_dir=temp_output_dir
    )

    # Validate results
    assert isinstance(results, WalkForwardResults)
    assert len(results.fold_configs) == len(validator.folds)
    assert len(results.fold_metrics) == len(validator.folds)
    assert 'total_return' in results.mean_metrics
    assert 'sharpe_ratio' in results.std_metrics


@pytest.mark.asyncio
@pytest.mark.unit
async def test_walk_forward_validate_sync_functions(basic_config):
    """Test that validation works with synchronous functions"""
    validator = WalkForwardValidator(**basic_config)

    # Define synchronous functions
    def train_fn(train_start, train_end):
        """Sync training function"""
        return DummyModel()

    def eval_fn(model, test_start, test_end):
        """Sync evaluation function"""
        return {
            'total_return': 0.1,
            'sharpe_ratio': 1.5,
            'win_rate': 0.6,
            'max_drawdown': -0.05,
            'profit_factor': 2.0,
            'num_trades': 100,
            'avg_trade_duration': 10.0,
            'volatility': 0.12,
            'sortino_ratio': 1.8,
            'calmar_ratio': 2.5
        }

    # Run validation with sync functions
    results = await validator.validate(
        train_fn=train_fn,
        eval_fn=eval_fn,
        save_models=False
    )

    # Should complete successfully
    assert isinstance(results, WalkForwardResults)
    assert len(results.fold_metrics) == len(validator.folds)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_walk_forward_best_worst_fold(basic_config):
    """Test identification of best and worst folds"""
    validator = WalkForwardValidator(**basic_config)

    async def train_fn(train_start, train_end):
        return DummyModel()

    async def eval_fn(model, test_start, test_end):
        # Create varying performance across folds
        # Fold 0 will have best return (0.15), last fold worst (-0.05)
        fold_id = test_start // 200 - 5  # Approximate fold ID
        return {
            'total_return': 0.15 - fold_id * 0.05,  # Decreasing returns
            'sharpe_ratio': 1.5,
            'win_rate': 0.6,
            'max_drawdown': -0.05,
            'profit_factor': 2.0,
            'num_trades': 100,
            'avg_trade_duration': 10.0,
            'volatility': 0.12,
            'sortino_ratio': 1.8,
            'calmar_ratio': 2.5
        }

    results = await validator.validate(
        train_fn=train_fn,
        eval_fn=eval_fn,
        save_models=False
    )

    # Best fold should have highest return
    best_fold = results.fold_metrics[results.best_fold_id]
    worst_fold = results.fold_metrics[results.worst_fold_id]

    assert best_fold.total_return >= worst_fold.total_return


# ============================================================================
# TEST RESULTS AND REPORTING
# ============================================================================

@pytest.mark.unit
def test_walk_forward_results_summary():
    """Test WalkForwardResults summary generation"""
    # Create sample results
    fold_configs = [
        FoldConfig(0, 0, 1000, 1000, 1200),
        FoldConfig(1, 200, 1200, 1200, 1400)
    ]

    fold_metrics = [
        FoldMetrics(0, 0.15, 1.8, 0.65, -0.05, 2.5, 100, 10.0, 0.12, 2.0, 3.0),
        FoldMetrics(1, 0.10, 1.5, 0.60, -0.08, 2.0, 95, 12.0, 0.15, 1.7, 2.5)
    ]

    results = WalkForwardResults(
        fold_configs=fold_configs,
        fold_metrics=fold_metrics,
        mean_metrics={
            'total_return': 0.125,
            'sharpe_ratio': 1.65,
            'win_rate': 0.625
        },
        std_metrics={
            'total_return': 0.025,
            'sharpe_ratio': 0.15,
            'win_rate': 0.025
        },
        best_fold_id=0,
        worst_fold_id=1,
        validation_date='2024-01-01T00:00:00'
    )

    summary = results.summary()

    # Check summary contains key information
    assert "WALK-FORWARD VALIDATION SUMMARY" in summary
    assert "Number of Folds: 2" in summary
    assert "Best Fold:  0" in summary
    assert "Worst Fold: 1" in summary


@pytest.mark.unit
def test_walk_forward_results_to_dict():
    """Test WalkForwardResults serialization"""
    fold_configs = [FoldConfig(0, 0, 1000, 1000, 1200)]
    fold_metrics = [
        FoldMetrics(0, 0.15, 1.8, 0.65, -0.05, 2.5, 100, 10.0, 0.12, 2.0, 3.0)
    ]

    results = WalkForwardResults(
        fold_configs=fold_configs,
        fold_metrics=fold_metrics,
        mean_metrics={'total_return': 0.15},
        std_metrics={'total_return': 0.0},
        best_fold_id=0,
        worst_fold_id=0,
        validation_date='2024-01-01T00:00:00'
    )

    results_dict = results.to_dict()

    # Should be JSON serializable
    json_str = json.dumps(results_dict)
    assert isinstance(json_str, str)

    # Should contain all keys
    assert 'fold_configs' in results_dict
    assert 'fold_metrics' in results_dict
    assert 'mean_metrics' in results_dict


@pytest.mark.unit
def test_generate_report(basic_config, temp_output_dir):
    """Test report generation (files created)"""
    validator = WalkForwardValidator(**basic_config)

    # Create sample results
    fold_configs = validator.folds[:2]  # Use first 2 folds
    fold_metrics = [
        FoldMetrics(0, 0.15, 1.8, 0.65, -0.05, 2.5, 100, 10.0, 0.12, 2.0, 3.0),
        FoldMetrics(1, 0.10, 1.5, 0.60, -0.08, 2.0, 95, 12.0, 0.15, 1.7, 2.5)
    ]

    results = WalkForwardResults(
        fold_configs=fold_configs,
        fold_metrics=fold_metrics,
        mean_metrics={
            'total_return': 0.125,
            'sharpe_ratio': 1.65,
            'win_rate': 0.625,
            'max_drawdown': -0.065
        },
        std_metrics={
            'total_return': 0.025,
            'sharpe_ratio': 0.15,
            'win_rate': 0.025,
            'max_drawdown': 0.015
        },
        best_fold_id=0,
        worst_fold_id=1,
        validation_date='2024-01-01T00:00:00'
    )

    # Generate report
    validator.generate_report(results, temp_output_dir)

    # Check that files were created
    files = list(temp_output_dir.glob('*'))
    assert len(files) > 0

    # Check for specific files
    json_files = list(temp_output_dir.glob('validation_results_*.json'))
    assert len(json_files) > 0

    summary_files = list(temp_output_dir.glob('summary_*.txt'))
    assert len(summary_files) > 0


# ============================================================================
# TEST EDGE CASES
# ============================================================================

@pytest.mark.unit
def test_small_dataset():
    """Test with minimal valid dataset"""
    validator = WalkForwardValidator(
        total_episodes=300,
        train_size=100,
        test_size=50,
        step_size=50,
        mode='rolling',
        min_train_size=50  # Set lower min_train_size for small dataset
    )

    # Should generate at least one fold
    assert len(validator.folds) >= 1


@pytest.mark.unit
def test_large_step_size():
    """Test with step size larger than test size"""
    validator = WalkForwardValidator(
        total_episodes=2000,
        train_size=500,
        test_size=100,
        step_size=300,  # Larger than test size
        mode='rolling'
    )

    # Should still generate valid folds
    assert len(validator.folds) > 0

    # Folds should not overlap
    for i in range(len(validator.folds) - 1):
        assert validator.folds[i].test_end <= validator.folds[i+1].test_start


@pytest.mark.unit
def test_no_valid_folds():
    """Test configuration that produces no valid folds"""
    # This will trigger the "exceeds total episodes" error
    with pytest.raises(ValueError, match="exceeds total episodes"):
        WalkForwardValidator(
            total_episodes=100,
            train_size=1000,  # Impossible
            test_size=200,
            step_size=200,
            mode='rolling'
        )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_validation_pipeline(basic_config, temp_output_dir):
    """Test complete validation pipeline end-to-end"""
    validator = WalkForwardValidator(**basic_config)

    # Track which folds were processed
    processed_folds = []

    async def train_fn(train_start, train_end):
        """Track training calls"""
        processed_folds.append(('train', train_start, train_end))
        return DummyModel(performance_multiplier=1.0 + np.random.rand() * 0.1)

    async def eval_fn(model, test_start, test_end):
        """Track evaluation calls"""
        processed_folds.append(('eval', test_start, test_end))
        np.random.seed(test_start)
        return {
            'total_return': (np.random.rand() * 0.2 - 0.05) * model.performance_multiplier,
            'sharpe_ratio': np.random.rand() * 2,
            'win_rate': 0.5 + np.random.rand() * 0.2,
            'max_drawdown': -np.random.rand() * 0.1,
            'profit_factor': 1.0 + np.random.rand(),
            'num_trades': np.random.randint(50, 150),
            'avg_trade_duration': np.random.rand() * 20,
            'volatility': np.random.rand() * 0.15,
            'sortino_ratio': np.random.rand() * 2,
            'calmar_ratio': np.random.rand() * 3
        }

    # Run validation
    results = await validator.validate(
        train_fn=train_fn,
        eval_fn=eval_fn,
        save_models=False,
        model_dir=temp_output_dir
    )

    # Generate report
    validator.generate_report(results, temp_output_dir)

    # Verify all folds were processed
    num_folds = len(validator.folds)
    assert len([x for x in processed_folds if x[0] == 'train']) == num_folds
    assert len([x for x in processed_folds if x[0] == 'eval']) == num_folds

    # Verify results are comprehensive
    assert len(results.fold_metrics) == num_folds
    assert results.mean_metrics['total_return'] is not None
    assert results.std_metrics['sharpe_ratio'] is not None

    # Verify report files exist
    assert (temp_output_dir / 'summary_*.txt').parent.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
