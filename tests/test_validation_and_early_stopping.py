#!/usr/bin/env python3
"""
Tests for Validation Monitoring and Early Stopping System
==========================================================
Comprehensive tests for validation data splitting, validation monitoring,
early stopping, training phase detection, and overfitting detection.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from nexlify.training.validation_monitor import (
    ValidationMonitor,
    ValidationDataSplitter,
    ValidationResult,
    DataSplit
)

from nexlify.training.early_stopping import (
    EarlyStopping,
    EarlyStoppingConfig,
    TrainingPhaseDetector,
    TrainingPhase,
    OverfittingDetector
)


class TestValidationDataSplitter:
    """Tests for ValidationDataSplitter"""

    def test_splitter_initialization(self):
        """Test basic initialization"""
        splitter = ValidationDataSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )

        assert splitter.train_ratio == 0.70
        assert splitter.val_ratio == 0.15
        assert splitter.test_ratio == 0.15

    def test_splitter_invalid_ratios(self):
        """Test that invalid ratios raise ValueError"""
        # Ratios don't sum to 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ValidationDataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.3)

        # Negative ratio
        with pytest.raises(ValueError, match="must be positive"):
            ValidationDataSplitter(train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)

    def test_split_temporal_ordering(self):
        """Test that data is split with temporal ordering preserved"""
        data = np.arange(1000)
        splitter = ValidationDataSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )

        split = splitter.split(data)

        # Check sizes
        assert len(split.train_data) == 700
        assert len(split.val_data) == 150
        assert len(split.test_data) == 150

        # Check temporal ordering (train < val < test)
        assert split.train_data[-1] < split.val_data[0]
        assert split.val_data[-1] < split.test_data[0]

        # Check no overlap
        assert np.all(split.train_data == data[:700])
        assert np.all(split.val_data == data[700:850])
        assert np.all(split.test_data == data[850:])

    def test_split_minimum_samples(self):
        """Test that split fails if data is too small"""
        splitter = ValidationDataSplitter(min_samples_per_split=100)
        small_data = np.arange(200)  # Too small for 3 splits of 100

        with pytest.raises(ValueError, match="too small for splitting"):
            splitter.split(small_data)

    def test_split_metadata(self):
        """Test that split metadata is correct"""
        data = np.arange(1000)
        splitter = ValidationDataSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )

        split = splitter.split(data)

        assert split.total_size == 1000
        assert split.train_indices == (0, 700)
        assert split.val_indices == (700, 850)
        assert split.test_indices == (850, 1000)
        assert split.split_ratios == (0.70, 0.15, 0.15)


class TestValidationMonitor:
    """Tests for ValidationMonitor"""

    def test_monitor_initialization(self):
        """Test basic initialization"""
        monitor = ValidationMonitor(
            validation_frequency=50,
            cache_results=True
        )

        assert monitor.validation_frequency == 50
        assert monitor.cache_results is True
        assert len(monitor.validation_results) == 0

    def test_should_validate(self):
        """Test validation frequency logic"""
        monitor = ValidationMonitor(validation_frequency=50)

        # First episode should always validate
        assert monitor.should_validate(0) is True

        # Should validate at frequency intervals
        monitor.last_validation_episode = 0
        assert monitor.should_validate(50) is True
        assert monitor.should_validate(25) is False

        monitor.last_validation_episode = 50
        assert monitor.should_validate(100) is True
        assert monitor.should_validate(75) is False

    def test_run_validation(self):
        """Test validation run"""
        # Create mock agent and environment
        mock_agent = Mock()
        mock_agent.act = Mock(return_value=0)  # Always hold

        mock_env = Mock()
        mock_env.reset = Mock(return_value=np.zeros(12))
        mock_env.step = Mock(return_value=(
            np.zeros(12),  # next_state
            0.1,  # reward
            False,  # done
            {'trade_executed': False}  # info
        ))
        mock_env._get_current_equity = Mock(return_value=11000)
        mock_env.initial_balance = 10000
        mock_env.equity_curve = [10000, 10500, 11000]
        mock_env.total_trades = 10
        mock_env.winning_trades = 6

        # Override step to simulate episode
        def step_side_effect(*args, **kwargs):
            # Increment internal counter (simulate episode progress)
            if not hasattr(mock_env, 'step_count'):
                mock_env.step_count = 0
            mock_env.step_count += 1

            # End episode after 10 steps
            done = mock_env.step_count >= 10
            if done:
                mock_env.step_count = 0  # Reset for next episode

            return (
                np.zeros(12),
                0.1,
                done,
                {'trade_executed': False}
            )

        mock_env.step = Mock(side_effect=step_side_effect)

        monitor = ValidationMonitor(validation_frequency=50)

        # Run validation
        val_result = monitor.run_validation(
            agent=mock_agent,
            val_env=mock_env,
            current_episode=50,
            num_episodes=1  # Just 1 episode for testing
        )

        # Check result
        assert isinstance(val_result, ValidationResult)
        assert val_result.episode == 50
        assert val_result.val_final_equity == 11000
        assert len(monitor.validation_results) == 1

    def test_update_best(self):
        """Test best validation tracking"""
        monitor = ValidationMonitor()

        # Create validation results
        result1 = ValidationResult(
            episode=50,
            val_return=500,
            val_return_pct=5.0,
            val_sharpe=0.5,
            val_win_rate=60.0,
            val_max_drawdown=-10.0,
            val_num_trades=10,
            val_final_equity=10500,
            timestamp="2025-01-01T00:00:00"
        )

        result2 = ValidationResult(
            episode=100,
            val_return=800,
            val_return_pct=8.0,
            val_sharpe=0.8,  # Better
            val_win_rate=65.0,
            val_max_drawdown=-8.0,
            val_num_trades=12,
            val_final_equity=10800,
            timestamp="2025-01-01T01:00:00"
        )

        # Update with first result
        is_best = monitor.update_best(result1, metric='val_sharpe')
        assert is_best is True
        assert monitor.best_val_result == result1

        # Update with better result
        is_best = monitor.update_best(result2, metric='val_sharpe')
        assert is_best is True
        assert monitor.best_val_result == result2

        # Update with worse result
        is_best = monitor.update_best(result1, metric='val_sharpe')
        assert is_best is False
        assert monitor.best_val_result == result2  # Should still be result2


class TestTrainingPhaseDetector:
    """Tests for TrainingPhaseDetector"""

    def test_phase_detection(self):
        """Test training phase detection based on epsilon"""
        detector = TrainingPhaseDetector(
            exploration_threshold=0.7,
            learning_threshold=0.3
        )

        # Exploration phase (epsilon > 0.7)
        phase = detector.detect_phase(epsilon=0.9, episode=10)
        assert phase == TrainingPhase.EXPLORATION

        # Learning phase (0.3 < epsilon <= 0.7)
        phase = detector.detect_phase(epsilon=0.5, episode=50)
        assert phase == TrainingPhase.LEARNING

        # Exploitation phase (epsilon <= 0.3)
        phase = detector.detect_phase(epsilon=0.1, episode=100)
        assert phase == TrainingPhase.EXPLOITATION

    def test_patience_multiplier(self):
        """Test patience multiplier for different phases"""
        detector = TrainingPhaseDetector()
        config = EarlyStoppingConfig(
            exploration_patience_multiplier=2.0,
            learning_patience_multiplier=1.5,
            exploitation_patience_multiplier=1.0
        )

        # Exploration
        multiplier = detector.get_patience_multiplier(TrainingPhase.EXPLORATION, config)
        assert multiplier == 2.0

        # Learning
        multiplier = detector.get_patience_multiplier(TrainingPhase.LEARNING, config)
        assert multiplier == 1.5

        # Exploitation
        multiplier = detector.get_patience_multiplier(TrainingPhase.EXPLOITATION, config)
        assert multiplier == 1.0


class TestOverfittingDetector:
    """Tests for OverfittingDetector"""

    def test_overfitting_detection(self):
        """Test overfitting detection logic"""
        detector = OverfittingDetector(
            overfitting_threshold=0.20,
            window_size=5
        )

        # No overfitting (train and val similar)
        is_overfitting, score = detector.update(
            train_metric=0.5,
            val_metric=0.48,
            episode=10
        )

        assert is_overfitting is False
        assert score < 0.20

        # Overfitting (train >> val)
        is_overfitting, score = detector.update(
            train_metric=0.8,
            val_metric=0.5,
            episode=11
        )

        assert is_overfitting is True
        assert score > 0.20

    def test_chronic_overfitting_detection(self):
        """Test chronic overfitting detection over window"""
        detector = OverfittingDetector(
            overfitting_threshold=0.20,
            window_size=3
        )

        # Simulate chronic overfitting over multiple episodes
        for i in range(10):
            detector.update(
                train_metric=0.8,
                val_metric=0.5,  # Consistent gap
                episode=i
            )

        # Check chronic overfitting was detected
        assert detector.chronic_overfitting_count >= 3


class TestEarlyStopping:
    """Tests for EarlyStopping"""

    def test_early_stopping_initialization(self):
        """Test basic initialization"""
        config = EarlyStoppingConfig(
            patience=30,
            min_delta=0.01,
            mode='max',
            metric='val_sharpe'
        )

        early_stopping = EarlyStopping(config=config)

        assert early_stopping.config.patience == 30
        assert early_stopping.config.min_delta == 0.01
        assert early_stopping.stopped is False

    def test_early_stopping_improvement_detection(self):
        """Test improvement detection logic"""
        config = EarlyStoppingConfig(
            patience=3,
            min_delta=0.01,
            mode='max',
            metric='val_sharpe',
            min_episodes=0  # Disable min episodes for testing
        )

        early_stopping = EarlyStopping(config=config)

        # First update (improvement)
        should_stop = early_stopping.update(
            metric_value=0.5,
            episode=1,
            epsilon=0.9
        )

        assert should_stop is False
        assert early_stopping.patience_counter == 0
        assert early_stopping.best_metric == 0.5

        # Second update (improvement)
        should_stop = early_stopping.update(
            metric_value=0.6,
            episode=2,
            epsilon=0.8
        )

        assert should_stop is False
        assert early_stopping.patience_counter == 0
        assert early_stopping.best_metric == 0.6

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers after patience exhausted"""
        config = EarlyStoppingConfig(
            patience=3,
            min_delta=0.01,
            mode='max',
            metric='val_sharpe',
            min_episodes=0
        )

        early_stopping = EarlyStopping(config=config)

        # First update (set baseline)
        early_stopping.update(metric_value=0.5, episode=1, epsilon=0.9)

        # No improvement for patience episodes
        early_stopping.update(metric_value=0.49, episode=2, epsilon=0.8)
        early_stopping.update(metric_value=0.48, episode=3, epsilon=0.7)
        should_stop = early_stopping.update(metric_value=0.47, episode=4, epsilon=0.6)

        assert should_stop is True
        assert early_stopping.stopped is True
        assert early_stopping.patience_counter >= config.patience

    def test_adaptive_patience_by_phase(self):
        """Test that patience adapts by training phase"""
        config = EarlyStoppingConfig(
            patience=10,
            min_delta=0.01,
            mode='max',
            metric='val_sharpe',
            min_episodes=0,
            exploration_patience_multiplier=2.0,
            learning_patience_multiplier=1.5,
            exploitation_patience_multiplier=1.0
        )

        early_stopping = EarlyStopping(config=config)

        # Exploration phase (epsilon = 0.9)
        early_stopping.update(metric_value=0.5, episode=1, epsilon=0.9)
        assert early_stopping.current_patience == 20  # 10 * 2.0

        # Learning phase (epsilon = 0.5)
        early_stopping.update(metric_value=0.5, episode=2, epsilon=0.5)
        assert early_stopping.current_patience == 15  # 10 * 1.5

        # Exploitation phase (epsilon = 0.1)
        early_stopping.update(metric_value=0.5, episode=3, epsilon=0.1)
        assert early_stopping.current_patience == 10  # 10 * 1.0

    def test_min_episodes_constraint(self):
        """Test that early stopping respects min_episodes"""
        config = EarlyStoppingConfig(
            patience=3,
            min_delta=0.01,
            mode='max',
            metric='val_sharpe',
            min_episodes=100
        )

        early_stopping = EarlyStopping(config=config)

        # Try to trigger early stopping before min_episodes
        early_stopping.update(metric_value=0.5, episode=1, epsilon=0.9)
        for i in range(10):
            should_stop = early_stopping.update(
                metric_value=0.49,
                episode=i + 2,
                epsilon=0.8
            )
            # Should not stop before episode 100
            if i + 2 < 100:
                assert should_stop is False

    def test_mode_minimize(self):
        """Test early stopping with mode='min' (for loss-like metrics)"""
        config = EarlyStoppingConfig(
            patience=3,
            min_delta=0.01,
            mode='min',  # Minimize
            metric='val_loss',
            min_episodes=0
        )

        early_stopping = EarlyStopping(config=config)
        early_stopping.best_metric = np.inf  # Initialize for min mode

        # Lower is better
        should_stop = early_stopping.update(metric_value=0.5, episode=1, epsilon=0.9)
        assert should_stop is False
        assert early_stopping.best_metric == 0.5

        # Improvement (lower value)
        should_stop = early_stopping.update(metric_value=0.3, episode=2, epsilon=0.8)
        assert should_stop is False
        assert early_stopping.best_metric == 0.3
        assert early_stopping.patience_counter == 0


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_validation_workflow(self):
        """Test full workflow: split data, run validation, early stopping"""
        # Generate synthetic data
        price_data = np.random.randn(1000) + 50000

        # Split data
        splitter = ValidationDataSplitter()
        split = splitter.split(price_data)

        assert len(split.train_data) > 0
        assert len(split.val_data) > 0
        assert len(split.test_data) > 0

        # Create validation monitor
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = ValidationMonitor(
                validation_frequency=50,
                save_dir=Path(tmpdir)
            )

            assert monitor.validation_frequency == 50

        # Create early stopping
        config = EarlyStoppingConfig(
            patience=30,
            metric='val_sharpe',
            mode='max'
        )

        early_stopping = EarlyStopping(config=config)

        assert early_stopping.stopped is False

    def test_overfitting_detection_integration(self):
        """Test overfitting detector with early stopping"""
        detector = OverfittingDetector(overfitting_threshold=0.20)
        config = EarlyStoppingConfig(patience=30, metric='val_sharpe', min_episodes=0)
        early_stopping = EarlyStopping(
            config=config,
            overfitting_detector=detector
        )

        # Simulate training with overfitting
        for i in range(20):
            # Train metric keeps improving, val metric plateaus
            train_metric = 0.5 + i * 0.05
            val_metric = 0.5 + (i * 0.01 if i < 5 else 0)

            early_stopping.update(
                metric_value=val_metric,
                episode=i,
                epsilon=0.5,
                train_metric=train_metric
            )

        # Check overfitting was detected
        summary = detector.get_overfitting_summary()
        assert summary['avg_overfitting_score'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
