"""
Tests for Training Monitoring System
Comprehensive test coverage for metrics logger, dashboard, alerts, and experiment tracking
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from nexlify.monitoring.metrics_logger import MetricsLogger
from nexlify.monitoring.alert_system import AlertSystem, AlertThresholds
from nexlify.monitoring.experiment_tracker import ExperimentTracker


# ============================================================================
# MetricsLogger Tests
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def metrics_logger(temp_dir):
    """Create MetricsLogger instance"""
    return MetricsLogger(
        experiment_name="test_exp",
        output_dir=temp_dir,
        buffer_size=10,
        auto_save_interval=50,
        enable_async=False  # Disable for testing
    )


@pytest.mark.unit
def test_metrics_logger_initialization(metrics_logger):
    """Test MetricsLogger initialization"""
    assert metrics_logger.experiment_name == "test_exp"
    assert metrics_logger.buffer_size == 10
    assert metrics_logger.auto_save_interval == 50
    assert len(metrics_logger.episode_metrics) == 0
    assert len(metrics_logger.model_metrics) == 0


@pytest.mark.unit
def test_log_episode(metrics_logger):
    """Test logging episode metrics"""
    metrics_logger.log_episode(
        episode=1,
        profit=100.0,
        sharpe=1.5,
        win_rate=0.6,
        drawdown=50.0,
        num_trades=10,
        epsilon=0.5,
        learning_rate=0.001
    )

    assert len(metrics_logger.episode_metrics) == 1
    episode = metrics_logger.episode_metrics[0]

    assert episode['episode'] == 1
    assert episode['profit'] == 100.0
    assert episode['sharpe'] == 1.5
    assert episode['win_rate'] == 0.6
    assert episode['epsilon'] == 0.5


@pytest.mark.unit
def test_log_model_metrics(metrics_logger):
    """Test logging model metrics"""
    q_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    metrics_logger.log_model_metrics(
        loss=0.5,
        q_values=q_values,
        gradients={'mean': 0.01, 'std': 0.001}
    )

    assert len(metrics_logger.model_metrics) == 1
    model = metrics_logger.model_metrics[0]

    assert model['loss'] == 0.5
    assert model['q_values']['mean'] == 3.0  # Mean of [1,2,3,4,5]
    assert model['gradients']['mean'] == 0.01


@pytest.mark.unit
def test_get_smoothed_metric(metrics_logger):
    """Test smoothed metric calculation"""
    # Log multiple episodes
    for i in range(20):
        metrics_logger.log_episode(
            episode=i,
            profit=float(i * 10),
            sharpe=1.0
        )

    smoothed = metrics_logger.get_smoothed_metric('profit', window=5)
    assert smoothed is not None
    assert smoothed > 0


@pytest.mark.unit
def test_get_latest_episode(metrics_logger):
    """Test getting latest episode"""
    metrics_logger.log_episode(episode=1, profit=100.0)
    metrics_logger.log_episode(episode=2, profit=200.0)

    latest = metrics_logger.get_latest_episode()
    assert latest['episode'] == 2
    assert latest['profit'] == 200.0


@pytest.mark.unit
def test_get_best_episode(metrics_logger):
    """Test getting best episode by metric"""
    metrics_logger.log_episode(episode=1, profit=100.0, sharpe=1.0)
    metrics_logger.log_episode(episode=2, profit=200.0, sharpe=1.5)
    metrics_logger.log_episode(episode=3, profit=150.0, sharpe=2.0)

    best_profit = metrics_logger.get_best_episode('profit')
    assert best_profit['episode'] == 2
    assert best_profit['profit'] == 200.0

    best_sharpe = metrics_logger.get_best_episode('sharpe')
    assert best_sharpe['episode'] == 3
    assert best_sharpe['sharpe'] == 2.0


@pytest.mark.unit
def test_get_statistics(metrics_logger):
    """Test statistics calculation"""
    # Log episodes
    for i in range(10):
        metrics_logger.log_episode(
            episode=i,
            profit=float(i * 10),
            sharpe=1.0 + i * 0.1
        )

    stats = metrics_logger.get_statistics()

    assert stats['total_episodes'] == 10
    assert stats['best_profit'] == 90.0
    assert stats['best_sharpe'] == pytest.approx(1.9, rel=0.1)
    assert 'recent_avg_profit' in stats
    assert 'training_time' in stats


@pytest.mark.unit
def test_save_metrics_json(metrics_logger, temp_dir):
    """Test saving metrics to JSON"""
    metrics_logger.log_episode(episode=1, profit=100.0)
    metrics_logger.log_episode(episode=2, profit=200.0)

    filepath = metrics_logger.save_metrics(format='json')

    assert filepath.exists()
    assert filepath.suffix == '.json'

    # Verify content
    with open(filepath) as f:
        data = json.load(f)

    assert data['experiment_name'] == 'test_exp'
    assert len(data['episode_metrics']) == 2


@pytest.mark.unit
def test_save_metrics_csv(metrics_logger, temp_dir):
    """Test saving metrics to CSV"""
    metrics_logger.log_episode(episode=1, profit=100.0)
    metrics_logger.log_episode(episode=2, profit=200.0)

    filepath = metrics_logger.save_metrics(format='csv')

    assert filepath.exists()
    assert filepath.suffix == '.csv'

    # Verify file has content
    assert filepath.stat().st_size > 0


@pytest.mark.unit
def test_minimal_overhead(metrics_logger):
    """Test that logging has minimal overhead"""
    num_logs = 1000

    start = time.time()
    for i in range(num_logs):
        metrics_logger.log_episode(episode=i, profit=100.0)
    elapsed = time.time() - start

    # Should complete 1000 logs in under 100ms (< 0.1ms per log)
    assert elapsed < 0.1
    assert metrics_logger.get_statistics()['avg_log_time_ms'] < 1.0


# ============================================================================
# AlertSystem Tests
# ============================================================================

@pytest.fixture
def alert_config():
    """Alert system configuration"""
    return {
        'enable_alerts': True,
        'email': {
            'enabled': False
        },
        'slack': {
            'enabled': False
        },
        'thresholds': AlertThresholds.MODERATE
    }


@pytest.fixture
def alert_system(alert_config):
    """Create AlertSystem instance"""
    return AlertSystem(alert_config)


@pytest.mark.unit
def test_alert_system_initialization(alert_system):
    """Test AlertSystem initialization"""
    assert alert_system.enabled is True
    assert alert_system.email_enabled is False
    assert alert_system.slack_enabled is False


@pytest.mark.unit
def test_send_alert(alert_system):
    """Test sending alert (without actual email/slack)"""
    result = alert_system.send_alert(
        level='warning',
        title='Test Alert',
        message='This is a test',
        force=True
    )

    # Should succeed even without email/slack configured
    assert len(alert_system.alert_history) == 1
    alert = alert_system.alert_history[0]

    assert alert['level'] == 'warning'
    assert alert['title'] == 'Test Alert'
    assert alert['message'] == 'This is a test'


@pytest.mark.unit
def test_alert_throttling(alert_system):
    """Test alert throttling"""
    # First alert should succeed
    alert_system.send_alert(level='info', title='Test', message='Message 1')
    assert len(alert_system.alert_history) == 1

    # Second alert with same title should be throttled
    alert_system.send_alert(level='info', title='Test', message='Message 2')
    assert len(alert_system.alert_history) == 1  # Still 1

    # Force send should bypass throttling
    alert_system.send_alert(
        level='info',
        title='Test',
        message='Message 3',
        force=True
    )
    assert len(alert_system.alert_history) == 2


@pytest.mark.unit
def test_check_no_improvement(alert_system):
    """Test no improvement detection"""
    # Create history with no positive profits
    history = [
        {'episode': i, 'profit': -10.0}
        for i in range(100)
    ]

    alert_system.check_training_health(
        latest_episode=history[-1],
        recent_history=history
    )

    # Should generate warning alert
    alerts = [a for a in alert_system.alert_history if a['level'] == 'warning']
    assert len(alerts) > 0


@pytest.mark.unit
def test_check_critical_loss(alert_system):
    """Test critical loss detection"""
    episode = {
        'episode': 1,
        'loss': 2000.0  # Above threshold
    }

    alert_system._check_critical_loss(episode)

    # Should generate critical alert
    alerts = [a for a in alert_system.alert_history if a['level'] == 'critical']
    assert len(alerts) > 0


@pytest.mark.unit
def test_check_low_profit(alert_system):
    """Test low profit warning"""
    episode = {
        'episode': 1,
        'profit': -600.0  # Below threshold (-500)
    }

    alert_system._check_low_profit(episode)

    # Should generate warning
    alerts = [a for a in alert_system.alert_history if a['level'] == 'warning']
    assert len(alerts) > 0


@pytest.mark.unit
def test_send_training_complete(alert_system):
    """Test training complete notification"""
    alert_system.send_training_complete(
        total_episodes=1000,
        best_profit=5000.0,
        best_sharpe=2.5,
        training_time=3600.0
    )

    assert len(alert_system.alert_history) == 1
    alert = alert_system.alert_history[0]

    assert alert['level'] == 'info'
    assert alert['title'] == 'Training Complete'


@pytest.mark.unit
def test_get_alert_history(alert_system):
    """Test getting alert history"""
    alert_system.send_alert(level='info', title='Info', message='Info')
    alert_system.send_alert(level='warning', title='Warn', message='Warn', force=True)
    alert_system.send_alert(level='critical', title='Crit', message='Crit', force=True)

    # Get all alerts
    all_alerts = alert_system.get_alert_history()
    assert len(all_alerts) == 3

    # Get warnings only
    warnings = alert_system.get_alert_history(level='warning')
    assert len(warnings) == 1
    assert warnings[0]['level'] == 'warning'

    # Get last 2
    last_two = alert_system.get_alert_history(last_n=2)
    assert len(last_two) == 2


# ============================================================================
# ExperimentTracker Tests
# ============================================================================

@pytest.fixture
def experiment_tracker(temp_dir):
    """Create ExperimentTracker instance"""
    return ExperimentTracker(experiments_dir=temp_dir)


@pytest.mark.unit
def test_experiment_tracker_initialization(experiment_tracker):
    """Test ExperimentTracker initialization"""
    assert experiment_tracker.experiments_dir.exists()
    assert isinstance(experiment_tracker.experiments, dict)
    assert isinstance(experiment_tracker.leaderboard, list)


@pytest.mark.unit
def test_create_experiment(experiment_tracker):
    """Test creating experiment"""
    hyperparams = {
        'learning_rate': 0.001,
        'gamma': 0.99
    }

    exp_id = experiment_tracker.create_experiment(
        name='test_exp',
        hyperparameters=hyperparams,
        description='Test experiment',
        tags=['test', 'dqn']
    )

    assert exp_id.startswith('test_exp_')
    assert exp_id in experiment_tracker.experiments

    exp = experiment_tracker.experiments[exp_id]
    assert exp['name'] == 'test_exp'
    assert exp['hyperparameters'] == hyperparams
    assert exp['status'] == 'running'
    assert 'test' in exp['tags']


@pytest.mark.unit
def test_log_result(experiment_tracker):
    """Test logging results"""
    exp_id = experiment_tracker.create_experiment(
        name='test',
        hyperparameters={'lr': 0.001}
    )

    experiment_tracker.log_result(
        exp_id,
        episode=1,
        profit=100.0,
        sharpe=1.5
    )

    exp = experiment_tracker.experiments[exp_id]
    assert len(exp['results']) == 1
    assert exp['results'][0]['episode'] == 1
    assert exp['results'][0]['profit'] == 100.0


@pytest.mark.unit
def test_complete_experiment(experiment_tracker, temp_dir):
    """Test completing experiment"""
    exp_id = experiment_tracker.create_experiment(
        name='test',
        hyperparameters={'lr': 0.001}
    )

    experiment_tracker.log_result(exp_id, episode=1, profit=100.0, sharpe=1.5)

    final_metrics = {
        'total_episodes': 1,
        'best_profit': 100.0
    }

    experiment_tracker.complete_experiment(exp_id, final_metrics)

    exp = experiment_tracker.experiments[exp_id]
    assert exp['status'] == 'completed'
    assert exp['final_metrics'] == final_metrics


@pytest.mark.unit
def test_save_and_load_experiment(experiment_tracker):
    """Test saving and loading experiment"""
    exp_id = experiment_tracker.create_experiment(
        name='test',
        hyperparameters={'lr': 0.001}
    )

    experiment_tracker.log_result(exp_id, episode=1, profit=100.0)

    # Save
    filepath = experiment_tracker.save_experiment(exp_id)
    assert filepath.exists()

    # Clear from memory
    del experiment_tracker.experiments[exp_id]

    # Load
    loaded = experiment_tracker.load_experiment(exp_id)
    assert loaded is not None
    assert loaded['name'] == 'test'
    assert len(loaded['results']) == 1


@pytest.mark.unit
def test_compare_experiments(experiment_tracker):
    """Test comparing experiments"""
    # Create two experiments
    exp1 = experiment_tracker.create_experiment(
        name='exp1',
        hyperparameters={'lr': 0.001}
    )
    exp2 = experiment_tracker.create_experiment(
        name='exp2',
        hyperparameters={'lr': 0.01}
    )

    # Log results
    for i in range(10):
        experiment_tracker.log_result(exp1, episode=i, profit=i * 10.0, sharpe=1.0)
        experiment_tracker.log_result(exp2, episode=i, profit=i * 15.0, sharpe=1.5)

    # Compare
    comparison = experiment_tracker.compare_experiments([exp1, exp2])

    assert len(comparison['experiments']) == 2
    assert 'metrics_comparison' in comparison
    assert 'profit' in comparison['metrics_comparison']


@pytest.mark.unit
def test_get_leaderboard(experiment_tracker):
    """Test leaderboard"""
    # Create experiments
    exp1 = experiment_tracker.create_experiment('exp1', {'lr': 0.001})
    exp2 = experiment_tracker.create_experiment('exp2', {'lr': 0.01})

    # Log results
    experiment_tracker.log_result(exp1, episode=1, profit=100.0, sharpe=1.0)
    experiment_tracker.log_result(exp2, episode=1, profit=200.0, sharpe=1.5)

    # Complete to update leaderboard
    experiment_tracker.complete_experiment(exp1)
    experiment_tracker.complete_experiment(exp2)

    # Get leaderboard
    leaderboard = experiment_tracker.get_leaderboard(metric='profit', top_n=10)

    assert len(leaderboard) == 2
    # exp2 should be first (higher profit)
    assert leaderboard[0]['name'] == 'exp2'


@pytest.mark.unit
def test_export_comparison(experiment_tracker, temp_dir):
    """Test exporting comparison"""
    exp1 = experiment_tracker.create_experiment('exp1', {'lr': 0.001})
    exp2 = experiment_tracker.create_experiment('exp2', {'lr': 0.01})

    experiment_tracker.log_result(exp1, episode=1, profit=100.0)
    experiment_tracker.log_result(exp2, episode=1, profit=200.0)

    # Export as JSON
    filepath = experiment_tracker.export_comparison([exp1, exp2], format='json')
    assert filepath.exists()

    with open(filepath) as f:
        data = json.load(f)
    assert len(data['experiments']) == 2


@pytest.mark.unit
def test_list_experiments(experiment_tracker):
    """Test listing experiments"""
    exp1 = experiment_tracker.create_experiment(
        'exp1',
        {'lr': 0.001},
        tags=['test']
    )
    exp2 = experiment_tracker.create_experiment(
        'exp2',
        {'lr': 0.01},
        tags=['production']
    )

    # List all
    all_exp = experiment_tracker.list_experiments()
    assert len(all_exp) == 2

    # Filter by tags
    test_exp = experiment_tracker.list_experiments(tags=['test'])
    assert len(test_exp) == 1
    assert test_exp[0]['name'] == 'exp1'

    # Complete one and filter by status
    experiment_tracker.complete_experiment(exp1)
    running = experiment_tracker.list_experiments(status='running')
    assert len(running) == 1
    assert running[0]['name'] == 'exp2'


@pytest.mark.unit
def test_get_experiment_summary(experiment_tracker):
    """Test experiment summary"""
    exp_id = experiment_tracker.create_experiment('test', {'lr': 0.001})

    # Log multiple results
    for i in range(20):
        experiment_tracker.log_result(
            exp_id,
            episode=i,
            profit=i * 10.0,
            sharpe=1.0 + i * 0.1
        )

    summary = experiment_tracker.get_experiment_summary(exp_id)

    assert summary['name'] == 'test'
    assert summary['total_episodes'] == 20
    assert 'statistics' in summary
    assert 'profit' in summary['statistics']
    assert summary['statistics']['profit']['max'] == 190.0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
def test_full_monitoring_workflow(temp_dir):
    """Test complete monitoring workflow"""
    # Setup
    metrics_logger = MetricsLogger(
        experiment_name='integration_test',
        output_dir=temp_dir,
        enable_async=False
    )

    tracker = ExperimentTracker(experiments_dir=temp_dir)

    alert_system = AlertSystem({
        'enable_alerts': True,
        'email': {'enabled': False},
        'slack': {'enabled': False},
        'thresholds': AlertThresholds.MODERATE
    })

    # Create experiment
    exp_id = tracker.create_experiment(
        name='integration_test',
        hyperparameters={'lr': 0.001, 'gamma': 0.99}
    )

    # Simulate training
    for episode in range(100):
        profit = episode * 5.0
        sharpe = 1.0 + episode * 0.01

        # Log to metrics logger
        metrics_logger.log_episode(
            episode=episode,
            profit=profit,
            sharpe=sharpe,
            win_rate=0.6,
            epsilon=1.0 / (episode + 1)
        )

        # Log to tracker
        tracker.log_result(
            exp_id,
            episode=episode,
            profit=profit,
            sharpe=sharpe
        )

        # Check health every 10 episodes
        if episode % 10 == 0:
            latest = metrics_logger.get_latest_episode()
            recent = metrics_logger.get_episode_history(last_n=50)
            alert_system.check_training_health(latest, recent)

    # Complete
    stats = metrics_logger.get_statistics()
    tracker.complete_experiment(exp_id, final_metrics=stats)

    # Verify results
    assert len(metrics_logger.episode_metrics) == 100
    assert stats['total_episodes'] == 100
    assert stats['best_profit'] == 495.0

    exp = tracker.experiments[exp_id]
    assert exp['status'] == 'completed'
    assert len(exp['results']) == 100

    # Save
    metrics_filepath = metrics_logger.save_metrics()
    exp_filepath = tracker.save_experiment(exp_id)

    assert metrics_filepath.exists()
    assert exp_filepath.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
