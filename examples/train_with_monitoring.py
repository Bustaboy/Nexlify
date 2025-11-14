#!/usr/bin/env python3
"""
Example: Training with Real-Time Monitoring
Demonstrates integration of monitoring system with RL training
"""

import sys
import logging
from pathlib import Path
import asyncio
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexlify.monitoring.metrics_logger import MetricsLogger
from nexlify.monitoring.training_dashboard import TrainingDashboard
from nexlify.monitoring.alert_system import AlertSystem, AlertThresholds
from nexlify.monitoring.experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_training_with_monitoring():
    """
    Example training loop with full monitoring integration

    This demonstrates:
    1. Creating an experiment
    2. Logging metrics during training
    3. Running dashboard in background
    4. Sending alerts on critical events
    5. Tracking best models
    """

    # 1. Setup Experiment Tracker
    tracker = ExperimentTracker(experiments_dir="experiments")

    hyperparameters = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'buffer_size': 100000,
        'target_update': 10
    }

    exp_id = tracker.create_experiment(
        name="dqn_with_monitoring",
        hyperparameters=hyperparameters,
        description="Example training with full monitoring",
        tags=['dqn', 'example', 'monitoring']
    )

    logger.info(f"Created experiment: {exp_id}")

    # 2. Setup Metrics Logger
    metrics_logger = MetricsLogger(
        experiment_name=exp_id,
        output_dir="training_logs",
        auto_save_interval=50,
        enable_async=True
    )

    # 3. Setup Alert System
    alert_config = {
        'enable_alerts': True,
        'email': {
            'enabled': False,  # Set to True and configure for real alerts
        },
        'slack': {
            'enabled': False,  # Set to True and add webhook URL
        },
        'thresholds': AlertThresholds.MODERATE
    }

    alert_system = AlertSystem(alert_config)

    # 4. Setup Dashboard (runs in background)
    dashboard = TrainingDashboard(
        metrics_logger=metrics_logger,
        port=8050,
        update_interval=2000,
        theme='cyberpunk'
    )

    logger.info("Starting dashboard on http://localhost:8050")
    dashboard.start(blocking=False)

    # 5. Training Loop
    logger.info("Starting training with monitoring...")

    try:
        num_episodes = 500
        epsilon = hyperparameters['epsilon_start']
        lr = hyperparameters['learning_rate']
        best_profit = float('-inf')

        for episode in range(1, num_episodes + 1):
            # === SIMULATED TRAINING ===
            # In real training, replace this with your actual training code

            # Epsilon decay
            epsilon = max(
                hyperparameters['epsilon_end'],
                epsilon * hyperparameters['epsilon_decay']
            )

            # Learning rate decay (every 100 episodes)
            if episode % 100 == 0:
                lr *= 0.5

            # Simulate episode (replace with real environment)
            profit, sharpe, win_rate, drawdown, num_trades = simulate_episode(
                episode,
                epsilon
            )

            # Simulate model training (replace with real training)
            loss, q_values, gradients = simulate_training_step(episode)

            # === LOG METRICS ===

            # Log episode metrics
            metrics_logger.log_episode(
                episode=episode,
                profit=profit,
                sharpe=sharpe,
                win_rate=win_rate,
                drawdown=drawdown,
                num_trades=num_trades,
                epsilon=epsilon,
                learning_rate=lr
            )

            # Log model metrics
            metrics_logger.log_model_metrics(
                loss=loss,
                q_values=q_values,
                gradients=gradients
            )

            # Log to experiment tracker
            tracker.log_result(
                exp_id,
                episode=episode,
                profit=profit,
                sharpe=sharpe,
                win_rate=win_rate,
                loss=loss
            )

            # === CHECK FOR ALERTS ===

            if episode % 10 == 0:
                latest = metrics_logger.get_latest_episode()
                recent = metrics_logger.get_episode_history(last_n=50)
                alert_system.check_training_health(latest, recent)

            # === TRACK BEST MODEL ===

            if profit > best_profit:
                best_profit = profit
                logger.info(f"New best profit: ${profit:.2f} at episode {episode}")

                # Send alert for new best
                alert_system.send_new_best_model(
                    episode=episode,
                    profit=profit,
                    sharpe=sharpe
                )

                # In real training, save model checkpoint here
                # torch.save(agent.state_dict(), f'models/best_{exp_id}.pth')

            # === PROGRESS LOGGING ===

            if episode % 50 == 0:
                stats = metrics_logger.get_statistics()
                logger.info(
                    f"Episode {episode}/{num_episodes} | "
                    f"Profit: ${profit:.2f} | "
                    f"Sharpe: {sharpe:.2f} | "
                    f"Win Rate: {win_rate:.1%} | "
                    f"Best: ${best_profit:.2f} | "
                    f"Epsilon: {epsilon:.3f}"
                )

        # === TRAINING COMPLETE ===

        logger.info("Training complete!")

        # Get final statistics
        stats = metrics_logger.get_statistics()

        # Complete experiment
        tracker.complete_experiment(
            exp_id,
            final_metrics=stats
        )

        # Send completion alert
        alert_system.send_training_complete(
            total_episodes=num_episodes,
            best_profit=stats['best_profit'],
            best_sharpe=stats['best_sharpe'],
            training_time=stats['training_time']
        )

        # Save final metrics
        metrics_logger.save_metrics(format='json')
        metrics_logger.save_metrics(format='csv')

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Experiment ID: {exp_id}")
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Best Profit: ${stats['best_profit']:.2f}")
        print(f"Best Sharpe: {stats['best_sharpe']:.2f}")
        print(f"Training Time: {stats['training_time']:.1f}s")
        print(f"Avg Log Time: {stats['avg_log_time_ms']:.3f}ms")
        print("=" * 60)

        # Show leaderboard
        print("\nLEADERBOARD (Top 5):")
        print("-" * 60)
        leaderboard = tracker.get_leaderboard(metric='profit', top_n=5)
        for i, entry in enumerate(leaderboard, 1):
            print(
                f"{i}. {entry['name']}: "
                f"${entry['best_profit']:.2f} "
                f"(Sharpe: {entry['best_sharpe']:.2f})"
            )
        print("-" * 60)

        # Keep dashboard running
        print(f"\nDashboard still running at http://localhost:8050")
        print("Press Ctrl+C to exit...")

        # Keep running until interrupted
        import time
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    finally:
        # Cleanup
        metrics_logger.close()
        dashboard.stop()
        logger.info("Monitoring system shut down")


def simulate_episode(episode: int, epsilon: float):
    """
    Simulate an episode (replace with real environment)

    Returns:
        profit, sharpe, win_rate, drawdown, num_trades
    """
    import random
    import numpy as np

    # Simulate improving performance over time
    base_profit = episode * 1.5
    noise = random.gauss(0, 50)
    profit = base_profit + noise

    sharpe = min(2.5, (episode / 100) + random.gauss(0, 0.3))
    sharpe = max(0, sharpe)

    win_rate = min(0.75, 0.3 + (episode / 500) + random.gauss(0, 0.05))
    win_rate = max(0, min(1, win_rate))

    drawdown = abs(random.gauss(100, 30))

    num_trades = random.randint(15, 40)

    return profit, sharpe, win_rate, drawdown, num_trades


def simulate_training_step(episode: int):
    """
    Simulate a training step (replace with real training)

    Returns:
        loss, q_values, gradients
    """
    import random
    import numpy as np

    # Loss decreases over time
    loss = 10.0 / (episode ** 0.5) + random.gauss(0, 0.3)
    loss = max(0.01, loss)

    # Q-values increase over time
    q_values = (np.random.randn(100) * 5 + episode * 0.2).tolist()

    # Gradient statistics
    gradients = {
        'mean': random.gauss(0, 0.01),
        'std': random.gauss(0.01, 0.005),
        'max': random.gauss(0.05, 0.01),
        'min': random.gauss(-0.05, 0.01)
    }

    return loss, q_values, gradients


if __name__ == '__main__':
    try:
        example_training_with_monitoring()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
