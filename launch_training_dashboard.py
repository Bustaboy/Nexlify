#!/usr/bin/env python3
"""
Launch Training Dashboard
Standalone launcher for the real-time training monitoring dashboard
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nexlify.monitoring.metrics_logger import MetricsLogger
from nexlify.monitoring.training_dashboard import TrainingDashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Launch Nexlify Training Dashboard'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='default',
        help='Experiment name to monitor'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run dashboard on (default: 8050)'
    )
    parser.add_argument(
        '--update-interval',
        type=int,
        default=2000,
        help='Update interval in milliseconds (default: 2000)'
    )
    parser.add_argument(
        '--theme',
        type=str,
        default='cyberpunk',
        choices=['cyberpunk', 'dark', 'light'],
        help='Dashboard theme (default: cyberpunk)'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='training_logs',
        help='Training logs directory (default: training_logs)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode with simulated data'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Nexlify Training Dashboard")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Update Interval: {args.update_interval}ms")
    logger.info(f"Theme: {args.theme}")
    logger.info(f"Logs Directory: {args.logs_dir}")
    logger.info("=" * 60)

    try:
        # Create metrics logger
        metrics_logger = MetricsLogger(
            experiment_name=args.experiment,
            output_dir=args.logs_dir,
            enable_async=True
        )

        # Demo mode - populate with sample data
        if args.demo:
            logger.info("Running in DEMO mode - generating sample data...")
            populate_demo_data(metrics_logger)

        # Create and start dashboard
        dashboard = TrainingDashboard(
            metrics_logger=metrics_logger,
            port=args.port,
            update_interval=args.update_interval,
            theme=args.theme
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dashboard starting on http://localhost:{args.port}")
        logger.info(f"Press Ctrl+C to stop")
        logger.info(f"{'=' * 60}\n")

        # Start dashboard (blocking)
        dashboard.start(blocking=True)

    except KeyboardInterrupt:
        logger.info("\nShutting down dashboard...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


def populate_demo_data(metrics_logger: MetricsLogger):
    """Populate metrics logger with demo data"""
    import numpy as np
    import random

    logger.info("Generating 200 episodes of demo data...")

    # Simulate training progression
    epsilon = 1.0
    lr = 0.001
    base_profit = 0

    for episode in range(1, 201):
        # Epsilon decay
        epsilon = max(0.01, epsilon * 0.995)

        # Learning rate decay
        if episode % 50 == 0:
            lr *= 0.5

        # Profit - improving over time with noise
        trend = episode * 2
        noise = random.gauss(0, 50)
        profit = base_profit + trend + noise

        # Sharpe ratio - improving
        sharpe = min(2.0, (episode / 100) + random.gauss(0, 0.3))

        # Win rate - improving
        win_rate = min(0.7, 0.3 + (episode / 400) + random.gauss(0, 0.05))

        # Drawdown
        drawdown = abs(random.gauss(100, 50))

        # Number of trades
        num_trades = random.randint(10, 50)

        # Log episode
        metrics_logger.log_episode(
            episode=episode,
            profit=profit,
            sharpe=max(0, sharpe),
            win_rate=max(0, min(1, win_rate)),
            drawdown=drawdown,
            num_trades=num_trades,
            epsilon=epsilon,
            learning_rate=lr
        )

        # Log model metrics every 5 episodes
        if episode % 5 == 0:
            loss = 10.0 / (episode ** 0.5) + random.gauss(0, 0.5)
            q_values = np.random.randn(100) * 10 + episode * 0.1

            metrics_logger.log_model_metrics(
                loss=max(0, loss),
                q_values=q_values.tolist(),
                gradients={
                    'mean': random.gauss(0, 0.01),
                    'std': random.gauss(0.01, 0.005),
                    'max': random.gauss(0.05, 0.01),
                    'min': random.gauss(-0.05, 0.01)
                }
            )

    logger.info("Demo data generation complete!")


if __name__ == '__main__':
    sys.exit(main())
