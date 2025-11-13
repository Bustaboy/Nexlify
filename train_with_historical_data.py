#!/usr/bin/env python3
"""
Nexlify Advanced Training with Historical Data
Main entry point for comprehensive ML/RL training with automatic retraining

Features:
- Fetches extensive historical data from multiple sources
- Enriches data with external features (sentiment, on-chain, etc.)
- Curriculum learning (easy → hard progression)
- Automatic retraining until marginal improvements plateau
- Comprehensive evaluation and model selection
- Best model tracking and saving

Usage:
    python train_with_historical_data.py --symbol BTC/USDT --years 5
    python train_with_historical_data.py --symbol ETH/USDT --years 3 --no-curriculum
    python train_with_historical_data.py --quick-test  # Fast test with 1 year
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from nexlify_training.nexlify_advanced_training_orchestrator import AdvancedTrainingOrchestrator
from nexlify_training.nexlify_model_evaluator import ModelEvaluator
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig
from nexlify_rl_models.nexlify_ultra_optimized_rl_agent import UltraOptimizedDQNAgent
from nexlify_preflight_checker import PreFlightChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoRetrainingOrchestrator:
    """
    Orchestrates automatic retraining with marginal improvement tracking
    """

    def __init__(
        self,
        output_dir: str = "./training_output",
        improvement_threshold: float = 1.0,  # Minimum % improvement to continue
        patience: int = 3,  # Number of non-improving iterations before stopping
        max_iterations: int = 10  # Maximum retraining iterations
    ):
        """
        Initialize auto-retraining orchestrator

        Args:
            output_dir: Output directory
            improvement_threshold: Minimum improvement % to continue training
            patience: Number of iterations without improvement before stopping
            max_iterations: Maximum number of retraining iterations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.improvement_threshold = improvement_threshold
        self.patience = patience
        self.max_iterations = max_iterations

        self.training_history = []
        self.best_score = float('-inf')
        self.best_model_path = None
        self.no_improvement_count = 0

        logger.info(f"Auto-retraining orchestrator initialized")
        logger.info(f"Improvement threshold: {improvement_threshold}%")
        logger.info(f"Patience: {patience} iterations")
        logger.info(f"Max iterations: {max_iterations}")

    def run_training_with_auto_retrain(
        self,
        symbol: str = 'BTC/USDT',
        exchange: str = 'binance',
        years: int = 5,
        use_curriculum: bool = True,
        quick_test: bool = False
    ) -> Dict[str, Any]:
        """
        Run training with automatic retraining until marginal improvements plateau

        Args:
            symbol: Trading pair
            exchange: Exchange name
            years: Years of historical data
            use_curriculum: Use curriculum learning
            quick_test: Quick test mode (less data, fewer episodes)

        Returns:
            Final training summary
        """
        logger.info("\n" + "="*80)
        logger.info("NEXLIFY ADVANCED TRAINING WITH AUTO-RETRAINING")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Exchange: {exchange}")
        logger.info(f"Historical data: {years} years")
        logger.info(f"Curriculum learning: {use_curriculum}")
        logger.info(f"Quick test mode: {quick_test}")
        logger.info("="*80 + "\n")

        # Adjust parameters for quick test
        if quick_test:
            years = 1
            self.max_iterations = 3
            logger.info("⚡ Quick test mode: using 1 year of data and 3 iterations")

        # Fetch and prepare test data for evaluation
        logger.info("Fetching validation data for model evaluation...")
        end_date = datetime.now()
        validation_start = end_date - timedelta(days=90)  # Last 90 days for validation

        data_fetcher = HistoricalDataFetcher()
        validation_config = FetchConfig(
            exchange=exchange,
            symbol=symbol,
            timeframe='1h',
            start_date=validation_start,
            end_date=end_date,
            cache_enabled=True
        )

        validation_data, _ = data_fetcher.fetch_historical_data(validation_config)
        logger.info(f"✓ Prepared {len(validation_data)} candles for validation")

        # Initialize evaluator
        evaluator = ModelEvaluator(output_dir=str(self.output_dir / "evaluation"))

        # Start iterative training
        iteration = 1
        training_start_time = datetime.now()

        while iteration <= self.max_iterations:
            logger.info(f"\n{'#'*80}")
            logger.info(f"TRAINING ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'#'*80}\n")

            # Create fresh orchestrator for this iteration
            orchestrator = AdvancedTrainingOrchestrator(
                output_dir=str(self.output_dir / f"iteration_{iteration}")
            )

            # Run comprehensive training
            try:
                training_summary = orchestrator.run_comprehensive_training(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe='1h',
                    total_years=years,
                    use_curriculum=use_curriculum
                )

                # Get best model from this iteration
                if orchestrator.best_model:
                    best_model_path = orchestrator.best_model.model_path
                    logger.info(f"✓ Best model from iteration: {best_model_path}")

                    # Load and evaluate on validation data
                    logger.info("\nEvaluating model on validation data...")
                    agent = self._load_model(best_model_path)

                    eval_metrics = evaluator.evaluate_model(
                        agent=agent,
                        test_data=validation_data,
                        model_name=f"Iteration_{iteration}",
                        num_episodes=10
                    )

                    current_score = eval_metrics.overall_score

                    logger.info(f"\nIteration {iteration} Results:")
                    logger.info(f"  Score: {current_score:.2f}")
                    logger.info(f"  Return: {eval_metrics.total_return_pct:.2f}%")
                    logger.info(f"  Sharpe: {eval_metrics.sharpe_ratio:.2f}")
                    logger.info(f"  Win Rate: {eval_metrics.win_rate:.1%}")

                    # Calculate improvement
                    if self.best_score == float('-inf'):
                        improvement_pct = 100.0
                        is_improvement = True
                    else:
                        improvement_pct = ((current_score - self.best_score) / abs(self.best_score)) * 100
                        is_improvement = improvement_pct >= self.improvement_threshold

                    logger.info(f"  Improvement: {improvement_pct:+.2f}%")

                    # Track history
                    iteration_result = {
                        'iteration': iteration,
                        'score': current_score,
                        'improvement_pct': improvement_pct,
                        'metrics': eval_metrics.__dict__,
                        'model_path': best_model_path,
                        'is_improvement': is_improvement,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.training_history.append(iteration_result)

                    # Update best model if improved
                    if current_score > self.best_score:
                        logger.info(f"🏆 NEW BEST MODEL! (Previous: {self.best_score:.2f})")
                        self.best_score = current_score
                        self.best_model_path = best_model_path
                        self.no_improvement_count = 0

                        # Save best model to main directory
                        self._save_best_model(best_model_path, iteration, eval_metrics)
                    else:
                        self.no_improvement_count += 1
                        logger.info(f"No improvement ({self.no_improvement_count}/{self.patience})")

                    # Check stopping criteria
                    if self.no_improvement_count >= self.patience:
                        logger.info(f"\n⚠️ Stopping: No improvement for {self.patience} iterations")
                        break

                    if improvement_pct < self.improvement_threshold and iteration > 1:
                        logger.info(f"\n⚠️ Stopping: Improvement ({improvement_pct:.2f}%) below threshold ({self.improvement_threshold}%)")
                        break

                else:
                    logger.warning("No best model found in this iteration")

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()

            iteration += 1

        # Training complete
        training_end_time = datetime.now()
        total_time = (training_end_time - training_start_time).total_seconds()

        # Generate final report
        final_summary = self._generate_final_report(
            symbol=symbol,
            exchange=exchange,
            total_time=total_time,
            validation_data_info=f"{len(validation_data)} candles"
        )

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total iterations: {len(self.training_history)}")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Best score: {self.best_score:.2f}")
        logger.info(f"Best model: {self.best_model_path}")
        logger.info("="*80 + "\n")

        return final_summary

    def _load_model(self, model_path: str) -> UltraOptimizedDQNAgent:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location='cpu')

        agent = UltraOptimizedDQNAgent(state_size=8, action_size=3)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        agent.epsilon = 0  # No exploration during evaluation

        return agent

    def _save_best_model(self, model_path: str, iteration: int, metrics):
        """Save best model to main directory"""
        import shutil

        best_model_dir = self.output_dir / "best_model"
        best_model_dir.mkdir(exist_ok=True)

        # Copy model file
        dest_path = best_model_dir / f"best_model_iter{iteration}_score{self.best_score:.1f}.pt"
        shutil.copy(model_path, dest_path)

        # Save metadata
        metadata = {
            'iteration': iteration,
            'score': self.best_score,
            'metrics': metrics.__dict__,
            'original_path': model_path,
            'saved_at': datetime.now().isoformat()
        }

        metadata_path = best_model_dir / "best_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"✓ Saved best model: {dest_path}")

    def _generate_final_report(
        self,
        symbol: str,
        exchange: str,
        total_time: float,
        validation_data_info: str
    ) -> Dict[str, Any]:
        """Generate final training report"""
        report = {
            'training_info': {
                'symbol': symbol,
                'exchange': exchange,
                'total_iterations': len(self.training_history),
                'total_time_seconds': total_time,
                'total_time_hours': total_time / 3600,
                'completed_at': datetime.now().isoformat(),
                'validation_data': validation_data_info
            },
            'final_results': {
                'best_score': self.best_score,
                'best_model_path': self.best_model_path,
                'improvement_threshold': self.improvement_threshold,
                'patience': self.patience
            },
            'iteration_history': self.training_history,
            'improvement_curve': [
                {
                    'iteration': h['iteration'],
                    'score': h['score'],
                    'improvement_pct': h['improvement_pct']
                }
                for h in self.training_history
            ]
        }

        # Save report
        report_path = self.output_dir / "final_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"✓ Final report saved: {report_path}")

        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Nexlify Advanced Training with Historical Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on 5 years of BTC data with curriculum learning
  python train_with_historical_data.py --symbol BTC/USDT --years 5

  # Quick test (1 year, 3 iterations)
  python train_with_historical_data.py --quick-test

  # Train on ETH without curriculum
  python train_with_historical_data.py --symbol ETH/USDT --years 3 --no-curriculum

  # Custom improvement threshold and patience
  python train_with_historical_data.py --threshold 2.0 --patience 5
        """
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair (default: BTC/USDT)'
    )

    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange name (default: binance)'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Years of historical data (default: 5)'
    )

    parser.add_argument(
        '--no-curriculum',
        action='store_true',
        help='Disable curriculum learning'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=1.0,
        help='Minimum improvement %% to continue training (default: 1.0)'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Number of iterations without improvement before stopping (default: 3)'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum retraining iterations (default: 10)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./training_output',
        help='Output directory (default: ./training_output)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (1 year, 3 iterations)'
    )

    parser.add_argument(
        '--automated',
        action='store_true',
        help='Fully automated mode (skip all prompts, use fallbacks)'
    )

    parser.add_argument(
        '--skip-preflight',
        action='store_true',
        help='Skip pre-flight checks (not recommended)'
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "="*80)
    print("NEXLIFY ADVANCED TRAINING WITH HISTORICAL DATA")
    print("Comprehensive ML/RL Training with Automatic Retraining")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Historical data: {args.years} years")
    print(f"Curriculum learning: {not args.no_curriculum}")
    print(f"Improvement threshold: {args.threshold}%")
    print(f"Patience: {args.patience} iterations")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Output directory: {args.output}")
    print(f"Quick test: {args.quick_test}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*80 + "\n")

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Run pre-flight checks unless skipped
    if not args.skip_preflight:
        logger.info("Running pre-flight checks...")
        checker = PreFlightChecker(symbol=args.symbol, exchange=args.exchange)
        can_proceed, check_results = checker.run_all_checks(automated_mode=args.automated)

        # Save pre-flight report
        checker.save_report(f"{args.output}/preflight_report.json")

        if not can_proceed:
            logger.error("Pre-flight checks failed. Aborting training.")
            logger.info("\nTo skip pre-flight checks (not recommended):")
            logger.info("  python train_with_historical_data.py --skip-preflight ...")
            return 1

        logger.info("\n✓ Pre-flight checks passed. Starting training...\n")
    else:
        logger.warning("⚠ Pre-flight checks skipped (not recommended)\n")

    # Create orchestrator
    orchestrator = AutoRetrainingOrchestrator(
        output_dir=args.output,
        improvement_threshold=args.threshold,
        patience=args.patience,
        max_iterations=args.max_iterations
    )

    # Run training
    try:
        summary = orchestrator.run_training_with_auto_retrain(
            symbol=args.symbol,
            exchange=args.exchange,
            years=args.years,
            use_curriculum=not args.no_curriculum,
            quick_test=args.quick_test
        )

        print("\n✓ Training completed successfully!")
        print(f"Best model score: {summary['final_results']['best_score']:.2f}")
        print(f"Best model path: {summary['final_results']['best_model_path']}")
        print(f"\nResults saved to: {args.output}")
        print("\nNext steps:")
        print(f"  1. Check training report: {args.output}/final_training_report.json")
        print(f"  2. Review best model: {args.output}/best_model/")
        print(f"  3. Run paper trading with best model")
        print(f"  4. Deploy to live trading (use caution!)")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Partial results may be available in output directory")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
