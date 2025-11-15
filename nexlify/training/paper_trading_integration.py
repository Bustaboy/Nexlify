"""
Paper Trading Integration for Trained Models

Allows testing walk-forward trained models using paper trading before live deployment.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import json

from nexlify.backtesting.nexlify_paper_trading import PaperTradingEngine
from nexlify.models.model_manifest import ModelManifest, ModelManager
from nexlify.utils.error_handler import get_error_handler

logger = logging.getLogger(__name__)


class ModelPaperTester:
    """
    Test trained models using paper trading

    This allows validation of trained walk-forward models in a simulated
    trading environment before deploying them to live trading.

    Example:
        >>> manifest = ModelManifest.load('models/walk_forward/fold_0_manifest.json')
        >>> tester = ModelPaperTester(manifest)
        >>> results = await tester.run_paper_test(duration_days=7)
        >>> print(f"Paper trading Sharpe: {results['sharpe_ratio']:.2f}")
    """

    def __init__(
        self,
        manifest: ModelManifest,
        initial_balance: float = 10000.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize paper trading tester

        Args:
            manifest: Model manifest with training capabilities
            initial_balance: Starting paper trading balance
            config: Optional configuration overrides
        """
        self.manifest = manifest
        self.config = config or {}
        self.error_handler = get_error_handler()

        # Initialize paper trading engine
        paper_config = {
            'paper_balance': initial_balance,
            'fee_rate': self.config.get('fee_rate', 0.001),
            'slippage': self.config.get('slippage', 0.0005),
        }
        self.paper_engine = PaperTradingEngine(paper_config)

        logger.info(f"Paper tester initialized for model: {manifest.model_name}")

    async def run_paper_test(
        self,
        duration_days: int = 7,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run paper trading test for trained model

        Args:
            duration_days: Number of days to run paper trading
            symbol: Trading symbol (uses manifest default if None)
            timeframe: Trading timeframe (uses manifest default if None)

        Returns:
            Dict with paper trading results and metrics
        """
        # Use manifest capabilities if not specified
        if symbol is None and self.manifest.capabilities.symbols:
            symbol = self.manifest.capabilities.symbols[0]

        if timeframe is None and self.manifest.capabilities.timeframes:
            timeframe = self.manifest.capabilities.timeframes[0]

        if not symbol or not timeframe:
            raise ValueError("Symbol and timeframe required for paper trading")

        # Validate model can trade this
        is_valid, reason = self.manifest.validate_trade(symbol, timeframe)
        if not is_valid:
            raise ValueError(f"Model cannot trade {symbol} on {timeframe}: {reason}")

        logger.info(f"Starting {duration_days}-day paper test for {symbol} on {timeframe}")

        # Simulate trading for duration
        start_time = datetime.now()
        end_time = start_time + timedelta(days=duration_days)

        # Placeholder for actual paper trading logic
        # In production, this would:
        # 1. Load the trained model
        # 2. Fetch real market data
        # 3. Generate trading signals
        # 4. Execute trades via paper engine
        # 5. Track performance

        logger.warning("Paper trading simulation not yet fully implemented")
        logger.info("In production, this would run model on live data without real money")

        # Return mock results for now
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'duration_days': duration_days,
            'start_balance': self.paper_engine.initial_balance,
            'end_balance': self.paper_engine.current_balance,
            'total_trades': self.paper_engine.total_trades,
            'winning_trades': self.paper_engine.winning_trades,
            'losing_trades': self.paper_engine.losing_trades,
            'win_rate': (
                self.paper_engine.winning_trades / max(self.paper_engine.total_trades, 1)
            ),
            'total_fees': self.paper_engine.total_fees_paid,
            'sharpe_ratio': 0.0,  # Would be calculated from actual trades
            'max_drawdown': 0.0,
            'message': 'Paper trading engine ready - full implementation pending'
        }

        logger.info(f"Paper test completed - {results['total_trades']} trades executed")

        return results

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """
        Save paper trading results to file

        Args:
            results: Results from run_paper_test()
            output_path: Path to save JSON results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Paper trading results saved to {output_path}")


def create_paper_test_from_manifest(
    manifest_path: Path,
    initial_balance: float = 10000.0
) -> ModelPaperTester:
    """
    Create paper tester from manifest file

    Args:
        manifest_path: Path to model manifest JSON
        initial_balance: Starting balance for paper trading

    Returns:
        Configured ModelPaperTester instance
    """
    manifest = ModelManifest.load(manifest_path)
    return ModelPaperTester(manifest, initial_balance)


async def test_all_models(
    models_dir: Path = Path('models/walk_forward'),
    duration_days: int = 7,
    initial_balance: float = 10000.0
) -> Dict[str, Dict[str, Any]]:
    """
    Run paper trading tests on all trained models

    Args:
        models_dir: Directory containing models and manifests
        duration_days: Days to run each test
        initial_balance: Starting balance for each test

    Returns:
        Dict mapping model_id to paper trading results
    """
    logger.info(f"Testing all models in {models_dir}")

    results = {}
    manifest_files = list(models_dir.glob('*_manifest.json'))

    logger.info(f"Found {len(manifest_files)} models to test")

    for manifest_file in manifest_files:
        try:
            manifest = ModelManifest.load(manifest_file)
            model_id = manifest.model_id

            logger.info(f"Testing model: {model_id}")

            tester = ModelPaperTester(manifest, initial_balance)
            test_results = await tester.run_paper_test(duration_days)

            results[model_id] = test_results

        except Exception as e:
            logger.error(f"Failed to test {manifest_file}: {e}")
            results[manifest_file.stem] = {
                'error': str(e),
                'success': False
            }

    logger.info(f"Completed testing {len(results)} models")

    return results


# Helper function for training UI integration
async def run_quick_paper_test(
    model_path: Path,
    manifest_path: Path,
    symbol: str = None,
    duration_days: int = 1
) -> Dict[str, Any]:
    """
    Quick paper trading test for trained model

    Args:
        model_path: Path to trained model file
        manifest_path: Path to model manifest
        symbol: Trading symbol to test
        duration_days: Test duration

    Returns:
        Paper trading results
    """
    manifest = ModelManifest.load(manifest_path)

    # Validate model meets thresholds
    if not manifest.meets_performance_thresholds():
        logger.warning("Model does not meet performance thresholds")

    tester = ModelPaperTester(manifest)
    results = await tester.run_paper_test(duration_days, symbol)

    return results
