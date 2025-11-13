#!/usr/bin/env python3
"""
Nexlify Comprehensive Multi-Strategy Training
Trains on ALL available trading strategies for maximum profitability

Strategies Trained:
1. Multi-Pair Spot Trading
2. DeFi Staking
3. Yield Farming / Liquidity Provision
4. Cross-Exchange Arbitrage
5. Portfolio Rebalancing

This ensures the model learns to use ALL features of the AI trader.
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from nexlify_environments.nexlify_multi_strategy_env import MultiStrategyEnvironment
from nexlify_rl_models.nexlify_ultra_optimized_rl_agent import UltraOptimizedDQNAgent
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig
from nexlify_data.nexlify_external_features import ExternalFeatureEnricher
from nexlify_preflight_checker import PreFlightChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveMultiStrategyTrainer:
    """
    Trains agent on ALL trading strategies simultaneously
    """

    def __init__(
        self,
        trading_pairs: List[str],
        exchange: str = 'binance',
        output_dir: str = './multi_strategy_output',
        automated_mode: bool = False
    ):
        """
        Initialize comprehensive trainer

        Args:
            trading_pairs: List of trading pairs to train on
            exchange: Exchange to fetch data from
            output_dir: Output directory
            automated_mode: Automated mode flag
        """
        self.trading_pairs = trading_pairs
        self.exchange = exchange
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.automated_mode = automated_mode

        logger.info(f"Comprehensive Multi-Strategy Trainer initialized")
        logger.info(f"  Trading pairs: {trading_pairs}")
        logger.info(f"  Exchange: {exchange}")
        logger.info(f"  Strategies: Spot + Staking + DeFi + Arbitrage")

    def fetch_multi_pair_data(
        self,
        years: int = 2
    ) -> Dict[str, np.ndarray]:
        """
        Fetch historical data for all trading pairs

        Args:
            years: Years of historical data

        Returns:
            Dictionary mapping pair to price arrays
        """
        logger.info(f"\nFetching historical data for {len(self.trading_pairs)} pairs...")

        fetcher = HistoricalDataFetcher(automated_mode=self.automated_mode)
        enricher = ExternalFeatureEnricher(automated_mode=self.automated_mode)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        market_data = {}

        for pair in self.trading_pairs:
            logger.info(f"\nFetching {pair}...")

            config = FetchConfig(
                exchange=self.exchange,
                symbol=pair,
                timeframe='1h',
                start_date=start_date,
                end_date=end_date,
                cache_enabled=True
            )

            df, quality = fetcher.fetch_historical_data(config)

            if not df.empty:
                # Enrich with features
                df = enricher.enrich_dataframe(df, symbol=pair)

                # Extract close prices
                market_data[pair] = df['close'].values

                logger.info(f"✓ {pair}: {len(df)} candles, quality: {quality.quality_score:.1f}/100")
            else:
                logger.warning(f"✗ {pair}: No data fetched")

        if not market_data:
            raise ValueError("Failed to fetch data for any pairs")

        return market_data

    def train_comprehensive_agent(
        self,
        market_data: Dict[str, np.ndarray],
        episodes: int = 500,
        initial_balance: float = 10000.0
    ) -> UltraOptimizedDQNAgent:
        """
        Train agent on all strategies

        Args:
            market_data: Market data for all pairs
            episodes: Number of training episodes
            initial_balance: Starting balance

        Returns:
            Trained agent
        """
        logger.info(f"\nTraining comprehensive multi-strategy agent...")
        logger.info(f"  Episodes: {episodes}")
        logger.info(f"  Initial balance: ${initial_balance:,.2f}")

        # Create multi-strategy environment
        env = MultiStrategyEnvironment(
            trading_pairs=list(market_data.keys()),
            initial_balance=initial_balance,
            market_data=market_data,
            enable_staking=True,
            enable_defi=True,
            enable_arbitrage=True
        )

        logger.info(f"\nEnvironment Details:")
        logger.info(f"  State size: {env.state_size}")
        logger.info(f"  Action size: {env.action_size}")
        logger.info(f"  Strategies enabled:")
        logger.info(f"    ✓ Spot Trading ({len(env.trading_pairs)} pairs)")
        logger.info(f"    ✓ Staking ({len(env.staking_pools)} pools)")
        logger.info(f"    ✓ DeFi LP ({len(env.liquidity_pools)} pools)")
        logger.info(f"    ✓ Arbitrage (enabled)")

        # Create agent
        agent = UltraOptimizedDQNAgent(
            state_size=env.state_size,
            action_size=env.action_size
        )

        logger.info(f"\nStarting training...\n")

        # Training metrics
        best_return = float('-inf')
        best_model_path = None

        for episode in range(1, episodes + 1):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Agent selects action
                action = agent.act(state)

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                agent.replay()

                episode_reward += reward
                state = next_state

            # Get episode stats
            stats = env.get_episode_stats()

            # Log progress
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}/{episodes} | "
                    f"Return: {stats['total_return_pct']:+.2f}% | "
                    f"Equity: ${stats['final_equity']:,.2f} | "
                    f"Trades: {stats['total_trades']} | "
                    f"Staking: ${stats['total_staking_rewards']:.2f} | "
                    f"LP Fees: ${stats['total_lp_fees']:.2f} | "
                    f"Arbitrage: ${stats['total_arbitrage_profits']:.2f} | "
                    f"Passive: ${stats['total_passive_income']:.2f} | "
                    f"ε: {agent.epsilon:.3f}"
                )

            # Save best model
            if stats['total_return_pct'] > best_return:
                best_return = stats['total_return_pct']

                model_path = self.output_dir / f"best_multi_strategy_model_return{best_return:.1f}.pt"

                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.model.state_dict(),
                    'target_model_state_dict': agent.target_model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'stats': stats,
                    'return_pct': best_return
                }, model_path)

                best_model_path = model_path
                logger.info(f"🏆 New best model! Return: {best_return:.2f}%")

            # Save checkpoint every 50 episodes
            if episode % 50 == 0:
                checkpoint_path = self.output_dir / f"checkpoint_ep{episode}.pt"
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.model.state_dict(),
                    'target_model_state_dict': agent.target_model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'stats': stats
                }, checkpoint_path)

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Best Return: {best_return:.2f}%")
        logger.info(f"Best Model: {best_model_path}")
        logger.info(f"{'='*80}\n")

        return agent


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Nexlify Comprehensive Multi-Strategy Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script trains on ALL available trading strategies:
  ✓ Multi-pair spot trading
  ✓ DeFi staking (passive income)
  ✓ Yield farming / liquidity provision
  ✓ Cross-exchange arbitrage
  ✓ Portfolio rebalancing

Examples:
  # Train on top 3 crypto pairs
  python train_comprehensive_multi_strategy.py --pairs BTC/USDT ETH/USDT SOL/USDT

  # Quick test
  python train_comprehensive_multi_strategy.py --quick-test

  # Fully automated
  python train_comprehensive_multi_strategy.py --automated
        """
    )

    parser.add_argument(
        '--pairs',
        nargs='+',
        default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        help='Trading pairs to train on (default: BTC/USDT ETH/USDT SOL/USDT)'
    )

    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange to fetch data from (default: binance)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Number of training episodes (default: 500)'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=2,
        help='Years of historical data (default: 2)'
    )

    parser.add_argument(
        '--balance',
        type=float,
        default=10000.0,
        help='Initial balance in USDT (default: 10000)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./multi_strategy_output',
        help='Output directory (default: ./multi_strategy_output)'
    )

    parser.add_argument(
        '--automated',
        action='store_true',
        help='Fully automated mode (no prompts, uses fallbacks)'
    )

    parser.add_argument(
        '--skip-preflight',
        action='store_true',
        help='Skip pre-flight checks (not recommended)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (1 pair, 1 year, 100 episodes)'
    )

    args = parser.parse_args()

    # Quick test adjustments
    if args.quick_test:
        args.pairs = ['BTC/USDT']
        args.years = 1
        args.episodes = 100
        logger.info("⚡ Quick test mode enabled")

    # Print banner
    print("\n" + "="*80)
    print("NEXLIFY COMPREHENSIVE MULTI-STRATEGY TRAINING")
    print("Train on ALL strategies for maximum profitability")
    print("="*80)
    print(f"Trading pairs: {', '.join(args.pairs)}")
    print(f"Exchange: {args.exchange}")
    print(f"Historical data: {args.years} years")
    print(f"Episodes: {args.episodes}")
    print(f"Initial balance: ${args.balance:,.2f}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("\nStrategies to be trained:")
    print(f"  ✓ Spot Trading ({len(args.pairs)} pairs)")
    print(f"  ✓ DeFi Staking (BTC, ETH, SOL, USDT)")
    print(f"  ✓ Liquidity Provision (BTC/ETH, ETH/USDT, BTC/USDT)")
    print(f"  ✓ Arbitrage Detection & Execution")
    print(f"  ✓ Portfolio Optimization")
    print("="*80 + "\n")

    # Pre-flight checks
    if not args.skip_preflight:
        logger.info("Running pre-flight checks for all pairs...")

        all_checks_passed = True

        for pair in args.pairs:
            logger.info(f"\nChecking {pair}...")
            checker = PreFlightChecker(symbol=pair, exchange=args.exchange)
            can_proceed, _ = checker.run_all_checks(automated_mode=args.automated)

            if not can_proceed:
                logger.error(f"✗ Pre-flight check failed for {pair}")
                all_checks_passed = False

                if not args.automated:
                    response = input(f"\nContinue without {pair}? (yes/no): ").strip().lower()
                    if response in ['yes', 'y']:
                        args.pairs.remove(pair)
                        logger.info(f"Removed {pair} from training")
                    else:
                        logger.error("Training aborted by user")
                        return 1
                else:
                    logger.warning(f"Automated mode: removing {pair} from training")
                    args.pairs.remove(pair)

        if not args.pairs:
            logger.error("No valid trading pairs remaining. Aborting.")
            return 1

        if not all_checks_passed:
            logger.warning(f"\n⚠ Some pairs failed pre-flight checks")
            logger.info(f"Training will proceed with {len(args.pairs)} pairs: {args.pairs}")

            if not args.automated:
                response = input("\nContinue with remaining pairs? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    logger.info("Training aborted by user")
                    return 1

        logger.info("\n✓ Pre-flight checks complete. Starting training...\n")

    # Create trainer
    trainer = ComprehensiveMultiStrategyTrainer(
        trading_pairs=args.pairs,
        exchange=args.exchange,
        output_dir=args.output,
        automated_mode=args.automated
    )

    # Fetch data
    try:
        market_data = trainer.fetch_multi_pair_data(years=args.years)
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        return 1

    # Train agent
    try:
        agent = trainer.train_comprehensive_agent(
            market_data=market_data,
            episodes=args.episodes,
            initial_balance=args.balance
        )

        print("\n✓ Training completed successfully!")
        print(f"\nResults saved to: {args.output}")
        print(f"\nNext steps:")
        print(f"  1. Review best model in {args.output}/")
        print(f"  2. Test in paper trading")
        print(f"  3. Deploy to live trading (use caution!)")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Partial results may be available in output directory")
        return 1

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
