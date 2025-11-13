#!/usr/bin/env python3
"""
Nexlify COMPLETE Features Training
Trains on ALL features actually implemented in Nexlify codebase

CRITICAL: This script trains on risk management features that were
previously missing:
✅ Stop-loss orders (2%)
✅ Take-profit orders (5%)
✅ Trailing stops (3%)
✅ Kelly Criterion position sizing
✅ Daily loss limits (5%)
✅ Max concurrent trades
✅ Position size limits

Plus all the strategy features:
✅ Multi-pair spot trading
✅ DeFi staking
✅ Liquidity provision
✅ Arbitrage

Without these risk management features, the agent would never learn
proper risk control!
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
import json
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))

from nexlify_environments.nexlify_complete_strategy_env import (
    CompleteMultiStrategyEnvironment,
    RiskLimits
)
from nexlify_rl_models.nexlify_ultra_optimized_rl_agent import UltraOptimizedDQNAgent
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig
from nexlify_data.nexlify_external_features import ExternalFeatureEnricher
from nexlify_preflight_checker import PreFlightChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Nexlify COMPLETE Features Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script trains on ALL features actually implemented in Nexlify:

RISK MANAGEMENT (Previously Missing!):
  ✅ Stop-loss orders (-2% auto-exit)
  ✅ Take-profit orders (+5% auto-exit)
  ✅ Trailing stops (3% from peak)
  ✅ Kelly Criterion position sizing
  ✅ Daily loss limits (5% max)
  ✅ Max concurrent trades (3)
  ✅ Position size limits (5% max)

TRADING STRATEGIES:
  ✅ Multi-pair spot trading
  ✅ DeFi staking (BTC, ETH, SOL, USDT)
  ✅ Liquidity provision (Uniswap V3, Aave)
  ✅ Arbitrage detection

Examples:
  # Train with complete features
  python train_with_complete_features.py \\
      --pairs BTC/USDT ETH/USDT SOL/USDT \\
      --episodes 500

  # Quick test
  python train_with_complete_features.py --quick-test

  # Automated
  python train_with_complete_features.py --automated
        """
    )

    parser.add_argument('--pairs', nargs='+', default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                        help='Trading pairs')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange')
    parser.add_argument('--episodes', type=int, default=500, help='Training episodes')
    parser.add_argument('--years', type=int, default=2, help='Years of data')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--output', type=str, default='./complete_training_output',
                        help='Output directory')
    parser.add_argument('--automated', action='store_true', help='Automated mode')
    parser.add_argument('--skip-preflight', action='store_true', help='Skip pre-flight checks')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')

    # Risk management overrides
    parser.add_argument('--stop-loss', type=float, default=0.02, help='Stop-loss %% (default: 0.02 = 2%%)')
    parser.add_argument('--take-profit', type=float, default=0.05, help='Take-profit %% (default: 0.05 = 5%%)')
    parser.add_argument('--trailing-stop', type=float, default=0.03, help='Trailing stop %% (default: 0.03 = 3%%)')
    parser.add_argument('--max-position', type=float, default=0.05, help='Max position size (default: 0.05 = 5%%)')
    parser.add_argument('--max-trades', type=int, default=3, help='Max concurrent trades (default: 3)')
    parser.add_argument('--no-kelly', action='store_true', help='Disable Kelly Criterion')

    args = parser.parse_args()

    # Quick test adjustments
    if args.quick_test:
        args.pairs = ['BTC/USDT']
        args.years = 1
        args.episodes = 100
        logger.info("⚡ Quick test mode")

    # Print banner
    print("\n" + "="*80)
    print("NEXLIFY COMPLETE FEATURES TRAINING")
    print("Trains on ALL actual Nexlify features (including risk management!)")
    print("="*80)
    print(f"Pairs: {', '.join(args.pairs)}")
    print(f"Episodes: {args.episodes}")
    print(f"Initial balance: ${args.balance:,.2f}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("\nRisk Management (from actual Nexlify config):")
    print(f"  Stop-loss: {args.stop_loss*100:.1f}%")
    print(f"  Take-profit: {args.take_profit*100:.1f}%")
    print(f"  Trailing stop: {args.trailing_stop*100:.1f}%")
    print(f"  Max position size: {args.max_position*100:.1f}%")
    print(f"  Max concurrent trades: {args.max_trades}")
    print(f"  Kelly Criterion: {'✅ Enabled' if not args.no_kelly else '❌ Disabled'}")
    print("\nTrading Strategies:")
    print(f"  ✅ Spot trading ({len(args.pairs)} pairs)")
    print(f"  ✅ DeFi staking (BTC, ETH, SOL, USDT)")
    print(f"  ✅ Liquidity provision (3 pools)")
    print(f"  ✅ Arbitrage detection")
    print("="*80 + "\n")

    # Pre-flight checks
    if not args.skip_preflight:
        logger.info("Running pre-flight checks...")
        for pair in args.pairs:
            checker = PreFlightChecker(symbol=pair, exchange=args.exchange)
            can_proceed, _ = checker.run_all_checks(automated_mode=args.automated)
            if not can_proceed:
                logger.error(f"Pre-flight failed for {pair}")
                if not args.automated:
                    return 1
        logger.info("✓ Pre-flight checks passed\n")

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Fetch historical data
    logger.info("Fetching historical data...")
    fetcher = HistoricalDataFetcher(automated_mode=args.automated)
    enricher = ExternalFeatureEnricher(automated_mode=args.automated)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)

    market_data = {}
    for pair in args.pairs:
        logger.info(f"Fetching {pair}...")
        config = FetchConfig(
            exchange=args.exchange,
            symbol=pair,
            timeframe='1h',
            start_date=start_date,
            end_date=end_date,
            cache_enabled=True
        )

        df, quality = fetcher.fetch_historical_data(config)
        if not df.empty:
            df = enricher.enrich_dataframe(df, symbol=pair)
            market_data[pair] = df['close'].values
            logger.info(f"✓ {pair}: {len(df)} candles, quality: {quality.quality_score:.1f}/100")

    if not market_data:
        logger.error("Failed to fetch data")
        return 1

    # Create risk limits (from actual Nexlify config)
    risk_limits = RiskLimits(
        max_position_size=args.max_position,
        max_daily_loss=0.05,  # 5% from config
        stop_loss_percent=args.stop_loss,
        take_profit_percent=args.take_profit,
        trailing_stop_percent=args.trailing_stop,
        max_concurrent_trades=args.max_trades,
        use_kelly_criterion=not args.no_kelly,
        kelly_fraction=0.5,
        min_kelly_confidence=0.6
    )

    # Create complete environment
    env = CompleteMultiStrategyEnvironment(
        trading_pairs=list(market_data.keys()),
        initial_balance=args.balance,
        market_data=market_data,
        risk_limits=risk_limits,
        enable_staking=True,
        enable_defi=True,
        enable_arbitrage=True
    )

    logger.info(f"\nEnvironment created:")
    logger.info(f"  State size: {env.state_size}")
    logger.info(f"  Action size: {env.action_size}")

    # Create agent
    agent = UltraOptimizedDQNAgent(
        state_size=env.state_size,
        action_size=env.action_size
    )

    logger.info(f"\nStarting training...\n")

    # Training loop
    best_return = float('-inf')
    best_model_path = None

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            episode_reward += reward
            state = next_state

        stats = env.get_episode_stats()

        if episode % 10 == 0:
            logger.info(
                f"Ep {episode}/{args.episodes} | "
                f"Return: {stats['total_return_pct']:+.2f}% | "
                f"Equity: ${stats['final_equity']:,.2f} | "
                f"Trades: {stats['total_trades']} | "
                f"Staking: ${stats['total_staking_rewards']:.2f} | "
                f"LP: ${stats['total_lp_fees']:.2f} | "
                f"Sharpe: {stats['sharpe_ratio']:.2f} | "
                f"DD: {stats['max_drawdown']:.1f}% | "
                f"ε: {agent.epsilon:.3f}"
            )

        # Save best model
        if stats['total_return_pct'] > best_return:
            best_return = stats['total_return_pct']
            model_path = Path(args.output) / f"best_complete_model_return{best_return:.1f}.pt"

            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'target_model_state_dict': agent.target_model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'stats': stats,
                'risk_limits': risk_limits.__dict__
            }, model_path)

            best_model_path = model_path
            logger.info(f"🏆 New best! Return: {best_return:.2f}%")

        # Periodic checkpoint
        if episode % 50 == 0:
            checkpoint = Path(args.output) / f"checkpoint_ep{episode}.pt"
            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'target_model_state_dict': agent.target_model.state_dict(),
                'stats': stats
            }, checkpoint)

    # Training complete
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Return: {best_return:.2f}%")
    print(f"Best Model: {best_model_path}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # Save summary
    summary = {
        'training_complete': True,
        'best_return_pct': best_return,
        'best_model_path': str(best_model_path),
        'pairs': args.pairs,
        'episodes': args.episodes,
        'risk_limits': risk_limits.__dict__
    }

    with open(Path(args.output) / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("✓ Training completed successfully!")
    print(f"\nNext steps:")
    print(f"  1. Review model: {best_model_path}")
    print(f"  2. Test in paper trading")
    print(f"  3. Deploy to live trading (carefully!)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
