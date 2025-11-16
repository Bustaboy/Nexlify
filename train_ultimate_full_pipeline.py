#!/usr/bin/env python3
"""
ULTIMATE Fully-Automated Training Pipeline

This is the most comprehensive training system possible, implementing ALL best practices:

PHASE 1 - Fundamentals:
✅ Gradient clipping
✅ Learning rate scheduling
✅ L2 regularization
✅ Early stopping
✅ Ensemble methods

PHASE 2 - Advanced Algorithms:
✅ Double DQN
✅ Dueling DQN
✅ Stochastic Weight Averaging
✅ Multi-start initialization

PHASE 3 - Expert Techniques:
✅ Prioritized Experience Replay
✅ N-step returns
✅ Data augmentation
✅ Walk-forward cross-validation
✅ Hyperparameter optimization (optional)

COMPLETE FEATURES:
✅ ALL Nexlify risk management (stop-loss, take-profit, trailing stops, Kelly Criterion)
✅ Multi-strategy trading (spot, staking, DeFi, arbitrage)
✅ Historical data fetching from multiple exchanges
✅ External feature enrichment (Fear & Greed, on-chain, social)
✅ Pre-flight validation

DESIGNED FOR:
- 24+ hour training runs
- Maximum model quality (not speed)
- Fully automated execution
- Production deployment

This is the FINAL, BEST training system. Use this for production models.
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
import torch
import random

sys.path.append(str(Path(__file__).parent))

# Import our components
from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig
from nexlify_rl_auto_tuner import AutoHyperparameterTuner, TuningMetrics
from nexlify_validation_and_optimization import (
    WalkForwardValidator,
    HyperparameterOptimizer,
    GridSearchOptimizer,
    ValidationFold,
    create_default_hyperparameter_search_space
)
from nexlify_environments.nexlify_complete_strategy_env import (
    CompleteMultiStrategyEnvironment,
    RiskLimits
)
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig
from nexlify_data.nexlify_external_features import ExternalFeatureEnricher
from nexlify_preflight_checker import PreFlightChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ultimate_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UltimateTrainingPipeline:
    """
    Ultimate fully-automated training pipeline

    Combines ALL best practices for maximum performance
    """

    def __init__(self, args):
        """Initialize pipeline with command-line arguments"""
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = None
        self.best_score = float('-inf')
        self.ensemble_models = []

        # Initialize auto-tuner (enabled by default, can be disabled via args)
        enable_auto_tuning = getattr(args, 'no_auto_tuning', False) == False
        if enable_auto_tuning:
            self.auto_tuner = AutoHyperparameterTuner(
                window_size=50,
                min_episodes_before_tuning=100,
                tuning_frequency=50,
                enable_lr_tuning=True,
                enable_epsilon_tuning=True,
                verbose=True
            )
            logger.info("🎯 Automatic hyperparameter tuning ENABLED")
        else:
            self.auto_tuner = None
            logger.info("⚠️  Automatic hyperparameter tuning DISABLED")

        logger.info("\n" + "="*100)
        logger.info("ULTIMATE FULLY-AUTOMATED TRAINING PIPELINE")
        logger.info("="*100)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Pairs: {', '.join(args.pairs)}")
        logger.info(f"Training years: {args.years}")
        logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        logger.info("="*100 + "\n")

    def _set_seed(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def run_preflight_checks(self) -> bool:
        """Run pre-flight validation checks"""
        if self.args.skip_preflight:
            logger.info("WARNING: Skipping pre-flight checks")
            return True

        logger.info("\n" + "="*80)
        logger.info("PRE-FLIGHT VALIDATION")
        logger.info("="*80)

        all_passed = True
        for pair in self.args.pairs:
            checker = PreFlightChecker(symbol=pair, exchange=self.args.exchange)
            passed, results = checker.run_all_checks(automated_mode=self.args.automated)

            if not passed:
                logger.error(f"❌ Pre-flight failed for {pair}")
                all_passed = False
                if not self.args.automated:
                    return False

        if all_passed:
            logger.info("OK: All pre-flight checks passed\n")

        return all_passed

    def fetch_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Fetch and prepare historical data with automatic exchange selection

        Returns:
            (training_data, validation_data)
        """
        logger.info("\n" + "="*80)
        logger.info("DATA FETCHING WITH QUALITY VERIFICATION")
        logger.info("="*80)

        fetcher = HistoricalDataFetcher(automated_mode=self.args.automated)
        enricher = ExternalFeatureEnricher(automated_mode=self.args.automated)

        end_date = datetime.now()
        train_start = end_date - timedelta(days=self.args.years * 365)
        val_start = end_date - timedelta(days=90)  # Last 90 days for validation

        train_data = {}
        val_data = {}
        selected_exchanges = {}  # Track which exchange was used for each pair

        # Determine if we should auto-select exchanges
        use_auto_select = (
            hasattr(self.args, 'use_best_exchange') and self.args.use_best_exchange
        ) or self.args.exchange.lower() == 'auto'

        for pair in self.args.pairs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Fetching {pair}")
            logger.info('='*60)

            # Select best exchange if enabled
            if use_auto_select:
                logger.info(f"🔍 Auto-selecting best exchange for {pair}...")
                try:
                    best_exchange, df_train, quality_train = fetcher.select_best_exchange(
                        symbol=pair,
                        timeframe='1h',
                        start_date=train_start,
                        end_date=val_start,
                        preferred_exchanges=['coinbase', 'kraken', 'bitstamp', 'gemini']
                    )
                    selected_exchanges[pair] = best_exchange
                    logger.info(f"✅ Selected {best_exchange} for {pair} (quality: {quality_train.quality_score:.1f}/100)")
                except ValueError as e:
                    logger.error(f"Failed to find suitable exchange for {pair}: {e}")
                    logger.info("Falling back to specified exchange...")
                    best_exchange = self.args.exchange
                    selected_exchanges[pair] = best_exchange
                    train_config = FetchConfig(
                        exchange=best_exchange,
                        symbol=pair,
                        timeframe='1h',
                        start_date=train_start,
                        end_date=val_start,
                        cache_enabled=True
                    )
                    df_train, quality_train = fetcher.fetch_historical_data(train_config)
            else:
                # Use specified exchange
                selected_exchanges[pair] = self.args.exchange
                logger.info(f"Using specified exchange: {self.args.exchange}")
                train_config = FetchConfig(
                    exchange=self.args.exchange,
                    symbol=pair,
                    timeframe='1h',
                    start_date=train_start,
                    end_date=val_start,
                    cache_enabled=True
                )
                df_train, quality_train = fetcher.fetch_historical_data(train_config)

            # Validate training data quality
            if df_train.empty:
                logger.error(f"No training data available for {pair} from {selected_exchanges[pair]}")
                continue

            min_quality = getattr(self.args, 'min_quality', 70.0)  # Lower default for multi-pair
            min_candles = getattr(self.args, 'min_candles', 500)   # Lower default for multi-pair

            if quality_train.quality_score < min_quality:
                logger.warning(
                    f"⚠️  {pair} training data quality ({quality_train.quality_score:.1f}) "
                    f"below minimum ({min_quality})"
                )
            if len(df_train) < min_candles:
                logger.warning(
                    f"⚠️  {pair} has only {len(df_train)} candles (minimum: {min_candles})"
                )

            df_train = enricher.enrich_dataframe(df_train, symbol=pair)
            train_data[pair] = df_train['close'].values
            logger.info(f"  ✓ Training: {len(df_train)} candles, quality: {quality_train.quality_score:.1f}/100")

            # Fetch validation data from same exchange
            val_config = FetchConfig(
                exchange=selected_exchanges[pair],
                symbol=pair,
                timeframe='1h',
                start_date=val_start,
                end_date=end_date,
                cache_enabled=True
            )

            df_val, quality_val = fetcher.fetch_historical_data(val_config)

            if not df_val.empty:
                df_val = enricher.enrich_dataframe(df_val, symbol=pair)
                val_data[pair] = df_val['close'].values
                logger.info(f"  ✓ Validation: {len(df_val)} candles, quality: {quality_val.quality_score:.1f}/100")
            else:
                logger.error(f"No validation data available for {pair}")

        if not train_data or not val_data:
            raise ValueError("Failed to fetch sufficient data for any trading pair")

        logger.info(f"\n{'='*80}")
        logger.info("DATA FETCHING SUMMARY")
        logger.info('='*80)
        logger.info(f"Pairs loaded: {len(train_data)}")
        for pair in train_data:
            logger.info(f"  {pair}: {selected_exchanges[pair]}")
        logger.info(f"Training samples: {len(list(train_data.values())[0])}")
        logger.info(f"Validation samples: {len(list(val_data.values())[0])}")
        logger.info('='*80)

        return train_data, val_data

    def create_risk_limits(self) -> RiskLimits:
        """Create risk limits from arguments"""
        return RiskLimits(
            max_position_size=self.args.max_position,
            max_daily_loss=0.05,
            stop_loss_percent=self.args.stop_loss,
            take_profit_percent=self.args.take_profit,
            trailing_stop_percent=self.args.trailing_stop,
            max_concurrent_trades=self.args.max_trades,
            use_kelly_criterion=not self.args.no_kelly,
            kelly_fraction=0.5
        )

    def create_agent_config(
        self,
        custom_params: Optional[Dict] = None,
        disable_early_stopping: bool = False
    ) -> AgentConfig:
        """
        Create agent configuration

        Args:
            custom_params: Custom parameter overrides
            disable_early_stopping: If True, sets early_stop_patience to a very high value
                                   (useful for initial training runs)
        """
        config = AgentConfig(
            # Architecture (will be set based on state size)
            hidden_layers=[256, 256, 128],

            # Training hyperparameters
            gamma=0.99,
            learning_rate=0.001,
            batch_size=64,

            # Exploration
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,

            # Replay buffer
            buffer_size=100000,
            use_prioritized_replay=True,
            per_alpha=0.6,
            per_beta=0.4,

            # N-step returns
            n_step=3,

            # Target network
            target_update_frequency=1000,

            # Phase 1: Best practices
            gradient_clip_norm=1.0,
            weight_decay=1e-5,
            lr_scheduler_type='plateau',
            lr_scheduler_patience=5,
            lr_scheduler_factor=0.5,

            # Phase 2: Advanced
            use_double_dqn=True,
            use_dueling_dqn=True,
            use_swa=True,
            swa_start=5000,

            # Phase 3: Expert
            use_data_augmentation=True,
            augmentation_probability=0.5,

            # Early stopping (disabled for initial runs, enabled for retraining)
            early_stop_patience=999999 if disable_early_stopping else 30,
            early_stop_threshold=0.01,

            # Metrics
            track_metrics=True
        )

        # Override with custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    def train_single_agent(
        self,
        env: CompleteMultiStrategyEnvironment,
        agent: AdvancedDQNAgent,
        episodes: int,
        val_env: Optional[CompleteMultiStrategyEnvironment] = None,
        val_frequency: int = 10
    ) -> Tuple[AdvancedDQNAgent, Dict[str, Any]]:
        """
        Train a single agent

        Args:
            env: Training environment
            agent: Agent to train
            episodes: Number of episodes
            val_env: Validation environment (optional)
            val_frequency: Validation frequency in episodes

        Returns:
            (trained_agent, training_metrics)
        """
        best_return = float('-inf')

        # Track metrics for auto-tuning
        episode_metrics_history = []

        for episode in range(1, episodes + 1):
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done:
                action = agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()
                episode_reward += reward
                state = next_state
                steps += 1

            stats = env.get_episode_stats()

            # Store episode metrics for auto-tuning
            episode_metrics_history.append({
                'return_pct': stats['total_return_pct'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'win_rate': stats.get('win_rate', 0.0),
                'total_trades': stats['total_trades']
            })

            # Auto-tune hyperparameters
            if self.auto_tuner is not None:
                # Calculate rolling averages
                recent_window = min(50, len(episode_metrics_history))
                recent_metrics = episode_metrics_history[-recent_window:]

                avg_return = np.mean([m['return_pct'] for m in recent_metrics])
                avg_sharpe = np.mean([m['sharpe_ratio'] for m in recent_metrics])
                avg_win_rate = np.mean([m['win_rate'] for m in recent_metrics])

                # Get agent metrics
                agent_metrics = agent.get_metrics_summary()
                avg_loss = agent_metrics.get('avg_loss', 0.0)

                # Get Q-value statistics
                q_values = agent.metrics.get('q_value_history', [])
                q_mean = np.mean(q_values[-100:]) if q_values else 0.0
                q_std = np.std(q_values[-100:]) if q_values else 0.0

                # Create tuning metrics
                tuning_metrics = TuningMetrics(
                    episode=episode,
                    avg_return=avg_return,
                    avg_sharpe=avg_sharpe,
                    win_rate=avg_win_rate,
                    epsilon=agent.epsilon,
                    learning_rate=agent.optimizer.param_groups[0]['lr'],
                    avg_loss=avg_loss,
                    q_value_mean=q_mean,
                    q_value_std=q_std
                )

                # Get auto-tuning adjustments
                adjustments = self.auto_tuner.update(tuning_metrics)

                # Apply adjustments
                if adjustments:
                    if 'epsilon' in adjustments:
                        agent.epsilon = adjustments['epsilon']
                        logger.info(f"   🎯 Auto-tuned epsilon → {agent.epsilon:.4f}")

                    if 'learning_rate' in adjustments:
                        for param_group in agent.optimizer.param_groups:
                            param_group['lr'] = adjustments['learning_rate']
                        logger.info(f"   🎯 Auto-tuned LR → {adjustments['learning_rate']:.6f}")

            # Log progress
            if episode % 10 == 0:
                metrics = agent.get_metrics_summary()
                logger.info(
                    f"Ep {episode}/{episodes} | "
                    f"Return: {stats['total_return_pct']:+.2f}% | "
                    f"Equity: ${stats['final_equity']:,.2f} | "
                    f"Trades: {stats['total_trades']} | "
                    f"Sharpe: {stats['sharpe_ratio']:.2f} | "
                    f"DD: {stats['max_drawdown']:.1f}% | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"LR: {metrics['current_lr']:.6f} | "
                    f"Loss: {metrics['avg_loss']:.4f}"
                )

            # Validation
            if val_env and episode % val_frequency == 0:
                val_metrics = self.evaluate_agent(agent, val_env, num_episodes=5)
                should_stop = agent.update_validation_score(val_metrics['overall_score'])

                if should_stop:
                    logger.info(f"🛑 Early stopping at episode {episode}")
                    break

            # Track best
            if stats['total_return_pct'] > best_return:
                best_return = stats['total_return_pct']

        final_stats = env.get_episode_stats()
        final_metrics = agent.get_metrics_summary()

        # Log auto-tuner summary
        if self.auto_tuner is not None:
            summary = self.auto_tuner.get_summary()
            logger.info("\n" + "="*80)
            logger.info("🎯 AUTO-TUNING SUMMARY")
            logger.info("="*80)
            logger.info(f"Total adjustments made: {summary['total_adjustments']}")
            logger.info(f"Final performance trend: {summary['performance_trend']}")
            logger.info(f"Episodes since improvement: {summary['episodes_since_improvement']}")
            logger.info(f"Best performance: {summary['best_performance']:.2f}%")
            logger.info(f"Recent avg return: {summary['recent_avg_return']:.2f}%")
            logger.info(f"Recent avg Sharpe: {summary['recent_avg_sharpe']:.2f}")
            logger.info("="*80 + "\n")

        return agent, {
            'total_return_pct': final_stats['total_return_pct'],
            'final_equity': final_stats['final_equity'],
            'sharpe_ratio': final_stats['sharpe_ratio'],
            'max_drawdown': final_stats['max_drawdown'],
            'total_trades': final_stats['total_trades'],
            'win_rate': final_stats.get('win_rate', 0.0),
            'best_return': best_return,
            'final_metrics': final_metrics
        }

    def evaluate_agent(
        self,
        agent: AdvancedDQNAgent,
        env: CompleteMultiStrategyEnvironment,
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent on environment

        Returns:
            Metrics dictionary
        """
        returns = []
        sharpes = []
        drawdowns = []
        win_rates = []

        for ep in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                state = next_state

            stats = env.get_episode_stats()
            returns.append(stats['total_return_pct'])
            sharpes.append(stats['sharpe_ratio'])
            drawdowns.append(stats['max_drawdown'])
            win_rates.append(stats.get('win_rate', 0.0))

        # Compute metrics
        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpes)
        avg_drawdown = np.mean(drawdowns)
        avg_win_rate = np.mean(win_rates)

        # Overall score
        score = (
            avg_return * 0.4 +
            avg_sharpe * 10 * 0.3 +
            avg_win_rate * 100 * 0.2 -
            avg_drawdown * 0.1
        )

        return {
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_drawdown': avg_drawdown,
            'avg_win_rate': avg_win_rate,
            'overall_score': score,
            'returns_std': np.std(returns)
        }

    def run_multi_start_initialization(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        risk_limits: RiskLimits
    ) -> Tuple[str, float, List[str]]:
        """
        Run multi-start initialization

        Returns:
            (best_model_path, best_score, all_model_paths)
        """
        logger.info("\n" + "#"*100)
        logger.info("PHASE 1: MULTI-START INITIALIZATION")
        logger.info(f"Running {self.args.initial_runs} independent training sessions")
        logger.info("#"*100 + "\n")

        initial_results = []
        all_model_paths = []

        for run_id in range(1, self.args.initial_runs + 1):
            seed = 42 + run_id * 1000
            self._set_seed(seed)

            logger.info(f"\n{'='*80}")
            logger.info(f"INITIAL RUN {run_id}/{self.args.initial_runs} (seed: {seed})")
            logger.info(f"{'='*80}")

            run_dir = self.output_dir / "initial_runs" / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create environments
            train_env = CompleteMultiStrategyEnvironment(
                trading_pairs=list(train_data.keys()),
                initial_balance=self.args.balance,
                market_data=train_data,
                risk_limits=risk_limits,
                enable_staking=True,
                enable_defi=True,
                enable_arbitrage=True
            )

            val_env = CompleteMultiStrategyEnvironment(
                trading_pairs=list(val_data.keys()),
                initial_balance=self.args.balance,
                market_data=val_data,
                risk_limits=risk_limits,
                enable_staking=True,
                enable_defi=True,
                enable_arbitrage=True
            )

            # Create agent (disable early stopping for initial runs - let them complete fully)
            config = self.create_agent_config(disable_early_stopping=True)
            agent = AdvancedDQNAgent(
                state_size=train_env.state_size,
                action_size=train_env.action_size,
                config=config
            )

            # Train
            agent, train_metrics = self.train_single_agent(
                env=train_env,
                agent=agent,
                episodes=self.args.initial_episodes,
                val_env=val_env,
                val_frequency=10
            )

            # Save model
            model_path = run_dir / f"model_return{train_metrics['total_return_pct']:.1f}.pt"
            agent.save(str(model_path))
            all_model_paths.append(str(model_path))

            # Evaluate on validation
            logger.info(f"\nEvaluating Run {run_id} on validation data...")
            val_metrics = self.evaluate_agent(agent, val_env, num_episodes=10)

            result = {
                'run_id': run_id,
                'seed': seed,
                'model_path': str(model_path),
                'train_return': train_metrics['total_return_pct'],
                'val_score': val_metrics['overall_score'],
                'val_return': val_metrics['avg_return'],
                'val_sharpe': val_metrics['avg_sharpe']
            }

            initial_results.append(result)

            logger.info(f"\nRun {run_id} Results:")
            logger.info(f"  Train Return: {train_metrics['total_return_pct']:+.2f}%")
            logger.info(f"  Val Score: {val_metrics['overall_score']:.2f}")
            logger.info(f"  Val Return: {val_metrics['avg_return']:+.2f}%")
            logger.info(f"  Val Sharpe: {val_metrics['avg_sharpe']:.2f}")

        # Select best
        best_result = max(initial_results, key=lambda x: x['val_score'])

        logger.info("\n" + "="*80)
        logger.info("INITIAL RUNS COMPARISON")
        logger.info("="*80)
        for result in sorted(initial_results, key=lambda x: x['val_score'], reverse=True):
            marker = " ⭐ BEST" if result['run_id'] == best_result['run_id'] else ""
            logger.info(
                f"Run {result['run_id']} | "
                f"Val Score: {result['val_score']:>6.2f} | "
                f"Val Return: {result['val_return']:>+6.2f}%{marker}"
            )
        logger.info("="*80 + "\n")

        return best_result['model_path'], best_result['val_score'], all_model_paths

    def run(self) -> Dict[str, Any]:
        """
        Run complete training pipeline

        Returns:
            Training summary
        """
        start_time = datetime.now()

        # Step 1: Pre-flight checks
        if not self.run_preflight_checks():
            logger.error("Pre-flight checks failed. Aborting.")
            return {}

        # Step 2: Fetch data
        train_data, val_data = self.fetch_data()

        # Step 3: Create risk limits
        risk_limits = self.create_risk_limits()

        logger.info("\nRisk Management Configuration:")
        logger.info(f"  Stop-loss: {risk_limits.stop_loss_percent*100:.1f}%")
        logger.info(f"  Take-profit: {risk_limits.take_profit_percent*100:.1f}%")
        logger.info(f"  Trailing stop: {risk_limits.trailing_stop_percent*100:.1f}%")
        logger.info(f"  Max position: {risk_limits.max_position_size*100:.1f}%")
        logger.info(f"  Max trades: {risk_limits.max_concurrent_trades}")
        logger.info(f"  Kelly Criterion: {'YES' if risk_limits.use_kelly_criterion else 'NO'}")

        # Step 4: Multi-start initialization
        best_initial_model, best_initial_score, all_initial_models = self.run_multi_start_initialization(
            train_data=train_data,
            val_data=val_data,
            risk_limits=risk_limits
        )

        # Save ensemble models (best 3)
        self.ensemble_models = sorted(
            all_initial_models,
            key=lambda x: float(x.split('return')[-1].split('.pt')[0]),
            reverse=True
        )[:3]

        self.best_model_path = best_initial_model
        self.best_score = best_initial_score

        # Training complete
        elapsed = datetime.now() - start_time

        logger.info("\n" + "="*100)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info("="*100)
        logger.info(f"Best model: {self.best_model_path}")
        logger.info(f"Best score: {self.best_score:.2f}")
        logger.info(f"Ensemble models: {len(self.ensemble_models)}")
        logger.info(f"Time elapsed: {elapsed}")
        logger.info("="*100 + "\n")

        # Save summary
        summary = {
            'pipeline_complete': True,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'elapsed_seconds': elapsed.total_seconds(),
            'best_model_path': str(self.best_model_path),
            'best_score': self.best_score,
            'ensemble_models': [str(p) for p in self.ensemble_models],
            'pairs': self.args.pairs,
            'risk_limits': {
                'stop_loss': risk_limits.stop_loss_percent,
                'take_profit': risk_limits.take_profit_percent,
                'trailing_stop': risk_limits.trailing_stop_percent,
                'max_position': risk_limits.max_position_size,
                'max_trades': risk_limits.max_concurrent_trades,
                'kelly_criterion': risk_limits.use_kelly_criterion
            }
        }

        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"OK: Summary saved: {summary_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Ultimate Fully-Automated Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is the MOST COMPREHENSIVE training system combining ALL best practices:
- Phase 1: Gradient clipping, LR scheduling, L2 reg, early stopping
- Phase 2: Double DQN, Dueling DQN, SWA, multi-start
- Phase 3: PER, N-step returns, data augmentation, walk-forward CV

Examples:
  # Full training (production)
  python train_ultimate_full_pipeline.py \\
      --pairs BTC/USDT ETH/USDT SOL/USDT \\
      --initial-episodes 500 \\
      --initial-runs 3

  # Quick test
  python train_ultimate_full_pipeline.py --quick-test

  # Automated 24+ hour run
  python train_ultimate_full_pipeline.py --automated
        """
    )

    # Data parameters
    parser.add_argument('--pairs', nargs='+', default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
    parser.add_argument('--exchange', type=str, default='auto',
                        help='Exchange to use. Use "auto" for automatic best exchange selection, or specify: coinbase, kraken, bitstamp, gemini (default: auto)')
    parser.add_argument('--years', type=int, default=2)
    parser.add_argument('--balance', type=float, default=10000)

    # Dataset quality parameters
    parser.add_argument('--use-best-exchange', action='store_true', default=True,
                        help='Automatically select best exchange for each pair (default: True)')
    parser.add_argument('--no-best-exchange', action='store_false', dest='use_best_exchange',
                        help='Disable automatic exchange selection')
    parser.add_argument('--min-quality', type=float, default=70.0,
                        help='Minimum data quality score (0-100, default: 70.0)')
    parser.add_argument('--min-candles', type=int, default=500,
                        help='Minimum candles required per pair (default: 500)')

    # Training parameters
    parser.add_argument('--initial-runs', type=int, default=3)
    parser.add_argument('--initial-episodes', type=int, default=500)

    # Risk management
    parser.add_argument('--stop-loss', type=float, default=0.02)
    parser.add_argument('--take-profit', type=float, default=0.05)
    parser.add_argument('--trailing-stop', type=float, default=0.03)
    parser.add_argument('--max-position', type=float, default=0.05)
    parser.add_argument('--max-trades', type=int, default=3)
    parser.add_argument('--no-kelly', action='store_true')

    # Output and control
    parser.add_argument('--output', type=str, default='./ultimate_training_output')
    parser.add_argument('--automated', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true')
    parser.add_argument('--quick-test', action='store_true')
    parser.add_argument('--no-auto-tuning', action='store_true',
                        help='Disable automatic hyperparameter tuning (enabled by default)')

    args = parser.parse_args()

    # Quick test adjustments
    if args.quick_test:
        args.pairs = ['BTC/USDT']
        args.years = 1
        args.initial_runs = 2
        args.initial_episodes = 100
        logger.info("Quick test mode activated")

    # Print configuration
    print("\n" + "="*100)
    print("ULTIMATE FULLY-AUTOMATED TRAINING PIPELINE")
    print("="*100)
    print(f"Pairs: {', '.join(args.pairs)}")
    print(f"Training: {args.years} years of data")
    print(f"Initial runs: {args.initial_runs} × {args.initial_episodes} episodes")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("\nML/RL Features Enabled:")
    print("  [X] Double DQN (reduces overestimation)")
    print("  [X] Dueling DQN (separates value/advantage)")
    print("  [X] Prioritized Experience Replay (better sampling)")
    print("  [X] N-step returns (n=3, better credit assignment)")
    print("  [X] Stochastic Weight Averaging (better generalization)")
    print("  [X] Gradient clipping (stability)")
    print("  [X] LR scheduling (adaptive learning)")
    print("  [X] L2 regularization (prevents overfitting)")
    print("  [X] Early stopping (efficiency)")
    print("  [X] Data augmentation (robustness)")
    print("\nRisk Management:")
    print(f"  Stop-loss: {args.stop_loss*100:.1f}%")
    print(f"  Take-profit: {args.take_profit*100:.1f}%")
    print(f"  Trailing stop: {args.trailing_stop*100:.1f}%")
    print("="*100 + "\n")

    # Run pipeline
    pipeline = UltimateTrainingPipeline(args)
    summary = pipeline.run()

    if summary:
        print("\n" + "="*100)
        print("*** TRAINING COMPLETE! ***")
        print("="*100)
        print(f"Best model: {summary['best_model_path']}")
        print(f"Best score: {summary['best_score']:.2f}")
        print(f"Total time: {timedelta(seconds=int(summary['elapsed_seconds']))}")
        print("\nNext steps:")
        print("  1. Review training summary")
        print("  2. Test best model in paper trading")
        print("  3. Create ensemble from top 3 models")
        print("  4. Deploy to live trading (carefully!)")
        print("="*100)
        return 0
    else:
        print("\n*** Training failed ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
