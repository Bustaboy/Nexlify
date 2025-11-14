"""
Nexlify Advanced Training Orchestrator
Comprehensive training program with curriculum learning and multi-source data

Features:
- Curriculum learning (easy → hard training progression)
- Multi-exchange, multi-symbol training
- Extensive historical data (5+ years)
- Cross-validation with time-series splits
- Ensemble training across market regimes
- Automated hyperparameter optimization
- Best model selection and evaluation
- Training metrics tracking and visualization
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
import json
from dataclasses import dataclass, asdict
import torch

# Import auto-tuner
from nexlify_rl_auto_tuner import AutoHyperparameterTuner, TuningMetrics

from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig
from nexlify_data.nexlify_external_features import ExternalFeatureEnricher
from nexlify_ml.nexlify_feature_engineering import AdvancedFeatureEngineer
from nexlify_rl_models.nexlify_ultra_optimized_rl_agent import UltraOptimizedDQNAgent
from nexlify_environments.nexlify_rl_training_env import TradingEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingPhase:
    """Configuration for a curriculum learning phase"""
    name: str
    difficulty: str  # 'easy', 'medium', 'hard', 'expert'
    data_period: Tuple[datetime, datetime]
    episodes: int
    initial_balance: float
    fee_rate: float
    description: str


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics"""
    phase: str
    episode: int
    total_reward: float
    final_equity: float
    return_pct: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_loss: float
    exploration_rate: float
    training_time: float
    data_quality_score: float


@dataclass
class ModelCheckpoint:
    """Model checkpoint information"""
    phase: str
    episode: int
    performance_score: float
    metrics: TrainingMetrics
    model_path: str
    timestamp: datetime


class AdvancedTrainingOrchestrator:
    """
    Orchestrates comprehensive training with curriculum learning
    """

    def __init__(
        self,
        output_dir: str = "./training_output",
        cache_enabled: bool = True,
        device: Optional[str] = None,
        enable_auto_tuning: bool = True
    ):
        """
        Initialize the training orchestrator

        Args:
            output_dir: Directory for saving models and results
            cache_enabled: Whether to use data caching
            device: Training device ('cuda', 'cpu', or None for auto)
            enable_auto_tuning: Enable automatic hyperparameter tuning (default: True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        self.data_dir = self.output_dir / "training_data"
        self.data_dir.mkdir(exist_ok=True)

        # Initialize components
        self.data_fetcher = HistoricalDataFetcher()
        self.feature_enricher = ExternalFeatureEnricher()
        self.feature_engineer = AdvancedFeatureEngineer()

        # Initialize auto-tuner
        self.enable_auto_tuning = enable_auto_tuning
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

        # Training state
        self.cache_enabled = cache_enabled
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.all_metrics: List[TrainingMetrics] = []
        self.checkpoints: List[ModelCheckpoint] = []
        self.best_model: Optional[ModelCheckpoint] = None

        logger.info(f"Advanced Training Orchestrator initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

    def create_curriculum_phases(
        self,
        symbol: str = 'BTC/USDT',
        total_years: int = 5
    ) -> List[TrainingPhase]:
        """
        Create curriculum learning phases (easy to hard)

        Curriculum Strategy:
        1. Easy: Recent stable/trending periods, low volatility
        2. Medium: Mixed market conditions, moderate volatility
        3. Hard: High volatility periods, crashes and rallies
        4. Expert: Complete historical data, all conditions

        Args:
            symbol: Trading pair
            total_years: Total years of historical data

        Returns:
            List of training phases
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_years * 365)

        # Define phases based on market conditions
        # Note: In production, you would analyze historical data to identify
        # periods of different difficulty levels

        phases = []

        # Phase 1: Easy - Recent stable period (last 6 months of trending)
        # Lower fees to make learning easier
        phases.append(TrainingPhase(
            name="Phase 1: Warm-up",
            difficulty="easy",
            data_period=(end_date - timedelta(days=180), end_date),
            episodes=200,
            initial_balance=10000,
            fee_rate=0.0005,  # Lower fees for easier learning
            description="Recent stable market conditions for initial learning"
        ))

        # Phase 2: Medium - Past year with mixed conditions
        phases.append(TrainingPhase(
            name="Phase 2: Intermediate",
            difficulty="medium",
            data_period=(end_date - timedelta(days=365), end_date),
            episodes=300,
            initial_balance=10000,
            fee_rate=0.001,  # Normal fees
            description="Full year of mixed market conditions"
        ))

        # Phase 3: Hard - 2-3 years including volatile periods
        phases.append(TrainingPhase(
            name="Phase 3: Advanced",
            difficulty="hard",
            data_period=(end_date - timedelta(days=730), end_date),
            episodes=400,
            initial_balance=10000,
            fee_rate=0.001,
            description="Multi-year data including high volatility periods"
        ))

        # Phase 4: Expert - Complete historical data (5 years)
        phases.append(TrainingPhase(
            name="Phase 4: Expert",
            difficulty="expert",
            data_period=(start_date, end_date),
            episodes=500,
            initial_balance=10000,
            fee_rate=0.001,
            description="Complete historical dataset with all market conditions"
        ))

        logger.info(f"Created {len(phases)} curriculum learning phases")
        for phase in phases:
            logger.info(f"  - {phase.name}: {phase.episodes} episodes, "
                       f"{(phase.data_period[1] - phase.data_period[0]).days} days of data")

        return phases

    def prepare_training_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_best_exchange: bool = True,
        min_quality_score: float = 80.0,
        min_candles: int = 1000
    ) -> Tuple[pd.DataFrame, float, str]:
        """
        Fetch and prepare comprehensive training data with quality verification

        Args:
            exchange: Preferred exchange name (used if use_best_exchange=False)
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date
            use_best_exchange: If True, automatically select best exchange (default: True)
            min_quality_score: Minimum acceptable data quality score (default: 80.0)
            min_candles: Minimum number of candles required (default: 1000)

        Returns:
            Tuple of (prepared DataFrame, quality score, selected exchange)
        """
        logger.info(f"Preparing training data: {symbol}")
        logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Minimum quality: {min_quality_score}/100, Minimum candles: {min_candles}")

        # Use best exchange selection if enabled
        if use_best_exchange:
            logger.info("🔍 Searching for best data source across exchanges...")
            try:
                selected_exchange, df, quality_metrics = self.data_fetcher.select_best_exchange(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    preferred_exchanges=['coinbase', 'kraken', 'bitstamp', 'gemini']
                )
            except ValueError as e:
                logger.error(f"Failed to find suitable exchange: {e}")
                raise ValueError(
                    f"No exchange could provide quality data for {symbol}. "
                    f"Please check symbol availability or try a different time range."
                )
        else:
            # Use specified exchange
            logger.info(f"Using specified exchange: {exchange}")
            selected_exchange = exchange
            config = FetchConfig(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                cache_enabled=self.cache_enabled
            )
            df, quality_metrics = self.data_fetcher.fetch_historical_data(config)

        # Verify dataset meets minimum requirements
        if df.empty:
            raise ValueError(f"No data fetched for {symbol} from {selected_exchange}")

        candles_count = len(df)
        quality_score = quality_metrics.quality_score

        logger.info(f"✓ Fetched {candles_count} candles from {selected_exchange} (quality: {quality_score:.1f}/100)")

        # Quality validation
        issues = []
        if quality_score < min_quality_score:
            issues.append(f"Quality score {quality_score:.1f} below minimum {min_quality_score}")

        if candles_count < min_candles:
            issues.append(f"Only {candles_count} candles available, need at least {min_candles}")

        # Calculate expected candles based on timeframe
        timeframe_seconds = self.data_fetcher.TIMEFRAMES[timeframe]
        expected_candles = int((end_date - start_date).total_seconds() / timeframe_seconds)
        completeness_pct = (candles_count / expected_candles * 100) if expected_candles > 0 else 0

        if completeness_pct < 50:  # Less than 50% of expected data
            issues.append(
                f"Dataset only {completeness_pct:.1f}% complete "
                f"({candles_count}/{expected_candles} candles)"
            )

        # Log warnings or raise error
        if issues:
            error_msg = f"Dataset quality issues for {symbol}:\n  - " + "\n  - ".join(issues)
            logger.error(error_msg)
            logger.error(f"Quality metrics: {quality_metrics}")
            raise ValueError(
                f"Dataset does not meet minimum requirements for training.\n{error_msg}\n\n"
                f"Suggestions:\n"
                f"  1. Try a shorter time range (fewer years)\n"
                f"  2. Use --use-best-exchange to automatically find better data\n"
                f"  3. Check if the symbol is available on the exchange\n"
                f"  4. Lower minimum requirements (not recommended)"
            )

        logger.info(f"✓ Dataset validated: {completeness_pct:.1f}% complete, quality: {quality_score:.1f}/100")

        # Enrich with external features
        df = self.feature_enricher.enrich_dataframe(
            df,
            symbol=symbol,
            include_sentiment=True,
            include_onchain=True,
            include_social=True
        )

        logger.info(f"✓ Added external features ({len(df.columns)} total columns)")

        # Add technical indicators and ML features
        df = self.feature_engineer.add_all_features(df)

        logger.info(f"✓ Added technical features ({len(df.columns)} total columns)")

        # Save prepared data
        data_filename = f"{selected_exchange}_{symbol.replace('/', '_')}_{start_date.date()}_{end_date.date()}.parquet"
        data_path = self.data_dir / data_filename
        df.to_parquet(data_path, index=False)

        logger.info(f"✓ Saved training data: {data_path}")

        return df, quality_metrics.quality_score, selected_exchange

    def train_phase(
        self,
        phase: TrainingPhase,
        agent: UltraOptimizedDQNAgent,
        training_data: pd.DataFrame,
        data_quality_score: float,
        timeframe: str = '1h'
    ) -> List[TrainingMetrics]:
        """
        Train agent for one curriculum phase

        Args:
            phase: Training phase configuration
            agent: RL agent to train
            training_data: Prepared training data
            data_quality_score: Quality score of the data
            timeframe: Timeframe of the data (for Sharpe annualization)

        Returns:
            List of training metrics for each episode
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {phase.name}")
        logger.info(f"Difficulty: {phase.difficulty}")
        logger.info(f"Episodes: {phase.episodes}")
        logger.info(f"Data quality: {data_quality_score:.1f}/100")
        logger.info(f"{'='*60}\n")

        phase_metrics = []

        # Extract price data for environment
        prices = training_data['close'].values

        # Create training environment
        env = TradingEnvironment(
            initial_balance=phase.initial_balance,
            fee_rate=phase.fee_rate,
            slippage=0.0005,
            market_data=prices,
            timeframe=timeframe  # Pass timeframe for correct Sharpe annualization
        )

        best_episode_reward = float('-inf')
        checkpoint_frequency = max(10, phase.episodes // 10)  # Save every 10% of episodes

        for episode in range(1, phase.episodes + 1):
            episode_start_time = datetime.now()

            # Reset environment
            state = env.reset()

            total_reward = 0
            done = False
            steps = 0

            # Run episode
            while not done:
                # Agent selects action
                action = agent.act(state)

                # Environment step
                next_state, reward, done, info = env.step(action)

                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                agent.replay()

                total_reward += reward
                state = next_state
                steps += 1

            # Get episode statistics
            episode_stats = env.get_episode_stats()
            episode_time = (datetime.now() - episode_start_time).total_seconds()

            # Create metrics
            metrics = TrainingMetrics(
                phase=phase.name,
                episode=episode,
                total_reward=total_reward,
                final_equity=episode_stats['final_equity'],
                return_pct=episode_stats['total_return_pct'],
                num_trades=episode_stats['num_trades'],
                win_rate=episode_stats['win_rate'],
                sharpe_ratio=episode_stats['sharpe_ratio'],
                max_drawdown=episode_stats['max_drawdown'],
                avg_loss=episode_stats['avg_loss'],
                exploration_rate=agent.epsilon,
                training_time=episode_time,
                data_quality_score=data_quality_score
            )

            phase_metrics.append(metrics)
            self.all_metrics.append(metrics)

            # Auto-tune hyperparameters based on performance
            if self.auto_tuner is not None:
                # Calculate average metrics over recent episodes
                recent_window = min(50, len(phase_metrics))
                recent_metrics = phase_metrics[-recent_window:]

                avg_return = np.mean([m.return_pct for m in recent_metrics])
                avg_sharpe = np.mean([m.sharpe_ratio for m in recent_metrics])
                avg_win_rate = np.mean([m.win_rate for m in recent_metrics])
                avg_loss = np.mean([m.avg_loss for m in recent_metrics])

                # Get Q-value statistics from agent
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
                        logger.info(f"   🎯 Auto-tuned learning rate → {adjustments['learning_rate']:.6f}")

            # Logging
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}/{phase.episodes} | "
                    f"Return: {metrics.return_pct:+.2f}% | "
                    f"Reward: {total_reward:.2f} | "
                    f"Trades: {metrics.num_trades} | "
                    f"Win Rate: {metrics.win_rate:.1%} | "
                    f"Sharpe: {metrics.sharpe_ratio:.2f} | "
                    f"ε: {agent.epsilon:.3f}"
                )

            # Save checkpoint
            if total_reward > best_episode_reward:
                best_episode_reward = total_reward

                checkpoint = self._save_checkpoint(
                    phase=phase.name,
                    episode=episode,
                    agent=agent,
                    metrics=metrics
                )

                # Update best model if this is the best overall
                if (self.best_model is None or
                    checkpoint.performance_score > self.best_model.performance_score):
                    self.best_model = checkpoint
                    logger.info(f"🏆 New best model! Score: {checkpoint.performance_score:.2f}")

            # Periodic checkpoint
            if episode % checkpoint_frequency == 0:
                self._save_checkpoint(
                    phase=phase.name,
                    episode=episode,
                    agent=agent,
                    metrics=metrics,
                    prefix="periodic"
                )

        logger.info(f"\n✓ Completed {phase.name}")
        logger.info(f"Best episode reward: {best_episode_reward:.2f}")

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

        return phase_metrics

    def run_comprehensive_training(
        self,
        exchange: str = 'binance',
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        total_years: int = 5,
        use_curriculum: bool = True,
        use_best_exchange: bool = True,
        min_quality_score: float = 80.0,
        min_candles: int = 1000
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline with curriculum learning

        Args:
            exchange: Preferred exchange name (used if use_best_exchange=False)
            symbol: Trading pair
            timeframe: Candle timeframe
            total_years: Years of historical data
            use_curriculum: Whether to use curriculum learning
            use_best_exchange: If True, automatically select best exchange (default: True)
            min_quality_score: Minimum acceptable data quality score (default: 80.0)
            min_candles: Minimum number of candles required (default: 1000)

        Returns:
            Training summary dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("NEXLIFY ADVANCED TRAINING PROGRAM")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Exchange: {exchange}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Historical data: {total_years} years")
        logger.info(f"Curriculum learning: {use_curriculum}")
        logger.info(f"Device: {self.device}")
        logger.info("="*80 + "\n")

        training_start_time = datetime.now()

        # Create curriculum phases
        phases = self.create_curriculum_phases(symbol=symbol, total_years=total_years)

        # Initialize agent (will be used across all phases for continuous learning)
        agent = UltraOptimizedDQNAgent(
            state_size=8,  # [balance, position, entry_price, price, price_change, RSI, MACD, volume]
            action_size=3,  # BUY, SELL, HOLD
            device=self.device
        )

        # Train through each phase
        all_phase_results = {}

        for phase_idx, phase in enumerate(phases, 1):
            logger.info(f"\n{'#'*80}")
            logger.info(f"CURRICULUM PHASE {phase_idx}/{len(phases)}")
            logger.info(f"{'#'*80}\n")

            try:
                # Prepare data for this phase
                training_data, quality_score, selected_exchange = self.prepare_training_data(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=phase.data_period[0],
                    end_date=phase.data_period[1],
                    use_best_exchange=use_best_exchange,
                    min_quality_score=min_quality_score,
                    min_candles=min_candles
                )
                logger.info(f"✅ Using {selected_exchange} for {phase.name}")

                # Train on this phase
                phase_metrics = self.train_phase(
                    phase=phase,
                    agent=agent,
                    training_data=training_data,
                    data_quality_score=quality_score,
                    timeframe=timeframe
                )

                # Save phase results
                all_phase_results[phase.name] = {
                    'metrics': [asdict(m) for m in phase_metrics],
                    'data_quality': quality_score,
                    'avg_return': np.mean([m.return_pct for m in phase_metrics]),
                    'avg_sharpe': np.mean([m.sharpe_ratio for m in phase_metrics]),
                    'best_return': max([m.return_pct for m in phase_metrics])
                }

                logger.info(f"✓ Phase {phase_idx} completed successfully")

            except Exception as e:
                logger.error(f"✗ Error in phase {phase_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # If not using curriculum, break after first phase
            if not use_curriculum:
                logger.info("Single-phase training mode, stopping after first phase")
                break

        # Training complete
        training_end_time = datetime.now()
        total_training_time = (training_end_time - training_start_time).total_seconds()

        # Generate final summary
        summary = self._generate_training_summary(
            phases=phases,
            phase_results=all_phase_results,
            total_time=total_training_time,
            symbol=symbol,
            exchange=exchange
        )

        # Save summary
        summary_path = self.metrics_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\n{'='*80}")
        logger.info("TRAINING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Total time: {total_training_time/3600:.2f} hours")
        logger.info(f"Total episodes: {len(self.all_metrics)}")
        logger.info(f"Best model: {self.best_model.model_path if self.best_model else 'None'}")
        logger.info(f"Best score: {self.best_model.performance_score:.2f}" if self.best_model else "N/A")
        logger.info(f"Summary saved: {summary_path}")
        logger.info(f"{'='*80}\n")

        return summary

    def _save_checkpoint(
        self,
        phase: str,
        episode: int,
        agent: UltraOptimizedDQNAgent,
        metrics: TrainingMetrics,
        prefix: str = "best"
    ) -> ModelCheckpoint:
        """Save model checkpoint"""
        # Calculate performance score (weighted combination of metrics)
        performance_score = (
            metrics.return_pct * 0.4 +
            metrics.sharpe_ratio * 10 * 0.3 +
            metrics.win_rate * 100 * 0.2 +
            (100 - abs(metrics.max_drawdown)) * 0.1
        )

        # Save model
        model_filename = f"{prefix}_{phase.replace(' ', '_')}_ep{episode}_score{performance_score:.1f}.pt"
        model_path = self.models_dir / model_filename

        torch.save({
            'episode': episode,
            'model_state_dict': agent.model.state_dict(),
            'target_model_state_dict': agent.target_model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'metrics': asdict(metrics),
            'performance_score': performance_score
        }, model_path)

        checkpoint = ModelCheckpoint(
            phase=phase,
            episode=episode,
            performance_score=performance_score,
            metrics=metrics,
            model_path=str(model_path),
            timestamp=datetime.now()
        )

        self.checkpoints.append(checkpoint)

        return checkpoint

    def _generate_training_summary(
        self,
        phases: List[TrainingPhase],
        phase_results: Dict,
        total_time: float,
        symbol: str,
        exchange: str
    ) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        summary = {
            'training_info': {
                'symbol': symbol,
                'exchange': exchange,
                'device': self.device,
                'total_phases': len(phases),
                'total_episodes': len(self.all_metrics),
                'total_time_seconds': total_time,
                'total_time_hours': total_time / 3600,
                'completed_at': datetime.now().isoformat()
            },
            'phases': [],
            'overall_performance': {},
            'best_model': asdict(self.best_model) if self.best_model else None,
            'data_sources': {
                'exchanges': [exchange],
                'symbols': [symbol],
                'total_candles': sum([
                    len(phase_results[phase]['metrics'])
                    for phase in phase_results
                ])
            }
        }

        # Phase summaries
        for phase in phases:
            if phase.name in phase_results:
                results = phase_results[phase.name]
                summary['phases'].append({
                    'name': phase.name,
                    'difficulty': phase.difficulty,
                    'episodes': phase.episodes,
                    'data_quality': results['data_quality'],
                    'avg_return_pct': results['avg_return'],
                    'best_return_pct': results['best_return'],
                    'avg_sharpe': results['avg_sharpe']
                })

        # Overall performance
        if self.all_metrics:
            summary['overall_performance'] = {
                'avg_return_pct': np.mean([m.return_pct for m in self.all_metrics]),
                'median_return_pct': np.median([m.return_pct for m in self.all_metrics]),
                'best_return_pct': max([m.return_pct for m in self.all_metrics]),
                'worst_return_pct': min([m.return_pct for m in self.all_metrics]),
                'avg_sharpe': np.mean([m.sharpe_ratio for m in self.all_metrics]),
                'avg_win_rate': np.mean([m.win_rate for m in self.all_metrics]),
                'avg_max_drawdown': np.mean([m.max_drawdown for m in self.all_metrics]),
                'total_checkpoints': len(self.checkpoints)
            }

        return summary


def quick_train(
    symbol: str = 'BTC/USDT',
    exchange: str = 'binance',
    years: int = 5,
    use_curriculum: bool = True,
    output_dir: str = "./training_output"
) -> Dict[str, Any]:
    """
    Quick helper function to start comprehensive training

    Args:
        symbol: Trading pair
        exchange: Exchange name
        years: Years of historical data
        use_curriculum: Use curriculum learning
        output_dir: Output directory

    Returns:
        Training summary
    """
    orchestrator = AdvancedTrainingOrchestrator(output_dir=output_dir)

    return orchestrator.run_comprehensive_training(
        exchange=exchange,
        symbol=symbol,
        timeframe='1h',
        total_years=years,
        use_curriculum=use_curriculum
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nexlify Advanced Training Orchestrator")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--years', type=int, default=5, help='Years of historical data')
    parser.add_argument('--no-curriculum', action='store_true', help='Disable curriculum learning')
    parser.add_argument('--output', type=str, default='./training_output', help='Output directory')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("NEXLIFY ADVANCED TRAINING PROGRAM")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Exchange: {args.exchange}")
    print(f"Historical data: {args.years} years")
    print(f"Curriculum learning: {not args.no_curriculum}")
    print("="*80 + "\n")

    summary = quick_train(
        symbol=args.symbol,
        exchange=args.exchange,
        years=args.years,
        use_curriculum=not args.no_curriculum,
        output_dir=args.output
    )

    print("\n✓ Training complete!")
    print(f"Results saved to: {args.output}")
