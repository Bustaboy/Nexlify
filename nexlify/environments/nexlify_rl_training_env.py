#!/usr/bin/env python3
"""
Nexlify RL Training Environment

Gym-style environment for training RL agents with paper trading integration.
Supports both simulated and live market data for continuous learning.

Features:
- OpenAI Gym-compatible interface
- Paper trading integration for risk-free training
- Real market data support
- Comprehensive reward function
- State space with technical indicators
- Episode management and performance tracking
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nexlify.backtesting.nexlify_paper_trading import PaperTradingEngine
from nexlify.ml.nexlify_feature_engineering import FeatureEngineer
from nexlify.config.fee_providers import FeeProvider, FeeEstimate
from nexlify.config.crypto_trading_config import CRYPTO_24_7_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    """Statistics for a single training episode"""

    episode_num: int
    total_reward: float
    final_equity: float
    total_return: float
    total_return_percent: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    steps: int


class TradingEnvironment:
    """
    Gym-style trading environment with paper trading integration

    This environment provides a standard interface for training RL agents
    with realistic trading conditions including fees, slippage, and risk management.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,  # DEPRECATED: use fee_provider instead
        slippage: float = 0.0005,
        state_size: int = 12,  # Updated for crypto-specific features
        action_size: int = 3,
        max_steps: int = 1000,
        use_paper_trading: bool = True,
        market_data: Optional[pd.DataFrame] = None,
        engineer_features: bool = False,
        timeframe: str = "1h",
        use_improved_rewards: bool = True,
        fee_provider: Optional[FeeProvider] = None,  # NEW: dynamic fee support
    ):
        """
        Initialize trading environment

        Args:
            initial_balance: Starting balance
            fee_rate: DEPRECATED - Static trading fee rate (use fee_provider instead)
            slippage: Slippage rate (0.0005 = 0.05%)
            state_size: Size of state vector
            action_size: Number of possible actions
            max_steps: Maximum steps per episode
            use_paper_trading: Whether to use paper trading engine
            market_data: Historical market data for training
            engineer_features: Whether to engineer features from market data
            timeframe: Timeframe of data ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            use_improved_rewards: Use improved reward function (default: True)
        """
        # Environment config
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate  # Kept for backward compatibility
        self.slippage = slippage
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.timeframe = timeframe

        # Fee provider setup (dynamic fees)
        if fee_provider is not None:
            self.fee_provider = fee_provider
        else:
            # Use from global config if available
            self.fee_provider = getattr(CRYPTO_24_7_CONFIG, 'fee_provider', None)

        # Log fee configuration
        if self.fee_provider:
            logger.info(f"Using dynamic fees from: {self.fee_provider.get_network_name()}")
        else:
            logger.info(f"Using static fee rate: {self.fee_rate * 100:.2f}%")

        # Calculate periods per year for Sharpe ratio annualization
        # Crypto markets trade 24/7/365
        timeframe_to_periods = {
            "1m": 525600,  # 365 * 24 * 60
            "5m": 105120,  # 365 * 24 * 12
            "15m": 35040,  # 365 * 24 * 4
            "1h": 8760,  # 365 * 24
            "4h": 2190,  # 365 * 6
            "1d": 365,  # 365
        }
        self.periods_per_year = timeframe_to_periods.get(
            timeframe, 8760
        )  # Default to hourly

        # Paper trading integration
        self.use_paper_trading = use_paper_trading
        if use_paper_trading:
            self.paper_engine = PaperTradingEngine(
                {
                    "paper_balance": initial_balance,
                    "fee_rate": fee_rate,
                    "slippage": slippage,
                }
            )
        else:
            self.paper_engine = None

        # Market data
        self.market_data = market_data
        self.current_step = 0
        self.current_price = 0.0
        self.price_history = []

        # Feature engineering
        self.engineer_features = engineer_features
        if engineer_features:
            self.feature_engineer = FeatureEngineer(enable_sentiment=False)
        else:
            self.feature_engineer = None

        # State tracking
        self.balance = initial_balance
        self.position = 0.0  # Current position size
        self.entry_price = 0.0  # Entry price for current position
        self.equity_curve = [initial_balance]

        # Episode tracking
        self.episode = 0
        self.episode_history: List[EpisodeStats] = []
        self.total_trades = 0
        self.winning_trades = 0

        # Reward function configuration
        self.use_improved_rewards = use_improved_rewards
        self.recent_returns = []  # Track recent returns for volatility calculation
        self.recent_returns_window = 20  # Window for calculating return volatility

        # Action space: 0=Buy, 1=Sell, 2=Hold
        self.ACTION_BUY = 0
        self.ACTION_SELL = 1
        self.ACTION_HOLD = 2

        logger.info(f"🎮 Trading Environment initialized")
        logger.info(f"   Balance: ${initial_balance:,.2f}")
        logger.info(f"   State size: {state_size}, Action size: {action_size}")
        logger.info(
            f"   Paper trading: {'enabled' if use_paper_trading else 'disabled'}"
        )
        logger.info(
            f"   Reward function: {'Improved (risk-adjusted)' if use_improved_rewards else 'Legacy'}"
        )

    def _get_fee_estimate(self, trade_size_usd: float) -> FeeEstimate:
        """
        Get fee estimate for a trade

        Args:
            trade_size_usd: Size of trade in USD

        Returns:
            FeeEstimate with current fees

        Raises:
            RuntimeError: If strict mode enabled and fees cannot be retrieved
        """
        if self.fee_provider is not None:
            return self.fee_provider.get_fee_estimate(trade_size_usd=trade_size_usd)
        else:
            # Check if we're in strict mode
            strict_mode = getattr(self, 'strict_fee_mode', False)
            trading_mode = getattr(self, 'trading_mode', 'backtest')

            if strict_mode or trading_mode == "live":
                raise RuntimeError(
                    "TRADING BLOCKED: Cannot retrieve real-time fees. "
                    f"Trading mode: {trading_mode}, Strict mode: {strict_mode}. "
                    "Fee provider must be configured for live/strict mode trading."
                )

            # Fallback to static fee_rate (ONLY for backtesting)
            logger.warning(f"Using static fallback fees ({self.fee_rate * 100:.2f}%) - only safe for backtesting!")
            return FeeEstimate(
                entry_fee_rate=self.fee_rate,
                exit_fee_rate=self.fee_rate,
                network="static_fallback",
                fee_type="percentage"
            )

    def _run_async_in_context(self, coro):
        """
        Run async coroutine when already in an async context.
        Uses a separate thread to avoid 'event loop already running' error.
        """
        import asyncio
        import concurrent.futures
        import threading

        result = [None]
        exception = [None]

        def run_in_thread():
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result[0] = loop.run_until_complete(coro)
                finally:
                    loop.close()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        if exception[0]:
            raise exception[0]

        return result[0]

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state

        Returns:
            Initial state vector
        """
        # Reset paper trading engine
        if self.use_paper_trading:
            self.paper_engine = PaperTradingEngine(
                {
                    "paper_balance": self.initial_balance,
                    "fee_rate": self.fee_rate,
                    "slippage": self.slippage,
                }
            )

        # Reset state
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.equity_curve = [self.initial_balance]
        self.price_history = []

        # Reset episode tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.recent_returns = []  # Reset returns tracking for volatility calculation

        # Generate or load initial price
        if self.market_data is not None and len(self.market_data) > 0:
            self.current_price = self.market_data.iloc[0]["close"]
        else:
            self.current_price = np.random.uniform(30000, 50000)  # Random BTC price

        self.price_history.append(self.current_price)

        self.episode += 1
        logger.debug(f"Episode {self.episode} started")

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state

        Args:
            action: Action to take (0=Buy, 1=Sell, 2=Hold)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Execute action
        trade_executed = self._execute_action(action)

        # Update market (get next price)
        self._update_market()

        # Calculate reward
        reward = self._calculate_reward(action, trade_executed)

        # Update equity
        current_equity = self._get_current_equity()
        self.equity_curve.append(current_equity)

        # Check if episode is done
        done = self._is_done()

        # Prepare info dict
        info = {
            "step": self.current_step,
            "price": self.current_price,
            "balance": self.balance,
            "position": self.position,
            "equity": current_equity,
            "total_return": current_equity - self.initial_balance,
            "total_return_percent": (
                (current_equity - self.initial_balance) / self.initial_balance
            )
            * 100,
            "trade_executed": trade_executed,
        }

        # If episode done, record stats
        if done:
            self._record_episode_stats(current_equity)

        return self._get_state(), reward, done, info

    def _execute_action(self, action: int) -> bool:
        """
        Execute trading action

        Args:
            action: Action index

        Returns:
            True if trade was executed, False otherwise
        """
        if action == self.ACTION_BUY:
            return self._execute_buy()
        elif action == self.ACTION_SELL:
            return self._execute_sell()
        else:  # HOLD
            return False

    def _execute_buy(self) -> bool:
        """Execute buy action"""
        # Check if already in position
        if self.position > 0:
            return False

        # Calculate position size (use 95% of balance to leave room for fees)
        max_cost = self.balance * 0.95
        amount = max_cost / (self.current_price * (1 + self.fee_rate + self.slippage))

        if amount <= 0:
            return False

        # Execute in paper trading engine if enabled
        if self.use_paper_trading:
            import asyncio
            import inspect

            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use asyncio.ensure_future and wait
                # Since we can't await in a non-async function, we need to run synchronously
                # The best approach is to make the paper trading calls synchronous
                result = self._run_async_in_context(
                    self.paper_engine.place_order(
                        "BTC/USDT",
                        "buy",
                        amount,
                        self.current_price,
                        strategy="rl_training",
                    )
                )
            except RuntimeError:
                # No event loop running - create one and run
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.paper_engine.place_order(
                            "BTC/USDT",
                            "buy",
                            amount,
                            self.current_price,
                            strategy="rl_training",
                        )
                    )
                finally:
                    loop.close()

            if not result.get("success"):
                return False

            # Update local state from paper engine
            self.balance = self.paper_engine.current_balance
            self.position = amount
            self.entry_price = self.current_price
        else:
            # Manual execution
            cost = amount * self.current_price
            fees = cost * self.fee_rate
            total_cost = cost + fees

            if total_cost > self.balance:
                return False

            self.balance -= total_cost
            self.position = amount
            self.entry_price = self.current_price

        self.total_trades += 1
        logger.debug(f"BUY executed: {amount:.4f} @ ${self.current_price:.2f}")
        return True

    def _execute_sell(self) -> bool:
        """Execute sell action"""
        # Check if in position
        if self.position <= 0:
            return False

        amount = self.position

        # Execute in paper trading engine if enabled
        if self.use_paper_trading:
            import asyncio

            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use helper
                result = self._run_async_in_context(
                    self.paper_engine.place_order(
                        "BTC/USDT",
                        "sell",
                        amount,
                        self.current_price,
                        strategy="rl_training",
                    )
                )
            except RuntimeError:
                # No event loop running - create one and run
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.paper_engine.place_order(
                            "BTC/USDT",
                            "sell",
                            amount,
                            self.current_price,
                            strategy="rl_training",
                        )
                    )
                finally:
                    loop.close()

            if not result.get("success"):
                return False

            # Check if profitable
            pnl = result.get("pnl", 0)
            if pnl > 0:
                self.winning_trades += 1

            # Update local state from paper engine
            self.balance = self.paper_engine.current_balance
            self.position = 0
            self.entry_price = 0
        else:
            # Manual execution
            proceeds = amount * self.current_price
            fees = proceeds * self.fee_rate
            net_proceeds = proceeds - fees

            # Check if profitable
            cost = amount * self.entry_price
            if net_proceeds > cost:
                self.winning_trades += 1

            self.balance += net_proceeds
            self.position = 0
            self.entry_price = 0

        self.total_trades += 1
        logger.debug(f"SELL executed: {amount:.4f} @ ${self.current_price:.2f}")
        return True

    def _update_market(self):
        """Update market with next price"""
        self.current_step += 1

        if self.market_data is not None and self.current_step < len(self.market_data):
            # Use historical data
            self.current_price = self.market_data.iloc[self.current_step]["close"]
        else:
            # Simulate price movement (random walk with slight upward bias)
            change_percent = np.random.normal(0.0001, 0.02)  # 0.01% mean, 2% std
            self.current_price *= 1 + change_percent

        self.price_history.append(self.current_price)

        # Update paper trading positions if enabled
        if self.use_paper_trading and self.position > 0:
            import asyncio

            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use helper
                self._run_async_in_context(
                    self.paper_engine.update_positions({"BTC/USDT": self.current_price})
                )
            except RuntimeError:
                # No event loop running - create one and run
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self.paper_engine.update_positions({"BTC/USDT": self.current_price})
                    )
                finally:
                    loop.close()

    def _calculate_reward(self, action: int, trade_executed: bool) -> float:
        """
        Calculate reward for current step

        Uses improved reward function if enabled, otherwise uses legacy version.

        Args:
            action: Action taken
            trade_executed: Whether trade was actually executed

        Returns:
            Reward value
        """
        if self.use_improved_rewards:
            return self._calculate_improved_reward(action, trade_executed)
        else:
            return self._calculate_legacy_reward(action, trade_executed)

    def _calculate_improved_reward(self, action: int, trade_executed: bool) -> float:
        """
        Improved reward function with risk-adjusted returns

        Key improvements:
        1. Risk-adjusted equity growth (Sharpe-inspired)
        2. Proper transaction cost penalty (realistic, not arbitrary)
        3. Win rate bonus (encourages quality trades)
        4. Consistency reward (penalizes excessive volatility)
        5. No conflicting signals

        Args:
            action: Action taken
            trade_executed: Whether trade was actually executed

        Returns:
            Normalized reward value
        """
        reward = 0.0
        current_equity = self._get_current_equity()

        # 1. PRIMARY REWARD: Equity percentage change (risk-adjusted)
        if len(self.equity_curve) > 1:
            equity_return = (
                current_equity - self.equity_curve[-1]
            ) / self.equity_curve[-1]

            # Track returns for volatility calculation
            self.recent_returns.append(equity_return)
            if len(self.recent_returns) > self.recent_returns_window:
                self.recent_returns.pop(0)

            # Risk-adjusted reward (higher reward for consistent gains, lower for volatile swings)
            if len(self.recent_returns) >= 5:
                returns_std = np.std(self.recent_returns)
                # Normalize by volatility (Sharpe-style): reward / risk
                # Add small epsilon to avoid division by zero
                risk_adjusted_return = equity_return / (returns_std + 1e-6)
                reward += risk_adjusted_return * 10.0  # Scale to reasonable range
            else:
                # Not enough data for risk adjustment, use raw return
                reward += equity_return * 100.0  # Scale to percentage

        # 2. TRANSACTION COST: Realistic penalty based on actual fees
        if trade_executed:
            # Actual cost is ~0.15% (0.1% fee + 0.05% slippage)
            # Penalty should reflect this realistically
            transaction_cost_pct = (self.fee_rate + self.slippage) * 100
            reward -= transaction_cost_pct  # Penalty proportional to actual cost

        # 3. WIN RATE BONUS: Reward for profitable trades
        if trade_executed and self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
            # Bonus for maintaining good win rate (0.5 = break-even)
            if win_rate > 0.5:
                reward += (win_rate - 0.5) * 2.0  # Max +1.0 for 100% win rate

        # 4. CONSISTENCY REWARD: Penalize excessive volatility
        if len(self.recent_returns) >= 10:
            returns_volatility = np.std(self.recent_returns)
            # Penalize high volatility (want stable growth)
            if returns_volatility > 0.02:  # More than 2% volatility
                reward -= (returns_volatility - 0.02) * 50.0

        # 5. POSITION QUALITY: Small incentive for being in market during strong trends
        # (This encourages participation without conflicting with transaction costs)
        if self.position > 0 and len(self.price_history) >= 5:
            # Check if price is trending
            recent_prices = self.price_history[-5:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            if price_trend > 0:  # Uptrend
                reward += min(price_trend * 5.0, 0.5)  # Small bonus, capped

        return reward

    def _calculate_legacy_reward(self, action: int, trade_executed: bool) -> float:
        """
        Legacy reward function (original version)

        Kept for A/B testing comparison.
        """
        reward = 0.0
        current_equity = self._get_current_equity()

        # Equity change reward (normalized)
        if len(self.equity_curve) > 1:
            equity_change = current_equity - self.equity_curve[-1]
            reward += equity_change / self.initial_balance * 100  # Scale to percentage

        # Penalize transaction costs
        if trade_executed:
            transaction_cost = self.fee_rate + self.slippage
            reward -= transaction_cost * 10  # Penalty for trading

        # Reward for holding profitable positions
        if self.position > 0 and self.current_price > self.entry_price:
            unrealized_gain = (self.current_price - self.entry_price) / self.entry_price
            reward += unrealized_gain * 5  # Reward unrealized gains

        # Penalize holding losing positions
        if self.position > 0 and self.current_price < self.entry_price:
            unrealized_loss = (self.entry_price - self.current_price) / self.entry_price
            reward -= (
                unrealized_loss * 3
            )  # Smaller penalty to encourage holding through dips

        # Small penalty for doing nothing when not in position
        if action == self.ACTION_HOLD and self.position == 0:
            reward -= 0.01

        return reward

    def _get_current_equity(self) -> float:
        """Calculate current total equity"""
        if self.use_paper_trading:
            return self.paper_engine.get_total_equity({"BTC/USDT": self.current_price})
        else:
            return self.balance + (self.position * self.current_price)

    def _is_done(self) -> bool:
        """Check if episode is complete"""
        # Max steps reached
        if self.current_step >= self.max_steps:
            return True

        # Out of market data
        if (
            self.market_data is not None
            and self.current_step >= len(self.market_data) - 1
        ):
            return True

        # Catastrophic loss (lost 90% of capital)
        if self._get_current_equity() < self.initial_balance * 0.1:
            logger.warning(f"Episode {self.episode} terminated: catastrophic loss")
            return True

        return False

    def _get_state(self) -> np.ndarray:
        """
        Get current state vector with crypto-specific features

        Returns:
            State vector with normalized features (12 features)
        """
        current_equity = self._get_current_equity()

        # Calculate technical indicators from price history
        rsi = self._calculate_rsi()
        macd = self._calculate_macd()
        volatility = self._calculate_volatility()

        # Calculate price change
        if len(self.price_history) > 1:
            price_change = (
                self.current_price - self.price_history[-2]
            ) / self.price_history[-2]
        else:
            price_change = 0.0

        # Crypto-specific features
        momentum = self._calculate_momentum()
        vol_clustering = self._calculate_volatility_clustering()
        drawdown = self._calculate_drawdown()
        sharpe = self._calculate_sharpe_from_equity()

        # Build state vector (12 features - crypto optimized)
        state = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                self.position
                / (self.initial_balance / self.current_price),  # Normalized position
                (
                    (self.entry_price / self.current_price)
                    if self.entry_price > 0
                    else 1.0
                ),  # Relative entry price
                self.current_price / self.initial_balance,  # Normalized price
                price_change,  # Price change
                rsi / 100,  # Normalized RSI
                macd,  # MACD (already normalized)
                volatility,  # Volatility
                momentum,  # Momentum (NEW)
                vol_clustering,  # Volatility clustering (NEW)
                drawdown,  # Drawdown from peak (NEW)
                sharpe,  # Sharpe ratio (NEW)
            ]
        )

        return state.astype(np.float32)

    def _calculate_rsi(self, period: int = 14) -> float:
        """Calculate RSI from price history"""
        if len(self.price_history) < period + 1:
            return 50.0  # Neutral RSI

        prices = np.array(self.price_history[-(period + 1) :])
        deltas = np.diff(prices)

        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self) -> float:
        """Calculate MACD from price history"""
        if len(self.price_history) < 26:
            return 0.0

        prices = np.array(self.price_history[-26:])

        # Simple implementation (not EMA, but fast)
        ema_12 = np.mean(prices[-12:])
        ema_26 = np.mean(prices)

        macd = (ema_12 - ema_26) / ema_26  # Normalized

        return macd

    def _calculate_volatility(self, period: int = 10) -> float:
        """Calculate price volatility"""
        if len(self.price_history) < period + 1:
            return 0.0

        prices = np.array(self.price_history[-(period + 1):])
        returns = np.diff(prices) / prices[:-1]

        return float(np.std(returns))

    def _calculate_momentum(self, period: int = 20) -> float:
        """
        Calculate price momentum (rate of change)
        Crypto markets show strong momentum effects
        """
        if len(self.price_history) < period + 1:
            return 0.0

        current_price = self.price_history[-1]
        past_price = self.price_history[-(period + 1)]

        if past_price == 0:
            return 0.0

        momentum = (current_price - past_price) / past_price
        return float(np.clip(momentum, -1, 1))

    def _calculate_volatility_clustering(self) -> float:
        """
        Calculate volatility clustering (GARCH-like effect)
        Crypto exhibits strong volatility clustering
        """
        if len(self.price_history) < 40:
            return 0.0

        # Recent volatility (10 periods)
        recent_vol = self._calculate_volatility(10)

        # Historical volatility (30 periods)
        if len(self.price_history) < 31:
            return 0.0

        prices = np.array(self.price_history[-31:])
        returns = np.diff(prices) / prices[:-1]
        historical_vol = np.std(returns)

        if historical_vol == 0:
            return 0.0

        vol_ratio = recent_vol / historical_vol
        return float(np.clip(vol_ratio - 1, -1, 1))

    def _calculate_drawdown(self) -> float:
        """
        Calculate current drawdown from peak equity
        Important risk metric for crypto trading
        """
        if len(self.equity_curve) < 2:
            return 0.0

        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        current_equity = equity_array[-1]
        peak_equity = running_max[-1]

        if peak_equity == 0:
            return 0.0

        drawdown = (peak_equity - current_equity) / peak_equity
        return float(np.clip(drawdown, 0, 1))

    def _calculate_sharpe_from_equity(self, window: int = 50) -> float:
        """
        Calculate rolling Sharpe ratio from equity curve
        Risk-adjusted return metric crucial for crypto
        """
        if len(self.equity_curve) < 10:
            return 0.0

        recent_equity = self.equity_curve[-window:] if len(self.equity_curve) >= window else self.equity_curve

        if len(recent_equity) < 2:
            return 0.0

        returns = np.diff(recent_equity) / np.array(recent_equity[:-1])

        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe
        sharpe = (mean_return / std_return) * np.sqrt(self.periods_per_year)
        return float(np.clip(sharpe, -3, 3))

    def _record_episode_stats(self, final_equity: float):
        """Record statistics for completed episode"""
        # Calculate metrics
        total_return = final_equity - self.initial_balance
        total_return_percent = (total_return / self.initial_balance) * 100
        win_rate = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0
        )

        # Calculate Sharpe ratio (annualized based on timeframe)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = (
                np.mean(returns) / np.std(returns) * np.sqrt(self.periods_per_year)
                if np.std(returns) > 0
                else 0
            )
        else:
            sharpe = 0

        # Calculate max drawdown
        equity_curve = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        stats = EpisodeStats(
            episode_num=self.episode,
            total_reward=total_return,  # Fixed: was incorrectly summing all equity values
            final_equity=final_equity,
            total_return=total_return,
            total_return_percent=total_return_percent,
            num_trades=self.total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown * 100,  # Convert to percentage
            steps=self.current_step,
        )

        self.episode_history.append(stats)

        logger.info(
            f"Episode {self.episode} completed: "
            f"Return={total_return_percent:.2f}%, "
            f"Trades={self.total_trades}, "
            f"Win Rate={win_rate:.1f}%"
        )

    def get_episode_stats(self) -> List[EpisodeStats]:
        """Get statistics for all episodes"""
        return self.episode_history

    def render(self, mode: str = "human"):
        """
        Render environment state

        Args:
            mode: Render mode ('human' for console output)
        """
        if mode == "human":
            current_equity = self._get_current_equity()
            print(f"\nStep: {self.current_step}/{self.max_steps}")
            print(f"Price: ${self.current_price:,.2f}")
            print(f"Balance: ${self.balance:,.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Equity: ${current_equity:,.2f}")
            print(
                f"Return: {((current_equity - self.initial_balance) / self.initial_balance * 100):.2f}%"
            )


# Convenience function
def create_training_environment(
    initial_balance: float = 10000.0,
    use_paper_trading: bool = True,
    market_data: Optional[pd.DataFrame] = None,
) -> TradingEnvironment:
    """
    Create trading environment with default settings

    Args:
        initial_balance: Starting balance
        use_paper_trading: Whether to use paper trading engine
        market_data: Historical market data

    Returns:
        Initialized TradingEnvironment
    """
    return TradingEnvironment(
        initial_balance=initial_balance,
        use_paper_trading=use_paper_trading,
        market_data=market_data,
    )


__all__ = ["TradingEnvironment", "EpisodeStats", "create_training_environment"]
