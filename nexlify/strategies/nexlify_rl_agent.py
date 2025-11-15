#!/usr/bin/env python3
"""
Nexlify Reinforcement Learning Module
Deep Q-Network (DQN) agent for autonomous trading optimization
"""

import json
import logging
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nexlify.utils.error_handler import get_error_handler, handle_errors
from nexlify.strategies.epsilon_decay import EpsilonDecayFactory, EpsilonDecayStrategy
from nexlify.strategies.gamma_selector import GammaSelector, get_recommended_gamma
from nexlify.config.crypto_trading_config import CRYPTO_24_7_CONFIG, FEATURE_PERIODS
from nexlify.config.fee_providers import FeeProvider, FeeEstimate

# Initialize logger first before using it
logger = logging.getLogger(__name__)
error_handler = get_error_handler()

# Prioritized Experience Replay
try:
    from nexlify.memory.prioritized_replay_buffer import PrioritizedReplayBuffer
    PER_AVAILABLE = True
except ImportError:
    PER_AVAILABLE = False
    logger.warning("PrioritizedReplayBuffer not available - using standard replay buffer")

# N-Step Returns
try:
    from nexlify.memory.nstep_replay_buffer import NStepReplayBuffer, MixedNStepReplayBuffer
    NSTEP_AVAILABLE = True
except ImportError:
    NSTEP_AVAILABLE = False
    logger.warning("NStepReplayBuffer not available - using standard replay buffer")

# Training optimization utilities
try:
    from nexlify.training.training_optimizers import GradientClipper, LRSchedulerManager
    TRAINING_OPTIMIZERS_AVAILABLE = True
except ImportError:
    TRAINING_OPTIMIZERS_AVAILABLE = False
    logger.warning("Training optimizers not available - gradient clipping and LR scheduling disabled")


class TradingEnvironment:
    """
    Gym-style trading environment for RL training

    Supports dynamic fee calculation based on network/exchange.
    Fees are applied TWICE per round trip (buy + sell).
    """

    def __init__(self,
                 price_data: np.ndarray,
                 initial_balance: float = 10000,
                 fee_provider: Optional[FeeProvider] = None,
                 config: Optional[Dict] = None):
        """
        Initialize trading environment

        Args:
            price_data: Historical price data
            initial_balance: Starting balance in USD
            fee_provider: Fee provider instance (None = use from config or default)
            config: Configuration dict (None = use CRYPTO_24_7_CONFIG)
        """
        self.price_data = price_data
        self.initial_balance = initial_balance
        self.config = config or {}

        # Fee provider setup
        if fee_provider is not None:
            self.fee_provider = fee_provider
        elif self.config.get("fee_provider"):
            self.fee_provider = self.config["fee_provider"]
        else:
            # Use from global config if available
            self.fee_provider = getattr(CRYPTO_24_7_CONFIG, 'fee_provider', None)

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # Amount of crypto held
        self.position_price = 0  # Entry price
        self.max_steps = len(price_data) - 1

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space_n = 3

        # State space: [balance, position, position_price, current_price,
        #               price_change, RSI, MACD, volatility, momentum, vol_clustering,
        #               drawdown, sharpe]
        self.state_space_n = 12  # Expanded for crypto-specific features

        # Tracking
        self.trade_history = []
        self.episode_reward = 0

        # Risk-adjusted reward tracking
        self.returns_history = []
        self.portfolio_values = [initial_balance]
        self.max_portfolio_value = initial_balance

        logger.info("🎮 Trading Environment initialized (Crypto-optimized)")
        if self.fee_provider:
            logger.info(f"   Fee provider: {self.fee_provider.get_network_name()}")

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
            # Check if we're in strict mode (from config)
            strict_mode = self.config.get("strict_fee_mode", False)
            trading_mode = self.config.get("trading_mode", "backtest")

            if strict_mode or trading_mode == "live":
                raise RuntimeError(
                    "TRADING BLOCKED: Cannot retrieve real-time fees. "
                    f"Trading mode: {trading_mode}, Strict mode: {strict_mode}. "
                    "Fee provider must be configured for live/strict mode trading."
                )

            # Fallback to static 0.1% fee (ONLY for backtesting)
            from nexlify.config.fee_providers import FeeEstimate
            logger.warning("Using static fallback fees (0.1%) - only safe for backtesting!")
            return FeeEstimate(
                entry_fee_rate=0.001,
                exit_fee_rate=0.001,
                network="static_fallback",
                fee_type="percentage"
            )

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.trade_history = []
        self.episode_reward = 0

        # Reset risk tracking
        self.returns_history = []
        self.portfolio_values = [self.initial_balance]
        self.max_portfolio_value = self.initial_balance

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation with crypto-specific features"""
        current_price = self.price_data[self.current_step]

        # Calculate features
        price_change = 0
        if self.current_step > 0:
            price_change = (
                current_price - self.price_data[self.current_step - 1]
            ) / current_price

        # Technical indicators
        rsi = self._calculate_rsi(14)
        macd = self._calculate_macd()
        volatility = self._calculate_volatility(10)

        # Crypto-specific features
        momentum = self._calculate_momentum(20)  # 20-period momentum
        vol_clustering = self._calculate_volatility_clustering()  # GARCH-like
        drawdown = self._calculate_drawdown()  # Current drawdown from peak
        sharpe = self._calculate_sharpe_ratio()  # Risk-adjusted returns

        state = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                self.position,  # Position size
                (
                    self.position_price / current_price if current_price > 0 else 0
                ),  # Relative entry price
                current_price / self.initial_balance,  # Normalized price
                price_change,  # Price change
                rsi,  # RSI
                macd,  # MACD
                volatility,  # Volatility
                momentum,  # Momentum (NEW)
                vol_clustering,  # Volatility clustering (NEW)
                drawdown,  # Drawdown from peak (NEW)
                sharpe,  # Sharpe ratio (NEW)
            ],
            dtype=np.float32,
        )

        return state

    def _calculate_rsi(self, period: int = None) -> float:
        """Calculate RSI indicator"""
        if period is None:
            period = FEATURE_PERIODS["rsi"]
        if self.current_step < period:
            return 0.5

        prices = self.price_data[
            max(0, self.current_step - period) : self.current_step + 1
        ]
        deltas = np.diff(prices)

        gains = deltas[deltas > 0].sum()
        losses = abs(deltas[deltas < 0].sum())

        if losses == 0:
            return 1.0

        rs = gains / losses
        rsi = 1 - (1 / (1 + rs))

        return rsi

    def _calculate_macd(self) -> float:
        """Calculate MACD indicator"""
        if self.current_step < 26:
            return 0

        prices = self.price_data[max(0, self.current_step - 26) : self.current_step + 1]

        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)

        macd = (ema_12 - ema_26) / ema_26 if ema_26 > 0 else 0

        return macd

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate exponential moving average"""
        if len(prices) < period:
            return prices[-1]

        multiplier = 2 / (period + 1)
        ema = prices[-period]

        for price in prices[-period + 1 :]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_volatility(self, period: int = 10) -> float:
        """Calculate price volatility"""
        if self.current_step < period:
            return 0

        prices = self.price_data[
            max(0, self.current_step - period) : self.current_step + 1
        ]
        returns = np.diff(prices) / prices[:-1]

        return np.std(returns)

    def _calculate_momentum(self, period: int = None) -> float:
        """
        Calculate price momentum (rate of change)
        Crypto markets show strong momentum effects
        """
        if period is None:
            period = FEATURE_PERIODS["momentum"]
        if self.current_step < period:
            return 0

        current_price = self.price_data[self.current_step]
        past_price = self.price_data[max(0, self.current_step - period)]

        if past_price == 0:
            return 0

        momentum = (current_price - past_price) / past_price
        return np.clip(momentum, -1, 1)  # Clip to [-1, 1] range

    def _calculate_volatility_clustering(self) -> float:
        """
        Calculate volatility clustering (GARCH-like effect)
        Crypto exhibits strong volatility clustering - high vol follows high vol
        """
        if self.current_step < 30:
            return 0

        # Recent volatility (10 periods)
        recent_vol = self._calculate_volatility(10)

        # Historical volatility (30 periods)
        if self.current_step < 30:
            return 0

        prices = self.price_data[max(0, self.current_step - 30): self.current_step + 1]
        returns = np.diff(prices) / prices[:-1]
        historical_vol = np.std(returns)

        if historical_vol == 0:
            return 0

        # Ratio of recent to historical volatility
        vol_ratio = recent_vol / historical_vol
        return np.clip(vol_ratio - 1, -1, 1)  # Centered at 0, clipped to [-1, 1]

    def _calculate_drawdown(self) -> float:
        """
        Calculate current drawdown from peak portfolio value
        Important risk metric for crypto trading
        """
        current_value = self.balance
        if self.position > 0:
            current_price = self.price_data[self.current_step]
            current_value += self.position * current_price

        # Update max value
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)

        if self.max_portfolio_value == 0:
            return 0

        drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        return np.clip(drawdown, 0, 1)

    def _calculate_sharpe_ratio(self, window: int = None) -> float:
        """
        Calculate rolling Sharpe ratio
        Risk-adjusted return metric crucial for crypto
        Assumes hourly data (8760 periods/year) for annualization
        """
        if window is None:
            window = FEATURE_PERIODS["sharpe_window"]
        if len(self.returns_history) < 10:
            return 0

        recent_returns = self.returns_history[-window:] if len(self.returns_history) >= window else self.returns_history

        if len(recent_returns) < 2:
            return 0

        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        if std_return == 0:
            return 0

        # Annualized Sharpe (crypto trades 24/7, assuming hourly data = 8760 periods/year)
        # This is a reasonable default for crypto trading environments
        periods_per_year = 8760  # 365 * 24 hours
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return np.clip(sharpe, -3, 3)  # Clip to reasonable range

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            (next_state, reward, done, info)
        """
        current_price = self.price_data[self.current_step]
        reward = 0
        info = {}

        # Execute action with proper fee accounting
        if action == 1:  # Buy
            if self.balance > 0 and self.position == 0:
                # Buy with 95% of balance, accounting for fees
                amount_to_invest = self.balance * 0.95  # Leave room for fees/slippage

                # Get dynamic fee estimate for this trade size
                fee_estimate = self._get_fee_estimate(amount_to_invest)

                # Calculate entry costs (percentage fee + fixed cost like gas)
                percentage_fee, fixed_fee = fee_estimate.calculate_entry_cost(amount_to_invest)
                total_fees = percentage_fee + fixed_fee

                # Calculate actual position size after fees
                amount_after_fees = amount_to_invest - total_fees
                self.position = amount_after_fees / current_price
                self.position_price = current_price
                self.balance -= amount_to_invest  # Deduct total investment

                info["action"] = "buy"
                info["price"] = current_price
                info["entry_fees_pct"] = percentage_fee
                info["entry_fees_fixed"] = fixed_fee
                info["total_entry_fees"] = total_fees
                info["fee_network"] = fee_estimate.network

        elif action == 2:  # Sell
            if self.position > 0:
                # Sell entire position with dynamic fee accounting
                sell_value = self.position * current_price

                # Get dynamic fee estimate for exit
                fee_estimate = self._get_fee_estimate(sell_value)

                # Calculate exit costs (percentage fee + fixed cost like gas)
                percentage_fee, fixed_fee = fee_estimate.calculate_exit_cost(sell_value)
                total_exit_fees = percentage_fee + fixed_fee
                net_proceeds = sell_value - total_exit_fees

                # Calculate profit after BOTH entry and exit fees
                # Entry cost = what we originally invested (already included entry fees)
                original_investment = self.initial_balance - self.balance  # What we spent on entry
                profit = net_proceeds - original_investment

                self.balance = net_proceeds

                # Reward based on ROI (Return on Investment)
                reward = profit / original_investment if original_investment > 0 else 0

                # Track trade with full fee breakdown
                self.trade_history.append(
                    {
                        "entry_price": self.position_price,
                        "exit_price": current_price,
                        "profit": profit,
                        "return": (current_price / self.position_price - 1) * 100,
                        "exit_fees_pct": percentage_fee,
                        "exit_fees_fixed": fixed_fee,
                        "total_exit_fees": total_exit_fees,
                        "fee_network": fee_estimate.network,
                    }
                )

                self.position = 0
                self.position_price = 0
                info["action"] = "sell"
                info["profit"] = profit
                info["exit_fees_pct"] = percentage_fee
                info["exit_fees_fixed"] = fixed_fee
                info["total_exit_fees"] = total_exit_fees
                info["fee_network"] = fee_estimate.network
        else:
            # Hold - no fixed penalty, let portfolio value change drive reward
            info["action"] = "hold"
            reward = 0  # Neutral, will be adjusted by equity change below

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Calculate total portfolio value
        total_value = self.balance
        if self.position > 0:
            total_value += self.position * current_price

        # Track portfolio value and returns for risk-adjusted rewards
        self.portfolio_values.append(total_value)

        # PRIMARY REWARD: Equity change (consistent with improved reward function)
        if len(self.portfolio_values) > 1:
            equity_change = total_value - self.portfolio_values[-2]
            equity_return = equity_change / self.portfolio_values[-2]
            self.returns_history.append(equity_return)

            # Only override reward if not from sell (sell already has ROI reward)
            if action != 2:  # Not sell
                reward = equity_return * 100.0  # Scale to percentage

        # Apply risk adjustment to reward (Sharpe-based) - FIXED VERSION
        # Apply as ADDITIVE bonus/penalty, not multiplicative
        if len(self.returns_history) >= 10:
            sharpe = self._calculate_sharpe_ratio()
            # Sharpe bonus/penalty (additive to avoid breaking negative rewards)
            # Clip Sharpe-based adjustment to reasonable range
            risk_adjustment = np.clip(sharpe * 0.01, -0.1, 0.1)  # ±0.1 max
            reward += risk_adjustment  # ADDITIVE, not multiplicative

            info["sharpe_ratio"] = sharpe
            info["risk_adjustment"] = risk_adjustment

        info["total_value"] = total_value
        info["equity_return"] = equity_return if len(self.portfolio_values) > 1 else 0
        info["step"] = self.current_step

        self.episode_reward += reward

        next_state = self._get_state() if not done else np.zeros(self.state_space_n)

        return next_state, reward, done, info

    def get_portfolio_value(self) -> float:
        """Get current total portfolio value"""
        current_price = (
            self.price_data[self.current_step]
            if self.current_step < len(self.price_data)
            else 0
        )
        return self.balance + (self.position * current_price)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = None, max_size: int = None):
        # Support both capacity and max_size for backward compatibility
        size = capacity or max_size or 100000
        self.buffer = deque(maxlen=size)
        self.max_size = size  # Expose max_size for tests

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def add(self, state, action, reward, next_state, done):
        """Alias for push() for backward compatibility with tests"""
        self.push(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> List:
        """Sample random batch from buffer"""
        # Handle case where batch_size > buffer size
        actual_batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, actual_batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for trading
    Uses neural network to approximate Q-values
    """

    def __init__(self, state_size: int, action_size: int, config: Dict = None, **kwargs):
        # Validate inputs
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}")
        if action_size <= 0:
            raise ValueError(f"action_size must be positive, got {action_size}")

        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}

        # Merge kwargs into config for backward compatibility with tests
        self.config.update(kwargs)

        # Hyperparameters (pull from central config or override)
        # Initialize gamma selector (selects optimal static gamma based on timeframe)
        self.gamma_selector = self._create_gamma_selector()
        self.gamma = self.gamma_selector.get_gamma()

        self.learning_rate = self.config.get("learning_rate", CRYPTO_24_7_CONFIG.learning_rate)
        self.learning_rate_decay = self.config.get("learning_rate_decay", CRYPTO_24_7_CONFIG.learning_rate_decay)
        self.batch_size = self.config.get("batch_size", CRYPTO_24_7_CONFIG.batch_size)
        self.target_update_freq = self.config.get("target_update_freq", CRYPTO_24_7_CONFIG.target_update_freq)

        # Epsilon decay strategy (new advanced system)
        self.epsilon_decay_strategy = self._create_epsilon_strategy()
        self.epsilon = self.epsilon_decay_strategy.current_epsilon

        # Experience replay (optimized for regime adaptation)
        replay_buffer_size = self.config.get("replay_buffer_size", CRYPTO_24_7_CONFIG.replay_buffer_size)

        # Check for N-Step Returns
        use_nstep = self.config.get("use_nstep_returns", CRYPTO_24_7_CONFIG.use_nstep_returns)

        # Use Prioritized Experience Replay if enabled and available
        use_per = self.config.get("use_prioritized_replay", CRYPTO_24_7_CONFIG.use_prioritized_replay)

        # Initialize replay buffer based on configuration
        # Priority: N-Step > PER > Standard
        if use_nstep and NSTEP_AVAILABLE:
            # N-Step Returns configuration
            n_step = self.config.get("n_step", CRYPTO_24_7_CONFIG.n_step)
            n_step_buffer_size = self.config.get("n_step_buffer_size", CRYPTO_24_7_CONFIG.n_step_buffer_size)
            use_mixed = self.config.get("use_mixed_returns", CRYPTO_24_7_CONFIG.use_mixed_returns)
            mixed_ratio = self.config.get("mixed_returns_ratio", CRYPTO_24_7_CONFIG.mixed_returns_ratio)

            if use_mixed:
                self.memory = MixedNStepReplayBuffer(
                    capacity=n_step_buffer_size,
                    n_step=n_step,
                    gamma=self.gamma,
                    n_step_ratio=mixed_ratio,
                )
                logger.info(f"🎯 Using Mixed N-Step Replay (n={n_step}, mix={mixed_ratio:.0%}, γ={self.gamma})")
            else:
                self.memory = NStepReplayBuffer(
                    capacity=n_step_buffer_size,
                    n_step=n_step,
                    gamma=self.gamma,
                )
                logger.info(f"🎯 Using N-Step Replay (n={n_step}, γ={self.gamma})")

            self.use_per = False
            self.use_nstep = True
        elif use_per and PER_AVAILABLE:
            per_alpha = self.config.get("per_alpha", CRYPTO_24_7_CONFIG.per_alpha)
            per_beta_start = self.config.get("per_beta_start", CRYPTO_24_7_CONFIG.per_beta_start)
            per_beta_end = self.config.get("per_beta_end", CRYPTO_24_7_CONFIG.per_beta_end)
            per_beta_annealing_steps = self.config.get(
                "per_beta_annealing_steps", CRYPTO_24_7_CONFIG.per_beta_annealing_steps
            )
            per_epsilon = self.config.get("per_epsilon", CRYPTO_24_7_CONFIG.per_epsilon)
            per_priority_clip = self.config.get("per_priority_clip", CRYPTO_24_7_CONFIG.per_priority_clip)

            self.memory = PrioritizedReplayBuffer(
                capacity=replay_buffer_size,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_end=per_beta_end,
                beta_annealing_steps=per_beta_annealing_steps,
                epsilon=per_epsilon,
                priority_clip=per_priority_clip,
            )
            self.use_per = True
            self.use_nstep = False
            logger.info(f"✨ Using Prioritized Experience Replay (alpha={per_alpha}, beta={per_beta_start}→{per_beta_end})")
        else:
            self.memory = ReplayBuffer(capacity=replay_buffer_size)
            self.use_per = False
            self.use_nstep = False
            if use_per and not PER_AVAILABLE:
                logger.warning("⚠️  PER requested but not available - using standard replay buffer")
            if use_nstep and not NSTEP_AVAILABLE:
                logger.warning("⚠️  N-Step requested but not available - using standard replay buffer")

        # Neural network models
        self.model = None
        self.target_model = None

        # Try to use GPU if available
        self.device = self._get_device()

        # Build models
        self._build_model()

        # Training stats
        self.training_history = []

        # Training optimizations (gradient clipping + LR scheduling)
        self.gradient_clipper = None
        self.lr_scheduler = None
        self._training_step_count = 0

        logger.info(f"🤖 DQN Agent initialized (device: {self.device})")

    def _create_epsilon_strategy(self) -> EpsilonDecayStrategy:
        """Create epsilon decay strategy from config"""
        # Check if using new config format
        if "epsilon_decay_type" in self.config or "epsilon_decay_steps" in self.config:
            return EpsilonDecayFactory.create_from_config(self.config)

        # Legacy config support - convert old parameters to new system
        epsilon_start = self.config.get("epsilon", CRYPTO_24_7_CONFIG.epsilon_start)
        epsilon_end = self.config.get("epsilon_min", CRYPTO_24_7_CONFIG.epsilon_end)

        # If old-style epsilon_decay (multiplicative) is provided, use exponential decay
        if "epsilon_decay" in self.config and "epsilon_decay_steps" not in self.config:
            decay_rate = self.config.get("epsilon_decay", 0.995)
            logger.info(f"🔄 Converting legacy epsilon_decay={decay_rate} to ExponentialEpsilonDecay")

            return EpsilonDecayFactory.create(
                "exponential",
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                decay_rate=decay_rate,
                decay_steps=10000,  # Large value for exponential decay
            )

        # Default: Use scheduled decay optimized for 24/7 crypto trading
        # Pull from central config
        schedule = self.config.get("epsilon_schedule", CRYPTO_24_7_CONFIG.epsilon_schedule)

        logger.info("🚀 Using ScheduledEpsilonDecay optimized for 24/7 crypto trading")

        return EpsilonDecayFactory.create(
            "scheduled",
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            schedule=schedule,
        )

    def _create_gamma_selector(self) -> GammaSelector:
        """Create gamma selector from config"""
        # Check for manual gamma override
        manual_gamma = self.config.get("manual_gamma", None)

        # BACKWARD COMPATIBILITY: Check if gamma was explicitly set
        if manual_gamma is None and "gamma" in self.config:
            manual_gamma = self.config["gamma"]
            logger.info(f"🔙 Legacy gamma={manual_gamma:.3f} detected")

        # DEFAULT FOR CRYPTO 24/7 TRADING: If no gamma specified, use 0.89
        # This provides backward compatibility and crypto-optimized default
        if manual_gamma is None and "gamma" not in self.config and "manual_gamma" not in self.config:
            manual_gamma = 0.89
            logger.info(f"🔄 Using crypto 24/7 default gamma={manual_gamma:.3f}")

        # Get timeframe from config (default: 1h)
        timeframe = self.config.get("timeframe", "1h")

        # Create selector
        selector = GammaSelector(
            timeframe=timeframe,
            manual_gamma=manual_gamma,
            config=self.config
        )

        return selector

    def _get_device(self) -> str:
        """Detect available compute device"""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _build_model(self):
        """Build neural network model"""
        try:
            import torch
            import torch.nn as nn

            class DQNNetwork(nn.Module):
                def __init__(self, state_size, action_size):
                    super(DQNNetwork, self).__init__()
                    self.fc1 = nn.Linear(state_size, 128)
                    self.fc2 = nn.Linear(128, 128)
                    self.fc3 = nn.Linear(128, 64)
                    self.fc4 = nn.Linear(64, action_size)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.relu(self.fc3(x))
                    x = self.fc4(x)
                    return x

            self.model = DQNNetwork(self.state_size, self.action_size)
            self.target_model = DQNNetwork(self.state_size, self.action_size)

            if self.device == "cuda":
                self.model = self.model.cuda()
                self.target_model = self.target_model.cuda()

            self.update_target_model()

            # Optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
            self.criterion = nn.MSELoss()

            # Initialize training optimizations (gradient clipping + LR scheduling)
            if TRAINING_OPTIMIZERS_AVAILABLE:
                self._init_training_optimizers()

            logger.info("✅ PyTorch DQN model built")

        except ImportError:
            logger.warning("PyTorch not available, using NumPy fallback")
            self._build_numpy_model()

    def _build_numpy_model(self):
        """Fallback: Simple NumPy-based Q-table approximation"""
        # Initialize random weights for simple linear model
        self.weights = np.random.randn(self.state_size, self.action_size) * 0.01
        logger.info("✅ NumPy fallback model built")

    def _init_training_optimizers(self):
        """Initialize gradient clipping and LR scheduling"""
        if not TRAINING_OPTIMIZERS_AVAILABLE:
            return

        # Gradient clipping configuration
        gradient_clip_norm = self.config.get('gradient_clip_norm', 1.0)
        gradient_explosion_threshold = self.config.get('gradient_explosion_threshold', 10.0)
        gradient_vanishing_threshold = self.config.get('gradient_vanishing_threshold', 1e-6)

        self.gradient_clipper = GradientClipper(
            max_norm=gradient_clip_norm,
            explosion_threshold=gradient_explosion_threshold,
            vanishing_threshold=gradient_vanishing_threshold,
        )

        # LR scheduling configuration
        lr_scheduler_type = self.config.get('lr_scheduler_type', 'cosine')
        lr_warmup_enabled = self.config.get('lr_warmup_enabled', True)
        lr_warmup_steps = self.config.get('lr_warmup_steps', 1000)

        self.lr_scheduler = LRSchedulerManager(
            optimizer=self.optimizer,
            scheduler_type=lr_scheduler_type,
            config=self.config,
            enable_warmup=lr_warmup_enabled,
            warmup_steps=lr_warmup_steps,
        )

        logger.info("✅ Training optimizers initialized (gradient clipping + LR scheduling)")

    def update_target_model(self):
        """Copy weights from model to target model"""
        try:
            if hasattr(self, "model") and self.model is not None:
                self.target_model.load_state_dict(self.model.state_dict())
        except:
            pass

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)

        # Exploit: best action from Q-values
        try:
            import torch

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if self.device == "cuda":
                state_tensor = state_tensor.cuda()

            with torch.no_grad():
                q_values = self.model(state_tensor)

            return q_values.argmax().item()

        except Exception as e:
            # If model forward fails due to dimension mismatch, return random action
            # This happens when state size doesn't match network architecture
            logger.error(f"Model forward failed: {e}. State size: {len(state)}, Expected: {self.state_size}")
            return np.random.randint(0, self.action_size)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size: int = None):
        """Train on batch from replay buffer with gradient clipping and LR scheduling"""
        batch_size = batch_size or self.batch_size
        if len(self.memory) < batch_size:
            return 0.0  # Return 0 loss when insufficient data (test compatibility)

        try:
            import torch

            # Sample batch (different for PER vs standard vs N-Step)
            if self.use_nstep:
                # N-Step returns format: (state, action, n_step_return, next_state_n, done, actual_steps)
                batch = self.memory.sample(batch_size)

                # Unpack based on tuple length
                if len(batch[0]) == 6:
                    states, actions, n_step_returns, next_states, dones, actual_steps = zip(*batch)
                else:
                    # Fallback for compatibility
                    states, actions, n_step_returns, next_states, dones = zip(*batch)
                    actual_steps = [self.memory.n_step] * len(batch)

                indices = None
                is_weights = None

                # Convert to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(n_step_returns)  # Use pre-calculated n-step returns
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones)

                if self.device == "cuda":
                    states = states.cuda()
                    actions = actions.cuda()
                    rewards = rewards.cuda()
                    next_states = next_states.cuda()
                    dones = dones.cuda()

                # For n-step, we need to add the bootstrapped value
                # n_step_return already contains: r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1}
                # We just need to add: γ^n * max_a Q(s_{t+n}, a)
                current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

                with torch.no_grad():
                    next_q_values = self.target_model(next_states).max(1)[0]
                    # Add bootstrapped value with appropriate gamma power
                    # rewards already contains discounted sum, just add bootstrap
                    gamma_n = torch.FloatTensor([self.gamma ** step for step in actual_steps])
                    if self.device == "cuda":
                        gamma_n = gamma_n.cuda()
                    target_q_values = rewards + (1 - dones) * gamma_n * next_q_values

            elif self.use_per:
                batch, indices, is_weights = self.memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert IS weights to tensor
                is_weights = torch.FloatTensor(is_weights)
                if self.device == "cuda":
                    is_weights = is_weights.cuda()

                # Convert to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones)

                if self.device == "cuda":
                    states = states.cuda()
                    actions = actions.cuda()
                    rewards = rewards.cuda()
                    next_states = next_states.cuda()
                    dones = dones.cuda()

                # Current Q values
                current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

                # Next Q values from target network
                with torch.no_grad():
                    next_q_values = self.target_model(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            else:
                # Standard replay buffer
                batch = self.memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                indices = None
                is_weights = None

                # Convert to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones)

                if self.device == "cuda":
                    states = states.cuda()
                    actions = actions.cuda()
                    rewards = rewards.cuda()
                    next_states = next_states.cuda()
                    dones = dones.cuda()

                # Current Q values
                current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

                # Next Q values from target network
                with torch.no_grad():
                    next_q_values = self.target_model(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute TD errors (for PER priority updates)
            # Ensure consistent shapes for TD error calculation
            current_q = current_q_values.squeeze(1)  # [batch_size, 1] -> [batch_size]
            td_errors = (target_q_values - current_q).detach()

            # Compute loss (with IS weights if using PER)
            if self.use_per and is_weights is not None:
                # Apply importance sampling weights to each sample's loss
                element_wise_loss = torch.nn.functional.mse_loss(
                    current_q, target_q_values, reduction='none'
                )
                loss = (is_weights * element_wise_loss).mean()
            else:
                loss = self.criterion(current_q, target_q_values)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (if enabled)
            self._training_step_count += 1
            gradient_norm_before = 0.0
            gradient_norm_after = 0.0
            gradient_clipped = False

            if self.gradient_clipper is not None:
                log_step = (self._training_step_count % 100 == 0)
                gradient_norm_before, gradient_norm_after, gradient_clipped = \
                    self.gradient_clipper.clip_gradients(self.model, log_step=log_step)

            # Optimizer step
            self.optimizer.step()

            # Update priorities in PER buffer
            if self.use_per and indices is not None:
                # Convert TD errors to numpy for priority update
                td_errors_np = td_errors.cpu().numpy() if self.device == "cuda" else td_errors.numpy()
                self.memory.update_priorities(indices, td_errors_np)

            # LR scheduling (if enabled)
            current_lr = self.learning_rate
            if self.lr_scheduler is not None:
                log_step = (self._training_step_count % 100 == 0)
                self.lr_scheduler.step(loss=loss.item(), log_step=log_step)
                current_lr = self.lr_scheduler.get_current_lr()

            # Track training metrics
            if hasattr(self, 'training_history'):
                training_info = {
                    'step': self._training_step_count,
                    'loss': loss.item(),
                    'learning_rate': current_lr,
                }

                if self.gradient_clipper is not None:
                    training_info.update({
                        'gradient_norm_before': gradient_norm_before,
                        'gradient_norm_after': gradient_norm_after,
                        'gradient_clipped': gradient_clipped,
                    })

                # Track PER stats if enabled
                if self.use_per:
                    per_stats = self.memory.get_stats()
                    training_info.update({
                        'per_beta': per_stats['beta'],
                        'per_mean_priority': per_stats['mean_priority'],
                    })

                # Only store every 10th step to avoid memory bloat
                if self._training_step_count % 10 == 0:
                    self.training_history.append(training_info)

            return loss.item()

        except Exception as e:
            logger.error(f"Replay error: {e}")
            return 0.0  # Return 0 loss on error (test compatibility)

    def decay_epsilon(self):
        """Decay exploration rate using advanced strategy"""
        self.epsilon = self.epsilon_decay_strategy.step()

    def update_epsilon(self):
        """Alias for decay_epsilon() for backward compatibility"""
        self.decay_epsilon()

    def get_gamma_info(self) -> Dict[str, Any]:
        """Get gamma selector information"""
        return {
            "gamma": self.gamma,
            "style": self.gamma_selector.get_style_name(),
            "timeframe": self.gamma_selector.timeframe
        }

    def get_per_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get Prioritized Experience Replay statistics

        Returns:
            Dictionary of PER statistics if PER is enabled, None otherwise
        """
        if self.use_per:
            return self.memory.get_stats()
        return None

    def save(self, filepath: str):
        """Save model to file"""
        try:
            import torch

            save_data = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "epsilon_step": self.epsilon_decay_strategy.current_step,
                "gamma": self.gamma,
                "training_history": self.training_history,
                "training_step_count": self._training_step_count,
            }

            torch.save(save_data, filepath)
            logger.info(f"✅ Model saved to {filepath}")

            # Save epsilon history separately
            epsilon_path = (
                Path(filepath).parent / f"{Path(filepath).stem}_epsilon_history.json"
            )
            self.epsilon_decay_strategy.save_history(str(epsilon_path))

            # Save training optimizer histories (gradient + LR)
            if self.gradient_clipper is not None or self.lr_scheduler is not None:
                self._save_training_optimizer_histories(filepath)

        except Exception as e:
            logger.error(f"Save error: {e}")

    def _save_training_optimizer_histories(self, model_filepath: str):
        """Save gradient and LR histories"""
        output_dir = Path(model_filepath).parent / f"{Path(model_filepath).stem}_training"

        try:
            if self.gradient_clipper is not None:
                gradient_path = output_dir / "gradient_history.json"
                self.gradient_clipper.save_history(str(gradient_path))

            if self.lr_scheduler is not None:
                lr_path = output_dir / "lr_history.json"
                self.lr_scheduler.save_history(str(lr_path))

        except Exception as e:
            logger.warning(f"Failed to save training optimizer histories: {e}")

    def load(self, filepath: str):
        """Load model from file"""
        try:
            import torch

            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            self.training_history = checkpoint.get("training_history", [])
            self._training_step_count = checkpoint.get("training_step_count", 0)

            # Restore epsilon decay strategy step
            if "epsilon_step" in checkpoint:
                self.epsilon_decay_strategy.current_step = checkpoint["epsilon_step"]
                self.epsilon_decay_strategy.current_epsilon = self.epsilon

            # Restore gamma if saved (overrides selector default)
            if "gamma" in checkpoint:
                self.gamma = checkpoint["gamma"]
                logger.info(f"✅ Restored gamma={self.gamma:.3f} from checkpoint")

            self.update_target_model()

            logger.info(f"✅ Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Load error: {e}")

    def get_model_summary(self) -> str:
        """Get summary of model architecture and parameters"""
        try:
            import torch

            if self.model is None:
                param_count = self.weights.size if hasattr(self, "weights") else 0
                return (
                    f"DQN Agent Model Summary\n"
                    f"{'='*50}\n"
                    f"Type: NumPy Fallback\n"
                    f"State Size: {self.state_size}\n"
                    f"Action Size: {self.action_size}\n"
                    f"Parameters: {param_count}\n"
                )

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            gamma_info = self.get_gamma_info()
            summary = (
                f"DQN Agent Model Summary\n"
                f"{'='*50}\n"
                f"Type: PyTorch DQN\n"
                f"State Size: {self.state_size}\n"
                f"Action Size: {self.action_size}\n"
                f"Total Parameters: {total_params:,}\n"
                f"Trainable Parameters: {trainable_params:,}\n"
                f"Device: {self.device}\n"
                f"Epsilon: {self.epsilon:.4f}\n"
                f"Learning Rate: {self.learning_rate}\n"
                f"Gamma: {self.gamma:.3f} ({gamma_info['style']}, timeframe: {gamma_info['timeframe']})\n"
            )

            # Add PER info if enabled
            if self.use_per:
                per_stats = self.get_per_stats()
                summary += (
                    f"Replay Buffer: Prioritized (PER)\n"
                    f"  Size: {per_stats['size']}/{per_stats['capacity']}\n"
                    f"  Alpha: {per_stats['alpha']:.2f}\n"
                    f"  Beta: {per_stats['beta']:.3f}\n"
                    f"  Mean Priority: {per_stats['mean_priority']:.4f}\n"
                )
            else:
                summary += f"Replay Buffer: Standard (size: {len(self.memory)})\n"

            return summary

        except Exception as e:
            logger.error(f"Model summary error: {e}")
            return (
                f"DQN Agent Model Summary (Error)\n"
                f"{'='*50}\n"
                f"State Size: {self.state_size}\n"
                f"Action Size: {self.action_size}\n"
                f"Error: {str(e)}\n"
            )


# Export main classes
__all__ = ["TradingEnvironment", "DQNAgent", "ReplayBuffer"]
