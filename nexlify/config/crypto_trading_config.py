#!/usr/bin/env python3
"""
Centralized Crypto Trading Configuration
==========================================

Single source of truth for all crypto trading settings.
All hardcoded values should be replaced with imports from this file.

Usage:
    from nexlify.config.crypto_trading_config import CRYPTO_24_7_CONFIG, FEATURE_PERIODS

Author: Nexlify Team
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from nexlify.config.fee_providers import FeeProvider, get_fee_provider


# ============================================================================
# 24/7 CRYPTO TRADING CONFIGURATION
# ============================================================================

@dataclass
class CryptoTradingConfig:
    """
    Optimized configuration for 24/7 continuous crypto trading

    Based on research and testing for non-stationary crypto markets
    with volatility clustering, regime changes, and 24/7 operation.
    """

    # ========================================================================
    # DQN AGENT HYPERPARAMETERS
    # ========================================================================

    # Discount factor (default for 24/7 crypto trading)
    # Can be overridden by setting manual_gamma in config
    # Or auto-selected based on timeframe using GammaSelector
    gamma: float = 0.89

    # Trading timeframe (used by GammaSelector to choose optimal gamma)
    timeframe: str = "1h"

    # Learning rate (aggressive for rapid adaptation)
    learning_rate: float = 0.0015

    # Learning rate decay (gradual stabilization)
    learning_rate_decay: float = 0.9998

    # Batch size for training
    batch_size: int = 64

    # Target network update frequency (steps)
    target_update_freq: int = 1200

    # Replay buffer size (smaller for regime adaptation)
    replay_buffer_size: int = 60000

    # ========================================================================
    # EPSILON DECAY SETTINGS
    # ========================================================================

    # Epsilon decay strategy type
    epsilon_decay_type: str = "scheduled"

    # Starting epsilon (full exploration)
    epsilon_start: float = 1.0

    # Ending epsilon (high ongoing exploration for crypto)
    epsilon_end: float = 0.22

    # Scheduled epsilon decay milestones
    # Episodes: epsilon value (24/7 context: 200 eps ≈ 8 days)
    epsilon_schedule: Dict[int, float] = None  # Set in __post_init__

    def __post_init__(self):
        """Initialize default epsilon schedule"""
        if self.epsilon_schedule is None:
            self.epsilon_schedule = {
                0: 1.0,      # Full exploration
                200: 0.65,   # Learn basics quickly (~8 days)
                800: 0.35,   # Start exploiting patterns (~1 month)
                2000: 0.22,  # High ongoing exploration (~2.5 months)
            }

    # ========================================================================
    # TRADING ENVIRONMENT SETTINGS
    # ========================================================================

    # Initial balance
    initial_balance: float = 10000.0

    # Network/Exchange for trading
    # Options: 'binance', 'coinbase', 'ethereum', 'polygon', 'bsc', 'static'
    # IMPORTANT: Set this based on your actual trading venue
    # Default 'static' uses fixed 0.1% fees (conservative estimate)
    trading_network: str = "static"

    # Fee provider (None = auto-create from trading_network)
    # Set explicitly to override trading_network selection
    fee_provider: Optional[FeeProvider] = None

    # DEPRECATED: Static fee rate (only used if fee_provider disabled)
    # WARNING: Using static fees can cause 10-100x errors on ETH L1!
    fee_rate: float = 0.001  # CEX default (Binance: 0.1%)

    # Slippage rate (0.05%)
    slippage: float = 0.0005

    # Use dynamic fee provider (RECOMMENDED)
    # If False, falls back to static fee_rate (ONLY for backtesting/simulation)
    use_dynamic_fees: bool = True

    # Strict fee mode (CRITICAL for live trading)
    # If True, trading is BLOCKED when real fees cannot be retrieved
    # MUST be True for live trading, False only for backtesting
    strict_fee_mode: bool = False  # Default False for backward compatibility

    # Trading mode: 'backtest', 'paper', 'live'
    # In 'live' mode, strict_fee_mode is automatically enforced
    trading_mode: str = "backtest"

    # Maximum steps per episode
    max_steps_per_episode: int = 1000

    # State space size (crypto-optimized features)
    state_size: int = 12

    # Action space size (buy, sell, hold)
    action_size: int = 3

    # ========================================================================
    # TECHNICAL INDICATOR PERIODS
    # ========================================================================

    # RSI period
    rsi_period: int = 14

    # MACD fast period
    macd_fast_period: int = 12

    # MACD slow period
    macd_slow_period: int = 26

    # Volatility calculation period
    volatility_period: int = 10

    # Momentum calculation period
    momentum_period: int = 20

    # Volatility clustering short period
    vol_clustering_short_period: int = 10

    # Volatility clustering long period
    vol_clustering_long_period: int = 30

    # Drawdown calculation enabled
    enable_drawdown_tracking: bool = True

    # Sharpe ratio rolling window
    sharpe_window: int = 50

    # ========================================================================
    # TIME PERIOD SETTINGS (24/7 Crypto)
    # ========================================================================

    # Periods per year for different timeframes (crypto trades 24/7)
    periods_per_year: Dict[str, int] = None  # Set in __post_init__

    # Default timeframe assumption
    default_timeframe: str = "1h"

    def __post_init__(self):
        """Initialize time period settings and fee provider"""
        if self.epsilon_schedule is None:
            self.epsilon_schedule = {
                0: 1.0,
                200: 0.65,
                800: 0.35,
                2000: 0.22,
            }

        if self.periods_per_year is None:
            self.periods_per_year = {
                "1m": 525600,   # 365 * 24 * 60
                "5m": 105120,   # 365 * 24 * 12
                "15m": 35040,   # 365 * 24 * 4
                "1h": 8760,     # 365 * 24
                "4h": 2190,     # 365 * 6
                "1d": 365,      # 365
            }

        # Initialize fee provider if not provided and dynamic fees enabled
        if self.use_dynamic_fees and self.fee_provider is None:
            self.fee_provider = get_fee_provider(self.trading_network)

        # ENFORCE strict fee mode for live trading
        if self.trading_mode == "live":
            self.strict_fee_mode = True
            if not self.use_dynamic_fees or self.fee_provider is None:
                raise ValueError(
                    "CRITICAL: Live trading REQUIRES dynamic fee provider. "
                    "Set trading_network to your actual exchange/network and ensure "
                    "use_dynamic_fees=True"
                )

    # ========================================================================
    # RISK MANAGEMENT
    # ========================================================================

    # Use risk-adjusted rewards (Sharpe-based)
    use_risk_adjusted_rewards: bool = True

    # Risk adjustment scaling factor
    risk_adjustment_scale: float = 0.5

    # Enable improved reward function
    use_improved_rewards: bool = True

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def get_periods_per_year(self, timeframe: str = None) -> int:
        """Get periods per year for given timeframe"""
        tf = timeframe or self.default_timeframe
        return self.periods_per_year.get(tf, self.periods_per_year["1h"])

    def get_fee_estimate(self, trade_size_usd: float = 1000.0):
        """
        Get current fee estimate for a trade

        Args:
            trade_size_usd: Size of trade in USD

        Returns:
            FeeEstimate object with current fees

        Raises:
            RuntimeError: If strict_fee_mode=True and real fees cannot be retrieved

        Examples:
            >>> config = CryptoTradingConfig(trading_network="ethereum")
            >>> estimate = config.get_fee_estimate(1000.0)
            >>> print(f"Round trip cost: ${estimate.calculate_round_trip_cost(1000.0):.2f}")
        """
        if self.use_dynamic_fees and self.fee_provider is not None:
            return self.fee_provider.get_fee_estimate(trade_size_usd=trade_size_usd)
        else:
            # Check if we're in strict mode
            if self.strict_fee_mode or self.trading_mode == "live":
                raise RuntimeError(
                    "TRADING BLOCKED: Cannot retrieve real-time fees. "
                    f"Trading mode: {self.trading_mode}, Strict fee mode: {self.strict_fee_mode}. "
                    "Fee provider must be configured for live/strict mode trading. "
                    "DO NOT trade without accurate fee information - this could cause losses!"
                )

            # Fallback to static fees (ONLY for backtesting/simulation)
            from nexlify.config.fee_providers import FeeEstimate
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Using static fallback fees ({self.fee_rate * 100:.2f}%) - "
                "This is ONLY safe for backtesting/simulation!"
            )
            return FeeEstimate(
                entry_fee_rate=self.fee_rate,
                exit_fee_rate=self.fee_rate,
                network="static_fallback",
                fee_type="percentage"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            # DQN hyperparameters
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "replay_buffer_size": self.replay_buffer_size,

            # Timeframe for gamma selection
            "timeframe": self.timeframe,

            # Epsilon decay
            "epsilon_decay_type": self.epsilon_decay_type,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_schedule": self.epsilon_schedule,

            # Environment
            "initial_balance": self.initial_balance,
            "fee_rate": self.fee_rate,
            "slippage": self.slippage,
            "max_steps": self.max_steps_per_episode,
            "state_size": self.state_size,
            "action_size": self.action_size,

            # Risk management
            "use_improved_rewards": self.use_improved_rewards,
        }


# ============================================================================
# GLOBAL CONFIG INSTANCES
# ============================================================================

# Default crypto 24/7 configuration
CRYPTO_24_7_CONFIG = CryptoTradingConfig()

# Legacy compatibility - expose as dict
DEFAULT_CONFIG = CRYPTO_24_7_CONFIG.to_dict()


# ============================================================================
# FEATURE CALCULATION PERIODS
# ============================================================================

FEATURE_PERIODS = {
    "rsi": CRYPTO_24_7_CONFIG.rsi_period,
    "macd_fast": CRYPTO_24_7_CONFIG.macd_fast_period,
    "macd_slow": CRYPTO_24_7_CONFIG.macd_slow_period,
    "volatility": CRYPTO_24_7_CONFIG.volatility_period,
    "momentum": CRYPTO_24_7_CONFIG.momentum_period,
    "vol_clustering_short": CRYPTO_24_7_CONFIG.vol_clustering_short_period,
    "vol_clustering_long": CRYPTO_24_7_CONFIG.vol_clustering_long_period,
    "sharpe_window": CRYPTO_24_7_CONFIG.sharpe_window,
}


# ============================================================================
# TIMEFRAME PERIODS
# ============================================================================

PERIODS_PER_YEAR = CRYPTO_24_7_CONFIG.periods_per_year.copy()


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config(config: CryptoTradingConfig) -> bool:
    """
    Validate configuration values

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Validate gamma
    if not 0 < config.gamma <= 1:
        raise ValueError(f"gamma must be in (0, 1], got {config.gamma}")

    # Validate learning rate
    if not 0 < config.learning_rate < 1:
        raise ValueError(f"learning_rate must be in (0, 1), got {config.learning_rate}")

    # Validate epsilon values
    if not 0 <= config.epsilon_end <= config.epsilon_start <= 1:
        raise ValueError(
            f"epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1, "
            f"got epsilon_start={config.epsilon_start}, epsilon_end={config.epsilon_end}"
        )

    # Validate buffer size
    if config.replay_buffer_size <= 0:
        raise ValueError(f"replay_buffer_size must be positive, got {config.replay_buffer_size}")

    # Validate state/action sizes
    if config.state_size != 12:
        raise ValueError(f"state_size must be 12 for crypto-optimized features, got {config.state_size}")

    if config.action_size != 3:
        raise ValueError(f"action_size must be 3 (buy/sell/hold), got {config.action_size}")

    return True


# ============================================================================
# ALTERNATIVE CONFIGURATIONS
# ============================================================================

@dataclass
class ConservativeCryptoConfig(CryptoTradingConfig):
    """More conservative crypto trading configuration"""

    gamma: float = 0.93  # Longer planning horizon
    learning_rate: float = 0.001  # Slower learning
    epsilon_end: float = 0.15  # Less exploration
    replay_buffer_size: int = 100000  # More history

    def __post_init__(self):
        super().__post_init__()
        # More gradual epsilon decay
        self.epsilon_schedule = {
            0: 1.0,
            500: 0.70,
            1500: 0.40,
            3000: 0.15,
        }


@dataclass
class AggressiveCryptoConfig(CryptoTradingConfig):
    """More aggressive crypto trading configuration"""

    gamma: float = 0.85  # Very short horizon
    learning_rate: float = 0.002  # Fast learning
    epsilon_end: float = 0.28  # High exploration
    replay_buffer_size: int = 40000  # Only recent data
    target_update_freq: int = 800  # Very frequent updates

    def __post_init__(self):
        super().__post_init__()
        # Very aggressive epsilon decay
        self.epsilon_schedule = {
            0: 1.0,
            100: 0.60,
            500: 0.35,
            1000: 0.28,
        }


# Export alternative configs
CONSERVATIVE_CONFIG = ConservativeCryptoConfig()
AGGRESSIVE_CONFIG = AggressiveCryptoConfig()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "CryptoTradingConfig",
    "CRYPTO_24_7_CONFIG",
    "DEFAULT_CONFIG",
    "FEATURE_PERIODS",
    "PERIODS_PER_YEAR",
    "validate_config",
    "ConservativeCryptoConfig",
    "AggressiveCryptoConfig",
    "CONSERVATIVE_CONFIG",
    "AGGRESSIVE_CONFIG",
]


# Validate default config on import
validate_config(CRYPTO_24_7_CONFIG)
