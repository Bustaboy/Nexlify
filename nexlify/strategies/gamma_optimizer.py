#!/usr/bin/env python3
"""
Gamma Optimizer for Adaptive Trading Timeframes
================================================

Automatically selects and adjusts the gamma (discount factor) based on:
- Trading timeframe configuration
- Observed trade durations during training
- Market volatility patterns

The gamma parameter controls how much the RL agent values future rewards vs immediate rewards.
Different trading styles require different gamma values:
- Scalping (< 1h): Lower gamma (0.90) - focus on immediate profits
- Day trading (1-24h): Medium gamma (0.95) - balance short and medium term
- Swing trading (1-7d): Higher gamma (0.97) - value medium-term rewards
- Position trading (> 7d): Highest gamma (0.99) - long-term planning

Author: Nexlify Team
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# TRADING STYLE DEFINITIONS
# ============================================================================

@dataclass
class TradingStyle:
    """Definition of a trading style with its characteristics"""
    name: str
    gamma: float
    min_duration_hours: float
    max_duration_hours: float
    description: str


# Trading style configurations
TRADING_STYLES = {
    "scalping": TradingStyle(
        name="scalping",
        gamma=0.90,
        min_duration_hours=0.0,
        max_duration_hours=1.0,
        description="High-frequency trading, < 1 hour holding period"
    ),
    "day_trading": TradingStyle(
        name="day_trading",
        gamma=0.95,
        min_duration_hours=1.0,
        max_duration_hours=24.0,
        description="Day trading, 1-24 hour holding period"
    ),
    "swing_trading": TradingStyle(
        name="swing_trading",
        gamma=0.97,
        min_duration_hours=24.0,
        max_duration_hours=168.0,  # 7 days
        description="Swing trading, 1-7 day holding period"
    ),
    "position_trading": TradingStyle(
        name="position_trading",
        gamma=0.99,
        min_duration_hours=168.0,
        max_duration_hours=float('inf'),
        description="Position trading, > 7 day holding period"
    )
}


# Timeframe to hours mapping (crypto 24/7 trading)
TIMEFRAME_TO_HOURS = {
    "1m": 1/60,
    "5m": 5/60,
    "15m": 15/60,
    "30m": 0.5,
    "1h": 1.0,
    "2h": 2.0,
    "4h": 4.0,
    "6h": 6.0,
    "12h": 12.0,
    "1d": 24.0,
    "3d": 72.0,
    "1w": 168.0,
}


# ============================================================================
# GAMMA OPTIMIZER
# ============================================================================

class GammaOptimizer:
    """
    Adaptive gamma optimizer for RL trading agents

    Features:
    - Auto-selects gamma based on timeframe
    - Monitors actual trade durations
    - Dynamically adjusts gamma based on observed behavior
    - Accounts for market volatility
    - Logs all adjustments with rationale

    Usage:
        >>> optimizer = GammaOptimizer(timeframe="1h", auto_adjust=True)
        >>> gamma = optimizer.get_gamma()
        >>> optimizer.record_trade(entry_time, exit_time, profit)
        >>> optimizer.update_gamma(episode=100)  # Check for adjustments
    """

    def __init__(
        self,
        timeframe: str = "1h",
        auto_adjust: bool = True,
        manual_gamma: Optional[float] = None,
        adjustment_interval: int = 100,
        history_window: int = 50,
        adjustment_threshold: float = 0.10,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize gamma optimizer

        Args:
            timeframe: Trading timeframe (e.g., "1h", "4h", "1d")
            auto_adjust: Enable automatic gamma adjustment
            manual_gamma: Manual gamma override (disables auto-selection)
            adjustment_interval: Episodes between gamma adjustments
            history_window: Number of recent trades to consider
            adjustment_threshold: Minimum change to trigger adjustment (10% default)
            config: Additional configuration dict
        """
        self.timeframe = timeframe
        self.auto_adjust = auto_adjust
        self.manual_gamma = manual_gamma
        self.adjustment_interval = adjustment_interval
        self.history_window = history_window
        self.adjustment_threshold = adjustment_threshold
        self.config = config or {}

        # Trading style and gamma
        self.current_style: Optional[TradingStyle] = None
        self.current_gamma: float = 0.95  # Default

        # Trade history tracking
        self.trade_durations: deque = deque(maxlen=history_window)
        self.trade_profits: deque = deque(maxlen=history_window)
        self.trade_timestamps: List[Tuple[datetime, datetime]] = []

        # Adjustment tracking
        self.last_adjustment_episode: int = 0
        self.adjustment_history: List[Dict] = []
        self.episodes_since_last_adjustment: int = 0

        # Volatility tracking
        self.volatility_history: deque = deque(maxlen=history_window)

        # Initialize gamma
        if manual_gamma is not None:
            self.current_gamma = manual_gamma
            self.current_style = self._infer_style_from_gamma(manual_gamma)
            logger.info(f"🎯 GammaOptimizer initialized with MANUAL gamma={manual_gamma:.3f}")
        else:
            self._initialize_from_timeframe()

        logger.info(
            f"📊 GammaOptimizer initialized:\n"
            f"   Timeframe: {timeframe}\n"
            f"   Trading Style: {self.current_style.name if self.current_style else 'Unknown'}\n"
            f"   Initial Gamma: {self.current_gamma:.3f}\n"
            f"   Auto-adjust: {auto_adjust}\n"
            f"   Adjustment Interval: {adjustment_interval} episodes"
        )

    def _initialize_from_timeframe(self):
        """Initialize gamma based on configured timeframe"""
        # Get expected trade duration from timeframe
        expected_duration_hours = TIMEFRAME_TO_HOURS.get(self.timeframe, 1.0)

        # Find matching trading style
        self.current_style = self._get_style_for_duration(expected_duration_hours)
        self.current_gamma = self.current_style.gamma

        logger.info(
            f"🚀 Auto-selected trading style: {self.current_style.name.upper()}\n"
            f"   Expected trade duration: {expected_duration_hours:.2f} hours\n"
            f"   Recommended gamma: {self.current_gamma:.3f}\n"
            f"   Rationale: {self.current_style.description}"
        )

    def _get_style_for_duration(self, duration_hours: float) -> TradingStyle:
        """Get trading style for a given duration"""
        for style in TRADING_STYLES.values():
            if style.min_duration_hours <= duration_hours < style.max_duration_hours:
                return style

        # Default to position trading for very long durations
        return TRADING_STYLES["position_trading"]

    def _infer_style_from_gamma(self, gamma: float) -> TradingStyle:
        """Infer trading style from gamma value"""
        # Find closest matching style
        closest_style = None
        min_diff = float('inf')

        for style in TRADING_STYLES.values():
            diff = abs(style.gamma - gamma)
            if diff < min_diff:
                min_diff = diff
                closest_style = style

        return closest_style

    def get_gamma(self) -> float:
        """Get current gamma value"""
        return self.current_gamma

    def record_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        profit: float,
        volatility: Optional[float] = None
    ):
        """
        Record a completed trade for gamma optimization

        Args:
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
            profit: Trade profit/loss
            volatility: Market volatility during trade (optional)
        """
        # Calculate duration in hours
        duration = (exit_time - entry_time).total_seconds() / 3600

        # Store trade data
        self.trade_durations.append(duration)
        self.trade_profits.append(profit)
        self.trade_timestamps.append((entry_time, exit_time))

        if volatility is not None:
            self.volatility_history.append(volatility)

        logger.debug(
            f"📝 Recorded trade: duration={duration:.2f}h, profit={profit:.2f}, "
            f"volatility={volatility:.4f if volatility else 'N/A'}"
        )

    def record_trade_from_steps(
        self,
        entry_step: int,
        exit_step: int,
        profit: float,
        steps_per_hour: int = 1,
        volatility: Optional[float] = None
    ):
        """
        Record a trade using step counts instead of timestamps

        Args:
            entry_step: Step when trade entered
            exit_step: Step when trade exited
            profit: Trade profit/loss
            steps_per_hour: How many steps represent 1 hour
            volatility: Market volatility during trade (optional)
        """
        # Convert steps to hours
        duration_hours = (exit_step - entry_step) / steps_per_hour

        # Create synthetic timestamps
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=duration_hours)

        self.record_trade(entry_time, exit_time, profit, volatility)

    def update_gamma(self, episode: int) -> Tuple[bool, Optional[str]]:
        """
        Update gamma based on observed trade behavior

        Args:
            episode: Current training episode

        Returns:
            (adjusted, rationale) - Whether gamma was adjusted and why
        """
        self.episodes_since_last_adjustment += 1

        # Only check periodically
        if self.episodes_since_last_adjustment < self.adjustment_interval:
            return False, None

        # Need enough data
        if len(self.trade_durations) < 10:
            logger.debug(
                f"⏳ Insufficient trade data for gamma adjustment "
                f"({len(self.trade_durations)}/10 trades)"
            )
            return False, None

        # Check if auto-adjust is enabled
        if not self.auto_adjust:
            return False, None

        # Calculate average trade duration
        avg_duration = np.mean(self.trade_durations)
        median_duration = np.median(self.trade_durations)
        std_duration = np.std(self.trade_durations)

        # Get suggested style based on observed behavior
        suggested_style = self._get_style_for_duration(avg_duration)

        # Check if we should adjust
        old_gamma = self.current_gamma
        suggested_gamma = suggested_style.gamma

        # Calculate percentage change
        gamma_change_pct = abs(suggested_gamma - old_gamma) / old_gamma

        # Only adjust if change is significant
        if gamma_change_pct < self.adjustment_threshold:
            logger.debug(
                f"🔍 Gamma adjustment check (episode {episode}):\n"
                f"   Avg duration: {avg_duration:.2f}h (median: {median_duration:.2f}h)\n"
                f"   Current gamma: {old_gamma:.3f}\n"
                f"   Suggested gamma: {suggested_gamma:.3f}\n"
                f"   Change: {gamma_change_pct*100:.1f}% (below {self.adjustment_threshold*100:.0f}% threshold)"
            )
            self.episodes_since_last_adjustment = 0
            return False, None

        # Perform adjustment
        self.current_gamma = suggested_gamma
        old_style = self.current_style
        self.current_style = suggested_style

        # Build rationale
        rationale = (
            f"Gamma adjusted from {old_gamma:.3f} → {suggested_gamma:.3f} "
            f"({gamma_change_pct*100:.1f}% change)\n"
            f"   Trading style: {old_style.name if old_style else 'Unknown'} → {suggested_style.name}\n"
            f"   Observed avg trade duration: {avg_duration:.2f}h (±{std_duration:.2f}h)\n"
            f"   Expected duration range: {suggested_style.min_duration_hours:.1f}h - "
            f"{suggested_style.max_duration_hours:.1f}h\n"
            f"   Trades analyzed: {len(self.trade_durations)}\n"
            f"   Episode: {episode}"
        )

        # Add volatility info if available
        if len(self.volatility_history) > 0:
            avg_volatility = np.mean(self.volatility_history)
            rationale += f"\n   Avg market volatility: {avg_volatility:.4f}"

        # Log adjustment
        logger.info(f"🔧 GAMMA ADJUSTMENT:\n{rationale}")

        # Record adjustment
        adjustment_record = {
            "episode": episode,
            "old_gamma": old_gamma,
            "new_gamma": suggested_gamma,
            "old_style": old_style.name if old_style else None,
            "new_style": suggested_style.name,
            "avg_duration_hours": avg_duration,
            "median_duration_hours": median_duration,
            "std_duration_hours": std_duration,
            "trades_analyzed": len(self.trade_durations),
            "timestamp": datetime.now().isoformat(),
            "rationale": rationale
        }

        self.adjustment_history.append(adjustment_record)
        self.last_adjustment_episode = episode
        self.episodes_since_last_adjustment = 0

        return True, rationale

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics and configuration"""
        stats = {
            "timeframe": self.timeframe,
            "current_gamma": self.current_gamma,
            "current_style": self.current_style.name if self.current_style else None,
            "auto_adjust": self.auto_adjust,
            "manual_override": self.manual_gamma is not None,
            "trades_recorded": len(self.trade_durations),
            "adjustments_made": len(self.adjustment_history),
            "last_adjustment_episode": self.last_adjustment_episode,
        }

        # Add duration statistics if available
        if len(self.trade_durations) > 0:
            stats.update({
                "avg_trade_duration_hours": float(np.mean(self.trade_durations)),
                "median_trade_duration_hours": float(np.median(self.trade_durations)),
                "std_trade_duration_hours": float(np.std(self.trade_durations)),
                "min_trade_duration_hours": float(np.min(self.trade_durations)),
                "max_trade_duration_hours": float(np.max(self.trade_durations)),
            })

        # Add profit statistics if available
        if len(self.trade_profits) > 0:
            stats.update({
                "avg_trade_profit": float(np.mean(self.trade_profits)),
                "total_profit": float(np.sum(self.trade_profits)),
                "win_rate": float(np.mean([p > 0 for p in self.trade_profits])),
            })

        # Add volatility statistics if available
        if len(self.volatility_history) > 0:
            stats.update({
                "avg_volatility": float(np.mean(self.volatility_history)),
                "std_volatility": float(np.std(self.volatility_history)),
            })

        return stats

    def save_history(self, filepath: str):
        """Save adjustment history to file"""
        history_data = {
            "config": {
                "timeframe": self.timeframe,
                "auto_adjust": self.auto_adjust,
                "manual_gamma": self.manual_gamma,
                "adjustment_interval": self.adjustment_interval,
                "history_window": self.history_window,
            },
            "current_state": self.get_statistics(),
            "adjustments": self.adjustment_history,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"💾 Saved gamma adjustment history to {filepath}")

    def load_history(self, filepath: str):
        """Load adjustment history from file"""
        with open(filepath, 'r') as f:
            history_data = json.load(f)

        # Restore adjustment history
        self.adjustment_history = history_data.get("adjustments", [])

        if self.adjustment_history:
            self.last_adjustment_episode = self.adjustment_history[-1]["episode"]

        logger.info(
            f"📂 Loaded gamma adjustment history from {filepath}\n"
            f"   Adjustments: {len(self.adjustment_history)}"
        )

    def reset(self):
        """Reset all tracked data (keeps configuration)"""
        self.trade_durations.clear()
        self.trade_profits.clear()
        self.trade_timestamps.clear()
        self.volatility_history.clear()
        self.adjustment_history.clear()
        self.last_adjustment_episode = 0
        self.episodes_since_last_adjustment = 0

        logger.info("🔄 GammaOptimizer reset (configuration preserved)")

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"GammaOptimizer(timeframe='{self.timeframe}', "
            f"style='{self.current_style.name if self.current_style else 'Unknown'}', "
            f"gamma={self.current_gamma:.3f}, "
            f"auto_adjust={self.auto_adjust}, "
            f"trades={len(self.trade_durations)})"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_recommended_gamma(timeframe: str) -> float:
    """
    Get recommended gamma for a timeframe

    Args:
        timeframe: Trading timeframe (e.g., "1h", "4h", "1d")

    Returns:
        Recommended gamma value

    Example:
        >>> gamma = get_recommended_gamma("1h")
        >>> print(f"Recommended gamma for 1h trading: {gamma}")
    """
    duration_hours = TIMEFRAME_TO_HOURS.get(timeframe, 1.0)

    for style in TRADING_STYLES.values():
        if style.min_duration_hours <= duration_hours < style.max_duration_hours:
            return style.gamma

    # Default to position trading for unknown timeframes
    return TRADING_STYLES["position_trading"].gamma


def print_gamma_recommendations():
    """Print gamma recommendations for different trading styles"""
    print("\n" + "="*80)
    print("GAMMA RECOMMENDATIONS FOR DIFFERENT TRADING STYLES")
    print("="*80)
    print(f"{'Trading Style':<20} {'Gamma':<10} {'Duration Range':<25} {'Description'}")
    print("-"*80)

    for style in TRADING_STYLES.values():
        max_dur = f"{style.max_duration_hours:.0f}h" if style.max_duration_hours != float('inf') else "∞"
        dur_range = f"{style.min_duration_hours:.1f}h - {max_dur}"
        print(
            f"{style.name.replace('_', ' ').title():<20} "
            f"{style.gamma:<10.2f} "
            f"{dur_range:<25} "
            f"{style.description}"
        )

    print("="*80)
    print("\nTimeframe to Gamma Mapping (Crypto 24/7):")
    print("-"*80)

    for tf in ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]:
        gamma = get_recommended_gamma(tf)
        duration = TIMEFRAME_TO_HOURS[tf]
        print(f"  {tf:<6} → gamma={gamma:.2f}  (≈{duration:.2f} hours)")

    print("="*80 + "\n")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "GammaOptimizer",
    "TradingStyle",
    "TRADING_STYLES",
    "TIMEFRAME_TO_HOURS",
    "get_recommended_gamma",
    "print_gamma_recommendations",
]


# Demo if run as script
if __name__ == "__main__":
    # Print recommendations
    print_gamma_recommendations()

    # Demo optimizer
    print("\n" + "="*80)
    print("DEMO: GammaOptimizer in Action")
    print("="*80 + "\n")

    # Create optimizer for 1h trading
    optimizer = GammaOptimizer(timeframe="1h", auto_adjust=True)
    print(f"\n{optimizer}\n")

    # Simulate some trades
    print("Simulating trades...")
    base_time = datetime.now()

    # Simulate shorter trades than expected (scalping behavior)
    for i in range(30):
        entry = base_time + timedelta(hours=i)
        # Random duration between 15-45 minutes
        duration_hours = np.random.uniform(0.25, 0.75)
        exit = entry + timedelta(hours=duration_hours)
        profit = np.random.uniform(-10, 20)
        volatility = np.random.uniform(0.01, 0.05)

        optimizer.record_trade(entry, exit, profit, volatility)

    # Check for adjustment
    print("\nChecking for gamma adjustment after 100 episodes...")
    adjusted, rationale = optimizer.update_gamma(episode=100)

    if adjusted:
        print(f"\n✅ Gamma was adjusted!\n{rationale}")
    else:
        print("\n❌ No adjustment needed yet")

    # Print statistics
    print("\n" + "="*80)
    print("OPTIMIZER STATISTICS")
    print("="*80)
    stats = optimizer.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*80 + "\n")
