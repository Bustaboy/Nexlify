#!/usr/bin/env python3
"""
Gamma Selection for Trading Timeframes
=======================================

Selects the optimal static gamma (discount factor) based on trading timeframe.

The gamma parameter controls how much the RL agent values future rewards vs immediate rewards.
Different trading styles require different gamma values:
- Scalping (< 1h): gamma = 0.90 - focus on immediate profits
- Day trading (1-24h): gamma = 0.95 - balance short and medium term
- Swing trading (1-7d): gamma = 0.97 - value medium-term rewards
- Position trading (> 7d): gamma = 0.99 - long-term planning

This provides STATIC gamma selection only. Gamma does not change during training.

For multi-gamma parallel training, see MultiGammaTrainer.

Author: Nexlify Team
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

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
# GAMMA SELECTOR
# ============================================================================

class GammaSelector:
    """
    Static gamma selector for RL trading agents

    Selects optimal gamma based on trading timeframe.
    Gamma remains FIXED during training (no auto-adjustment).

    Usage:
        >>> selector = GammaSelector(timeframe="1h")
        >>> gamma = selector.get_gamma()  # Returns 0.95
        >>> style = selector.get_style()  # Returns "day_trading"
    """

    def __init__(
        self,
        timeframe: str = "1h",
        manual_gamma: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize gamma selector

        Args:
            timeframe: Trading timeframe (e.g., "1h", "4h", "1d")
            manual_gamma: Manual gamma override (disables timeframe selection)
            config: Additional configuration dict
        """
        self.timeframe = timeframe
        self.manual_gamma = manual_gamma
        self.config = config or {}

        # Select gamma and style
        if manual_gamma is not None:
            self.gamma = manual_gamma
            self.style = self._infer_style_from_gamma(manual_gamma)
            logger.info(f"📌 Using MANUAL gamma={manual_gamma:.3f} (style: {self.style.name})")
        else:
            self._initialize_from_timeframe()

        logger.info(
            f"📊 GammaSelector initialized:\n"
            f"   Timeframe: {timeframe}\n"
            f"   Trading Style: {self.style.name}\n"
            f"   Gamma: {self.gamma:.3f} (STATIC - no auto-adjustment)"
        )

    def _initialize_from_timeframe(self):
        """Initialize gamma based on configured timeframe"""
        # Get expected trade duration from timeframe
        expected_duration_hours = TIMEFRAME_TO_HOURS.get(self.timeframe, 1.0)

        # Find matching trading style
        self.style = self._get_style_for_duration(expected_duration_hours)
        self.gamma = self.style.gamma

        logger.info(
            f"🚀 Selected trading style: {self.style.name.upper()}\n"
            f"   Expected trade duration: {expected_duration_hours:.2f} hours\n"
            f"   Recommended gamma: {self.gamma:.3f}\n"
            f"   Rationale: {self.style.description}"
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
        """Get selected gamma value"""
        return self.gamma

    def get_style(self) -> TradingStyle:
        """Get selected trading style"""
        return self.style

    def get_style_name(self) -> str:
        """Get trading style name"""
        return self.style.name

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"GammaSelector(timeframe='{self.timeframe}', "
            f"style='{self.style.name}', "
            f"gamma={self.gamma:.3f})"
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
        Recommended gamma for 1h trading: 0.95
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
    "GammaSelector",
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

    # Demo selector
    print("\n" + "="*80)
    print("DEMO: GammaSelector in Action")
    print("="*80 + "\n")

    # Create selector for 1h trading
    selector = GammaSelector(timeframe="1h")
    print(f"\n{selector}")
    print(f"Gamma: {selector.get_gamma():.3f}")
    print(f"Style: {selector.get_style_name()}\n")

    # Test different timeframes
    print("\nGamma selection for different timeframes:")
    for tf in ["5m", "1h", "4h", "1d"]:
        sel = GammaSelector(timeframe=tf)
        print(f"  {tf}: gamma={sel.get_gamma():.2f} ({sel.get_style_name()})")
