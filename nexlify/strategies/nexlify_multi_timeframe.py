#!/usr/bin/env python3
"""
Nexlify Multi-Timeframe Analysis
Analyze market conditions across multiple timeframes for better signals
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class TimeframeSignal(Enum):
    """Signal types from timeframe analysis"""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MultiTimeframeAnalyzer:
    """
    Analyzes market across multiple timeframes for confluence trading
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Supported timeframes (in order of granularity)
        self.timeframes = self.config.get(
            "timeframes",
            [
                "5m",  # Short-term: Scalping
                "15m",  # Short-term: Day trading
                "1h",  # Medium-term: Swing entry
                "4h",  # Medium-term: Trend confirmation
                "1d",  # Long-term: Major trend
            ],
        )

        # Timeframe weights for scoring
        self.timeframe_weights = self.config.get(
            "timeframe_weights",
            {"5m": 0.10, "15m": 0.15, "1h": 0.25, "4h": 0.25, "1d": 0.25},
        )

        logger.info(
            f"📊 Multi-Timeframe Analyzer initialized with {len(self.timeframes)} timeframes"
        )

    @handle_errors("Multi-Timeframe Analysis", reraise=False)
    async def analyze_symbol(
        self, exchange: ccxt.Exchange, symbol: str, lookback_periods: int = 100
    ) -> Dict:
        """
        Analyze a symbol across all timeframes

        Args:
            exchange: CCXT exchange instance
            symbol: Trading pair (e.g., 'BTC/USDT')
            lookback_periods: Number of candles to fetch per timeframe

        Returns:
            Dict with multi-timeframe analysis results
        """
        logger.info(
            f"🔍 Analyzing {symbol} across {len(self.timeframes)} timeframes..."
        )

        # Fetch data for all timeframes
        timeframe_data = {}
        for tf in self.timeframes:
            try:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol, timeframe=tf, limit=lookback_periods
                )

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                timeframe_data[tf] = df

            except Exception as e:
                logger.error(f"Failed to fetch {tf} data for {symbol}: {e}")

        if not timeframe_data:
            return {"error": "Failed to fetch any timeframe data"}

        # Analyze each timeframe
        timeframe_signals = {}
        for tf, df in timeframe_data.items():
            signal_data = self._analyze_timeframe(df, tf)
            timeframe_signals[tf] = signal_data

        # Calculate confluence score
        confluence = self._calculate_confluence(timeframe_signals)

        # Determine overall signal
        overall_signal = self._determine_overall_signal(confluence)

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "timeframe_signals": timeframe_signals,
            "confluence_score": confluence,
            "overall_signal": overall_signal.value,
            "signal_strength": abs(confluence) / 100,
            "recommendation": self._generate_recommendation(overall_signal, confluence),
        }

    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze a single timeframe for trend and momentum"""
        if len(df) < 20:
            return {"signal": TimeframeSignal.NEUTRAL, "score": 0, "indicators": {}}

        # Calculate indicators
        indicators = self._calculate_indicators(df)

        # Trend analysis
        trend_score = self._analyze_trend(df, indicators)

        # Momentum analysis
        momentum_score = self._analyze_momentum(indicators)

        # Support/Resistance
        sr_score = self._analyze_support_resistance(df)

        # Volume analysis
        volume_score = self._analyze_volume(df)

        # Combine scores
        combined_score = (
            trend_score * 0.40
            + momentum_score * 0.30
            + sr_score * 0.20
            + volume_score * 0.10
        )

        # Determine signal
        if combined_score >= 60:
            signal = TimeframeSignal.STRONG_BUY
        elif combined_score >= 20:
            signal = TimeframeSignal.BUY
        elif combined_score <= -60:
            signal = TimeframeSignal.STRONG_SELL
        elif combined_score <= -20:
            signal = TimeframeSignal.SELL
        else:
            signal = TimeframeSignal.NEUTRAL

        return {
            "signal": signal,
            "score": combined_score,
            "indicators": indicators,
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "sr_score": sr_score,
            "volume_score": volume_score,
        }

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        close = df["close"].values

        # Moving averages
        sma_20 = (
            pd.Series(close).rolling(20).mean().iloc[-1]
            if len(close) >= 20
            else close[-1]
        )
        sma_50 = (
            pd.Series(close).rolling(50).mean().iloc[-1]
            if len(close) >= 50
            else close[-1]
        )
        ema_12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
        ema_26 = pd.Series(close).ewm(span=26).mean().iloc[-1]

        # RSI
        rsi = self._calculate_rsi(close)

        # MACD
        macd = ema_12 - ema_26
        signal_line = pd.Series(close).ewm(span=9).mean().iloc[-1]

        # Bollinger Bands
        sma_20_series = pd.Series(close).rolling(20).mean()
        std_20 = pd.Series(close).rolling(20).std()
        bb_upper = (
            (sma_20_series + 2 * std_20).iloc[-1] if len(close) >= 20 else close[-1]
        )
        bb_lower = (
            (sma_20_series - 2 * std_20).iloc[-1] if len(close) >= 20 else close[-1]
        )

        # ATR (Average True Range)
        atr = self._calculate_atr(df)

        return {
            "price": close[-1],
            "sma_20": sma_20,
            "sma_50": sma_50,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "rsi": rsi,
            "macd": macd,
            "signal_line": signal_line,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "atr": atr,
        }

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr_list = []
        for i in range(1, len(df)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            tr_list.append(tr)

        atr = np.mean(tr_list[-period:]) if len(tr_list) >= period else 0
        return atr

    def _analyze_trend(self, df: pd.DataFrame, indicators: Dict) -> float:
        """Analyze trend direction and strength (-100 to +100)"""
        price = indicators["price"]
        sma_20 = indicators["sma_20"]
        sma_50 = indicators["sma_50"]

        score = 0

        # Price vs MA
        if price > sma_20:
            score += 30
        elif price < sma_20:
            score -= 30

        if price > sma_50:
            score += 30
        elif price < sma_50:
            score -= 30

        # MA crossover
        if sma_20 > sma_50:
            score += 40  # Golden cross
        else:
            score -= 40  # Death cross

        return np.clip(score, -100, 100)

    def _analyze_momentum(self, indicators: Dict) -> float:
        """Analyze momentum (-100 to +100)"""
        rsi = indicators["rsi"]
        macd = indicators["macd"]

        score = 0

        # RSI analysis
        if rsi > 70:
            score -= 40  # Overbought
        elif rsi > 60:
            score += 20  # Strong momentum
        elif rsi < 30:
            score += 40  # Oversold (potential reversal)
        elif rsi < 40:
            score -= 20  # Weak momentum

        # MACD analysis
        if macd > 0:
            score += 30
        else:
            score -= 30

        # MACD vs Signal
        if macd > indicators["signal_line"]:
            score += 30  # Bullish crossover
        else:
            score -= 30  # Bearish crossover

        return np.clip(score, -100, 100)

    def _analyze_support_resistance(self, df: pd.DataFrame) -> float:
        """Analyze price position relative to S/R levels (-100 to +100)"""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        current_price = close[-1]

        # Calculate support (recent lows)
        support = np.min(low[-20:]) if len(low) >= 20 else low[-1]

        # Calculate resistance (recent highs)
        resistance = np.max(high[-20:]) if len(high) >= 20 else high[-1]

        # Position within range
        if resistance > support:
            position = (current_price - support) / (resistance - support)

            # Near support = bullish
            if position < 0.2:
                return 50
            # Near resistance = bearish
            elif position > 0.8:
                return -50
            # Middle = neutral
            else:
                return 0
        else:
            return 0

    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """Analyze volume trends (-100 to +100)"""
        volume = df["volume"].values
        close = df["close"].values

        if len(volume) < 20:
            return 0

        # Average volume
        avg_volume = np.mean(volume[-20:-1])
        current_volume = volume[-1]

        # Price change
        price_change = (close[-1] - close[-2]) / close[-2]

        score = 0

        # High volume with price increase
        if current_volume > avg_volume * 1.5 and price_change > 0:
            score += 50

        # High volume with price decrease
        elif current_volume > avg_volume * 1.5 and price_change < 0:
            score -= 50

        # Low volume
        elif current_volume < avg_volume * 0.5:
            score -= 20  # Weak trend

        return np.clip(score, -100, 100)

    def _calculate_confluence(self, timeframe_signals: Dict) -> float:
        """Calculate confluence score across all timeframes (-100 to +100)"""
        weighted_score = 0

        for tf, data in timeframe_signals.items():
            weight = self.timeframe_weights.get(tf, 0.2)
            score = data["score"]
            weighted_score += score * weight

        return weighted_score

    def _determine_overall_signal(self, confluence_score: float) -> TimeframeSignal:
        """Determine overall signal from confluence score"""
        if confluence_score >= 60:
            return TimeframeSignal.STRONG_BUY
        elif confluence_score >= 20:
            return TimeframeSignal.BUY
        elif confluence_score <= -60:
            return TimeframeSignal.STRONG_SELL
        elif confluence_score <= -20:
            return TimeframeSignal.SELL
        else:
            return TimeframeSignal.NEUTRAL

    def _generate_recommendation(
        self, signal: TimeframeSignal, confluence: float
    ) -> str:
        """Generate human-readable recommendation"""
        recommendations = {
            TimeframeSignal.STRONG_BUY: f"Strong buy signal with {confluence:.1f}% confluence. Multiple timeframes aligned bullish.",
            TimeframeSignal.BUY: f"Buy signal with {confluence:.1f}% confluence. Majority timeframes bullish.",
            TimeframeSignal.NEUTRAL: f"Neutral signal ({confluence:.1f}%). Mixed signals across timeframes. Wait for clarity.",
            TimeframeSignal.SELL: f"Sell signal with {confluence:.1f}% confluence. Majority timeframes bearish.",
            TimeframeSignal.STRONG_SELL: f"Strong sell signal with {confluence:.1f}% confluence. Multiple timeframes aligned bearish.",
        }

        return recommendations.get(signal, "No recommendation available")

    def get_timeframe_summary(self, analysis_result: Dict) -> str:
        """Get formatted summary of timeframe analysis"""
        if "error" in analysis_result:
            return f"Error: {analysis_result['error']}"

        lines = [
            f"Symbol: {analysis_result['symbol']}",
            f"Overall Signal: {analysis_result['overall_signal'].upper()}",
            f"Confluence Score: {analysis_result['confluence_score']:.1f}",
            f"Signal Strength: {analysis_result['signal_strength']:.0%}",
            "",
            "Timeframe Breakdown:",
            "-" * 50,
        ]

        for tf, data in analysis_result["timeframe_signals"].items():
            signal = data["signal"].value
            score = data["score"]
            lines.append(f"  {tf:>4} | {signal:<12} | Score: {score:>6.1f}")

        lines.append("-" * 50)
        lines.append(f"\n{analysis_result['recommendation']}")

        return "\n".join(lines)


if __name__ == "__main__":

    async def main():
        print("=" * 70)
        print("NEXLIFY MULTI-TIMEFRAME ANALYSIS DEMO")
        print("=" * 70)

        # Initialize
        analyzer = MultiTimeframeAnalyzer()
        exchange = ccxt.binance()

        # Analyze
        result = await analyzer.analyze_symbol(exchange, "BTC/USDT")

        # Print results
        print("\n" + analyzer.get_timeframe_summary(result))

        await exchange.close()

    asyncio.run(main())
