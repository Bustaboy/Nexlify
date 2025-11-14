#!/usr/bin/env python3
"""
Nexlify Predictive Features Module
AI-powered price prediction and market analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class PredictiveEngine:
    """
    AI-powered predictive analytics engine
    Uses machine learning for price prediction and trend analysis
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.price_history: Dict[str, deque] = {}
        self.predictions_cache: Dict[str, Dict] = {}
        self.max_history_length = 1000

        # Model parameters (simplified for now)
        self.prediction_window = 60  # minutes
        self.confidence_threshold = 0.6

        logger.info("🔮 Predictive Engine initialized")

    @handle_errors("Price Prediction", reraise=False)
    def predict_price(
        self, symbol: str, current_price: float, historical_data: List[float]
    ) -> Dict:
        """
        Predict future price movement

        Args:
            symbol: Trading pair symbol
            current_price: Current price
            historical_data: Historical price data

        Returns:
            Dictionary with prediction, confidence, and direction
        """
        try:
            if len(historical_data) < 10:
                return {
                    "predicted_price": current_price,
                    "confidence": 0.0,
                    "direction": "neutral",
                    "change_percent": 0.0,
                }

            # Simple moving average prediction (can be enhanced with ML)
            short_window = 5
            long_window = 20

            df = pd.DataFrame({"price": historical_data})

            # Calculate moving averages
            df["sma_short"] = df["price"].rolling(window=short_window).mean()
            df["sma_long"] = df["price"].rolling(window=long_window).mean()

            # Calculate momentum
            df["momentum"] = df["price"].diff()
            df["momentum_ma"] = df["momentum"].rolling(window=5).mean()

            # Simple trend prediction
            last_short = df["sma_short"].iloc[-1]
            last_long = df["sma_long"].iloc[-1]
            momentum = df["momentum_ma"].iloc[-1]

            # Predict direction
            if last_short > last_long and momentum > 0:
                direction = "bullish"
                predicted_change = abs(momentum) * 1.5
            elif last_short < last_long and momentum < 0:
                direction = "bearish"
                predicted_change = abs(momentum) * 1.5
            else:
                direction = "neutral"
                predicted_change = abs(momentum) * 0.5

            # Calculate predicted price
            if direction == "bullish":
                predicted_price = current_price + predicted_change
            elif direction == "bearish":
                predicted_price = current_price - predicted_change
            else:
                predicted_price = current_price

            # Calculate confidence based on trend strength
            trend_strength = abs(last_short - last_long) / current_price
            confidence = min(trend_strength * 100, 0.95)

            change_percent = ((predicted_price - current_price) / current_price) * 100

            prediction = {
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "confidence": confidence,
                "direction": direction,
                "change_percent": change_percent,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache prediction
            self.predictions_cache[symbol] = prediction

            return prediction

        except Exception as e:
            logger.error(f"Price prediction error for {symbol}: {e}")
            return {
                "predicted_price": current_price,
                "confidence": 0.0,
                "direction": "neutral",
                "change_percent": 0.0,
            }

    def analyze_volatility(self, symbol: str, prices: List[float]) -> Dict:
        """
        Analyze price volatility

        Returns:
            Dictionary with volatility metrics
        """
        try:
            if len(prices) < 2:
                return {"volatility": 0.0, "risk_level": "unknown"}

            # Calculate returns
            returns = pd.Series(prices).pct_change().dropna()

            # Calculate standard deviation (volatility)
            volatility = returns.std()

            # Calculate other metrics
            mean_return = returns.mean()
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0

            # Classify risk level
            if volatility < 0.01:
                risk_level = "low"
            elif volatility < 0.03:
                risk_level = "medium"
            elif volatility < 0.05:
                risk_level = "high"
            else:
                risk_level = "extreme"

            return {
                "symbol": symbol,
                "volatility": volatility,
                "mean_return": mean_return,
                "sharpe_ratio": sharpe_ratio,
                "risk_level": risk_level,
                "sample_size": len(prices),
            }

        except Exception as e:
            logger.error(f"Volatility analysis error: {e}")
            return {"volatility": 0.0, "risk_level": "unknown"}

    def detect_patterns(self, symbol: str, prices: List[float]) -> List[Dict]:
        """
        Detect technical analysis patterns

        Returns:
            List of detected patterns
        """
        patterns = []

        try:
            if len(prices) < 20:
                return patterns

            df = pd.DataFrame({"price": prices})

            # Moving averages
            df["sma_20"] = df["price"].rolling(window=20).mean()
            df["sma_50"] = (
                df["price"].rolling(window=50).mean()
                if len(prices) >= 50
                else df["price"].rolling(window=20).mean()
            )

            # Detect Golden Cross
            if len(prices) >= 50:
                if (
                    df["sma_20"].iloc[-1] > df["sma_50"].iloc[-1]
                    and df["sma_20"].iloc[-2] <= df["sma_50"].iloc[-2]
                ):
                    patterns.append(
                        {
                            "pattern": "golden_cross",
                            "signal": "bullish",
                            "confidence": 0.75,
                            "description": "Short-term MA crossed above long-term MA",
                        }
                    )

            # Detect Death Cross
            if len(prices) >= 50:
                if (
                    df["sma_20"].iloc[-1] < df["sma_50"].iloc[-1]
                    and df["sma_20"].iloc[-2] >= df["sma_50"].iloc[-2]
                ):
                    patterns.append(
                        {
                            "pattern": "death_cross",
                            "signal": "bearish",
                            "confidence": 0.75,
                            "description": "Short-term MA crossed below long-term MA",
                        }
                    )

            # Detect support/resistance breaks
            recent_high = df["price"].iloc[-20:].max()
            recent_low = df["price"].iloc[-20:].min()
            current_price = df["price"].iloc[-1]

            if current_price > recent_high * 0.99:
                patterns.append(
                    {
                        "pattern": "resistance_break",
                        "signal": "bullish",
                        "confidence": 0.65,
                        "description": f"Price breaking resistance at {recent_high:.2f}",
                    }
                )

            if current_price < recent_low * 1.01:
                patterns.append(
                    {
                        "pattern": "support_break",
                        "signal": "bearish",
                        "confidence": 0.65,
                        "description": f"Price breaking support at {recent_low:.2f}",
                    }
                )

        except Exception as e:
            logger.error(f"Pattern detection error: {e}")

        return patterns

    def calculate_indicators(
        self, prices: List[float], volumes: List[float] = None
    ) -> Dict:
        """
        Calculate technical indicators

        Returns:
            Dictionary of technical indicators
        """
        indicators = {}

        try:
            if len(prices) < 2:
                return indicators

            df = pd.DataFrame({"price": prices})
            if volumes:
                df["volume"] = volumes

            # Simple Moving Averages
            if len(prices) >= 7:
                indicators["sma_7"] = df["price"].rolling(window=7).mean().iloc[-1]
            if len(prices) >= 25:
                indicators["sma_25"] = df["price"].rolling(window=25).mean().iloc[-1]
            if len(prices) >= 99:
                indicators["sma_99"] = df["price"].rolling(window=99).mean().iloc[-1]

            # Exponential Moving Average
            if len(prices) >= 12:
                indicators["ema_12"] = df["price"].ewm(span=12).mean().iloc[-1]
            if len(prices) >= 26:
                indicators["ema_26"] = df["price"].ewm(span=26).mean().iloc[-1]

            # RSI (Relative Strength Index)
            if len(prices) >= 14:
                delta = df["price"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators["rsi"] = (100 - (100 / (1 + rs))).iloc[-1]

            # MACD
            if len(prices) >= 26:
                ema_12 = df["price"].ewm(span=12).mean()
                ema_26 = df["price"].ewm(span=26).mean()
                macd = ema_12 - ema_26
                signal = macd.ewm(span=9).mean()
                indicators["macd"] = macd.iloc[-1]
                indicators["macd_signal"] = signal.iloc[-1]
                indicators["macd_histogram"] = (macd - signal).iloc[-1]

            # Bollinger Bands
            if len(prices) >= 20:
                sma_20 = df["price"].rolling(window=20).mean()
                std_20 = df["price"].rolling(window=20).std()
                indicators["bb_upper"] = (sma_20 + 2 * std_20).iloc[-1]
                indicators["bb_middle"] = sma_20.iloc[-1]
                indicators["bb_lower"] = (sma_20 - 2 * std_20).iloc[-1]

            # Volume indicators (if volume data provided)
            if volumes and len(volumes) >= 20:
                indicators["volume_sma"] = (
                    pd.Series(volumes).rolling(window=20).mean().iloc[-1]
                )
                indicators["volume_ratio"] = volumes[-1] / indicators["volume_sma"]

        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")

        return indicators

    def score_trade_opportunity(
        self, symbol: str, prediction: Dict, indicators: Dict, patterns: List[Dict]
    ) -> Dict:
        """
        Score a trading opportunity based on all available data

        Returns:
            Dictionary with overall score and recommendation
        """
        try:
            score = 0.5  # Neutral starting point
            reasons = []

            # Prediction direction
            if prediction["direction"] == "bullish":
                score += prediction["confidence"] * 0.3
                reasons.append(
                    f"Bullish prediction (+{prediction['change_percent']:.2f}%)"
                )
            elif prediction["direction"] == "bearish":
                score -= prediction["confidence"] * 0.3
                reasons.append(
                    f"Bearish prediction ({prediction['change_percent']:.2f}%)"
                )

            # RSI analysis
            if "rsi" in indicators:
                rsi = indicators["rsi"]
                if rsi < 30:
                    score += 0.15
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    score -= 0.15
                    reasons.append(f"RSI overbought ({rsi:.1f})")

            # MACD analysis
            if "macd_histogram" in indicators:
                if indicators["macd_histogram"] > 0:
                    score += 0.1
                    reasons.append("MACD bullish")
                else:
                    score -= 0.1
                    reasons.append("MACD bearish")

            # Pattern analysis
            for pattern in patterns:
                if pattern["signal"] == "bullish":
                    score += pattern["confidence"] * 0.15
                    reasons.append(f"Pattern: {pattern['pattern']}")
                elif pattern["signal"] == "bearish":
                    score -= pattern["confidence"] * 0.15
                    reasons.append(f"Pattern: {pattern['pattern']}")

            # Normalize score to 0-1 range
            score = max(0, min(1, score))

            # Generate recommendation
            if score >= 0.7:
                recommendation = "strong_buy"
            elif score >= 0.6:
                recommendation = "buy"
            elif score >= 0.4:
                recommendation = "hold"
            elif score >= 0.3:
                recommendation = "sell"
            else:
                recommendation = "strong_sell"

            return {
                "symbol": symbol,
                "score": score,
                "recommendation": recommendation,
                "reasons": reasons,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Trade scoring error: {e}")
            return {
                "symbol": symbol,
                "score": 0.5,
                "recommendation": "hold",
                "reasons": ["Error in analysis"],
                "timestamp": datetime.now().isoformat(),
            }

    def update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.max_history_length)

        self.price_history[symbol].append({"price": price, "timestamp": datetime.now()})

    def get_price_history(self, symbol: str, limit: int = 100) -> List[float]:
        """Get price history for a symbol"""
        if symbol not in self.price_history:
            return []

        history = list(self.price_history[symbol])[-limit:]
        return [entry["price"] for entry in history]
