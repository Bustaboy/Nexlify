#!/usr/bin/env python3
"""Unit tests for PredictiveEngine blended-signal prediction flow."""

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("numpy")
pytest.importorskip("pandas")

from nexlify.strategies.nexlify_predictive_features import PredictiveEngine


def _generate_prices(length: int = 200, start: float = 100.0):
    """Generate deterministic non-flat price data for tests."""
    prices = [start]
    for idx in range(1, length):
        drift = 0.002 if (idx // 25) % 2 == 0 else -0.001
        prices.append(prices[-1] * (1 + drift))
    return prices


def test_predict_price_insufficient_history_returns_neutral():
    engine = PredictiveEngine()

    result = engine.predict_price("BTCUSDT", 100.0, [100.0, 100.5, 100.8])

    assert result["predicted_price"] == 100.0
    assert result["confidence"] == 0.0
    assert result["direction"] == "neutral"
    assert result["change_percent"] == 0.0


def test_predict_price_returns_enriched_prediction_payload():
    engine = PredictiveEngine()
    prices = _generate_prices(180)
    current_price = prices[-1]

    result = engine.predict_price("BTCUSDT", current_price, prices)

    assert result["symbol"] == "BTCUSDT"
    assert "expected_return" in result
    assert "model_version" in result
    assert result["model_version"] == "blended_signal_v2"
    assert 0.0 <= result["confidence"] <= 0.95
    assert result["direction"] in {"bullish", "bearish", "neutral"}


def test_tune_model_for_symbol_persists_best_params_with_signal_scale():
    engine = PredictiveEngine()
    prices = _generate_prices(240)

    result = engine.tune_model_for_symbol("ETHUSDT", prices, horizon=5)

    assert result["status"] == "tuned"
    assert "validation_accuracy" in result
    assert "best_params" in result
    assert "signal_scale" in result["best_params"]
    assert "ETHUSDT" in engine.model_params


def test_record_prediction_outcome_and_confidence_calibration():
    engine = PredictiveEngine()

    # Build history where bullish is correct ~80% of the time
    for idx in range(25):
        entry = 100.0
        realized = 101.0 if idx < 20 else 99.0
        engine.record_prediction_outcome("BTCUSDT", "bullish", entry, realized)

    history = engine.prediction_history["BTCUSDT"]
    assert len(history) == 25
    assert history[-1]["predicted_direction"] == "bullish"

    calibrated = engine._calibrate_confidence("BTCUSDT", 0.50, "bullish")
    # Pull toward empirical hit-rate (0.8) from baseline 0.50 => should increase
    assert calibrated > 0.50
    assert calibrated <= 0.95


def test_tune_model_for_symbol_insufficient_data():
    engine = PredictiveEngine()
    prices = _generate_prices(30)

    result = engine.tune_model_for_symbol("SOLUSDT", prices, horizon=5, min_samples=120)

    assert result["status"] == "insufficient_data"
    assert result["required_samples"] == 120
    assert result["available_samples"] == 30
