#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "nexlify" / "training" / "self_improvement_loop.py"
spec = importlib.util.spec_from_file_location("self_improvement_loop", MODULE_PATH)
self_improvement_loop = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = self_improvement_loop
spec.loader.exec_module(self_improvement_loop)
RecursiveSelfImprover = self_improvement_loop.RecursiveSelfImprover
AutomatedSelfImprovementRunner = self_improvement_loop.AutomatedSelfImprovementRunner


def test_generate_candidates_respects_cap():
    improver = RecursiveSelfImprover(
        {
            "max_candidates_per_generation": 3,
            "exchanges": ["binance", "kraken"],
            "strategy_families": ["trend_following", "mean_reversion"],
            "model_types": ["dqn"],
            "risk_profiles": ["balanced"],
        }
    )

    candidates = improver.generate_candidates(generation=1)

    assert len(candidates) == 3
    assert all(c.generation == 1 for c in candidates)


def test_recursive_cycles_promote_best_champion():
    improver = RecursiveSelfImprover(
        {
            "max_candidates_per_generation": 4,
            "exchanges": ["binance", "coinbase"],
            "strategy_families": ["trend_following", "breakout"],
            "model_types": ["dqn"],
            "risk_profiles": ["balanced"],
        }
    )

    def evaluate(candidate):
        bonus = 1.0 if candidate.exchange == "coinbase" else 0.5
        return {
            "score": bonus + (0.2 if candidate.strategy_family == "breakout" else 0.1),
            "sharpe": 1.25,
            "max_drawdown": 4.5,
        }

    champion = improver.run_recursive_cycles(cycles=2, evaluate_fn=evaluate)

    assert champion.score > 0
    assert improver.champion is not None
    assert improver.champion.candidate.exchange == "coinbase"
    assert len(improver.history) == 8


def test_recursive_cycles_invalid_input():
    improver = RecursiveSelfImprover()

    def evaluate(_candidate):
        return {"score": 1.0}

    try:
        improver.run_recursive_cycles(cycles=0, evaluate_fn=evaluate)
        assert False, "Expected ValueError for cycles <= 0"
    except ValueError:
        assert True


def test_automated_runner_scores_and_promotes():
    improver = RecursiveSelfImprover(
        {
            "max_candidates_per_generation": 2,
            "exchanges": ["binance", "coinbase"],
            "strategy_families": ["trend_following"],
            "model_types": ["dqn"],
            "risk_profiles": ["balanced"],
        }
    )

    runner = AutomatedSelfImprovementRunner(
        improver=improver,
        evaluators={
            "return": lambda c: 1.0 if c.exchange == "coinbase" else 0.2,
            "risk": lambda c: 0.8,
        },
        metric_weights={"return": 0.7, "risk": 0.3},
    )

    champion = runner.run_once(cycles=1)

    assert champion.score > 0
    assert improver.champion is not None
    assert improver.champion.candidate.exchange == "coinbase"
    assert len(runner.automation_log) == 1


def test_automated_runner_rejects_empty_evaluators():
    improver = RecursiveSelfImprover()

    try:
        AutomatedSelfImprovementRunner(improver=improver, evaluators={})
        assert False, "Expected ValueError for empty evaluator set"
    except ValueError:
        assert True
