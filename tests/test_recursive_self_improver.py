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


def test_generate_candidates_not_biased_to_first_exchange_when_capped():
    improver = RecursiveSelfImprover(
        {
            "max_candidates_per_generation": 12,
            "exchanges": ["binance", "coinbase"],
            "strategy_families": ["trend_following", "mean_reversion", "breakout", "market_making"],
            "model_types": ["dqn", "double_dqn", "ensemble"],
            "risk_profiles": ["conservative", "balanced", "aggressive"],
        }
    )

    candidates = improver.generate_candidates(generation=1)
    exchanges = {c.exchange for c in candidates}

    assert len(candidates) == 12
    assert "binance" in exchanges
    assert "coinbase" in exchanges


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



def test_champion_snapshot_empty_before_training():
    improver = RecursiveSelfImprover()
    assert improver.champion_snapshot() == {}


def test_seeded_generation_mutates_params_with_bounds():
    improver = RecursiveSelfImprover(
        {
            "max_candidates_per_generation": 1,
            "base_params": {"learning_rate": 0.001, "gamma": 0.99},
        }
    )

    seed = self_improvement_loop.StrategyCandidate(
        generation=1,
        name="seed",
        exchange="binance",
        strategy_family="trend_following",
        model_type="dqn",
        risk_profile="balanced",
        params={"learning_rate": 0.000001, "gamma": 0.9999},
    )

    candidates = improver.generate_candidates(generation=2, seed_candidate=seed)

    assert len(candidates) == 1
    params = candidates[0].params
    assert params["learning_rate"] == 1e-5
    assert params["gamma"] == 0.9999


def test_evaluate_generation_defaults_missing_score_and_preserves_champion():
    improver = RecursiveSelfImprover(
        {
            "max_candidates_per_generation": 2,
            "exchanges": ["binance", "coinbase"],
            "strategy_families": ["trend_following"],
            "model_types": ["dqn"],
            "risk_profiles": ["balanced"],
        }
    )

    winner_1 = improver.evaluate_generation(
        generation=1,
        evaluate_fn=lambda _c: {"score": 5, "sharpe": 1},
    )
    assert winner_1.score == 5.0

    winner_2 = improver.evaluate_generation(
        generation=2,
        evaluate_fn=lambda _c: {"sharpe": 0.5},
    )

    assert winner_2.score == 0.0
    assert improver.champion is not None
    assert improver.champion.score == 5.0


def test_runner_uses_default_weights_and_continuous_rounds():
    improver = RecursiveSelfImprover(
        {
            "max_candidates_per_generation": 1,
            "exchanges": ["binance"],
            "strategy_families": ["trend_following"],
            "model_types": ["dqn"],
            "risk_profiles": ["balanced"],
        }
    )

    runner = AutomatedSelfImprovementRunner(
        improver=improver,
        evaluators={"m1": lambda _c: 2.0, "m2": lambda _c: 4.0},
    )

    sample = self_improvement_loop.StrategyCandidate(
        generation=1,
        name="x",
        exchange="binance",
        strategy_family="trend_following",
        model_type="dqn",
        risk_profile="balanced",
    )
    metrics = runner.evaluate_candidate(sample)
    assert metrics["score"] == 3.0

    champion = runner.run_continuous(rounds=2, cycles_per_round=1)
    assert champion.score > 0
    assert len(runner.automation_log) == 2


def test_runner_continuous_runtime_error_when_no_champion_and_invalid_rounds():
    class BrokenImprover:
        champion = None

        def run_recursive_cycles(self, cycles, evaluate_fn):
            return None

        def champion_snapshot(self):
            return {}

    runner = AutomatedSelfImprovementRunner(
        improver=BrokenImprover(),
        evaluators={"m1": lambda _c: 1.0},
    )

    try:
        runner.run_continuous(rounds=0)
        assert False, "Expected ValueError for rounds <= 0"
    except ValueError:
        assert True

    try:
        runner.run_continuous(rounds=1)
        assert False, "Expected RuntimeError when no champion is produced"
    except RuntimeError:
        assert True
