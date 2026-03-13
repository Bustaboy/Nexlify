#!/usr/bin/env python3
"""
Recursive self-improvement loop for strategy/model/exchange selection.

This module provides a safe orchestration layer that can:
- generate strategy candidates from configurable templates,
- evaluate them with a pluggable scoring callback,
- promote the best candidate into the active policy,
- and iterate for multiple generations.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from typing import Callable, Dict, List, Optional


@dataclass
class StrategyCandidate:
    """Single strategy/model/exchange proposal in a generation."""

    generation: int
    name: str
    exchange: str
    strategy_family: str
    model_type: str
    risk_profile: str
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateEvaluation:
    """Evaluation output for a candidate."""

    candidate: StrategyCandidate
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class RecursiveSelfImprover:
    """Evolutionary coordinator for recursive strategy improvement."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.max_candidates_per_generation = int(cfg.get("max_candidates_per_generation", 12))
        self.exchanges = list(cfg.get("exchanges", ["binance", "coinbase"]))
        self.strategy_families = list(
            cfg.get(
                "strategy_families",
                ["trend_following", "mean_reversion", "breakout", "market_making"],
            )
        )
        self.model_types = list(cfg.get("model_types", ["dqn", "double_dqn", "ensemble"]))
        self.risk_profiles = list(cfg.get("risk_profiles", ["conservative", "balanced", "aggressive"]))
        self.base_params = dict(cfg.get("base_params", {"learning_rate": 0.001, "gamma": 0.99}))

        self._history: List[CandidateEvaluation] = []
        self._champion: Optional[CandidateEvaluation] = None

    @property
    def history(self) -> List[CandidateEvaluation]:
        return list(self._history)

    @property
    def champion(self) -> Optional[CandidateEvaluation]:
        return self._champion

    def _candidate_name(self, generation: int, index: int, exchange: str, family: str, model: str) -> str:
        return f"gen{generation}_c{index}_{exchange}_{family}_{model}"

    def generate_candidates(self, generation: int, seed_candidate: Optional[StrategyCandidate] = None) -> List[StrategyCandidate]:
        """Generate candidate pool for a generation."""
        candidates: List[StrategyCandidate] = []

        # IMPORTANT: keep exchanges as the innermost loop so that when
        # max_candidates_per_generation truncates the cartesian product,
        # candidate selection still spans all exchanges instead of being
        # biased toward the first configured exchange.
        for idx, (family, model, risk, exchange) in enumerate(
            product(self.strategy_families, self.model_types, self.risk_profiles, self.exchanges),
            start=1,
        ):
            params = dict(self.base_params)

            if seed_candidate is not None:
                params.update(seed_candidate.params)
                params["learning_rate"] = max(1e-5, float(params.get("learning_rate", 0.001)) * 0.98)
                params["gamma"] = min(0.9999, float(params.get("gamma", 0.99)) + 0.0005)

            candidates.append(
                StrategyCandidate(
                    generation=generation,
                    name=self._candidate_name(generation, idx, exchange, family, model),
                    exchange=exchange,
                    strategy_family=family,
                    model_type=model,
                    risk_profile=risk,
                    params=params,
                )
            )

            if len(candidates) >= self.max_candidates_per_generation:
                break

        return candidates

    def evaluate_generation(
        self,
        generation: int,
        evaluate_fn: Callable[[StrategyCandidate], Dict[str, float]],
    ) -> CandidateEvaluation:
        """Evaluate one generation and return the winner."""
        seed = self._champion.candidate if self._champion else None
        candidates = self.generate_candidates(generation=generation, seed_candidate=seed)

        evaluated: List[CandidateEvaluation] = []
        for candidate in candidates:
            result = evaluate_fn(candidate)
            score = float(result.get("score", 0.0))
            metrics = {k: float(v) for k, v in result.items() if k != "score"}
            evaluated.append(CandidateEvaluation(candidate=candidate, score=score, metrics=metrics))

        winner = max(evaluated, key=lambda item: item.score)
        self._history.extend(evaluated)

        if self._champion is None or winner.score > self._champion.score:
            self._champion = winner

        return winner

    def run_recursive_cycles(
        self,
        cycles: int,
        evaluate_fn: Callable[[StrategyCandidate], Dict[str, float]],
    ) -> CandidateEvaluation:
        """Run multiple generations and return the best overall strategy."""
        if cycles <= 0:
            raise ValueError("cycles must be > 0")

        latest_winner: Optional[CandidateEvaluation] = None
        for generation in range(1, cycles + 1):
            latest_winner = self.evaluate_generation(generation=generation, evaluate_fn=evaluate_fn)

        if self._champion is None:
            raise RuntimeError("No champion selected during recursive cycles")

        return self._champion if latest_winner is not None else self._champion

    def champion_snapshot(self) -> Dict:
        """Get serializable champion payload for logging/storage."""
        if self._champion is None:
            return {}

        return {
            "score": self._champion.score,
            "metrics": dict(self._champion.metrics),
            "candidate": asdict(self._champion.candidate),
            "timestamp_utc": self._champion.timestamp_utc,
        }


class AutomatedSelfImprovementRunner:
    """
    Fully automated wrapper around RecursiveSelfImprover.

    It evaluates every generated candidate through registered evaluators
    (e.g., backtest, paper trading, latency, risk), computes a weighted score,
    and continuously runs recursive improvement cycles.
    """

    def __init__(
        self,
        improver: RecursiveSelfImprover,
        evaluators: Dict[str, Callable[[StrategyCandidate], float]],
        metric_weights: Optional[Dict[str, float]] = None,
    ):
        if not evaluators:
            raise ValueError("At least one evaluator is required for automation")

        self.improver = improver
        self.evaluators = dict(evaluators)
        default_weight = 1.0 / len(self.evaluators)
        self.metric_weights = {
            name: float((metric_weights or {}).get(name, default_weight))
            for name in self.evaluators
        }
        self.automation_log: List[Dict] = []

    def evaluate_candidate(self, candidate: StrategyCandidate) -> Dict[str, float]:
        """Run all evaluators and combine to a single score."""
        metrics: Dict[str, float] = {}
        weighted_score = 0.0

        for metric_name, evaluator in self.evaluators.items():
            metric_value = float(evaluator(candidate))
            metrics[metric_name] = metric_value
            weighted_score += metric_value * self.metric_weights.get(metric_name, 0.0)

        metrics["score"] = weighted_score
        return metrics

    def run_once(self, cycles: int = 1) -> CandidateEvaluation:
        """Run one fully automated improvement session."""
        champion = self.improver.run_recursive_cycles(
            cycles=cycles,
            evaluate_fn=self.evaluate_candidate,
        )
        self.automation_log.append(
            {
                "cycles": cycles,
                "champion": self.improver.champion_snapshot(),
                "evaluators": list(self.evaluators.keys()),
            }
        )
        return champion

    def run_continuous(
        self,
        rounds: int,
        cycles_per_round: int = 1,
        sleep_seconds: float = 0.0,
    ) -> CandidateEvaluation:
        """Run repeated automated rounds and return current global champion."""
        if rounds <= 0:
            raise ValueError("rounds must be > 0")

        for _ in range(rounds):
            self.run_once(cycles=cycles_per_round)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if self.improver.champion is None:
            raise RuntimeError("Automation completed without selecting champion")
        return self.improver.champion
