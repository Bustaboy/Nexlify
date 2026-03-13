# Recursive Self-Improvement Guide

This guide explains how Nexlify's recursive self-improvement system works, what is automated, and how to plug in your own evaluators.

## What it is

The self-improvement system is implemented in:

- `nexlify/training/self_improvement_loop.py`

It contains two layers:

1. **`RecursiveSelfImprover`** (core loop)
   - generates candidate combinations (exchange, strategy family, model type, risk profile),
   - evaluates each candidate,
   - tracks full history,
   - promotes a champion (best-so-far candidate).

2. **`AutomatedSelfImprovementRunner`** (automation layer)
   - runs multiple evaluators per candidate,
   - combines metrics into a weighted score,
   - executes one-shot or continuous rounds automatically,
   - logs automation sessions.

## Is it automated?

**Yes, when you use `AutomatedSelfImprovementRunner`.**

- `RecursiveSelfImprover` by itself is orchestration logic and needs an evaluation function.
- `AutomatedSelfImprovementRunner` makes the process operational by calling registered evaluators, scoring candidates, and recursively running improvement rounds.

## Core data structures

- **`StrategyCandidate`**: one generated strategy proposal.
- **`CandidateEvaluation`**: candidate + numeric score + metric breakdown + timestamp.

## Typical flow

1. Configure search space (exchanges, strategy families, model types, risk profiles).
2. Register evaluator functions.
3. Set metric weights.
4. Run `run_once(cycles=...)` for one automated batch.
5. Run `run_continuous(rounds=..., cycles_per_round=...)` for long-running improvement.
6. Consume `improver.champion_snapshot()` and `runner.automation_log`.

## Minimal example

```python
from nexlify.training import RecursiveSelfImprover, AutomatedSelfImprovementRunner

improver = RecursiveSelfImprover(
    {
        "max_candidates_per_generation": 6,
        "exchanges": ["binance", "coinbase", "kraken"],
        "strategy_families": ["trend_following", "mean_reversion", "breakout"],
        "model_types": ["dqn", "double_dqn"],
        "risk_profiles": ["conservative", "balanced"],
        "base_params": {"learning_rate": 0.001, "gamma": 0.99},
    }
)

runner = AutomatedSelfImprovementRunner(
    improver=improver,
    evaluators={
        "return": lambda c: 1.2 if c.exchange == "coinbase" else 0.8,
        "risk": lambda c: 0.7,
        "latency": lambda c: 0.9,
    },
    metric_weights={
        "return": 0.6,
        "risk": 0.3,
        "latency": 0.1,
    },
)

champion = runner.run_continuous(rounds=3, cycles_per_round=2)
print(champion.score)
print(improver.champion_snapshot())
```

## Evaluator design recommendations

- Keep each evaluator focused (e.g., one for return, one for drawdown/risk, one for slippage).
- Return normalized values where possible, so weights are meaningful.
- Make evaluators deterministic in tests and robust in production.
- If integrating backtesting/paper trading, include timeouts and failure fallbacks.

## Safety notes

- This module **does not place trades by itself**.
- It selects the best candidate according to evaluator outputs.
- Execution should remain behind existing risk controls and paper-trading validation before deployment.

## Related APIs

Exports available from `nexlify.training`:

- `StrategyCandidate`
- `CandidateEvaluation`
- `RecursiveSelfImprover`
- `AutomatedSelfImprovementRunner`
