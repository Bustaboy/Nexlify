"""Trading strategies and ML models."""

from nexlify.strategies.nexlify_multi_strategy import MultiStrategyOptimizer
from nexlify.strategies.nexlify_multi_timeframe import MultiTimeframeAnalyzer
from nexlify.strategies.nexlify_predictive_features import PredictiveEngine
from nexlify.strategies.nexlify_rl_agent import DQNAgent

# Backward compatibility aliases
MultiStrategy = MultiStrategyOptimizer
MultiTimeframe = MultiTimeframeAnalyzer
PredictiveFeatures = PredictiveEngine
RLAgent = DQNAgent

__all__ = [
    'MultiStrategyOptimizer',
    'MultiStrategy',
    'MultiTimeframeAnalyzer',
    'MultiTimeframe',
    'PredictiveEngine',
    'PredictiveFeatures',
    'DQNAgent',
    'RLAgent',
]
