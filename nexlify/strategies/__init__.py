"""Trading strategies and ML models."""

from nexlify.strategies.nexlify_multi_strategy import MultiStrategyOptimizer
from nexlify.strategies.nexlify_multi_timeframe import MultiTimeframeAnalyzer
from nexlify.strategies.nexlify_predictive_features import PredictiveEngine
from nexlify.strategies.nexlify_rl_agent import RLAgent

# Ultra-Optimized RL Agent (optional - graceful if dependencies missing)
try:
    from nexlify.strategies.nexlify_ultra_optimized_rl_agent import UltraOptimizedDQNAgent

    __all__ = [
        'MultiStrategyOptimizer',
        'MultiTimeframeAnalyzer',
        'PredictiveEngine',
        'RLAgent',
        'UltraOptimizedDQNAgent',  # Ultra-optimized version with all optimizations
    ]
except ImportError:
    # Ultra-optimized agent not available (missing dependencies)
    __all__ = [
        'MultiStrategyOptimizer',
        'MultiTimeframeAnalyzer',
        'PredictiveEngine',
        'RLAgent',
    ]
