"""Trading strategies and ML models."""

from nexlify.strategies.nexlify_multi_strategy import MultiStrategyOptimizer
from nexlify.strategies.nexlify_multi_timeframe import MultiTimeframeAnalyzer
from nexlify.strategies.nexlify_predictive_features import PredictiveEngine
from nexlify.strategies.nexlify_rl_agent import DQNAgent
from nexlify.strategies.epsilon_decay import (
    EpsilonDecayStrategy,
    LinearEpsilonDecay,
    ScheduledEpsilonDecay,
    ExponentialEpsilonDecay,
    EpsilonDecayFactory,
)
from nexlify.strategies.gamma_selector import (
    GammaSelector,
    get_recommended_gamma,
    TRADING_STYLES,
)
from nexlify.strategies.multi_gamma_trainer import (
    MultiGammaTrainer,
    HardwareBenchmark,
)

# Backward compatibility aliases
MultiStrategy = MultiStrategyOptimizer
MultiTimeframe = MultiTimeframeAnalyzer
PredictiveFeatures = PredictiveEngine
RLAgent = DQNAgent

# Ultra-Optimized RL Agent (optional - graceful if dependencies missing)
try:
    from nexlify.strategies.nexlify_ultra_optimized_rl_agent import \
        UltraOptimizedDQNAgent

    __all__ = [
        "MultiStrategyOptimizer",
        "MultiStrategy",
        "MultiTimeframeAnalyzer",
        "MultiTimeframe",
        "PredictiveEngine",
        "PredictiveFeatures",
        "DQNAgent",
        "RLAgent",
        "UltraOptimizedDQNAgent",  # Ultra-optimized version with all optimizations
        "EpsilonDecayStrategy",
        "LinearEpsilonDecay",
        "ScheduledEpsilonDecay",
        "ExponentialEpsilonDecay",
        "EpsilonDecayFactory",
        "GammaSelector",
        "get_recommended_gamma",
        "TRADING_STYLES",
        "MultiGammaTrainer",
        "HardwareBenchmark",
    ]
except ImportError:
    # Ultra-optimized agent not available (missing dependencies)
    __all__ = [
        "MultiStrategyOptimizer",
        "MultiStrategy",
        "MultiTimeframeAnalyzer",
        "MultiTimeframe",
        "PredictiveEngine",
        "PredictiveFeatures",
        "DQNAgent",
        "RLAgent",
        "EpsilonDecayStrategy",
        "LinearEpsilonDecay",
        "ScheduledEpsilonDecay",
        "ExponentialEpsilonDecay",
        "EpsilonDecayFactory",
        "GammaSelector",
        "get_recommended_gamma",
        "TRADING_STYLES",
        "MultiGammaTrainer",
        "HardwareBenchmark",
    ]
