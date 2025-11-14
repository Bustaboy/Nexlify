"""
Nexlify Machine Learning Module
Comprehensive ML system for cryptocurrency trading
"""

from nexlify.ml.nexlify_feature_engineering import (FeatureEngineer,
                                                    quick_engineer_features)

# Ultra-Optimized System Components (optional imports - graceful if dependencies missing)
try:
    from nexlify.ml.nexlify_gpu_optimizations import GPUOptimizer
    from nexlify.ml.nexlify_model_compilation import (ModelCompiler,
                                                      compile_model)
    from nexlify.ml.nexlify_multi_gpu import MultiGPUManager
    from nexlify.ml.nexlify_optimization_manager import (OptimizationManager,
                                                         OptimizationProfile)
    from nexlify.ml.nexlify_quantization import AutoQuantizer, quantize_model
    from nexlify.ml.nexlify_sentiment_analysis import SentimentAnalyzer
    from nexlify.ml.nexlify_smart_cache import SmartCache
    from nexlify.ml.nexlify_thermal_monitor import ThermalMonitor

    __all__ = [
        "FeatureEngineer",
        "quick_engineer_features",
        # Ultra-Optimized System
        "OptimizationManager",
        "OptimizationProfile",
        "MultiGPUManager",
        "ThermalMonitor",
        "SmartCache",
        "ModelCompiler",
        "compile_model",
        "AutoQuantizer",
        "quantize_model",
        "SentimentAnalyzer",
        "GPUOptimizer",
    ]
except ImportError as e:
    # Ultra-optimized components not available (missing dependencies)
    __all__ = ["FeatureEngineer", "quick_engineer_features"]
