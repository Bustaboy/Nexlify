"""Core trading and neural network components."""

from nexlify.core.nexlify_auto_trader import AutoExecutionEngine

# Optional imports: avoid hard failures in lightweight test/dev environments.
try:
    from nexlify.core.arasaka_neural_net import ArasakaNeuralNet
except Exception:  # optional dependency tree (e.g., aiohttp/ccxt extras)
    ArasakaNeuralNet = None

try:
    from nexlify.core.nexlify_neural_net import NexlifyNeuralNet
except Exception:
    NexlifyNeuralNet = None

try:
    from nexlify.core.nexlify_trading_integration import TradingIntegrationManager
except Exception:
    TradingIntegrationManager = None

# Backward compatibility alias
AutoTrader = AutoExecutionEngine

__all__ = [
    "ArasakaNeuralNet",
    "NexlifyNeuralNet",
    "AutoExecutionEngine",
    "AutoTrader",
    "TradingIntegrationManager",
]
