"""Core trading and neural network components."""

from nexlify.core.arasaka_neural_net import ArasakaNeuralNet
from nexlify.core.nexlify_neural_net import NexlifyNeuralNet
from nexlify.core.nexlify_auto_trader import AutoExecutionEngine
from nexlify.core.nexlify_trading_integration import TradingIntegrationManager

# Backward compatibility alias
AutoTrader = AutoExecutionEngine

__all__ = [
    "ArasakaNeuralNet",
    "NexlifyNeuralNet",
    "AutoExecutionEngine",
    "AutoTrader",
    "TradingIntegrationManager",
]
