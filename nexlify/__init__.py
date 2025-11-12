"""
Nexlify - AI-Powered Cryptocurrency Trading Platform
"""
__version__ = "2.0.7.7"

# For now, make imports lazy to avoid circular dependencies during testing
# Users can import from specific submodules: from nexlify.core import ArasakaNeuralNet
__all__ = [
    'ArasakaNeuralNet',
    'NexlifyNeuralNet',
    'AutoTrader',
    'TradingIntegrationManager',
    'RiskManager',
    'CircuitBreaker',
    'FlashCrashProtection',
    'EmergencyKillSwitch',
    'SecuritySuite',
]
