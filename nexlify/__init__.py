"""
Nexlify - AI-Powered Cryptocurrency Trading Platform
"""
__version__ = "2.0.7.7"

# Re-export main components for easy access
from nexlify.core import (
    ArasakaNeuralNet,
    NexlifyNeuralNet,
    AutoTrader,
    TradingIntegrationManager
)

from nexlify.risk import (
    RiskManager,
    CircuitBreaker,
    FlashCrashProtection,
    EmergencyKillSwitch
)

from nexlify.security import SecuritySuite

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
