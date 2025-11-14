"""Risk management and protection systems."""

from nexlify.risk.nexlify_risk_manager import RiskManager
from nexlify.risk.nexlify_circuit_breaker import (
    ExchangeCircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
)
from nexlify.risk.nexlify_flash_crash_protection import (
    FlashCrashProtection,
    CrashSeverity,
)
from nexlify.risk.nexlify_emergency_kill_switch import (
    EmergencyKillSwitch,
    KillSwitchTrigger,
)

# Aliases for backward compatibility and cleaner imports
CircuitBreaker = ExchangeCircuitBreaker

__all__ = [
    "RiskManager",
    "CircuitBreaker",
    "ExchangeCircuitBreaker",
    "CircuitBreakerManager",
    "CircuitState",
    "FlashCrashProtection",
    "CrashSeverity",
    "EmergencyKillSwitch",
    "KillSwitchTrigger",
]
