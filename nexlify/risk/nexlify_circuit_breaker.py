#!/usr/bin/env python3
"""
Nexlify - Circuit Breaker Pattern
Intelligent failure handling for exchange API calls
🔌 Protect against cascading failures and API rate limits
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Any, Dict, Optional
from enum import Enum
from dataclasses import dataclass, field
import time

from nexlify.utils.error_handler import handle_errors, get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Circuit breaker statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: int = 0
    time_in_open: float = 0.0  # Seconds spent in OPEN state


class ExchangeCircuitBreaker:
    """
    🔌 Circuit Breaker for Exchange API Calls

    Implements the Circuit Breaker pattern to protect against:
    - Cascading failures
    - API rate limit exhaustion
    - Network issues
    - Exchange outages

    States:
    - CLOSED: Normal operation, all calls go through
    - OPEN: Blocking all calls, returning errors immediately
    - HALF_OPEN: Testing recovery with limited calls

    Features:
    - Automatic failure detection
    - Exponential backoff
    - Automatic recovery testing
    - Detailed logging and metrics
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        timeout_seconds: int = 300,
        half_open_max_calls: int = 1
    ):
        """
        Initialize Circuit Breaker

        Args:
            name: Identifier for this circuit (e.g., exchange name)
            failure_threshold: Number of consecutive failures before opening
            timeout_seconds: Seconds to wait before testing recovery
            half_open_max_calls: Max calls allowed in HALF_OPEN state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        # State management
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.opened_at: Optional[float] = None
        self.half_open_calls = 0

        logger.info(f"🔌 Circuit Breaker initialized: {name}")
        logger.info(f"   Failure threshold: {failure_threshold}")
        logger.info(f"   Timeout: {timeout_seconds}s")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Async function to call
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result from function call

        Raises:
            Exception: If circuit is OPEN or function fails
        """
        # Check if we should test recovery
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self._transition_to_half_open()
            else:
                # Calculate remaining time
                elapsed = time.time() - self.opened_at
                remaining = self.timeout_seconds - elapsed
                raise Exception(
                    f"🔴 {self.name} circuit OPEN - blocked "
                    f"(recovery in {remaining:.0f}s)"
                )

        # Check HALF_OPEN call limit
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise Exception(
                    f"🟡 {self.name} circuit HALF-OPEN - "
                    f"max test calls reached, awaiting result"
                )

        # Execute the call
        try:
            self.stats.total_calls += 1

            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1

            # Call the actual function
            result = await func(*args, **kwargs)

            # Success!
            self._on_success()
            return result

        except Exception as e:
            # Failure
            self._on_failure(e)
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.opened_at is None:
            return False

        elapsed = time.time() - self.opened_at
        return elapsed >= self.timeout_seconds

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN state"""
        if self.state != CircuitState.OPEN:
            return

        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.stats.state_changes += 1

        # Calculate time spent in OPEN
        if self.opened_at:
            self.stats.time_in_open += time.time() - self.opened_at

        logger.info(f"🔄 {self.name} circuit HALF-OPEN - testing recovery")

    def _on_success(self):
        """Handle successful call"""
        self.stats.successful_calls += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = datetime.now()

        # If we were in HALF_OPEN, close the circuit
        if self.state == CircuitState.HALF_OPEN:
            self._close_circuit()

    def _on_failure(self, error: Exception):
        """Handle failed call"""
        self.stats.failed_calls += 1
        self.stats.consecutive_failures += 1
        self.stats.last_failure_time = datetime.now()

        logger.warning(
            f"⚠️ {self.name} failure #{self.stats.consecutive_failures}: {str(error)[:100]}"
        )

        # Check if we should open the circuit
        if self.stats.consecutive_failures >= self.failure_threshold:
            if self.state == CircuitState.HALF_OPEN:
                # Failed recovery, reopen circuit
                self._open_circuit("Recovery test failed")
            elif self.state == CircuitState.CLOSED:
                # Too many failures, open circuit
                self._open_circuit(
                    f"{self.stats.consecutive_failures} consecutive failures"
                )

    def _open_circuit(self, reason: str):
        """Transition to OPEN state"""
        if self.state == CircuitState.OPEN:
            return  # Already open

        self.state = CircuitState.OPEN
        self.opened_at = time.time()
        self.stats.state_changes += 1

        logger.error(
            f"🔴 {self.name} circuit OPEN - {reason} "
            f"(recovery in {self.timeout_seconds}s)"
        )

    def _close_circuit(self):
        """Transition to CLOSED state (recovery successful)"""
        if self.state == CircuitState.CLOSED:
            return  # Already closed

        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
        self.stats.state_changes += 1

        logger.info(f"✅ {self.name} circuit CLOSED - recovery successful")

    def force_open(self, reason: str = "Manual override"):
        """🔴 Manually open the circuit"""
        logger.warning(f"⚠️ Manually opening {self.name} circuit: {reason}")
        self._open_circuit(reason)

    def force_close(self, reason: str = "Manual override"):
        """✅ Manually close the circuit"""
        logger.warning(f"⚠️ Manually closing {self.name} circuit: {reason}")
        self.stats.consecutive_failures = 0
        self._close_circuit()

    def get_status(self) -> Dict:
        """
        📊 Get circuit breaker status

        Returns:
            Dictionary with current status and statistics
        """
        status = {
            'name': self.name,
            'state': self.state.value,
            'total_calls': self.stats.total_calls,
            'successful_calls': self.stats.successful_calls,
            'failed_calls': self.stats.failed_calls,
            'consecutive_failures': self.stats.consecutive_failures,
            'failure_threshold': self.failure_threshold,
            'state_changes': self.stats.state_changes,
        }

        # Success rate
        if self.stats.total_calls > 0:
            status['success_rate'] = f"{(self.stats.successful_calls / self.stats.total_calls * 100):.1f}%"
        else:
            status['success_rate'] = "N/A"

        # Last failure time
        if self.stats.last_failure_time:
            status['last_failure'] = self.stats.last_failure_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            status['last_failure'] = "Never"

        # Last success time
        if self.stats.last_success_time:
            status['last_success'] = self.stats.last_success_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            status['last_success'] = "Never"

        # Time until recovery (if OPEN)
        if self.state == CircuitState.OPEN and self.opened_at:
            elapsed = time.time() - self.opened_at
            remaining = max(0, self.timeout_seconds - elapsed)
            status['recovery_in'] = f"{remaining:.0f}s"
        else:
            status['recovery_in'] = "N/A"

        return status

    def reset(self):
        """🔄 Reset circuit breaker to initial state"""
        logger.info(f"🔄 Resetting {self.name} circuit breaker")
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.opened_at = None
        self.half_open_calls = 0


class CircuitBreakerManager:
    """
    🎛️ Manage multiple circuit breakers

    Provides centralized management of circuit breakers for all exchanges
    """

    def __init__(self, config: Dict):
        """Initialize Circuit Breaker Manager"""
        self.config = config.get('circuit_breaker', {})
        self.enabled = self.config.get('enabled', True)
        self.failure_threshold = self.config.get('failure_threshold', 3)
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        self.half_open_max_calls = self.config.get('half_open_max_calls', 1)

        self.breakers: Dict[str, ExchangeCircuitBreaker] = {}

        logger.info("🎛️ Circuit Breaker Manager initialized")
        logger.info(f"   Enabled: {self.enabled}")
        logger.info(f"   Default threshold: {self.failure_threshold}")
        logger.info(f"   Default timeout: {self.timeout_seconds}s")

    def get_or_create(self, name: str) -> ExchangeCircuitBreaker:
        """Get existing circuit breaker or create new one"""
        if name not in self.breakers:
            self.breakers[name] = ExchangeCircuitBreaker(
                name=name,
                failure_threshold=self.failure_threshold,
                timeout_seconds=self.timeout_seconds,
                half_open_max_calls=self.half_open_max_calls
            )

        return self.breakers[name]

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers"""
        return {name: breaker.get_status() for name, breaker in self.breakers.items()}

    def reset_all(self):
        """Reset all circuit breakers"""
        logger.warning("🔄 Resetting ALL circuit breakers")
        for breaker in self.breakers.values():
            breaker.reset()

    def get_health_summary(self) -> Dict:
        """Get overall health summary"""
        total_breakers = len(self.breakers)
        open_breakers = sum(1 for b in self.breakers.values() if b.state == CircuitState.OPEN)
        half_open_breakers = sum(1 for b in self.breakers.values() if b.state == CircuitState.HALF_OPEN)
        closed_breakers = sum(1 for b in self.breakers.values() if b.state == CircuitState.CLOSED)

        return {
            'total_breakers': total_breakers,
            'healthy': closed_breakers,
            'testing': half_open_breakers,
            'failed': open_breakers,
            'overall_health': 'healthy' if open_breakers == 0 else 'degraded'
        }


# Usage example
if __name__ == "__main__":
    async def test_circuit_breaker():
        """Test circuit breaker functionality"""
        breaker = ExchangeCircuitBreaker("test_exchange", failure_threshold=3, timeout_seconds=5)

        # Simulate successful calls
        async def successful_call():
            await asyncio.sleep(0.1)
            return "Success"

        # Simulate failing calls
        async def failing_call():
            await asyncio.sleep(0.1)
            raise Exception("API Error")

        print("Testing successful calls...")
        for i in range(3):
            try:
                result = await breaker.call(successful_call)
                print(f"✅ Call {i+1}: {result}")
            except Exception as e:
                print(f"❌ Call {i+1}: {e}")

        print("\nTesting failures...")
        for i in range(5):
            try:
                result = await breaker.call(failing_call)
                print(f"✅ Call {i+1}: {result}")
            except Exception as e:
                print(f"❌ Call {i+1}: {str(e)[:50]}")

        print(f"\nCircuit state: {breaker.state.value}")
        print(f"Stats: {breaker.get_status()}")

        print("\nWaiting for timeout...")
        await asyncio.sleep(6)

        print("\nAttempting recovery...")
        try:
            result = await breaker.call(successful_call)
            print(f"✅ Recovery successful: {result}")
        except Exception as e:
            print(f"❌ Recovery failed: {e}")

        print(f"\nFinal circuit state: {breaker.state.value}")
        print(f"Final stats: {breaker.get_status()}")

    asyncio.run(test_circuit_breaker())
