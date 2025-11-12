#!/usr/bin/env python3
"""
Unit tests for Nexlify Circuit Breaker
Comprehensive testing of circuit breaker functionality
"""

import pytest
import asyncio
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify_circuit_breaker import (
    ExchangeCircuitBreaker,
    CircuitBreakerManager,
    CircuitState
)


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker with short timeout for testing"""
    return ExchangeCircuitBreaker(
        name="test_exchange",
        failure_threshold=3,
        timeout_seconds=2,
        half_open_max_calls=1
    )


@pytest.fixture
def circuit_manager():
    """Create a circuit breaker manager"""
    config = {
        'circuit_breaker': {
            'enabled': True,
            'failure_threshold': 3,
            'timeout_seconds': 2,
            'half_open_max_calls': 1
        }
    }
    return CircuitBreakerManager(config)


# Mock functions for testing
async def successful_call(delay=0.01):
    """Simulate successful API call"""
    await asyncio.sleep(delay)
    return "success"


async def failing_call(delay=0.01, error_msg="API Error"):
    """Simulate failing API call"""
    await asyncio.sleep(delay)
    raise Exception(error_msg)


class TestCircuitBreakerInitialization:
    """Test circuit breaker initialization"""

    def test_initialization(self, circuit_breaker):
        """Test proper initialization"""
        assert circuit_breaker.name == "test_exchange"
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.timeout_seconds == 2
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.stats.total_calls == 0

    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        breaker = ExchangeCircuitBreaker(
            name="custom",
            failure_threshold=5,
            timeout_seconds=60,
            half_open_max_calls=3
        )
        assert breaker.failure_threshold == 5
        assert breaker.timeout_seconds == 60
        assert breaker.half_open_max_calls == 3


class TestSuccessfulCalls:
    """Test successful call handling"""

    @pytest.mark.asyncio
    async def test_single_successful_call(self, circuit_breaker):
        """Test single successful call"""
        result = await circuit_breaker.call(successful_call)

        assert result == "success"
        assert circuit_breaker.stats.total_calls == 1
        assert circuit_breaker.stats.successful_calls == 1
        assert circuit_breaker.stats.failed_calls == 0
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_multiple_successful_calls(self, circuit_breaker):
        """Test multiple successful calls"""
        for i in range(5):
            result = await circuit_breaker.call(successful_call)
            assert result == "success"

        assert circuit_breaker.stats.total_calls == 5
        assert circuit_breaker.stats.successful_calls == 5
        assert circuit_breaker.stats.consecutive_failures == 0
        assert circuit_breaker.state == CircuitState.CLOSED


class TestFailureHandling:
    """Test failure detection and handling"""

    @pytest.mark.asyncio
    async def test_single_failure(self, circuit_breaker):
        """Test single failure doesn't open circuit"""
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_call)

        assert circuit_breaker.stats.failed_calls == 1
        assert circuit_breaker.stats.consecutive_failures == 1
        assert circuit_breaker.state == CircuitState.CLOSED  # Still closed

    @pytest.mark.asyncio
    async def test_failure_threshold(self, circuit_breaker):
        """Test circuit opens after threshold failures"""
        # Fail threshold times
        for i in range(circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_call)

        # Circuit should now be OPEN
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.stats.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_failure_resets_on_success(self, circuit_breaker):
        """Test consecutive failures reset on success"""
        # Two failures
        for i in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_call)

        assert circuit_breaker.stats.consecutive_failures == 2

        # One success
        await circuit_breaker.call(successful_call)

        # Consecutive failures should reset
        assert circuit_breaker.stats.consecutive_failures == 0
        assert circuit_breaker.state == CircuitState.CLOSED


class TestOpenState:
    """Test circuit breaker in OPEN state"""

    @pytest.mark.asyncio
    async def test_open_blocks_calls(self, circuit_breaker):
        """Test OPEN circuit blocks all calls"""
        # Open the circuit
        for i in range(circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_call)

        assert circuit_breaker.state == CircuitState.OPEN

        # Attempt another call - should be blocked immediately
        with pytest.raises(Exception) as exc_info:
            await circuit_breaker.call(successful_call)

        assert "circuit OPEN" in str(exc_info.value)
        assert "blocked" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_open_doesnt_execute_function(self, circuit_breaker):
        """Test OPEN circuit doesn't execute the actual function"""
        # Open the circuit
        circuit_breaker.force_open("Test")

        call_count = 0

        async def counting_call():
            nonlocal call_count
            call_count += 1
            return "called"

        # Try to call
        with pytest.raises(Exception):
            await circuit_breaker.call(counting_call)

        # Function should NOT have been called
        assert call_count == 0


class TestRecovery:
    """Test circuit breaker recovery process"""

    @pytest.mark.asyncio
    async def test_automatic_recovery_attempt(self, circuit_breaker):
        """Test circuit attempts recovery after timeout"""
        # Open the circuit
        for i in range(circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_call)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(circuit_breaker.timeout_seconds + 0.5)

        # Next call should transition to HALF_OPEN
        result = await circuit_breaker.call(successful_call)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED  # Recovered

    @pytest.mark.asyncio
    async def test_half_open_state(self, circuit_breaker):
        """Test HALF_OPEN state behavior"""
        # Open the circuit
        circuit_breaker.force_open("Test")

        # Wait for timeout
        await asyncio.sleep(circuit_breaker.timeout_seconds + 0.5)

        # Should be ready for HALF_OPEN
        assert circuit_breaker._should_attempt_recovery()

    @pytest.mark.asyncio
    async def test_failed_recovery(self, circuit_breaker):
        """Test circuit reopens if recovery fails"""
        # Open the circuit
        for i in range(circuit_breaker.failure_threshold):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_call)

        # Wait for timeout
        await asyncio.sleep(circuit_breaker.timeout_seconds + 0.5)

        # Try to recover but fail
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_call)

        # Should be OPEN again
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_successful_recovery_closes_circuit(self, circuit_breaker):
        """Test successful recovery closes circuit"""
        # Open circuit
        circuit_breaker.force_open("Test")
        await asyncio.sleep(circuit_breaker.timeout_seconds + 0.5)

        # Successful recovery
        result = await circuit_breaker.call(successful_call)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.stats.consecutive_failures == 0


class TestManualControl:
    """Test manual circuit control"""

    def test_force_open(self, circuit_breaker):
        """Test manually opening circuit"""
        assert circuit_breaker.state == CircuitState.CLOSED

        circuit_breaker.force_open("Manual test")

        assert circuit_breaker.state == CircuitState.OPEN

    def test_force_close(self, circuit_breaker):
        """Test manually closing circuit"""
        # Open circuit
        circuit_breaker.force_open("Test")
        assert circuit_breaker.state == CircuitState.OPEN

        # Force close
        circuit_breaker.force_close("Manual test")

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.stats.consecutive_failures == 0

    def test_reset(self, circuit_breaker):
        """Test resetting circuit breaker"""
        # Make some calls
        circuit_breaker.stats.total_calls = 10
        circuit_breaker.stats.successful_calls = 7
        circuit_breaker.stats.failed_calls = 3
        circuit_breaker.force_open("Test")

        # Reset
        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.stats.total_calls == 0
        assert circuit_breaker.stats.successful_calls == 0
        assert circuit_breaker.stats.failed_calls == 0


class TestStatistics:
    """Test statistics and status reporting"""

    @pytest.mark.asyncio
    async def test_get_status(self, circuit_breaker):
        """Test status dictionary"""
        # Make some calls
        await circuit_breaker.call(successful_call)
        await circuit_breaker.call(successful_call)

        status = circuit_breaker.get_status()

        assert status['name'] == "test_exchange"
        assert status['state'] == CircuitState.CLOSED.value
        assert status['total_calls'] == 2
        assert status['successful_calls'] == 2
        assert status['failed_calls'] == 0
        assert 'success_rate' in status
        assert status['success_rate'] == "100.0%"

    @pytest.mark.asyncio
    async def test_success_rate_calculation(self, circuit_breaker):
        """Test success rate calculation"""
        # 3 successes, 2 failures
        await circuit_breaker.call(successful_call)
        await circuit_breaker.call(successful_call)

        try:
            await circuit_breaker.call(failing_call)
        except:
            pass

        try:
            await circuit_breaker.call(failing_call)
        except:
            pass

        await circuit_breaker.call(successful_call)

        status = circuit_breaker.get_status()
        # 3 of 5 = 60%
        assert status['success_rate'] == "60.0%"

    @pytest.mark.asyncio
    async def test_last_failure_tracking(self, circuit_breaker):
        """Test last failure time tracking"""
        try:
            await circuit_breaker.call(failing_call)
        except:
            pass

        status = circuit_breaker.get_status()
        assert status['last_failure'] != "Never"

    @pytest.mark.asyncio
    async def test_recovery_time_display(self, circuit_breaker):
        """Test recovery time display in OPEN state"""
        circuit_breaker.force_open("Test")

        status = circuit_breaker.get_status()
        assert status['recovery_in'] != "N/A"


class TestCircuitBreakerManager:
    """Test Circuit Breaker Manager"""

    def test_manager_initialization(self, circuit_manager):
        """Test manager initialization"""
        assert circuit_manager.enabled is True
        assert circuit_manager.failure_threshold == 3
        assert circuit_manager.timeout_seconds == 2

    def test_get_or_create(self, circuit_manager):
        """Test getting or creating circuit breakers"""
        breaker1 = circuit_manager.get_or_create("binance")
        breaker2 = circuit_manager.get_or_create("kraken")
        breaker3 = circuit_manager.get_or_create("binance")  # Same as breaker1

        assert breaker1.name == "binance"
        assert breaker2.name == "kraken"
        assert breaker1 is breaker3  # Same instance

    @pytest.mark.asyncio
    async def test_get_all_status(self, circuit_manager):
        """Test getting status of all breakers"""
        # Create some breakers
        binance = circuit_manager.get_or_create("binance")
        kraken = circuit_manager.get_or_create("kraken")

        # Make some calls
        await binance.call(successful_call)
        await kraken.call(successful_call)

        # Get all status
        all_status = circuit_manager.get_all_status()

        assert "binance" in all_status
        assert "kraken" in all_status
        assert all_status["binance"]["total_calls"] == 1
        assert all_status["kraken"]["total_calls"] == 1

    def test_reset_all(self, circuit_manager):
        """Test resetting all breakers"""
        # Create and use some breakers
        binance = circuit_manager.get_or_create("binance")
        kraken = circuit_manager.get_or_create("kraken")

        binance.stats.total_calls = 10
        kraken.stats.total_calls = 20

        # Reset all
        circuit_manager.reset_all()

        assert binance.stats.total_calls == 0
        assert kraken.stats.total_calls == 0

    def test_health_summary(self, circuit_manager):
        """Test health summary"""
        # Create breakers in different states
        binance = circuit_manager.get_or_create("binance")
        kraken = circuit_manager.get_or_create("kraken")
        coinbase = circuit_manager.get_or_create("coinbase")

        binance.state = CircuitState.CLOSED
        kraken.state = CircuitState.OPEN
        coinbase.state = CircuitState.HALF_OPEN

        summary = circuit_manager.get_health_summary()

        assert summary['total_breakers'] == 3
        assert summary['healthy'] == 1
        assert summary['testing'] == 1
        assert summary['failed'] == 1
        assert summary['overall_health'] == 'degraded'


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_multiple_rapid_failures(self, circuit_breaker):
        """Test handling multiple rapid failures"""
        # Fire off many failures rapidly
        for i in range(10):
            try:
                await circuit_breaker.call(failing_call, delay=0.001)
            except:
                pass

        # Should be OPEN after threshold
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_mixed_success_failure_pattern(self, circuit_breaker):
        """Test mixed success/failure pattern"""
        # Alternating success and failure
        for i in range(4):
            if i % 2 == 0:
                await circuit_breaker.call(successful_call)
            else:
                try:
                    await circuit_breaker.call(failing_call)
                except:
                    pass

        # Should still be CLOSED (consecutive failures reset)
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_exception_types(self, circuit_breaker):
        """Test handling different exception types"""
        async def timeout_error():
            raise TimeoutError("Request timeout")

        async def value_error():
            raise ValueError("Invalid value")

        # Both should be treated as failures
        try:
            await circuit_breaker.call(timeout_error)
        except:
            pass

        try:
            await circuit_breaker.call(value_error)
        except:
            pass

        assert circuit_breaker.stats.failed_calls == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
