#!/usr/bin/env python3
"""
Nexlify Thermal and Power Monitoring

Monitors GPU/CPU temperature and power consumption to prevent:
- Thermal throttling (performance degradation)
- Hardware damage
- Power limit throttling
- Battery drain on laptops

LOW OVERHEAD: Checks run every 30 seconds in background thread
"""

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


class ThermalState(Enum):
    """Thermal states"""

    OPTIMAL = "optimal"  # < 70°C
    WARM = "warm"  # 70-80°C
    HOT = "hot"  # 80-85°C
    THROTTLING = "throttling"  # 85-95°C
    CRITICAL = "critical"  # > 95°C


class PowerState(Enum):
    """Power states"""

    UNLIMITED = "unlimited"  # Desktop, AC power
    BALANCED = "balanced"  # Balanced mode
    POWER_SAVER = "power_saver"  # Battery, power limit active
    THROTTLED = "throttled"  # Power limit exceeded


@dataclass
class ThermalSnapshot:
    """Snapshot of thermal state"""

    timestamp: float
    gpu_temps: List[float]
    gpu_power_watts: List[float]
    gpu_power_limits: List[float]
    cpu_temp: Optional[float]
    thermal_state: ThermalState
    power_state: PowerState
    is_throttling: bool
    on_battery: bool


class ThermalMonitor:
    """
    Monitor thermal and power state with LOW OVERHEAD

    Features:
    - Background monitoring (30s interval)
    - Thermal throttling detection
    - Power limit detection
    - Battery-aware optimization
    - Automatic batch size reduction on throttling
    """

    def __init__(self, check_interval: float = 30.0):
        """
        Args:
            check_interval: Seconds between checks (default: 30s for low overhead)
        """
        self.check_interval = check_interval
        self.history = deque(maxlen=20)  # Keep last 10 minutes (20 * 30s)

        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None

        # Cache
        self.last_snapshot: Optional[ThermalSnapshot] = None

        # Throttling detection
        self.throttle_count = 0
        self.throttle_cooldown = 0

        logger.info(
            f"🌡️  Thermal Monitor initialized (check interval: {check_interval}s)"
        )

    def start_monitoring(self):
        """Start background monitoring thread (LOW OVERHEAD)"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()
            logger.info("📊 Thermal monitoring started")

    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.check_interval + 1)
        logger.info("⏹️  Thermal monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop - runs every check_interval seconds"""
        while self.monitoring:
            try:
                snapshot = self.take_snapshot()
                self.history.append(snapshot)
                self.last_snapshot = snapshot

                # Check for throttling
                if snapshot.is_throttling:
                    self.throttle_count += 1
                    logger.warning(
                        f"⚠️  Thermal throttling detected! Count: {self.throttle_count}"
                    )
                    logger.warning(f"   GPU temps: {snapshot.gpu_temps}")
                    logger.warning(
                        f"   Recommendation: Reduce batch size or improve cooling"
                    )
                else:
                    # Decay throttle count
                    if self.throttle_count > 0:
                        self.throttle_count = max(0, self.throttle_count - 1)

                # Power state warnings
                if snapshot.power_state == PowerState.THROTTLED:
                    logger.warning("⚠️  GPU power limit exceeded - performance reduced")
                elif snapshot.on_battery:
                    logger.info(
                        "🔋 Running on battery - using power-efficient settings"
                    )

            except Exception as e:
                logger.debug(f"Thermal monitoring error: {e}")

            # Sleep for check_interval (low overhead!)
            time.sleep(self.check_interval)

    def take_snapshot(self) -> ThermalSnapshot:
        """Take snapshot of thermal/power state (CACHED for low overhead)"""
        timestamp = time.time()

        # GPU temps and power
        gpu_temps = []
        gpu_power_watts = []
        gpu_power_limits = []

        try:
            import torch

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()

                # Use nvidia-smi for detailed info (CACHED every 30s)
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=temperature.gpu,power.draw,power.limit",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )

                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")

                        for line in lines[:num_gpus]:
                            parts = line.split(",")
                            if len(parts) >= 3:
                                temp = float(parts[0].strip())
                                power = float(parts[1].strip())
                                limit = float(parts[2].strip())

                                gpu_temps.append(temp)
                                gpu_power_watts.append(power)
                                gpu_power_limits.append(limit)

                except Exception as e:
                    logger.debug(f"nvidia-smi query failed: {e}")

        except ImportError:
            pass

        # CPU temp (if available)
        cpu_temp = self._get_cpu_temp()

        # Determine thermal state
        if gpu_temps:
            max_temp = max(gpu_temps)
        elif cpu_temp:
            max_temp = cpu_temp
        else:
            max_temp = 50.0  # Assume optimal

        thermal_state = self._classify_thermal_state(max_temp)

        # Determine power state
        power_state = self._classify_power_state(gpu_power_watts, gpu_power_limits)

        # Check throttling
        is_throttling = (
            thermal_state in [ThermalState.THROTTLING, ThermalState.CRITICAL]
            or power_state == PowerState.THROTTLED
        )

        # Battery status
        on_battery = self._check_battery_status()

        return ThermalSnapshot(
            timestamp=timestamp,
            gpu_temps=gpu_temps,
            gpu_power_watts=gpu_power_watts,
            gpu_power_limits=gpu_power_limits,
            cpu_temp=cpu_temp,
            thermal_state=thermal_state,
            power_state=power_state,
            is_throttling=is_throttling,
            on_battery=on_battery,
        )

    def _get_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature (if available)"""
        try:
            # psutil.sensors_temperatures() on Linux
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()

                # Try common sensors
                for sensor_name in ["coretemp", "k10temp", "zenpower", "cpu_thermal"]:
                    if sensor_name in temps:
                        sensor_temps = temps[sensor_name]
                        if sensor_temps:
                            # Return max temperature
                            return max(t.current for t in sensor_temps)

        except Exception as e:
            logger.debug(f"CPU temperature read failed: {e}")

        return None

    def _classify_thermal_state(self, temp_celsius: float) -> ThermalState:
        """Classify thermal state from temperature"""
        if temp_celsius < 70:
            return ThermalState.OPTIMAL
        elif temp_celsius < 80:
            return ThermalState.WARM
        elif temp_celsius < 85:
            return ThermalState.HOT
        elif temp_celsius < 95:
            return ThermalState.THROTTLING
        else:
            return ThermalState.CRITICAL

    def _classify_power_state(
        self, power_watts: List[float], power_limits: List[float]
    ) -> PowerState:
        """Classify power state"""
        if not power_watts or not power_limits:
            return PowerState.UNLIMITED

        # Check if any GPU is at/near power limit
        for power, limit in zip(power_watts, power_limits):
            if power >= limit * 0.95:  # Within 5% of limit
                return PowerState.THROTTLED

        # Check power level
        avg_utilization = sum(p / l for p, l in zip(power_watts, power_limits)) / len(
            power_watts
        )

        if avg_utilization > 0.8:
            return PowerState.UNLIMITED
        elif avg_utilization > 0.5:
            return PowerState.BALANCED
        else:
            return PowerState.POWER_SAVER

    def _check_battery_status(self) -> bool:
        """Check if running on battery"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return not battery.power_plugged
        except:
            pass
        return False

    def is_throttling(self) -> bool:
        """Check if currently throttling (CACHED - no overhead)"""
        if self.last_snapshot:
            return self.last_snapshot.is_throttling
        return False

    def get_thermal_state(self) -> ThermalState:
        """Get current thermal state (CACHED)"""
        if self.last_snapshot:
            return self.last_snapshot.thermal_state
        return ThermalState.OPTIMAL

    def should_reduce_load(self) -> bool:
        """
        Check if load should be reduced due to thermal/power issues

        LOW OVERHEAD: Uses cached snapshot
        """
        if not self.last_snapshot:
            return False

        # Reduce load if:
        # 1. Throttling for 2+ consecutive checks (60s+)
        # 2. Critical temperature
        # 3. On battery with high power draw

        if self.throttle_count >= 2:
            return True

        if self.last_snapshot.thermal_state == ThermalState.CRITICAL:
            return True

        if (
            self.last_snapshot.on_battery
            and self.last_snapshot.power_state != PowerState.POWER_SAVER
        ):
            return True

        return False

    def get_recommended_batch_scale(self) -> float:
        """
        Get recommended batch size scaling factor (0.5-1.0)

        Returns:
            1.0 = no change
            0.75 = reduce by 25%
            0.5 = reduce by 50%
        """
        if not self.last_snapshot:
            return 1.0

        state = self.last_snapshot.thermal_state

        if state == ThermalState.CRITICAL:
            return 0.5  # Reduce by 50%
        elif state == ThermalState.THROTTLING:
            return 0.7  # Reduce by 30%
        elif state == ThermalState.HOT:
            return 0.85  # Reduce by 15%
        elif self.last_snapshot.on_battery:
            return 0.8  # Battery: reduce by 20%
        else:
            return 1.0  # No reduction

    def get_power_efficiency_mode(self) -> bool:
        """Check if power efficiency mode should be enabled"""
        if not self.last_snapshot:
            return False

        return (
            self.last_snapshot.on_battery
            or self.last_snapshot.power_state == PowerState.POWER_SAVER
        )

    def get_stats_summary(self) -> Dict:
        """Get thermal/power statistics summary"""
        if not self.last_snapshot:
            return {"available": False}

        snapshot = self.last_snapshot

        return {
            "available": True,
            "gpu_temps": snapshot.gpu_temps,
            "gpu_max_temp": max(snapshot.gpu_temps) if snapshot.gpu_temps else None,
            "cpu_temp": snapshot.cpu_temp,
            "thermal_state": snapshot.thermal_state.value,
            "power_state": snapshot.power_state.value,
            "is_throttling": snapshot.is_throttling,
            "throttle_count": self.throttle_count,
            "on_battery": snapshot.on_battery,
            "recommended_batch_scale": self.get_recommended_batch_scale(),
            "power_efficiency_mode": self.get_power_efficiency_mode(),
        }

    def get_thermal_history(self, minutes: int = 10) -> List[ThermalSnapshot]:
        """Get thermal history for last N minutes"""
        # Each snapshot is ~30s apart
        num_snapshots = min(
            len(self.history), (minutes * 60) // int(self.check_interval)
        )
        return list(self.history)[-num_snapshots:]


class ThermalOptimizer:
    """
    Automatically adjusts training parameters based on thermal state

    LOW OVERHEAD: Only checks every 30s via ThermalMonitor
    """

    def __init__(self, thermal_monitor: ThermalMonitor, initial_batch_size: int):
        self.monitor = thermal_monitor
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.last_adjustment_time = 0
        self.adjustment_cooldown = 60.0  # Wait 60s between adjustments

    def maybe_adjust_batch_size(self) -> Tuple[int, bool]:
        """
        Check if batch size should be adjusted

        Returns:
            (new_batch_size, changed)
        """
        current_time = time.time()

        # Cooldown to avoid thrashing
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return self.current_batch_size, False

        # Get recommendation (cached, no overhead)
        scale = self.monitor.get_recommended_batch_scale()

        new_batch_size = int(self.initial_batch_size * scale)

        # Only adjust if change is significant (>10%)
        if (
            abs(new_batch_size - self.current_batch_size)
            > self.current_batch_size * 0.1
        ):
            self.current_batch_size = new_batch_size
            self.last_adjustment_time = current_time
            logger.info(f"♨️  Thermal adjustment: batch size {self.current_batch_size}")
            return self.current_batch_size, True

        return self.current_batch_size, False


# Convenience functions
def create_thermal_monitor(check_interval: float = 30.0) -> ThermalMonitor:
    """Create thermal monitor with specified check interval"""
    return ThermalMonitor(check_interval=check_interval)


# Export
__all__ = [
    "ThermalState",
    "PowerState",
    "ThermalSnapshot",
    "ThermalMonitor",
    "ThermalOptimizer",
    "create_thermal_monitor",
]
