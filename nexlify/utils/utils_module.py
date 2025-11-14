#!/usr/bin/env python3
"""
Nexlify - Shared Utilities
Common functions and utilities used across the application
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp

from nexlify.utils.error_handler import get_error_handler, handle_errors

# Initialize components
error_handler = get_error_handler()
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = PROJECT_ROOT / "config" / "neural_config.json"
DEFAULT_TIMEOUT = 30


class CryptoUtils:
    """Cryptocurrency-related utilities"""

    @staticmethod
    def validate_btc_address(address: str) -> bool:
        """Validate Bitcoin address format"""
        # Basic validation - real implementation would be more complex
        if not address:
            return False

        # Check length
        if len(address) < 26 or len(address) > 35:
            return False

        # Check format (simplified)
        patterns = [
            r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$",  # Legacy
            r"^bc1[a-z0-9]{39,59}$",  # Bech32
            r"^[23][a-km-zA-HJ-NP-Z1-9]{25,34}$",  # P2SH
        ]

        return any(re.match(pattern, address) for pattern in patterns)

    @staticmethod
    def normalize_symbol(symbol: str, exchange: str = None) -> str:
        """Normalize trading pair symbol across exchanges"""
        # Remove common separators
        normalized = symbol.upper().replace("-", "").replace("_", "").replace("/", "")

        # Exchange-specific normalization
        if exchange == "binance":
            # Binance uses direct concatenation (BTCUSDT)
            return normalized
        elif exchange == "kraken":
            # Kraken may use XBT for BTC
            return normalized.replace("BTC", "XBT")

        # Default format with slash
        if len(normalized) >= 6:
            # Try to split common pairs
            for quote in ["USDT", "USDC", "USD", "EUR", "BTC", "ETH"]:
                if normalized.endswith(quote):
                    base = normalized[: -len(quote)]
                    return f"{base}/{quote}"

        return symbol

    @staticmethod
    def calculate_position_size(
        balance: float, risk_percent: float, stop_loss_percent: float
    ) -> float:
        """Calculate position size based on risk management"""
        if stop_loss_percent <= 0:
            return 0

        risk_amount = balance * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_percent / 100)

        return min(position_size, balance)  # Never risk more than balance


class NetworkUtils:
    """Network and API utilities"""

    @staticmethod
    async def rate_limited_request(
        session: aiohttp.ClientSession,
        url: str,
        method: str = "GET",
        rate_limit: float = 0.1,
        **kwargs,
    ) -> Dict:
        """Make rate-limited HTTP request"""
        await asyncio.sleep(rate_limit)

        try:
            async with session.request(
                method, url, timeout=DEFAULT_TIMEOUT, **kwargs
            ) as response:
                response.raise_for_status()
                return await response.json()
        except asyncio.TimeoutError:
            raise Exception(f"Request timeout: {url}")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {e}")

    @staticmethod
    def generate_signature(secret: str, message: str) -> str:
        """Generate HMAC signature for API requests"""
        return hmac.new(
            secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()


class TimeUtils:
    """Time and scheduling utilities"""

    @staticmethod
    def get_next_schedule_time(schedule: str, from_time: datetime = None) -> datetime:
        """Get next scheduled time based on schedule type"""
        if from_time is None:
            from_time = datetime.now()

        if schedule == "hourly":
            return from_time.replace(minute=0, second=0, microsecond=0) + timedelta(
                hours=1
            )
        elif schedule == "daily":
            return from_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        elif schedule == "weekly":
            days_ahead = 7 - from_time.weekday()  # Monday is 0
            return from_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_ahead)
        elif schedule == "monthly":
            if from_time.month == 12:
                return from_time.replace(
                    year=from_time.year + 1,
                    month=1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            else:
                return from_time.replace(
                    month=from_time.month + 1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
        else:
            return from_time + timedelta(hours=24)  # Default to daily

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"


class ValidationUtils:
    """Input validation utilities"""

    @staticmethod
    def validate_api_credentials(api_key: str, secret: str) -> tuple[bool, str]:
        """Validate API credentials format"""
        if not api_key or not secret:
            return False, "API key and secret are required"

        if len(api_key) < 16:
            return False, "API key seems too short"

        if len(secret) < 16:
            return False, "API secret seems too short"

        # Check for placeholder values
        placeholders = ["YOUR_API_KEY", "YOUR_SECRET", "xxx", "..."]
        if any(placeholder in api_key.upper() for placeholder in placeholders):
            return False, "API key contains placeholder text"

        return True, "Valid"

    @staticmethod
    def validate_port(port: Union[str, int]) -> tuple[bool, str]:
        """Validate network port"""
        try:
            port_num = int(port)
            if port_num < 1 or port_num > 65535:
                return False, "Port must be between 1 and 65535"

            # Check for commonly blocked ports
            blocked_ports = [21, 22, 23, 25, 110, 143]
            if port_num in blocked_ports:
                return False, f"Port {port_num} is commonly blocked"

            return True, "Valid"
        except ValueError:
            return False, "Port must be a number"

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format"""
        if not email:
            return True  # Empty is valid (optional)

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))


class FileUtils:
    """File system utilities"""

    @staticmethod
    @handle_errors("Creating Backup", reraise=False)
    def create_backup(file_path: Path, backup_dir: Path = None) -> Optional[Path]:
        """Create backup of file"""
        if not file_path.exists():
            return None

        if backup_dir is None:
            backup_dir = PROJECT_ROOT / "backups"

        backup_dir.mkdir(exist_ok=True)

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name

        # Copy file
        import shutil

        shutil.copy2(file_path, backup_path)

        # Cleanup old backups (keep last 7 days)
        FileUtils.cleanup_old_files(backup_dir, days=7)

        return backup_path

    @staticmethod
    def cleanup_old_files(directory: Path, days: int = 7):
        """Remove files older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)

        for file_path in directory.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        logger.debug(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old file {file_path}: {e}")

    @staticmethod
    def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
        """Safely load JSON file with error handling"""
        try:
            path = Path(file_path)
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        except Exception as e:
            error_handler.log_error(
                e, f"Failed to load JSON from {file_path}", severity="warning"
            )

        return default if default is not None else {}

    @staticmethod
    def safe_json_save(
        data: Any, file_path: Union[str, Path], create_backup: bool = True
    ) -> bool:
        """Safely save JSON file with error handling"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if requested
            if create_backup and path.exists():
                FileUtils.create_backup(path)

            # Write to temporary file first
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_path.replace(path)
            return True

        except Exception as e:
            error_handler.log_error(
                e, f"Failed to save JSON to {file_path}", severity="error"
            )
            return False


class AsyncUtils:
    """Async programming utilities"""

    @staticmethod
    def run_async(coro):
        """Run async coroutine in sync context"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @staticmethod
    def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Decorator for async retry logic"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            raise

                raise last_exception

            return wrapper

        return decorator


class MathUtils:
    """Mathematical utilities"""

    @staticmethod
    def safe_divide(
        numerator: float, denominator: float, default: float = 0.0
    ) -> float:
        """Safe division with zero check"""
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 0.0 if new_value == 0 else float("inf")
        return ((new_value - old_value) / old_value) * 100

    @staticmethod
    def moving_average(values: List[float], window: int) -> List[float]:
        """Calculate simple moving average"""
        if len(values) < window:
            return values

        result = []
        for i in range(len(values) - window + 1):
            avg = sum(values[i : i + window]) / window
            result.append(avg)

        return result


# Singleton instances for commonly used utilities
crypto_utils = CryptoUtils()
network_utils = NetworkUtils()
time_utils = TimeUtils()
validation_utils = ValidationUtils()
file_utils = FileUtils()
async_utils = AsyncUtils()
math_utils = MathUtils()

# Export main functions for convenience
validate_btc_address = crypto_utils.validate_btc_address
normalize_symbol = crypto_utils.normalize_symbol
safe_json_load = file_utils.safe_json_load
safe_json_save = file_utils.safe_json_save
create_backup = file_utils.create_backup
