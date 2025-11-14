#!/usr/bin/env python3
"""
Nexlify - Error Handler
Centralized error logging and crash reporting system
"""

import logging
import traceback
import sys
import os
import json
from datetime import datetime
from pathlib import Path
import platform
import psutil
from typing import Optional, Dict, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp


class NightCityErrorHandler:
    """
    Centralized error handling with separate error logs and crash reports
    """

    def __init__(self, config_path: str = "config/neural_config.json"):
        self.config_path = Path(config_path)
        self.error_log_path = Path("logs/errors.log")
        self.crash_report_path = Path("logs/crash_reports")
        self.config = self._load_config()

        # Create directories
        self.error_log_path.parent.mkdir(exist_ok=True)
        self.crash_report_path.mkdir(exist_ok=True)

        # Setup error logger
        self.error_logger = self._setup_error_logger()

        # Track error counts for monitoring
        self.error_counts = {"warning": 0, "error": 0, "critical": 0, "fatal": 0}

        # Install exception hooks
        self._install_exception_hooks()

    def _load_config(self) -> Dict:
        """Load configuration for notifications"""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {}

    def _setup_error_logger(self) -> logging.Logger:
        """Setup dedicated error logger"""
        error_logger = logging.getLogger("NightCityErrors")
        error_logger.setLevel(logging.WARNING)

        # Don't propagate to root logger
        error_logger.propagate = False

        # File handler for errors only
        error_handler = logging.FileHandler(self.error_log_path, encoding="utf-8")
        error_handler.setLevel(logging.WARNING)

        # Detailed formatter for errors
        error_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]\n"
            "%(message)s\n"
            "=" * 70
        )
        error_handler.setFormatter(error_formatter)

        # Console handler for critical errors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter("🚨 [%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)

        error_logger.addHandler(error_handler)
        error_logger.addHandler(console_handler)

        return error_logger

    def _install_exception_hooks(self):
        """Install global exception handlers"""
        # Handle uncaught exceptions
        sys.excepthook = self._handle_exception

        # Handle threading exceptions
        import threading

        threading.excepthook = self._handle_thread_exception

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupt to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the fatal error
        self.log_fatal_error(exc_type, exc_value, exc_traceback)

    def _handle_thread_exception(self, args):
        """Handle exceptions in threads"""
        if args.exc_type == SystemExit:
            return

        self.log_fatal_error(
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
            thread_info=f"Thread: {args.thread.name}",
        )

    def log_error(self, error: Exception, context: str = "", severity: str = "error"):
        """Log an error with context"""
        self.error_counts[severity] += 1

        # Create error message
        error_msg = f"Context: {context}\n"
        error_msg += f"Error Type: {type(error).__name__}\n"
        error_msg += f"Error Message: {str(error)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"

        # Log based on severity
        if severity == "warning":
            self.error_logger.warning(error_msg)
        elif severity == "error":
            self.error_logger.error(error_msg)
        elif severity == "critical":
            self.error_logger.critical(error_msg)
            self._notify_critical_error(error_msg)
        elif severity == "fatal":
            self.log_fatal_error(type(error), error, error.__traceback__)

    def log_fatal_error(
        self, exc_type, exc_value, exc_traceback, thread_info: str = ""
    ):
        """Log fatal error and generate crash report"""
        self.error_counts["fatal"] += 1

        # Generate crash report
        crash_report = self._generate_crash_report(
            exc_type, exc_value, exc_traceback, thread_info
        )

        # Save crash report
        crash_file = (
            self.crash_report_path
            / f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(crash_file, "w") as f:
            json.dump(crash_report, f, indent=2)

        # Log to error log
        self.error_logger.critical(
            f"FATAL ERROR - Crash report saved to: {crash_file}\n"
            f"{crash_report['error']['type']}: {crash_report['error']['message']}"
        )

        # Send notifications
        self._notify_fatal_crash(crash_report)

        # Display to console
        print(f"\n{'='*70}")
        print(f"🔴 FATAL ERROR DETECTED - Nexlify Crashed")
        print(f"{'='*70}")
        print(f"Error: {crash_report['error']['type']}")
        print(f"Message: {crash_report['error']['message']}")
        print(f"Crash report saved to: {crash_file}")
        print(f"{'='*70}\n")

    def _generate_crash_report(
        self, exc_type, exc_value, exc_traceback, thread_info: str = ""
    ) -> Dict:
        """Generate detailed crash report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.7.7",
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "memory_percent": psutil.virtual_memory().percent,
            },
            "error": {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "thread_info": thread_info,
                "traceback": traceback.format_tb(exc_traceback),
            },
            "error_counts": self.error_counts.copy(),
            "config": self._get_safe_config(),
        }

        # Add recent errors from log
        report["recent_errors"] = self._get_recent_errors()

        return report

    def _get_safe_config(self) -> Dict:
        """Get configuration without sensitive data"""
        safe_config = {}

        if self.config:
            safe_config = {
                "exchanges_configured": list(self.config.get("exchanges", {}).keys()),
                "risk_level": self.config.get("risk_level", "unknown"),
                "auto_trade": self.config.get("auto_trade", False),
                "environment": {
                    "debug": self.config.get("environment", {}).get("debug", False),
                    "log_level": self.config.get("environment", {}).get(
                        "log_level", "INFO"
                    ),
                    "notifications_enabled": bool(
                        self.config.get("environment", {}).get("emergency_contact")
                        or self.config.get("environment", {}).get("telegram_bot_token")
                    ),
                },
            }

        return safe_config

    def _get_recent_errors(self, limit: int = 10) -> list:
        """Get recent errors from error log"""
        recent_errors = []

        if self.error_log_path.exists():
            with open(self.error_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse last few error entries
            current_error = []
            for line in reversed(lines[-200:]):  # Check last 200 lines
                if line.startswith("[") and current_error:
                    recent_errors.append("".join(reversed(current_error)))
                    current_error = [line]
                    if len(recent_errors) >= limit:
                        break
                else:
                    current_error.append(line)

        return recent_errors

    def _notify_critical_error(self, error_msg: str):
        """Send notification for critical errors"""
        env_config = self.config.get("environment", {})

        # Telegram notification
        if env_config.get("telegram_bot_token") and env_config.get("telegram_chat_id"):
            asyncio.create_task(
                self._send_telegram_notification(
                    f"⚠️ CRITICAL ERROR in Nexlify\n\n{error_msg[:500]}..."
                )
            )

        # Email notification (if configured)
        if env_config.get("emergency_contact"):
            self._send_email_notification("Critical Error - Nexlify", error_msg)

    def _notify_fatal_crash(self, crash_report: Dict):
        """Send notifications for fatal crashes"""
        env_config = self.config.get("environment", {})

        message = (
            f"🔴 FATAL CRASH - Nexlify\n\n"
            f"Time: {crash_report['timestamp']}\n"
            f"Error: {crash_report['error']['type']}\n"
            f"Message: {crash_report['error']['message']}\n"
            f"Platform: {crash_report['system']['platform']}\n"
            f"Memory Usage: {crash_report['system']['memory_percent']}%"
        )

        # Telegram
        if env_config.get("telegram_bot_token") and env_config.get("telegram_chat_id"):
            asyncio.create_task(self._send_telegram_notification(message))

        # Email
        if env_config.get("emergency_contact"):
            self._send_email_notification(
                "FATAL CRASH - Nexlify", json.dumps(crash_report, indent=2)
            )

    async def _send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        env_config = self.config.get("environment", {})
        token = env_config.get("telegram_bot_token")
        chat_id = env_config.get("telegram_chat_id")

        if not token or not chat_id:
            return

        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
                await session.post(url, data=data)
        except Exception as e:
            self.error_logger.error(f"Failed to send Telegram notification: {e}")

    def _send_email_notification(self, subject: str, body: str):
        """Send email notification (placeholder for future implementation)"""
        # This would require SMTP configuration
        # For now, just log that we would send an email
        self.error_logger.info(f"Email notification would be sent: {subject}")

    def get_error_summary(self) -> Dict:
        """Get summary of errors for display"""
        return {
            "counts": self.error_counts.copy(),
            "last_error_time": self._get_last_error_time(),
            "error_rate": self._calculate_error_rate(),
        }

    def _get_last_error_time(self) -> Optional[str]:
        """Get timestamp of last error"""
        if self.error_log_path.exists():
            stat = os.stat(self.error_log_path)
            return datetime.fromtimestamp(stat.st_mtime).isoformat()
        return None

    def _calculate_error_rate(self) -> float:
        """Calculate errors per hour"""
        # Simplified calculation - in production would track over time
        total_errors = sum(self.error_counts.values())
        return total_errors  # Would divide by hours running


# Global error handler instance
error_handler = None


def get_error_handler() -> NightCityErrorHandler:
    """Get or create global error handler"""
    global error_handler
    if error_handler is None:
        error_handler = NightCityErrorHandler()
    return error_handler


# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in a specific context"""

    def __init__(self, context: str, severity: str = "error", reraise: bool = True):
        self.context = context
        self.severity = severity
        self.reraise = reraise
        self.handler = get_error_handler()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            self.handler.log_error(exc_value, self.context, self.severity)
            return not self.reraise
        return True


# Decorator for error handling
def handle_errors(context: str = "", severity: str = "error", reraise: bool = True):
    """Decorator to handle errors in functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            ctx = context or f"{func.__module__}.{func.__name__}"
            with ErrorContext(ctx, severity, reraise):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Import for asyncio if available
try:
    import asyncio
except ImportError:
    asyncio = None
