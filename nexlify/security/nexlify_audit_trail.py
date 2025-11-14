#!/usr/bin/env python3
"""
Nexlify Audit Trail Module
Comprehensive logging and auditing of all system activities
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import hashlib

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class AuditEvent:
    """Represents a single audit event"""

    def __init__(
        self,
        event_type: str,
        severity: str,
        user: str,
        details: Dict[str, Any],
        ip_address: str = None,
    ):
        self.timestamp = datetime.now()
        self.event_type = event_type
        self.severity = severity
        self.user = user
        self.details = details
        self.ip_address = ip_address or "127.0.0.1"
        self.event_id = self._generate_event_id()

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        data = f"{self.timestamp.isoformat()}{self.event_type}{self.user}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity,
            "user": self.user,
            "ip_address": self.ip_address,
            "details": self.details,
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class AuditManager:
    """
    Manages audit trail for all system activities
    """

    def __init__(self, audit_dir: Path = None):
        self.audit_dir = audit_dir or Path("logs/audit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Current audit file (rotated daily)
        self.current_audit_file = self._get_current_audit_file()

        # In-memory cache for recent events
        self.recent_events: List[AuditEvent] = []
        self.max_recent_events = 1000

        logger.info("📋 Audit Manager initialized")

    def _get_current_audit_file(self) -> Path:
        """Get the current audit file (one per day)"""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.audit_dir / f"audit_{date_str}.jsonl"

    @handle_errors("Audit Logging", reraise=False)
    def log_event(self, event: AuditEvent):
        """Log an audit event"""
        try:
            # Add to recent events
            self.recent_events.append(event)
            if len(self.recent_events) > self.max_recent_events:
                self.recent_events.pop(0)

            # Check if we need to rotate file
            current_file = self._get_current_audit_file()
            if current_file != self.current_audit_file:
                self.current_audit_file = current_file

            # Write to audit file (JSONL format - one JSON per line)
            with open(self.current_audit_file, "a") as f:
                f.write(event.to_json() + "\n")

            # Log to application logger based on severity
            log_message = f"[AUDIT] {event.event_type} by {event.user}: {event.details}"
            if event.severity == "critical":
                logger.critical(log_message)
            elif event.severity == "error":
                logger.error(log_message)
            elif event.severity == "warning":
                logger.warning(log_message)
            else:
                logger.info(log_message)

        except Exception as e:
            error_handler.log_error(e, "Failed to log audit event", severity="error")

    # Convenience methods for common audit events

    async def audit_login(self, username: str, ip_address: str, success: bool):
        """Audit a login attempt"""
        event = AuditEvent(
            event_type="login_attempt",
            severity="info" if success else "warning",
            user=username,
            details={"success": success, "method": "password"},
            ip_address=ip_address,
        )
        self.log_event(event)

    async def audit_logout(self, username: str, ip_address: str):
        """Audit a logout"""
        event = AuditEvent(
            event_type="logout",
            severity="info",
            user=username,
            details={"voluntary": True},
            ip_address=ip_address,
        )
        self.log_event(event)

    async def audit_trade(
        self,
        username: str,
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        order_type: str,
        success: bool,
    ):
        """Audit a trade execution"""
        event = AuditEvent(
            event_type="trade_execution",
            severity="info" if success else "error",
            user=username,
            details={
                "exchange": exchange,
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "order_type": order_type,
                "success": success,
            },
        )
        self.log_event(event)

    async def audit_withdrawal(
        self, username: str, amount: float, address: str, success: bool
    ):
        """Audit a withdrawal"""
        event = AuditEvent(
            event_type="withdrawal",
            severity="warning" if success else "error",
            user=username,
            details={
                "amount_usd": amount,
                "address": address[:8] + "..." + address[-8:],  # Partial for privacy
                "success": success,
            },
        )
        self.log_event(event)

    async def audit_config_change(
        self, username: str, config_section: str, old_value: Any, new_value: Any
    ):
        """Audit a configuration change"""
        event = AuditEvent(
            event_type="config_change",
            severity="warning",
            user=username,
            details={
                "section": config_section,
                "old_value": str(old_value)[:100],  # Truncate for safety
                "new_value": str(new_value)[:100],
            },
        )
        self.log_event(event)

    async def audit_api_key_change(self, username: str, exchange: str, action: str):
        """Audit API key changes"""
        event = AuditEvent(
            event_type="api_key_change",
            severity="warning",
            user=username,
            details={
                "exchange": exchange,
                "action": action,  # 'added', 'updated', 'removed'
            },
        )
        self.log_event(event)

    async def audit_security_event(
        self, username: str, event_type: str, details: Dict[str, Any]
    ):
        """Audit security-related events"""
        event = AuditEvent(
            event_type=f"security_{event_type}",
            severity="warning",
            user=username,
            details=details,
        )
        self.log_event(event)

    async def audit_system_event(
        self, event_type: str, severity: str, details: Dict[str, Any]
    ):
        """Audit system-level events"""
        event = AuditEvent(
            event_type=f"system_{event_type}",
            severity=severity,
            user="system",
            details=details,
        )
        self.log_event(event)

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        user: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get recent audit events with optional filtering

        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            severity: Filter by severity
            user: Filter by user

        Returns:
            List of event dictionaries
        """
        filtered_events = self.recent_events

        # Apply filters
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        if user:
            filtered_events = [e for e in filtered_events if e.user == user]

        # Sort by timestamp descending and limit
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        filtered_events = filtered_events[:limit]

        return [e.to_dict() for e in filtered_events]

    def get_events_from_file(
        self, date: Optional[datetime] = None, limit: int = 1000
    ) -> List[Dict]:
        """
        Load audit events from file

        Args:
            date: Date to load events from (defaults to today)
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y%m%d")
        audit_file = self.audit_dir / f"audit_{date_str}.jsonl"

        events = []
        if audit_file.exists():
            try:
                with open(audit_file, "r") as f:
                    for line in f:
                        if line.strip():
                            event = json.loads(line)
                            events.append(event)
                            if len(events) >= limit:
                                break
            except Exception as e:
                logger.error(f"Error reading audit file: {e}")

        return events

    def search_events(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Search audit events by query string

        Args:
            query: Search query
            start_date: Start date for search
            end_date: End date for search

        Returns:
            List of matching event dictionaries
        """
        matching_events = []

        # Search recent events first
        for event in self.recent_events:
            if start_date and event.timestamp < start_date:
                continue
            if end_date and event.timestamp > end_date:
                continue

            # Search in JSON representation
            if query.lower() in event.to_json().lower():
                matching_events.append(event.to_dict())

        return matching_events

    def generate_audit_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate an audit report for a date range

        Returns:
            Report dictionary with statistics and events
        """
        report = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "events_by_user": {},
            "critical_events": [],
        }

        # Iterate through date range
        current_date = start_date
        while current_date <= end_date:
            events = self.get_events_from_file(current_date)

            for event_dict in events:
                report["total_events"] += 1

                # Count by type
                event_type = event_dict["event_type"]
                report["events_by_type"][event_type] = (
                    report["events_by_type"].get(event_type, 0) + 1
                )

                # Count by severity
                severity = event_dict["severity"]
                report["events_by_severity"][severity] = (
                    report["events_by_severity"].get(severity, 0) + 1
                )

                # Count by user
                user = event_dict["user"]
                report["events_by_user"][user] = (
                    report["events_by_user"].get(user, 0) + 1
                )

                # Collect critical events
                if severity == "critical":
                    report["critical_events"].append(event_dict)

            current_date += datetime.timedelta(days=1)

        return report

    def cleanup_old_audits(self, days: int = 90):
        """Remove audit files older than specified days"""
        try:
            cutoff_date = datetime.now() - datetime.timedelta(days=days)

            for audit_file in self.audit_dir.glob("audit_*.jsonl"):
                try:
                    # Extract date from filename
                    date_str = audit_file.stem.split("_")[1]
                    file_date = datetime.strptime(date_str, "%Y%m%d")

                    if file_date < cutoff_date:
                        audit_file.unlink()
                        logger.info(f"Removed old audit file: {audit_file}")
                except Exception as e:
                    logger.warning(f"Error processing audit file {audit_file}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning up old audits: {e}")
