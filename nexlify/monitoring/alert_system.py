"""
Alert System for Training Monitoring
Email and Slack notifications for critical events
"""

import logging
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Multi-channel alert system for training events

    Supports:
    - Email notifications (SMTP)
    - Slack webhooks
    - Configurable alert levels and thresholds
    - Alert throttling to prevent spam

    Example:
        >>> alert_system = AlertSystem(config)
        >>> alert_system.send_alert(
        ...     level='warning',
        ...     title='Slow Learning',
        ...     message='No improvement in 100 episodes'
        ... )
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert system

        Config structure:
        {
            'enable_alerts': True,
            'email': {
                'enabled': True,
                'smtp_host': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your@email.com',
                'password': 'your_password',
                'from_addr': 'nexlify@bot.com',
                'to_addrs': ['recipient@email.com']
            },
            'slack': {
                'enabled': True,
                'webhook_url': 'https://hooks.slack.com/...'
            },
            'thresholds': {
                'no_improvement_episodes': 100,
                'critical_loss_threshold': 1000.0,
                'min_profit_warning': -500.0
            }
        }
        """
        self.config = config
        self.enabled = config.get('enable_alerts', True)

        # Email config
        self.email_config = config.get('email', {})
        self.email_enabled = self.email_config.get('enabled', False)

        # Slack config
        self.slack_config = config.get('slack', {})
        self.slack_enabled = self.slack_config.get('enabled', False)

        # Thresholds
        self.thresholds = config.get('thresholds', {})

        # Alert history for throttling
        self.alert_history: List[Dict[str, Any]] = []
        self.throttle_window = 300  # 5 minutes

        # Alert log
        self.log_file = Path("training_logs/alerts.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"AlertSystem initialized (email={self.email_enabled}, "
            f"slack={self.slack_enabled})"
        )

    def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool:
        """
        Send alert via configured channels

        Args:
            level: Alert level ('info', 'warning', 'critical')
            title: Alert title
            message: Alert message
            data: Additional data to include
            force: Force send even if throttled

        Returns:
            True if alert sent successfully
        """
        if not self.enabled:
            return False

        # Check throttling
        if not force and self._is_throttled(title):
            logger.debug(f"Alert throttled: {title}")
            return False

        # Create alert record
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'title': title,
            'message': message,
            'data': data or {}
        }

        # Log alert
        self._log_alert(alert)

        # Add to history
        self.alert_history.append(alert)

        # Send via channels
        success = False

        if self.email_enabled:
            success |= self._send_email(alert)

        if self.slack_enabled:
            success |= self._send_slack(alert)

        return success

    def check_training_health(
        self,
        latest_episode: Dict[str, Any],
        recent_history: List[Dict[str, Any]]
    ) -> None:
        """
        Check training health and send alerts if needed

        Args:
            latest_episode: Latest episode metrics
            recent_history: Recent episode history
        """
        if not self.enabled:
            return

        # Check for no improvement
        self._check_no_improvement(recent_history)

        # Check for critical loss
        self._check_critical_loss(latest_episode)

        # Check for low profit
        self._check_low_profit(latest_episode)

        # Check for NaN values
        self._check_nan_values(latest_episode)

    def send_training_complete(
        self,
        total_episodes: int,
        best_profit: float,
        best_sharpe: float,
        training_time: float
    ) -> None:
        """Send training complete notification"""
        message = (
            f"Training completed!\n\n"
            f"Total Episodes: {total_episodes}\n"
            f"Best Profit: ${best_profit:.2f}\n"
            f"Best Sharpe: {best_sharpe:.2f}\n"
            f"Training Time: {training_time:.1f}s"
        )

        self.send_alert(
            level='info',
            title='Training Complete',
            message=message,
            force=True
        )

    def send_new_best_model(
        self,
        episode: int,
        profit: float,
        sharpe: float
    ) -> None:
        """Send new best model notification"""
        message = (
            f"New best model found!\n\n"
            f"Episode: {episode}\n"
            f"Profit: ${profit:.2f}\n"
            f"Sharpe: {sharpe:.2f}"
        )

        self.send_alert(
            level='info',
            title='New Best Model',
            message=message
        )

    def _check_no_improvement(self, history: List[Dict[str, Any]]) -> None:
        """Check for no improvement over threshold episodes"""
        threshold = self.thresholds.get('no_improvement_episodes', 100)

        if len(history) < threshold:
            return

        recent = history[-threshold:]
        profits = [e.get('profit', 0) for e in recent]

        if profits and max(profits) <= 0:
            self.send_alert(
                level='warning',
                title='No Improvement',
                message=f'No positive profit in last {threshold} episodes'
            )

    def _check_critical_loss(self, episode: Dict[str, Any]) -> None:
        """Check for critically high loss"""
        loss = episode.get('loss')
        threshold = self.thresholds.get('critical_loss_threshold', 1000.0)

        if loss and loss > threshold:
            self.send_alert(
                level='critical',
                title='Critical Loss',
                message=f'Loss exceeds threshold: {loss:.2f} > {threshold}',
                data={'loss': loss}
            )

    def _check_low_profit(self, episode: Dict[str, Any]) -> None:
        """Check for critically low profit"""
        profit = episode.get('profit', 0)
        threshold = self.thresholds.get('min_profit_warning', -500.0)

        if profit < threshold:
            self.send_alert(
                level='warning',
                title='Low Profit Warning',
                message=f'Profit below threshold: ${profit:.2f} < ${threshold}',
                data={'profit': profit}
            )

    def _check_nan_values(self, episode: Dict[str, Any]) -> None:
        """Check for NaN values in metrics"""
        import math

        for key, value in episode.items():
            if isinstance(value, (int, float)) and math.isnan(value):
                self.send_alert(
                    level='critical',
                    title='NaN Value Detected',
                    message=f'NaN detected in {key}',
                    data={'key': key, 'episode': episode.get('episode')}
                )
                break

    def _is_throttled(self, title: str) -> bool:
        """Check if alert is throttled"""
        now = datetime.now()
        recent = [
            a for a in self.alert_history
            if a['title'] == title and
            (now - datetime.fromisoformat(a['timestamp'])).seconds < self.throttle_window
        ]

        return len(recent) > 0

    def _send_email(self, alert: Dict[str, Any]) -> bool:
        """Send alert via email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('from_addr', 'nexlify@bot.com')
            msg['To'] = ', '.join(self.email_config.get('to_addrs', []))
            msg['Subject'] = f"[Nexlify {alert['level'].upper()}] {alert['title']}"

            # Email body
            body = f"""
Nexlify Training Alert

Level: {alert['level'].upper()}
Time: {alert['timestamp']}

{alert['message']}

---
Additional Data:
{json.dumps(alert.get('data', {}), indent=2)}

---
This is an automated message from Nexlify Training Monitor.
            """

            msg.attach(MIMEText(body, 'plain'))

            # Connect and send
            with smtplib.SMTP(
                self.email_config['smtp_host'],
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['username'],
                    self.email_config['password']
                )
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert['title']}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _send_slack(self, alert: Dict[str, Any]) -> bool:
        """Send alert via Slack webhook"""
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available for Slack alerts")
            return False

        try:
            # Color coding
            colors = {
                'info': '#00ff9f',
                'warning': '#ffaa00',
                'critical': '#ff0080'
            }

            # Build Slack message
            payload = {
                'attachments': [{
                    'color': colors.get(alert['level'], '#cccccc'),
                    'title': alert['title'],
                    'text': alert['message'],
                    'fields': [
                        {
                            'title': 'Level',
                            'value': alert['level'].upper(),
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': alert['timestamp'],
                            'short': True
                        }
                    ],
                    'footer': 'Nexlify Training Monitor',
                    'ts': int(datetime.now().timestamp())
                }]
            }

            # Add data if present
            if alert.get('data'):
                payload['attachments'][0]['fields'].append({
                    'title': 'Additional Data',
                    'value': f"```{json.dumps(alert['data'], indent=2)}```",
                    'short': False
                })

            # Send webhook
            response = requests.post(
                self.slack_config['webhook_url'],
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert['title']}")
                return True
            else:
                logger.error(
                    f"Slack webhook failed: {response.status_code} "
                    f"{response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def _log_alert(self, alert: Dict[str, Any]) -> None:
        """Log alert to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

    def get_alert_history(
        self,
        level: Optional[str] = None,
        last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history

        Args:
            level: Filter by level (None for all)
            last_n: Return last N alerts (None for all)

        Returns:
            List of alerts
        """
        alerts = self.alert_history

        if level:
            alerts = [a for a in alerts if a['level'] == level]

        if last_n:
            alerts = alerts[-last_n:]

        return alerts

    def clear_history(self) -> None:
        """Clear alert history"""
        self.alert_history.clear()
        logger.info("Alert history cleared")


class AlertThresholds:
    """Predefined alert thresholds"""

    CONSERVATIVE = {
        'no_improvement_episodes': 50,
        'critical_loss_threshold': 500.0,
        'min_profit_warning': -200.0
    }

    MODERATE = {
        'no_improvement_episodes': 100,
        'critical_loss_threshold': 1000.0,
        'min_profit_warning': -500.0
    }

    AGGRESSIVE = {
        'no_improvement_episodes': 200,
        'critical_loss_threshold': 2000.0,
        'min_profit_warning': -1000.0
    }
