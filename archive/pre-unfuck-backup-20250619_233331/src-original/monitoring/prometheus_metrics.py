#!/usr/bin/env python3
"""
src/monitoring/prometheus_metrics.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXLIFY PROMETHEUS METRICS - CYBERPUNK MONITORING SYSTEM v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Real-time monitoring with Prometheus, Grafana, and custom alerts.
Tracks everything from latency to profits with cyberpunk style dashboards.
"""

import os
import asyncio
import time
import json
import psutil
import GPUtil
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import structlog
import aiohttp
from pathlib import Path

# Prometheus
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest,
    start_http_server, push_to_gateway,
    CONTENT_TYPE_LATEST
)
from prometheus_client.exposition import basic_auth_handler

# Alerting
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx

# Import our components
from ..utils.config_loader import get_config_loader, CyberColors

# Initialize logger
logger = structlog.get_logger("NEXLIFY.MONITORING.PROMETHEUS")

# Custom registry for better control
REGISTRY = CollectorRegistry()

# Core Metrics
SYSTEM_INFO = Info(
    'nexlify_system_info',
    'System information',
    registry=REGISTRY
)

UPTIME = Gauge(
    'nexlify_uptime_seconds',
    'System uptime in seconds',
    registry=REGISTRY
)

# Trading Metrics
TRADES_TOTAL = Counter(
    'nexlify_trades_total',
    'Total number of trades executed',
    ['exchange', 'symbol', 'side', 'status'],
    registry=REGISTRY
)

TRADE_VOLUME = Counter(
    'nexlify_trade_volume_usd',
    'Total trade volume in USD',
    ['exchange', 'symbol', 'side'],
    registry=REGISTRY
)

TRADE_PROFIT = Gauge(
    'nexlify_trade_profit_usd',
    'Current trading profit in USD',
    ['strategy', 'symbol'],
    registry=REGISTRY
)

TRADE_LATENCY = Histogram(
    'nexlify_trade_latency_seconds',
    'Trade execution latency',
    ['exchange', 'operation'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY
)

# Strategy Metrics
STRATEGY_SIGNALS = Counter(
    'nexlify_strategy_signals_total',
    'Total strategy signals generated',
    ['strategy', 'signal_type', 'symbol'],
    registry=REGISTRY
)

STRATEGY_ACCURACY = Gauge(
    'nexlify_strategy_accuracy_percent',
    'Strategy prediction accuracy',
    ['strategy'],
    registry=REGISTRY
)

STRATEGY_SHARPE_RATIO = Gauge(
    'nexlify_strategy_sharpe_ratio',
    'Strategy Sharpe ratio',
    ['strategy'],
    registry=REGISTRY
)

# Exchange Metrics
EXCHANGE_STATUS = Gauge(
    'nexlify_exchange_status',
    'Exchange connection status (1=connected, 0=disconnected)',
    ['exchange'],
    registry=REGISTRY
)

EXCHANGE_BALANCE = Gauge(
    'nexlify_exchange_balance',
    'Exchange balance by asset',
    ['exchange', 'asset'],
    registry=REGISTRY
)

ORDERBOOK_SPREAD = Gauge(
    'nexlify_orderbook_spread',
    'Current bid-ask spread',
    ['exchange', 'symbol'],
    registry=REGISTRY
)

ORDERBOOK_DEPTH = Gauge(
    'nexlify_orderbook_depth_usd',
    'Orderbook depth in USD',
    ['exchange', 'symbol', 'side'],
    registry=REGISTRY
)

# ML Model Metrics
ML_PREDICTIONS = Counter(
    'nexlify_ml_predictions_total',
    'Total ML predictions made',
    ['model', 'symbol'],
    registry=REGISTRY
)

ML_PREDICTION_LATENCY = Histogram(
    'nexlify_ml_prediction_latency_seconds',
    'ML model prediction latency',
    ['model'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY
)

ML_MODEL_ACCURACY = Gauge(
    'nexlify_ml_model_accuracy_percent',
    'ML model accuracy percentage',
    ['model', 'timeframe'],
    registry=REGISTRY
)

# System Metrics
CPU_USAGE = Gauge(
    'nexlify_cpu_usage_percent',
    'CPU usage percentage',
    ['core'],
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'nexlify_memory_usage_bytes',
    'Memory usage in bytes',
    ['type'],
    registry=REGISTRY
)

GPU_USAGE = Gauge(
    'nexlify_gpu_usage_percent',
    'GPU usage percentage',
    ['gpu_id', 'metric'],
    registry=REGISTRY
)

DISK_USAGE = Gauge(
    'nexlify_disk_usage_bytes',
    'Disk usage in bytes',
    ['mount_point', 'type'],
    registry=REGISTRY
)

# Network Metrics
NETWORK_IO = Counter(
    'nexlify_network_io_bytes',
    'Network I/O in bytes',
    ['interface', 'direction'],
    registry=REGISTRY
)

WEBSOCKET_CONNECTIONS = Gauge(
    'nexlify_websocket_connections',
    'Active WebSocket connections',
    ['exchange'],
    registry=REGISTRY
)

API_REQUESTS = Counter(
    'nexlify_api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status'],
    registry=REGISTRY
)

API_LATENCY = Histogram(
    'nexlify_api_latency_seconds',
    'API request latency',
    ['endpoint', 'method'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY
)

# Error Metrics
ERRORS_TOTAL = Counter(
    'nexlify_errors_total',
    'Total errors by type',
    ['component', 'error_type'],
    registry=REGISTRY
)

CRITICAL_ALERTS = Counter(
    'nexlify_critical_alerts_total',
    'Critical alerts triggered',
    ['alert_type'],
    registry=REGISTRY
)

# Custom Business Metrics
DAILY_PNL = Gauge(
    'nexlify_daily_pnl_usd',
    'Daily profit and loss in USD',
    registry=REGISTRY
)

WIN_RATE = Gauge(
    'nexlify_win_rate_percent',
    'Overall win rate percentage',
    ['timeframe'],
    registry=REGISTRY
)

MAX_DRAWDOWN = Gauge(
    'nexlify_max_drawdown_percent',
    'Maximum drawdown percentage',
    ['timeframe'],
    registry=REGISTRY
)

RISK_SCORE = Gauge(
    'nexlify_risk_score',
    'Current risk score (0-100)',
    registry=REGISTRY
)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    condition: str  # e.g., "> 90", "< 0.5", "== 0"
    threshold: float
    duration: int  # seconds
    severity: str  # info, warning, critical
    message_template: str
    cooldown: int = 300  # 5 minutes default
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot"""
    timestamp: datetime
    name: str
    value: float
    labels: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class NexlifyMetricsCollector:
    """
    📊 NEXLIFY Prometheus Metrics Collector
    
    Features:
    - Real-time system and trading metrics
    - Custom business KPI tracking
    - Alerting with multiple channels
    - Grafana-ready dashboards
    - Historical data retention
    - Anomaly detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config_loader().get('monitoring', {})
        
        # Metrics server config
        self.metrics_port = self.config.get('prometheus_port', 9090)
        self.pushgateway_url = self.config.get('pushgateway_url')
        self.auth_enabled = self.config.get('auth_enabled', False)
        self.auth_username = self.config.get('auth_username', 'nexlify')
        self.auth_password = self.config.get('auth_password', 'metrics')
        
        # System info
        self.start_time = time.time()
        self.hostname = os.uname().nodename
        
        # Alert configuration
        self.alert_rules: List[AlertRule] = []
        self.alert_channels = {
            'email': self.config.get('alerts.email', {}),
            'discord': self.config.get('alerts.discord', {}),
            'telegram': self.config.get('alerts.telegram', {}),
            'webhook': self.config.get('alerts.webhook', {})
        }
        
        # Metric history for anomaly detection
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_threshold = self.config.get('anomaly_threshold', 3.0)  # Standard deviations
        
        # Performance tracking
        self.collection_times = deque(maxlen=100)
        
        logger.info(
            f"{CyberColors.NEON_CYAN}📊 Initializing Prometheus Metrics Collector...{CyberColors.RESET}"
        )
    
    async def initialize(self):
        """Initialize metrics collector and start server"""
        logger.info(f"{CyberColors.NEON_CYAN}Starting metrics collection...{CyberColors.RESET}")
        
        try:
            # Set system info
            SYSTEM_INFO.info({
                'version': '3.0',
                'hostname': self.hostname,
                'platform': os.uname().sysname,
                'python_version': '.'.join(map(str, os.sys.version_info[:3])),
                'start_time': datetime.now().isoformat()
            })
            
            # Load alert rules
            self._load_alert_rules()
            
            # Start metrics HTTP server
            if self.auth_enabled:
                start_http_server(
                    self.metrics_port,
                    registry=REGISTRY,
                    auth_handler=lambda: basic_auth_handler(
                        self.auth_username,
                        self.auth_password
                    )
                )
            else:
                start_http_server(self.metrics_port, registry=REGISTRY)
            
            logger.info(
                f"{CyberColors.NEON_GREEN}✓ Metrics server started on port {self.metrics_port}{CyberColors.RESET}"
            )
            
            # Start collection tasks
            asyncio.create_task(self._collect_system_metrics())
            asyncio.create_task(self._check_alerts())
            asyncio.create_task(self._detect_anomalies())
            
            # Push to gateway if configured
            if self.pushgateway_url:
                asyncio.create_task(self._push_metrics())
            
        except Exception as e:
            logger.error(f"{CyberColors.NEON_RED}Metrics initialization failed: {e}{CyberColors.RESET}")
            raise
    
    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric="nexlify_cpu_usage_percent",
                condition="> 90",
                threshold=90,
                duration=300,
                severity="warning",
                message_template="High CPU usage detected: {value}%"
            ),
            AlertRule(
                name="low_memory",
                metric="nexlify_memory_usage_bytes",
                condition="> 90",
                threshold=90,
                duration=180,
                severity="critical",
                message_template="Low memory warning: {value}% used"
            ),
            AlertRule(
                name="exchange_disconnected",
                metric="nexlify_exchange_status",
                condition="== 0",
                threshold=0,
                duration=60,
                severity="critical",
                message_template="Exchange {labels[exchange]} disconnected"
            ),
            AlertRule(
                name="high_error_rate",
                metric="nexlify_errors_total",
                condition="> 100",
                threshold=100,
                duration=300,
                severity="warning",
                message_template="High error rate: {value} errors in 5 minutes"
            ),
            AlertRule(
                name="large_drawdown",
                metric="nexlify_max_drawdown_percent",
                condition="> 10",
                threshold=10,
                duration=60,
                severity="critical",
                message_template="Large drawdown detected: {value}%"
            ),
            AlertRule(
                name="ml_model_degradation",
                metric="nexlify_ml_model_accuracy_percent",
                condition="< 60",
                threshold=60,
                duration=600,
                severity="warning",
                message_template="ML model accuracy degraded: {value}%"
            )
        ]
        
        # Load custom rules from config
        custom_rules = self.config.get('alert_rules', [])
        for rule_config in custom_rules:
            try:
                rule = AlertRule(**rule_config)
                self.alert_rules.append(rule)
            except Exception as e:
                logger.error(f"Failed to load alert rule: {e}")
        
        # Add default rules if no custom rules
        if not self.alert_rules:
            self.alert_rules = default_rules
        
        logger.info(f"Loaded {len(self.alert_rules)} alert rules")
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        while True:
            try:
                start_time = time.perf_counter()
                
                # Uptime
                UPTIME.set(time.time() - self.start_time)
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(percpu=True)
                for i, percent in enumerate(cpu_percent):
                    CPU_USAGE.labels(core=str(i)).set(percent)
                    self._record_metric('cpu_usage', percent, {'core': str(i)})
                
                # Memory metrics
                memory = psutil.virtual_memory()
                MEMORY_USAGE.labels(type='used').set(memory.used)
                MEMORY_USAGE.labels(type='available').set(memory.available)
                MEMORY_USAGE.labels(type='total').set(memory.total)
                self._record_metric('memory_percent', memory.percent, {})
                
                # GPU metrics (if available)
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        GPU_USAGE.labels(gpu_id=str(gpu.id), metric='utilization').set(gpu.load * 100)
                        GPU_USAGE.labels(gpu_id=str(gpu.id), metric='memory').set(gpu.memoryUtil * 100)
                        GPU_USAGE.labels(gpu_id=str(gpu.id), metric='temperature').set(gpu.temperature)
                        self._record_metric('gpu_usage', gpu.load * 100, {'gpu_id': str(gpu.id)})
                except:
                    pass
                
                # Disk metrics
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        DISK_USAGE.labels(
                            mount_point=partition.mountpoint,
                            type='used'
                        ).set(usage.used)
                        DISK_USAGE.labels(
                            mount_point=partition.mountpoint,
                            type='free'
                        ).set(usage.free)
                        DISK_USAGE.labels(
                            mount_point=partition.mountpoint,
                            type='total'
                        ).set(usage.total)
                    except:
                        pass
                
                # Network metrics
                net_io = psutil.net_io_counters(pernic=True)
                for interface, counters in net_io.items():
                    NETWORK_IO.labels(interface=interface, direction='sent').inc(counters.bytes_sent)
                    NETWORK_IO.labels(interface=interface, direction='recv').inc(counters.bytes_recv)
                
                # Track collection time
                collection_time = time.perf_counter() - start_time
                self.collection_times.append(collection_time)
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                ERRORS_TOTAL.labels(component='metrics', error_type='collection').inc()
                await asyncio.sleep(30)
    
    def _record_metric(self, name: str, value: float, labels: Dict[str, str]):
        """Record metric for history and anomaly detection"""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            name=name,
            value=value,
            labels=labels
        )
        
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.metric_history[key].append(snapshot)
    
    async def _check_alerts(self):
        """Check alert rules periodically"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for rule in self.alert_rules:
                    if not rule.enabled:
                        continue
                    
                    # Check cooldown
                    if rule.last_triggered:
                        elapsed = (datetime.now() - rule.last_triggered).total_seconds()
                        if elapsed < rule.cooldown:
                            continue
                    
                    # Get metric value
                    metric_value = self._get_metric_value(rule.metric)
                    
                    if metric_value is None:
                        continue
                    
                    # Check condition
                    if self._evaluate_condition(metric_value, rule.condition, rule.threshold):
                        # Check duration
                        if await self._check_duration(rule):
                            await self._trigger_alert(rule, metric_value)
                
            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                ERRORS_TOTAL.labels(component='alerts', error_type='check').inc()
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        # This would interface with the actual Prometheus metrics
        # For now, return sample values
        sample_values = {
            'nexlify_cpu_usage_percent': psutil.cpu_percent(),
            'nexlify_memory_usage_bytes': psutil.virtual_memory().percent,
            'nexlify_exchange_status': 1,
            'nexlify_errors_total': 0,
            'nexlify_max_drawdown_percent': 5.2,
            'nexlify_ml_model_accuracy_percent': 85.5
        }
        
        return sample_values.get(metric_name)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        operators = {
            '>': lambda v, t: v > t,
            '<': lambda v, t: v < t,
            '>=': lambda v, t: v >= t,
            '<=': lambda v, t: v <= t,
            '==': lambda v, t: v == t,
            '!=': lambda v, t: v != t
        }
        
        for op, func in operators.items():
            if condition.startswith(op):
                return func(value, threshold)
        
        return False
    
    async def _check_duration(self, rule: AlertRule) -> bool:
        """Check if condition has been true for required duration"""
        # This would track condition state over time
        # For now, return True
        return True
    
    async def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger alert through configured channels"""
        rule.last_triggered = datetime.now()
        
        # Format message
        message = rule.message_template.format(
            value=value,
            labels={},  # Would include actual labels
            rule=rule.name,
            severity=rule.severity
        )
        
        logger.warning(
            f"{CyberColors.NEON_RED}🚨 Alert triggered: {rule.name} - {message}{CyberColors.RESET}"
        )
        
        CRITICAL_ALERTS.labels(alert_type=rule.name).inc()
        
        # Send through channels
        tasks = []
        
        if self.alert_channels['email'].get('enabled'):
            tasks.append(self._send_email_alert(rule, message))
        
        if self.alert_channels['discord'].get('enabled'):
            tasks.append(self._send_discord_alert(rule, message))
        
        if self.alert_channels['telegram'].get('enabled'):
            tasks.append(self._send_telegram_alert(rule, message))
        
        if self.alert_channels['webhook'].get('enabled'):
            tasks.append(self._send_webhook_alert(rule, message))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_email_alert(self, rule: AlertRule, message: str):
        """Send email alert"""
        try:
            config = self.alert_channels['email']
            
            msg = MIMEMultipart()
            msg['From'] = config['from_address']
            msg['To'] = ', '.join(config['to_addresses'])
            msg['Subject'] = f"[{rule.severity.upper()}] Nexlify Alert: {rule.name}"
            
            body = f"""
            <html>
                <body style="font-family: monospace; background-color: #0a0a0a; color: #00ff00;">
                    <h2 style="color: #ff0066;">🚨 NEXLIFY ALERT</h2>
                    <p><strong>Rule:</strong> {rule.name}</p>
                    <p><strong>Severity:</strong> <span style="color: {'#ff0000' if rule.severity == 'critical' else '#ffaa00'}">{rule.severity.upper()}</span></p>
                    <p><strong>Message:</strong> {message}</p>
                    <p><strong>Time:</strong> {datetime.now().isoformat()}</p>
                    <hr style="border-color: #00ff00;">
                    <p style="color: #666;">This is an automated alert from Nexlify Trading Matrix v3.0</p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                if config.get('use_tls'):
                    server.starttls()
                if config.get('username'):
                    server.login(config['username'], config['password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_discord_alert(self, rule: AlertRule, message: str):
        """Send Discord webhook alert"""
        try:
            webhook_url = self.alert_channels['discord']['webhook_url']
            
            embed = {
                "title": f"🚨 {rule.name}",
                "description": message,
                "color": 0xff0000 if rule.severity == "critical" else 0xffaa00,
                "fields": [
                    {"name": "Severity", "value": rule.severity.upper(), "inline": True},
                    {"name": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": True}
                ],
                "footer": {"text": "Nexlify Trading Matrix v3.0"}
            }
            
            async with httpx.AsyncClient() as client:
                await client.post(
                    webhook_url,
                    json={"embeds": [embed]}
                )
            
            logger.info(f"Discord alert sent for {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    async def _send_telegram_alert(self, rule: AlertRule, message: str):
        """Send Telegram alert"""
        try:
            config = self.alert_channels['telegram']
            bot_token = config['bot_token']
            chat_id = config['chat_id']
            
            text = f"""
🚨 *NEXLIFY ALERT*

*Rule:* {rule.name}
*Severity:* {rule.severity.upper()}
*Message:* {message}
*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            async with httpx.AsyncClient() as client:
                await client.post(
                    url,
                    json={
                        'chat_id': chat_id,
                        'text': text,
                        'parse_mode': 'Markdown'
                    }
                )
            
            logger.info(f"Telegram alert sent for {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    async def _send_webhook_alert(self, rule: AlertRule, message: str):
        """Send generic webhook alert"""
        try:
            webhook_url = self.alert_channels['webhook']['url']
            
            payload = {
                'alert': {
                    'rule': rule.name,
                    'severity': rule.severity,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    'metric': rule.metric,
                    'threshold': rule.threshold,
                    'condition': rule.condition
                }
            }
            
            async with httpx.AsyncClient() as client:
                await client.post(webhook_url, json=payload)
            
            logger.info(f"Webhook alert sent for {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    async def _detect_anomalies(self):
        """Detect anomalies in metrics using statistical methods"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for key, history in self.metric_history.items():
                    if len(history) < 100:  # Need enough data
                        continue
                    
                    # Calculate statistics
                    values = [h.value for h in history]
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5
                    
                    if std_dev == 0:
                        continue
                    
                    # Check latest value
                    latest = history[-1]
                    z_score = abs((latest.value - mean) / std_dev)
                    
                    if z_score > self.anomaly_threshold:
                        logger.warning(
                            f"{CyberColors.NEON_YELLOW}📈 Anomaly detected in {latest.name}: "
                            f"value={latest.value:.2f}, z-score={z_score:.2f}{CyberColors.RESET}"
                        )
                        
                        # Create alert
                        anomaly_rule = AlertRule(
                            name=f"anomaly_{latest.name}",
                            metric=latest.name,
                            condition=f"z_score > {self.anomaly_threshold}",
                            threshold=self.anomaly_threshold,
                            duration=0,
                            severity="warning",
                            message_template=f"Anomaly in {latest.name}: {{value}} (z-score: {z_score:.2f})"
                        )
                        
                        await self._trigger_alert(anomaly_rule, latest.value)
                
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                ERRORS_TOTAL.labels(component='anomaly', error_type='detection').inc()
    
    async def _push_metrics(self):
        """Push metrics to Prometheus Pushgateway"""
        while True:
            try:
                await asyncio.sleep(30)  # Push every 30 seconds
                
                if self.pushgateway_url:
                    # Generate metrics
                    metrics_data = generate_latest(REGISTRY)
                    
                    # Push to gateway
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.pushgateway_url}/metrics/job/nexlify/instance/{self.hostname}",
                            data=metrics_data,
                            headers={'Content-Type': CONTENT_TYPE_LATEST}
                        ) as response:
                            if response.status != 200:
                                logger.error(f"Failed to push metrics: {response.status}")
                
            except Exception as e:
                logger.error(f"Metrics push error: {e}")
    
    # Public methods for updating business metrics
    
    def record_trade(
        self,
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        status: str = "success"
    ):
        """Record trade metrics"""
        TRADES_TOTAL.labels(
            exchange=exchange,
            symbol=symbol,
            side=side,
            status=status
        ).inc()
        
        volume_usd = amount * price
        TRADE_VOLUME.labels(
            exchange=exchange,
            symbol=symbol,
            side=side
        ).inc(volume_usd)
    
    def update_profit(self, strategy: str, symbol: str, profit: float):
        """Update profit metrics"""
        TRADE_PROFIT.labels(strategy=strategy, symbol=symbol).set(profit)
        self._record_metric('profit', profit, {'strategy': strategy, 'symbol': symbol})
    
    def record_ml_prediction(self, model: str, symbol: str, latency: float):
        """Record ML prediction metrics"""
        ML_PREDICTIONS.labels(model=model, symbol=symbol).inc()
        ML_PREDICTION_LATENCY.labels(model=model).observe(latency)
    
    def update_strategy_metrics(
        self,
        strategy: str,
        accuracy: float,
        sharpe_ratio: float,
        signals: Dict[str, int]
    ):
        """Update strategy performance metrics"""
        STRATEGY_ACCURACY.labels(strategy=strategy).set(accuracy)
        STRATEGY_SHARPE_RATIO.labels(strategy=strategy).set(sharpe_ratio)
        
        for signal_type, count in signals.items():
            STRATEGY_SIGNALS.labels(
                strategy=strategy,
                signal_type=signal_type,
                symbol='all'
            ).inc(count)
    
    def update_exchange_status(self, exchange: str, connected: bool):
        """Update exchange connection status"""
        EXCHANGE_STATUS.labels(exchange=exchange).set(1 if connected else 0)
        
        if not connected:
            ERRORS_TOTAL.labels(
                component='exchange',
                error_type='disconnection'
            ).inc()
    
    def update_balance(self, exchange: str, balances: Dict[str, float]):
        """Update exchange balance metrics"""
        for asset, balance in balances.items():
            EXCHANGE_BALANCE.labels(exchange=exchange, asset=asset).set(balance)
    
    def update_orderbook(
        self,
        exchange: str,
        symbol: str,
        bid: float,
        ask: float,
        bid_depth: float,
        ask_depth: float
    ):
        """Update orderbook metrics"""
        spread = ask - bid
        ORDERBOOK_SPREAD.labels(exchange=exchange, symbol=symbol).set(spread)
        ORDERBOOK_DEPTH.labels(exchange=exchange, symbol=symbol, side='bid').set(bid_depth)
        ORDERBOOK_DEPTH.labels(exchange=exchange, symbol=symbol, side='ask').set(ask_depth)
    
    def update_risk_metrics(
        self,
        daily_pnl: float,
        win_rate: float,
        max_drawdown: float,
        risk_score: float
    ):
        """Update risk and performance metrics"""
        DAILY_PNL.set(daily_pnl)
        WIN_RATE.labels(timeframe='daily').set(win_rate)
        MAX_DRAWDOWN.labels(timeframe='daily').set(max_drawdown)
        RISK_SCORE.set(risk_score)
        
        # Check for risk alerts
        if risk_score > 80:
            logger.warning(
                f"{CyberColors.NEON_RED}High risk score: {risk_score}{CyberColors.RESET}"
            )
    
    def record_error(self, component: str, error_type: str):
        """Record error metrics"""
        ERRORS_TOTAL.labels(component=component, error_type=error_type).inc()
    
    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        latency: float
    ):
        """Record API request metrics"""
        API_REQUESTS.labels(
            endpoint=endpoint,
            method=method,
            status=str(status)
        ).inc()
        
        API_LATENCY.labels(endpoint=endpoint, method=method).observe(latency)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        # This would query actual Prometheus metrics
        # For now, return sample data
        return {
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'total_trades': 1234,
            'total_volume_usd': 567890.12,
            'current_profit_usd': 12345.67,
            'win_rate_percent': 65.4,
            'active_alerts': len([r for r in self.alert_rules if r.last_triggered]),
            'error_rate_per_hour': 2.3,
            'avg_latency_ms': 45.6,
            'cpu_usage_percent': psutil.cpu_percent(),
            'memory_usage_percent': psutil.virtual_memory().percent
        }
    
    async def export_dashboards(self, output_dir: Path):
        """Export Grafana dashboard configurations"""
        dashboards = {
            'nexlify_overview': self._create_overview_dashboard(),
            'nexlify_trading': self._create_trading_dashboard(),
            'nexlify_system': self._create_system_dashboard(),
            'nexlify_ml': self._create_ml_dashboard()
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, dashboard in dashboards.items():
            with open(output_dir / f"{name}.json", 'w') as f:
                json.dump(dashboard, f, indent=2)
        
        logger.info(
            f"{CyberColors.NEON_GREEN}✓ Exported {len(dashboards)} Grafana dashboards{CyberColors.RESET}"
        )
    
    def _create_overview_dashboard(self) -> Dict[str, Any]:
        """Create overview Grafana dashboard"""
        return {
            "dashboard": {
                "title": "Nexlify Overview - Cyberpunk Trading Matrix",
                "panels": [
                    {
                        "title": "System Uptime",
                        "type": "stat",
                        "targets": [{"expr": "nexlify_uptime_seconds / 3600"}]
                    },
                    {
                        "title": "Total Profit (USD)",
                        "type": "stat",
                        "targets": [{"expr": "sum(nexlify_trade_profit_usd)"}]
                    },
                    {
                        "title": "Win Rate",
                        "type": "gauge",
                        "targets": [{"expr": "nexlify_win_rate_percent"}]
                    },
                    {
                        "title": "Active Exchanges",
                        "type": "stat",
                        "targets": [{"expr": "sum(nexlify_exchange_status)"}]
                    }
                ]
            }
        }
    
    def _create_trading_dashboard(self) -> Dict[str, Any]:
        """Create trading Grafana dashboard"""
        return {
            "dashboard": {
                "title": "Nexlify Trading Performance",
                "panels": [
                    {
                        "title": "Trade Volume by Exchange",
                        "type": "graph",
                        "targets": [{"expr": "rate(nexlify_trade_volume_usd[5m])"}]
                    },
                    {
                        "title": "Strategy Performance",
                        "type": "table",
                        "targets": [{"expr": "nexlify_strategy_sharpe_ratio"}]
                    }
                ]
            }
        }
    
    def _create_system_dashboard(self) -> Dict[str, Any]:
        """Create system monitoring dashboard"""
        return {
            "dashboard": {
                "title": "Nexlify System Metrics",
                "panels": [
                    {
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [{"expr": "nexlify_cpu_usage_percent"}]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [{"expr": "nexlify_memory_usage_bytes / 1024 / 1024 / 1024"}]
                    }
                ]
            }
        }
    
    def _create_ml_dashboard(self) -> Dict[str, Any]:
        """Create ML monitoring dashboard"""
        return {
            "dashboard": {
                "title": "Nexlify ML Models",
                "panels": [
                    {
                        "title": "Model Accuracy",
                        "type": "graph",
                        "targets": [{"expr": "nexlify_ml_model_accuracy_percent"}]
                    },
                    {
                        "title": "Prediction Latency",
                        "type": "heatmap",
                        "targets": [{"expr": "nexlify_ml_prediction_latency_seconds"}]
                    }
                ]
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown metrics collector"""
        logger.info(f"{CyberColors.NEURAL_PURPLE}Shutting down metrics collector...{CyberColors.RESET}")
        
        # Final metrics push
        if self.pushgateway_url:
            try:
                await self._push_metrics()
            except:
                pass
        
        logger.info(f"{CyberColors.NEURAL_PURPLE}Metrics collector offline{CyberColors.RESET}")


# Global metrics instance
_metrics_collector: Optional[NexlifyMetricsCollector] = None

def get_metrics_collector() -> NexlifyMetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = NexlifyMetricsCollector()
    return _metrics_collector


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize config
        config_loader = get_config_loader()
        await config_loader.initialize()
        
        # Create metrics collector
        metrics = NexlifyMetricsCollector()
        await metrics.initialize()
        
        # Simulate some metrics
        print(f"\n{CyberColors.NEON_CYAN}=== Nexlify Metrics Collector ==={CyberColors.RESET}")
        
        # Record some trades
        metrics.record_trade("coinbase", "BTC/USDT", "buy", 0.1, 50000)
        metrics.record_trade("binance", "ETH/USDT", "sell", 1.0, 3000)
        
        # Update profits
        metrics.update_profit("arbitrage", "BTC/USDT", 1234.56)
        metrics.update_profit("momentum", "ETH/USDT", 567.89)
        
        # Update strategy metrics
        metrics.update_strategy_metrics(
            "arbitrage",
            accuracy=85.5,
            sharpe_ratio=2.1,
            signals={"buy": 10, "sell": 8}
        )
        
        # Update risk metrics
        metrics.update_risk_metrics(
            daily_pnl=1802.45,
            win_rate=65.4,
            max_drawdown=5.2,
            risk_score=35
        )
        
        # Get summary
        summary = metrics.get_metrics_summary()
        print(f"\n{CyberColors.NEON_GREEN}Metrics Summary:{CyberColors.RESET}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Export dashboards
        await metrics.export_dashboards(Path("./dashboards"))
        
        # Let it run for a bit
        print(f"\n{CyberColors.NEON_PINK}Metrics server running on port {metrics.metrics_port}{CyberColors.RESET}")
        print(f"{CyberColors.NEON_YELLOW}Press Ctrl+C to stop{CyberColors.RESET}")
        
        try:
            await asyncio.sleep(3600)
        except KeyboardInterrupt:
            print(f"\n{CyberColors.NEURAL_PURPLE}Stopping...{CyberColors.RESET}")
        finally:
            await metrics.shutdown()
    
    asyncio.run(main())
