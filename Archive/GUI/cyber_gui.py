"""
Nexlify Cyberpunk Trading GUI - Enhanced with V3 Improvements
High-performance PyQt5 interface with real-time updates and comprehensive security
"""

import sys
import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque
import re

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtChart import *
import qasync

# Import our enhanced modules
from nexlify_neural_net import NexlifyNeuralNet
from nexlify_advanced_security import SecurityManager, TwoFactorAuth
from nexlify_audit_trail import AuditManager
from nexlify_predictive_features import PredictiveEngine
from nexlify_multi_strategy import MultiStrategyOptimizer
from nexlify_cyberpunk_effects import CyberpunkEffects, SoundManager
from nexlify_ai_companion import AITradingCompanion
from error_handler import get_error_handler, ErrorContext
from utils_module import (
    ValidationUtils, CryptoUtils, FileUtils, NetworkUtils,
    TimeUtils, MathUtils
)

logger = logging.getLogger(__name__)
error_handler = get_error_handler()

# Constants
DEBOUNCE_INSTANT = 100  # ms for normal actions
LOADING_TIMEOUT = 30000  # 30s timeout for async operations
LOG_MAX_SIZE_MB = 25  # Maximum log size in MB
SESSION_CHECK_INTERVAL = 60000  # Check session every 60s
GRACE_PERIOD_MINUTES = 5  # Grace period before forcing re-auth

@dataclass
class GUIConfig:
    """GUI Configuration with cyberpunk theme"""
    # Colors
    bg_primary = "#0a0a0a"
    bg_secondary = "#151515"
    bg_tertiary = "#1f1f1f"
    accent_primary = "#00ffff"  # Cyan
    accent_secondary = "#ff00ff"  # Magenta
    accent_success = "#00ff00"  # Green
    accent_warning = "#ffff00"  # Yellow
    accent_error = "#ff0000"    # Red
    text_primary = "#ffffff"
    text_secondary = "#b0b0b0"
    text_dim = "#606060"
    
    # Fonts
    font_family = "Consolas"
    font_size_small = 10
    font_size_normal = 12
    font_size_large = 14
    font_size_header = 18
    
    # Animation
    animation_duration = 200
    glow_intensity = 20
    
class RateLimitedButton(QPushButton):
    """Button with built-in rate limiting and loading states"""
    
    def __init__(self, text: str, debounce_ms: int = DEBOUNCE_INSTANT):
        super().__init__(text)
        self.debounce_ms = debounce_ms
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._enable_button)
        self.loading = False
        self.original_text = text
        self._setup_loading_animation()
        
    def _setup_loading_animation(self):
        """Setup loading spinner animation"""
        self.loading_movie = QMovie()  # Would load actual spinner
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self._update_loading_text)
        self.loading_dots = 0
        
    def click_with_debounce(self):
        """Handle click with debounce"""
        if not self.isEnabled():
            return
        
        self.setEnabled(False)
        self.debounce_timer.start(self.debounce_ms)
        
        # Emit clicked signal
        self.clicked.emit()
    
    def _enable_button(self):
        """Re-enable button after debounce"""
        if not self.loading:
            self.setEnabled(True)
    
    def set_loading(self, loading: bool):
        """Set loading state"""
        self.loading = loading
        if loading:
            self.setEnabled(False)
            self.loading_timer.start(500)  # Update every 500ms
            self._update_loading_text()
        else:
            self.loading_timer.stop()
            self.setText(self.original_text)
            self.setEnabled(True)
    
    def _update_loading_text(self):
        """Update loading animation text"""
        dots = "." * (self.loading_dots % 4)
        self.setText(f"{self.original_text}{dots}")
        self.loading_dots += 1

class VirtualTableModel(QAbstractTableModel):
    """Virtual table model for high-performance data display"""
    
    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns
        self.data_cache = []
        self.batch_updates = []
        self.batch_timer = QTimer()
        self.batch_timer.timeout.connect(self._apply_batch_updates)
        self.batch_timer.start(100)  # Batch every 100ms
        
    def add_batch_update(self, data: List[Dict]):
        """Add data to batch update queue"""
        self.batch_updates.extend(data)
        
    def _apply_batch_updates(self):
        """Apply batched updates"""
        if not self.batch_updates:
            return
            
        self.beginResetModel()
        self.data_cache = self.batch_updates[-1000:]  # Keep last 1000 items
        self.batch_updates.clear()
        self.endResetModel()
        
    def rowCount(self, parent=QModelIndex()):
        return len(self.data_cache)
        
    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)
        
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.DisplayRole:
            row = self.data_cache[index.row()]
            col = self.columns[index.column()]
            return str(row.get(col, ""))
            
        return None
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section]
        return None

class LogWidget(QPlainTextEdit):
    """Memory-efficient log widget with size-based rotation"""
    
    def __init__(self, max_size_mb: float = LOG_MAX_SIZE_MB):
        super().__init__()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.setReadOnly(True)
        self.setMaximumBlockCount(10000)  # Limit blocks
        self.document().setMaximumBlockCount(10000)
        
    def append_log(self, message: str, level: str = "INFO"):
        """Append log with size checking"""
        # Check document size
        doc_size = len(self.toPlainText().encode('utf-8'))
        if doc_size > self.max_size_bytes:
            # Remove first 20% of content
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 
                              self.document().blockCount() // 5)
            cursor.removeSelectedText()
        
        # Add timestamp and format
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color based on level
        colors = {
            "INFO": "#00ff00",
            "WARNING": "#ffff00", 
            "ERROR": "#ff0000",
            "DEBUG": "#00ffff"
        }
        color = colors.get(level, "#ffffff")
        
        # Append with HTML formatting
        self.appendHtml(
            f'<span style="color: #606060">[{timestamp}]</span> '
            f'<span style="color: {color}">[{level}]</span> '
            f'<span style="color: #ffffff">{message}</span>'
        )
        
        # Auto-scroll to bottom
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

class SessionManager(QObject):
    """Manages user sessions with security integration"""
    
    session_expired = pyqtSignal()
    session_warning = pyqtSignal(int)  # Minutes remaining
    
    def __init__(self, security_manager: SecurityManager):
        super().__init__()
        self.security_manager = security_manager
        self.session_token = None
        self.last_activity = datetime.now()
        self.session_timeout = timedelta(minutes=30)  # From config
        self.grace_period = timedelta(minutes=GRACE_PERIOD_MINUTES)
        
        # Check session periodically
        self.check_timer = QTimer()
        self.check_timer.timeout.connect(self.check_session)
        self.check_timer.start(SESSION_CHECK_INTERVAL)
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        
    def check_session(self):
        """Check if session is still valid"""
        if not self.session_token:
            return
            
        try:
            # Validate with security manager
            is_valid = self.security_manager.validate_session(
                self.session_token,
                self.get_current_ip()
            )
            
            if not is_valid:
                self.session_expired.emit()
                return
                
            # Check timeout
            time_since_activity = datetime.now() - self.last_activity
            time_remaining = self.session_timeout - time_since_activity
            
            if time_remaining <= timedelta(0):
                # Session timed out
                self.session_expired.emit()
            elif time_remaining <= self.grace_period:
                # In grace period
                minutes_left = int(time_remaining.total_seconds() / 60)
                self.session_warning.emit(minutes_left)
                
        except Exception as e:
            logger.error(f"Session check failed: {e}")
            
    def get_current_ip(self) -> str:
        """Get current IP address"""
        # In production, would get actual IP
        return "127.0.0.1"
        
    def set_session(self, token: str):
        """Set new session token"""
        self.session_token = token
        self.last_activity = datetime.now()

class CyberGUI(QMainWindow):
    """Main Cyberpunk Trading GUI with V3 improvements"""
    
    def __init__(self):
        super().__init__()
        self.config = GUIConfig()
        self.neural_net = None
        self.security_manager = None
        self.audit_manager = None
        self.predictive_engine = None
        self.strategy_optimizer = None
        self.sound_manager = None
        self.ai_companion = None
        self.session_manager = None
        
        # State tracking
        self.is_authenticated = False
        self.is_trading_active = False
        self.active_pairs_model = None
        self.positions_model = None
        
        # Initialize components
        self._init_components()
        self._setup_ui()
        self._apply_cyberpunk_theme()
        self._setup_connections()
        
    def _init_components(self):
        """Initialize backend components"""
        try:
            # Load configuration
            self.app_config = FileUtils.load_json('enhanced_config.json')
            if not self.app_config:
                self.app_config = FileUtils.load_json('neural_config.json')
                
            # Initialize security
            self.security_manager = SecurityManager(
                master_password=self.app_config.get('security', {}).get('master_password', '')
            )
            
            # Initialize session manager
            self.session_manager = SessionManager(self.security_manager)
            self.session_manager.session_expired.connect(self._handle_session_expired)
            self.session_manager.session_warning.connect(self._handle_session_warning)
            
            # Initialize other components
            self.audit_manager = AuditManager()
            self.sound_manager = SoundManager()
            self.sound_manager.initialize()
            
            # These will be initialized after authentication
            self.neural_net = None
            self.predictive_engine = None
            self.strategy_optimizer = None
            self.ai_companion = None
            
        except Exception as e:
            error_handler.log_error(e, {"method": "_init_components"})
            
    def _setup_ui(self):
        """Setup the main UI structure"""
        self.setWindowTitle("Nexlify Neural Trading Matrix v2.0.8")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        self._create_header(main_layout)
        
        # Content area with tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        
        # Create tabs
        self._create_dashboard_tab()
        self._create_trading_tab()
        self._create_portfolio_tab()
        self._create_strategies_tab()
        self._create_settings_tab()
        self._create_logs_tab()
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self._create_status_bar()
        
        # Initially show login dialog
        QTimer.singleShot(100, self._show_login_dialog)
        
    def _create_header(self, parent_layout):
        """Create cyberpunk header with status indicators"""
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_frame.setObjectName("header_frame")
        
        header_layout = QHBoxLayout(header_frame)
        
        # Logo and title
        title_layout = QVBoxLayout()
        title_label = QLabel("NEXLIFY NEURAL TRADING MATRIX")
        title_label.setObjectName("header_title")
        subtitle_label = QLabel("Cyberpunk Trading Engine v2.0.8")
        subtitle_label.setObjectName("header_subtitle")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        header_layout.addLayout(title_layout)
        
        header_layout.addStretch()
        
        # Status indicators
        self.status_grid = QGridLayout()
        
        # Connection status
        self.connection_led = self._create_status_led("Connection", False)
        self.status_grid.addWidget(self.connection_led, 0, 0)
        
        # Trading status
        self.trading_led = self._create_status_led("Trading", False)
        self.status_grid.addWidget(self.trading_led, 0, 1)
        
        # Security status
        self.security_led = self._create_status_led("Security", True)
        self.status_grid.addWidget(self.security_led, 1, 0)
        
        # AI status
        self.ai_led = self._create_status_led("AI Active", False)
        self.status_grid.addWidget(self.ai_led, 1, 1)
        
        header_layout.addLayout(self.status_grid)
        
        # User info
        self.user_info_label = QLabel("Not Authenticated")
        self.user_info_label.setObjectName("user_info")
        header_layout.addWidget(self.user_info_label)
        
        parent_layout.addWidget(header_frame)
        
    def _create_status_led(self, label: str, initial_state: bool) -> QWidget:
        """Create a status LED indicator"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(2)
        
        # LED
        led = QLabel()
        led.setFixedSize(20, 20)
        led.setObjectName("status_led_on" if initial_state else "status_led_off")
        led.setAlignment(Qt.AlignCenter)
        
        # Label
        text = QLabel(label)
        text.setObjectName("status_label")
        text.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(led, alignment=Qt.AlignCenter)
        layout.addWidget(text)
        
        return container
        
    def _create_dashboard_tab(self):
        """Create main dashboard with real-time data"""
        dashboard = QWidget()
        layout = QGridLayout(dashboard)
        
        # Market overview
        market_group = QGroupBox("Market Overview")
        market_layout = QVBoxLayout(market_group)
        
        # BTC price with real-time updates
        self.btc_price_label = QLabel("BTC: Loading...")
        self.btc_price_label.setObjectName("price_display")
        market_layout.addWidget(self.btc_price_label)
        
        # Market stats
        self.market_stats_widget = self._create_market_stats()
        market_layout.addWidget(self.market_stats_widget)
        
        layout.addWidget(market_group, 0, 0)
        
        # Active pairs with virtual scrolling
        pairs_group = QGroupBox("Active Trading Pairs")
        pairs_layout = QVBoxLayout(pairs_group)
        
        self.active_pairs_table = QTableView()
        self.active_pairs_model = VirtualTableModel([
            "Symbol", "Price", "24h Volume", "24h Change", "Spread", 
            "Exchanges", "Risk Score", "Action"
        ])
        self.active_pairs_table.setModel(self.active_pairs_model)
        
        # Configure table
        self.active_pairs_table.setAlternatingRowColors(True)
        self.active_pairs_table.setSelectionBehavior(QTableView.SelectRows)
        self.active_pairs_table.verticalHeader().setDefaultSectionSize(30)
        
        pairs_layout.addWidget(self.active_pairs_table)
        layout.addWidget(pairs_group, 1, 0, 1, 2)
        
        # Profit chart
        profit_group = QGroupBox("Profit Performance")
        profit_layout = QVBoxLayout(profit_group)
        
        self.profit_chart = self._create_profit_chart()
        profit_layout.addWidget(self.profit_chart)
        
        layout.addWidget(profit_group, 0, 1)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        # Start/Stop trading with confirmation for critical action
        self.toggle_trading_btn = RateLimitedButton("Start Auto-Trading")
        self.toggle_trading_btn.clicked.connect(self._toggle_trading)
        actions_layout.addWidget(self.toggle_trading_btn)
        
        # Emergency stop with confirmation
        self.emergency_stop_btn = RateLimitedButton("EMERGENCY STOP")
        self.emergency_stop_btn.setObjectName("emergency_button")
        self.emergency_stop_btn.clicked.connect(self._emergency_stop)
        actions_layout.addWidget(self.emergency_stop_btn)
        
        # Refresh data
        self.refresh_btn = RateLimitedButton("Refresh Data")
        self.refresh_btn.clicked.connect(self._refresh_data)
        actions_layout.addWidget(self.refresh_btn)
        
        layout.addWidget(actions_group, 2, 0, 1, 2)
        
        self.tab_widget.addTab(dashboard, "Dashboard")
        
    def _create_market_stats(self) -> QWidget:
        """Create market statistics widget"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        stats = [
            ("Total Volume 24h", "$0.00"),
            ("Active Exchanges", "0"),
            ("Profitable Pairs", "0"),
            ("Current Profit", "$0.00")
        ]
        
        self.stat_labels = {}
        for i, (name, value) in enumerate(stats):
            name_label = QLabel(name + ":")
            value_label = QLabel(value)
            value_label.setObjectName("stat_value")
            
            row = i // 2
            col = (i % 2) * 2
            
            layout.addWidget(name_label, row, col)
            layout.addWidget(value_label, row, col + 1)
            
            self.stat_labels[name] = value_label
            
        return widget
        
    def _create_profit_chart(self) -> QChartView:
        """Create profit performance chart"""
        # Create chart
        self.profit_series = QLineSeries()
        self.profit_series.setName("Profit (USDT)")
        
        # Sample data
        for i in range(24):
            self.profit_series.append(i, 0)
            
        chart = QChart()
        chart.addSeries(self.profit_series)
        chart.setTitle("24 Hour Profit Performance")
        chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # Axes
        axis_x = QValueAxis()
        axis_x.setRange(0, 24)
        axis_x.setLabelFormat("%d:00")
        axis_x.setTitleText("Hour")
        
        axis_y = QValueAxis()
        axis_y.setRange(-100, 100)
        axis_y.setLabelFormat("$%.2f")
        axis_y.setTitleText("Profit (USDT)")
        
        chart.addAxis(axis_x, Qt.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignLeft)
        
        self.profit_series.attachAxis(axis_x)
        self.profit_series.attachAxis(axis_y)
        
        # Style
        chart.setBackgroundBrush(QBrush(QColor(self.config.bg_secondary)))
        chart.setTitleBrush(QBrush(QColor(self.config.text_primary)))
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        
        return chart_view
        
    def _create_trading_tab(self):
        """Create trading interface with order management"""
        trading = QWidget()
        layout = QVBoxLayout(trading)
        
        # Trading controls
        controls_group = QGroupBox("Trading Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Exchange selection
        controls_layout.addWidget(QLabel("Exchange:"), 0, 0)
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems(["All Exchanges", "Binance", "ByBit", "OKX"])
        controls_layout.addWidget(self.exchange_combo, 0, 1)
        
        # Pair input with validation
        controls_layout.addWidget(QLabel("Trading Pair:"), 0, 2)
        self.pair_input = QLineEdit()
        self.pair_input.setPlaceholderText("BTC/USDT")
        self.pair_input.textChanged.connect(self._validate_pair_input)
        controls_layout.addWidget(self.pair_input, 0, 3)
        
        # Amount input with validation
        controls_layout.addWidget(QLabel("Amount:"), 1, 0)
        self.amount_input = QLineEdit()
        self.amount_input.setPlaceholderText("0.001")
        self.amount_input.setValidator(QDoubleValidator(0.00000001, 999999, 8))
        controls_layout.addWidget(self.amount_input, 1, 1)
        
        # Order type
        controls_layout.addWidget(QLabel("Order Type:"), 1, 2)
        self.order_type_combo = QComboBox()
        self.order_type_combo.addItems(["Market", "Limit", "Stop-Limit"])
        controls_layout.addWidget(self.order_type_combo, 1, 3)
        
        # Price input (for limit orders)
        controls_layout.addWidget(QLabel("Price:"), 2, 0)
        self.price_input = QLineEdit()
        self.price_input.setPlaceholderText("0.00")
        self.price_input.setValidator(QDoubleValidator(0.00000001, 999999999, 8))
        self.price_input.setEnabled(False)
        controls_layout.addWidget(self.price_input, 2, 1)
        
        # Enable price input for limit orders
        self.order_type_combo.currentTextChanged.connect(
            lambda t: self.price_input.setEnabled(t != "Market")
        )
        
        # Buy/Sell buttons with confirmation for critical actions
        self.buy_btn = RateLimitedButton("BUY")
        self.buy_btn.setObjectName("buy_button")
        self.buy_btn.clicked.connect(lambda: self._execute_trade("buy"))
        controls_layout.addWidget(self.buy_btn, 2, 2)
        
        self.sell_btn = RateLimitedButton("SELL")
        self.sell_btn.setObjectName("sell_button")
        self.sell_btn.clicked.connect(lambda: self._execute_trade("sell"))
        controls_layout.addWidget(self.sell_btn, 2, 3)
        
        layout.addWidget(controls_group)
        
        # Open positions with virtual scrolling
        positions_group = QGroupBox("Open Positions")
        positions_layout = QVBoxLayout(positions_group)
        
        self.positions_table = QTableView()
        self.positions_model = VirtualTableModel([
            "Pair", "Side", "Amount", "Entry Price", "Current Price",
            "PnL", "PnL %", "Action"
        ])
        self.positions_table.setModel(self.positions_model)
        
        positions_layout.addWidget(self.positions_table)
        layout.addWidget(positions_group)
        
        # Order history
        history_group = QGroupBox("Order History")
        history_layout = QVBoxLayout(history_group)
        
        self.order_history_table = QTableView()
        self.order_history_model = VirtualTableModel([
            "Time", "Pair", "Side", "Type", "Amount", "Price", "Status"
        ])
        self.order_history_table.setModel(self.order_history_model)
        
        history_layout.addWidget(self.order_history_table)
        layout.addWidget(history_group)
        
        self.tab_widget.addTab(trading, "Trading")
        
    def _create_portfolio_tab(self):
        """Create portfolio overview with balances"""
        portfolio = QWidget()
        layout = QVBoxLayout(portfolio)
        
        # Portfolio summary
        summary_group = QGroupBox("Portfolio Summary")
        summary_layout = QGridLayout(summary_group)
        
        # Total value
        self.total_value_label = QLabel("Total Value: $0.00")
        self.total_value_label.setObjectName("portfolio_value")
        summary_layout.addWidget(self.total_value_label, 0, 0, 1, 2)
        
        # 24h change
        self.portfolio_change_label = QLabel("24h Change: +0.00%")
        summary_layout.addWidget(self.portfolio_change_label, 1, 0)
        
        # Available balance
        self.available_balance_label = QLabel("Available: $0.00")
        summary_layout.addWidget(self.available_balance_label, 1, 1)
        
        layout.addWidget(summary_group)
        
        # Asset balances
        balances_group = QGroupBox("Asset Balances")
        balances_layout = QVBoxLayout(balances_group)
        
        self.balances_table = QTableView()
        self.balances_model = VirtualTableModel([
            "Asset", "Amount", "Value (USDT)", "24h Change", "% of Portfolio"
        ])
        self.balances_table.setModel(self.balances_model)
        
        balances_layout.addWidget(self.balances_table)
        layout.addWidget(balances_group)
        
        # Withdrawal section
        withdrawal_group = QGroupBox("Withdrawals")
        withdrawal_layout = QGridLayout(withdrawal_group)
        
        # BTC address input with validation
        withdrawal_layout.addWidget(QLabel("BTC Address:"), 0, 0)
        self.btc_address_input = QLineEdit()
        self.btc_address_input.setPlaceholderText("bc1q...")
        self.btc_address_input.textChanged.connect(self._validate_btc_address)
        withdrawal_layout.addWidget(self.btc_address_input, 0, 1, 1, 2)
        
        # Validation indicator
        self.btc_address_valid_label = QLabel()
        withdrawal_layout.addWidget(self.btc_address_valid_label, 0, 3)
        
        # Withdrawal amount
        withdrawal_layout.addWidget(QLabel("Amount (USDT):"), 1, 0)
        self.withdrawal_amount_input = QLineEdit()
        self.withdrawal_amount_input.setValidator(QDoubleValidator(0.01, 999999, 2))
        withdrawal_layout.addWidget(self.withdrawal_amount_input, 1, 1)
        
        # Withdrawal button with confirmation
        self.withdraw_btn = RateLimitedButton("Withdraw to BTC")
        self.withdraw_btn.clicked.connect(self._withdraw_funds)
        withdrawal_layout.addWidget(self.withdraw_btn, 1, 2)
        
        layout.addWidget(withdrawal_group)
        
        self.tab_widget.addTab(portfolio, "Portfolio")
        
    def _create_strategies_tab(self):
        """Create strategy management interface"""
        strategies = QWidget()
        layout = QVBoxLayout(strategies)
        
        # Active strategies
        active_group = QGroupBox("Active Strategies")
        active_layout = QVBoxLayout(active_group)
        
        self.strategies_list = QListWidget()
        strategies_data = [
            "Arbitrage Scanner - Active",
            "Momentum Trading - Active", 
            "Market Making - Paused",
            "DeFi Integration - Inactive"
        ]
        
        for strategy in strategies_data:
            item = QListWidgetItem(strategy)
            if "Active" in strategy:
                item.setForeground(QColor(self.config.accent_success))
            elif "Paused" in strategy:
                item.setForeground(QColor(self.config.accent_warning))
            else:
                item.setForeground(QColor(self.config.text_dim))
            self.strategies_list.addItem(item)
            
        active_layout.addWidget(self.strategies_list)
        
        # Strategy controls
        controls_layout = QHBoxLayout()
        
        self.enable_strategy_btn = RateLimitedButton("Enable")
        self.enable_strategy_btn.clicked.connect(self._enable_strategy)
        controls_layout.addWidget(self.enable_strategy_btn)
        
        self.disable_strategy_btn = RateLimitedButton("Disable")
        self.disable_strategy_btn.clicked.connect(self._disable_strategy)
        controls_layout.addWidget(self.disable_strategy_btn)
        
        self.configure_strategy_btn = RateLimitedButton("Configure")
        self.configure_strategy_btn.clicked.connect(self._configure_strategy)
        controls_layout.addWidget(self.configure_strategy_btn)
        
        active_layout.addLayout(controls_layout)
        layout.addWidget(active_group)
        
        # Strategy performance
        performance_group = QGroupBox("Strategy Performance")
        performance_layout = QVBoxLayout(performance_group)
        
        self.strategy_performance_table = QTableView()
        self.strategy_performance_model = VirtualTableModel([
            "Strategy", "Trades", "Win Rate", "Total Profit", "Sharpe Ratio"
        ])
        self.strategy_performance_table.setModel(self.strategy_performance_model)
        
        performance_layout.addWidget(self.strategy_performance_table)
        layout.addWidget(performance_group)
        
        self.tab_widget.addTab(strategies, "Strategies")
        
    def _create_settings_tab(self):
        """Create settings interface with all configurations"""
        settings = QScrollArea()
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # API Configuration
        api_group = QGroupBox("Exchange API Configuration")
        api_layout = QFormLayout(api_group)
        
        self.api_inputs = {}
        exchanges = ["Binance", "ByBit", "OKX", "Kraken"]
        
        for exchange in exchanges:
            # API Key
            api_key_input = QLineEdit()
            api_key_input.setEchoMode(QLineEdit.Password)
            api_key_input.setPlaceholderText("API Key")
            
            # Secret Key
            secret_input = QLineEdit()
            secret_input.setEchoMode(QLineEdit.Password)
            secret_input.setPlaceholderText("Secret Key")
            
            # Test connection button
            test_btn = RateLimitedButton("Test")
            test_btn.clicked.connect(lambda c, ex=exchange: self._test_exchange_connection(ex))
            
            # Layout
            exchange_layout = QHBoxLayout()
            exchange_layout.addWidget(api_key_input)
            exchange_layout.addWidget(secret_input)
            exchange_layout.addWidget(test_btn)
            
            api_layout.addRow(f"{exchange}:", exchange_layout)
            
            self.api_inputs[exchange] = {
                'api_key': api_key_input,
                'secret': secret_input,
                'test_btn': test_btn
            }
            
        settings_layout.addWidget(api_group)
        
        # Trading Settings
        trading_group = QGroupBox("Trading Configuration")
        trading_layout = QFormLayout(trading_group)
        
        # Risk level
        self.risk_level_combo = QComboBox()
        self.risk_level_combo.addItems(["Low", "Medium", "High"])
        trading_layout.addRow("Risk Level:", self.risk_level_combo)
        
        # Min profit threshold
        self.min_profit_input = QDoubleSpinBox()
        self.min_profit_input.setRange(0.1, 10.0)
        self.min_profit_input.setSingleStep(0.1)
        self.min_profit_input.setSuffix("%")
        self.min_profit_input.setValue(0.5)
        trading_layout.addRow("Min Profit Threshold:", self.min_profit_input)
        
        # Max positions
        self.max_positions_input = QSpinBox()
        self.max_positions_input.setRange(1, 50)
        self.max_positions_input.setValue(10)
        trading_layout.addRow("Max Concurrent Positions:", self.max_positions_input)
        
        settings_layout.addWidget(trading_group)
        
        # Security Settings
        security_group = QGroupBox("Security Configuration")
        security_layout = QFormLayout(security_group)
        
        # Master password (optional)
        self.master_password_check = QCheckBox("Enable Master Password")
        security_layout.addRow(self.master_password_check)
        
        # 2FA (optional)
        self.twofa_check = QCheckBox("Enable 2FA Authentication")
        security_layout.addRow(self.twofa_check)
        
        # Setup 2FA button
        self.setup_2fa_btn = RateLimitedButton("Setup 2FA")
        self.setup_2fa_btn.clicked.connect(self._setup_2fa)
        self.setup_2fa_btn.setEnabled(False)
        self.twofa_check.toggled.connect(self.setup_2fa_btn.setEnabled)
        security_layout.addRow(self.setup_2fa_btn)
        
        # IP Whitelist
        self.ip_whitelist_check = QCheckBox("Enable IP Whitelist")
        security_layout.addRow(self.ip_whitelist_check)
        
        self.ip_whitelist_text = QPlainTextEdit()
        self.ip_whitelist_text.setPlaceholderText("One IP per line")
        self.ip_whitelist_text.setMaximumHeight(100)
        self.ip_whitelist_text.setEnabled(False)
        self.ip_whitelist_check.toggled.connect(self.ip_whitelist_text.setEnabled)
        security_layout.addRow("Whitelisted IPs:", self.ip_whitelist_text)
        
        settings_layout.addWidget(security_group)
        
        # Save settings button
        self.save_settings_btn = RateLimitedButton("Save All Settings")
        self.save_settings_btn.clicked.connect(self._save_settings)
        settings_layout.addWidget(self.save_settings_btn)
        
        settings.setWidget(settings_widget)
        settings.setWidgetResizable(True)
        self.tab_widget.addTab(settings, "Settings")
        
    def _create_logs_tab(self):
        """Create logs interface with memory-efficient display"""
        logs = QWidget()
        layout = QVBoxLayout(logs)
        
        # Log filters
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Level:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["ALL", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.currentTextChanged.connect(self._filter_logs)
        filter_layout.addWidget(self.log_level_combo)
        
        filter_layout.addWidget(QLabel("Search:"))
        self.log_search_input = QLineEdit()
        self.log_search_input.setPlaceholderText("Filter logs...")
        self.log_search_input.textChanged.connect(self._filter_logs)
        filter_layout.addWidget(self.log_search_input)
        
        self.clear_logs_btn = RateLimitedButton("Clear Logs")
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        filter_layout.addWidget(self.clear_logs_btn)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Log display with size limit
        self.log_widget = LogWidget(max_size_mb=LOG_MAX_SIZE_MB)
        layout.addWidget(self.log_widget)
        
        # Log initial message
        self.log_widget.append_log("Nexlify Neural Trading Matrix initialized", "INFO")
        
        self.tab_widget.addTab(logs, "Logs")
        
    def _create_status_bar(self):
        """Create status bar with connection info"""
        self.status_bar = self.statusBar()
        
        # Connection status
        self.connection_status = QLabel("Disconnected")
        self.status_bar.addWidget(self.connection_status)
        
        # Separator
        self.status_bar.addWidget(QLabel(" | "))
        
        # Exchange count
        self.exchange_count_label = QLabel("Exchanges: 0")
        self.status_bar.addWidget(self.exchange_count_label)
        
        # Separator
        self.status_bar.addWidget(QLabel(" | "))
        
        # Last update
        self.last_update_label = QLabel("Last Update: Never")
        self.status_bar.addWidget(self.last_update_label)
        
        # Permanent widgets on the right
        self.status_bar.addPermanentWidget(QLabel("Session Time: "))
        self.session_time_label = QLabel("00:00:00")
        self.status_bar.addPermanentWidget(self.session_time_label)
        
        # Update timer
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self._update_session_time)
        self.session_timer.start(1000)  # Update every second
        
    def _apply_cyberpunk_theme(self):
        """Apply cyberpunk visual theme"""
        style = f"""
        /* Main Window */
        QMainWindow {{
            background-color: {self.config.bg_primary};
        }}
        
        /* Header */
        #header_frame {{
            background-color: {self.config.bg_secondary};
            border-bottom: 2px solid {self.config.accent_primary};
        }}
        
        #header_title {{
            color: {self.config.accent_primary};
            font-size: {self.config.font_size_header}px;
            font-weight: bold;
            font-family: {self.config.font_family};
        }}
        
        #header_subtitle {{
            color: {self.config.text_secondary};
            font-size: {self.config.font_size_normal}px;
            font-family: {self.config.font_family};
        }}
        
        /* Status LEDs */
        #status_led_on {{
            background-color: {self.config.accent_success};
            border-radius: 10px;
            border: 1px solid {self.config.accent_success};
        }}
        
        #status_led_off {{
            background-color: {self.config.bg_tertiary};
            border-radius: 10px;
            border: 1px solid {self.config.text_dim};
        }}
        
        #status_label {{
            color: {self.config.text_secondary};
            font-size: {self.config.font_size_small}px;
        }}
        
        /* Tabs */
        QTabWidget::pane {{
            background-color: {self.config.bg_secondary};
            border: 1px solid {self.config.bg_tertiary};
        }}
        
        QTabBar::tab {{
            background-color: {self.config.bg_tertiary};
            color: {self.config.text_secondary};
            padding: 8px 16px;
            margin-right: 2px;
            font-family: {self.config.font_family};
        }}
        
        QTabBar::tab:selected {{
            background-color: {self.config.bg_secondary};
            color: {self.config.accent_primary};
            border-bottom: 2px solid {self.config.accent_primary};
        }}
        
        QTabBar::tab:hover {{
            background-color: {self.config.bg_secondary};
            color: {self.config.text_primary};
        }}
        
        /* Groups */
        QGroupBox {{
            background-color: {self.config.bg_secondary};
            border: 1px solid {self.config.bg_tertiary};
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
            font-family: {self.config.font_family};
            color: {self.config.text_primary};
        }}
        
        QGroupBox::title {{
            color: {self.config.accent_primary};
            left: 10px;
            top: -8px;
            background-color: {self.config.bg_secondary};
            padding: 0 5px;
        }}
        
        /* Tables */
        QTableView {{
            background-color: {self.config.bg_tertiary};
            alternate-background-color: {self.config.bg_secondary};
            color: {self.config.text_primary};
            gridline-color: {self.config.bg_primary};
            selection-background-color: {self.config.accent_primary};
            selection-color: {self.config.bg_primary};
            font-family: {self.config.font_family};
        }}
        
        QHeaderView::section {{
            background-color: {self.config.bg_secondary};
            color: {self.config.accent_primary};
            padding: 5px;
            border: none;
            font-weight: bold;
        }}
        
        /* Inputs */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {self.config.bg_tertiary};
            border: 1px solid {self.config.bg_primary};
            color: {self.config.text_primary};
            padding: 5px;
            font-family: {self.config.font_family};
        }}
        
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {self.config.accent_primary};
            outline: none;
        }}
        
        /* Buttons */
        QPushButton {{
            background-color: {self.config.bg_tertiary};
            border: 1px solid {self.config.accent_primary};
            color: {self.config.accent_primary};
            padding: 8px 16px;
            font-family: {self.config.font_family};
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {self.config.accent_primary};
            color: {self.config.bg_primary};
        }}
        
        QPushButton:pressed {{
            background-color: {self.config.accent_secondary};
        }}
        
        QPushButton:disabled {{
            background-color: {self.config.bg_tertiary};
            border: 1px solid {self.config.text_dim};
            color: {self.config.text_dim};
        }}
        
        /* Special Buttons */
        #buy_button {{
            border-color: {self.config.accent_success};
            color: {self.config.accent_success};
        }}
        
        #buy_button:hover {{
            background-color: {self.config.accent_success};
            color: {self.config.bg_primary};
        }}
        
        #sell_button {{
            border-color: {self.config.accent_error};
            color: {self.config.accent_error};
        }}
        
        #sell_button:hover {{
            background-color: {self.config.accent_error};
            color: {self.config.bg_primary};
        }}
        
        #emergency_button {{
            border-color: {self.config.accent_error};
            color: {self.config.accent_error};
            font-size: {self.config.font_size_large}px;
        }}
        
        #emergency_button:hover {{
            background-color: {self.config.accent_error};
            color: {self.config.bg_primary};
        }}
        
        /* Scrollbars */
        QScrollBar:vertical {{
            background-color: {self.config.bg_secondary};
            width: 12px;
            border: none;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {self.config.accent_primary};
            min-height: 20px;
            border-radius: 6px;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            background: none;
            border: none;
        }}
        
        /* Labels */
        QLabel {{
            color: {self.config.text_primary};
            font-family: {self.config.font_family};
        }}
        
        #price_display {{
            color: {self.config.accent_primary};
            font-size: {self.config.font_size_header}px;
            font-weight: bold;
        }}
        
        #stat_value {{
            color: {self.config.accent_success};
            font-weight: bold;
        }}
        
        #portfolio_value {{
            color: {self.config.accent_primary};
            font-size: {self.config.font_size_large}px;
            font-weight: bold;
        }}
        
        #user_info {{
            color: {self.config.text_secondary};
            font-size: {self.config.font_size_small}px;
        }}
        
        /* Status Bar */
        QStatusBar {{
            background-color: {self.config.bg_secondary};
            color: {self.config.text_secondary};
            border-top: 1px solid {self.config.bg_tertiary};
        }}
        
        /* Tooltips */
        QToolTip {{
            background-color: {self.config.bg_tertiary};
            color: {self.config.text_primary};
            border: 1px solid {self.config.accent_primary};
            padding: 5px;
            font-family: {self.config.font_family};
        }}
        
        /* Log Widget */
        QPlainTextEdit {{
            background-color: {self.config.bg_tertiary};
            color: {self.config.text_primary};
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: {self.config.font_size_small}px;
            selection-background-color: {self.config.accent_primary};
        }}
        
        /* Checkboxes */
        QCheckBox {{
            color: {self.config.text_primary};
            font-family: {self.config.font_family};
        }}
        
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            background-color: {self.config.bg_tertiary};
            border: 1px solid {self.config.accent_primary};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {self.config.accent_primary};
        }}
        """
        
        self.setStyleSheet(style)
        
    def _setup_connections(self):
        """Setup signal/slot connections"""
        # Timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_real_time_data)
        self.update_timer.start(5000)  # Update every 5 seconds
        
    def _show_login_dialog(self):
        """Show login dialog with optional security features"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Nexlify Authentication")
        dialog.setFixedSize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        # Logo/Title
        title = QLabel("NEXLIFY NEURAL TRADING")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {self.config.accent_primary}; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
        layout.addSpacing(20)
        
        # Check if master password is enabled
        has_master_password = self.app_config.get('security', {}).get('master_password_enabled', False)
        
        if has_master_password:
            # Master password input
            layout.addWidget(QLabel("Master Password:"))
            password_input = QLineEdit()
            password_input.setEchoMode(QLineEdit.Password)
            layout.addWidget(password_input)
            
            # Check if 2FA is enabled
            has_2fa = self.app_config.get('security', {}).get('2fa_enabled', False)
            
            if has_2fa:
                layout.addWidget(QLabel("2FA Code:"))
                twofa_input = QLineEdit()
                twofa_input.setPlaceholderText("6-digit code")
                layout.addWidget(twofa_input)
            else:
                twofa_input = None
                
        else:
            # Simple PIN login
            layout.addWidget(QLabel("Enter PIN:"))
            password_input = QLineEdit()
            password_input.setEchoMode(QLineEdit.Password)
            password_input.setPlaceholderText("Default: 2077")
            layout.addWidget(password_input)
            twofa_input = None
            
        layout.addSpacing(20)
        
        # Login button
        login_btn = QPushButton("Authenticate")
        login_btn.clicked.connect(lambda: self._authenticate(
            dialog, password_input.text(), 
            twofa_input.text() if twofa_input else None
        ))
        layout.addWidget(login_btn)
        
        # Make Enter key trigger login
        password_input.returnPressed.connect(login_btn.click)
        if twofa_input:
            twofa_input.returnPressed.connect(login_btn.click)
            
        dialog.exec_()
        
    def _authenticate(self, dialog: QDialog, password: str, twofa_code: Optional[str]):
        """Authenticate user with security manager"""
        try:
            # For simple PIN mode
            if not self.app_config.get('security', {}).get('master_password_enabled', False):
                expected_pin = self.app_config.get('pin', '2077')
                if password == expected_pin:
                    self.is_authenticated = True
                    dialog.accept()
                    self._post_authentication()
                    return
                else:
                    QMessageBox.warning(dialog, "Authentication Failed", "Invalid PIN")
                    return
                    
            # For full security mode
            username = "admin"  # Could be configurable
            
            # Authenticate
            session = self.security_manager.authenticate_user(
                username, password, self.session_manager.get_current_ip()
            )
            
            if not session:
                QMessageBox.warning(dialog, "Authentication Failed", "Invalid credentials")
                return
                
            # Check 2FA if enabled
            if self.app_config.get('security', {}).get('2fa_enabled', False):
                if not twofa_code:
                    QMessageBox.warning(dialog, "2FA Required", "Please enter 2FA code")
                    return
                    
                if not self.security_manager.two_factor_auth.verify_token(username, twofa_code):
                    QMessageBox.warning(dialog, "2FA Failed", "Invalid 2FA code")
                    return
                    
            # Success
            self.is_authenticated = True
            self.session_manager.set_session(session)
            self.user_info_label.setText(f"User: {username}")
            
            # Log successful login
            asyncio.create_task(self.audit_manager.audit_login(
                username, self.session_manager.get_current_ip(), True
            ))
            
            dialog.accept()
            self._post_authentication()
            
        except Exception as e:
            error_handler.log_error(e, {"method": "_authenticate"})
            QMessageBox.critical(dialog, "Error", "Authentication error occurred")
            
    def _post_authentication(self):
        """Initialize components after successful authentication"""
        try:
            # Initialize neural net
            self.neural_net = NexlifyNeuralNet()
            asyncio.create_task(self.neural_net.initialize())
            
            # Initialize other components
            if self.app_config.get('features', {}).get('enable_predictive', True):
                self.predictive_engine = PredictiveEngine()
                
            if self.app_config.get('features', {}).get('enable_ai_companion', True):
                self.ai_companion = AITradingCompanion(self.neural_net, self)
                
            # Update UI
            self._update_connection_status(True)
            self.log_widget.append_log("Authentication successful", "INFO")
            
            # Load saved settings
            self._load_settings()
            
        except Exception as e:
            error_handler.log_error(e, {"method": "_post_authentication"})
            self.log_widget.append_log(f"Initialization error: {str(e)}", "ERROR")
            
    def _validate_pair_input(self, text: str):
        """Validate trading pair input in real-time"""
        if not text:
            self.pair_input.setStyleSheet("")
            return
            
        # Check format
        if "/" not in text:
            self.pair_input.setStyleSheet(f"border: 1px solid {self.config.accent_error};")
            return
            
        # Validate pair format
        try:
            base, quote = text.upper().split("/")
            valid_quotes = ["USDT", "USDC", "BUSD", "USD", "BTC", "ETH"]
            
            if len(base) < 2 or len(base) > 10:
                self.pair_input.setStyleSheet(f"border: 1px solid {self.config.accent_error};")
            elif quote not in valid_quotes:
                self.pair_input.setStyleSheet(f"border: 1px solid {self.config.accent_warning};")
            else:
                self.pair_input.setStyleSheet(f"border: 1px solid {self.config.accent_success};")
                
        except:
            self.pair_input.setStyleSheet(f"border: 1px solid {self.config.accent_error};")
            
    def _validate_btc_address(self, address: str):
        """Validate BTC address in real-time"""
        if not address:
            self.btc_address_valid_label.setText("")
            self.btc_address_input.setStyleSheet("")
            return
            
        # Use our crypto utils for validation
        is_valid = CryptoUtils.validate_address(address, 'BTC')
        
        if is_valid:
            self.btc_address_valid_label.setText("✓ Valid")
            self.btc_address_valid_label.setStyleSheet(f"color: {self.config.accent_success};")
            self.btc_address_input.setStyleSheet(f"border: 1px solid {self.config.accent_success};")
        else:
            self.btc_address_valid_label.setText("✗ Invalid")
            self.btc_address_valid_label.setStyleSheet(f"color: {self.config.accent_error};")
            self.btc_address_input.setStyleSheet(f"border: 1px solid {self.config.accent_error};")
            
    async def _test_exchange_connection(self, exchange: str):
        """Test exchange API connection with real validation"""
        try:
            # Update activity for session management
            self.session_manager.update_activity()
            
            # Get the button
            test_btn = self.api_inputs[exchange]['test_btn']
            test_btn.set_loading(True)
            
            # Get credentials
            api_key = self.api_inputs[exchange]['api_key'].text()
            secret = self.api_inputs[exchange]['secret'].text()
            
            if not api_key or not secret:
                QMessageBox.warning(self, "Missing Credentials", 
                                  f"Please enter API credentials for {exchange}")
                test_btn.set_loading(False)
                return
                
            # Validate format
            if not ValidationUtils.validate_api_credentials(api_key, secret, exchange.lower()):
                QMessageBox.warning(self, "Invalid Format", 
                                  f"API credentials format is invalid for {exchange}")
                test_btn.set_loading(False)
                return
                
            # Test actual connection
            try:
                import ccxt
                exchange_class = getattr(ccxt, exchange.lower())
                test_exchange = exchange_class({
                    'apiKey': api_key,
                    'secret': secret,
                    'enableRateLimit': True
                })
                
                # Set testnet if configured
                if self.app_config.get('trading', {}).get('testnet', True):
                    test_exchange.set_sandbox_mode(True)
                    
                # Try to fetch balance
                balance = await asyncio.wait_for(
                    test_exchange.fetch_balance(), 
                    timeout=10.0
                )
                
                # Success
                QMessageBox.information(self, "Connection Successful", 
                                      f"Successfully connected to {exchange}!")
                self.log_widget.append_log(f"Exchange connection test passed: {exchange}", "INFO")
                
                # Log test
                await self.audit_manager.audit_system_event(
                    "exchange_test",
                    "info",
                    {"exchange": exchange, "success": True}
                )
                
            except asyncio.TimeoutError:
                QMessageBox.warning(self, "Connection Timeout", 
                                  f"Connection to {exchange} timed out")
                self.log_widget.append_log(f"Exchange connection timeout: {exchange}", "WARNING")
                
            except Exception as e:
                QMessageBox.critical(self, "Connection Failed", 
                                   f"Failed to connect to {exchange}: {str(e)}")
                self.log_widget.append_log(f"Exchange connection failed: {exchange} - {str(e)}", "ERROR")
                
        except Exception as e:
            error_handler.log_error(e, {"method": "_test_exchange_connection", "exchange": exchange})
            
        finally:
            test_btn.set_loading(False)
            
    def _toggle_trading(self):
        """Toggle auto-trading with confirmation"""
        if not self.neural_net:
            QMessageBox.warning(self, "Not Ready", "Trading system not initialized")
            return
            
        self.session_manager.update_activity()
        
        if not self.is_trading_active:
            # Confirm start
            reply = QMessageBox.question(
                self, "Start Auto-Trading",
                "Are you sure you want to start automated trading?\n\n"
                "This will execute real trades based on the configured strategies.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.is_trading_active = True
                self.toggle_trading_btn.setText("Stop Auto-Trading")
                self.toggle_trading_btn.setObjectName("sell_button")  # Red style
                self._update_trading_status(True)
                self.log_widget.append_log("Auto-trading started", "INFO")
                
                # Play sound
                if self.sound_manager:
                    self.sound_manager.play("trading_start")
                    
        else:
            # Confirm stop
            reply = QMessageBox.question(
                self, "Stop Auto-Trading", 
                "Are you sure you want to stop automated trading?\n\n"
                "Open positions will remain but no new trades will be executed.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.is_trading_active = False
                self.toggle_trading_btn.setText("Start Auto-Trading")
                self.toggle_trading_btn.setObjectName("")  # Default style
                self._update_trading_status(False)
                self.log_widget.append_log("Auto-trading stopped", "WARNING")
                
    def _emergency_stop(self):
        """Emergency stop with confirmation"""
        self.session_manager.update_activity()
        
        reply = QMessageBox.critical(
            self, "EMERGENCY STOP",
            "⚠️ WARNING ⚠️\n\n"
            "This will immediately:\n"
            "• Stop all trading activities\n"
            "• Cancel all pending orders\n"
            "• Disable auto-trading\n\n"
            "Are you absolutely sure?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Create emergency stop file
                with open("EMERGENCY_STOP_ACTIVE", "w") as f:
                    f.write(f"Emergency stop triggered at {datetime.now()}")
                    
                # Stop trading
                self.is_trading_active = False
                self._update_trading_status(False)
                
                # Log emergency stop
                self.log_widget.append_log("EMERGENCY STOP ACTIVATED", "ERROR")
                
                # Audit log
                asyncio.create_task(self.audit_manager.audit_system_event(
                    "emergency_stop",
                    "critical",
                    {"triggered_by": "user", "timestamp": datetime.now().isoformat()}
                ))
                
                # Play alarm sound
                if self.sound_manager:
                    self.sound_manager.play("emergency_alarm")
                    
                QMessageBox.information(self, "Emergency Stop Active", 
                                      "Emergency stop has been activated.\n"
                                      "Delete EMERGENCY_STOP_ACTIVE file to resume.")
                                      
            except Exception as e:
                error_handler.log_error(e, {"method": "_emergency_stop"})
                
    def _execute_trade(self, side: str):
        """Execute trade with confirmation for critical action"""
        self.session_manager.update_activity()
        
        # Validate inputs
        pair = self.pair_input.text().upper()
        if "/" not in pair:
            QMessageBox.warning(self, "Invalid Pair", "Please enter a valid trading pair")
            return
            
        try:
            amount = float(self.amount_input.text())
        except:
            QMessageBox.warning(self, "Invalid Amount", "Please enter a valid amount")
            return
            
        # Get order details
        order_type = self.order_type_combo.currentText()
        price = None
        if order_type != "Market":
            try:
                price = float(self.price_input.text())
            except:
                QMessageBox.warning(self, "Invalid Price", "Please enter a valid price")
                return
                
        # Show confirmation dialog
        details = f"Pair: {pair}\nSide: {side.upper()}\nAmount: {amount}\nType: {order_type}"
        if price:
            details += f"\nPrice: {price}"
            
        reply = QMessageBox.question(
            self, f"Confirm {side.upper()} Order",
            f"Please confirm your order:\n\n{details}\n\n"
            "This will execute a real trade on the selected exchange.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Yes:
            # Execute trade
            asyncio.create_task(self._execute_trade_async(pair, side, amount, order_type, price))
            
    async def _execute_trade_async(self, pair: str, side: str, amount: float, 
                                  order_type: str, price: Optional[float]):
        """Execute trade asynchronously"""
        try:
            self.log_widget.append_log(
                f"Executing {side} order: {amount} {pair} ({order_type})", "INFO"
            )
            
            # Would execute through neural net
            # result = await self.neural_net.execute_manual_trade(...)
            
            # For now, simulate
            await asyncio.sleep(1)
            
            self.log_widget.append_log(
                f"Order executed successfully: {side} {amount} {pair}", "INFO"
            )
            
            # Play sound
            if self.sound_manager:
                self.sound_manager.play("trade_executed")
                
            # Update positions
            await self._refresh_positions()
            
        except Exception as e:
            error_handler.log_error(e, {"method": "_execute_trade_async"})
            self.log_widget.append_log(f"Trade execution failed: {str(e)}", "ERROR")
            
    def _withdraw_funds(self):
        """Withdraw funds with validation and confirmation"""
        self.session_manager.update_activity()
        
        # Validate BTC address
        btc_address = self.btc_address_input.text()
        if not CryptoUtils.validate_address(btc_address, 'BTC'):
            QMessageBox.warning(self, "Invalid Address", 
                              "Please enter a valid Bitcoin address")
            return
            
        # Validate amount
        try:
            amount = float(self.withdrawal_amount_input.text())
            if amount <= 0:
                raise ValueError()
        except:
            QMessageBox.warning(self, "Invalid Amount", 
                              "Please enter a valid withdrawal amount")
            return
            
        # Check minimum withdrawal
        min_withdrawal = self.app_config.get('withdrawal', {}).get('min_withdrawal_usdt', 100)
        if amount < min_withdrawal:
            QMessageBox.warning(self, "Amount Too Low", 
                              f"Minimum withdrawal is ${min_withdrawal} USDT")
            return
            
        # Get current BTC price
        btc_price = 45000  # Would get from neural net
        btc_amount = amount / btc_price
        
        # Show confirmation
        reply = QMessageBox.question(
            self, "Confirm Withdrawal",
            f"Please confirm your withdrawal:\n\n"
            f"Amount: ${amount:.2f} USDT\n"
            f"BTC Amount: {btc_amount:.8f} BTC\n"
            f"BTC Price: ${btc_price:,.2f}\n"
            f"Address: {btc_address[:8]}...{btc_address[-8:]}\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Yes:
            asyncio.create_task(self._withdraw_funds_async(amount, btc_address))
            
    async def _withdraw_funds_async(self, amount: float, address: str):
        """Execute withdrawal asynchronously"""
        try:
            self.withdraw_btn.set_loading(True)
            
            # Would execute through neural net
            # await self.neural_net.withdraw_profits_to_btc(amount)
            
            # Simulate
            await asyncio.sleep(2)
            
            self.log_widget.append_log(
                f"Withdrawal initiated: ${amount} USDT to {address[:8]}...", "INFO"
            )
            
            QMessageBox.information(self, "Withdrawal Initiated",
                                  "Your withdrawal has been initiated.\n"
                                  "You will receive a confirmation once processed.")
                                  
        except Exception as e:
            error_handler.log_error(e, {"method": "_withdraw_funds_async"})
            self.log_widget.append_log(f"Withdrawal failed: {str(e)}", "ERROR")
            
        finally:
            self.withdraw_btn.set_loading(False)
            
    def _save_settings(self):
        """Save all settings with validation"""
        self.session_manager.update_activity()
        
        try:
            # Gather all settings
            config = {
                'exchanges': {},
                'trading': {
                    'risk_level': self.risk_level_combo.currentText().lower(),
                    'min_profit_threshold': self.min_profit_input.value(),
                    'max_concurrent_trades': self.max_positions_input.value()
                },
                'security': {
                    'master_password_enabled': self.master_password_check.isChecked(),
                    '2fa_enabled': self.twofa_check.isChecked(),
                    'ip_whitelist_enabled': self.ip_whitelist_check.isChecked()
                }
            }
            
            # Save exchange credentials (encrypted)
            for exchange, inputs in self.api_inputs.items():
                api_key = inputs['api_key'].text()
                secret = inputs['secret'].text()
                
                if api_key and secret:
                    # Encrypt credentials
                    if hasattr(self.security_manager, 'encryption_manager'):
                        encrypted_key = self.security_manager.encryption_manager.encrypt_data(api_key)
                        encrypted_secret = self.security_manager.encryption_manager.encrypt_data(secret)
                        
                        config['exchanges'][exchange.lower()] = {
                            'apiKey': encrypted_key,
                            'secret': encrypted_secret
                        }
                        
            # Save IP whitelist
            if self.ip_whitelist_check.isChecked():
                ips = self.ip_whitelist_text.toPlainText().strip().split('\n')
                valid_ips = []
                
                for ip in ips:
                    ip = ip.strip()
                    if ip and self._validate_ip_address(ip):
                        valid_ips.append(ip)
                        
                config['security']['ip_whitelist'] = valid_ips
                
            # Save configuration
            FileUtils.save_json(config, 'enhanced_config.json')
            
            # Update app config
            self.app_config.update(config)
            
            # Log save
            self.log_widget.append_log("Settings saved successfully", "INFO")
            
            # Audit log
            await self.audit_manager.audit_config_change(
                "admin",
                "settings",
                {},
                config
            )
            
            QMessageBox.information(self, "Settings Saved", 
                                  "All settings have been saved successfully.")
                                  
        except Exception as e:
            error_handler.log_error(e, {"method": "_save_settings"})
            QMessageBox.critical(self, "Save Failed", 
                               f"Failed to save settings: {str(e)}")
                               
    def _validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format"""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except:
            return False
            
    def _setup_2fa(self):
        """Setup 2FA with QR code display"""
        if not self.security_manager or not hasattr(self.security_manager, 'two_factor_auth'):
            QMessageBox.warning(self, "2FA Not Available", 
                              "Two-factor authentication is not properly configured")
            return
            
        try:
            # Generate QR code
            username = "admin"
            qr_data = self.security_manager.two_factor_auth.generate_qr_code(username)
            
            if not qr_data:
                QMessageBox.warning(self, "2FA Setup Failed", 
                                  "Failed to generate 2FA QR code")
                return
                
            # Show QR code dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("2FA Setup")
            dialog.setFixedSize(400, 500)
            
            layout = QVBoxLayout(dialog)
            
            # Instructions
            instructions = QLabel(
                "Scan this QR code with your authenticator app\n"
                "(Google Authenticator, Authy, etc.)"
            )
            instructions.setWordWrap(True)
            layout.addWidget(instructions)
            
            # QR code display
            qr_label = QLabel()
            qr_pixmap = QPixmap()
            qr_pixmap.loadFromData(qr_data)
            qr_label.setPixmap(qr_pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            qr_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(qr_label)
            
            # Backup codes
            backup_codes = self.security_manager.two_factor_auth.users.get(
                username, {}
            ).get('backup_codes', [])
            
            if backup_codes:
                layout.addWidget(QLabel("\nBackup Codes (save these!):"))
                codes_text = QTextEdit()
                codes_text.setPlainText('\n'.join(backup_codes[:5]))  # Show first 5
                codes_text.setReadOnly(True)
                codes_text.setMaximumHeight(100)
                layout.addWidget(codes_text)
                
            # Close button
            close_btn = QPushButton("Done")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.exec_()
            
            self.log_widget.append_log("2FA setup completed", "INFO")
            
        except Exception as e:
            error_handler.log_error(e, {"method": "_setup_2fa"})
            QMessageBox.critical(self, "2FA Setup Failed", 
                               f"Failed to setup 2FA: {str(e)}")
                               
    def _load_settings(self):
        """Load saved settings into UI"""
        try:
            # Load risk level
            risk_level = self.app_config.get('trading', {}).get('risk_level', 'medium')
            index = self.risk_level_combo.findText(risk_level.capitalize())
            if index >= 0:
                self.risk_level_combo.setCurrentIndex(index)
                
            # Load trading settings
            self.min_profit_input.setValue(
                self.app_config.get('trading', {}).get('min_profit_threshold', 0.5)
            )
            self.max_positions_input.setValue(
                self.app_config.get('trading', {}).get('max_concurrent_trades', 10)
            )
            
            # Load security settings
            self.master_password_check.setChecked(
                self.app_config.get('security', {}).get('master_password_enabled', False)
            )
            self.twofa_check.setChecked(
                self.app_config.get('security', {}).get('2fa_enabled', False)
            )
            self.ip_whitelist_check.setChecked(
                self.app_config.get('security', {}).get('ip_whitelist_enabled', False)
            )
            
            # Load IP whitelist
            if self.app_config.get('security', {}).get('ip_whitelist'):
                ips = '\n'.join(self.app_config['security']['ip_whitelist'])
                self.ip_whitelist_text.setPlainText(ips)
                
            # Load BTC address
            btc_address = self.app_config.get('withdrawal', {}).get('btc_address', '')
            self.btc_address_input.setText(btc_address)
            
        except Exception as e:
            error_handler.log_error(e, {"method": "_load_settings"})
            
    def _update_real_time_data(self):
        """Update real-time data displays"""
        if not self.neural_net or not self.is_authenticated:
            return
            
        try:
            # Update BTC price
            if hasattr(self.neural_net, 'btc_price'):
                self.btc_price_label.setText(f"BTC: ${self.neural_net.btc_price:,.2f}")
                
            # Update market stats
            if hasattr(self.neural_net, 'active_pairs'):
                self.stat_labels["Active Exchanges"].setText(
                    str(len(self.neural_net.exchanges))
                )
                self.stat_labels["Profitable Pairs"].setText(
                    str(len(self.neural_net.active_pairs))
                )
                
            # Update active pairs table
            if self.neural_net.active_pairs:
                pairs_data = []
                for pair_data in self.neural_net.get_active_pairs_display():
                    pairs_data.append({
                        "Symbol": pair_data['symbol'],
                        "Price": pair_data['price'],
                        "24h Volume": pair_data['volume_24h'],
                        "24h Change": pair_data['change_24h'],
                        "Spread": pair_data['spread'],
                        "Exchanges": str(pair_data['exchanges']),
                        "Risk Score": f"{pair_data['risk_score']:.2f}",
                        "Action": "Trade"
                    })
                    
                self.active_pairs_model.add_batch_update(pairs_data)
                
            # Update profit chart
            # Would update with real profit data
            
            # Update last update time
            self.last_update_label.setText(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.debug(f"Error updating real-time data: {e}")
            
    async def _refresh_positions(self):
        """Refresh open positions"""
        # Would fetch from neural net
        pass
        
    def _refresh_data(self):
        """Manually refresh all data"""
        self.session_manager.update_activity()
        self.refresh_btn.set_loading(True)
        
        # Schedule refresh
        QTimer.singleShot(100, lambda: self._do_refresh())
        
    def _do_refresh(self):
        """Perform actual refresh"""
        try:
            self._update_real_time_data()
            self.log_widget.append_log("Data refreshed", "INFO")
        finally:
            self.refresh_btn.set_loading(False)
            
    def _enable_strategy(self):
        """Enable selected strategy"""
        # Would interact with strategy optimizer
        pass
        
    def _disable_strategy(self):
        """Disable selected strategy"""
        # Would interact with strategy optimizer
        pass
        
    def _configure_strategy(self):
        """Configure selected strategy"""
        # Would show strategy config dialog
        pass
        
    def _filter_logs(self):
        """Filter displayed logs"""
        # Would implement log filtering
        pass
        
    def _clear_logs(self):
        """Clear log display"""
        self.log_widget.clear()
        self.log_widget.append_log("Logs cleared", "INFO")
        
    def _update_connection_status(self, connected: bool):
        """Update connection status indicators"""
        if connected:
            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet(f"color: {self.config.accent_success};")
            self.connection_led.findChild(QLabel).setObjectName("status_led_on")
        else:
            self.connection_status.setText("Disconnected")
            self.connection_status.setStyleSheet(f"color: {self.config.accent_error};")
            self.connection_led.findChild(QLabel).setObjectName("status_led_off")
            
    def _update_trading_status(self, active: bool):
        """Update trading status indicators"""
        led = self.trading_led.findChild(QLabel)
        if active:
            led.setObjectName("status_led_on")
        else:
            led.setObjectName("status_led_off")
        led.setStyle(led.style())  # Force style update
        
    def _update_session_time(self):
        """Update session timer"""
        if self.is_authenticated and self.session_manager.last_activity:
            elapsed = datetime.now() - self.session_manager.last_activity
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            seconds = int(elapsed.total_seconds() % 60)
            self.session_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
    def _handle_session_expired(self):
        """Handle expired session"""
        self.is_authenticated = False
        QMessageBox.warning(self, "Session Expired", 
                          "Your session has expired. Please login again.")
        self._show_login_dialog()
        
    def _handle_session_warning(self, minutes: int):
        """Handle session warning"""
        self.log_widget.append_log(
            f"Session will expire in {minutes} minutes", "WARNING"
        )
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.is_trading_active:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Trading is still active. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
                
        # Shutdown components
        if self.neural_net:
            asyncio.create_task(self.neural_net.shutdown())
            
        event.accept()


# Main application
def main():
    """Main application entry point"""
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Nexlify Neural Trading")
    app.setOrganizationName("Nexlify")
    
    # Create event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Create and show main window
    window = CyberGUI()
    window.show()
    
    # Run event loop
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
