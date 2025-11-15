"""
ML/RL Training UI

Dedicated user interface for training and validating trading agents with
walk-forward validation. Provides real-time monitoring, configuration
controls, and performance visualization.
"""

import sys
import json
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from nexlify.security.api_key_manager import APIKeyManager, get_api_key_manager

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
        QTextEdit, QGroupBox, QProgressBar, QTabWidget, QCheckBox,
        QLineEdit, QFileDialog, QMessageBox, QGridLayout
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QColor, QPalette
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)


# Cyberpunk color scheme
COLORS = {
    'bg': '#0a0e27',
    'card_bg': '#1a1f3a',
    'text': '#00ff9f',
    'accent': '#ff00ff',
    'profit': '#00ff9f',
    'loss': '#ff0080',
    'warning': '#ffaa00',
    'neutral': '#4a5568'
}


class TrainingWorker(QThread):
    """Worker thread for running training without blocking UI"""

    progress_updated = pyqtSignal(str, float)  # message, progress_pct
    training_complete = pyqtSignal(dict)  # results dict
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.should_stop = False

    def run(self):
        """Run training in background thread"""
        try:
            # Import here to avoid circular imports
            from nexlify.training.walk_forward_trainer import WalkForwardTrainer

            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create trainer with progress callback
            trainer = WalkForwardTrainer(
                config=self.config,
                progress_callback=self._on_progress
            )

            # Run training
            results = loop.run_until_complete(trainer.train())

            # Emit completion signal
            self.training_complete.emit(results.to_dict())

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

    def _on_progress(self, message: str, progress: float):
        """Handle progress updates from trainer"""
        if not self.should_stop:
            self.progress_updated.emit(message, progress)

    def stop(self):
        """Request training to stop"""
        self.should_stop = True


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas widget for embedding plots"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=COLORS['bg'])
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor(COLORS['card_bg'])

        # Style axes
        self.axes.tick_params(colors=COLORS['text'])
        self.axes.spines['bottom'].set_color(COLORS['text'])
        self.axes.spines['top'].set_color(COLORS['text'])
        self.axes.spines['left'].set_color(COLORS['text'])
        self.axes.spines['right'].set_color(COLORS['text'])
        self.axes.xaxis.label.set_color(COLORS['text'])
        self.axes.yaxis.label.set_color(COLORS['text'])

        super().__init__(self.fig)
        self.setParent(parent)

    def clear_plot(self):
        """Clear the plot"""
        self.axes.clear()
        self.axes.set_facecolor(COLORS['card_bg'])
        self.draw()


class TrainingUI(QMainWindow):
    """
    Main training UI window

    Features:
    - Configuration panel for walk-forward parameters
    - Real-time progress monitoring
    - Performance visualization
    - Training history
    - Model management
    """

    def __init__(self):
        super().__init__()

        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 required for training UI")

        self.config = self.load_config()
        self.training_worker: Optional[TrainingWorker] = None
        self.training_history: list = []
        self.current_results: Optional[Dict[str, Any]] = None
        self.api_key_manager: Optional[APIKeyManager] = None

        self.init_ui()
        self.apply_theme()
        self.init_api_manager()

        logger.info("TrainingUI initialized")

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path('config/neural_config.json')

        if not config_path.exists():
            logger.warning("Config file not found, using defaults")
            return {
                'walk_forward': {
                    'enabled': True,
                    'total_episodes': 2000,
                    'train_size': 1000,
                    'test_size': 200,
                    'step_size': 200,
                    'mode': 'rolling',
                    'min_train_size': 500,
                    'save_models': True,
                    'risk_free_rate': 0.02
                },
                'rl_agent': {},
                'trading': {}
            }

        with open(config_path) as f:
            return json.load(f)

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Nexlify ML/RL Training Dashboard")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Configuration
        left_panel = self.create_config_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel: Monitoring and results
        right_panel = self.create_monitoring_panel()
        main_layout.addWidget(right_panel, stretch=2)

    def create_config_panel(self) -> QWidget:
        """Create configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Training Configuration")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Walk-Forward Parameters
        wf_group = self.create_walk_forward_config()
        layout.addWidget(wf_group)

        # RL Agent Parameters
        rl_group = self.create_rl_config()
        layout.addWidget(rl_group)

        # API Settings
        api_group = self.create_api_config()
        layout.addWidget(api_group)

        # Control buttons
        control_layout = QVBoxLayout()

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        self.save_config_btn = QPushButton("Save Configuration")
        self.save_config_btn.clicked.connect(self.save_configuration)
        control_layout.addWidget(self.save_config_btn)

        self.load_config_btn = QPushButton("Load Configuration")
        self.load_config_btn.clicked.connect(self.load_configuration_file)
        control_layout.addWidget(self.load_config_btn)

        layout.addLayout(control_layout)
        layout.addStretch()

        return panel

    def create_walk_forward_config(self) -> QGroupBox:
        """Create walk-forward configuration group"""
        group = QGroupBox("Walk-Forward Validation")
        layout = QGridLayout()

        wf_config = self.config.get('walk_forward', {})

        # Total Episodes
        layout.addWidget(QLabel("Total Episodes:"), 0, 0)
        self.total_episodes_spin = QSpinBox()
        self.total_episodes_spin.setRange(100, 100000)
        self.total_episodes_spin.setValue(wf_config.get('total_episodes', 2000))
        self.total_episodes_spin.setSingleStep(100)
        layout.addWidget(self.total_episodes_spin, 0, 1)

        # Train Size
        layout.addWidget(QLabel("Train Size:"), 1, 0)
        self.train_size_spin = QSpinBox()
        self.train_size_spin.setRange(100, 50000)
        self.train_size_spin.setValue(wf_config.get('train_size', 1000))
        self.train_size_spin.setSingleStep(100)
        layout.addWidget(self.train_size_spin, 1, 1)

        # Test Size
        layout.addWidget(QLabel("Test Size:"), 2, 0)
        self.test_size_spin = QSpinBox()
        self.test_size_spin.setRange(50, 10000)
        self.test_size_spin.setValue(wf_config.get('test_size', 200))
        self.test_size_spin.setSingleStep(50)
        layout.addWidget(self.test_size_spin, 2, 1)

        # Step Size
        layout.addWidget(QLabel("Step Size:"), 3, 0)
        self.step_size_spin = QSpinBox()
        self.step_size_spin.setRange(50, 5000)
        self.step_size_spin.setValue(wf_config.get('step_size', 200))
        self.step_size_spin.setSingleStep(50)
        layout.addWidget(self.step_size_spin, 3, 1)

        # Mode
        layout.addWidget(QLabel("Mode:"), 4, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['rolling', 'expanding'])
        self.mode_combo.setCurrentText(wf_config.get('mode', 'rolling'))
        layout.addWidget(self.mode_combo, 4, 1)

        # Save Models
        self.save_models_check = QCheckBox("Save Models")
        self.save_models_check.setChecked(wf_config.get('save_models', True))
        layout.addWidget(self.save_models_check, 5, 0, 1, 2)

        # Risk-free rate
        layout.addWidget(QLabel("Risk-Free Rate:"), 6, 0)
        self.risk_free_spin = QDoubleSpinBox()
        self.risk_free_spin.setRange(0.0, 0.1)
        self.risk_free_spin.setValue(wf_config.get('risk_free_rate', 0.02))
        self.risk_free_spin.setSingleStep(0.01)
        self.risk_free_spin.setDecimals(3)
        layout.addWidget(self.risk_free_spin, 6, 1)

        group.setLayout(layout)
        return group

    def create_rl_config(self) -> QGroupBox:
        """Create RL agent configuration group"""
        group = QGroupBox("RL Agent Parameters")
        layout = QGridLayout()

        rl_config = self.config.get('rl_agent', {})

        # Learning Rate
        layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(rl_config.get('learning_rate', 0.001))
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(5)
        layout.addWidget(self.lr_spin, 0, 1)

        # Discount Factor
        layout.addWidget(QLabel("Discount Factor:"), 1, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.9, 0.999)
        self.gamma_spin.setValue(rl_config.get('discount_factor', 0.99))
        self.gamma_spin.setSingleStep(0.001)
        self.gamma_spin.setDecimals(3)
        layout.addWidget(self.gamma_spin, 1, 1)

        # Batch Size
        layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 512)
        self.batch_size_spin.setValue(rl_config.get('batch_size', 64))
        self.batch_size_spin.setSingleStep(16)
        layout.addWidget(self.batch_size_spin, 2, 1)

        # Architecture
        layout.addWidget(QLabel("Architecture:"), 3, 0)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(['tiny', 'small', 'medium', 'large', 'xlarge'])
        self.arch_combo.setCurrentText(rl_config.get('default_architecture', 'medium'))
        layout.addWidget(self.arch_combo, 3, 1)

        group.setLayout(layout)
        return group

    def create_api_config(self) -> QGroupBox:
        """Create API configuration group"""
        group = QGroupBox("Exchange API Settings")
        layout = QGridLayout()

        # Exchange selection
        layout.addWidget(QLabel("Exchange:"), 0, 0)
        self.exchange_combo = QComboBox()
        self.exchange_combo.addItems([
            'binance', 'kraken', 'coinbase', 'bitfinex', 'bitstamp',
            'gemini', 'poloniex', 'kucoin', 'huobi', 'okx'
        ])
        self.exchange_combo.currentTextChanged.connect(self.on_exchange_changed)
        layout.addWidget(self.exchange_combo, 0, 1)

        # API Key
        layout.addWidget(QLabel("API Key:"), 1, 0)
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter API key...")
        layout.addWidget(self.api_key_input, 1, 1)

        # Secret Key
        layout.addWidget(QLabel("Secret:"), 2, 0)
        self.secret_input = QLineEdit()
        self.secret_input.setEchoMode(QLineEdit.Password)
        self.secret_input.setPlaceholderText("Enter secret...")
        layout.addWidget(self.secret_input, 2, 1)

        # Testnet checkbox
        self.testnet_check = QCheckBox("Use Testnet")
        self.testnet_check.setChecked(False)
        layout.addWidget(self.testnet_check, 3, 0, 1, 2)

        # Buttons
        button_layout = QHBoxLayout()

        self.test_connection_btn = QPushButton("Test Connection")
        self.test_connection_btn.clicked.connect(self.test_api_connection)
        button_layout.addWidget(self.test_connection_btn)

        self.save_api_btn = QPushButton("Save API Keys")
        self.save_api_btn.clicked.connect(self.save_api_keys)
        button_layout.addWidget(self.save_api_btn)

        layout.addLayout(button_layout, 4, 0, 1, 2)

        # Status label
        self.api_status_label = QLabel("No API keys loaded")
        self.api_status_label.setWordWrap(True)
        layout.addWidget(self.api_status_label, 5, 0, 1, 2)

        group.setLayout(layout)
        return group

    def init_api_manager(self):
        """Initialize API key manager"""
        try:
            # For now, use a default password
            # In production, this should be user's PIN or secure password
            password = "nexlify_default_password_change_me"

            self.api_key_manager = APIKeyManager(password)

            # Load API keys for current exchange
            self.on_exchange_changed(self.exchange_combo.currentText())

        except Exception as e:
            logger.error(f"Failed to initialize API key manager: {e}")
            self.api_status_label.setText(f"Error: {str(e)}")

    def on_exchange_changed(self, exchange: str):
        """Handle exchange selection change"""
        if not self.api_key_manager:
            return

        # Load API keys for selected exchange
        keys = self.api_key_manager.get_api_key(exchange)

        if keys:
            self.api_key_input.setText(keys['api_key'])
            self.secret_input.setText(keys['secret'])
            self.testnet_check.setChecked(keys.get('testnet', False))
            self.api_status_label.setText(f"✓ API keys loaded for {exchange}")
            self.api_status_label.setStyleSheet(f"color: {COLORS['profit']};")
        else:
            self.api_key_input.clear()
            self.secret_input.clear()
            self.testnet_check.setChecked(False)
            self.api_status_label.setText(f"No API keys stored for {exchange}")
            self.api_status_label.setStyleSheet(f"color: {COLORS['neutral']};")

    def test_api_connection(self):
        """Test API connection to selected exchange"""
        exchange = self.exchange_combo.currentText()
        api_key = self.api_key_input.text()
        secret = self.secret_input.text()

        if not api_key or not secret:
            QMessageBox.warning(
                self,
                "Missing Credentials",
                "Please enter both API key and secret"
            )
            return

        # Save temporarily to test
        testnet = self.testnet_check.isChecked()
        if self.api_key_manager:
            self.api_key_manager.add_api_key(exchange, api_key, secret, testnet)

        # Update status
        self.api_status_label.setText(f"Testing connection to {exchange}...")
        self.api_status_label.setStyleSheet(f"color: {COLORS['warning']};")
        self.test_connection_btn.setEnabled(False)

        # Run test in background
        QTimer.singleShot(100, lambda: self._run_connection_test(exchange))

    def _run_connection_test(self, exchange: str):
        """Run connection test asynchronously"""
        try:
            # Create event loop for async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if self.api_key_manager:
                success, message = loop.run_until_complete(
                    self.api_key_manager.test_connection(exchange)
                )
            else:
                success, message = False, "API key manager not initialized"

            # Update UI
            if success:
                self.api_status_label.setText(f"✓ {message}")
                self.api_status_label.setStyleSheet(f"color: {COLORS['profit']};")
                QMessageBox.information(
                    self,
                    "Connection Successful",
                    f"Successfully connected to {exchange}!"
                )
            else:
                self.api_status_label.setText(f"✗ {message}")
                self.api_status_label.setStyleSheet(f"color: {COLORS['loss']};")
                QMessageBox.warning(
                    self,
                    "Connection Failed",
                    f"Failed to connect to {exchange}:\n\n{message}"
                )

        except Exception as e:
            self.api_status_label.setText(f"Error: {str(e)}")
            self.api_status_label.setStyleSheet(f"color: {COLORS['loss']};")
            QMessageBox.critical(
                self,
                "Test Error",
                f"Error testing connection:\n\n{str(e)}"
            )

        finally:
            self.test_connection_btn.setEnabled(True)

    def save_api_keys(self):
        """Save API keys to encrypted storage"""
        exchange = self.exchange_combo.currentText()
        api_key = self.api_key_input.text()
        secret = self.secret_input.text()

        if not api_key or not secret:
            QMessageBox.warning(
                self,
                "Missing Credentials",
                "Please enter both API key and secret"
            )
            return

        try:
            testnet = self.testnet_check.isChecked()

            if self.api_key_manager:
                self.api_key_manager.add_api_key(exchange, api_key, secret, testnet)

                self.api_status_label.setText(f"✓ API keys saved for {exchange}")
                self.api_status_label.setStyleSheet(f"color: {COLORS['profit']};")

                QMessageBox.information(
                    self,
                    "Keys Saved",
                    f"API keys for {exchange} have been saved to encrypted storage."
                )
            else:
                raise Exception("API key manager not initialized")

        except Exception as e:
            self.api_status_label.setText(f"Error: {str(e)}")
            self.api_status_label.setStyleSheet(f"color: {COLORS['loss']};")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save API keys:\n\n{str(e)}"
            )

    def create_monitoring_panel(self) -> QWidget:
        """Create monitoring and results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tab widget for different views
        tabs = QTabWidget()

        # Progress tab
        progress_tab = self.create_progress_tab()
        tabs.addTab(progress_tab, "Training Progress")

        # Results tab
        results_tab = self.create_results_tab()
        tabs.addTab(results_tab, "Results & Metrics")

        # Visualization tab
        viz_tab = self.create_visualization_tab()
        tabs.addTab(viz_tab, "Performance Charts")

        # History tab
        history_tab = self.create_history_tab()
        tabs.addTab(history_tab, "Training History")

        layout.addWidget(tabs)

        return panel

    def create_progress_tab(self) -> QWidget:
        """Create training progress tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to train")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Log output
        layout.addWidget(QLabel("Training Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)

        return tab

    def create_results_tab(self) -> QWidget:
        """Create results and metrics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Summary metrics
        metrics_group = QGroupBox("Summary Metrics")
        metrics_layout = QGridLayout()

        self.metrics_labels = {}
        metrics = [
            ('Total Return', 'total_return'),
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Win Rate', 'win_rate'),
            ('Max Drawdown', 'max_drawdown'),
            ('Profit Factor', 'profit_factor'),
            ('Sortino Ratio', 'sortino_ratio')
        ]

        for i, (label, key) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2

            metrics_layout.addWidget(QLabel(f"{label}:"), row, col)
            value_label = QLabel("--")
            value_label.setFont(QFont("Arial", 12, QFont.Bold))
            metrics_layout.addWidget(value_label, row, col + 1)
            self.metrics_labels[key] = value_label

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Fold breakdown
        layout.addWidget(QLabel("Fold Results:"))
        self.fold_results_text = QTextEdit()
        self.fold_results_text.setReadOnly(True)
        self.fold_results_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.fold_results_text)

        return tab

    def create_visualization_tab(self) -> QWidget:
        """Create visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if not MATPLOTLIB_AVAILABLE:
            layout.addWidget(QLabel("Matplotlib not available for visualization"))
            return tab

        # Create matplotlib canvases
        self.performance_canvas = MatplotlibCanvas(tab, width=8, height=4)
        layout.addWidget(self.performance_canvas)

        self.metrics_canvas = MatplotlibCanvas(tab, width=8, height=4)
        layout.addWidget(self.metrics_canvas)

        return tab

    def create_history_tab(self) -> QWidget:
        """Create training history tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Training History:"))

        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.history_text)

        # Buttons
        btn_layout = QHBoxLayout()

        load_history_btn = QPushButton("Load History")
        load_history_btn.clicked.connect(self.load_history)
        btn_layout.addWidget(load_history_btn)

        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        btn_layout.addWidget(clear_history_btn)

        layout.addLayout(btn_layout)

        return tab

    def apply_theme(self):
        """Apply cyberpunk theme to UI"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(COLORS['bg']))
        palette.setColor(QPalette.WindowText, QColor(COLORS['text']))
        palette.setColor(QPalette.Base, QColor(COLORS['card_bg']))
        palette.setColor(QPalette.AlternateBase, QColor(COLORS['bg']))
        palette.setColor(QPalette.Text, QColor(COLORS['text']))
        palette.setColor(QPalette.Button, QColor(COLORS['card_bg']))
        palette.setColor(QPalette.ButtonText, QColor(COLORS['text']))
        palette.setColor(QPalette.Highlight, QColor(COLORS['accent']))
        palette.setColor(QPalette.HighlightedText, QColor(COLORS['bg']))

        self.setPalette(palette)

        # Style sheets for specific widgets
        self.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {COLORS['text']};
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: {COLORS['text']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }}
            QPushButton {{
                background-color: {COLORS['card_bg']};
                color: {COLORS['text']};
                border: 2px solid {COLORS['text']};
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['text']};
                color: {COLORS['bg']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['accent']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['neutral']};
                color: {COLORS['card_bg']};
                border-color: {COLORS['neutral']};
            }}
            QProgressBar {{
                border: 2px solid {COLORS['text']};
                border-radius: 5px;
                text-align: center;
                color: {COLORS['text']};
                background-color: {COLORS['card_bg']};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['profit']};
            }}
        """)

    def get_config_from_ui(self) -> Dict[str, Any]:
        """Get configuration from UI controls"""
        config = self.config.copy()

        # Update walk-forward config
        config['walk_forward'] = {
            'enabled': True,
            'total_episodes': self.total_episodes_spin.value(),
            'train_size': self.train_size_spin.value(),
            'test_size': self.test_size_spin.value(),
            'step_size': self.step_size_spin.value(),
            'mode': self.mode_combo.currentText(),
            'save_models': self.save_models_check.isChecked(),
            'risk_free_rate': self.risk_free_spin.value(),
            'min_train_size': 500,
            'output_dir': 'reports/walk_forward',
            'model_dir': 'models/walk_forward',
            'integration': {
                'select_best_model': True,
                'validation_metric': 'sharpe_ratio'
            }
        }

        # Update RL config
        config['rl_agent'] = config.get('rl_agent', {})
        config['rl_agent'].update({
            'learning_rate': self.lr_spin.value(),
            'discount_factor': self.gamma_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'default_architecture': self.arch_combo.currentText()
        })

        return config

    def start_training(self):
        """Start training process"""
        # Get configuration from UI
        config = self.get_config_from_ui()

        # Validate configuration
        if config['walk_forward']['train_size'] + config['walk_forward']['test_size'] > config['walk_forward']['total_episodes']:
            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Train size + test size cannot exceed total episodes!"
            )
            return

        # Reset UI
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting training...")
        self.log("Training started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Disable start button, enable stop button
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Create and start worker
        self.training_worker = TrainingWorker(config)
        self.training_worker.progress_updated.connect(self.on_progress)
        self.training_worker.training_complete.connect(self.on_training_complete)
        self.training_worker.error_occurred.connect(self.on_training_error)
        self.training_worker.start()

    def stop_training(self):
        """Stop training process"""
        if self.training_worker:
            self.training_worker.stop()
            self.log("Training stop requested...")
            self.status_label.setText("Stopping training...")

    def on_progress(self, message: str, progress: float):
        """Handle progress updates"""
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(message)
        self.log(f"[{progress:.1f}%] {message}")

    def on_training_complete(self, results: Dict[str, Any]):
        """Handle training completion"""
        self.current_results = results
        self.status_label.setText("Training complete!")
        self.progress_bar.setValue(100)
        self.log("Training completed successfully")

        # Update results display
        self.display_results(results)

        # Add to history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        self.update_history_display()

        # Re-enable buttons
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        QMessageBox.information(
            self,
            "Training Complete",
            f"Walk-forward validation completed with {len(results['fold_metrics'])} folds.\n"
            f"Mean Sharpe Ratio: {results['mean_metrics']['sharpe_ratio']:.2f}"
        )

    def on_training_error(self, error_msg: str):
        """Handle training errors"""
        self.status_label.setText("Training failed!")
        self.log(f"ERROR: {error_msg}")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        QMessageBox.critical(
            self,
            "Training Error",
            f"Training failed with error:\n{error_msg}"
        )

    def display_results(self, results: Dict[str, Any]):
        """Display training results"""
        # Update metric labels
        mean_metrics = results['mean_metrics']

        for key, label in self.metrics_labels.items():
            if key in mean_metrics:
                value = mean_metrics[key]
                std = results['std_metrics'].get(key, 0)

                # Format based on metric type
                if 'rate' in key or 'return' in key or 'drawdown' in key:
                    text = f"{value:.2%} ± {std:.2%}"
                else:
                    text = f"{value:.2f} ± {std:.2f}"

                label.setText(text)

                # Color code based on value
                if value > 0 and key != 'max_drawdown':
                    label.setStyleSheet(f"color: {COLORS['profit']}")
                elif value < 0:
                    label.setStyleSheet(f"color: {COLORS['loss']}")

        # Update fold results
        fold_text = "Fold Results:\n" + "="*60 + "\n\n"
        for fold_metrics in results['fold_metrics']:
            fold_text += f"Fold {fold_metrics['fold_id']}:\n"
            fold_text += f"  Return: {fold_metrics['total_return']:.2%}\n"
            fold_text += f"  Sharpe: {fold_metrics['sharpe_ratio']:.2f}\n"
            fold_text += f"  Win Rate: {fold_metrics['win_rate']:.2%}\n"
            fold_text += f"  Max DD: {fold_metrics['max_drawdown']:.2%}\n\n"

        self.fold_results_text.setText(fold_text)

        # Update visualizations
        if MATPLOTLIB_AVAILABLE:
            self.plot_performance(results)
            self.plot_metrics(results)

    def plot_performance(self, results: Dict[str, Any]):
        """Plot performance across folds"""
        self.performance_canvas.clear_plot()
        ax = self.performance_canvas.axes

        fold_ids = [fm['fold_id'] for fm in results['fold_metrics']]
        returns = [fm['total_return'] * 100 for fm in results['fold_metrics']]

        ax.bar(fold_ids, returns, color=COLORS['profit'], alpha=0.7)
        ax.axhline(results['mean_metrics']['total_return'] * 100,
                   color=COLORS['accent'], linestyle='--', label='Mean')
        ax.set_xlabel('Fold ID', color=COLORS['text'])
        ax.set_ylabel('Total Return (%)', color=COLORS['text'])
        ax.set_title('Returns per Fold', color=COLORS['text'])
        ax.legend(facecolor=COLORS['card_bg'], edgecolor=COLORS['text'])
        ax.grid(True, alpha=0.3, color=COLORS['neutral'])

        self.performance_canvas.draw()

    def plot_metrics(self, results: Dict[str, Any]):
        """Plot key metrics comparison"""
        self.metrics_canvas.clear_plot()
        ax = self.metrics_canvas.axes

        metrics = ['sharpe_ratio', 'win_rate', 'profit_factor']
        values = [results['mean_metrics'][m] for m in metrics]
        labels = ['Sharpe', 'Win Rate', 'Profit Factor']

        bars = ax.bar(labels, values, color=[COLORS['profit'], COLORS['accent'], COLORS['text']], alpha=0.7)

        ax.set_ylabel('Value', color=COLORS['text'])
        ax.set_title('Key Metrics Summary', color=COLORS['text'])
        ax.grid(True, alpha=0.3, color=COLORS['neutral'], axis='y')

        self.metrics_canvas.draw()

    def log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def save_configuration(self):
        """Save current configuration to file"""
        config = self.get_config_from_ui()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "config/training_config.json",
            "JSON Files (*.json)"
        )

        if file_path:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Success", f"Configuration saved to {file_path}")

    def load_configuration_file(self):
        """Load configuration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "config/",
            "JSON Files (*.json)"
        )

        if file_path:
            with open(file_path) as f:
                config = json.load(f)

            self.config = config
            self.update_ui_from_config(config)
            QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")

    def update_ui_from_config(self, config: Dict[str, Any]):
        """Update UI controls from configuration"""
        wf = config.get('walk_forward', {})
        self.total_episodes_spin.setValue(wf.get('total_episodes', 2000))
        self.train_size_spin.setValue(wf.get('train_size', 1000))
        self.test_size_spin.setValue(wf.get('test_size', 200))
        self.step_size_spin.setValue(wf.get('step_size', 200))
        self.mode_combo.setCurrentText(wf.get('mode', 'rolling'))
        self.save_models_check.setChecked(wf.get('save_models', True))
        self.risk_free_spin.setValue(wf.get('risk_free_rate', 0.02))

        rl = config.get('rl_agent', {})
        self.lr_spin.setValue(rl.get('learning_rate', 0.001))
        self.gamma_spin.setValue(rl.get('discount_factor', 0.99))
        self.batch_size_spin.setValue(rl.get('batch_size', 64))
        self.arch_combo.setCurrentText(rl.get('default_architecture', 'medium'))

    def update_history_display(self):
        """Update training history display"""
        history_text = "Training History:\n" + "="*60 + "\n\n"

        for i, entry in enumerate(self.training_history[-10:]):  # Show last 10
            history_text += f"Training Run {i+1}:\n"
            history_text += f"  Time: {entry['timestamp']}\n"

            results = entry['results']
            history_text += f"  Folds: {len(results['fold_metrics'])}\n"
            history_text += f"  Mean Return: {results['mean_metrics']['total_return']:.2%}\n"
            history_text += f"  Mean Sharpe: {results['mean_metrics']['sharpe_ratio']:.2f}\n\n"

        self.history_text.setText(history_text)

    def load_history(self):
        """Load training history from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Training History",
            "training_logs/",
            "JSON Files (*.json)"
        )

        if file_path:
            with open(file_path) as f:
                data = json.load(f)

            if 'validation_results' in data:
                self.current_results = data['validation_results']
                self.display_results(self.current_results)

                QMessageBox.information(self, "Success", f"History loaded from {file_path}")

    def clear_history(self):
        """Clear training history"""
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear the training history?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.training_history.clear()
            self.history_text.clear()


def main():
    """Main entry point for training UI"""
    if not PYQT_AVAILABLE:
        print("Error: PyQt5 is required for the training UI")
        print("Install with: pip install PyQt5")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = TrainingUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
