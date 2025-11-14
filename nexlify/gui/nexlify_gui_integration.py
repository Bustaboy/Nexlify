#!/usr/bin/env python3
"""
Nexlify GUI Integration for Phase 1 & 2 Features
Adds user-accessible controls for all new security and financial features
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from nexlify.financial.nexlify_defi_integration import DeFiIntegration
from nexlify.financial.nexlify_profit_manager import (ProfitManager,
                                                      WithdrawalDestination,
                                                      WithdrawalFrequency,
                                                      WithdrawalStrategy)
from nexlify.financial.nexlify_tax_reporter import TaxReporter
from nexlify.risk.nexlify_emergency_kill_switch import KillSwitchTrigger
# Import Phase 1 & 2 modules
from nexlify.security.nexlify_security_suite import SecuritySuite

logger = logging.getLogger(__name__)


class PINAuthDialog(QDialog):
    """PIN Authentication Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔐 Nexlify Authentication")
        self.setModal(True)
        self.setFixedSize(400, 250)
        self.pin_value = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Enter your PIN to access Nexlify")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #1e293b;")
        layout.addWidget(title)

        # PIN input
        self.pin_input = QLineEdit()
        self.pin_input.setEchoMode(QLineEdit.Password)
        self.pin_input.setPlaceholderText("Enter PIN")
        self.pin_input.setAlignment(Qt.AlignCenter)
        self.pin_input.setStyleSheet(
            """
            QLineEdit {
                font-size: 24px;
                padding: 10px;
                border: 2px solid #2563eb;
                border-radius: 6px;
                background: #ffffff;
                color: #1e293b;
            }
        """
        )
        self.pin_input.returnPressed.connect(self.validate_pin)
        layout.addWidget(self.pin_input)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #ef4444; font-size: 12px;")
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.validate_pin)
        self.login_btn.setStyleSheet(
            """
            QPushButton {
                background: #2563eb;
                color: white;
                font-weight: 500;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #3b82f6;
            }
        """
        )
        btn_layout.addWidget(self.login_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet(
            """
            QPushButton {
                background: #e8ecf1;
                color: #1e293b;
                padding: 10px;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #cbd5e1;
            }
        """
        )
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.pin_input.setFocus()

    def validate_pin(self):
        self.pin_value = self.pin_input.text()
        if self.pin_value:
            self.accept()
        else:
            self.status_label.setText("Please enter a PIN")


class EmergencyKillSwitchWidget(QWidget):
    """Emergency Kill Switch Control Panel"""

    def __init__(self, security_suite: SecuritySuite, parent=None):
        super().__init__(parent)
        self.security_suite = security_suite
        self.setup_ui()
        self.update_status()

        # Auto-refresh every 5 seconds
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_status)
        self.refresh_timer.start(5000)

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("🚨 Emergency Kill Switch")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #ef4444;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 14px; color: #2563eb;")
        layout.addWidget(self.status_label)

        # Big red button
        self.kill_btn = QPushButton("🛑 EMERGENCY STOP")
        self.kill_btn.setMinimumHeight(100)
        self.kill_btn.clicked.connect(self.trigger_kill_switch)
        self.kill_btn.setStyleSheet(
            """
            QPushButton {
                background: #ef4444;
                color: white;
                font-size: 24px;
                font-weight: 600;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #dc2626;
            }
            QPushButton:pressed {
                background: #b91c1c;
            }
        """
        )
        layout.addWidget(self.kill_btn)

        # Reset button
        self.reset_btn = QPushButton("Reset Kill Switch")
        self.reset_btn.clicked.connect(self.reset_kill_switch)
        self.reset_btn.setEnabled(False)
        layout.addWidget(self.reset_btn)

        # Event history
        history_label = QLabel("Recent Events:")
        history_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(history_label)

        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(150)
        layout.addWidget(self.history_text)

        layout.addStretch()
        self.setLayout(layout)

    def update_status(self):
        if not self.security_suite.kill_switch:
            return

        status = self.security_suite.kill_switch.get_status()

        if status["is_active"]:
            self.status_label.setText("⚠️ KILL SWITCH ACTIVE - Trading Stopped")
            self.status_label.setStyleSheet(
                "font-size: 14px; color: #ef4444; font-weight: 600;"
            )
            self.kill_btn.setEnabled(False)
            self.reset_btn.setEnabled(not status["is_locked"])
        else:
            self.status_label.setText("✅ System Operational")
            self.status_label.setStyleSheet("font-size: 14px; color: #10b981;")
            self.kill_btn.setEnabled(True)
            self.reset_btn.setEnabled(False)

        # Update history
        events = self.security_suite.kill_switch.get_event_history(5)
        history_html = ""
        for event in events:
            history_html += f"<p><b>{event['timestamp'][:19]}</b>: {event['trigger']} - {event['reason']}</p>"
        self.history_text.setHtml(history_html)

    def trigger_kill_switch(self):
        reply = QMessageBox.question(
            self,
            "Confirm Emergency Stop",
            "This will immediately:\n• Stop all trading\n• Close all positions\n• Cancel all orders\n• Lock the system\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            asyncio.create_task(self._do_trigger())

    async def _do_trigger(self):
        result = await self.security_suite.trigger_emergency_shutdown(
            reason="Manual activation from GUI"
        )
        self.update_status()

        if result["success"]:
            QMessageBox.information(
                self, "Success", "Emergency kill switch activated successfully"
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to activate kill switch")

    def reset_kill_switch(self):
        if not self.security_suite.kill_switch.is_locked:
            reply = QMessageBox.question(
                self,
                "Reset Kill Switch",
                "Are you sure you want to reset and resume trading?",
                QMessageBox.Yes | QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                success = self.security_suite.kill_switch.reset(authorized=True)
                self.update_status()

                if success:
                    QMessageBox.information(
                        self, "Success", "Kill switch reset - trading resumed"
                    )


class TaxReportingWidget(QWidget):
    """Tax Reporting Interface"""

    def __init__(self, tax_reporter: TaxReporter, parent=None):
        super().__init__(parent)
        self.tax_reporter = tax_reporter
        self.setup_ui()
        self.update_summary()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("💰 Tax Reporting")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #10b981;")
        layout.addWidget(title)

        # Summary
        summary_group = QGroupBox("Tax Summary (Current Year)")
        summary_layout = QVBoxLayout()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        summary_layout.addWidget(self.summary_text)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Actions
        actions_group = QGroupBox("Generate Reports")
        actions_layout = QVBoxLayout()

        self.year_input = QSpinBox()
        self.year_input.setRange(2020, datetime.now().year + 1)
        self.year_input.setValue(datetime.now().year)
        self.year_input.setPrefix("Tax Year: ")
        actions_layout.addWidget(self.year_input)

        btn_layout = QHBoxLayout()

        form8949_btn = QPushButton("Generate Form 8949")
        form8949_btn.clicked.connect(self.generate_form_8949)
        btn_layout.addWidget(form8949_btn)

        turbotax_btn = QPushButton("Export TurboTax")
        turbotax_btn.clicked.connect(self.export_turbotax)
        btn_layout.addWidget(turbotax_btn)

        actions_layout.addLayout(btn_layout)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # Refresh button
        refresh_btn = QPushButton("Refresh Summary")
        refresh_btn.clicked.connect(self.update_summary)
        layout.addWidget(refresh_btn)

        layout.addStretch()
        self.setLayout(layout)

    def update_summary(self):
        summary = self.tax_reporter.calculate_tax_summary(datetime.now().year)
        liability = self.tax_reporter.get_current_tax_liability()

        summary_html = f"""
        <h3>Tax Summary {datetime.now().year}</h3>
        <p><b>Total Trades:</b> {summary.total_trades}</p>
        <p><b>Total Proceeds:</b> ${float(summary.total_proceeds):,.2f}</p>
        <p><b>Total Cost Basis:</b> ${float(summary.total_cost_basis):,.2f}</p>
        <p><b>Short-term Gain:</b> <span style='color: {"green" if summary.short_term_gain > 0 else "red"}'>${float(summary.short_term_gain):,.2f}</span></p>
        <p><b>Long-term Gain:</b> <span style='color: {"green" if summary.long_term_gain > 0 else "red"}'>${float(summary.long_term_gain):,.2f}</span></p>
        <p><b>Total Gain/Loss:</b> <span style='color: {"green" if summary.total_gain_loss > 0 else "red"}'>${float(summary.total_gain_loss):,.2f}</span></p>
        <hr>
        <p><b>Estimated Tax Liability:</b> <span style='color: red'>${liability['estimated_total_tax']:,.2f}</span></p>
        <p style='font-size: 10px; color: gray;'>*Consult tax professional for actual rates</p>
        """

        self.summary_text.setHtml(summary_html)

    def generate_form_8949(self):
        year = self.year_input.value()
        file_path = self.tax_reporter.generate_form_8949(year)

        QMessageBox.information(
            self,
            "Form 8949 Generated",
            f"Form 8949 for {year} has been generated:\n{file_path}",
        )

    def export_turbotax(self):
        year = self.year_input.value()
        file_path = self.tax_reporter.export_for_turbotax(year)

        QMessageBox.information(
            self,
            "TurboTax Export Complete",
            f"TurboTax file for {year} has been created:\n{file_path}",
        )


class DeFiPositionsWidget(QWidget):
    """DeFi Positions and Pool Management"""

    def __init__(self, defi_integration: DeFiIntegration, parent=None):
        super().__init__(parent)
        self.defi = defi_integration
        self.setup_ui()
        self.update_positions()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("🌊 DeFi Integration")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #2563eb;")
        layout.addWidget(title)

        # Portfolio summary
        self.portfolio_label = QLabel()
        self.portfolio_label.setStyleSheet("font-size: 14px; color: #10b981;")
        layout.addWidget(self.portfolio_label)

        # Positions table
        positions_group = QGroupBox("Active Positions")
        positions_layout = QVBoxLayout()

        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(6)
        self.positions_table.setHorizontalHeaderLabels(
            ["Protocol", "Pair", "Value (USD)", "Rewards", "IL%", "Actions"]
        )
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        positions_layout.addWidget(self.positions_table)

        positions_group.setLayout(positions_layout)
        layout.addWidget(positions_group)

        # Actions
        actions_layout = QHBoxLayout()

        harvest_btn = QPushButton("🌾 Harvest All Rewards")
        harvest_btn.clicked.connect(self.harvest_rewards)
        actions_layout.addWidget(harvest_btn)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.update_positions)
        actions_layout.addWidget(refresh_btn)

        layout.addLayout(actions_layout)

        layout.addStretch()
        self.setLayout(layout)

    def update_positions(self):
        # Update portfolio summary
        yield_data = self.defi.get_portfolio_yield()
        self.portfolio_label.setText(
            f"Portfolio Value: ${yield_data['total_value_usd']:,.2f} | "
            f"Total Rewards: ${yield_data['total_rewards']:.2f} | "
            f"Active Positions: {yield_data['positions_count']}"
        )

        # Update table
        self.positions_table.setRowCount(0)
        for position_id, position in self.defi.active_positions.items():
            row = self.positions_table.rowCount()
            self.positions_table.insertRow(row)

            self.positions_table.setItem(row, 0, QTableWidgetItem(position.protocol))
            self.positions_table.setItem(
                row, 1, QTableWidgetItem(f"{position.token0}/{position.token1}")
            )
            self.positions_table.setItem(
                row, 2, QTableWidgetItem(f"${float(position.value_usd):,.2f}")
            )
            self.positions_table.setItem(
                row, 3, QTableWidgetItem(f"${float(position.rewards_earned):.2f}")
            )
            self.positions_table.setItem(
                row, 4, QTableWidgetItem(f"{float(position.impermanent_loss):.2f}%")
            )

            # Actions button
            withdraw_btn = QPushButton("Withdraw")
            withdraw_btn.clicked.connect(
                lambda checked, pid=position_id: self.withdraw_position(pid)
            )
            self.positions_table.setCellWidget(row, 5, withdraw_btn)

    def harvest_rewards(self):
        asyncio.create_task(self._do_harvest())

    async def _do_harvest(self):
        results = await self.defi.harvest_rewards()
        self.update_positions()

        QMessageBox.information(
            self,
            "Rewards Harvested",
            f"Successfully harvested ${results['total_harvested']:.2f}",
        )

    def withdraw_position(self, position_id: str):
        reply = QMessageBox.question(
            self,
            "Withdraw Liquidity",
            "Withdraw 100% of this position?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            asyncio.create_task(self._do_withdraw(position_id))

    async def _do_withdraw(self, position_id: str):
        success = await self.defi.withdraw_liquidity(position_id, 100)
        self.update_positions()

        if success:
            QMessageBox.information(self, "Success", "Liquidity withdrawn successfully")


class ProfitManagementWidget(QWidget):
    """Profit Withdrawal Management"""

    def __init__(self, profit_manager: ProfitManager, parent=None):
        super().__init__(parent)
        self.profit_manager = profit_manager
        self.setup_ui()
        self.update_summary()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("💸 Profit Management")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #10b981;")
        layout.addWidget(title)

        # Summary
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.summary_label)

        # Manual withdrawal
        withdrawal_group = QGroupBox("Manual Withdrawal")
        withdrawal_layout = QVBoxLayout()

        amount_layout = QHBoxLayout()
        amount_layout.addWidget(QLabel("Amount (USD):"))
        self.amount_input = QDoubleSpinBox()
        self.amount_input.setRange(0, 1000000)
        self.amount_input.setDecimals(2)
        self.amount_input.setPrefix("$ ")
        amount_layout.addWidget(self.amount_input)
        withdrawal_layout.addLayout(amount_layout)

        dest_layout = QHBoxLayout()
        dest_layout.addWidget(QLabel("Destination:"))
        self.dest_combo = QComboBox()
        self.dest_combo.addItems(["Cold Wallet", "Bank Account", "Reinvest"])
        dest_layout.addWidget(self.dest_combo)
        withdrawal_layout.addLayout(dest_layout)

        withdraw_btn = QPushButton("Execute Withdrawal")
        withdraw_btn.clicked.connect(self.execute_withdrawal)
        withdrawal_layout.addWidget(withdraw_btn)

        withdrawal_group.setLayout(withdrawal_layout)
        layout.addWidget(withdrawal_group)

        # Withdrawal history
        history_group = QGroupBox("Recent Withdrawals")
        history_layout = QVBoxLayout()

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(
            ["Date", "Amount", "Destination", "Status"]
        )
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setMaximumHeight(200)
        history_layout.addWidget(self.history_table)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_summary)
        layout.addWidget(refresh_btn)

        layout.addStretch()
        self.setLayout(layout)

    def update_summary(self):
        summary = self.profit_manager.get_withdrawal_summary()

        self.summary_label.setText(
            f"Total Profit: ${summary['total_profit']:,.2f} | "
            f"Withdrawn: ${summary['total_withdrawn']:,.2f} | "
            f"Available: ${summary['available_for_withdrawal']:,.2f}"
        )

        # Update history
        status = self.profit_manager.get_status()
        self.history_table.setRowCount(0)

        for withdrawal in status["recent_withdrawals"][:10]:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)

            self.history_table.setItem(
                row, 0, QTableWidgetItem(withdrawal["timestamp"][:19])
            )
            self.history_table.setItem(
                row, 1, QTableWidgetItem(f"${withdrawal['amount']:,.2f}")
            )
            self.history_table.setItem(
                row, 2, QTableWidgetItem(withdrawal["destination"])
            )
            self.history_table.setItem(row, 3, QTableWidgetItem(withdrawal["status"]))

    def execute_withdrawal(self):
        amount = self.amount_input.value()
        dest_map = {
            "Cold Wallet": WithdrawalDestination.COLD_WALLET,
            "Bank Account": WithdrawalDestination.BANK_ACCOUNT,
            "Reinvest": WithdrawalDestination.REINVEST,
        }
        destination = dest_map[self.dest_combo.currentText()]

        if amount <= 0:
            QMessageBox.warning(self, "Invalid Amount", "Please enter a valid amount")
            return

        asyncio.create_task(self._do_withdrawal(amount, destination))

    async def _do_withdrawal(self, amount: float, destination: WithdrawalDestination):
        withdrawal_id = await self.profit_manager.execute_withdrawal(
            amount, destination, "Manual withdrawal from GUI"
        )

        self.update_summary()

        if withdrawal_id:
            QMessageBox.information(
                self,
                "Withdrawal Successful",
                f"Withdrawal of ${amount:,.2f} executed successfully.\nID: {withdrawal_id}",
            )
        else:
            QMessageBox.warning(
                self, "Withdrawal Failed", "Withdrawal failed. Check logs for details."
            )


class GUIIntegration:
    """
    Main GUI Integration class for Phase 1 & 2 features
    """

    def __init__(self, config: Dict):
        """Initialize GUI Integration with configuration"""
        self.config = config
        self.security_suite = None
        self.tax_reporter = None
        self.defi_integration = None
        self.profit_manager = None

    async def initialize(self):
        """Initialize all Phase 1 & 2 managers"""
        self.security_suite = SecuritySuite(self.config)
        await self.security_suite.initialize()

        self.tax_reporter = TaxReporter(self.config)
        self.defi_integration = DeFiIntegration(self.config)
        self.profit_manager = ProfitManager(self.config)

        logger.info("✅ GUIIntegration initialized")

    def inject_dependencies(
        self, risk_manager=None, exchange_manager=None, telegram_bot=None
    ):
        """Inject external dependencies into security suite"""
        if self.security_suite:
            self.security_suite.inject_external_dependencies(
                risk_manager=risk_manager,
                exchange_manager=exchange_manager,
                telegram_bot=telegram_bot,
            )

    def integrate_into_main_window(self, main_window: QMainWindow):
        """
        Integrate Phase 1 & 2 features into existing GUI

        Call this method from cyber_gui.py to add new tabs
        """
        # Get main tab widget (assuming it exists)
        if hasattr(main_window, "tab_widget"):
            tabs = main_window.tab_widget

            # Add Phase 1 tabs
            if self.security_suite:
                kill_switch_tab = EmergencyKillSwitchWidget(self.security_suite)
                tabs.addTab(kill_switch_tab, "🚨 Emergency")

            # Add Phase 2 tabs
            if self.tax_reporter:
                tax_tab = TaxReportingWidget(self.tax_reporter)
                tabs.addTab(tax_tab, "💰 Tax Reports")

            if self.defi_integration:
                defi_tab = DeFiPositionsWidget(self.defi_integration)
                tabs.addTab(defi_tab, "🌊 DeFi")

            if self.profit_manager:
                profit_tab = ProfitManagementWidget(self.profit_manager)
                tabs.addTab(profit_tab, "💸 Withdrawals")

            logger.info("✅ Phase 1 & 2 features integrated into GUI")

    def get_managers(self):
        """Get all initialized managers"""
        return {
            "security_suite": self.security_suite,
            "tax_reporter": self.tax_reporter,
            "defi_integration": self.defi_integration,
            "profit_manager": self.profit_manager,
        }


def integrate_phase1_phase2_into_gui(main_window: QMainWindow, config: Dict):
    """
    Legacy function for backward compatibility

    Integrate Phase 1 & 2 features into existing GUI

    Call this function from cyber_gui.py to add new tabs
    """
    # Initialize all Phase 1 & 2 managers
    security_suite = SecuritySuite(config)
    asyncio.create_task(security_suite.initialize())

    tax_reporter = TaxReporter(config)
    defi_integration = DeFiIntegration(config)
    profit_manager = ProfitManager(config)

    # Inject dependencies
    security_suite.inject_external_dependencies(
        risk_manager=None,  # Will be injected by main GUI
        exchange_manager=None,
        telegram_bot=None,
    )

    # Get main tab widget (assuming it exists)
    if hasattr(main_window, "tab_widget"):
        tabs = main_window.tab_widget

        # Add Phase 1 tabs
        kill_switch_tab = EmergencyKillSwitchWidget(security_suite)
        tabs.addTab(kill_switch_tab, "🚨 Emergency")

        # Add Phase 2 tabs
        tax_tab = TaxReportingWidget(tax_reporter)
        tabs.addTab(tax_tab, "💰 Tax Reports")

        defi_tab = DeFiPositionsWidget(defi_integration)
        tabs.addTab(defi_tab, "🌊 DeFi")

        profit_tab = ProfitManagementWidget(profit_manager)
        tabs.addTab(profit_tab, "💸 Withdrawals")

        logger.info("✅ Phase 1 & 2 features integrated into GUI")

    return {
        "security_suite": security_suite,
        "tax_reporter": tax_reporter,
        "defi_integration": defi_integration,
        "profit_manager": profit_manager,
    }
