"""
PIN Setup Dialog

Forces user to set a secure PIN on first boot or when using default PIN.
Blocks all other UI interactions until PIN is properly configured.
"""

import hashlib
import logging
from typing import Optional
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QLineEdit, QMessageBox, QProgressBar
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QIcon
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

logger = logging.getLogger(__name__)


class PINSetupDialog(QDialog):
    """
    Modal dialog for setting up PIN on first boot

    Features:
    - Blocks all other UI until PIN is set
    - Enforces PIN strength requirements
    - Confirms PIN with re-entry
    - Cannot be closed without setting PIN
    """

    # PIN strength requirements
    MIN_PIN_LENGTH = 6
    MAX_PIN_LENGTH = 20

    def __init__(self, parent=None, reason: str = "first_boot"):
        """
        Initialize PIN setup dialog

        Args:
            parent: Parent widget
            reason: Why PIN setup is required ('first_boot', 'weak_pin', 'default_pin')
        """
        super().__init__(parent)

        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 required for PIN setup dialog")

        self.reason = reason
        self.new_pin = None

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Security Setup Required")
        self.setModal(True)  # Block all other windows
        self.setWindowFlags(
            Qt.Dialog |
            Qt.WindowTitleHint |
            Qt.CustomizeWindowHint  # Remove close button
        )
        self.setFixedSize(500, 400)

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Warning icon and title
        title = QLabel("🔒 PIN Setup Required")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Reason message
        if self.reason == "first_boot":
            message = (
                "Welcome to Nexlify!\n\n"
                "For your security, you must set up a PIN before using the application.\n"
                "This PIN will protect your API keys and trading operations."
            )
        elif self.reason == "default_pin":
            message = (
                "⚠️ Default PIN Detected!\n\n"
                "You are using the default PIN which is not secure.\n"
                "Please set a custom PIN to protect your account."
            )
        elif self.reason == "weak_pin":
            message = (
                "⚠️ Weak PIN Detected!\n\n"
                "Your current PIN does not meet security requirements.\n"
                "Please set a stronger PIN."
            )
        else:
            message = "Please set up your security PIN."

        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(message_label)

        # PIN requirements
        requirements = QLabel(
            f"PIN Requirements:\n"
            f"• {self.MIN_PIN_LENGTH}-{self.MAX_PIN_LENGTH} characters\n"
            f"• Mix of numbers and letters recommended\n"
            f"• Avoid common patterns (123456, password, etc.)"
        )
        requirements.setWordWrap(True)
        layout.addWidget(requirements)

        # PIN input
        pin_layout = QVBoxLayout()

        pin_layout.addWidget(QLabel("New PIN:"))
        self.pin_input = QLineEdit()
        self.pin_input.setEchoMode(QLineEdit.Password)
        self.pin_input.setPlaceholderText("Enter new PIN...")
        self.pin_input.textChanged.connect(self.on_pin_changed)
        pin_layout.addWidget(self.pin_input)

        pin_layout.addWidget(QLabel("Confirm PIN:"))
        self.confirm_input = QLineEdit()
        self.confirm_input.setEchoMode(QLineEdit.Password)
        self.confirm_input.setPlaceholderText("Re-enter PIN...")
        self.confirm_input.textChanged.connect(self.on_pin_changed)
        pin_layout.addWidget(self.confirm_input)

        layout.addLayout(pin_layout)

        # Strength indicator
        self.strength_bar = QProgressBar()
        self.strength_bar.setMaximum(100)
        self.strength_bar.setValue(0)
        self.strength_bar.setTextVisible(True)
        self.strength_bar.setFormat("PIN Strength: %p%")
        layout.addWidget(self.strength_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.set_pin_btn = QPushButton("Set PIN")
        self.set_pin_btn.clicked.connect(self.on_set_pin)
        self.set_pin_btn.setEnabled(False)
        button_layout.addWidget(self.set_pin_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

    def apply_theme(self):
        """Apply cyberpunk theme to dialog"""
        self.setStyleSheet("""
            QDialog {
                background-color: #0a0e27;
                color: #00ff9f;
            }
            QLabel {
                color: #00ff9f;
            }
            QLineEdit {
                background-color: #1a1f3a;
                color: #00ff9f;
                border: 2px solid #00ff9f;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #ff00ff;
            }
            QPushButton {
                background-color: #1a1f3a;
                color: #00ff9f;
                border: 2px solid #00ff9f;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00ff9f;
                color: #0a0e27;
            }
            QPushButton:disabled {
                background-color: #1a1f3a;
                color: #4a5568;
                border: 2px solid #4a5568;
            }
            QProgressBar {
                border: 2px solid #00ff9f;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1f3a;
            }
            QProgressBar::chunk {
                background-color: #00ff9f;
            }
        """)

    def calculate_pin_strength(self, pin: str) -> int:
        """
        Calculate PIN strength score (0-100)

        Args:
            pin: PIN string

        Returns:
            Strength score (0-100)
        """
        if not pin:
            return 0

        score = 0

        # Length score (max 40 points)
        length_score = min(40, (len(pin) / self.MAX_PIN_LENGTH) * 40)
        score += length_score

        # Character variety (max 30 points)
        has_lower = any(c.islower() for c in pin)
        has_upper = any(c.isupper() for c in pin)
        has_digit = any(c.isdigit() for c in pin)
        has_special = any(not c.isalnum() for c in pin)

        variety_score = 0
        if has_lower: variety_score += 8
        if has_upper: variety_score += 8
        if has_digit: variety_score += 7
        if has_special: variety_score += 7
        score += variety_score

        # Not common patterns (max 30 points)
        common_patterns = [
            '123456', 'password', 'admin', '111111', 'qwerty',
            'abc123', 'letmein', 'welcome', '123123', 'default'
        ]

        is_common = any(pattern in pin.lower() for pattern in common_patterns)
        if not is_common:
            score += 30
        else:
            score -= 20  # Penalty for common patterns

        return max(0, min(100, int(score)))

    def on_pin_changed(self):
        """Handle PIN input changes"""
        pin = self.pin_input.text()
        confirm = self.confirm_input.text()

        # Calculate strength
        strength = self.calculate_pin_strength(pin)
        self.strength_bar.setValue(strength)

        # Update strength bar color
        if strength < 30:
            color = "#ff0080"  # Red
            self.strength_bar.setStyleSheet(f"""
                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)
        elif strength < 60:
            color = "#ffaa00"  # Orange
            self.strength_bar.setStyleSheet(f"""
                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)
        else:
            color = "#00ff9f"  # Green
            self.strength_bar.setStyleSheet(f"""
                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)

        # Validate
        is_valid = True
        message = ""

        if len(pin) < self.MIN_PIN_LENGTH:
            is_valid = False
            message = f"PIN must be at least {self.MIN_PIN_LENGTH} characters"
        elif len(pin) > self.MAX_PIN_LENGTH:
            is_valid = False
            message = f"PIN must not exceed {self.MAX_PIN_LENGTH} characters"
        elif strength < 40:
            is_valid = False
            message = "PIN is too weak - try adding numbers, letters, or special characters"
        elif pin != confirm:
            is_valid = False
            if confirm:  # Only show mismatch if user has started typing
                message = "PINs do not match"
        else:
            message = "✓ PIN is strong and valid"

        self.status_label.setText(message)
        self.set_pin_btn.setEnabled(is_valid and len(confirm) > 0)

    def on_set_pin(self):
        """Handle Set PIN button click"""
        pin = self.pin_input.text()
        confirm = self.confirm_input.text()

        # Final validation
        if pin != confirm:
            QMessageBox.warning(
                self,
                "PIN Mismatch",
                "The PINs you entered do not match. Please try again."
            )
            return

        strength = self.calculate_pin_strength(pin)
        if strength < 40:
            reply = QMessageBox.question(
                self,
                "Weak PIN",
                "Your PIN is weak. Are you sure you want to use it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Save PIN
        try:
            self.save_pin(pin)
            self.new_pin = pin

            QMessageBox.information(
                self,
                "PIN Set Successfully",
                "Your PIN has been set successfully. Please remember it!"
            )

            self.accept()  # Close dialog with success

        except Exception as e:
            logger.error(f"Failed to save PIN: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save PIN: {str(e)}"
            )

    def save_pin(self, pin: str):
        """
        Save PIN to secure storage

        Args:
            pin: PIN to save
        """
        from nexlify.security.nexlify_pin_manager import PINManager

        pin_manager = PINManager()

        # Hash the PIN
        pin_hash = hashlib.sha256(pin.encode()).hexdigest()

        # Save to file
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        pin_file = config_dir / ".pin_hash"
        with open(pin_file, 'w') as f:
            f.write(pin_hash)

        # Set restrictive permissions
        import os
        os.chmod(pin_file, 0o600)

        logger.info("PIN hash saved successfully")

    @staticmethod
    def check_if_setup_required() -> bool:
        """
        Check if PIN setup is required

        Returns:
            True if setup is required
        """
        pin_file = Path("config/.pin_hash")

        # No PIN file = first boot
        if not pin_file.exists():
            return True

        # Check if default PIN is in use
        try:
            with open(pin_file) as f:
                stored_hash = f.read().strip()

            # Hash of default PIN "nexlify_default_password_change_me"
            default_hash = hashlib.sha256(
                "nexlify_default_password_change_me".encode()
            ).hexdigest()

            if stored_hash == default_hash:
                return True  # Using default PIN

        except Exception as e:
            logger.error(f"Failed to check PIN: {e}")
            return True  # Safer to require setup

        return False

    @staticmethod
    def get_setup_reason() -> str:
        """
        Determine why PIN setup is required

        Returns:
            Reason string ('first_boot', 'default_pin', 'weak_pin')
        """
        pin_file = Path("config/.pin_hash")

        if not pin_file.exists():
            return "first_boot"

        try:
            with open(pin_file) as f:
                stored_hash = f.read().strip()

            default_hash = hashlib.sha256(
                "nexlify_default_password_change_me".encode()
            ).hexdigest()

            if stored_hash == default_hash:
                return "default_pin"

        except Exception:
            return "first_boot"

        return "weak_pin"


def show_pin_setup_dialog(parent=None) -> Optional[str]:
    """
    Show PIN setup dialog and return the new PIN

    Args:
        parent: Parent widget

    Returns:
        New PIN string if set, None if cancelled
    """
    if not PINSetupDialog.check_if_setup_required():
        return None

    reason = PINSetupDialog.get_setup_reason()
    dialog = PINSetupDialog(parent, reason)

    result = dialog.exec_()

    if result == QDialog.Accepted:
        return dialog.new_pin
    else:
        return None
