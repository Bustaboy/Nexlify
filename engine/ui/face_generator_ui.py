"""
Face Generator UI
Provides UI for generating character faces with various expressions and customization options.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QSlider, QColorDialog,
    QFileDialog, QMessageBox, QGridLayout, QScrollArea,
    QCheckBox, QFrame, QSizePolicy, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QColor, QPixmap, QPainter, QImage, QFont

logger = logging.getLogger(__name__)


@dataclass
class FaceGeneratorConfig:
    """Configuration for face generator UI theme and colors"""

    # Background Colors
    bg_primary: str = "#ffffff"
    bg_secondary: str = "#f5f7fa"
    bg_tertiary: str = "#e8ecf1"
    bg_card: str = "#ffffff"

    # Accent Colors
    accent_primary: str = "#2563eb"
    accent_secondary: str = "#3b82f6"
    accent_success: str = "#10b981"
    accent_warning: str = "#f59e0b"
    accent_error: str = "#ef4444"

    # Text Colors
    text_primary: str = "#1e293b"
    text_secondary: str = "#64748b"
    text_dim: str = "#94a3b8"

    # Structure
    border_color: str = "#e2e8f0"
    shadow_color: str = "rgba(0, 0, 0, 0.1)"

    # Typography
    font_family: str = "Segoe UI, -apple-system, system-ui, sans-serif"
    font_size_small: int = 11
    font_size_normal: int = 13
    font_size_large: int = 15
    font_size_header: int = 20


class FacePreviewWidget(QLabel):
    """Widget for displaying face preview with custom rendering"""

    def __init__(self, size: int = 128, parent=None):
        super().__init__(parent)
        self.face_size = size
        self.face_data = {}
        self.setFixedSize(size, size)
        self.setFrameStyle(QFrame.Box)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #f5f7fa; border: 2px solid #e2e8f0; border-radius: 8px;")

    def update_face(self, face_data: Dict[str, Any]):
        """Update face preview with new data"""
        self.face_data = face_data
        self.render_face()

    def render_face(self):
        """Render face based on current data"""
        # Create a pixmap to draw on
        pixmap = QPixmap(self.face_size, self.face_size)
        pixmap.fill(QColor("#f5f7fa"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get colors from face data
        skin_color = QColor(self.face_data.get('skin_color', '#ffdbac'))
        hair_color = QColor(self.face_data.get('hair_color', '#8b4513'))
        eye_color = QColor(self.face_data.get('eye_color', '#0066cc'))
        expression = self.face_data.get('expression', 'neutral')

        # Draw face (simplified representation)
        center_x = self.face_size // 2
        center_y = self.face_size // 2
        face_radius = int(self.face_size * 0.35)

        # Draw head
        painter.setBrush(skin_color)
        painter.setPen(QColor("#000000"))
        painter.drawEllipse(center_x - face_radius, center_y - face_radius,
                          face_radius * 2, face_radius * 2)

        # Draw hair
        painter.setBrush(hair_color)
        hair_width = int(face_radius * 2.2)
        hair_height = int(face_radius * 1.2)
        painter.drawEllipse(center_x - hair_width // 2,
                          center_y - face_radius - hair_height // 3,
                          hair_width, hair_height)

        # Draw eyes
        eye_y = center_y - face_radius // 3
        eye_spacing = face_radius // 2
        eye_radius = face_radius // 5

        # Expression affects eye shape
        if expression == 'happy':
            eye_height = eye_radius // 2
            mouth_curve = 10
        elif expression == 'sad':
            eye_height = eye_radius // 2
            mouth_curve = -10
        elif expression == 'angry':
            eye_height = eye_radius
            mouth_curve = -5
        elif expression == 'surprised':
            eye_height = eye_radius * 2
            mouth_curve = 0
        else:  # neutral
            eye_height = eye_radius
            mouth_curve = 0

        # Left eye
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(center_x - eye_spacing - eye_radius,
                          eye_y - eye_height // 2,
                          eye_radius * 2, eye_height)
        painter.setBrush(eye_color)
        painter.drawEllipse(center_x - eye_spacing - eye_radius // 2,
                          eye_y - eye_height // 4,
                          eye_radius, eye_height // 2)

        # Right eye
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(center_x + eye_spacing - eye_radius,
                          eye_y - eye_height // 2,
                          eye_radius * 2, eye_height)
        painter.setBrush(eye_color)
        painter.drawEllipse(center_x + eye_spacing - eye_radius // 2,
                          eye_y - eye_height // 4,
                          eye_radius, eye_height // 2)

        # Draw mouth
        mouth_y = center_y + face_radius // 2
        mouth_width = face_radius

        painter.setBrush(QColor("#000000"))
        if expression in ['happy', 'sad', 'angry']:
            # Curved mouth
            painter.drawArc(center_x - mouth_width // 2,
                          mouth_y - mouth_curve,
                          mouth_width,
                          abs(mouth_curve) * 2,
                          0, 180 * 16 if mouth_curve > 0 else -180 * 16)
        elif expression == 'surprised':
            # Open mouth
            painter.drawEllipse(center_x - mouth_width // 4,
                              mouth_y,
                              mouth_width // 2,
                              mouth_width // 2)
        else:
            # Neutral line
            painter.drawLine(center_x - mouth_width // 2, mouth_y,
                           center_x + mouth_width // 2, mouth_y)

        painter.end()
        self.setPixmap(pixmap)


class FaceGeneratorWidget(QWidget):
    """Main face generator widget with expression selector and customization"""

    # Signals
    face_updated = pyqtSignal(dict)
    export_completed = pyqtSignal(str)

    EXPRESSIONS = ['neutral', 'happy', 'sad', 'angry', 'surprised']
    EXPORT_SIZES = [96, 128, 256, 512]

    def __init__(self, config: Optional[FaceGeneratorConfig] = None, parent=None):
        super().__init__(parent)
        self.config = config or FaceGeneratorConfig()

        # Current face data
        self.face_data = {
            'skin_color': '#ffdbac',
            'hair_color': '#8b4513',
            'eye_color': '#0066cc',
            'expression': 'neutral',
            'export_size': 128
        }

        # Preview widgets for all expressions
        self.expression_previews: Dict[str, FacePreviewWidget] = {}

        # Color sync with body generator (placeholder for integration)
        self.body_colors = {}

        self.setup_ui()
        self.apply_theme()
        self.update_all_previews()

    def setup_ui(self):
        """Create face generator interface"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Left panel: Controls
        controls_panel = self._create_controls_panel()
        main_layout.addWidget(controls_panel, 1)

        # Right panel: Previews
        preview_panel = self._create_preview_panel()
        main_layout.addWidget(preview_panel, 2)

    def _create_controls_panel(self) -> QWidget:
        """Create controls panel with color pickers and settings"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Title
        title = QLabel("Face Customization")
        title.setFont(QFont(self.config.font_family, self.config.font_size_header, QFont.Bold))
        layout.addWidget(title)

        # Expression Selector
        expression_group = self._create_expression_selector()
        layout.addWidget(expression_group)

        # Color Controls
        color_group = self._create_color_controls()
        layout.addWidget(color_group)

        # Export Settings
        export_group = self._create_export_settings()
        layout.addWidget(export_group)

        # Action Buttons
        actions_group = self._create_action_buttons()
        layout.addWidget(actions_group)

        layout.addStretch()

        return panel

    def _create_expression_selector(self) -> QGroupBox:
        """Create expression selector group"""
        group = QGroupBox("Expression")
        layout = QVBoxLayout()

        # Dropdown for expression selection
        expr_layout = QHBoxLayout()
        expr_layout.addWidget(QLabel("Current:"))

        self.expression_combo = QComboBox()
        self.expression_combo.addItems([e.capitalize() for e in self.EXPRESSIONS])
        self.expression_combo.setCurrentText(self.face_data['expression'].capitalize())
        self.expression_combo.currentTextChanged.connect(self._on_expression_changed)
        expr_layout.addWidget(self.expression_combo)

        layout.addLayout(expr_layout)

        # Quick select buttons
        quick_layout = QHBoxLayout()
        for expression in self.EXPRESSIONS:
            btn = QPushButton(expression.capitalize())
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda checked, e=expression: self._set_expression(e))
            quick_layout.addWidget(btn)

        layout.addLayout(quick_layout)

        group.setLayout(layout)
        return group

    def _create_color_controls(self) -> QGroupBox:
        """Create color picker controls"""
        group = QGroupBox("Colors")
        layout = QVBoxLayout()

        # Skin Color
        skin_layout = self._create_color_picker_row("Skin:", 'skin_color')
        layout.addLayout(skin_layout)

        # Hair Color
        hair_layout = self._create_color_picker_row("Hair:", 'hair_color')
        layout.addLayout(hair_layout)

        # Eye Color
        eye_layout = self._create_color_picker_row("Eyes:", 'eye_color')
        layout.addLayout(eye_layout)

        # Sync checkbox
        self.sync_checkbox = QCheckBox("Sync with Body Generator")
        self.sync_checkbox.setChecked(False)
        self.sync_checkbox.stateChanged.connect(self._on_sync_toggled)
        layout.addWidget(self.sync_checkbox)

        group.setLayout(layout)
        return group

    def _create_color_picker_row(self, label: str, color_key: str) -> QHBoxLayout:
        """Create a color picker row"""
        layout = QHBoxLayout()

        label_widget = QLabel(label)
        label_widget.setMinimumWidth(50)
        layout.addWidget(label_widget)

        # Color preview button
        color_btn = QPushButton()
        color_btn.setFixedSize(60, 30)
        color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.face_data[color_key]};
                border: 2px solid {self.config.border_color};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border-color: {self.config.accent_primary};
            }}
        """)
        color_btn.clicked.connect(lambda: self._pick_color(color_key, color_btn))
        layout.addWidget(color_btn)

        # Color hex display
        color_label = QLabel(self.face_data[color_key])
        color_label.setObjectName(f"{color_key}_label")
        layout.addWidget(color_label)

        layout.addStretch()

        return layout

    def _create_export_settings(self) -> QGroupBox:
        """Create export settings group"""
        group = QGroupBox("Export Settings")
        layout = QVBoxLayout()

        # Size selector
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))

        self.size_combo = QComboBox()
        self.size_combo.addItems([f"{size}x{size}" for size in self.EXPORT_SIZES])
        self.size_combo.setCurrentText(f"{self.face_data['export_size']}x{self.face_data['export_size']}")
        self.size_combo.currentTextChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.size_combo)
        size_layout.addStretch()

        layout.addLayout(size_layout)

        # Format selector
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))

        self.format_combo = QComboBox()
        self.format_combo.addItems(['PNG', 'JPEG', 'BMP'])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()

        layout.addLayout(format_layout)

        group.setLayout(layout)
        return group

    def _create_action_buttons(self) -> QGroupBox:
        """Create action buttons group"""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()

        # Export current expression
        export_current_btn = QPushButton("💾 Export Current Expression")
        export_current_btn.clicked.connect(self._export_current)
        layout.addWidget(export_current_btn)

        # Export all expressions (batch)
        export_all_btn = QPushButton("📦 Batch Export All Expressions")
        export_all_btn.clicked.connect(self._export_all_expressions)
        layout.addWidget(export_all_btn)

        # Save configuration
        save_config_btn = QPushButton("⚙️ Save Configuration")
        save_config_btn.clicked.connect(self._save_configuration)
        layout.addWidget(save_config_btn)

        # Load configuration
        load_config_btn = QPushButton("📂 Load Configuration")
        load_config_btn.clicked.connect(self._load_configuration)
        layout.addWidget(load_config_btn)

        # Reset button
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        layout.addWidget(reset_btn)

        group.setLayout(layout)
        return group

    def _create_preview_panel(self) -> QWidget:
        """Create preview panel showing all expressions"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Expression Previews")
        title.setFont(QFont(self.config.font_family, self.config.font_size_header, QFont.Bold))
        layout.addWidget(title)

        # Current large preview
        current_group = QGroupBox("Current Expression (Large Preview)")
        current_layout = QVBoxLayout()
        current_layout.setAlignment(Qt.AlignCenter)

        self.current_preview = FacePreviewWidget(size=256)
        current_layout.addWidget(self.current_preview, 0, Qt.AlignCenter)

        current_group.setLayout(current_layout)
        layout.addWidget(current_group)

        # All expressions grid
        all_group = QGroupBox("All Expressions")
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        # Create preview for each expression
        for i, expression in enumerate(self.EXPRESSIONS):
            col = i % 3
            row = i // 3

            expr_widget = QWidget()
            expr_layout = QVBoxLayout(expr_widget)
            expr_layout.setAlignment(Qt.AlignCenter)

            # Expression label
            label = QLabel(expression.capitalize())
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont(self.config.font_family, self.config.font_size_small))
            expr_layout.addWidget(label)

            # Preview
            preview = FacePreviewWidget(size=96)
            self.expression_previews[expression] = preview
            expr_layout.addWidget(preview)

            # Select button
            select_btn = QPushButton("Select")
            select_btn.setFixedHeight(25)
            select_btn.clicked.connect(lambda checked, e=expression: self._set_expression(e))
            expr_layout.addWidget(select_btn)

            grid_layout.addWidget(expr_widget, row, col)

        all_group.setLayout(grid_layout)
        layout.addWidget(all_group)

        layout.addStretch()

        return panel

    def apply_theme(self):
        """Apply theme styling to the widget"""
        style = f"""
        QGroupBox {{
            background-color: {self.config.bg_card};
            border: 1px solid {self.config.border_color};
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            font-family: {self.config.font_family};
            font-size: {self.config.font_size_normal}px;
            font-weight: bold;
            color: {self.config.text_primary};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }}

        QPushButton {{
            background-color: {self.config.accent_primary};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-family: {self.config.font_family};
            font-size: {self.config.font_size_normal}px;
            font-weight: bold;
        }}

        QPushButton:hover {{
            background-color: {self.config.accent_secondary};
        }}

        QPushButton:pressed {{
            background-color: {self.config.accent_primary};
        }}

        QComboBox {{
            background-color: {self.config.bg_secondary};
            border: 1px solid {self.config.border_color};
            border-radius: 4px;
            padding: 5px;
            font-family: {self.config.font_family};
            font-size: {self.config.font_size_normal}px;
        }}

        QComboBox:hover {{
            border-color: {self.config.accent_primary};
        }}

        QLabel {{
            color: {self.config.text_primary};
            font-family: {self.config.font_family};
            font-size: {self.config.font_size_normal}px;
        }}

        QCheckBox {{
            color: {self.config.text_primary};
            font-family: {self.config.font_family};
            font-size: {self.config.font_size_normal}px;
        }}

        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
        }}

        QCheckBox::indicator:checked {{
            background-color: {self.config.accent_success};
            border: 2px solid {self.config.accent_success};
            border-radius: 3px;
        }}
        """
        self.setStyleSheet(style)

    # Event Handlers

    def _on_expression_changed(self, text: str):
        """Handle expression combo box change"""
        self._set_expression(text.lower())

    def _set_expression(self, expression: str):
        """Set current expression and update previews"""
        self.face_data['expression'] = expression
        self.expression_combo.setCurrentText(expression.capitalize())
        self.update_current_preview()
        self.face_updated.emit(self.face_data.copy())
        logger.info(f"Expression changed to: {expression}")

    def _pick_color(self, color_key: str, button: QPushButton):
        """Open color picker dialog"""
        current_color = QColor(self.face_data[color_key])
        color = QColorDialog.getColor(current_color, self, f"Select {color_key.replace('_', ' ').title()}")

        if color.isValid():
            color_hex = color.name()
            self.face_data[color_key] = color_hex

            # Update button color
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color_hex};
                    border: 2px solid {self.config.border_color};
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    border-color: {self.config.accent_primary};
                }}
            """)

            # Update label
            label = self.findChild(QLabel, f"{color_key}_label")
            if label:
                label.setText(color_hex)

            # Update all previews
            self.update_all_previews()
            self.face_updated.emit(self.face_data.copy())
            logger.info(f"Color {color_key} changed to: {color_hex}")

    def _on_size_changed(self, text: str):
        """Handle export size change"""
        size = int(text.split('x')[0])
        self.face_data['export_size'] = size
        logger.info(f"Export size changed to: {size}")

    def _on_sync_toggled(self, state: int):
        """Handle body generator sync toggle"""
        is_checked = state == Qt.Checked
        if is_checked:
            logger.info("Body generator sync enabled")
            # Placeholder for actual sync implementation
            QMessageBox.information(
                self,
                "Sync Enabled",
                "Colors will sync with body generator.\n\n"
                "Note: Body generator integration pending."
            )
        else:
            logger.info("Body generator sync disabled")

    def update_current_preview(self):
        """Update the large current preview"""
        self.current_preview.update_face(self.face_data)

    def update_all_previews(self):
        """Update all expression previews"""
        # Update current preview
        self.update_current_preview()

        # Update all expression previews with current colors
        for expression, preview in self.expression_previews.items():
            face_data = self.face_data.copy()
            face_data['expression'] = expression
            preview.update_face(face_data)

    def _export_current(self):
        """Export current expression to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Current Expression",
            f"face_{self.face_data['expression']}.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp)"
        )

        if file_path:
            try:
                size = self.face_data['export_size']
                format_ext = self.format_combo.currentText().lower()

                # Create face at export size
                temp_preview = FacePreviewWidget(size=size)
                temp_preview.update_face(self.face_data)

                # Save pixmap
                pixmap = temp_preview.pixmap()
                if pixmap and pixmap.save(file_path, format_ext):
                    logger.info(f"Exported face to: {file_path}")
                    self.export_completed.emit(file_path)
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Face exported successfully to:\n{file_path}"
                    )
                else:
                    raise Exception("Failed to save image")

            except Exception as e:
                logger.error(f"Export failed: {e}")
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export face:\n{str(e)}"
                )

    def _export_all_expressions(self):
        """Batch export all expressions"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "",
            QFileDialog.ShowDirsOnly
        )

        if directory:
            try:
                size = self.face_data['export_size']
                format_ext = self.format_combo.currentText().lower()
                exported_files = []

                # Create progress dialog
                progress = QMessageBox(self)
                progress.setWindowTitle("Batch Export")
                progress.setText("Exporting expressions...")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()

                # Export each expression
                for expression in self.EXPRESSIONS:
                    face_data = self.face_data.copy()
                    face_data['expression'] = expression

                    # Create face at export size
                    temp_preview = FacePreviewWidget(size=size)
                    temp_preview.update_face(face_data)

                    # Save pixmap
                    file_name = f"face_{expression}.{format_ext}"
                    file_path = Path(directory) / file_name

                    pixmap = temp_preview.pixmap()
                    if pixmap and pixmap.save(str(file_path), format_ext):
                        exported_files.append(str(file_path))
                        logger.info(f"Exported {expression} to: {file_path}")

                progress.close()

                # Show summary
                QMessageBox.information(
                    self,
                    "Batch Export Complete",
                    f"Successfully exported {len(exported_files)} expressions to:\n{directory}\n\n"
                    f"Files:\n" + "\n".join([Path(f).name for f in exported_files])
                )

                logger.info(f"Batch export completed: {len(exported_files)} files")

            except Exception as e:
                logger.error(f"Batch export failed: {e}")
                QMessageBox.critical(
                    self,
                    "Batch Export Failed",
                    f"Failed to export expressions:\n{str(e)}"
                )

    def _save_configuration(self):
        """Save current configuration to JSON"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "face_config.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                config_data = {
                    'face_data': self.face_data,
                    'format': self.format_combo.currentText(),
                    'sync_enabled': self.sync_checkbox.isChecked()
                }

                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)

                logger.info(f"Configuration saved to: {file_path}")
                QMessageBox.information(
                    self,
                    "Configuration Saved",
                    f"Configuration saved successfully to:\n{file_path}"
                )

            except Exception as e:
                logger.error(f"Save configuration failed: {e}")
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"Failed to save configuration:\n{str(e)}"
                )

    def _load_configuration(self):
        """Load configuration from JSON"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)

                # Update face data
                if 'face_data' in config_data:
                    self.face_data.update(config_data['face_data'])

                # Update UI controls
                if 'format' in config_data:
                    self.format_combo.setCurrentText(config_data['format'])

                if 'sync_enabled' in config_data:
                    self.sync_checkbox.setChecked(config_data['sync_enabled'])

                # Update expression combo
                self.expression_combo.setCurrentText(self.face_data['expression'].capitalize())

                # Update size combo
                size = self.face_data['export_size']
                self.size_combo.setCurrentText(f"{size}x{size}")

                # Update color buttons and labels
                for color_key in ['skin_color', 'hair_color', 'eye_color']:
                    # Find and update button
                    for btn in self.findChildren(QPushButton):
                        if hasattr(btn, 'styleSheet') and color_key in str(btn.parent()):
                            btn.setStyleSheet(f"""
                                QPushButton {{
                                    background-color: {self.face_data[color_key]};
                                    border: 2px solid {self.config.border_color};
                                    border-radius: 4px;
                                }}
                            """)

                    # Update label
                    label = self.findChild(QLabel, f"{color_key}_label")
                    if label:
                        label.setText(self.face_data[color_key])

                # Update all previews
                self.update_all_previews()

                logger.info(f"Configuration loaded from: {file_path}")
                QMessageBox.information(
                    self,
                    "Configuration Loaded",
                    f"Configuration loaded successfully from:\n{file_path}"
                )

            except Exception as e:
                logger.error(f"Load configuration failed: {e}")
                QMessageBox.critical(
                    self,
                    "Load Failed",
                    f"Failed to load configuration:\n{str(e)}"
                )

    def _reset_to_defaults(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Reset face data
            self.face_data = {
                'skin_color': '#ffdbac',
                'hair_color': '#8b4513',
                'eye_color': '#0066cc',
                'expression': 'neutral',
                'export_size': 128
            }

            # Reset UI controls
            self.expression_combo.setCurrentText('Neutral')
            self.size_combo.setCurrentText('128x128')
            self.format_combo.setCurrentText('PNG')
            self.sync_checkbox.setChecked(False)

            # Update all previews
            self.update_all_previews()

            logger.info("Reset to default settings")
            QMessageBox.information(
                self,
                "Reset Complete",
                "All settings have been reset to defaults."
            )

    # Public methods for integration

    def set_body_colors(self, colors: Dict[str, str]):
        """
        Set colors from body generator (for synchronization)

        Args:
            colors: Dictionary with keys like 'skin', 'hair', etc.
        """
        self.body_colors = colors

        if self.sync_checkbox.isChecked():
            # Map body colors to face colors
            if 'skin' in colors:
                self.face_data['skin_color'] = colors['skin']
            if 'hair' in colors:
                self.face_data['hair_color'] = colors['hair']
            if 'eyes' in colors:
                self.face_data['eye_color'] = colors['eyes']

            self.update_all_previews()
            logger.info("Synchronized colors with body generator")

    def get_face_data(self) -> Dict[str, Any]:
        """Get current face configuration"""
        return self.face_data.copy()

    def set_face_data(self, data: Dict[str, Any]):
        """Set face configuration"""
        self.face_data.update(data)
        self.update_all_previews()
