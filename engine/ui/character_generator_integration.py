"""
Character Generator Integration
Demonstrates how to integrate FaceGeneratorWidget with a character/body generator.
"""

import logging
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QTabWidget, QPushButton, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from engine.ui.face_generator_ui import FaceGeneratorWidget, FaceGeneratorConfig

logger = logging.getLogger(__name__)


class CharacterGeneratorWidget(QWidget):
    """
    Integrated character generator with face and body customization.

    This widget demonstrates how to integrate the FaceGeneratorWidget
    with a body generator (placeholder) to create a complete character
    creation system with color synchronization.
    """

    # Signals
    character_updated = pyqtSignal(dict)
    export_completed = pyqtSignal(str)

    def __init__(self, config: Optional[FaceGeneratorConfig] = None, parent=None):
        super().__init__(parent)
        self.config = config or FaceGeneratorConfig()

        # Character data combining face and body
        self.character_data = {
            'face': {},
            'body': {},
            'metadata': {
                'name': '',
                'version': '1.0'
            }
        }

        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        """Create integrated character generator interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_layout = self._create_title_bar()
        layout.addLayout(title_layout)

        # Main content with tabs or panels
        content = self._create_content_area()
        layout.addWidget(content)

        # Action bar
        action_bar = self._create_action_bar()
        layout.addWidget(action_bar)

    def _create_title_bar(self) -> QHBoxLayout:
        """Create title bar"""
        layout = QHBoxLayout()

        from PyQt5.QtWidgets import QLabel
        title = QLabel("🎨 Character Generator")
        title.setFont(QFont(self.config.font_family, 24, QFont.Bold))
        layout.addWidget(title)

        layout.addStretch()

        return layout

    def _create_content_area(self) -> QWidget:
        """
        Create main content area.

        Provides two integration modes:
        1. Tabbed mode - Face and body in separate tabs
        2. Split mode - Face and body side-by-side
        """
        # Create tab widget for different modes
        tab_widget = QTabWidget()

        # Mode 1: Tabbed Interface
        tabbed_view = self._create_tabbed_interface()
        tab_widget.addTab(tabbed_view, "📑 Tabbed Mode")

        # Mode 2: Split View
        split_view = self._create_split_interface()
        tab_widget.addTab(split_view, "⚡ Split Mode")

        return tab_widget

    def _create_tabbed_interface(self) -> QWidget:
        """Create tabbed interface with face and body generators"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        tabs = QTabWidget()

        # Face tab
        self.face_generator = FaceGeneratorWidget(self.config)
        self.face_generator.face_updated.connect(self._on_face_updated)
        self.face_generator.export_completed.connect(self._on_face_exported)
        tabs.addTab(self.face_generator, "😊 Face Generator")

        # Body tab (placeholder)
        body_tab = self._create_body_generator_placeholder()
        tabs.addTab(body_tab, "🧍 Body Generator")

        layout.addWidget(tabs)

        return widget

    def _create_split_interface(self) -> QWidget:
        """Create split view with face and body side-by-side"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        splitter = QSplitter(Qt.Horizontal)

        # Left: Face generator (smaller)
        self.face_generator_split = FaceGeneratorWidget(self.config)
        self.face_generator_split.face_updated.connect(self._on_face_updated)
        self.face_generator_split.export_completed.connect(self._on_face_exported)

        face_group = QGroupBox("Face Generator")
        face_layout = QVBoxLayout()
        face_layout.addWidget(self.face_generator_split)
        face_group.setLayout(face_layout)

        splitter.addWidget(face_group)

        # Right: Body generator (larger)
        body_group = QGroupBox("Body Generator")
        body_layout = QVBoxLayout()
        body_placeholder = self._create_body_generator_placeholder()
        body_layout.addWidget(body_placeholder)
        body_group.setLayout(body_layout)

        splitter.addWidget(body_group)

        # Set initial sizes (30% face, 70% body)
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

        return widget

    def _create_body_generator_placeholder(self) -> QWidget:
        """
        Create placeholder for body generator.

        Replace this with actual body generator implementation.
        """
        from PyQt5.QtWidgets import QLabel, QColorDialog, QVBoxLayout

        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_label = QLabel(
            "Body Generator Placeholder\n\n"
            "This is where the body generator component would be integrated.\n"
            "Features to implement:\n"
            "• Body type selection\n"
            "• Clothing customization\n"
            "• Color synchronization with face\n"
            "• Export functionality\n\n"
            "Click the button below to simulate color sync."
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        # Simulate color sync button
        sync_btn = QPushButton("🎨 Pick Body Colors (Simulate Sync)")
        sync_btn.clicked.connect(self._simulate_body_color_sync)
        layout.addWidget(sync_btn)

        layout.addStretch()

        return widget

    def _create_action_bar(self) -> QWidget:
        """Create bottom action bar with export and save options"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Export complete character
        export_char_btn = QPushButton("💾 Export Complete Character")
        export_char_btn.clicked.connect(self._export_complete_character)
        layout.addWidget(export_char_btn)

        # Save character data
        save_data_btn = QPushButton("📄 Save Character Data")
        save_data_btn.clicked.connect(self._save_character_data)
        layout.addWidget(save_data_btn)

        # Load character data
        load_data_btn = QPushButton("📂 Load Character Data")
        load_data_btn.clicked.connect(self._load_character_data)
        layout.addWidget(load_data_btn)

        layout.addStretch()

        return widget

    def apply_theme(self):
        """Apply theme styling"""
        style = f"""
        QGroupBox {{
            background-color: {self.config.bg_card};
            border: 2px solid {self.config.border_color};
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 15px;
            font-family: {self.config.font_family};
            font-size: {self.config.font_size_normal}px;
            font-weight: bold;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 5px;
        }}

        QPushButton {{
            background-color: {self.config.accent_primary};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-family: {self.config.font_family};
            font-size: {self.config.font_size_normal}px;
            font-weight: bold;
        }}

        QPushButton:hover {{
            background-color: {self.config.accent_secondary};
        }}

        QTabWidget::pane {{
            border: 1px solid {self.config.border_color};
            border-radius: 4px;
            background-color: {self.config.bg_secondary};
        }}

        QTabBar::tab {{
            background-color: {self.config.bg_tertiary};
            border: 1px solid {self.config.border_color};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 8px 16px;
            margin-right: 2px;
        }}

        QTabBar::tab:selected {{
            background-color: {self.config.bg_secondary};
            border-bottom: 2px solid {self.config.accent_primary};
        }}

        QTabBar::tab:hover {{
            background-color: {self.config.bg_secondary};
        }}
        """
        self.setStyleSheet(style)

    # Event Handlers

    def _on_face_updated(self, face_data: Dict[str, Any]):
        """Handle face generator updates"""
        self.character_data['face'] = face_data
        self.character_updated.emit(self.character_data.copy())
        logger.info("Face data updated in character")

    def _on_face_exported(self, file_path: str):
        """Handle face export completion"""
        logger.info(f"Face exported: {file_path}")

    def _simulate_body_color_sync(self):
        """Simulate color synchronization from body to face"""
        from PyQt5.QtWidgets import QColorDialog
        from PyQt5.QtGui import QColor

        # Pick a color to simulate body color
        color = QColorDialog.getColor(QColor("#ffdbac"), self, "Pick Body Color")

        if color.isValid():
            # Simulate body colors
            body_colors = {
                'skin': color.name(),
                'hair': '#8b4513',  # Keep hair separate
                'eyes': '#0066cc'   # Keep eyes separate
            }

            # Sync to face generator
            if hasattr(self, 'face_generator'):
                self.face_generator.set_body_colors(body_colors)

            if hasattr(self, 'face_generator_split'):
                self.face_generator_split.set_body_colors(body_colors)

            QMessageBox.information(
                self,
                "Color Sync",
                f"Body color {color.name()} synced to face generator!"
            )

            logger.info(f"Simulated body color sync: {body_colors}")

    def _export_complete_character(self):
        """Export complete character (face + body)"""
        from PyQt5.QtWidgets import QFileDialog
        import json

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Complete Character",
            "character_complete.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                # Get latest face data
                if hasattr(self, 'face_generator'):
                    self.character_data['face'] = self.face_generator.get_face_data()

                # Add export metadata
                self.character_data['metadata']['export_format'] = 'complete'

                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(self.character_data, f, indent=2)

                logger.info(f"Complete character exported to: {file_path}")
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Complete character exported to:\n{file_path}"
                )

                self.export_completed.emit(file_path)

            except Exception as e:
                logger.error(f"Export failed: {e}")
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export character:\n{str(e)}"
                )

    def _save_character_data(self):
        """Save character data to file"""
        from PyQt5.QtWidgets import QFileDialog
        import json

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Character Data",
            "character_data.json",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                # Get latest face data
                if hasattr(self, 'face_generator'):
                    self.character_data['face'] = self.face_generator.get_face_data()

                with open(file_path, 'w') as f:
                    json.dump(self.character_data, f, indent=2)

                logger.info(f"Character data saved to: {file_path}")
                QMessageBox.information(
                    self,
                    "Save Successful",
                    f"Character data saved to:\n{file_path}"
                )

            except Exception as e:
                logger.error(f"Save failed: {e}")
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"Failed to save character data:\n{str(e)}"
                )

    def _load_character_data(self):
        """Load character data from file"""
        from PyQt5.QtWidgets import QFileDialog
        import json

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Character Data",
            "",
            "JSON Files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.character_data = json.load(f)

                # Update face generator if face data exists
                if 'face' in self.character_data:
                    if hasattr(self, 'face_generator'):
                        self.face_generator.set_face_data(self.character_data['face'])
                    if hasattr(self, 'face_generator_split'):
                        self.face_generator_split.set_face_data(self.character_data['face'])

                logger.info(f"Character data loaded from: {file_path}")
                QMessageBox.information(
                    self,
                    "Load Successful",
                    f"Character data loaded from:\n{file_path}"
                )

                self.character_updated.emit(self.character_data.copy())

            except Exception as e:
                logger.error(f"Load failed: {e}")
                QMessageBox.critical(
                    self,
                    "Load Failed",
                    f"Failed to load character data:\n{str(e)}"
                )
