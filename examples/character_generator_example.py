#!/usr/bin/env python3
"""
Character Generator Example
Demonstrates how to use the integrated CharacterGeneratorWidget.

Usage:
    python examples/character_generator_example.py
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtWidgets import QApplication, QMainWindow
from engine.ui import CharacterGeneratorWidget, FaceGeneratorConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CharacterGeneratorWindow(QMainWindow):
    """Main window for character generator example"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Character Generator Example - Integrated Face & Body")
        self.setGeometry(100, 100, 1400, 900)

        # Create character generator widget
        config = FaceGeneratorConfig()
        self.character_generator = CharacterGeneratorWidget(config)

        # Connect signals
        self.character_generator.character_updated.connect(self.on_character_updated)
        self.character_generator.export_completed.connect(self.on_export_completed)

        # Set as central widget
        self.setCentralWidget(self.character_generator)

        logger.info("Character Generator window initialized")

    def on_character_updated(self, character_data):
        """Handle character updates"""
        logger.info(f"Character updated: {character_data.keys()}")

    def on_export_completed(self, file_path):
        """Handle export completion"""
        logger.info(f"Export completed: {file_path}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Character Generator")
    app.setOrganizationName("Nexlify")

    # Create and show window
    window = CharacterGeneratorWindow()
    window.show()

    logger.info("Application started")

    # Run event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
