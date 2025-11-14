#!/usr/bin/env python3
"""
Face Generator Example
Demonstrates how to use the FaceGeneratorWidget standalone.

Usage:
    python examples/face_generator_example.py
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtWidgets import QApplication, QMainWindow
from engine.ui import FaceGeneratorWidget, FaceGeneratorConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceGeneratorWindow(QMainWindow):
    """Main window for face generator example"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Generator Example")
        self.setGeometry(100, 100, 1200, 800)

        # Create face generator widget
        config = FaceGeneratorConfig()
        self.face_generator = FaceGeneratorWidget(config)

        # Connect signals
        self.face_generator.face_updated.connect(self.on_face_updated)
        self.face_generator.export_completed.connect(self.on_export_completed)

        # Set as central widget
        self.setCentralWidget(self.face_generator)

        logger.info("Face Generator window initialized")

    def on_face_updated(self, face_data):
        """Handle face updates"""
        logger.info(f"Face updated: {face_data}")

    def on_export_completed(self, file_path):
        """Handle export completion"""
        logger.info(f"Export completed: {file_path}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Face Generator")
    app.setOrganizationName("Nexlify")

    # Create and show window
    window = FaceGeneratorWindow()
    window.show()

    logger.info("Application started")

    # Run event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
