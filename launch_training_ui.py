#!/usr/bin/env python3
"""
Nexlify Training UI Launcher

Launches the ML/RL training dashboard for walk-forward validation
and agent training with real-time monitoring.

Usage:
    python launch_training_ui.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Launch the training UI"""
    try:
        from nexlify.gui.training_ui import main as training_ui_main

        logger.info("Launching Nexlify Training UI...")
        training_ui_main()

    except ImportError as e:
        logger.error(f"Failed to import training UI: {e}")
        logger.error("Make sure PyQt5 is installed: pip install PyQt5")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to launch training UI: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
