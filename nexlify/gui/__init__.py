"""GUI components and cyberpunk effects."""

from nexlify.gui.cyber_gui import CyberGUI
from nexlify.gui.nexlify_cyberpunk_effects import CyberpunkEffects
from nexlify.gui.nexlify_gui_integration import GUIIntegration
from nexlify.gui.nexlify_hardware_detection import HardwareDetector

# Ultra-Optimized System UI (optional - graceful if dependencies missing)
try:
    from nexlify.gui.nexlify_ultra_optimized_ui import (
        ULTRA_OPTIMIZED_AVAILABLE, UltraOptimizedConfigDialog,
        show_ultra_optimized_config)

    __all__ = [
        "CyberGUI",
        "GUIIntegration",
        "CyberpunkEffects",
        "HardwareDetector",
        # Ultra-Optimized UI
        "UltraOptimizedConfigDialog",
        "show_ultra_optimized_config",
        "ULTRA_OPTIMIZED_AVAILABLE",
    ]
except ImportError:
    # Ultra-optimized UI not available (missing dependencies)
    __all__ = [
        "CyberGUI",
        "GUIIntegration",
        "CyberpunkEffects",
        "HardwareDetector",
    ]
