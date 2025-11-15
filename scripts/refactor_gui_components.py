#!/usr/bin/env python3
"""
Script to refactor cyber_gui.py to use extracted components
"""

import re
from pathlib import Path

def refactor_cyber_gui():
    """Refactor cyber_gui.py to import from components"""
    cyber_gui_path = Path("nexlify/gui/cyber_gui.py")

    with open(cyber_gui_path, 'r') as f:
        content = f.read()

    # Add import for components after the existing nexlify.core imports
    old_import = """from nexlify.core.nexlify_neural_net import NexlifyNeuralNet
from nexlify.gui.nexlify_cyberpunk_effects import (CyberpunkEffects,
                                                   SoundManager)"""

    new_import = """from nexlify.core.nexlify_neural_net import NexlifyNeuralNet
from nexlify.gui.components import RateLimitedButton, VirtualTableModel, LogWidget
from nexlify.gui.nexlify_cyberpunk_effects import (CyberpunkEffects,
                                                   SoundManager)"""

    content = content.replace(old_import, new_import)

    # Find and remove RateLimitedButton class definition (lines 87-151)
    # Pattern to match the entire class
    rate_limited_pattern = r'class RateLimitedButton\(QPushButton\):.*?(?=\n\nclass |\nclass )'
    content = re.sub(rate_limited_pattern, '', content, flags=re.DOTALL)

    # Find and remove VirtualTableModel class definition (lines 153-200)
    virtual_table_pattern = r'class VirtualTableModel\(QAbstractTableModel\):.*?(?=\n\nclass |\nclass )'
    content = re.sub(virtual_table_pattern, '', content, flags=re.DOTALL)

    # Find and remove LogWidget class definition (lines 203-249)
    log_widget_pattern = r'class LogWidget\(QPlainTextEdit\):.*?(?=\n\nclass |\nclass )'
    content = re.sub(log_widget_pattern, '', content, flags=re.DOTALL)

    # Clean up multiple blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)

    # Write back
    with open(cyber_gui_path, 'w') as f:
        f.write(content)

    print(f"✅ Refactored {cyber_gui_path}")
    print(f"   - Added component imports")
    print(f"   - Removed RateLimitedButton class (~65 lines)")
    print(f"   - Removed VirtualTableModel class (~45 lines)")
    print(f"   - Removed LogWidget class (~45 lines)")
    print(f"   - Total lines saved: ~155 lines")

if __name__ == "__main__":
    refactor_cyber_gui()
