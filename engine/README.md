# Face Generator UI

A comprehensive PyQt5-based face generator with expression customization, color synchronization, and batch export capabilities.

## Features

### Core Features

- **Expression Selector**: Choose from 5 different expressions
  - Neutral
  - Happy
  - Sad
  - Angry
  - Surprised

- **Color Customization**: Full color control for
  - Skin tone
  - Hair color
  - Eye color

- **Live Preview**: Real-time preview of all expressions
  - Large preview of current selection (256x256)
  - Grid preview of all expressions (96x96 each)
  - Instant updates when colors change

- **Export Functionality**:
  - Single expression export
  - Batch export all expressions
  - Multiple sizes: 96x96, 128x128, 256x256, 512x512
  - Multiple formats: PNG, JPEG, BMP

- **Configuration Management**:
  - Save/load face configurations
  - Reset to defaults
  - JSON-based storage

- **Color Synchronization**:
  - Sync with body generator (when integrated)
  - Maintains consistency across character parts

## File Structure

```
engine/
├── __init__.py                          # Package initialization
├── ui/
│   ├── __init__.py                      # UI module exports
│   ├── face_generator_ui.py             # Main face generator widget
│   └── character_generator_integration.py  # Integration example
└── README.md                            # This file

examples/
├── face_generator_example.py            # Standalone face generator
└── character_generator_example.py       # Integrated character generator
```

## Quick Start

### Standalone Face Generator

```python
from PyQt5.QtWidgets import QApplication
from engine.ui import FaceGeneratorWidget, FaceGeneratorConfig

app = QApplication([])

# Create widget
config = FaceGeneratorConfig()
widget = FaceGeneratorWidget(config)

# Connect signals (optional)
widget.face_updated.connect(lambda data: print(f"Face updated: {data}"))
widget.export_completed.connect(lambda path: print(f"Exported to: {path}"))

widget.show()
app.exec_()
```

### Integrated with Character Generator

```python
from PyQt5.QtWidgets import QApplication
from engine.ui import CharacterGeneratorWidget, FaceGeneratorConfig

app = QApplication([])

# Create integrated widget
config = FaceGeneratorConfig()
widget = CharacterGeneratorWidget(config)

widget.show()
app.exec_()
```

### Running Examples

```bash
# Face generator only
python examples/face_generator_example.py

# Full character generator (face + body)
python examples/character_generator_example.py
```

## Usage Guide

### Creating a Face Generator

```python
from engine.ui import FaceGeneratorWidget, FaceGeneratorConfig

# Create with default config
widget = FaceGeneratorWidget()

# Or customize config
config = FaceGeneratorConfig(
    accent_primary="#ff0000",  # Red accent
    bg_primary="#ffffff"       # White background
)
widget = FaceGeneratorWidget(config)
```

### Customizing Configuration

```python
from dataclasses import dataclass
from engine.ui import FaceGeneratorConfig

# Inherit and customize
@dataclass
class CustomConfig(FaceGeneratorConfig):
    accent_primary: str = "#9b59b6"     # Purple
    accent_secondary: str = "#8e44ad"   # Dark purple
    bg_primary: str = "#2c3e50"         # Dark bg
    text_primary: str = "#ecf0f1"       # Light text
```

### Programmatically Setting Face Data

```python
# Get current data
face_data = widget.get_face_data()
print(face_data)
# {
#     'skin_color': '#ffdbac',
#     'hair_color': '#8b4513',
#     'eye_color': '#0066cc',
#     'expression': 'neutral',
#     'export_size': 128
# }

# Set face data
widget.set_face_data({
    'skin_color': '#f4c2a0',
    'hair_color': '#000000',
    'eye_color': '#228b22',
    'expression': 'happy'
})
```

### Synchronizing with Body Generator

```python
# When body colors change, sync to face
body_colors = {
    'skin': '#ffdbac',
    'hair': '#8b4513',
    'eyes': '#0066cc'
}

face_widget.set_body_colors(body_colors)
```

### Connecting Signals

```python
def on_face_updated(face_data):
    print(f"Expression: {face_data['expression']}")
    print(f"Colors: {face_data['skin_color']}, {face_data['hair_color']}")

def on_export_completed(file_path):
    print(f"Face saved to: {file_path}")

widget.face_updated.connect(on_face_updated)
widget.export_completed.connect(on_export_completed)
```

### Batch Export Example

```python
import asyncio
from pathlib import Path

# Export all expressions programmatically
async def batch_export_faces():
    output_dir = Path("exports/faces")
    output_dir.mkdir(parents=True, exist_ok=True)

    expressions = ['neutral', 'happy', 'sad', 'angry', 'surprised']

    for expr in expressions:
        widget.set_face_data({'expression': expr})
        # Trigger export via UI or programmatically save pixmap
        print(f"Exported {expr}")

# Or use the built-in batch export button
# This opens a directory picker and exports all automatically
```

## API Reference

### FaceGeneratorWidget

**Constructor**:
```python
FaceGeneratorWidget(config: Optional[FaceGeneratorConfig] = None, parent=None)
```

**Signals**:
- `face_updated(dict)` - Emitted when face data changes
- `export_completed(str)` - Emitted when export completes (receives file path)

**Methods**:

| Method | Description | Returns |
|--------|-------------|---------|
| `get_face_data()` | Get current face configuration | `Dict[str, Any]` |
| `set_face_data(data)` | Set face configuration | `None` |
| `set_body_colors(colors)` | Sync colors from body generator | `None` |
| `update_current_preview()` | Refresh large preview | `None` |
| `update_all_previews()` | Refresh all expression previews | `None` |

### FaceGeneratorConfig

**Dataclass Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bg_primary` | `str` | `"#ffffff"` | Primary background color |
| `bg_secondary` | `str` | `"#f5f7fa"` | Secondary background |
| `accent_primary` | `str` | `"#2563eb"` | Primary accent (buttons) |
| `accent_success` | `str` | `"#10b981"` | Success color |
| `accent_error` | `str` | `"#ef4444"` | Error color |
| `text_primary` | `str` | `"#1e293b"` | Primary text color |
| `font_family` | `str` | `"Segoe UI, ..."` | Font family |
| `font_size_normal` | `int` | `13` | Normal font size |

### FacePreviewWidget

**Constructor**:
```python
FacePreviewWidget(size: int = 128, parent=None)
```

**Methods**:
- `update_face(face_data: Dict[str, Any])` - Update preview with face data

### CharacterGeneratorWidget

**Constructor**:
```python
CharacterGeneratorWidget(config: Optional[FaceGeneratorConfig] = None, parent=None)
```

**Signals**:
- `character_updated(dict)` - Emitted when character data changes
- `export_completed(str)` - Emitted when export completes

## Integration Examples

### Integrating into Existing PyQt5 Application

```python
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from engine.ui import FaceGeneratorWidget, FaceGeneratorConfig

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create central widget
        central = QWidget()
        layout = QVBoxLayout(central)

        # Add face generator
        self.face_gen = FaceGeneratorWidget(FaceGeneratorConfig())
        layout.addWidget(self.face_gen)

        self.setCentralWidget(central)
```

### Adding as a Tab

```python
from PyQt5.QtWidgets import QTabWidget
from engine.ui import FaceGeneratorWidget

tabs = QTabWidget()

# Add face generator tab
face_tab = FaceGeneratorWidget()
tabs.addTab(face_tab, "😊 Face Generator")

# Add other tabs...
```

### Embedding in Existing Nexlify GUI

```python
# In nexlify/gui/cyber_gui.py, add to _create_dashboard_tab():

def _create_dashboard_tab(self):
    tab = QWidget()
    layout = QVBoxLayout(tab)

    # ... existing code ...

    # Add face generator group
    from engine.ui import FaceGeneratorWidget, FaceGeneratorConfig

    face_group = QGroupBox("Character Face Generator")
    face_layout = QVBoxLayout()

    face_config = FaceGeneratorConfig()
    face_widget = FaceGeneratorWidget(face_config)
    face_layout.addWidget(face_widget)

    face_group.setLayout(face_layout)
    layout.addWidget(face_group)

    # ... rest of code ...
```

## Advanced Usage

### Custom Expression Rendering

To customize how expressions are rendered, you can subclass `FacePreviewWidget`:

```python
from engine.ui.face_generator_ui import FacePreviewWidget

class CustomFacePreviewWidget(FacePreviewWidget):
    def render_face(self):
        # Custom rendering logic
        # Access self.face_data for current configuration
        # Draw using QPainter on self
        pass
```

### Extending with More Expressions

```python
from engine.ui import FaceGeneratorWidget

class ExtendedFaceGenerator(FaceGeneratorWidget):
    EXPRESSIONS = [
        'neutral', 'happy', 'sad', 'angry', 'surprised',
        'confused', 'excited', 'scared'  # Additional expressions
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization
```

### Custom Export Formats

```python
def export_to_custom_format(widget, output_path):
    """Export face data to custom format"""
    face_data = widget.get_face_data()

    # Generate pixmap
    from engine.ui.face_generator_ui import FacePreviewWidget
    preview = FacePreviewWidget(size=512)
    preview.update_face(face_data)

    # Convert to custom format
    pixmap = preview.pixmap()
    # ... custom processing ...

    # Save
    pixmap.save(output_path, 'PNG')
```

## Troubleshooting

### Common Issues

**Issue**: Widget doesn't show up
```python
# Make sure to call show() on parent window
window.show()
app.exec_()
```

**Issue**: Colors not updating
```python
# Ensure you call update_all_previews() after color changes
widget.update_all_previews()
```

**Issue**: Export fails
```python
# Check file permissions and path validity
import os
output_dir = "exports"
os.makedirs(output_dir, exist_ok=True)
```

**Issue**: Signals not working
```python
# Make sure to connect signals BEFORE showing widget
widget.face_updated.connect(my_handler)
widget.show()
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Performance Considerations

- **Preview Updates**: Updates are triggered on every color change. For batch updates, consider temporarily disconnecting signals.
- **Large Export Sizes**: Exporting at 512x512 may take longer. Consider showing a progress indicator.
- **Memory**: Each preview widget holds a QPixmap. For memory-constrained environments, reduce preview count.

## Dependencies

- **PyQt5** >= 5.15.0
- **Python** >= 3.9

Install dependencies:
```bash
pip install PyQt5>=5.15.0
```

## Architecture

### Component Hierarchy

```
QWidget
└── FaceGeneratorWidget
    ├── Controls Panel (QWidget)
    │   ├── Expression Selector (QGroupBox)
    │   ├── Color Controls (QGroupBox)
    │   ├── Export Settings (QGroupBox)
    │   └── Action Buttons (QGroupBox)
    └── Preview Panel (QWidget)
        ├── Current Preview (FacePreviewWidget 256x256)
        └── All Expressions Grid (QGroupBox)
            └── 5x FacePreviewWidget (96x96 each)
```

### Design Patterns Used

1. **Dataclass Configuration**: Centralized theme configuration
2. **Signal/Slot Pattern**: Qt's event-driven architecture
3. **Composition**: Widgets composed of smaller components
4. **Separation of Concerns**: UI logic separate from rendering logic

## Future Enhancements

Potential features for future development:

- [ ] More facial features (nose, ears, accessories)
- [ ] Animation preview (expression transitions)
- [ ] Import/export sprite sheets
- [ ] Integration with AI face generation
- [ ] Undo/redo functionality
- [ ] Preset gallery (save favorite faces)
- [ ] Export to game engine formats (Unity, Godot, etc.)
- [ ] Multiplayer avatar synchronization

## License

This module is part of the Nexlify project. See main repository for license details.

## Contributing

When contributing to this module:

1. Follow the existing code style
2. Add type hints to all functions
3. Update this README for new features
4. Add example usage for new functionality
5. Test with multiple PyQt5 versions

## Support

For issues or questions:
- Check the examples directory
- Review the API reference above
- Consult the main Nexlify documentation

---

**Version**: 1.0.0
**Last Updated**: 2025-11-14
**Author**: Nexlify Development Team
