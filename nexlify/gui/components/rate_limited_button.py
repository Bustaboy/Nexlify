"""
Rate Limited Button Component
Button with built-in rate limiting and loading states
"""

from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QTimer

# Default debounce constant
DEBOUNCE_INSTANT = 100  # ms for normal actions


class RateLimitedButton(QPushButton):
    """Button with built-in rate limiting and loading states"""

    def __init__(self, text: str, debounce_ms: int = DEBOUNCE_INSTANT):
        super().__init__(text)
        self.debounce_ms = debounce_ms
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._enable_button)
        self.loading = False
        self.original_text = text
        self._setup_loading_animation()

    def _setup_loading_animation(self):
        """Setup loading spinner animation"""
        # Use text-based loading animation (no external spinner file needed)
        self.loading_timer = QTimer()
        self.loading_timer.timeout.connect(self._update_loading_text)
        self.loading_dots = 0
        self.loading_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.loading_frame_index = 0

    def click_with_debounce(self):
        """Handle click with debounce"""
        if not self.isEnabled():
            return

        self.setEnabled(False)
        self.debounce_timer.start(self.debounce_ms)

        # Emit clicked signal
        self.clicked.emit()

    def _enable_button(self):
        """Re-enable button after debounce"""
        if not self.loading:
            self.setEnabled(True)

    def set_loading(self, loading: bool):
        """Set loading state"""
        self.loading = loading
        if loading:
            self.setEnabled(False)
            self.loading_timer.start(500)  # Update every 500ms
            self._update_loading_text()
        else:
            self.loading_timer.stop()
            self.setText(self.original_text)
            self.setEnabled(True)

    def _update_loading_text(self):
        """Update loading animation text"""
        if hasattr(self, "loading_frames"):
            # Use spinner animation
            frame = self.loading_frames[
                self.loading_frame_index % len(self.loading_frames)
            ]
            self.setText(f"{frame} {self.original_text}")
            self.loading_frame_index += 1
        else:
            # Fallback to dots
            dots = "." * (self.loading_dots % 4)
            self.setText(f"{self.original_text}{dots}")
            self.loading_dots += 1
