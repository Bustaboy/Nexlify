"""
Log Widget Component
Memory-efficient log widget with size-based rotation
"""

from datetime import datetime
from PyQt5.QtWidgets import QPlainTextEdit
from PyQt5.QtGui import QTextCursor

# Default log size constant
LOG_MAX_SIZE_MB = 25  # Maximum log size in MB


class LogWidget(QPlainTextEdit):
    """Memory-efficient log widget with size-based rotation"""

    def __init__(self, max_size_mb: float = LOG_MAX_SIZE_MB):
        super().__init__()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.setReadOnly(True)
        self.setMaximumBlockCount(10000)  # Limit blocks
        self.document().setMaximumBlockCount(10000)

    def append_log(self, message: str, level: str = "INFO"):
        """Append log with size checking"""
        # Check document size
        doc_size = len(self.toPlainText().encode("utf-8"))
        if doc_size > self.max_size_bytes:
            # Remove first 20% of content
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(
                QTextCursor.Down,
                QTextCursor.KeepAnchor,
                self.document().blockCount() // 5,
            )
            cursor.removeSelectedText()

        # Add timestamp and format
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color based on level - Modern colors for light theme
        colors = {
            "INFO": "#10b981",  # Green
            "WARNING": "#f59e0b",  # Amber
            "ERROR": "#ef4444",  # Red
            "DEBUG": "#2563eb",  # Blue
        }
        color = colors.get(level, "#1e293b")

        # Append with HTML formatting
        self.appendHtml(
            f'<span style="color: #64748b">[{timestamp}]</span> '
            f'<span style="color: {color}; font-weight: 600">[{level}]</span> '
            f'<span style="color: #1e293b">{message}</span>'
        )

        # Auto-scroll to bottom
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
