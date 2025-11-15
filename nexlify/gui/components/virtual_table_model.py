"""
Virtual Table Model Component
High-performance table model with batch updates
"""

from typing import Dict, List
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer


class VirtualTableModel(QAbstractTableModel):
    """Virtual table model for high-performance data display"""

    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns
        self.data_cache = []
        self.batch_updates = []
        self.batch_timer = QTimer()
        self.batch_timer.timeout.connect(self._apply_batch_updates)
        self.batch_timer.start(100)  # Batch every 100ms

    def add_batch_update(self, data: List[Dict]):
        """Add data to batch update queue"""
        self.batch_updates.extend(data)

    def _apply_batch_updates(self):
        """Apply batched updates"""
        if not self.batch_updates:
            return

        self.beginResetModel()
        self.data_cache = self.batch_updates[-1000:]  # Keep last 1000 items
        self.batch_updates.clear()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.data_cache)

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            row = self.data_cache[index.row()]
            col = self.columns[index.column()]
            return str(row.get(col, ""))

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section]
        return None
