#!/usr/bin/env python3
"""
SumTree Data Structure for Prioritized Experience Replay
Efficient binary tree for O(log n) sampling based on priorities
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SumTree:
    """
    Binary tree data structure for efficient priority-based sampling

    The tree maintains a sum of priorities at each node, allowing
    O(log n) sampling and updates.

    Tree structure:
        - Leaf nodes store priorities and data
        - Internal nodes store sum of children's priorities
        - Root node stores total sum of all priorities

    Operations:
        - add(priority, data): O(log n)
        - sample(value): O(log n)
        - update(idx, priority): O(log n)
        - total(): O(1)

    Example:
        >>> tree = SumTree(capacity=100)
        >>> tree.add(1.0, experience1)
        >>> tree.add(2.0, experience2)
        >>> idx, priority, data = tree.sample(1.5)
    """

    def __init__(self, capacity: int):
        """
        Initialize SumTree with given capacity

        Args:
            capacity: Maximum number of leaf nodes (experiences)

        Raises:
            ValueError: If capacity is not positive
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.capacity = capacity
        self.write_index = 0
        self.n_entries = 0

        # Tree is stored as a flat array
        # Parent at index i has children at 2*i+1 and 2*i+2
        # Tree has capacity-1 internal nodes and capacity leaf nodes
        # Total size = capacity - 1 + capacity = 2*capacity - 1
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

        # Data array stores the actual experiences at leaf positions
        self.data = np.zeros(capacity, dtype=object)

        # Track max priority for new experiences
        self.max_priority = 1.0

        logger.debug(f"SumTree initialized with capacity={capacity}")

    def _propagate(self, idx: int, change: float):
        """
        Propagate priority change up the tree

        Args:
            idx: Tree index where change occurred
            change: Amount of priority change
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        # Recursively propagate to root
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """
        Retrieve leaf index for given cumulative value

        Traverses tree to find leaf where cumulative priority
        contains the given value.

        Args:
            idx: Current tree index (start from root=0)
            value: Target cumulative priority value

        Returns:
            Leaf index in tree
        """
        left = 2 * idx + 1
        right = left + 1

        # If we've reached a leaf (no children), return this index
        if left >= len(self.tree):
            return idx

        # Decide whether to go left or right
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            # Subtract left subtree's sum and go right
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        """
        Get total sum of all priorities

        Returns:
            Sum of all priorities (value at root node)
        """
        return self.tree[0]

    def add(self, priority: float, data: Any):
        """
        Add new experience with given priority

        Args:
            priority: Priority value (must be non-negative)
            data: Experience data to store

        Raises:
            ValueError: If priority is negative
        """
        if priority < 0:
            raise ValueError(f"Priority must be non-negative, got {priority}")

        # Leaf indices start at capacity-1
        idx = self.write_index + self.capacity - 1

        # Store data
        self.data[self.write_index] = data

        # Update tree
        self.update(idx, priority)

        # Update write position (circular buffer)
        self.write_index = (self.write_index + 1) % self.capacity

        # Track number of entries
        if self.n_entries < self.capacity:
            self.n_entries += 1

        # Track max priority
        if priority > self.max_priority:
            self.max_priority = priority

    def update(self, idx: int, priority: float):
        """
        Update priority at given tree index

        Args:
            idx: Tree index to update
            priority: New priority value (must be non-negative)

        Raises:
            ValueError: If priority is negative
        """
        if priority < 0:
            raise ValueError(f"Priority must be non-negative, got {priority}")

        # Calculate change in priority
        change = priority - self.tree[idx]

        # Update this node
        self.tree[idx] = priority

        # Propagate change up the tree
        self._propagate(idx, change)

        # Track max priority
        if priority > self.max_priority:
            self.max_priority = priority

    def sample(self, value: float) -> Tuple[int, float, Any]:
        """
        Sample an experience based on priority value

        Args:
            value: Cumulative priority value (0 <= value <= total())

        Returns:
            Tuple of (tree_idx, priority, data)

        Raises:
            ValueError: If value is out of range
        """
        if value < 0 or value > self.total():
            raise ValueError(
                f"Sample value must be in [0, {self.total():.2f}], got {value:.2f}"
            )

        # Retrieve leaf index
        idx = self._retrieve(0, value)

        # Convert tree index to data index
        data_idx = idx - self.capacity + 1

        # Get priority and data
        priority = self.tree[idx]
        data = self.data[data_idx]

        return idx, priority, data

    def get_leaf(self, idx: int) -> Tuple[float, Any]:
        """
        Get priority and data at given tree index

        Args:
            idx: Tree index

        Returns:
            Tuple of (priority, data)

        Raises:
            IndexError: If index is out of range
        """
        if idx < 0 or idx >= len(self.tree):
            raise IndexError(f"Index {idx} out of range [0, {len(self.tree)})")

        data_idx = idx - self.capacity + 1

        if data_idx < 0 or data_idx >= self.capacity:
            raise IndexError(f"Not a leaf index: {idx}")

        return self.tree[idx], self.data[data_idx]

    def get_max_priority(self) -> float:
        """
        Get maximum priority in tree

        Returns:
            Maximum priority value
        """
        return self.max_priority

    def __len__(self) -> int:
        """
        Get number of entries in tree

        Returns:
            Number of stored experiences
        """
        return self.n_entries

    def __repr__(self) -> str:
        return (
            f"SumTree(capacity={self.capacity}, "
            f"n_entries={self.n_entries}, "
            f"total={self.total():.2f}, "
            f"max_priority={self.max_priority:.4f})"
        )


# Export main class
__all__ = ["SumTree"]
