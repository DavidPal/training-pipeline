"""Tests demonstrating the __getitem__() and __setitem__() methods."""

from __future__ import annotations

import unittest


class Indexer:
    """Class that overrides __getitem__() and __setitem__() methods."""

    def __init__(self) -> None:
        """Initialize the object."""
        self.data = [[1, 2, 3], [4, 5, 6]]

    def __getitem__(self, key: int | slice | tuple) -> int | slice | tuple:
        """Returns the value at the given key."""
        return key

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """Returns the value at the given key."""
        self.data[key[0]][key[1]] = value


class TestIndexer(unittest.TestCase):
    """Unit tests for Indexer class."""

    def test_getitem(self) -> None:
        """Tests __getitem__() method."""
        a = Indexer()
        self.assertEqual(a[0], 0)
        self.assertEqual(a[0:1:2], slice(0, 1, 2))
        self.assertEqual(a[0, 1], (0, 1))
        self.assertEqual(a[0:1:2, 3, 4], (slice(0, 1, 2), 3, 4))
        self.assertEqual(a[0:1:2, 3:4:5, 6:7:8], (slice(0, 1, 2), slice(3, 4, 5), slice(6, 7, 8)))

    def test_setitem(self) -> None:
        """Tests __setitem__() method."""
        a = Indexer()
        a[0, 1] = 47
        self.assertEqual(a.data, [[1, 47, 3], [4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
