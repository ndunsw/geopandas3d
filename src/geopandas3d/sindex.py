
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SpatialIndex3D:
    """A lightweight 3D spatial index for point geometries using cKDTree."""
    coords: np.ndarray
    leafsize: int = 16
    _tree: cKDTree | None = None

    def __post_init__(self):
        """Validate coordinates after initialization."""
        if self.coords.size == 0:
            raise ValueError("Cannot create spatial index with empty coordinates")

        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError(f"coords must be (n,3) array, got {self.coords.shape}")

        if not np.isfinite(self.coords).all():
            raise ValueError("Coordinates must be finite numbers")

    def build(self) -> SpatialIndex3D:
        """Build the cKDTree spatial index."""
        if self._tree is not None:
            return self

        self._tree = cKDTree(
            self.coords,
            leafsize=self.leafsize,
            compact_nodes=True,
            balanced_tree=True
        )
        return self

    @property
    def ready(self) -> bool:
        """Check if the spatial index is built and ready."""
        return self._tree is not None

    def query(self, points: Iterable[tuple[float,float,float]], k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for given points.

        Args:
            points: Iterable of (x, y, z) tuples
            k: Number of nearest neighbors to return (default: 1)

        Returns:
            Tuple of (indices, distances) arrays

        Raises:
            ValueError: If k > number of points in index
        """
        if not self.ready:
            self.build()

        if k > len(self.coords):
            raise ValueError(f"k ({k}) cannot be larger than number of points ({len(self.coords)})")

        pts = np.asarray(list(points), dtype='float64')
        if pts.size == 0:
            return np.array([]), np.array([])

        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        if pts.shape[1] != 3:
            raise ValueError(f"Points must be 3D, got shape {pts.shape}")

        if not np.isfinite(pts).all():
            raise ValueError("Query points must be finite numbers")

        try:
            dist, idx = self._tree.query(pts, k=k)
        except Exception as err:
            raise RuntimeError(f"Error querying spatial index: {err}") from err

        # Normalize to arrays
        if k == 1:
            idx = np.asarray(idx)
            dist = np.asarray(dist)
        else:
            idx = np.asarray(idx)
            dist = np.asarray(dist)

        return idx, dist

    def query_ball(self, points: Iterable[tuple[float,float,float]], r: float) -> list[list[int]]:
        """Find all neighbors within radius r for each query point.

        Args:
            points: Iterable of (x, y, z) tuples
            r: Search radius (must be positive)

        Returns:
            List of lists of indices for each query point

        Raises:
            ValueError: If radius is not positive
        """
        if not self.ready:
            self.build()

        if r <= 0:
            raise ValueError(f"Radius must be positive, got {r}")

        pts = np.asarray(list(points), dtype='float64')
        if pts.size == 0:
            return []

        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        if pts.shape[1] != 3:
            raise ValueError(f"Points must be 3D, got shape {pts.shape}")

        if not np.isfinite(pts).all():
            raise ValueError("Query points must be finite numbers")

        try:
            return self._tree.query_ball_point(pts, r)
        except Exception as err:
            raise RuntimeError(f"Error querying spatial index: {err}") from err

    def __len__(self) -> int:
        """Return the number of points in the index."""
        return len(self.coords)

    def __repr__(self) -> str:
        status = "ready" if self.ready else "not built"
        return f"SpatialIndex3D({len(self)} points, {status})"
