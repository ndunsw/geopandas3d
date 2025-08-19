"""
Point3D class for 3D coordinate representation.

This module provides a lightweight dataclass for representing 3D points
with x, y, z coordinates. It's designed to work seamlessly with GeoPandas
and the GeoDataFrame3D class.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from shapely.geometry import Point


@dataclass(frozen=True)
class Point3D:
    """
    A lightweight, immutable 3D point representation.

    This class provides a clear way to represent 3D coordinates and
    integrates well with GeoPandas operations. It's designed to be
    a bridge between simple coordinate tuples and full Shapely geometries.

    Attributes:
        x: X coordinate (longitude or easting)
        y: Y coordinate (latitude or northing)
        z: Z coordinate (altitude or height)
        crs: Optional CRS specification (string or CRS object)
    """
    x: float
    y: float
    z: float
    crs: Optional[Union[str, object]] = None

    def __post_init__(self):
        """Validate coordinates after initialization."""
        if not isinstance(self.x, (int, float)) or not isinstance(self.y, (int, float)) or not isinstance(self.z, (int, float)):
            raise ValueError("Coordinates must be numeric values")

        if not np.isfinite(self.x) or not np.isfinite(self.y) or not np.isfinite(self.z):
            raise ValueError("Coordinates must be finite values")

    def __iter__(self):
        """Allow unpacking as (x, y, z)."""
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index):
        """Allow indexing as point[0], point[1], point[2]."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Point3D index must be 0, 1, or 2")

    def __len__(self):
        """Return 3 for 3D points."""
        return 3

    def __repr__(self):
        """String representation."""
        crs_str = f", crs={self.crs}" if self.crs is not None else ""
        return f"Point3D({self.x}, {self.y}, {self.z}{crs_str})"

    def __str__(self):
        """String representation."""
        return f"({self.x}, {self.y}, {self.z})"

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)

    def to_list(self) -> list[float]:
        """Convert to list."""
        return [self.x, self.y, self.z]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    def to_shapely(self) -> Point:
        """Convert to Shapely Point (2D, z coordinate is lost)."""
        return Point(self.x, self.y)

    def distance_to(self, other: 'Point3D') -> float:
        """Calculate 3D Euclidean distance to another point."""
        if not isinstance(other, Point3D):
            raise TypeError("Can only calculate distance to another Point3D")

        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def distance_to_2d(self, other: 'Point3D') -> float:
        """Calculate 2D Euclidean distance to another point (ignoring z)."""
        if not isinstance(other, Point3D):
            raise TypeError("Can only calculate distance to another Point3D")

        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx*dx + dy*dy)

    def midpoint(self, other: 'Point3D') -> 'Point3D':
        """Calculate midpoint between this point and another."""
        if not isinstance(other, Point3D):
            raise TypeError("Can only calculate midpoint with another Point3D")

        mid_x = (self.x + other.x) / 2
        mid_y = (self.y + other.y) / 2
        mid_z = (self.z + other.z) / 2

        # Preserve CRS if both points have the same CRS
        crs = self.crs if self.crs == other.crs else None

        return Point3D(mid_x, mid_y, mid_z, crs)

    def transform(self, transformer) -> 'Point3D':
        """
        Transform coordinates using a pyproj.Transformer.

        Args:
            transformer: pyproj.Transformer instance

        Returns:
            New Point3D with transformed coordinates
        """
        try:
            # Transform x, y coordinates
            x_new, y_new = transformer.transform(self.x, self.y)

            # For 3D transformations, we need to handle z separately
            # This is a simplified approach - in practice, you might want
            # more sophisticated 3D transformation handling
            z_new = self.z

            return Point3D(x_new, y_new, z_new, crs=transformer.target_crs)
        except Exception as e:
            raise ValueError(f"Transformation failed: {e}")

    @classmethod
    def from_tuple(cls, coords: tuple[float, float, float], crs: Optional[Union[str, object]] = None) -> 'Point3D':
        """Create Point3D from tuple."""
        if len(coords) != 3:
            raise ValueError("Tuple must have exactly 3 elements")
        return cls(coords[0], coords[1], coords[2], crs)

    @classmethod
    def from_list(cls, coords: list[float], crs: Optional[Union[str, object]] = None) -> 'Point3D':
        """Create Point3D from list."""
        if len(coords) != 3:
            raise ValueError("List must have exactly 3 elements")
        return cls(coords[0], coords[1], coords[2], crs)

    @classmethod
    def from_array(cls, coords: np.ndarray, crs: Optional[Union[str, object]] = None) -> 'Point3D':
        """Create Point3D from numpy array."""
        if coords.size != 3:
            raise ValueError("Array must have exactly 3 elements")
        return cls(float(coords[0]), float(coords[1]), float(coords[2]), crs)

    @classmethod
    def from_shapely(cls, point: Point, z: float, crs: Optional[Union[str, object]] = None) -> 'Point3D':
        """Create Point3D from Shapely Point and z coordinate."""
        if not isinstance(point, Point):
            raise TypeError("Input must be a Shapely Point")
        return cls(point.x, point.y, z, crs)

    @classmethod
    def origin(cls, crs: Optional[Union[str, object]] = None) -> 'Point3D':
        """Create Point3D at origin (0, 0, 0)."""
        return cls(0.0, 0.0, 0.0, crs)

    def is_origin(self) -> bool:
        """Check if point is at origin."""
        return self.x == 0.0 and self.y == 0.0 and self.z == 0.0

    def is_finite(self) -> bool:
        """Check if all coordinates are finite."""
        return np.isfinite(self.x) and np.isfinite(self.y) and np.isfinite(self.z)

    def is_valid(self) -> bool:
        """Check if point is valid (finite coordinates)."""
        return self.is_finite()
