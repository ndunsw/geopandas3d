"""
3D extensions to GeoPandas for altitude-aware spatial operations.

This package extends GeoPandas with 3D spatial capabilities, providing
efficient 3D spatial indexing, joins, and operations while maintaining
full compatibility with existing GeoPandas workflows.
"""

from .geodataframe3d import GeoDataFrame3D
from .plotting import plot3d, plot_points_3d, plot_polygons_3d
from .point3d import Point3D
from .utils import (
    bounds3d,
    centroid3d,
    distance3d,
    get_crs_info,
    is_point_in_polygon3d,
    polygon_area3d,
    transform_point3d_batch,
    validate_3d_coordinates,
    validate_crs_compatibility,
)

__version__ = "0.3.0-dev"
__author__ = "geopandas3d contributors"

__all__ = [
    "GeoDataFrame3D",
    "Point3D",
    "distance3d",
    "centroid3d",
    "polygon_area3d",
    "is_point_in_polygon3d",
    "bounds3d",
    "validate_3d_coordinates",
    "transform_point3d_batch",
    "get_crs_info",
    "validate_crs_compatibility",
    "plot3d",
    "plot_points_3d",
    "plot_polygons_3d",
]
