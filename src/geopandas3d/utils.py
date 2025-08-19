"""
Utility functions for 3D spatial operations.
"""


import numpy as np


def validate_3d_coordinates(coords: np.ndarray) -> bool:
    """Validate that coordinates are valid 3D coordinates.

    Args:
        coords: Array of coordinates

    Returns:
        True if valid, False otherwise
    """
    if coords.size == 0:
        return True

    if coords.ndim != 2 or coords.shape[1] != 3:
        return False

    if not np.isfinite(coords).all():
        return False

    return True


def distance3d(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance between two points.

    Args:
        p1: First point as (x, y, z) tuple
        p2: Second point as (x, y, z) tuple

    Returns:
        3D distance between points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def centroid3d(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Calculate 3D centroid of a list of points.

    Args:
        points: List of (x, y, z) tuples

    Returns:
        Centroid as (x, y, z) tuple
    """
    if not points:
        return (np.nan, np.nan, np.nan)

    points_array = np.array(points)
    centroid = points_array.mean(axis=0)
    return tuple(centroid)


def polygon_area3d(vertices: list[tuple[float, float, float]]) -> float:
    """Calculate area of a 3D polygon projected to 2D.

    Args:
        vertices: List of polygon vertices as (x, y, z) tuples

    Returns:
        Area of the 2D projection
    """
    if len(vertices) < 3:
        return 0.0

    # Extract 2D coordinates for area calculation
    coords_2d = [(v[0], v[1]) for v in vertices]

    # Use shoelace formula for polygon area
    n = len(coords_2d)
    area = 0.0

    for i in range(n):
        j = (i + 1) % n
        area += coords_2d[i][0] * coords_2d[j][1]
        area -= coords_2d[j][0] * coords_2d[i][1]

    return abs(area) / 2.0


def is_point_in_polygon3d(point: tuple[float, float, float],
                          vertices: list[tuple[float, float, float]]) -> bool:
    """Check if a 3D point is inside a 3D polygon (using 2D projection).

    Args:
        point: Point to test as (x, y, z) tuple
        vertices: Polygon vertices as list of (x, y, z) tuples

    Returns:
        True if point is inside polygon, False otherwise
    """
    if len(vertices) < 3:
        return False

    # Extract 2D coordinates for point-in-polygon test
    coords_2d = [(v[0], v[1]) for v in vertices]
    point_2d = (point[0], point[1])

    # Use ray casting algorithm
    n = len(coords_2d)
    inside = False

    for i in range(n):
        j = (i + 1) % n
        if (((coords_2d[i][1] > point_2d[1]) != (coords_2d[j][1] > point_2d[1])) and
            (point_2d[0] < (coords_2d[j][0] - coords_2d[i][0]) *
             (point_2d[1] - coords_2d[i][1]) / (coords_2d[j][1] - coords_2d[i][1]) +
             coords_2d[i][0])):
            inside = not inside

    return inside


def bounds3d(coords_3d):
    """
    Calculate 3D bounds from a numpy array of 3D coordinates.

    Args:
        coords_3d: numpy array of shape (n, 3) with x, y, z coordinates

    Returns:
        tuple: (min_x, min_y, min_z, max_x, max_y, max_z)
    """
    if len(coords_3d) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # Filter out NaN values
    valid_mask = ~np.isnan(coords_3d).any(axis=1)
    if not valid_mask.any():
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    valid_coords = coords_3d[valid_mask]

    min_coords = np.min(valid_coords, axis=0)
    max_coords = np.max(valid_coords, axis=0)

    return (min_coords[0], min_coords[1], min_coords[2],
            max_coords[0], max_coords[1], max_coords[2])


def transform_point3d_batch(points, transformer, preserve_z=True):
    """
    Transform a batch of Point3D objects using a pyproj.Transformer.

    Args:
        points: List or array of Point3D objects
        transformer: pyproj.Transformer instance
        preserve_z: If True, preserve z coordinates (with potential unit conversion)

    Returns:
        List of transformed Point3D objects
    """
    from .point3d import Point3D

    if not points:
        return []

    # Extract coordinates
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    z_coords = [p.z for p in points]

    # Transform x, y coordinates
    try:
        x_new, y_new = transformer.transform(x_coords, y_coords)
    except Exception as err:
        raise ValueError(f"Batch coordinate transformation failed: {err}") from err

    # Handle z coordinates
    z_new = z_coords.copy() if preserve_z else z_coords

    # Create new Point3D objects
    transformed_points = []
    for i, _point in enumerate(points):
        new_point = Point3D(
            x_new[i],
            y_new[i],
            z_new[i],
            crs=transformer.target_crs
        )
        transformed_points.append(new_point)

    return transformed_points


def get_crs_info(crs):
    """
    Get information about a CRS including its type and units.

    Args:
        crs: CRS object (pyproj.CRS, string, or other CRS type)

    Returns:
        dict: Information about the CRS
    """
    from pyproj import CRS

    if isinstance(crs, str):
        crs_obj = CRS.from_string(crs)
    else:
        crs_obj = CRS(crs)

    info = {
        'is_geographic': crs_obj.is_geographic,
        'is_projected': crs_obj.is_projected,
        'is_compound': crs_obj.is_compound,
        'name': crs_obj.name,
        'to_string': str(crs_obj)
    }

    # Handle authority information safely
    try:
        if hasattr(crs_obj, 'authority'):
            info['authority'] = crs_obj.authority
        elif hasattr(crs_obj, 'to_authority'):
            info['authority'] = crs_obj.to_authority()
        else:
            info['authority'] = None
    except Exception:
        info['authority'] = None

    # Get units information
    try:
        if hasattr(crs_obj, 'axis_info'):
            info['units'] = [axis.unit_name for axis in crs_obj.axis_info]
        else:
            info['units'] = None
    except Exception:
        info['units'] = None

    return info


def validate_crs_compatibility(source_crs, target_crs):
    """
    Validate if two CRS are compatible for transformation.

    Args:
        source_crs: Source CRS
        target_crs: Target CRS

    Returns:
        dict: Compatibility information and warnings
    """
    source_info = get_crs_info(source_crs)
    target_info = get_crs_info(target_crs)

    compatibility = {
        'compatible': True,
        'warnings': [],
        'source_info': source_info,
        'target_info': target_info
    }

    # Check for potential issues
    if source_info['is_geographic'] and target_info['is_projected']:
        compatibility['warnings'].append(
            "Transforming from geographic to projected CRS. "
            "Z coordinates may need unit conversion."
        )

    if source_info['is_projected'] and target_info['is_geographic']:
        compatibility['warnings'].append(
            "Transforming from projected to geographic CRS. "
            "Z coordinates may need unit conversion."
        )

    if source_info['is_compound'] or target_info['is_compound']:
        compatibility['warnings'].append(
            "One or both CRS are compound. "
            "Z coordinate handling may be complex."
        )

    return compatibility
