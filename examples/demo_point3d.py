"""
Demonstration of the Point3D class functionality.

This script shows how to use the Point3D class for 3D coordinate
representation and operations.
"""

import numpy as np
from shapely.geometry import Point

# Import our Point3D class
from geopandas3d import Point3D


def demo_basic_usage():
    """Demonstrate basic Point3D usage."""
    print("=== Basic Point3D Usage ===")
    
    # Create points
    point1 = Point3D(1.0, 2.0, 3.0)
    point2 = Point3D(4.0, 5.0, 6.0)
    point3 = Point3D(1.0, 2.0, 3.0, crs="EPSG:4326")
    
    print(f"Point 1: {point1}")
    print(f"Point 2: {point2}")
    print(f"Point 3 (with CRS): {point3}")
    print(f"Point 1 coordinates: x={point1.x}, y={point1.y}, z={point1.z}")
    print(f"Point 3 CRS: {point3.crs}")
    print()


def demo_iteration_and_indexing():
    """Demonstrate iteration and indexing."""
    print("=== Iteration and Indexing ===")
    
    point = Point3D(10.0, 20.0, 30.0)
    
    # Iteration
    print(f"Unpacking: x, y, z = {point}")
    x, y, z = point
    print(f"Unpacked: x={x}, y={y}, z={z}")
    
    # Indexing
    print(f"point[0] = {point[0]}")
    print(f"point[1] = {point[1]}")
    print(f"point[2] = {point[2]}")
    print(f"Length: {len(point)}")
    print()


def demo_conversions():
    """Demonstrate conversion methods."""
    print("=== Conversion Methods ===")
    
    point = Point3D(1.5, 2.5, 3.5)
    
    # Convert to different formats
    coord_tuple = point.to_tuple()
    coord_list = point.to_list()
    coord_array = point.to_array()
    shapely_point = point.to_shapely()
    
    print(f"Original: {point}")
    print(f"To tuple: {coord_tuple}")
    print(f"To list: {coord_list}")
    print(f"To numpy array: {coord_array}")
    print(f"To Shapely Point: {shapely_point}")
    print(f"Shapely Point type: {type(shapely_point)}")
    print()


def demo_class_methods():
    """Demonstrate class methods for creation."""
    print("=== Class Methods for Creation ===")
    
    # From tuple
    point_tuple = Point3D.from_tuple((100.0, 200.0, 300.0))
    print(f"From tuple: {point_tuple}")
    
    # From list
    point_list = Point3D.from_list([150.0, 250.0, 350.0])
    print(f"From list: {point_list}")
    
    # From numpy array
    point_array = Point3D.from_array(np.array([200.0, 300.0, 400.0]))
    print(f"From numpy array: {point_array}")
    
    # From Shapely Point
    shapely_point = Point(250.0, 350.0)
    point_shapely = Point3D.from_shapely(shapely_point, z=450.0)
    print(f"From Shapely Point: {point_shapely}")
    
    # Origin
    origin = Point3D.origin()
    print(f"Origin: {origin}")
    
    origin_crs = Point3D.origin(crs="EPSG:4326")
    print(f"Origin with CRS: {origin_crs}")
    print()


def demo_distance_calculations():
    """Demonstrate distance calculations."""
    print("=== Distance Calculations ===")
    
    # Create points for distance calculations
    origin = Point3D(0.0, 0.0, 0.0)
    point_2d = Point3D(3.0, 4.0, 0.0)  # 5 units away in 2D
    point_3d = Point3D(3.0, 4.0, 12.0)  # 13 units away in 3D (5-12-13 triangle)
    
    # Calculate distances
    dist_2d = origin.distance_to_2d(point_2d)
    dist_3d = origin.distance_to(point_3d)
    
    print(f"Origin: {origin}")
    print(f"2D point: {point_2d}")
    print(f"3D point: {point_3d}")
    print(f"2D distance: {dist_2d}")
    print(f"3D distance: {dist_3d}")
    print()


def demo_midpoint():
    """Demonstrate midpoint calculation."""
    print("=== Midpoint Calculation ===")
    
    point1 = Point3D(0.0, 0.0, 0.0)
    point2 = Point3D(10.0, 20.0, 30.0)
    
    midpoint = point1.midpoint(point2)
    
    print(f"Point 1: {point1}")
    print(f"Point 2: {point2}")
    print(f"Midpoint: {midpoint}")
    print(f"Expected: (5.0, 10.0, 15.0)")
    print()


def demo_validation():
    """Demonstrate validation features."""
    print("=== Validation Features ===")
    
    # Valid points
    valid_point = Point3D(1.0, 2.0, 3.0)
    print(f"Valid point: {valid_point}")
    print(f"Is valid: {valid_point.is_valid()}")
    print(f"Is finite: {valid_point.is_finite()}")
    print(f"Is origin: {valid_point.is_origin()}")
    
    # Origin point
    origin = Point3D.origin()
    print(f"Origin point: {origin}")
    print(f"Is origin: {origin.is_origin()}")
    print()


def demo_error_handling():
    """Demonstrate error handling."""
    print("=== Error Handling ===")
    
    try:
        # Invalid coordinates (string)
        invalid_point = Point3D("1", 2, 3)
    except ValueError as e:
        print(f"Invalid coordinates error: {e}")
    
    try:
        # Invalid coordinates (infinity)
        invalid_point = Point3D(1.0, np.inf, 3.0)
    except ValueError as e:
        print(f"Non-finite coordinates error: {e}")
    
    try:
        # Wrong length for from_tuple
        invalid_point = Point3D.from_tuple((1.0, 2.0))
    except ValueError as e:
        print(f"Wrong length error: {e}")
    
    try:
        # Wrong type for from_shapely
        invalid_point = Point3D.from_shapely("not a point", z=3.0)
    except TypeError as e:
        print(f"Wrong type error: {e}")
    
    print()


def main():
    """Run all demonstrations."""
    print("Point3D Class Demonstration")
    print("=" * 50)
    print()
    
    demo_basic_usage()
    demo_iteration_and_indexing()
    demo_conversions()
    demo_class_methods()
    demo_distance_calculations()
    demo_midpoint()
    demo_validation()
    demo_error_handling()
    
    print("Demonstration complete!")


if __name__ == "__main__":
    main()
