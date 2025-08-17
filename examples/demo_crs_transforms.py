"""
Demonstration of CRS transformation functionality with 3D awareness.

This script shows how to use the new to_crs3d method and related
utility functions for transforming coordinates between different CRS
while preserving 3D information.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Import our geopandas3d functionality
from geopandas3d import (
    GeoDataFrame3D, 
    Point3D, 
    transform_point3d_batch,
    get_crs_info,
    validate_crs_compatibility
)


def demo_basic_crs_transformation():
    """Demonstrate basic CRS transformation."""
    print("=== Basic CRS Transformation ===")
    
    # Create sample data in WGS84 (EPSG:4326)
    coords_3d = [
        (-122.4194, 37.7749, 100),  # San Francisco
        (-74.0060, 40.7128, 50),   # New York
        (-118.2437, 34.0522, 200),  # Los Angeles
    ]
    
    # Create GeoDataFrame3D
    gdf3d = GeoDataFrame3D.from_points(
        coords_3d, 
        crs="EPSG:4326"
    )
    
    print(f"Original CRS: {gdf3d.crs}")
    print(f"Original coordinates:")
    print(gdf3d.get_3d_coordinates())
    print()
    
    # Transform to Web Mercator (EPSG:3857)
    gdf3d_mercator = gdf3d.to_crs3d("EPSG:3857")
    
    print(f"Transformed CRS: {gdf3d_mercator.crs}")
    print(f"Transformed coordinates:")
    print(gdf3d_mercator.get_3d_coordinates())
    print()
    
    # Transform back to WGS84
    gdf3d_back = gdf3d_mercator.to_crs3d("EPSG:4326")
    
    print(f"Back to original CRS: {gdf3d_back.crs}")
    print(f"Coordinates after round-trip:")
    print(gdf3d_back.get_3d_coordinates())
    print()


def demo_advanced_transformation():
    """Demonstrate advanced transformation features."""
    print("=== Advanced Transformation Features ===")
    
    # Create data in a projected CRS (UTM Zone 10N)
    coords_3d = [
        (500000, 4000000, 150),  # UTM coordinates
        (510000, 4010000, 75),
        (505000, 4005000, 300),
    ]
    
    gdf3d_utm = GeoDataFrame3D.from_points(
        coords_3d, 
        crs="EPSG:32610"  # UTM Zone 10N
    )
    
    print(f"UTM CRS: {gdf3d_utm.crs}")
    print(f"UTM coordinates:")
    print(gdf3d_utm.get_3d_coordinates())
    print()
    
    # Transform to WGS84
    gdf3d_wgs84 = gdf3d_utm.to_crs3d("EPSG:4326")
    
    print(f"WGS84 coordinates:")
    print(gdf3d_wgs84.get_3d_coordinates())
    print()
    
    # Transform to another UTM zone
    gdf3d_utm11 = gdf3d_utm.to_crs3d("EPSG:32611")  # UTM Zone 11N
    
    print(f"UTM Zone 11N coordinates:")
    print(gdf3d_utm11.get_3d_coordinates())
    print()


def demo_custom_transformer():
    """Demonstrate using custom transformers."""
    print("=== Custom Transformer Usage ===")
    
    # Create sample data
    coords_3d = [
        (-122.4194, 37.7749, 100),
        (-74.0060, 40.7128, 50),
    ]
    
    gdf3d = GeoDataFrame3D.from_points(
        coords_3d, 
        crs="EPSG:4326"
    )
    
    # Get a custom transformer
    transformer = gdf3d.get_transformer("EPSG:3857")
    
    print(f"Transformer info:")
    print(f"  Source CRS: {transformer.source_crs}")
    print(f"  Target CRS: {transformer.target_crs}")
    print()
    
    # Use the transformer
    gdf3d_transformed = gdf3d.transform3d(transformer)
    
    print(f"Transformed coordinates:")
    print(gdf3d_transformed.get_3d_coordinates())
    print()


def demo_point3d_batch_transformation():
    """Demonstrate batch transformation of Point3D objects."""
    print("=== Point3D Batch Transformation ===")
    
    # Create Point3D objects
    point3d_list = [
        Point3D(-122.4194, 37.7749, 100, crs="EPSG:4326"),
        Point3D(-74.0060, 40.7128, 50, crs="EPSG:4326"),
        Point3D(-118.2437, 34.0522, 200, crs="EPSG:4326"),
    ]
    
    print("Original Point3D objects:")
    for p in point3d_list:
        print(f"  {p}")
    print()
    
    # Create transformer
    import pyproj
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    
    # Transform batch
    transformed_points = transform_point3d_batch(point3d_list, transformer)
    
    print("Transformed Point3D objects:")
    for p in transformed_points:
        print(f"  {p}")
    print()


def demo_crs_validation():
    """Demonstrate CRS validation and compatibility checking."""
    print("=== CRS Validation and Compatibility ===")
    
    # Check different CRS combinations
    crs_combinations = [
        ("EPSG:4326", "EPSG:3857"),  # Geographic to Projected
        ("EPSG:3857", "EPSG:4326"),  # Projected to Geographic
        ("EPSG:32610", "EPSG:32611"), # Projected to Projected
        ("EPSG:4326", "EPSG:4326"),  # Same CRS
    ]
    
    for source, target in crs_combinations:
        print(f"Source: {source} -> Target: {target}")
        
        compatibility = validate_crs_compatibility(source, target)
        
        print(f"  Compatible: {compatibility['compatible']}")
        if compatibility['warnings']:
            print("  Warnings:")
            for warning in compatibility['warnings']:
                print(f"    - {warning}")
        
        source_info = compatibility['source_info']
        target_info = compatibility['target_info']
        
        print(f"  Source: {source_info['name']} ({'Geographic' if source_info['is_geographic'] else 'Projected'})")
        print(f"  Target: {target_info['name']} ({'Geographic' if target_info['is_geographic'] else 'Projected'})")
        print()


def demo_inplace_transformation():
    """Demonstrate inplace transformation."""
    print("=== Inplace Transformation ===")
    
    # Create sample data
    coords_3d = [
        (-122.4194, 37.7749, 100),
        (-74.0060, 40.7128, 50),
    ]
    
    gdf3d = GeoDataFrame3D.from_points(
        coords_3d, 
        crs="EPSG:4326"
    )
    
    print(f"Original CRS: {gdf3d.crs}")
    print(f"Original coordinates:")
    print(gdf3d.get_3d_coordinates())
    print()
    
    # Transform inplace
    gdf3d.to_crs3d("EPSG:3857", inplace=True)
    
    print(f"After inplace transformation:")
    print(f"CRS: {gdf3d.crs}")
    print(f"Coordinates:")
    print(gdf3d.get_3d_coordinates())
    print()


def main():
    """Run all demonstrations."""
    print("CRS Transformation with 3D Awareness Demonstration")
    print("=" * 60)
    print()
    
    demo_basic_crs_transformation()
    demo_advanced_transformation()
    demo_custom_transformer()
    demo_point3d_batch_transformation()
    demo_crs_validation()
    demo_inplace_transformation()
    
    print("Demonstration complete!")


if __name__ == "__main__":
    main()
