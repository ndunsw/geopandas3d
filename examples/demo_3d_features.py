#!/usr/bin/env python3
"""
Comprehensive demo of geopandas3d 3D features.
This script demonstrates how geopandas3d extends GeoPandas with 3D capabilities.
"""

import pandas as pd

# Try to import GeoPandas (required for geopandas3d)
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    print("Warning: GeoPandas not available. Install with: pip install geopandas")
    GEOPANDAS_AVAILABLE = False

# Import the new geopandas3d
try:
    from geopandas3d import (
        GeoDataFrame3D,
        centroid3d,
        distance3d,
        is_point_in_polygon3d,
        # plot3d,
        polygon_area3d,
    )
    GEOPANDAS3D_AVAILABLE = True
except ImportError:
    print("Warning: geopandas3d not available. Install with: pip install -e .")
    GEOPANDAS3D_AVAILABLE = False

def demo_basic_3d_creation():
    """Demonstrate basic 3D GeoDataFrame creation."""
    print("=== Basic 3D Creation Demo ===")

    if not GEOPANDAS3D_AVAILABLE:
        print("Skipping demo - geopandas3d not available")
        return None

    # Create sample data with x, y, z coordinates
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "x": [0, 10, 20, 15, 25],
        "y": [0, 10, 20, 15, 25],
        "z": [0, 5, 15, 10, 20],
        "value": [100, 200, 300, 250, 400]
    })

    print("Sample data with x, y, z coordinates:")
    print(df)
    print()

    # Create GeoDataFrame3D using the from_xyz constructor
    print("1. Creating GeoDataFrame3D from x, y, z columns:")
    gdf3d = GeoDataFrame3D.from_xyz(df, "x", "y", "z", crs="EPSG:4979", height_col="altitude")
    print(f"   Result: {gdf3d}")
    print()

    # Show that we can access the underlying data like a regular GeoDataFrame
    print("2. Accessing data like a regular GeoDataFrame:")
    print(f"   Number of records: {len(gdf3d)}")
    print(f"   Columns: {list(gdf3d.columns)}")
    print(f"   CRS: {gdf3d.crs}")
    print(f"   Height column: {gdf3d.height_col}")
    print(f"   Geometry type: {gdf3d.geometry_type}")
    print()

    # Show 3D-specific capabilities
    print("3. 3D-specific capabilities:")
    print(f"   3D bounds: {gdf3d.bounds3d()}")
    print(f"   3D coordinates: {gdf3d.get_3d_coordinates()}")
    print()

    return gdf3d

def demo_3d_points():
    """Demonstrate 3D point operations."""
    print("\n=== 3D Points Demo ===")

    if not GEOPANDAS3D_AVAILABLE:
        print("Skipping demo - geopandas3d not available")
        return None

    # Create sample 3D point data
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "x": [0, 10, 20, 15, 25],
        "y": [0, 10, 20, 15, 25],
        "z": [0, 5, 15, 10, 20],
        "value": [100, 200, 300, 250, 400]
    })

    # Create GeoDataFrame3D using the from_xyz constructor
    gdf = GeoDataFrame3D.from_xyz(df, "x", "y", "z", crs="EPSG:4979", height_col="altitude")
    print(f"Created: {gdf}")

    # Basic operations
    print(f"Number of points: {len(gdf)}")
    print(f"3D bounds: {gdf.bounds3d()}")

    # Show the structure
    print(f"Height column: {gdf.height_col}")
    print("Sample data:")
    for i, row in gdf.head().iterrows():
        geom = row["geometry"]
        height = row[gdf.height_col]
        print(f"  Row {i}: 2D geometry {geom}, height {height}")

    # Spatial indexing
    gdf.build_sindex()
    print("Built spatial index")

    # Nearest neighbor queries
    query_point = (12, 12, 8)
    idx, dist = gdf.nearest3d([query_point], k=2)
    print(f"2 nearest neighbors to {query_point}:")
    for i, (neighbor_idx, distance) in enumerate(zip(idx[0], dist[0])):
        neighbor_point = gdf.iloc[neighbor_idx]["geometry"]
        neighbor_height = gdf.iloc[neighbor_idx][gdf.height_col]
        print(f"  {i+1}. Point {neighbor_idx}: 2D {neighbor_point}, height {neighbor_height}, Distance: {distance:.2f}")

    # Radius queries
    neighbors = gdf.query_ball3d([query_point], r=10.0)
    print(f"Points within radius 10: {neighbors[0]}")

    return gdf

def demo_3d_polygons():
    """Demonstrate 3D polygon operations."""
    print("\n=== 3D Polygons Demo ===")

    if not GEOPANDAS3D_AVAILABLE:
        print("Skipping demo - geopandas3d not available")
        return None

    # Create sample 3D polygon data
    polygons = [
        # Square polygon at z=0
        [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0)],
        # Triangle polygon at z=10
        [(10, 10, 10), (15, 10, 10), (12.5, 15, 10)],
        # Complex polygon at z=20
        [(20, 20, 20), (25, 20, 20), (25, 25, 20), (22.5, 27, 20), (20, 25, 20)]
    ]

    # Create GeoDataFrame3D from polygons
    gdf = GeoDataFrame3D.from_polygons(polygons, crs="EPSG:4979", height_col="altitude")
    print(f"Created: {gdf}")

    # Basic operations
    print(f"Number of polygons: {len(gdf)}")
    print(f"3D bounds: {gdf.bounds3d()}")

    # Show the structure
    print(f"Height column: {gdf.height_col}")
    print("Sample data:")
    for i, row in gdf.head().iterrows():
        geom = row["geometry"]
        height = row[gdf.height_col]
        print(f"  Row {i}: 2D geometry {geom}, height {height}")

    # Individual geometry bounds
    bounds_list = gdf.get_geometry_bounds3d()
    for i, bounds in enumerate(bounds_list):
        print(f"Polygon {i+1} bounds: {bounds}")

    # Polygon properties
    for i, geom in enumerate(gdf.geometry):
        if geom is not None:
            # Extract vertices for area calculation
            if hasattr(geom, 'exterior'):
                vertices = list(geom.exterior.coords)
            else:
                vertices = list(geom.coords)

            # Add height to vertices for 3D calculations
            height = gdf.iloc[i][gdf.height_col]
            vertices_3d = [(v[0], v[1], height) for v in vertices]

            area = polygon_area3d(vertices_3d)
            centroid = centroid3d(vertices_3d)
            print(f"Polygon {i+1}: Area = {area:.2f}, Centroid = {centroid}")

    # Point-in-polygon tests
    test_points = [(2.5, 2.5, 0), (12.5, 12.5, 10), (30, 30, 30)]
    for _i, point in enumerate(test_points):
        for j, geom in enumerate(gdf.geometry):
            if geom is not None:
                # Extract vertices and add height
                if hasattr(geom, 'exterior'):
                    vertices = list(geom.exterior.coords)
                else:
                    vertices = list(geom.coords)

                height = gdf.iloc[j][gdf.height_col]
                vertices_3d = [(v[0], v[1], height) for v in vertices]

                inside = is_point_in_polygon3d(point, vertices_3d)
                print(f"Point {point} {'inside' if inside else 'outside'} polygon {j+1}")

    return gdf

def demo_spatial_joins():
    """Demonstrate spatial join operations."""
    print("\n=== Spatial Joins Demo ===")

    if not GEOPANDAS3D_AVAILABLE:
        print("Skipping demo - geopandas3d not available")
        return None

    # Create two datasets
    left_df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "x": [2, 12, 22, 17],
        "y": [2, 12, 22, 17],
        "z": [2, 7, 17, 12]
    })

    right_df = pd.DataFrame({
        "oid": [100, 101, 102],
        "x": [1, 30, 18],
        "y": [1, 30, 18],
        "z": [1, 30, 18]
    })

    left_gdf = GeoDataFrame3D.from_xyz(left_df, "x", "y", "z", crs="EPSG:4979", height_col="altitude")
    right_gdf = GeoDataFrame3D.from_xyz(right_df, "x", "y", "z", crs="EPSG:4979", height_col="altitude")

    # Build spatial indexes
    left_gdf.build_sindex()
    right_gdf.build_sindex()

    # Nearest neighbor join
    nearest_join = left_gdf.sjoin_nearest3d(right_gdf, k=1, how="left")
    print(f"Nearest neighbor join: {len(nearest_join)} rows")
    if len(nearest_join) > 0:
        print(nearest_join[["id_l", "oid_r", "distance3d"]].head())

    # Within distance join
    within_join = left_gdf.sjoin_within_distance3d(right_gdf, max_distance=8.0, how="inner")
    print(f"Within distance join (r=8): {len(within_join)} rows")
    if len(within_join) > 0:
        print(within_join[["id_l", "oid_r", "distance3d"]].head())

    return left_gdf, right_gdf

def demo_plotting():
    """Demonstrate 3D plotting capabilities."""
    print("\n=== 3D Plotting Demo ===")

    if not GEOPANDAS3D_AVAILABLE:
        print("Skipping demo - geopandas3d not available")
        return None

    # Create sample data for plotting
    points_df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "x": [0, 10, 20, 15, 25],
        "y": [0, 10, 20, 15, 25],
        "z": [0, 5, 15, 10, 20],
        "value": [100, 200, 300, 250, 400]
    })

    points_gdf = GeoDataFrame3D.from_xyz(points_df, "x", "y", "z", crs="EPSG:4979", height_col="altitude")

    # Create polygon data
    polygons = [
        [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0)],
        [(10, 10, 10), (15, 10, 10), (12.5, 15, 10)]
    ]

    polygons_gdf = GeoDataFrame3D.from_polygons(polygons, crs="EPSG:4979", height_col="altitude")

    print("Created sample data for plotting")
    print(f"Points: {points_gdf}")
    print(f"Polygons: {polygons_gdf}")

    # Note: In a real environment, you would call:
    # fig, ax = points_gdf.plot3d(column="value", cmap="viridis")
    # plt.show()

    print("Plotting functions available:")
    print("- gdf.plot3d() for matplotlib 3D plots")
    print("- plot3d(gdf) for standalone plotting function")

    return points_gdf, polygons_gdf

def demo_utility_functions():
    """Demonstrate utility functions."""
    print("\n=== Utility Functions Demo ===")

    if not GEOPANDAS3D_AVAILABLE:
        print("Skipping demo - geopandas3d not available")
        return

    # Distance calculations
    p1 = (0, 0, 0)
    p2 = (1, 1, 1)
    dist = distance3d(p1, p2)
    print(f"Distance between {p1} and {p2}: {dist:.3f}")

    # Centroid calculations
    points = [(0, 0, 0), (2, 0, 0), (1, 2, 0)]
    centroid = centroid3d(points)
    print(f"Centroid of triangle {points}: {centroid}")

    # Polygon area
    polygon_vertices = [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0)]
    area = polygon_area3d(polygon_vertices)
    print(f"Area of square polygon: {area:.2f}")

    # Point in polygon test
    test_point = (2.5, 2.5, 0)
    inside = is_point_in_polygon3d(test_point, polygon_vertices)
    print(f"Point {test_point} is {'inside' if inside else 'outside'} the polygon")

def demo_realistic_workflow():
    """Demonstrate a realistic workflow extending existing GeoPandas data."""
    print("\n=== Realistic Workflow Demo ===")

    if not GEOPANDAS_AVAILABLE or not GEOPANDAS3D_AVAILABLE:
        print("Skipping realistic workflow demo - GeoPandas or geopandas3d not available")
        return None

    try:
        from shapely.geometry import Point

        print("1. Starting with existing 2D GeoPandas data:")
        # Simulate loading existing 2D data
        existing_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Point A", "Point B", "Point C"],
            "category": ["A", "B", "A"]
        })

        # Create 2D geometries (as you might have from a file)
        geometries_2d = [Point(0, 0), Point(10, 10), Point(20, 20)]

        gdf_2d = gpd.GeoDataFrame(existing_df, geometry=geometries_2d, crs="EPSG:4326")
        print(f"   Original 2D GeoDataFrame: {gdf_2d}")
        print(f"   CRS: {gdf_2d.crs}")
        print(f"   Geometry type: {gdf_2d.geometry.geom_type.iloc[0]}")
        print()

        print("2. Adding 3D coordinates to existing data:")
        # Add z-coordinates (maybe from elevation data, LiDAR, etc.)
        gdf_2d["z"] = [0, 5, 15]  # Add elevation data
        gdf_2d["x"] = gdf_2d.geometry.x  # Extract x from 2D geometry
        gdf_2d["y"] = gdf_2d.geometry.y  # Extract y from 2D geometry

        print("   Added z-coordinates and extracted x, y from 2D geometries:")
        print(f"   Columns: {list(gdf_2d.columns)}")
        print(gdf_2d[["id", "name", "x", "y", "z"]])
        print()

        print("3. Creating GeoDataFrame3D for 3D operations:")
        # Create 3D version for 3D operations
        gdf3d = GeoDataFrame3D.from_xyz(
            gdf_2d[["id", "name", "category", "x", "y", "z"]],
            "x", "y", "z",
            crs="EPSG:4979"  # 3D CRS
        )
        print(f"   Result: {gdf3d}")
        print()

        print("4. Using both 2D and 3D capabilities:")
        print("   Original 2D operations still work:")
        print(f"   - 2D bounds: {gdf_2d.bounds}")
        print(f"   - 2D centroid: {gdf_2d.geometry.centroid.iloc[0]}")
        print()

        print("   New 3D operations available:")
        print(f"   - 3D bounds: {gdf3d.bounds3d()}")
        print(f"   - 3D geometry type: {gdf3d.geometry_type}")

        # Build 3D spatial index
        gdf3d.build_sindex()
        print("   - Built 3D spatial index")

        # Demonstrate 3D query
        query_point = (5, 5, 2)
        idx, dist = gdf3d.nearest3d([query_point], k=1)

        # Handle the result structure properly - idx and dist are numpy arrays
        if idx.size > 0:
            # For k=1, idx and dist are 1D arrays
            nearest_idx = idx[0]
            nearest_name = gdf3d.iloc[nearest_idx]["name"]
            print(f"   - Nearest to {query_point}: {nearest_name} (distance: {dist[0]:.2f})")
        else:
            print(f"   - No nearest neighbor found for {query_point}")
        print()

        print("5. Workflow benefits:")
        print("   ✓ Keep existing 2D GeoPandas workflows unchanged")
        print("   ✓ Add 3D capabilities incrementally")
        print("   ✓ Use 2D methods for 2D operations")
        print("   ✓ Use 3D methods for 3D operations")
        print("   ✓ No need to rewrite existing code")

        return gdf3d

    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure shapely is installed: pip install shapely")
        return None

def main():
    """Run all demos."""
    print("geopandas3d 3D Features Demo")
    print("=" * 50)
    print("This package extends GeoPandas with 3D capabilities")
    print("=" * 50)

    try:
        # Run all demos
        demo_basic_3d_creation()
        demo_3d_points()
        demo_3d_polygons()
        left_gdf, right_gdf = demo_spatial_joins()
        plot_points, plot_polygons = demo_plotting()
        demo_utility_functions()
        demo_realistic_workflow()

        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Properly inherits from GeoPandas GeoDataFrame")
        print("- Required altitude/height column for 3D operations")
        print("- 3D point and polygon support")
        print("- Automatic geometry type detection")
        print("- 3D spatial indexing with cKDTree")
        print("- 3D nearest neighbor and radius queries")
        print("- 3D spatial joins (nearest and within distance)")
        print("- 3D plotting capabilities with matplotlib")
        print("- Utility functions for geometry operations")
        print("\nIntegration Benefits:")
        print("- All existing GeoPandas methods work unchanged")
        print("- Seamless addition of 3D capabilities to existing workflows")
        print("- No need to rewrite existing code")
        print("- Incremental adoption of 3D features")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
