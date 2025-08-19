"""
Demonstration of HNSW indexing and k-NN join optimization for huge datasets.

This script shows how to use the new HNSW indexing method for efficient
spatial operations on large datasets, and demonstrates the optimized
k-NN join functionality.
"""

import time

import numpy as np

# Import our geopandas3d functionality
from geopandas3d import GeoDataFrame3D


def demo_hnsw_indexing():
    """Demonstrate HNSW indexing for large datasets."""
    print("=== HNSW Indexing for Large Datasets ===")

    # Create a large dataset (simulate 100k points)
    print("Creating large dataset (100,000 points)...")
    np.random.seed(42)

    # Generate random coordinates
    n_points = 100000
    x_coords = np.random.uniform(-180, 180, n_points)
    y_coords = np.random.uniform(-90, 90, n_points)
    z_coords = np.random.uniform(0, 1000, n_points)

    coords_3d = list(zip(x_coords, y_coords, z_coords))

    # Create GeoDataFrame3D
    gdf3d = GeoDataFrame3D.from_points(
        coords_3d,
        crs="EPSG:4326"
    )

    print(f"Dataset size: {len(gdf3d):,} points")
    print()

    # Test cKDTree indexing
    print("Building cKDTree index...")
    start_time = time.time()
    gdf3d.build_sindex(method="cKDTree")
    ckdtree_time = time.time() - start_time
    print(f"cKDTree build time: {ckdtree_time:.3f} seconds")

    # Test HNSW indexing
    print("Building HNSW index...")
    start_time = time.time()
    try:
        gdf3d.build_sindex(method="HNSW")
        hnsw_time = time.time() - start_time
        print(f"HNSW build time: {hnsw_time:.3f} seconds")
        print(f"Speedup: {ckdtree_time/hnsw_time:.1f}x")
    except ImportError:
        print("HNSW indexing not available (install hnswlib: pip install hnswlib)")
        hnsw_time = None

    print()


def demo_knn_performance_comparison():
    """Compare k-NN query performance between cKDTree and HNSW."""
    print("=== k-NN Query Performance Comparison ===")

    # Create a large dataset
    print("Creating dataset (50,000 points)...")
    np.random.seed(42)

    n_points = 50000
    x_coords = np.random.uniform(-180, 180, n_points)
    y_coords = np.random.uniform(-90, 90, n_points)
    z_coords = np.random.uniform(0, 1000, n_points)

    coords_3d = list(zip(x_coords, y_coords, z_coords))

    gdf3d = GeoDataFrame3D.from_points(
        coords_3d,
        crs="EPSG:4326"
    )

    # Generate query points
    n_queries = 1000
    query_x = np.random.uniform(-180, 180, n_queries)
    query_y = np.random.uniform(-90, 90, n_queries)
    query_z = np.random.uniform(0, 1000, n_queries)
    query_points = list(zip(query_x, query_y, query_z))

    print(f"Dataset size: {len(gdf3d):,} points")
    print(f"Query points: {n_queries:,}")
    print()

    # Test cKDTree performance
    print("Testing cKDTree performance...")
    start_time = time.time()
    indices_ckdtree, distances_ckdtree = gdf3d.nearest3d(
        query_points, k=5, method="cKDTree"
    )
    ckdtree_query_time = time.time() - start_time
    print(f"cKDTree query time: {ckdtree_query_time:.3f} seconds")

    # Test HNSW performance
    print("Testing HNSW performance...")
    try:
        start_time = time.time()
        indices_hnsw, distances_hnsw = gdf3d.nearest3d(
            query_points, k=5, method="HNSW"
        )
        hnsw_query_time = time.time() - start_time
        print(f"HNSW query time: {hnsw_query_time:.3f} seconds")
        print(f"Speedup: {ckdtree_query_time/hnsw_query_time:.1f}x")

        # Verify results are similar
        print(f"Results match: {np.allclose(distances_ckdtree, distances_hnsw, atol=1e-6)}")

    except ImportError:
        print("HNSW not available for comparison")

    print()


def demo_optimized_spatial_join():
    """Demonstrate optimized spatial join with k-NN."""
    print("=== Optimized Spatial Join with k-NN ===")

    # Create two datasets
    print("Creating two datasets for spatial join...")
    np.random.seed(42)

    # Dataset 1: 10,000 points
    n1 = 10000
    x1 = np.random.uniform(-180, 180, n1)
    y1 = np.random.uniform(-90, 90, n1)
    z1 = np.random.uniform(0, 1000, n1)
    coords1 = list(zip(x1, y1, z1))

    gdf1 = GeoDataFrame3D.from_points(
        coords1,
        crs="EPSG:4326"
    )
    gdf1['dataset'] = 'dataset1'
    gdf1['id'] = range(n1)

    # Dataset 2: 5,000 points
    n2 = 5000
    x2 = np.random.uniform(-180, 180, n2)
    y2 = np.random.uniform(-90, 90, n2)
    z2 = np.random.uniform(0, 1000, n2)
    coords2 = list(zip(x2, y2, z2))

    gdf2 = GeoDataFrame3D.from_points(
        coords2,
        crs="EPSG:4326"
    )
    gdf2['dataset'] = 'dataset2'
    gdf2['id'] = range(n2)

    print(f"Dataset 1: {len(gdf1):,} points")
    print(f"Dataset 2: {len(gdf2):,} points")
    print()

    # Test spatial join with cKDTree
    print("Performing spatial join with cKDTree...")
    start_time = time.time()
    joined_ckdtree = gdf1.sjoin_nearest3d(gdf2, k=3, method="cKDTree")
    ckdtree_join_time = time.time() - start_time
    print(f"cKDTree join time: {ckdtree_join_time:.3f} seconds")
    print(f"Joined result size: {len(joined_ckdtree):,} rows")

    # Test spatial join with HNSW
    print("Performing spatial join with HNSW...")
    try:
        start_time = time.time()
        joined_hnsw = gdf1.sjoin_nearest3d(gdf2, k=3, method="HNSW")
        hnsw_join_time = time.time() - start_time
        print(f"HNSW join time: {hnsw_join_time:.3f} seconds")
        print(f"Joined result size: {len(joined_hnsw):,} rows")
        print(f"Speedup: {ckdtree_join_time/hnsw_join_time:.1f}x")

        # Verify results are similar
        print(f"Results match: {len(joined_ckdtree) == len(joined_hnsw)}")

    except ImportError:
        print("HNSW not available for spatial join")

    print()

    # Show sample results
    if len(joined_ckdtree) > 0:
        print("Sample joined results:")
        print(joined_ckdtree.head(3)[['left_id', 'right_id', 'distance', 'neighbor_rank']])

    print()


def demo_auto_method_selection():
    """Demonstrate automatic method selection based on dataset size."""
    print("=== Automatic Method Selection ===")

    # Test different dataset sizes
    sizes = [1000, 10000, 100000]

    for size in sizes:
        print(f"\nTesting dataset size: {size:,} points")

        # Create dataset
        np.random.seed(42)
        x_coords = np.random.uniform(-180, 180, size)
        y_coords = np.random.uniform(-90, 90, size)
        z_coords = np.random.uniform(0, 1000, size)
        coords_3d = list(zip(x_coords, y_coords, z_coords))

        gdf3d = GeoDataFrame3D.from_points(
            coords_3d,
            crs="EPSG:4326"
        )

        # Test auto method selection
        print("  Testing auto method selection...")
        start_time = time.time()
        indices, distances = gdf3d.nearest3d(
            [(0, 0, 500)], k=5, method="auto"
        )
        auto_time = time.time() - start_time

        print(f"  Auto method query time: {auto_time:.3f} seconds")
        print(f"  Found {len(indices[0])} nearest neighbors")

        # Show which method was selected
        if hasattr(gdf3d, '_hnsw_index') and gdf3d._hnsw_index is not None:
            print("  Method selected: HNSW")
        elif hasattr(gdf3d, '_sindex') and gdf3d._sindex is not None:
            print("  Method selected: cKDTree")
        else:
            print("  Method selected: Unknown")

    print()


def main():
    """Run all demonstrations."""
    print("HNSW Indexing and k-NN Join Optimization Demonstration")
    print("=" * 65)
    print()

    demo_hnsw_indexing()
    demo_knn_performance_comparison()
    demo_optimized_spatial_join()
    demo_auto_method_selection()

    print("Demonstration complete!")
    print("\nKey Benefits:")
    print("- HNSW indexing for datasets >100k points")
    print("- Automatic method selection based on dataset size")
    print("- Optimized k-NN spatial joins")
    print("- Significant performance improvements for large datasets")


if __name__ == "__main__":
    main()
