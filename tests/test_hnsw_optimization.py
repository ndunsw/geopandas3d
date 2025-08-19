"""
Tests for HNSW indexing and k-NN join optimization features.
"""

import numpy as np
import pytest

from geopandas3d import GeoDataFrame3D


class TestHNSWOptimization:
    """Test HNSW indexing and optimization features."""

    def test_build_sindex_methods(self):
        """Test building different types of spatial indexes."""
        # Create test data
        coords = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        gdf3d = GeoDataFrame3D.from_points(coords, crs="EPSG:4326")

        # Test cKDTree indexing
        ckdtree_index = gdf3d.build_sindex(method="cKDTree")
        assert ckdtree_index is not None

        # Test HNSW indexing (if available)
        try:
            hnsw_index = gdf3d.build_sindex(method="HNSW")
            assert hnsw_index is not None
        except ImportError:
            # HNSW not available, skip test
            pytest.skip("HNSW indexing not available")

    def test_nearest3d_method_selection(self):
        """Test automatic method selection in nearest3d."""
        # Create small dataset (should use cKDTree)
        coords = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        gdf3d = GeoDataFrame3D.from_points(coords, crs="EPSG:4326")

        # Test auto method (should select cKDTree for small dataset)
        indices, distances = gdf3d.nearest3d([(0.5, 0.5, 0.5)], k=2, method="auto")
        assert len(indices) == 1
        assert len(indices[0]) == 2

        # Test explicit cKDTree method
        indices, distances = gdf3d.nearest3d([(0.5, 0.5, 0.5)], k=2, method="cKDTree")
        assert len(indices) == 1
        assert len(indices[0]) == 2

    def test_nearest3d_hnsw(self):
        """Test HNSW nearest neighbor search."""
        try:
            # Create test data
            coords = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
            gdf3d = GeoDataFrame3D.from_points(coords, crs="EPSG:4326")

            # Test HNSW method
            indices, distances = gdf3d.nearest3d([(0.5, 0.5, 0.5)], k=2, method="HNSW")
            assert len(indices) == 1
            assert len(indices[0]) == 2
            assert len(distances) == 1
            assert len(distances[0]) == 2

        except ImportError:
            pytest.skip("HNSW indexing not available")

    def test_sjoin_nearest3d(self):
        """Test optimized spatial join with k-NN."""
        # Create two datasets
        coords1 = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        coords2 = [(0.5, 0.5, 0.5), (1.5, 1.5, 1.5)]

        gdf1 = GeoDataFrame3D.from_points(coords1, crs="EPSG:4326")
        gdf1["id"] = range(len(coords1))

        gdf2 = GeoDataFrame3D.from_points(coords2, crs="EPSG:4326")
        gdf2["id"] = range(len(coords2))

        # Test spatial join
        joined = gdf1.sjoin_nearest3d(gdf2, k=2, method="cKDTree")
        assert len(joined) > 0
        assert "distance3d" in joined.columns
        assert "neighbor_rank" in joined.columns

    def test_large_dataset_auto_selection(self):
        """Test automatic method selection for large datasets."""
        # Create a dataset that should trigger HNSW selection
        np.random.seed(42)
        n_points = 150000  # Above the 100k threshold

        x_coords = np.random.uniform(-180, 180, n_points)
        y_coords = np.random.uniform(-90, 90, n_points)
        z_coords = np.random.uniform(0, 1000, n_points)

        coords = list(zip(x_coords, y_coords, z_coords))

        gdf3d = GeoDataFrame3D.from_points(coords, crs="EPSG:4326")

        # Test auto method selection
        try:
            indices, distances = gdf3d.nearest3d([(0, 0, 500)], k=5, method="auto")
            assert len(indices) == 1
            assert len(indices[0]) == 5

            # Should have used HNSW for large dataset
            assert hasattr(gdf3d, "_hnsw_index") or hasattr(gdf3d, "_sindex")

        except ImportError:
            # If HNSW not available, should fall back to cKDTree
            indices, distances = gdf3d.nearest3d([(0, 0, 500)], k=5, method="auto")
            assert len(indices) == 1
            assert len(indices[0]) == 5

    def test_invalid_methods(self):
        """Test error handling for invalid methods."""
        coords = [(0, 0, 0), (1, 1, 1)]
        gdf3d = GeoDataFrame3D.from_points(coords, crs="EPSG:4326")

        # Test invalid indexing method
        with pytest.raises(ValueError, match="Unknown indexing method"):
            gdf3d.build_sindex(method="invalid")

        # Test invalid search method
        with pytest.raises(ValueError, match="Unknown search method"):
            gdf3d.nearest3d([(0.5, 0.5, 0.5)], method="invalid")


if __name__ == "__main__":
    pytest.main([__file__])
