"""
Basic tests for GeoDataFrame3D.
"""

import numpy as np
import pandas as pd
import pytest

# Test if geopandas3d is available
try:
    from geopandas3d import GeoDataFrame3D

    GEOPANDAS3D_AVAILABLE = True
except ImportError:
    GEOPANDAS3D_AVAILABLE = False

# Test if GeoPandas is available
try:
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


@pytest.mark.skipif(not GEOPANDAS3D_AVAILABLE, reason="geopandas3d not available")
class TestGeoDataFrame3D:
    """Test GeoDataFrame3D functionality."""

    def test_creation_from_xyz(self):
        """Test creating GeoDataFrame3D from x, y, z columns."""
        df = pd.DataFrame(
            {"x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2], "value": [10, 20, 30]}
        )

        gdf3d = GeoDataFrame3D.from_xyz(
            df, "x", "y", "z", crs="EPSG:4979", height_col="altitude"
        )

        assert len(gdf3d) == 3
        assert gdf3d.height_col == "altitude"
        assert gdf3d.geometry_type == "point"
        assert gdf3d.crs == "EPSG:4979"
        assert "altitude" in gdf3d.columns
        assert "geometry" in gdf3d.columns

    def test_creation_from_points(self):
        """Test creating GeoDataFrame3D from list of 3D points."""
        points = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        data = pd.DataFrame({"value": [10, 20, 30]})

        gdf3d = GeoDataFrame3D.from_points(
            points, data, crs="EPSG:4979", height_col="altitude"
        )

        assert len(gdf3d) == 3
        assert gdf3d.height_col == "altitude"
        assert gdf3d.geometry_type == "point"
        assert "altitude" in gdf3d.columns
        assert "geometry" in gdf3d.columns

    def test_creation_from_polygons(self):
        """Test creating GeoDataFrame3D from list of 3D polygon vertices."""
        polygons = [
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            [(2, 2, 1), (3, 2, 1), (2.5, 3, 1)],
        ]

        gdf3d = GeoDataFrame3D.from_polygons(
            polygons, crs="EPSG:4979", height_col="altitude"
        )

        assert len(gdf3d) == 2
        assert gdf3d.height_col == "altitude"
        assert gdf3d.geometry_type == "polygon"
        assert "altitude" in gdf3d.columns
        assert "geometry" in gdf3d.columns

    def test_3d_coordinates(self):
        """Test getting 3D coordinates."""
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2]})

        gdf3d = GeoDataFrame3D.from_xyz(
            df, "x", "y", "z", crs="EPSG:4979", height_col="altitude"
        )
        coords = gdf3d.get_3d_coordinates()

        assert coords.shape == (3, 3)
        assert np.array_equal(coords, np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))

    def test_3d_bounds(self):
        """Test 3D bounds calculation."""
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2]})

        gdf3d = GeoDataFrame3D.from_xyz(
            df, "x", "y", "z", crs="EPSG:4979", height_col="altitude"
        )
        bounds = gdf3d.bounds3d()

        assert bounds == (0.0, 0.0, 0.0, 2.0, 2.0, 2.0)

    def test_spatial_indexing(self):
        """Test spatial indexing functionality."""
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2]})

        gdf3d = GeoDataFrame3D.from_xyz(
            df, "x", "y", "z", crs="EPSG:4979", height_col="altitude"
        )
        gdf3d.build_sindex()

        assert gdf3d._sindex is not None
        assert gdf3d._sindex.ready

    def test_nearest_neighbor(self):
        """Test nearest neighbor queries."""
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2]})

        gdf3d = GeoDataFrame3D.from_xyz(
            df, "x", "y", "z", crs="EPSG:4979", height_col="altitude"
        )
        gdf3d.build_sindex()

        query_point = (0.5, 0.5, 0.5)
        idx, dist = gdf3d.nearest3d([query_point], k=1)

        # For k=1, idx and dist are 1D numpy arrays
        assert idx.size == 1
        assert dist.size == 1
        assert idx[0] == 0  # Should find the first point (0, 0, 0)
        assert dist[0] > 0  # Distance should be positive

    def test_radius_query(self):
        """Test radius queries."""
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "z": [0, 1, 2]})

        gdf3d = GeoDataFrame3D.from_xyz(
            df, "x", "y", "z", crs="EPSG:4979", height_col="altitude"
        )
        gdf3d.build_sindex()

        query_point = (0.5, 0.5, 0.5)
        neighbors = gdf3d.query_ball3d([query_point], r=1.0)

        assert len(neighbors) == 1
        assert len(neighbors[0]) > 0  # Should find at least one neighbor

    def test_inheritance_from_geopandas(self):
        """Test that GeoDataFrame3D properly inherits from GeoPandas GeoDataFrame."""
        df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "z": [0, 1]})

        gdf3d = GeoDataFrame3D.from_xyz(
            df, "x", "y", "z", crs="EPSG:4979", height_col="altitude"
        )

        # Test that standard GeoPandas methods work
        assert hasattr(gdf3d, "geometry")
        assert hasattr(gdf3d, "crs")
        assert hasattr(gdf3d, "bounds")

        # Test that we can access columns like a regular DataFrame
        assert "altitude" in gdf3d.columns
        assert "geometry" in gdf3d.columns

        # Test that we can iterate over rows
        for _idx, row in gdf3d.iterrows():
            assert "altitude" in row
            assert "geometry" in row


@pytest.mark.skipif(not GEOPANDAS3D_AVAILABLE, reason="geopandas3d not available")
class TestUtilityFunctions:
    """Test utility functions."""

    def test_distance3d(self):
        """Test 3D distance calculation."""
        from geopandas3d import distance3d

        p1 = (0, 0, 0)
        p2 = (1, 1, 1)
        dist = distance3d(p1, p2)

        expected = np.sqrt(3)
        assert abs(dist - expected) < 1e-10

    def test_centroid3d(self):
        """Test 3D centroid calculation."""
        from geopandas3d import centroid3d

        points = [(0, 0, 0), (2, 0, 0), (1, 2, 0)]
        centroid = centroid3d(points)

        expected = (1.0, 2 / 3, 0.0)
        assert all(abs(c - e) < 1e-10 for c, e in zip(centroid, expected))

    def test_polygon_area3d(self):
        """Test 3D polygon area calculation."""
        from geopandas3d import polygon_area3d

        vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        area = polygon_area3d(vertices)

        assert abs(area - 1.0) < 1e-10

    def test_point_in_polygon3d(self):
        """Test point-in-polygon test."""
        from geopandas3d import is_point_in_polygon3d

        vertices = [(0, 0, 0), (2, 0, 0), (1, 2, 0)]

        # Point inside
        inside_point = (1, 1, 0)
        assert is_point_in_polygon3d(inside_point, vertices)

        # Point outside
        outside_point = (3, 3, 0)
        assert not is_point_in_polygon3d(outside_point, vertices)


if __name__ == "__main__":
    pytest.main([__file__])
