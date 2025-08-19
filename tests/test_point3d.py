"""
Tests for the Point3D class.
"""

import numpy as np
import pytest
from shapely.geometry import Point

# Skip if geopandas3d is not available
try:
    from geopandas3d import Point3D
except ImportError:
    Point3D = None


@pytest.mark.skipif(Point3D is None, reason="geopandas3d not available")
class TestPoint3D:
    """Test the Point3D class."""

    def test_basic_creation(self):
        """Test basic Point3D creation."""
        point = Point3D(1.0, 2.0, 3.0)
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.z == 3.0
        assert point.crs is None

    def test_creation_with_crs(self):
        """Test Point3D creation with CRS."""
        point = Point3D(1.0, 2.0, 3.0, crs="EPSG:4326")
        assert point.crs == "EPSG:4326"

    def test_immutable(self):
        """Test that Point3D is immutable."""
        point = Point3D(1.0, 2.0, 3.0)
        with pytest.raises(Exception):  # dataclass frozen=True raises FrozenInstanceError
            point.x = 5.0

    def test_validation_numeric(self):
        """Test coordinate validation for numeric types."""
        # Valid
        Point3D(1, 2, 3)  # integers
        Point3D(1.0, 2.0, 3.0)  # floats

        # Invalid
        with pytest.raises(ValueError):
            Point3D("1", 2, 3)
        with pytest.raises(ValueError):
            Point3D(1, "2", 3)
        with pytest.raises(ValueError):
            Point3D(1, 2, "3")

    def test_validation_finite(self):
        """Test coordinate validation for finite values."""
        # Valid
        Point3D(1.0, 2.0, 3.0)

        # Invalid
        with pytest.raises(ValueError):
            Point3D(np.inf, 2.0, 3.0)
        with pytest.raises(ValueError):
            Point3D(1.0, np.nan, 3.0)
        with pytest.raises(ValueError):
            Point3D(1.0, 2.0, -np.inf)

    def test_iteration(self):
        """Test iteration and unpacking."""
        point = Point3D(1.0, 2.0, 3.0)
        x, y, z = point
        assert x == 1.0
        assert y == 2.0
        assert z == 3.0

    def test_indexing(self):
        """Test indexing access."""
        point = Point3D(1.0, 2.0, 3.0)
        assert point[0] == 1.0
        assert point[1] == 2.0
        assert point[2] == 3.0

        with pytest.raises(IndexError):
            _ = point[3]
        with pytest.raises(IndexError):
            _ = point[-1]

    def test_length(self):
        """Test length property."""
        point = Point3D(1.0, 2.0, 3.0)
        assert len(point) == 3

    def test_string_representations(self):
        """Test string representations."""
        point = Point3D(1.0, 2.0, 3.0)
        assert str(point) == "(1.0, 2.0, 3.0)"
        assert repr(point) == "Point3D(1.0, 2.0, 3.0)"

        point_with_crs = Point3D(1.0, 2.0, 3.0, crs="EPSG:4326")
        assert "crs=EPSG:4326" in repr(point_with_crs)

    def test_conversion_methods(self):
        """Test conversion to other formats."""
        point = Point3D(1.0, 2.0, 3.0)

        # To tuple
        assert point.to_tuple() == (1.0, 2.0, 3.0)

        # To list
        assert point.to_list() == [1.0, 2.0, 3.0]

        # To numpy array
        arr = point.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

        # To Shapely Point (2D)
        shapely_point = point.to_shapely()
        assert isinstance(shapely_point, Point)
        assert shapely_point.x == 1.0
        assert shapely_point.y == 2.0
        # z coordinate is lost in Shapely Point

    def test_distance_calculations(self):
        """Test distance calculations."""
        point1 = Point3D(0.0, 0.0, 0.0)
        point2 = Point3D(3.0, 4.0, 0.0)
        point3 = Point3D(3.0, 4.0, 12.0)

        # 2D distance
        assert point1.distance_to_2d(point2) == 5.0

        # 3D distance
        assert point1.distance_to(point3) == 13.0  # 5-12-13 triangle

        # Distance to self
        assert point1.distance_to(point1) == 0.0

    def test_distance_errors(self):
        """Test distance calculation errors."""
        point = Point3D(1.0, 2.0, 3.0)

        with pytest.raises(TypeError):
            point.distance_to("not a point")
        with pytest.raises(TypeError):
            point.distance_to_2d("not a point")

    def test_midpoint(self):
        """Test midpoint calculation."""
        point1 = Point3D(0.0, 0.0, 0.0)
        point2 = Point3D(2.0, 4.0, 6.0)

        mid = point1.midpoint(point2)
        assert mid.x == 1.0
        assert mid.y == 2.0
        assert mid.z == 3.0

        # Midpoint with CRS
        point1_crs = Point3D(0.0, 0.0, 0.0, crs="EPSG:4326")
        point2_crs = Point3D(2.0, 4.0, 6.0, crs="EPSG:4326")
        mid_crs = point1_crs.midpoint(point2_crs)
        assert mid_crs.crs == "EPSG:4326"

        # Midpoint with different CRS
        point1_diff_crs = Point3D(0.0, 0.0, 0.0, crs="EPSG:4326")
        point2_diff_crs = Point3D(2.0, 4.0, 6.0, crs="EPSG:3857")
        mid_diff_crs = point1_diff_crs.midpoint(point2_diff_crs)
        assert mid_diff_crs.crs is None

    def test_midpoint_errors(self):
        """Test midpoint calculation errors."""
        point = Point3D(1.0, 2.0, 3.0)

        with pytest.raises(TypeError):
            point.midpoint("not a point")

    def test_class_methods(self):
        """Test class methods for creating Point3D instances."""
        # From tuple
        point_tuple = Point3D.from_tuple((1.0, 2.0, 3.0))
        assert point_tuple.x == 1.0
        assert point_tuple.y == 2.0
        assert point_tuple.z == 3.0

        # From list
        point_list = Point3D.from_list([1.0, 2.0, 3.0])
        assert point_list.x == 1.0
        assert point_list.y == 2.0
        assert point_list.z == 3.0

        # From numpy array
        point_array = Point3D.from_array(np.array([1.0, 2.0, 3.0]))
        assert point_array.x == 1.0
        assert point_array.y == 2.0
        assert point_array.z == 3.0

        # From Shapely Point
        shapely_point = Point(1.0, 2.0)
        point_shapely = Point3D.from_shapely(shapely_point, z=3.0)
        assert point_shapely.x == 1.0
        assert point_shapely.y == 2.0
        assert point_shapely.z == 3.0

        # Origin
        origin = Point3D.origin()
        assert origin.x == 0.0
        assert origin.y == 0.0
        assert origin.z == 0.0

        # Origin with CRS
        origin_crs = Point3D.origin(crs="EPSG:4326")
        assert origin_crs.crs == "EPSG:4326"

    def test_class_method_errors(self):
        """Test class method error handling."""
        # Wrong length
        with pytest.raises(ValueError):
            Point3D.from_tuple((1.0, 2.0))
        with pytest.raises(ValueError):
            Point3D.from_list([1.0, 2.0])
        with pytest.raises(ValueError):
            Point3D.from_array(np.array([1.0, 2.0]))

        # Wrong type for Shapely
        with pytest.raises(TypeError):
            Point3D.from_shapely("not a point", z=3.0)

    def test_utility_methods(self):
        """Test utility methods."""
        # Valid point
        point = Point3D(1.0, 2.0, 3.0)
        assert point.is_valid()
        assert point.is_finite()
        assert not point.is_origin()

        # Origin point
        origin = Point3D.origin()
        assert origin.is_origin()
        assert origin.is_valid()
        assert origin.is_finite()

    def test_transform_method(self):
        """Test transform method (basic functionality)."""
        point = Point3D(1.0, 2.0, 3.0, crs="EPSG:4326")

        # Mock transformer (in practice, this would be a pyproj.Transformer)
        class MockTransformer:
            def transform(self, x, y):
                return x + 1, y + 1
            @property
            def target_crs(self):
                return "EPSG:3857"

        transformer = MockTransformer()
        transformed = point.transform(transformer)

        assert transformed.x == 2.0  # 1 + 1
        assert transformed.y == 3.0  # 2 + 1
        assert transformed.z == 3.0  # z unchanged
        assert transformed.crs == "EPSG:3857"

    def test_equality_and_hash(self):
        """Test equality and hash behavior."""
        point1 = Point3D(1.0, 2.0, 3.0)
        point2 = Point3D(1.0, 2.0, 3.0)
        point3 = Point3D(1.0, 2.0, 4.0)

        # Equality
        assert point1 == point2
        assert point1 != point3

        # Hash (frozen dataclass should be hashable)
        assert hash(point1) == hash(point2)
        assert hash(point1) != hash(point3)
