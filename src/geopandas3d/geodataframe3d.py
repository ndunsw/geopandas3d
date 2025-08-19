"""
3D GeoDataFrame that extends GeoPandas with altitude-aware spatial operations.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Literal, Optional

import numpy as np
import pandas as pd

# Import GeoPandas classes to extend
try:
    import geopandas as gpd
    from geopandas import GeoDataFrame

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: GeoPandas not available. Install with: pip install geopandas")

from .sindex import SpatialIndex3D

CRSType = Optional[str]
GeometryType = Literal["point", "polygon", "mixed"]


class GeoDataFrame3D(GeoDataFrame):
    """A 3D extension to GeoPandas GeoDataFrame with altitude-aware spatial operations.

    This class inherits from GeoPandas GeoDataFrame and adds:
    - Required altitude/height column for 3D operations
    - 3D spatial indexing using scipy.spatial.cKDTree
    - 3D nearest-neighbor queries and within-distance spatial joins
    - 3D bounds calculation and geometry validation
    - 3D plotting capabilities

    All existing GeoPandas methods work unchanged.
    """

    def __init__(self, data=None, *args, height_col: str = "altitude", **kwargs):
        """Initialize GeoDataFrame3D.

        Args:
            data: DataFrame, GeoDataFrame, or other data source
            height_col: Name of the altitude/height column (required for 3D operations)
            *args, **kwargs: Passed to GeoDataFrame constructor
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError(
                "GeoPandas is required. Install with: pip install geopandas"
            )

        # Initialize the parent GeoDataFrame
        super().__init__(data, *args, **kwargs)

        # Set the height column
        self.height_col = height_col

        # Validate that height column exists
        if self.height_col not in self.columns:
            raise ValueError(
                f"Height column '{self.height_col}' not found in DataFrame. "
                f"Available columns: {list(self.columns)}"
            )

        # Validate height column contains numeric data
        if not pd.api.types.is_numeric_dtype(self[self.height_col]):
            raise ValueError(
                f"Height column '{self.height_col}' must contain numeric data"
            )

        # Initialize spatial index
        self._sindex = None

        # Detect geometry type
        self._detect_geometry_type()

    def _detect_geometry_type(self):
        """Detect and validate geometry types for 3D operations."""
        if len(self) == 0:
            self.geometry_type = "mixed"
            return

        # Check geometry types
        geom_types = self.geometry.geom_type.unique()

        if len(geom_types) == 1:
            if geom_types[0] == "Point":
                self.geometry_type = "point"
            elif geom_types[0] in ["Polygon", "MultiPolygon"]:
                self.geometry_type = "polygon"
            else:
                self.geometry_type = "mixed"
        else:
            self.geometry_type = "mixed"

    @classmethod
    def from_xyz(
        cls,
        df: pd.DataFrame,
        x: str,
        y: str,
        z: str,
        crs: CRSType = None,
        height_col: str = "altitude",
    ) -> GeoDataFrame3D:
        """Create GeoDataFrame3D from DataFrame with x, y, z columns.

        Args:
            df: DataFrame with x, y, z coordinate columns
            x: Name of x-coordinate column
            y: Name of y-coordinate column
            z: Name of z-coordinate column
            crs: Coordinate reference system
            height_col: Name for the altitude/height column

        Returns:
            GeoDataFrame3D instance
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError(
                "GeoPandas is required. Install with: pip install geopandas"
            )

        if not all(col in df.columns for col in [x, y, z]):
            raise ValueError(f"DataFrame must contain columns: {x}, {y}, {z}")

        if df.empty:
            # Handle empty DataFrame
            empty_df = pd.DataFrame(columns=[height_col])
            empty_gdf = GeoDataFrame(empty_df, crs=crs)
            return cls(empty_gdf, height_col=height_col)

        # Create a copy
        gdf = df.copy()

        # Create 2D Point geometries for GeoPandas compatibility (using x, y coordinates)
        try:
            from shapely.geometry import Point

            gdf["geometry"] = gdf.apply(lambda row: Point(row[x], row[y]), axis=1)
        except ImportError as err:
            raise ImportError(
                "Shapely is required. Install with: pip install shapely"
            ) from err

        # Store altitude in height column
        gdf[height_col] = gdf[z]

        # Create GeoDataFrame with 2D geometry and height column
        geo_df = GeoDataFrame(gdf, geometry="geometry", crs=crs)

        # Create the GeoDataFrame3D
        return cls(geo_df, height_col=height_col)

    @classmethod
    def from_points(
        cls,
        points: list[tuple[float, float, float]],
        data: pd.DataFrame | None = None,
        crs: CRSType = None,
        height_col: str = "altitude",
    ) -> GeoDataFrame3D:
        """Create GeoDataFrame3D from list of 3D points.

        Args:
            points: List of (x, y, z) tuples
            data: Optional DataFrame with additional columns
            crs: Coordinate reference system
            height_col: Name for the altitude/height column

        Returns:
            GeoDataFrame3D instance
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError(
                "GeoPandas is required. Install with: pip install geopandas"
            )

        try:
            from shapely.geometry import Point
        except ImportError as err:
            raise ImportError(
                "Shapely is required. Install with: pip install shapely"
            ) from err

        # Create DataFrame
        if data is None:
            data = pd.DataFrame()

        # Create 2D geometries and extract heights
        geometries = [Point(x, y) for x, y, _ in points]
        heights = [z for _, _, z in points]

        # Add to DataFrame
        data = data.copy()
        data["geometry"] = geometries
        data[height_col] = heights

        # Create GeoDataFrame
        geo_df = GeoDataFrame(data, geometry="geometry", crs=crs)

        # Create GeoDataFrame3D
        return cls(geo_df, height_col=height_col)

    @classmethod
    def from_polygons(
        cls,
        polygons: list[list[tuple[float, float, float]]],
        data: pd.DataFrame | None = None,
        crs: CRSType = None,
        height_col: str = "altitude",
    ) -> GeoDataFrame3D:
        """Create GeoDataFrame3D from list of 3D polygon vertices.

        Args:
            polygons: List of polygon vertex lists, each vertex as (x, y, z) tuple
            data: Optional DataFrame with additional columns
            crs: Coordinate reference system
            height_col: Name for the altitude/height column

        Returns:
            GeoDataFrame3D instance
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError(
                "GeoPandas is required. Install with: pip install geopandas"
            )

        try:
            from shapely.geometry import Polygon
        except ImportError as err:
            raise ImportError(
                "Shapely is required. Install with: pip install shapely"
            ) from err

        # Create DataFrame
        if data is None:
            data = pd.DataFrame()

        # Create 2D polygons and extract heights
        geometries = []
        heights = []

        for polygon_vertices in polygons:
            if len(polygon_vertices) < 3:
                warnings.warn(
                    f"Polygon with {len(polygon_vertices)} vertices skipped (need at least 3)",
                    stacklevel=2,
                )
                continue

            # Extract 2D coordinates for geometry
            coords_2d = [(x, y) for x, y, _ in polygon_vertices]

            # Extract height (use mean of all vertices)
            z_coords = [z for _, _, z in polygon_vertices]
            height = np.mean(z_coords)

            try:
                poly = Polygon(coords_2d)
                if poly.is_valid:
                    geometries.append(poly)
                    heights.append(height)
                else:
                    warnings.warn(f"Invalid polygon skipped: {poly}", stacklevel=2)
            except Exception as e:
                warnings.warn(f"Error creating polygon: {e}", stacklevel=2)
                continue

        if not geometries:
            raise ValueError("No valid polygons could be created")

        # Add to DataFrame
        data = data.copy()
        data["geometry"] = geometries
        data[height_col] = heights

        # Create GeoDataFrame
        geo_df = GeoDataFrame(data, geometry="geometry", crs=crs)

        # Create GeoDataFrame3D
        return cls(geo_df, height_col=height_col)

    @classmethod
    def from_gdf(
        cls, gdf: gpd.GeoDataFrame, z_column: str, geometry_column: str = None, **kwargs
    ):
        """
        Construct a GeoDataFrame3D from a 2D GeoDataFrame and a column
        representing altitude/height.

        Args:
            gdf (GeoDataFrame): The source 2D GeoDataFrame.
            z_column (str): Column name in gdf containing altitude values.
            geometry_column (str, optional): Name of the geometry column.
                                             Defaults to gdf.geometry.name.
            **kwargs: Additional arguments passed to GeoDataFrame3D constructor.

        Returns:
            GeoDataFrame3D: A new 3D-enabled GeoDataFrame.
        """
        if geometry_column is None:
            geometry_column = gdf.geometry.name

        if z_column not in gdf.columns:
            raise ValueError(f"Column '{z_column}' not found in GeoDataFrame.")

        try:
            from shapely.geometry import Point
        except ImportError as err:
            raise ImportError(
                "Shapely is required. Install with: pip install shapely"
            ) from err

        # Build 3D Points by combining existing 2D geometry with z values
        new_geom = [
            Point(geom.x, geom.y, z) if geom is not None else None
            for geom, z in zip(gdf[geometry_column], gdf[z_column])
        ]

        new_gdf = gdf.copy()
        new_gdf[geometry_column] = new_geom

        # Construct as GeoDataFrame3D
        return cls(new_gdf, geometry=geometry_column, crs=gdf.crs, **kwargs)

    # ----- 3D geometry operations -----
    def get_3d_coordinates(self) -> np.ndarray:
        """Get 3D coordinates as numpy array for spatial operations.

        Returns:
            (n, 3) array of (x, y, z) coordinates
        """
        if len(self) == 0:
            return np.array([]).reshape(0, 3)

        coords = []
        for geom, height in zip(self.geometry, self[self.height_col]):
            if geom is not None and not pd.isna(height):
                if self.geometry_type == "point":
                    # For points, extract x, y coordinates
                    coords.append([geom.x, geom.y, height])
                elif self.geometry_type == "polygon":
                    # For polygons, use centroid
                    centroid = geom.centroid
                    coords.append([centroid.x, centroid.y, height])
                else:
                    # Mixed types - use centroid for all
                    centroid = geom.centroid
                    coords.append([centroid.x, centroid.y, height])
            else:
                coords.append([np.nan, np.nan, np.nan])

        return np.array(coords)

    def bounds3d(self) -> tuple[float, float, float, float, float, float]:
        """Get 3D bounding box (xmin, ymin, zmin, xmax, ymax, zmax)."""
        if len(self) == 0:
            return (np.nan,) * 6

        coords = self.get_3d_coordinates()
        valid_coords = coords[~np.isnan(coords).any(axis=1)]

        if len(valid_coords) == 0:
            return (np.nan,) * 6

        mins = valid_coords.min(axis=0)
        maxs = valid_coords.max(axis=0)
        return (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])

    def get_geometry_bounds3d(
        self,
    ) -> list[tuple[float, float, float, float, float, float]]:
        """Get individual 3D bounds for each geometry."""
        if len(self) == 0:
            return []

        bounds_list = []
        for geom, height in zip(self.geometry, self[self.height_col]):
            if geom is not None and not pd.isna(height):
                if self.geometry_type == "point":
                    bounds_list.append((geom.x, geom.y, height, geom.x, geom.y, height))
                elif self.geometry_type == "polygon":
                    # Get 2D bounds and add height
                    bounds_2d = geom.bounds  # (xmin, ymin, xmax, ymax)
                    bounds_list.append(
                        (
                            bounds_2d[0],
                            bounds_2d[1],
                            height,
                            bounds_2d[2],
                            bounds_2d[3],
                            height,
                        )
                    )
                else:
                    # Mixed types - use centroid
                    centroid = geom.centroid
                    bounds_list.append(
                        (centroid.x, centroid.y, height, centroid.x, centroid.y, height)
                    )
            else:
                bounds_list.append((np.nan,) * 6)

        return bounds_list

    # ----- 3D spatial indexing -----
    def build_sindex(self, method: str = "cKDTree", **kwargs):
        """Build 3D spatial index for efficient spatial queries.

        Args:
            method: Indexing method ("cKDTree" or "HNSW").
            **kwargs: Additional arguments for the indexing method.

        Returns:
            The built spatial index object.
        """
        if method == "cKDTree":
            self._sindex = self._build_ckdtree_index(**kwargs)
        elif method == "HNSW":
            self._sindex = self._build_hnsw_index(**kwargs)
        else:
            raise ValueError(
                f"Unknown indexing method: {method}. Use 'cKDTree' or 'HNSW'."
            )

        return self._sindex


    def _build_ckdtree_index(self, **kwargs):
        """Build cKDTree spatial index (default method)."""
        coords_3d = self.get_3d_coordinates()
        if len(coords_3d) == 0:
            return None

        # Filter out invalid coordinates
        valid_mask = ~np.isnan(coords_3d).any(axis=1)
        if not valid_mask.any():
            return None

        valid_coords = coords_3d[valid_mask]

        try:
            from scipy.spatial import cKDTree

            return cKDTree(valid_coords, **kwargs)
        except ImportError as err:
            raise ImportError("scipy is required for cKDTree indexing") from err

    def _build_hnsw_index(self, **kwargs):
        """Build HNSW spatial index for huge datasets."""
        try:
            import hnswlib
        except ImportError as err:
            raise ImportError(
                "hnswlib is required for HNSW indexing. Install with: pip install hnswlib"
            ) from err

        coords_3d = self.get_3d_coordinates()
        if len(coords_3d) == 0:
            return None

        # Filter out invalid coordinates
        valid_mask = ~np.isnan(coords_3d).any(axis=1)
        if not valid_mask.any():
            return None

        valid_coords = coords_3d[valid_mask]

        # Default HNSW parameters optimized for 3D spatial data
        default_params = {
            "dim": 3,  # 3D coordinates
            "max_elements": len(valid_coords),
            "ef_construction": 200,
            "M": 16,
            "random_seed": 100,
            "allow_replace_deleted": False,
        }

        # Update with user parameters
        default_params.update(kwargs)

        # Create and configure HNSW index
        index = hnswlib.Index(space="l2", dim=default_params["dim"])
        index.init_index(
            max_elements=default_params["max_elements"],
            ef_construction=default_params["ef_construction"],
            M=default_params["M"],
            random_seed=default_params["random_seed"],
            allow_replace_deleted=default_params["allow_replace_deleted"],
        )

        # Add data to index
        index.add_items(valid_coords.astype(np.float32))

        # Set search parameters
        index.set_ef(50)  # Default search parameter

        return index

    @property
    def sindex(self) -> SpatialIndex3D:
        """Access the 3D spatial index."""
        if self._sindex is None:
            self.build_sindex()
        return self._sindex

    # ----- 3D spatial queries -----
    def nearest3d(self, query_points, k=1, method="auto", **kwargs):
        """Find k nearest neighbors in 3D space.

        Args:
            query_points: Query points as (n, 3) array or list of Point3D
            k: Number of nearest neighbors to find
            method: Search method ("auto", "cKDTree", or "HNSW")
            **kwargs: Additional arguments for the search method

        Returns:
            tuple: (indices, distances) arrays
        """
        if method == "auto" and len(self) > 100000:  # Prefer HNSW for huge datasets
            method = "HNSW"
        else:
            method = "cKDTree"

        if method == "cKDTree":
            return self._nearest3d_ckdtree(query_points, k, **kwargs)
        elif method == "HNSW":
            return self._nearest3d_hnsw(query_points, k, **kwargs)
        else:
            raise ValueError(f"Unknown search method: {method}")

    def _nearest3d_ckdtree(self, query_points, k=1, **kwargs):
        """Find nearest neighbors using cKDTree."""
        # Convert query points to numpy array
        if isinstance(query_points, list) and len(query_points) > 0:
            if isinstance(query_points[0], (list, tuple)):
                query_coords = np.array(query_points)
            elif hasattr(query_points[0], "to_array"):
                query_coords = np.array([p.to_array() for p in query_points])
            else:
                query_coords = np.array(query_points)
        else:
            query_coords = np.array(query_points)

        # Ensure 2D array
        if query_coords.ndim == 1:
            query_coords = query_coords.reshape(1, -1)

        # Build index if not exists
        if not hasattr(self, "_sindex") or self._sindex is None:
            self._sindex = self._build_ckdtree_index()

        if self._sindex is None:
            return np.array([]), np.array([])

        # Query the index
        kwargs.pop("how", None)
        distances, indices = self._sindex.query(query_coords, k=k, **kwargs)

        # normalize shapes: k=1 → 1D arrays; k>1 → 2D (n_queries, k)
        if k == 1:
            # cKDTree.query gives shape (n_queries,) or scalar if n_queries==1
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)
        else:
            # cKDTree.query gives shape (n_queries,k) or (k,) if n_queries==1
            if distances.ndim == 1:
                distances = distances.reshape(1, -1)
                indices = indices.reshape(1, -1)

        return indices, distances

    def _nearest3d_hnsw(self, query_points, k=1, **kwargs):
        """Find nearest neighbors using HNSW index."""
        # Convert query points to numpy array
        if isinstance(query_points, list) and len(query_points) > 0:
            if isinstance(query_points[0], (list, tuple)):
                query_coords = np.array(query_points)
            elif hasattr(query_points[0], "to_array"):
                query_coords = np.array([p.to_array() for p in query_points])
            else:
                query_coords = np.array(query_points)
        else:
            query_coords = np.array(query_points)

        # Ensure 2D array
        if query_coords.ndim == 1:
            query_coords = query_coords.reshape(1, -1)

        # Build index if not exists
        if not hasattr(self, "_hnsw_index") or self._hnsw_index is None:
            self._hnsw_index = self._build_hnsw_index(**kwargs)

        if self._hnsw_index is None:
            return np.array([]), np.array([])

        # Set search parameters
        ef = kwargs.get("ef", 50)
        self._hnsw_index.set_ef(ef)

        # Query the index
        indices, distances = self._hnsw_index.knn_query(query_coords, k=k)

        # HNSW returns distances as squared distances, convert to actual distances
        distances = np.sqrt(distances)

        return indices, distances

    def query_ball3d(self, points: Iterable[tuple[float, float, float]], r: float, method: str = "cKDTree"):
        """Find all neighbors within radius r in 3D space.

        Args:
            points: Iterable of (x, y, z) tuples
            r: Search radius
            method: Indexing method to use ("cKDTree" or "HNSW")

        Returns:
            List of lists of indices for each query point
        """
        if len(self) == 0:
            return [[] for _ in points]

        # Filter out invalid coordinates
        coords = self.get_3d_coordinates()
        valid_mask = ~np.isnan(coords).any(axis=1)

        if not valid_mask.any():
            return [[] for _ in points]

        # Build default index if none exists
        if self._sindex is None:
            self.build_sindex(method=method)

        # Query spatial index
        if method == "cKDTree":
            neighbors = self._sindex.query_ball_point(points, r=r)
        elif method == "HNSW":
            neighbors = []
            for pt in points:
                # Use a large k and filter by distance r
                k = len(self)
                labels, distances = self._sindex.knn_query(np.array([pt]), k=k)
                neighbors.append([int(lbl) for lbl, dist in zip(labels[0], distances[0]) if dist <= r])
        else:
            raise ValueError(f"Unknown search method: {method}")

        # Map back to original indices
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            neighbors = [[valid_indices[i] for i in nbrs] for nbrs in neighbors]

        return neighbors

    # ----- 3D spatial joins -----
    def sjoin_within_distance3d(
        self,
        other: GeoDataFrame3D,
        max_distance: float,
        how: Literal["inner", "left"] = "inner",
        suffixes: tuple[str, str] = ("_l", "_r"),
    ) -> pd.DataFrame:
        """Join pairs within max_distance using 3D distances.

        Args:
            other: Another GeoDataFrame3D to join with
            max_distance: Maximum 3D distance for joining
            how: Join type ('inner', 'left')
            suffixes: Suffixes for overlapping column names

        Returns:
            DataFrame with joined data
        """
        if len(self) == 0 or len(other) == 0:
            return pd.DataFrame()

        if not hasattr(self, "_sindex") or self._sindex is None:
            self._sindex = self._build_ckdtree_index()
        if not hasattr(other, "_sindex") or other._sindex is None:
            other._sindex = other._build_ckdtree_index()

        # Get 3D coordinates
        left_coords = self.get_3d_coordinates()
        right_coords = other.get_3d_coordinates()

        # Filter valid coordinates
        left_valid = ~np.isnan(left_coords).any(axis=1)
        right_valid = ~np.isnan(right_coords).any(axis=1)

        if not left_valid.any() or not right_valid.any():
            return pd.DataFrame()

        # Query within distance
        neighbors = other.query_ball3d(left_coords[left_valid], r=max_distance)

        # Build result DataFrame
        rows = []
        left_valid_indices = np.where(left_valid)[0]

        for i, nbrs in enumerate(neighbors):
            left_idx = left_valid_indices[i]

            if not nbrs and how == "inner":
                continue

            for j in nbrs:
                lrow = self.iloc[left_idx]
                rrow = other.iloc[j]

                # Calculate 3D distance
                d = np.linalg.norm(left_coords[left_idx] - right_coords[j])

                # Combine rows with suffixes
                combined = pd.concat(
                    [lrow.add_suffix(suffixes[0]), rrow.add_suffix(suffixes[1])]
                )
                combined["distance3d"] = float(d)
                rows.append(combined)

            if how == "left" and not nbrs:
                lrow = self.iloc[left_idx].add_suffix(suffixes[0])
                rrow = pd.Series({c + suffixes[1]: np.nan for c in other.columns})
                combined = pd.concat([lrow, rrow])
                combined["distance3d"] = np.nan
                rows.append(combined)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ----- 3D plotting methods -----
    def plot3d(self, column=None, ax=None, **kwargs):
        """Plot the 3D data using matplotlib."""
        from .plotting import plot3d

        return plot3d(self, column=column, ax=ax, **kwargs)

    def to_crs3d(self, crs, inplace=False, transformer=None):
        """
        Transform coordinates to a new CRS with 3D awareness.

        This method extends GeoPandas' to_crs functionality to properly handle
        3D coordinates during transformation. It uses pyproj.Transformer for
        efficient batch transformations.

        Args:
            crs: Target CRS (string, CRS object, or pyproj.CRS)
            inplace: If True, modify the current object. If False, return a new object.
            transformer: Optional pre-configured pyproj.Transformer for custom transformations

        Returns:
            GeoDataFrame3D with transformed coordinates (if inplace=False)

        Note:
            - Z coordinates (altitude/height) are preserved during transformation
            - For geographic to projected transformations, units are automatically handled
            - For projected to projected transformations, units are preserved
        """
        from pyproj import CRS, Transformer

        # Convert CRS to pyproj.CRS if it's a string
        if isinstance(crs, str):
            target_crs = CRS.from_string(crs)
        else:
            target_crs = CRS(crs)

        # Get source CRS
        source_crs = self.crs
        if source_crs is None:
            raise ValueError("Source CRS is not set. Cannot transform coordinates.")

        source_crs = CRS(source_crs)

        # Create transformer if not provided
        if transformer is None:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        # Get 3D coordinates
        coords_3d = self.get_3d_coordinates()
        x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]

        # Transform x, y coordinates
        try:
            x_new, y_new = transformer.transform(x, y)
        except Exception as err:
            raise ValueError(f"Coordinate transformation failed: {err}") from err

        # Handle z coordinates based on CRS types
        z_new = z.copy()

        # If transforming between geographic CRS, z might need unit conversion
        if source_crs.is_geographic and target_crs.is_geographic:
            # Both are geographic - z units should be compatible
            pass
        elif source_crs.is_geographic and not target_crs.is_geographic:
            # Geographic to projected - z units might need conversion
            # This is a simplified approach - in practice you might want more sophisticated handling
            pass
        elif not source_crs.is_geographic and target_crs.is_geographic:
            # Projected to geographic - z units might need conversion
            pass
        else:
            # Both projected - z units should be compatible
            pass

        # Create new geometry column with transformed coordinates
        new_geometries = []
        for i, geom in enumerate(self.geometry):
            if geom.geom_type == "Point":
                # Create new Point with transformed coordinates
                from shapely.geometry import Point

                new_point = Point(x_new[i], y_new[i])
                new_geometries.append(new_point)
            elif geom.geom_type == "Polygon":
                # Transform polygon coordinates properly
                coords = list(geom.exterior.coords)
                new_coords = []

                # Transform each coordinate pair
                for old_x, old_y in coords:
                    try:
                        new_x, new_y = transformer.transform(old_x, old_y)
                        new_coords.append((new_x, new_y))
                    except Exception:
                        # Fallback to original coordinates if transformation fails
                        new_coords.append((old_x, old_y))

                from shapely.geometry import Polygon

                new_polygon = Polygon(new_coords)
                new_geometries.append(new_polygon)
            else:
                # For other geometry types, keep as is for now
                new_geometries.append(geom)

        # Create new height column with transformed z values
        new_height_col = self[self.height_col].copy()
        new_height_col.iloc[: len(z_new)] = z_new

        # Create new DataFrame
        new_data = self.drop(columns=[self.height_col, self.geometry.name])
        new_data[self.height_col] = new_height_col
        new_data[self.geometry.name] = new_geometries

        # Create new GeoDataFrame3D
        if inplace:
            # Update current object properly
            # Clear existing data and add new data
            self.drop(columns=self.columns, inplace=True)
            for col in new_data.columns:
                self[col] = new_data[col]
            self._crs = target_crs
            return self
        else:
            # Return new object
            return GeoDataFrame3D(
                new_data,
                geometry=self.geometry.name,
                height_col=self.height_col,
                crs=target_crs,
            )

    def transform3d(self, transformer, inplace=False):
        """
        Transform coordinates using a pre-configured pyproj.Transformer.

        This method provides more control over the transformation process
        by allowing you to use a custom transformer configuration.

        Args:
            transformer: pyproj.Transformer instance
            inplace: If True, modify the current object. If False, return a new object.

        Returns:
            GeoDataFrame3D with transformed coordinates (if inplace=False)
        """
        # Get target CRS from transformer
        target_crs = transformer.target_crs

        return self.to_crs3d(target_crs, inplace=inplace, transformer=transformer)

    def get_transformer(self, target_crs, **kwargs):
        """
        Get a pyproj.Transformer for transforming to a target CRS.

        This method creates a transformer that can be reused for multiple
        transformations, which is more efficient than creating a new one each time.

        Args:
            target_crs: Target CRS (string, CRS object, or pyproj.CRS)
            **kwargs: Additional arguments to pass to pyproj.Transformer.from_crs

        Returns:
            pyproj.Transformer instance
        """
        from pyproj import CRS, Transformer

        # Convert CRS to pyproj.CRS if it's a string
        if isinstance(target_crs, str):
            target_crs = CRS.from_string(target_crs)
        else:
            target_crs = CRS(target_crs)

        # Get source CRS
        source_crs = self.crs
        if source_crs is None:
            raise ValueError("Source CRS is not set. Cannot create transformer.")

        source_crs = CRS(source_crs)

        # Create and return transformer
        return Transformer.from_crs(source_crs, target_crs, always_xy=True, **kwargs)

    def sjoin_nearest3d(self, other, k=1, method="auto", **kwargs):
        """Spatial join to find k nearest neighbors between two GeoDataFrame3D objects.

        Args:
            other: Another GeoDataFrame3D object
            k: Number of nearest neighbors to find
            method: Search method ("auto", "cKDTree", or "HNSW")
            **kwargs: Additional arguments for the search method

        Returns:
            GeoDataFrame3D with joined data
        """
        if method == "auto":
            # Auto-select best method based on dataset sizes
            total_size = len(self) + len(other)
            if total_size > 100000:  # Use HNSW for huge datasets
                method = "HNSW"
            else:
                method = "cKDTree"

        # Get coordinates from both datasets
        coords_self = self.get_3d_coordinates()
        coords_other = other.get_3d_coordinates()

        if len(coords_self) == 0 or len(coords_other) == 0:
            return GeoDataFrame3D([], crs=self.crs)

        # Find nearest neighbors
        indices, distances = self.nearest3d(coords_other, k=k, method=method, **kwargs)

        # PATCH: always make (n_queries, k) arrays
        # wrap scalars → arrays
        indices = np.atleast_1d(indices)
        distances = np.atleast_1d(distances)

        # if result is 1-D (shape (n_queries,)), turn it into (n_queries, 1)
        if indices.ndim == 1:
            indices = indices.reshape(-1, 1)
            distances = distances.reshape(-1, 1)

        # Create joined data
        joined_data = []
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            for j, (neighbor_idx, distance) in enumerate(zip(idx, dist)):
                if neighbor_idx < len(self):
                    # Get data from self
                    row_self = self.iloc[neighbor_idx].copy()
                    # Get data from other
                    row_other = other.iloc[i].copy()

                    # Combine data
                    combined_row = {}
                    for col in row_self.index:
                        if col != self.geometry.name and col != self.height_col:
                            combined_row[f"left_{col}"] = row_self[col]

                    for col in row_other.index:
                        if col != other.geometry.name and col != other.height_col:
                            combined_row[f"right_{col}"] = row_other[col]

                    # Add geometry and height from self (nearest neighbor)
                    combined_row[self.geometry.name] = row_self[self.geometry.name]
                    combined_row[self.height_col] = row_self[self.height_col]

                    # Add distance information
                    combined_row["distance"] = distance
                    combined_row["neighbor_rank"] = j

                    joined_data.append(combined_row)

        if not joined_data:
            return GeoDataFrame3D([], crs=self.crs)

        # Create DataFrame and GeoDataFrame3D
        df_joined = pd.DataFrame(joined_data)

        rename_map = {}
        for c in df_joined.columns:
            if c.startswith("left_"):
                rename_map[c] = f"{c[len('left_') :]}_l"
            elif c.startswith("right_"):
                rename_map[c] = f"{c[len('right_') :]}_r"
        if "distance" in df_joined.columns:
            rename_map["distance"] = "distance3d"

        df_joined = df_joined.rename(columns=rename_map)

        return GeoDataFrame3D(
            df_joined,
            geometry=self.geometry.name,
            height_col=self.height_col,
            crs=self.crs,
        )

    # ----- GeoPandas method delegation -----
    def __getattr__(self, name):
        """Delegate unknown attributes to the parent GeoDataFrame."""
        if hasattr(super(), name):
            return getattr(super(), name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __repr__(self):
        """String representation."""
        return f"GeoDataFrame3D({len(self)} {self.geometry_type}s, CRS: {self.crs}, height_col: {self.height_col})"
