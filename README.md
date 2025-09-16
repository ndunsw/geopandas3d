# GeoPandas3D

3D extensions to GeoPandas for altitude-aware spatial operations.

## Overview

GeoPandas3D extends GeoPandas with altitude awareness. It supports 3D distance calculations, centroids, point-in-polygon tests, spatial indexing and plotting without losing any built-in 2D features.

---

## Key Features

- Full 3D spatial operations: distance3d, centroid3d, point-in-polygon3d  
- Explicit height column requirement for all 3D workflows  
- cKDTree and HNSW spatial indexing for scalable performance  
- Nearest-neighbor and within-distance joins that include altitude  
- Matplotlib-based 3D plotting (plot3d and standalone plot3d function)  
- CRS transformations with pyproj that preserve 3D coordinates  
- Immutable Point3D class for lightweight 3D point representation  
- Automatic method selection (cKDTree vs HNSW) based on data size  
- HNSW indexing optimized for large datasets

---

## Installation

```bash
# Basic installation & Full feature set with HNSW support
# To be added to PyPi, for now refer to the development installation

# Development install
git clone https://github.com/ndunsw/geopandas3d.git
cd geopandas3d
pip install -e .
```

**Optional dependencies**

- hnswlib (>=0.7.0) for large-scale indexing  
- matplotlib (>=3.5) for 3D visualization  
- pyproj (>=3.0) for advanced CRS operations  

---

## Dependencies

- Required: geopandas, pandas, numpy, scipy, shapely  
- Optional: pyproj, matplotlib, hnsw

---

## Quick Start

### Creating a GeoDataFrame3D from x, y, z

```python
import pandas as pd
from geopandas3d import GeoDataFrame3D

df = pd.DataFrame({
    "id":    [1, 2, 3],
    "x":     [0, 10, 20],
    "y":     [0, 10, 20],
    "z":     [0, 5, 15],
    "value": [100, 200, 300]
})

gdf3d = GeoDataFrame3D.from_xyz(
    df, "x", "y", "z",
    crs="EPSG:4979",
    height_col="altitude"
)
print(gdf3d)
```

### Performing 3D Operations

```python
# 3D distance between query points
distances = gdf3d.distance3d([(0, 0, 0), (20, 20, 15)])

# 3D centroids
centroids = gdf3d.centroid3d()

# 3D bounds
bounds3d = gdf3d.bounds3d()

# 3D point-in-polygon test
inside = gdf3d.is_point_in_polygon3d([(10, 10, 5)])
```

### Performance Optimization with HNSW

By default GeoPandas3D switches to HNSW for data sets over 100 000 points. You can also specify the method manually.

```python
# Automatic selection
indices, dist = gdf3d.nearest3d(query_points, k=5, method="auto")

# Force HNSW
indices, dist = gdf3d.nearest3d(query_points, k=5, method="HNSW")
```

---

## Advanced Spatial Joins

```python
import pandas as pd
from geopandas3d import GeoDataFrame3D

other_df = pd.DataFrame({
    "oid": [100, 101],
    "x":   [1, 30],
    "y":   [1, 30],
    "z":   [1, 30]
})

other_gdf = GeoDataFrame3D.from_xyz(
    other_df, "x", "y", "z",
    crs="EPSG:4979",
    height_col="altitude"
)

nearest = gdf3d.sjoin_nearest3d(other_gdf, k=1, how="left")
print(nearest)

within = gdf3d.sjoin_within_distance3d(
    other_gdf, max_distance=8.0, how="inner"
)
print(within)
```

---

## 3D Visualization

```python
fig, ax = gdf3d.plot3d(column="value", cmap="viridis")

from geopandas3d import plot3d
fig, ax = plot3d(gdf3d, column="value")
```

---

## Integration with Existing GeoPandas Data

```python
import geopandas as gpd
from geopandas3d import GeoDataFrame3D

gdf2d = gpd.read_file("points.shp")
gdf2d["z"] = [0, 5, 15]
gdf2d["x"] = gdf2d.geometry.x
gdf2d["y"] = gdf2d.geometry.y

gdf3d = GeoDataFrame3D.from_xyz(
    gdf2d[["id", "name", "x", "y", "z"]],
    "x", "y", "z",
    crs="EPSG:4979"
)

print("2D bounds", gdf2d.bounds)
print("3D bounds", gdf3d.bounds3d())
```

---

## API Reference

### GeoDataFrame3D

#### Constructors

- `from_xyz(data, x, y, z, crs=None, height_col="altitude")`  
- `from_points(points, data=None, crs=None, height_col="altitude")`  
- `from_polygons(polygons, data=None, crs=None, height_col="altitude")`  

#### Key Methods

- `get_3d_coordinates()`  
- `bounds3d()`  
- `get_geometry_bounds3d()`  
- `build_sindex()`  
- `nearest3d(points, k=1)`  
- `query_ball3d(points, r)`  
- `sjoin_nearest3d(other, k=1, how="left")`  
- `sjoin_within_distance3d(other, max_distance, how="inner")`  
- `plot3d(**kwargs)`  

#### Properties

- `height_col`  
- `geometry_type`  

### Utility Functions

- `distance3d(p1, p2)`  
- `centroid3d(points)`  
- `polygon_area3d(vertices)`  
- `is_point_in_polygon3d(point, vertices)`  
- `validate_3d_coordinates(coords)`  

### Plotting Functions

- `plot3d(gdf3d, **kwargs)`  
- `plot_points_3d(points, values=None, **kwargs)`  
- `plot_polygons_3d(polygons, heights, **kwargs)`  

---

## Examples

See the `examples/` directory for full demos, including:

- Basic 3D creation and operations  
- 3D spatial indexing and queries  
- 3D spatial joins  
- 3D visualization  
- Integration with existing GeoPandas workflows  

---

## Contributing

Pull requests and issues are welcome.  

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Changelog

### 0.3.0-dev (2025-08-17)

- Added HNSW indexing for large data sets  
- Automatic index method selection  
- New nearest-neighbor spatial join  
- Added Point3D dataclass  
- Batch CRS transformation support  
- Improved error handling and validation  
- Documentation enhancements  

### 0.2.0 (2025-08-17)

- Full rewrite to inherit from GeoDataFrame  
- Required height column for 3D operations  
- New 3D spatial operations and joins  
- Matplotlib-based 3D plots  
- Removed Cython and Plotly dependencies  
- Improved integration with GeoPandas workflows