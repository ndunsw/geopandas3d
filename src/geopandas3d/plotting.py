"""
3D plotting capabilities for GeoDataFrame3D using matplotlib.
"""

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn(
        "Matplotlib not available. Install with: pip install matplotlib", stacklevel=2
    )


def plot3d(
    gdf3d,
    column: Optional[str] = None,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    s: int = 50,
    **kwargs,
) -> tuple[Any, Any]:
    """Create a 3D plot of the GeoDataFrame3D.

    Args:
        gdf3d: GeoDataFrame3D to plot
        column: Column to use for coloring points/polygons
        cmap: Colormap for the plot
        figsize: Figure size as (width, height)
        alpha: Transparency (0-1)
        s: Point size for scatter plots
        **kwargs: Additional plotting parameters

    Returns:
        Tuple of (figure, axes)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for this method. Install with: pip install matplotlib"
        )

    if len(gdf3d) == 0:
        warnings.warn("Empty GeoDataFrame3D - nothing to plot", stacklevel=2)
        return None, None

    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Get 3D coordinates
    coords = gdf3d.get_3d_coordinates()
    valid_mask = ~np.isnan(coords).any(axis=1)

    if not valid_mask.any():
        warnings.warn("No valid 3D coordinates found for plotting", stacklevel=2)
        return fig, ax

    valid_coords = coords[valid_mask]

    # Prepare color data
    if column is not None and column in gdf3d.columns:
        color_data = gdf3d[column].iloc[valid_mask]
        if pd.api.types.is_numeric_dtype(color_data):
            scatter = ax.scatter(
                valid_coords[:, 0],
                valid_coords[:, 1],
                valid_coords[:, 2],
                c=color_data,
                cmap=cmap,
                s=s,
                alpha=alpha,
                **kwargs,
            )
            plt.colorbar(scatter, ax=ax, label=column)
        else:
            # Categorical data
            unique_categories = color_data.unique()
            colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_categories)))

            for i, category in enumerate(unique_categories):
                mask = color_data == category
                if mask.any():
                    cat_coords = valid_coords[mask]
                    ax.scatter(
                        cat_coords[:, 0],
                        cat_coords[:, 1],
                        cat_coords[:, 2],
                        c=[colors[i]],
                        label=str(category),
                        s=s,
                        alpha=alpha,
                        **kwargs,
                    )
            ax.legend()
    else:
        # No column specified - use default colors
        ax.scatter(
            valid_coords[:, 0],
            valid_coords[:, 1],
            valid_coords[:, 2],
            s=s,
            alpha=alpha,
            **kwargs,
        )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Plot of {len(gdf3d)} {gdf3d.geometry_type}s")

    # Set equal aspect ratio for better visualization
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        # Older matplotlib versions don't have set_box_aspect
        pass

    return fig, ax


def plot_points_3d(
    points: np.ndarray,
    values: Optional[np.ndarray] = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    s: int = 50,
    alpha: float = 0.7,
    **kwargs,
) -> tuple[Any, Any]:
    """Create a 3D scatter plot of points.

    Args:
        points: Array of (n, 3) coordinates
        values: Optional array of values for coloring
        figsize: Figure size as (width, height)
        cmap: Colormap for the plot
        s: Point size
        alpha: Transparency (0-1)
        **kwargs: Additional plotting parameters

    Returns:
        Tuple of (figure, axes)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required. Install with: pip install matplotlib"
        )

    if len(points) == 0:
        warnings.warn("No points to plot", stacklevel=2)
        return None, None

    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Filter valid coordinates
    valid_mask = ~np.isnan(points).any(axis=1)
    if not valid_mask.any():
        warnings.warn("No valid 3D coordinates found for plotting", stacklevel=2)
        return fig, ax

    valid_points = points[valid_mask]

    # Plot points
    if values is not None:
        valid_values = values[valid_mask]
        if pd.api.types.is_numeric_dtype(valid_values):
            scatter = ax.scatter(
                valid_points[:, 0],
                valid_points[:, 1],
                valid_points[:, 2],
                c=valid_values,
                cmap=cmap,
                s=s,
                alpha=alpha,
                **kwargs,
            )
            plt.colorbar(scatter, ax=ax)
        else:
            # Categorical values
            unique_categories = pd.Series(valid_values).unique()
            colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_categories)))

            for i, category in enumerate(unique_categories):
                mask = valid_values == category
                if mask.any():
                    cat_points = valid_points[mask]
                    ax.scatter(
                        cat_points[:, 0],
                        cat_points[:, 1],
                        cat_points[:, 2],
                        c=[colors[i]],
                        label=str(category),
                        s=s,
                        alpha=alpha,
                        **kwargs,
                    )
            ax.legend()
    else:
        ax.scatter(
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
            s=s,
            alpha=alpha,
            **kwargs,
        )

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Scatter Plot of {len(valid_points)} Points")

    # Set equal aspect ratio
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    return fig, ax


def plot_polygons_3d(
    polygons: list,
    heights: list,
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    **kwargs,
) -> tuple[Any, Any]:
    """Create a 3D plot of polygons at different heights.

    Args:
        polygons: List of polygon geometries (Shapely objects)
        heights: List of heights for each polygon
        figsize: Figure size as (width, height)
        alpha: Transparency (0-1)
        **kwargs: Additional plotting parameters

    Returns:
        Tuple of (figure, axes)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required. Install with: pip install matplotlib"
        )

    if len(polygons) == 0:
        warnings.warn("No polygons to plot", stacklevel=2)
        return None, None

    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot each polygon
    for i, (polygon, height) in enumerate(zip(polygons, heights)):
        if polygon is None or pd.isna(height):
            continue

        try:
            # Extract polygon coordinates
            if hasattr(polygon, "exterior"):
                coords = list(polygon.exterior.coords)
            else:
                coords = list(polygon.coords)

            if len(coords) < 3:
                continue

            # Create 3D coordinates
            x_coords = [coord[0] for coord in coords]
            y_coords = [coord[1] for coord in coords]
            z_coords = [height] * len(coords)

            # Plot polygon as a filled surface
            ax.plot_trisurf(x_coords, y_coords, z_coords, alpha=alpha, **kwargs)

        except Exception as e:
            warnings.warn(f"Error plotting polygon {i}: {e}", stacklevel=2)
            continue

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Polygon Plot of {len(polygons)} Polygons")

    # Set equal aspect ratio
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    return fig, ax
