# cython: language_level=3
# distutils: language=c

"""
C-optimized spatial operations for geopandas3d.
This module provides fast implementations of spatial operations using Cython.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from cpython cimport array
import array

# Define numpy data types
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def distance3d_fast(np.ndarray[DTYPE_t, ndim=1] p1, np.ndarray[DTYPE_t, ndim=1] p2):
    """Fast 3D Euclidean distance calculation using C."""
    cdef DTYPE_t dx = p1[0] - p2[0]
    cdef DTYPE_t dy = p1[1] - p2[1]
    cdef DTYPE_t dz = p1[2] - p2[2]
    return sqrt(dx*dx + dy*dy + dz*dz)

def distances3d_batch(np.ndarray[DTYPE_t, ndim=2] points1, np.ndarray[DTYPE_t, ndim=2] points2):
    """Fast batch calculation of 3D distances between two sets of points."""
    cdef int n1 = points1.shape[0]
    cdef int n2 = points2.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] distances = np.zeros((n1, n2), dtype=DTYPE)
    cdef int i, j
    cdef DTYPE_t dx, dy, dz
    
    for i in range(n1):
        for j in range(n2):
            dx = points1[i, 0] - points2[j, 0]
            dy = points1[i, 1] - points2[j, 1]
            dz = points1[i, 2] - points2[j, 2]
            distances[i, j] = sqrt(dx*dx + dy*dy + dz*dz)
    
    return distances

def nearest_neighbor_fast(np.ndarray[DTYPE_t, ndim=2] query_points, np.ndarray[DTYPE_t, ndim=2] data_points, int k=1):
    """Fast k-nearest neighbor search using brute force (for small datasets)."""
    cdef int n_queries = query_points.shape[0]
    cdef int n_data = data_points.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] distances = np.zeros((n_queries, n_data), dtype=DTYPE)
    cdef np.ndarray[np.int64_t, ndim=2] indices = np.zeros((n_queries, k), dtype=np.int64)
    cdef np.ndarray[DTYPE_t, ndim=2] k_distances = np.zeros((n_queries, k), dtype=DTYPE)
    cdef int i, j, m
    cdef DTYPE_t dx, dy, dz, dist
    cdef DTYPE_t max_dist
    
    # Calculate all distances
    for i in range(n_queries):
        for j in range(n_data):
            dx = query_points[i, 0] - data_points[j, 0]
            dy = query_points[i, 1] - data_points[j, 1]
            dz = query_points[i, 2] - data_points[j, 2]
            distances[i, j] = sqrt(dx*dx + dy*dy + dz*dz)
    
    # Find k nearest for each query point
    for i in range(n_queries):
        # Simple selection sort for k nearest
        for m in range(k):
            min_idx = m
            min_dist = distances[i, m]
            for j in range(m+1, n_data):
                if distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    min_idx = j
            
            # Swap
            if min_idx != m:
                distances[i, m], distances[i, min_idx] = distances[i, min_idx], distances[i, m]
                indices[i, m] = min_idx
                k_distances[i, m] = min_dist
            else:
                indices[i, m] = m
                k_distances[i, m] = min_dist
    
    return indices, k_distances

def point_in_polygon3d_fast(np.ndarray[DTYPE_t, ndim=1] point, np.ndarray[DTYPE_t, ndim=2] polygon):
    """Fast point-in-polygon test using ray casting algorithm."""
    cdef int n_vertices = polygon.shape[0]
    cdef int i, j
    cdef bint inside = False
    cdef DTYPE_t x = point[0]
    cdef DTYPE_t y = point[1]
    cdef DTYPE_t xi, yi, xj, yj
    
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        xi, yi = polygon[i, 0], polygon[i, 1]
        xj, yj = polygon[j, 0], polygon[j, 1]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
    
    return inside

def polygon_area3d_fast(np.ndarray[DTYPE_t, ndim=2] vertices):
    """Fast polygon area calculation using shoelace formula."""
    cdef int n = vertices.shape[0]
    cdef int i, j
    cdef DTYPE_t area = 0.0
    cdef DTYPE_t xi, yi, xj, yj
    
    for i in range(n):
        j = (i + 1) % n
        xi, yi = vertices[i, 0], vertices[i, 1]
        xj, yj = vertices[j, 0], vertices[j, 1]
        area += xi * yj
        area -= xj * yi
    
    return fabs(area) / 2.0

def bounds3d_fast(np.ndarray[DTYPE_t, ndim=2] points):
    """Fast 3D bounds calculation."""
    cdef int n = points.shape[0]
    cdef int i
    cdef DTYPE_t min_x, min_y, min_z, max_x, max_y, max_z
    
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    min_x = max_x = points[0, 0]
    min_y = max_y = points[0, 1]
    min_z = max_z = points[0, 2]
    
    for i in range(1, n):
        if points[i, 0] < min_x:
            min_x = points[i, 0]
        elif points[i, 0] > max_x:
            max_x = points[i, 0]
        
        if points[i, 1] < min_y:
            min_y = points[i, 1]
        elif points[i, 1] > max_y:
            max_y = points[i, 1]
        
        if points[i, 2] < min_z:
            min_z = points[i, 2]
        elif points[i, 2] > max_z:
            max_z = points[i, 2]
    
    return (min_x, min_y, min_z, max_x, max_y, max_z)

def spatial_join_within_distance_fast(np.ndarray[DTYPE_t, ndim=2] left_points, 
                                     np.ndarray[DTYPE_t, ndim=2] right_points, 
                                     DTYPE_t max_distance):
    """Fast spatial join within distance using brute force."""
    cdef int n_left = left_points.shape[0]
    cdef int n_right = right_points.shape[0]
    cdef list neighbors = []
    cdef int i, j
    cdef DTYPE_t dx, dy, dz, dist
    
    for i in range(n_left):
        point_neighbors = []
        for j in range(n_right):
            dx = left_points[i, 0] - right_points[j, 0]
            dy = left_points[i, 1] - right_points[j, 1]
            dz = left_points[i, 2] - right_points[j, 2]
            dist = sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist <= max_distance:
                point_neighbors.append(j)
        
        neighbors.append(point_neighbors)
    
    return neighbors
