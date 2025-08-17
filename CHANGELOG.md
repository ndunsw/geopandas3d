# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-08-17

### **Major Rewrite - Complete Architecture Overhaul**

This release represents a complete rewrite of the package with a much better architecture that properly extends GeoPandas instead of wrapping it.

#### **Added**
- **Proper GeoPandas Inheritance**: `GeoDataFrame3D` now inherits directly from `GeoDataFrame`
- **Required Altitude Column**: Every 3D dataset must include an altitude/height column for 3D operations
- **3D Spatial Indexing**: Efficient 3D spatial queries using scipy.spatial.cKDTree
- **3D Spatial Operations**: 
  - Nearest neighbor queries in 3D space
  - Radius queries within 3D distance
  - 3D spatial joins (nearest and within distance)
- **3D Geometry Support**:
  - 3D points with automatic geometry type detection
  - 3D polygons with area calculations and point-in-polygon tests
  - Mixed geometry types supported
- **3D Visualization**: Matplotlib-based 3D plotting capabilities
- **Utility Functions**: 3D distance, centroid, area, and point-in-polygon calculations
- **Comprehensive Testing**: Full test suite covering all functionality
- **Complete Documentation**: Comprehensive README and API documentation

#### **Changed**
- **Architecture**: Complete rewrite from wrapper pattern to inheritance pattern
- **API Design**: Cleaner, more intuitive API that extends rather than replaces functionality
- **Performance**: Improved spatial indexing and query performance
- **Dependencies**: Updated to require shapely and matplotlib for core functionality

#### **Removed**
- **Wrapper Pattern**: No more complex delegation to underlying GeoDataFrame
- **Cython Extensions**: Simplified to pure Python implementation for better maintainability
- **Plotly Dependencies**: Focused on matplotlib for 3D visualization
- **Complex Geometry Handling**: Streamlined to focus on core 3D operations

#### **Fixed**
- **Geometry Type Detection**: Proper handling of different geometry types (points vs polygons)
- **3D Coordinate Extraction**: Fixed issues with polygon geometry handling
- **Spatial Indexing**: Improved error handling and validation
- **Test Coverage**: All tests now passing with proper numpy array handling

#### **Documentation**
- **Complete API Reference**: All methods and classes documented
- **Usage Examples**: Comprehensive demonstration scripts
- **Design Philosophy**: Clear explanation of architectural decisions
- **Integration Guide**: How to use with existing GeoPandas workflows

#### **Testing**
- **13 Test Cases**: Covering all major functionality
- **Demo Script**: Complete demonstration of all features
- **Error Handling**: Proper validation and error messages

---

## [0.1.0] - 2024-01-01

### **Initial Release**

#### **Added**
- Basic 3D GeoDataFrame wrapper around GeoPandas
- 3D spatial indexing using cKDTree
- Basic 3D plotting capabilities
- Cython extensions for performance optimization

#### **Notes**
- This was a prototype implementation with wrapper pattern
- Replaced by v0.2.0 complete rewrite
