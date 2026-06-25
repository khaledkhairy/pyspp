# Changelog

## Version 0.1.0 - Initial Release

### Core Classes
- ✅ `sh_basis` - Spherical harmonics basis computation
- ✅ `sh_surface` - Single function SH surface representation
- ✅ `shp_surface` - 3D shape representation using SH parameterization
- ✅ `surface_mesh` - Mesh operations and properties

### File I/O
- ✅ `.off` file reading/writing (using meshio)
- ✅ `.shp3` file reading (text format parser)

### Level Functions
- ✅ Level 0: `bin2shp` - Binary/mesh to SH conversion
- ✅ Level 1: 
  - `mesh_segmentation_rw` - Random walk mesh segmentation
  - `patch_info_gen` - Patch information generation
- ✅ Level 2:
  - `spherical_parameterization` - Main parameterization pipeline
  - `build_patch_data_structure` - Patch data structure builder

### Visualization
- ✅ PyVista 3D plotting support
- ✅ Matplotlib fallback
- ✅ Mesh display with labels/colors

### Test Suite
- ✅ Unit tests for all core classes
- ✅ Integration tests for file I/O
- ✅ Test scripts converted from MATLAB

### Documentation
- ✅ README.md - Main documentation
- ✅ README_TESTS.md - Test documentation
- ✅ QUICKSTART.md - Quick start guide
- ✅ Examples in `examples/` directory
