# pySHP - Python Spherical Harmonics Parameterization

A Python translation of the MATLAB shp_toolbox for spherical parameterization and spherical harmonics analysis of 3D meshes.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package (optional, for development):
```bash
pip install -e .
```

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- pyshtools >= 4.10.0 (for spherical harmonics)
- trimesh >= 3.15.0 (for mesh operations)
- meshio >= 5.3.0 (for file I/O)
- pyvista >= 0.38.0 (for 3D visualization)
- matplotlib >= 3.5.0 (for plotting fallback)

## Usage

### Loading .off Files

```python
from pySHP import surface_mesh
from pySHP.utils import readoff

# Read a mesh file
X, F = readoff('mesh.off')
m = surface_mesh(X, F)

# Optimize mesh
m = m.optimize_mesh()

# Compute properties
m.props()
print(f"Area: {m.A}, Volume: {m.V}")

# Plot
m.plot()
```

### Loading .shp3 Files

```python
from pySHP import shp_surface

# Method 1: Direct initialization
s = shp_surface('path/to/file.shp3')

# Method 2: Import after creation
s = shp_surface()
s.import_shp3('path/to/file.shp3', dim=60)

# Update and display
s.update()
s.plot()
```

### Converting Mesh to Spherical Harmonics

```python
from pySHP import surface_mesh, sh_basis, shp_surface
from pySHP.utils import readoff, kk_cart2sph

# Read mesh
X, F = readoff('mesh.off')
m = surface_mesh(X, F)

# Optimize
m = m.optimize_mesh()

# Create spherical harmonics basis
L_max = 16
gdim = 60
b = sh_basis(L_max, gdim)

# Convert to SHP
s = shp_surface(L_max, b, m)

# Plot
s.plot()
```

### Running Test Scripts

```bash
# Test with echinocyte mesh
python pySHP/test_scripts/wbd_echinocyte.py

# Test with hand mesh
python pySHP/test_scripts/wbd_hand.py

# Test with mushroom mesh
python pySHP/test_scripts/wbd_mushroom.py

# Test with brain mesh
python pySHP/test_scripts/wbb_BDH6230.py
```

### Running Tests

```bash
# Run all tests
python -m pySHP.tests.run_tests

# Run specific test module
python -m pySHP.tests.run_tests test_integration
```

See `README_TESTS.md` for more details.

## Classes

### `sh_basis`
Spherical harmonics basis class for computing SH basis functions.

### `sh_surface`
Spherical harmonics surface representation (single function).

### `shp_surface`
Spherical harmonics parameterized surface (3D shape representation).

### `surface_mesh`
Surface mesh class for triangular/quad meshes with operations like:
- Mesh optimization
- Property calculation (area, volume, curvature)
- Sphere mesh generation
- Edge information computation

## Status

This is an initial translation of the MATLAB codebase. Some features are still being implemented:

- ✅ Core classes (sh_basis, sh_surface, shp_surface, surface_mesh)
- ✅ Basic mesh operations
- ✅ Spherical harmonics basis computation
- ✅ File I/O (readoff/writeoff)
- ✅ Test scripts with visualization
- ⏳ Full parameterization pipeline (level1, level2 functions)
- ⏳ Advanced mesh segmentation
- ⏳ Complete curvature calculations

## Notes

- The code uses PyVista for 3D visualization when available, with matplotlib as fallback
- Mesh file paths in test scripts may need to be adjusted to match your file structure
- Some MATLAB-specific optimizations may need Python equivalents

## License

See LICENSE file (inherited from original MATLAB toolbox).
