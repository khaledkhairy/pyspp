# pySHP Test Suite

## Running Tests

### Run All Tests
```bash
python -m pySHP.tests.run_tests
```

Or from the tests directory:
```bash
cd pySHP/tests
python run_tests.py
```

### Run Specific Test Module
```bash
python -m pySHP.tests.run_tests test_sh_basis
python -m pySHP.tests.run_tests test_surface_mesh
python -m pySHP.tests.run_tests test_shp_surface
python -m pySHP.tests.run_tests test_integration
```

### Run with unittest
```bash
python -m unittest discover pySHP/tests
```

## Test Files

### test_sh_basis.py
Tests for `sh_basis` class:
- Basic initialization
- Gaussian quadrature
- Normalization factors
- Spherical harmonic computation

### test_surface_mesh.py
Tests for `surface_mesh` class:
- Mesh initialization
- Sphere mesh generation
- Edge information computation
- Mesh properties (area, volume)
- Reading .off files

### test_shp_surface.py
Tests for `shp_surface` class:
- Initialization modes
- Importing .shp3 files
- Coefficient extraction
- Surface updates

### test_integration.py
Integration tests:
- Loading .off files from disk
- Loading .shp3 files
- Converting meshes to spherical harmonics
- End-to-end workflows

## Test Data

Tests look for data files in:
- `.off` files: `Matlab/shp_toolbox-main/shp_toolbox-main/test_data/off/scientific/`
- `.shp3` files: `Matlab/shp_toolbox-main/shp_toolbox-main/level1/flip_template.shp3`

If test data is not found, tests will be skipped with appropriate messages.

## Example Test Output

```
Running all pySHP tests...
test_init (test_sh_basis.TestSHBasis) ... ok
test_gaussquad (test_sh_basis.TestSHBasis) ... ok
test_N_LK_bosh (test_sh_basis.TestSHBasis) ... ok
test_ylk_bosh (test_sh_basis.TestSHBasis) ... ok
test_init (test_surface_mesh.TestSurfaceMesh) ... ok
test_sphere_mesh_gen (test_surface_mesh.TestSurfaceMesh) ... ok
test_edge_info (test_surface_mesh.TestSurfaceMesh) ... ok
test_props (test_surface_mesh.TestSurfaceMesh) ... ok
test_readoff (test_surface_mesh.TestSurfaceMesh) ... ok
test_init_sphere (test_shp_surface.TestSHPSurface) ... ok
test_init_with_L_max (test_shp_surface.TestSHPSurface) ... ok
test_init_with_basis (test_shp_surface.TestSHPSurface) ... ok
test_import_shp3 (test_shp_surface.TestSHPSurface) ... ok
test_load_off_file (test_integration.TestIntegration) ... ok
test_load_shp3_file (test_integration.TestIntegration) ... ok
test_mesh_to_shp (test_integration.TestIntegration) ... ok

----------------------------------------------------------------------
Ran 15 tests in X.XXXs

OK
```
