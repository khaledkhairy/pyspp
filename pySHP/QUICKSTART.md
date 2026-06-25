# pySHP Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Quick Examples

### 1. Load and Display .off Mesh

```python
from pySHP import surface_mesh
from pySHP.utils import readoff

# Load mesh
X, F = readoff('path/to/mesh.off')
m = surface_mesh(X, F)

# Compute properties
m.props()
print(f"Area: {m.A:.2f}, Volume: {m.V:.2f}")

# Display
m.plot()
```

### 2. Load and Display .shp3 File

```python
from pySHP import shp_surface

# Load SHP surface
s = shp_surface('path/to/file.shp3')

# Display
s.plot()
```

### 3. Convert Mesh to Spherical Harmonics

```python
from pySHP import surface_mesh, sh_basis, shp_surface
from pySHP.utils import readoff, kk_cart2sph

# Load and optimize mesh
X, F = readoff('mesh.off')
m = surface_mesh(X, F)
m = m.optimize_mesh()

# Create SH representation
L_max = 16
b = sh_basis(L_max, 60)
s = shp_surface(L_max, b, m)

# Display
s.plot()
```

### 4. Mesh Segmentation

```python
from pySHP.level1 import mesh_segmentation_rw, patch_info_gen

# Segment mesh into patches
m, L, slix, P, Pconn = mesh_segmentation_rw(m, nseeds=20)

# Generate patch information
m, PM, Pconn = patch_info_gen(m, P, Pconn)

# Display segmented mesh
m.plot()  # Will show different colors for different patches
```

### 5. Full Pipeline

```python
from pySHP import surface_mesh, sh_basis, shp_surface
from pySHP.level1 import mesh_segmentation_rw, patch_info_gen
from pySHP.level2 import spherical_parameterization
from pySHP.utils import readoff

# 1. Load mesh
X, F = readoff('mesh.off')
m = surface_mesh(X, F)
m = m.optimize_mesh()

# 2. Segment
m, L, slix, P, Pconn = mesh_segmentation_rw(m, nseeds=20)
m, PM, Pconn = patch_info_gen(m, P, Pconn)

# 3. Parameterize
opts = {'lambdaA': 1.0, 'lambda': 1e2, 'maxiter': 1000}
mp, PM, failed = spherical_parameterization(m, 20, opts)

# 4. Convert to SH
L_max = 16
b = sh_basis(L_max, 60)
s = shp_surface(L_max, b, mp)

# 5. Display
s.plot()
```

## Running Tests

```bash
# All tests
python -m pySHP.tests.run_tests

# Specific test
python -m pySHP.tests.run_tests test_integration

# Test script
python pySHP/test_scripts/wbd_echinocyte.py
```

## File Locations

- **Test data**: `Matlab/shp_toolbox-main/shp_toolbox-main/test_data/off/`
- **.shp3 files**: `Matlab/shp_toolbox-main/shp_toolbox-main/level1/`
- **Test scripts**: `pySHP/test_scripts/`
- **Examples**: `pySHP/examples/`
