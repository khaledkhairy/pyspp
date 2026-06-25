# Final Step: Spherical Harmonics Projection and Export

## Overview

This document describes the final step of the spherical parameterization pipeline: assembling the parameterized mesh from all patches, converting it to spherical harmonics, and exporting to `.shp3` format.

## Implementation

### 1. Assemble Parameterized Mesh (`assemble_parameterized_mesh.py`)

**Function:** `assemble_parameterized_mesh(m_original, PM)`

Collects theta (t) and phi (p) values from all parameterized patches and assigns them to the original mesh vertices.

- Creates a copy of the original mesh
- Iterates through all patches in `PM['P']`
- Copies t and p values from each patch to the full mesh
- Reports statistics on coverage

**Key Points:**
- Patches use the same vertex indices as the original mesh
- Only non-zero t/p values are assigned (zeros indicate unparameterized vertices)
- If a vertex appears in multiple patches, the first non-zero value is used

### 2. Convert to Spherical Harmonics (`shp_surface.py`)

**Method:** `shp_surface.mesh2shp(m, L_max=None)`

Converts a mesh with t and p values to spherical harmonics representation.

- Creates a `sh_basis` object for the specified L_max
- Creates a `shp_surface` object
- Calls `mesh2shp()` which internally calls `shp_analysis()`
- `shp_analysis()` performs SVD-based least squares fitting to compute spherical harmonics coefficients

**Parameters:**
- `L_max`: Maximum spherical harmonics degree (default: 12, adjust based on mesh complexity)
- `gdim`: Grid dimension for basis computation (default: 60)

**Output:**
- `s.xc`, `s.yc`, `s.zc`: Spherical harmonics coefficients for x, y, z components
- `s.X_o`: Flattened coefficient vector (column-major order, matching MATLAB)
- `s.residual`: Reconstruction error

### 3. Export to .shp3 Format (`shp_surface.export_shp3()`)

**Method:** `shp_surface.export_shp3(filename)`

Exports spherical harmonics coefficients to `.shp3` ASCII format (compatible with MATLAB toolbox).

**File Format:**
```
n_shapes = 1
L_max = 12
n_components = 3
x	y	z
1.234567e+00	2.345678e+00	3.456789e+00
...
```

**Structure:**
- Header: n_shapes, L_max, n_components
- Component tags: x, y, z (and any scalar fields)
- Data rows: coefficients for each (L,K) index, tab-separated

## Usage

### In Notebook

Add a new cell after cell 11 (after `parameterize_patches_cart`) with the following code:

```python
# Final Step: Spherical Harmonics Projection and Export
from pySHP.level1.assemble_parameterized_mesh import assemble_parameterized_mesh
from pySHP.sh_basis import sh_basis
from pySHP.shp_surface import shp_surface

print("="*60)
print("Spherical Harmonics Projection and Export")
print("="*60)

# Step 1: Assemble full parameterized mesh
print("\n[Step 1] Assembling full parameterized mesh from all patches...")
m_param = assemble_parameterized_mesh(m_seg, PM)

# Step 2: Convert to spherical harmonics
print("\n[Step 2] Converting to spherical harmonics...")
L_max = 12  # Adjust as needed
gdim = 60
b = sh_basis(L_max, gdim)
s = shp_surface(L_max, b, m_param)
s.mesh2shp(m_param, L_max=L_max)

# Step 3: Export to .shp3
print("\n[Step 3] Exporting to .shp3 file...")
output_filename = "parameterized_mesh.shp3"
s.export_shp3(output_filename)

print("\nComplete! Output file: " + output_filename)
```

### Standalone Script

See `pySHP/tests/cell_final_shp_export.py` for a complete standalone script with visualization.

## Parameters

### L_max (Spherical Harmonics Degree)

- **Lower values (8-12):** Faster computation, smoother surfaces, less detail
- **Higher values (16-20):** More detail, slower computation, may overfit noise
- **Recommendation:** Start with 12, increase if more detail needed

### gdim (Grid Dimension)

- Controls resolution of basis function computation
- Default: 60 (good balance)
- Higher values: More accurate but slower

## Output Files

### .shp3 File

- ASCII format, tab-separated values
- Contains spherical harmonics coefficients
- Can be imported into MATLAB using `import_shp3()`
- Can be used for shape analysis, morphing, etc.

## Verification

After export, verify the file:

1. **Check file size:** Should be non-zero
2. **Check header:** Should match expected format
3. **Import test:** Try importing back into MATLAB/Python
4. **Visualization:** Compare original mesh with SH reconstruction

## Troubleshooting

### "Cannot export: X_o not computed"
- Run `s.mesh2shp(m_param)` before exporting

### "NaN found in phi or theta"
- Some vertices may not have been parameterized
- Check `assemble_parameterized_mesh` output for unassigned vertices
- May need to fix patch parameterization issues

### Low coverage (< 100%)
- Some vertices not assigned from patches
- Check patch parameterization completeness
- May need to handle unassigned vertices (interpolation or default values)

### Large reconstruction error
- Increase L_max for more detail
- Check parameterization quality
- Verify t and p values are correct

## Next Steps

After export, you can:
1. Import into MATLAB for further analysis
2. Use for shape morphing and comparison
3. Perform statistical shape analysis
4. Generate smooth mesh reconstructions at different resolutions
