# Regression Test: Surface Mesh Segmentation Pipeline

## Overview

`test_surface_mesh_segmentation_regression.py` is a comprehensive regression test that validates the complete surface mesh segmentation and spherical parameterization pipeline. It converts the workflow from `test_surface_mesh_segmentation.ipynb` into an automated test.

## Purpose

This test serves as a **regression test** that should pass whenever you make larger changes to the codebase. It ensures that:

1. The entire pipeline works end-to-end
2. All components integrate correctly
3. Outputs are generated as expected
4. No regressions are introduced

## Test Workflow

The test validates these stages:

1. **Mesh Loading and Preprocessing**
   - Loads mesh from .off file
   - Repairs mesh (removes duplicates, fixes normals)
   - Checks mesh integrity and quality

2. **Mesh Segmentation**
   - Performs random walk segmentation
   - Verifies correct number of patches

3. **Patch Info Generation**
   - Generates patch structure (PM)
   - Creates simplified mesh
   - Identifies key vertices and edges

4. **Spherical Conformal Parameterization**
   - Parameterizes simplified mesh
   - Verifies coverage (>90% of vertices)

5. **Patch Parameterization**
   - Parameterizes all fine mesh patches
   - Verifies each patch has valid t,p values

6. **Mesh Assembly**
   - Assembles full parameterized mesh from patches
   - Verifies coverage (>80% of vertices)

7. **Spherical Harmonics Conversion**
   - Converts to spherical harmonics representation
   - Verifies coefficients are generated

8. **Export**
   - Exports to .shp3 file
   - Verifies file is created and non-empty

## Running the Test

### Run all tests:
```bash
python -m pytest pySHP/tests/test_surface_mesh_segmentation_regression.py -v
```

Or using unittest:
```bash
python -m unittest pySHP.tests.test_surface_mesh_segmentation_regression -v
```

### Run just the regression test:
```bash
cd pySHP/tests
python test_surface_mesh_segmentation_regression.py
```

### Run from project root:
```bash
python -m pySHP.tests.test_surface_mesh_segmentation_regression
```

## Output Files

All outputs are saved to `pySHP/tests/test_outputs/`:

- **`test_output_YYYYMMDD_HHMMSS.shp3`**: Exported spherical harmonics file
- **`test_results_YYYYMMDD_HHMMSS.json`**: Complete test results with statistics

The JSON file contains:
- Test configuration
- Results from each pipeline stage
- Statistics (vertex counts, coverage, etc.)
- Timestamp

## Test Configuration

You can modify the test configuration in `setUpClass()`:

```python
cls.test_config = {
    'topic': 'scientific',           # Test data subdirectory
    'file_name': 'BDH6230_whole_brain.off',  # Mesh file
    'nseeds': 11,                     # Number of segmentation seeds
    'sigma': 1.0,                     # Segmentation parameter
    'curvature_weight': -1e-6,        # Curvature weighting
    'L_max': 12,                      # Spherical harmonics degree
    'gdim': 60                        # Grid dimension
}
```

## Assertions

The test includes assertions at each stage:

- **Mesh loading**: Validates mesh has vertices and faces
- **Segmentation**: Verifies correct number of patches
- **Patch info**: Checks PM structure completeness
- **Parameterization**: Validates coverage thresholds
- **Export**: Verifies file creation

## Integration with CI/CD

This test can be integrated into continuous integration:

```yaml
# Example GitHub Actions workflow
- name: Run regression tests
  run: |
    python -m unittest pySHP.tests.test_surface_mesh_segmentation_regression -v
```

## Troubleshooting

### Test fails at a specific stage

1. Check the output JSON file for detailed statistics
2. Verify the test mesh file exists
3. Check that all dependencies are installed
4. Review the assertion messages for specific failures

### Output files not generated

- Check that `test_outputs/` directory is writable
- Verify file paths are correct
- Check disk space

### Test takes too long

- Reduce `nseeds` for faster segmentation
- Use a smaller test mesh
- Set `verbose=False` and `plot_intermediate=False`

## Adding New Assertions

To add new validation checks, modify the test methods:

```python
def test_complete_pipeline(self):
    # ... existing code ...
    
    # Add new assertion
    self.assertGreater(some_value, threshold, "Custom error message")
    
    # Save result
    self.save_stage_result('custom_stage', {'value': some_value})
```

## Maintenance

- **Update when pipeline changes**: If you modify the pipeline, update the test
- **Review outputs periodically**: Check that outputs are reasonable
- **Adjust thresholds**: Coverage thresholds may need adjustment for different meshes
- **Document custom configurations**: If you add test variants, document them

## Related Files

- `test_surface_mesh_segmentation.ipynb`: Original notebook workflow
- `test_surface_mesh_segmentation.py`: Simpler test version
- `run_tests.py`: Test runner for all tests
