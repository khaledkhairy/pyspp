"""
Regression test for complete surface mesh segmentation and spherical parameterization pipeline.

This test converts the workflow from test_surface_mesh_segmentation.ipynb into an automated
test that can be run as part of the test suite. It verifies that the entire pipeline works
correctly and saves outputs for inspection.

Test workflow:
1. Load and preprocess mesh
2. Mesh segmentation (random walk)
3. Patch info generation
4. Spherical conformal parameterization
5. Patch parameterization
6. Assemble full parameterized mesh
7. Convert to spherical harmonics
8. Export to .shp3 file

Outputs are saved to test_outputs/ directory for inspection.
"""

import unittest
import numpy as np
import sys
import os
import json
from datetime import datetime
from io import StringIO

# Add code directory to path
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from pySHP.surface_mesh import surface_mesh
from pySHP.shp_surface import shp_surface
from pySHP.sh_basis import sh_basis
from pySHP.utils import readoff
from pySHP.level1.mesh_segmentation_rw import mesh_segmentation_rw
from pySHP.level1.patch_info_gen import patch_info_gen
from pySHP.level1.parameterize_patches_cart import parameterize_patches_cart
from pySHP.level1.assemble_parameterized_mesh import assemble_parameterized_mesh
from pySHP.level2.spherical_conformal_parameterization import spherical_conformal_parameterization


class TestSurfaceMeshSegmentationRegression(unittest.TestCase):
    """Regression test for complete segmentation and parameterization pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests"""
        # Create output directory
        cls.output_dir = os.path.join(os.path.dirname(__file__), 'test_outputs')
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Test configuration
        cls.test_config = {
            'topic': 'scientific',
            'file_name': 'BDH6230_whole_brain.off',
            'nseeds': 11,
            'sigma': 1.0,
            'curvature_weight': -1e-6,
            'L_max': 12,
            'gdim': 60
        }
        
        # Find test data file
        base_dir = os.path.join(code_dir, 'Matlab', 'shp_toolbox-main', 'shp_toolbox-main')
        cls.test_file = os.path.join(base_dir, 'test_data', 'off', 
                                     cls.test_config['topic'], cls.test_config['file_name'])
        
        if not os.path.exists(cls.test_file):
            # Try alternative file
            cls.test_file = os.path.join(base_dir, 'test_data', 'off', 
                                        'misc_shapes', 'mushroom.off')
            if not os.path.exists(cls.test_file):
                raise FileNotFoundError(f"Test mesh file not found. Tried: {cls.test_file}")
        
        # Initialize results dictionary
        cls.results = {
            'test_file': cls.test_file,
            'config': cls.test_config,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
    
    def setUp(self):
        """Set up for each test"""
        self.stdout_capture = StringIO()
        self.stderr_capture = StringIO()
    
    def save_stage_result(self, stage_name, data):
        """Save result from a pipeline stage"""
        self.__class__.results['stages'][stage_name] = data
    
    def test_complete_pipeline(self):
        """Test the complete segmentation and parameterization pipeline"""
        print("\n" + "="*60)
        print("Regression Test: Complete Surface Mesh Segmentation Pipeline")
        print("="*60)
        
        # Stage 1: Load and preprocess mesh
        print("\n[Stage 1] Loading and preprocessing mesh...")
        X, F = readoff(self.__class__.test_file)
        m = surface_mesh(X, F)
        
        # Mesh preprocessing
        m.repair_mesh()
        components = m.find_disconnected_surfaces()
        m.keep_largest_surface()
        mesh_info = m.info()
        integrity_check = m.check_mesh_integrity()
        quality_stats = m.print_mesh_quality()
        
        # Assertions
        self.assertGreater(len(m.X), 0, "Mesh should have vertices")
        self.assertGreater(len(m.F), 0, "Mesh should have faces")
        self.assertTrue(integrity_check, "Mesh integrity check should pass")
        
        self.save_stage_result('mesh_loading', {
            'n_vertices': len(m.X),
            'n_faces': len(m.F),
            'mesh_info': mesh_info,
            'quality_stats': quality_stats
        })
        print(f"  ✓ Loaded mesh: {len(m.X)} vertices, {len(m.F)} faces")
        
        # Stage 2: Mesh segmentation
        print("\n[Stage 2] Performing mesh segmentation...")
        nseeds = self.__class__.test_config['nseeds']
        sigma = self.__class__.test_config['sigma']
        curvature_weight = self.__class__.test_config['curvature_weight']
        
        ms, L, slix, P, Pconn = mesh_segmentation_rw(
            m, nseeds, sigma, 
            curvature_weight=curvature_weight, 
            verbose=False,  # Set to False for cleaner test output
            plot_intermediate=False
        )
        
        # Assertions
        self.assertIsNotNone(ms, "Segmented mesh should not be None")
        self.assertTrue(hasattr(ms, 'face_labels'), "Segmented mesh should have face_labels")
        self.assertIsNotNone(ms.face_labels, "face_labels should not be None")
        unique_labels = np.unique(ms.face_labels)
        self.assertEqual(len(unique_labels), nseeds, 
                        f"Should have {nseeds} unique labels, got {len(unique_labels)}")
        
        self.save_stage_result('segmentation', {
            'n_patches': len(unique_labels),
            'unique_labels': unique_labels.tolist(),
            'n_seeds': nseeds
        })
        print(f"  ✓ Segmentation complete: {len(unique_labels)} patches")
        
        # Stage 3: Patch info generation
        print("\n[Stage 3] Generating patch information...")
        m_seg, PM, Pconn = patch_info_gen(ms, P, Pconn)
        
        # Assertions
        self.assertIsNotNone(PM, "PM structure should not be None")
        self.assertIn('npatches', PM, "PM should have 'npatches' key")
        self.assertEqual(PM['npatches'], len(unique_labels), 
                        "PM.npatches should match number of unique labels")
        self.assertIn('P', PM, "PM should have 'P' key")
        self.assertIn('pm', PM, "PM should have 'pm' key")
        
        self.save_stage_result('patch_info', {
            'n_patches': PM['npatches'],
            'simplified_mesh_faces': len(PM['pm'].F) if 'pm' in PM else 0,
            'has_keys': 'keys' in PM and len(PM['keys']) > 0,
            'n_edges': len(PM['Edges']) if 'Edges' in PM else 0
        })
        print(f"  ✓ Patch info generated: {PM['npatches']} patches")
        
        # Stage 4: Spherical conformal parameterization
        print("\n[Stage 4] Performing spherical conformal parameterization...")
        PM['spm'] = spherical_conformal_parameterization(PM['pm'])
        
        # Assertions
        self.assertIsNotNone(PM['spm'], "Spherical mesh should not be None")
        self.assertTrue(hasattr(PM['spm'], 't'), "Spherical mesh should have t")
        self.assertTrue(hasattr(PM['spm'], 'p'), "Spherical mesh should have p")
        self.assertIsNotNone(PM['spm'].t, "t should not be None")
        self.assertIsNotNone(PM['spm'].p, "p should not be None")
        
        valid_tp = (PM['spm'].t != 0) & (PM['spm'].p != 0)
        n_valid = np.sum(valid_tp)
        self.assertGreater(n_valid, len(PM['spm'].X) * 0.9, 
                          "At least 90% of vertices should have valid t,p values")
        
        self.save_stage_result('spherical_parameterization', {
            'n_vertices': len(PM['spm'].X),
            'n_valid_tp': int(n_valid),
            'coverage': float(n_valid / len(PM['spm'].X))
        })
        print(f"  ✓ Spherical parameterization complete: {n_valid}/{len(PM['spm'].X)} vertices parameterized")
        
        # Stage 5: Patch parameterization
        print("\n[Stage 5] Parameterizing patches...")
        PM = parameterize_patches_cart(PM, plot_flag=0)  # Set to 0 for cleaner output
        
        # Assertions
        self.assertIsNotNone(PM, "PM should not be None after patch parameterization")
        for pix in range(PM['npatches']):
            patm = PM['P'][pix][0]
            self.assertTrue(hasattr(patm, 't'), f"Patch {pix} should have t")
            self.assertTrue(hasattr(patm, 'p'), f"Patch {pix} should have p")
            
            # Check that at least some vertices are parameterized
            if patm.t is not None and patm.p is not None:
                valid_tp = (patm.t != 0) & (patm.p != 0)
                n_valid = np.sum(valid_tp)
                # At least 50% of border vertices should be parameterized
                if hasattr(patm, 'border_vertex') and patm.border_vertex is not None:
                    n_border = np.sum(patm.border_vertex)
                    if n_border > 0:
                        border_valid = np.sum(valid_tp & patm.border_vertex)
                        coverage = border_valid / n_border if n_border > 0 else 0
                        self.assertGreater(coverage, 0.5, 
                                          f"Patch {pix} should have >50% border vertices parameterized")
        
        self.save_stage_result('patch_parameterization', {
            'n_patches': PM['npatches'],
            'all_patches_parameterized': True
        })
        print(f"  ✓ Patch parameterization complete: {PM['npatches']} patches")
        
        # Stage 6: Assemble full parameterized mesh
        print("\n[Stage 6] Assembling full parameterized mesh...")
        m_param = assemble_parameterized_mesh(m_seg, PM)
        
        # Assertions
        self.assertIsNotNone(m_param, "Parameterized mesh should not be None")
        self.assertTrue(hasattr(m_param, 't'), "Parameterized mesh should have t")
        self.assertTrue(hasattr(m_param, 'p'), "Parameterized mesh should have p")
        
        valid_tp = (m_param.t != 0) & (m_param.p != 0)
        n_valid = np.sum(valid_tp)
        coverage = n_valid / len(m_param.X) if len(m_param.X) > 0 else 0
        
        self.assertGreater(coverage, 0.8, 
                          "At least 80% of vertices should be parameterized")
        
        self.save_stage_result('mesh_assembly', {
            'n_vertices': len(m_param.X),
            'n_valid_tp': int(n_valid),
            'coverage': float(coverage)
        })
        print(f"  ✓ Mesh assembly complete: {n_valid}/{len(m_param.X)} vertices ({coverage*100:.1f}%)")
        
        # Stage 7: Convert to spherical harmonics
        print("\n[Stage 7] Converting to spherical harmonics...")
        L_max = self.__class__.test_config['L_max']
        gdim = self.__class__.test_config['gdim']
        
        # Ensure mesh doesn't trigger map2sphere
        m_param.needs_map2sphere = False
        
        b = sh_basis(L_max, gdim)
        s = shp_surface(L_max, b, m_param)
        
        # Assertions
        self.assertIsNotNone(s, "Spherical harmonics surface should not be None")
        self.assertIsNotNone(s.xc, "Should have x coefficients")
        self.assertIsNotNone(s.yc, "Should have y coefficients")
        self.assertIsNotNone(s.zc, "Should have z coefficients")
        self.assertEqual(s.L_max, L_max, "L_max should match")
        
        self.save_stage_result('spherical_harmonics', {
            'L_max': s.L_max,
            'n_coefficients': len(s.xc),
            'has_residual': hasattr(s, 'residual') and s.residual is not None
        })
        print(f"  ✓ Spherical harmonics conversion complete: L_max={s.L_max}, {len(s.xc)} coefficients")
        
        # Stage 8: Export to .shp3
        print("\n[Stage 8] Exporting to .shp3 file...")
        output_filename = os.path.join(self.__class__.output_dir, 
                                      f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.shp3")
        s.export_shp3(output_filename)
        
        # Assertions
        self.assertTrue(os.path.exists(output_filename), 
                       f"Output file should exist: {output_filename}")
        file_size = os.path.getsize(output_filename)
        self.assertGreater(file_size, 0, "Output file should not be empty")
        
        self.save_stage_result('export', {
            'output_file': output_filename,
            'file_size': file_size
        })
        print(f"  ✓ Export complete: {output_filename} ({file_size} bytes)")
        
        # Save test results summary
        results_file = os.path.join(self.__class__.output_dir, 
                                   f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(self.__class__.results, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("All pipeline stages completed successfully!")
        print(f"Results saved to: {results_file}")
        print("="*60)
    
    def test_pipeline_components(self):
        """Test individual pipeline components for quick debugging"""
        # This is a lighter test that can be used for quick validation
        print("\n" + "="*60)
        print("Quick Component Test")
        print("="*60)
        
        # Load mesh
        X, F = readoff(self.__class__.test_file)
        m = surface_mesh(X, F)
        m.repair_mesh()
        m.keep_largest_surface()
        
        # Quick segmentation with fewer seeds
        ms, L, slix, P, Pconn = mesh_segmentation_rw(
            m, nseeds=3, sigma=1.0, 
            curvature_weight=-1e-6, 
            verbose=False, 
            plot_intermediate=False
        )
        
        # Generate patch info
        m_seg, PM, Pconn = patch_info_gen(ms, P, Pconn)
        
        # Basic assertions
        self.assertGreater(PM['npatches'], 0, "Should have at least one patch")
        self.assertIn('pm', PM, "Should have simplified mesh")
        
        print("  ✓ Quick component test passed")


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSurfaceMeshSegmentationRegression)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
