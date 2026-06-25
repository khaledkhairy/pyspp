"""
Integration tests: loading .off files and .shp3 files
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pySHP.surface_mesh import surface_mesh
from pySHP.shp_surface import shp_surface
from pySHP.sh_basis import sh_basis
from pySHP.utils import readoff


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test paths"""
        self.base_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'Matlab',
            'shp_toolbox-main', 'shp_toolbox-main'
        )
        self.test_data_dir = os.path.join(self.base_dir, 'test_data', 'off')
        self.shp3_file = os.path.join(self.base_dir, 'level1', 'flip_template.shp3')
    
    def test_load_off_file(self):
        """Test loading .off file and creating surface_mesh"""
        scientific_dir = os.path.join(self.test_data_dir, 'scientific')
        if not os.path.exists(scientific_dir):
            self.skipTest("Test data directory not found")
        
        off_files = [f for f in os.listdir(scientific_dir) if f.endswith('.off')]
        if not off_files:
            self.skipTest("No .off files found")
        
        test_file = os.path.join(scientific_dir, off_files[0])
        print(f"\nTesting with file: {test_file}")
        
        try:
            X, F = readoff(test_file)
            m = surface_mesh(X, F)
            
            # Verify mesh
            self.assertGreater(len(m.X), 0, "Mesh has no vertices")
            self.assertGreater(len(m.F), 0, "Mesh has no faces")
            print(f"  Loaded mesh: {len(m.X)} vertices, {len(m.F)} faces")
            
            # Compute properties
            m.props()
            self.assertIsNotNone(m.A, "Area not computed")
            self.assertIsNotNone(m.V, "Volume not computed")
            print(f"  Area: {m.A:.2f}, Volume: {m.V:.2f}")
            
        except Exception as e:
            self.fail(f"Failed to load .off file: {e}")
    
    def test_load_shp3_file(self):
        """Test loading .shp3 file and displaying"""
        if not os.path.exists(self.shp3_file):
            self.skipTest("Test .shp3 file not found")
        
        print(f"\nTesting with file: {self.shp3_file}")
        
        try:
            s = shp_surface()
            s.import_shp3(self.shp3_file)
            
            # Verify surface
            self.assertIsNotNone(s.xc, "xc coefficients not loaded")
            self.assertIsNotNone(s.yc, "yc coefficients not loaded")
            self.assertIsNotNone(s.zc, "zc coefficients not loaded")
            print(f"  Loaded SHP surface: L_max={s.L_max}, gdim={s.gdim}")
            
            # Update surface
            s.update()
            self.assertIsNotNone(s.x, "x coordinates not computed")
            self.assertIsNotNone(s.y, "y coordinates not computed")
            self.assertIsNotNone(s.z, "z coordinates not computed")
            print(f"  Surface updated: shape {s.x.shape}")
            
        except Exception as e:
            self.fail(f"Failed to load .shp3 file: {e}")
    
    def test_mesh_to_shp(self):
        """Test converting mesh to spherical harmonics"""
        scientific_dir = os.path.join(self.test_data_dir, 'scientific')
        if not os.path.exists(scientific_dir):
            self.skipTest("Test data directory not found")
        
        off_files = [f for f in os.listdir(scientific_dir) if f.endswith('.off')]
        if not off_files:
            self.skipTest("No .off files found")
        
        test_file = os.path.join(scientific_dir, off_files[0])
        
        try:
            # Load mesh
            X, F = readoff(test_file)
            m = surface_mesh(X, F)
            
            # Optimize mesh
            m.meshresample_keepratio = 0.8
            m = m.optimize_mesh()
            
            # Create SH representation
            L_max = 8
            b = sh_basis(L_max, 60)
            s = shp_surface(L_max, b, m)
            
            self.assertIsNotNone(s.xc)
            self.assertIsNotNone(s.yc)
            self.assertIsNotNone(s.zc)
            print(f"\n  Converted mesh to SH: L_max={s.L_max}")
            
        except Exception as e:
            self.fail(f"Failed mesh to SH conversion: {e}")
    
    def test_plot_surface_mesh(self):
        """Test plotting surface mesh"""
        scientific_dir = os.path.join(self.test_data_dir, 'scientific')
        if not os.path.exists(scientific_dir):
            self.skipTest("Test data directory not found")
        
        off_files = [f for f in os.listdir(scientific_dir) if f.endswith('.off')]
        if not off_files:
            self.skipTest("No .off files found")
        
        test_file = os.path.join(scientific_dir, off_files[0])
        
        try:
            X, F = readoff(test_file)
            m = surface_mesh(X, F)
            m.props()
            
            # Test all plot functions exist
            self.assertTrue(hasattr(m, 'plot'))
            self.assertTrue(hasattr(m, 'plot_H'))
            self.assertTrue(hasattr(m, 'plot_fast'))
            self.assertTrue(hasattr(m, 'plot_quality'))
            self.assertTrue(hasattr(m, 'plot_map_quality'))
            
        except Exception as e:
            self.fail(f"Failed plotting test: {e}")
    
    def test_plot_shp_surface(self):
        """Test plotting SHP surface"""
        if not os.path.exists(self.shp3_file):
            self.skipTest("Test .shp3 file not found")
        
        try:
            s = shp_surface()
            s.import_shp3(self.shp3_file)
            s.update()
            
            # Test all plot functions exist
            self.assertTrue(hasattr(s, 'plot'))
            self.assertTrue(hasattr(s, 'plot_H'))
            self.assertTrue(hasattr(s, 'plot_fast'))
            
        except Exception as e:
            self.fail(f"Failed plotting test: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
