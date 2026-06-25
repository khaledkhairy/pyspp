"""
Tests for shp_surface class including .shp3 file reading
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pySHP.shp_surface import shp_surface
from pySHP.sh_basis import sh_basis
from pySHP.surface_mesh import surface_mesh


class TestSHPSurface(unittest.TestCase):
    
    def test_init_sphere(self):
        """Test initialization as sphere"""
        s = shp_surface()
        self.assertIsNotNone(s.xc)
        self.assertIsNotNone(s.yc)
        self.assertIsNotNone(s.zc)
        self.assertEqual(s.L_max, 12)
    
    def test_init_with_L_max(self):
        """Test initialization with L_max"""
        s = shp_surface(8)
        self.assertEqual(s.L_max, 8)
        self.assertIsNotNone(s.basis)
    
    def test_init_with_basis(self):
        """Test initialization with basis"""
        b = sh_basis(10, 60)
        s = shp_surface(10, b)
        self.assertEqual(s.L_max, 10)
        self.assertEqual(s.basis, b)
    
    def test_import_shp3(self):
        """Test importing .shp3 file and verify geometric properties are computed"""
        # Try to find test .shp3 file (look in tests directory first, then try MATLAB examples)
        test_file = os.path.join(os.path.dirname(__file__), 'parameterized_mesh.shp3')
        
        # If not found, try MATLAB example file
        if not os.path.exists(test_file):
            test_file = os.path.join(
                os.path.dirname(__file__), '..', '..', 'Matlab',
                'shp_toolbox-main', 'shp_toolbox-main', 'level1', 'flip_template.shp3'
            )
        
        if os.path.exists(test_file):
            s = shp_surface()
            s.import_shp3(test_file)
            # Verify coefficients are loaded
            self.assertIsNotNone(s.xc)
            self.assertIsNotNone(s.yc)
            self.assertIsNotNone(s.zc)
            self.assertEqual(s.L_max, 12)
            s.plot_H()
            # Verify geometric properties are computed after import
            self.assertIsNotNone(s.H, "Mean curvature H should be computed after import")
            self.assertIsNotNone(s.KG, "Gaussian curvature KG should be computed after import")
            self.assertIsNotNone(s.x, "Surface coordinates x should be computed")
            self.assertIsNotNone(s.y, "Surface coordinates y should be computed")
            self.assertIsNotNone(s.z, "Surface coordinates z should be computed")
            self.assertFalse(s.needs_updating, "Surface should be marked as updated")
        else:
            self.skipTest(f"Test .shp3 file not found: {test_file}")
    
    def test_get_xyz_clks(self):
        """Test extracting xyz coefficients"""
        s = shp_surface(6)
        xc, yc, zc = s.get_xyz_clks(s.X_o)
        self.assertEqual(len(xc), (6 + 1)**2)
        self.assertEqual(len(yc), (6 + 1)**2)
        self.assertEqual(len(zc), (6 + 1)**2)
    
    def test_update(self):
        """Test surface update"""
        s = shp_surface(6)
        s.update()
        self.assertIsNotNone(s.x)
        self.assertIsNotNone(s.y)
        self.assertIsNotNone(s.z)
        self.assertFalse(s.needs_updating)
    
    def test_plot(self):
        """Test plot function"""
        s = shp_surface(6)
        s.update()
        # Test that plot function exists
        self.assertTrue(hasattr(s, 'plot'))
        self.assertTrue(callable(s.plot))
    
    def test_plot_H(self):
        """Test plot_H function"""
        s = shp_surface(6)
        s.update_full()
        # Test that plot_H function exists
        self.assertTrue(hasattr(s, 'plot_H'))
        self.assertTrue(callable(s.plot_H))
        # Test that H is computed
        self.assertIsNotNone(s.H, "Mean curvature H should be computed after update_full()")
        self.assertIsNotNone(s.KG, "Gaussian curvature KG should be computed after update_full()")
        s.plot_H()
    
    def test_plot_fast(self):
        """Test plot_fast function"""
        s = shp_surface(6)
        s.update()
        # Test that plot_fast function exists
        self.assertTrue(hasattr(s, 'plot_fast'))
        self.assertTrue(callable(s.plot_fast))


if __name__ == '__main__':
    unittest.main()
