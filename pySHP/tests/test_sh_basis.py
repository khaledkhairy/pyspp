"""
Tests for sh_basis class
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pySHP.sh_basis import sh_basis


class TestSHBasis(unittest.TestCase):
    
    def test_init(self):
        """Test basic initialization"""
        b = sh_basis(6, 30)
        self.assertEqual(b.L_max, 6)
        self.assertEqual(b.gdim, 30)
        self.assertIsNotNone(b.Y)
        self.assertIsNotNone(b.w)
    
    def test_gaussquad(self):
        """Test Gaussian quadrature"""
        x, w = sh_basis.gaussquad(5, -1, 1)
        self.assertEqual(len(x), 5)
        self.assertEqual(len(w), 5)
        # Check weights sum to interval length
        self.assertAlmostEqual(np.sum(w), 2.0, places=5)
    
    def test_N_LK_bosh(self):
        """Test normalization factor"""
        nlk = sh_basis.N_LK_bosh(0, 0)
        self.assertGreater(nlk, 0)
        
        nlk = sh_basis.N_LK_bosh(1, 0)
        self.assertGreater(nlk, 0)
        
        nlk = sh_basis.N_LK_bosh(1, 1)
        self.assertGreater(nlk, 0)
    
    def test_ylk_bosh(self):
        """Test spherical harmonic computation"""
        phi = np.array([0, np.pi/2, np.pi])
        theta = np.array([0, np.pi/4, np.pi/2])
        
        Y = sh_basis.ylk_bosh(0, 0, phi, theta)
        self.assertEqual(Y.shape, phi.shape)
        
        Y = sh_basis.ylk_bosh(1, 0, phi, theta)
        self.assertEqual(Y.shape, phi.shape)


if __name__ == '__main__':
    unittest.main()
