"""
Tests for surface_mesh class
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pySHP.surface_mesh import surface_mesh
from pySHP.shp_surface import shp_surface
from pySHP.utils import readoff



"""Test spherical parameterization"""
# read off file
fn_shape = os.path.join('C:\\Users\\Khaled Khairy\\Dropbox\\Projects\\hot\\Project_spherical_parameterization\\code', 'Matlab','shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off', 'misc_shapes', 'bunny.off'
)
X, F = readoff(fn_shape)

m = surface_mesh(X, F)
m.info()  # Check initial topology

#m.subdivide(1)  # 4x more faces
#m.remesh(method='isotropic', n_iterations=2, smooth_iterations=2)
#m.info()  # Verify topology preserved
m.check_mesh_integrity()  # Check for problems
m.print_mesh_quality()  # See triangle quality
m.plot()