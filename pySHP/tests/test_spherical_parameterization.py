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
fn_cube = os.path.join('C:\\Users\\Khaled Khairy\\Dropbox\\Projects\\hot\\Project_spherical_parameterization\\code', 'Matlab','shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off', 'basic_shapes', 'cube.off'
)
X, F = readoff(fn_cube)
m = surface_mesh(X, F)
# Ensure Newton steps are enabled for optimization
m.optimization_method = 1
m.newton_niter = 30
m.newton_step = 0.2
# Enable verbose output to see what's happening
# bijective_plot_flag: 0=none, 1=final plot, 2=iterations, 3=iterations+histogram
m.bijective_plot_flag = 0  # Set to 1 for final plot only, or 2+ for iteration plots
m.map2sphere()
m.plot()
print("Spherical parameterization completed successfully!")
s = shp_surface(m)
s.plot(nico=3)

# Display Newton steps residual progression
if hasattr(m, 'newton_residuals') and m.newton_residuals is not None:
    print(f"\nNewton Steps Residual Progression:")
    print(f"  Initial residual: {m.newton_residuals[0]:.6e}")
    print(f"  Final residual: {m.newton_residuals[-1]:.6e}")
    print(f"  Reduction factor: {m.newton_residuals[0]/m.newton_residuals[-1]:.2f}x")
    



