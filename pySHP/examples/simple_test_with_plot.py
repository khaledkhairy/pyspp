"""
Simple test script without plotting
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pySHP import surface_mesh, sh_basis, shp_surface
from pySHP.utils import readoff

print("=" * 60)
print("Simple pySHP Test")
print("=" * 60)

# Test 1: Create sh_basis
print("\nTest 1: Creating sh_basis...")
b = sh_basis(4, 30)
print(f"  [OK] Created sh_basis: L_max={b.L_max}, gdim={b.gdim}")

# Test 2: Load mesh
print("\nTest 2: Loading .off file...")
#test_file = os.path.join('Matlab', 'shp_toolbox-main', 'shp_toolbox-main', 
#                         'test_data', 'off', 'scientific', '1dmp_pocket.off')
test_file = os.path.join('Matlab', 'shp_toolbox-main', 'shp_toolbox-main', 
                         'test_data', 'off', 'scientific', 'echinocyte.off')
#test_file = os.path.join('Matlab', 'shp_toolbox-main', 'shp_toolbox-main', 
#                         'test_data', 'off', 'basic_shapes', 'cube.off')
if os.path.exists(test_file):
    X, F = readoff(test_file)
    print(f"  [OK] Loaded mesh: {len(X)} vertices, {len(F)} faces")
    
    # Test 3: Create surface_mesh
    print("\nTest 3: Creating surface_mesh...")
    m = surface_mesh(X, F)
    m.props()
    print(f"  [OK] Created surface_mesh: Area={m.A:.2f}, Volume={m.V:.2f}")
    
    # Test 4: Plot mesh
    print("\nTest 4: Plotting mesh...")
    m.plot()
    print(f"  [OK] Plotted mesh")
    # Test 5: Create shp_surface
    print("\nTest 4: Creating shp_surface...")
    s = shp_surface(4, b, m)
    m, X, F, Y_LK, t, p = s.get_mesh()
    m.plot()
    print(f"  [OK] Created shp_surface: L_max={s.L_max}, coefficients={len(s.xc)}")
else:
    print(f"  [ERROR] Test file not found: {test_file}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
