"""
Test script for BDH6230 brain mesh
Translated from MATLAB wbb_BDH6230.m
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pySHP.surface_mesh import surface_mesh
from pySHP.sh_basis import sh_basis
from pySHP.shp_surface import shp_surface
from pySHP.utils import readoff
import time

def plot_mesh_pyvista(mesh, title="Mesh"):
    """Plot mesh using pyvista"""
    try:
        import pyvista as pv
        
        plotter = pv.Plotter()
        if hasattr(mesh, 'X') and hasattr(mesh, 'F'):
            mesh_pv = pv.PolyData(mesh.X, mesh.F)
        else:
            mesh_pv = pv.PolyData(mesh)
        plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True)
        plotter.add_text(title, font_size=12)
        plotter.show()
    except ImportError:
        print("PyVista not available")

def main():
    print("=== BDH6230 Brain Mesh Parameterization Test ===")
    tic = time.time()
    
    # Read mesh file
    dir_top = os.path.join(os.path.dirname(__file__), '..', '..', 'Matlab', 
                           'shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off')
    
    fn = os.path.join(dir_top, 'scientific', 'BDH6230_whole_brain.off')
    
    if not os.path.exists(fn):
        print(f"Warning: Mesh file not found at {fn}")
        print("Creating a test sphere mesh instead...")
        X, F = surface_mesh.sphere_mesh_gen(3)
        m = surface_mesh(X, F)
    else:
        print(f"Reading mesh from {fn}")
        X, F = readoff(fn)
        m = surface_mesh(X, F)
    
    print(f"Mesh loaded: {len(m.X)} vertices, {len(m.F)} faces")
    
    # Optimize mesh
    dim = 220
    nseeds = 30
    
    print("Remeshing...")
    m.meshresample_keepratio = 0.5
    m = m.optimize_mesh()
    
    print(f"Optimized mesh: {len(m.X)} vertices, {len(m.F)} faces")
    
    # Plot original mesh
    plot_mesh_pyvista(m, "Original Brain Mesh")
    
    # Spherical Harmonics projection
    L_max = 44
    gdim = 320
    
    print(f"\nComputing spherical harmonics with L_max={L_max}, gdim={gdim}")
    
    b = sh_basis(L_max, gdim)
    print("Basis computed")
    
    # Create shp_surface
    s = shp_surface(L_max, b)
    print("SHP surface created")
    
    # Try to fit mesh to SH
    try:
        from pySHP.utils import kk_cart2sph
        t, p, r = kk_cart2sph(m.X[:, 0], m.X[:, 1], m.X[:, 2])
        m.t = t
        m.p = p
        
        s.mesh2shp(m, L_max)
        print("Mesh fitted to spherical harmonics")
        
        from pySHP.sh_surface import sh_surface
        sh_x = sh_surface(s.xc)
        mshp, X_mesh, C_mesh, Y_LK, t_mesh, p_mesh = sh_surface.get_mesh(sh_x, 6)
        print(f"SH mesh: {len(X_mesh)} vertices, {len(C_mesh)} faces")
        
        plot_mesh_pyvista(mshp, f"Spherical Harmonics (L={L_max})")
        
    except Exception as e:
        print(f"Error in SH fitting: {e}")
        import traceback
        traceback.print_exc()
    
    toc = time.time()
    print(f"\nTotal time: {toc - tic:.2f} seconds")

if __name__ == "__main__":
    main()
