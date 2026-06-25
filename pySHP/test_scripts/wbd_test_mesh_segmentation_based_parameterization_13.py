"""
Test script for mesh segmentation-based parameterization
Translated from MATLAB wbd_test_mesh_segmentation_based_parameterization_13.m
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pySHP.surface_mesh import surface_mesh
from pySHP.sh_basis import sh_basis
from pySHP.shp_surface import shp_surface
from pySHP.level1.mesh_segmentation_rw import mesh_segmentation_rw
from pySHP.level1.patch_info_gen import patch_info_gen
from pySHP.level2.spherical_parameterization import spherical_parameterization
from pySHP.utils import readoff
import time

def plot_mesh_pyvista(mesh, title="Mesh"):
    """Plot mesh using pyvista"""
    try:
        import pyvista as pv
        
        plotter = pv.Plotter()
        if hasattr(mesh, 'X') and hasattr(mesh, 'F'):
            mesh_pv = pv.PolyData(mesh.X, mesh.F)
            if hasattr(mesh, 'face_labels') and mesh.face_labels is not None:
                plotter.add_mesh(mesh_pv, scalars=mesh.face_labels, 
                               show_edges=True, cmap='tab20')
            else:
                plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True)
        else:
            mesh_pv = pv.PolyData(mesh)
            plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True)
        plotter.add_text(title, font_size=12)
        plotter.show()
    except ImportError:
        print("PyVista not available")

def main():
    print("=== Mesh Segmentation-Based Parameterization Test ===")
    tic = time.time()
    
    # Read mesh file
    dir_top = os.path.join(os.path.dirname(__file__), '..', '..', 'Matlab', 
                           'shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off')
    
    # Try different test files
    test_files = [
        os.path.join(dir_top, 'scientific', 'echinocyte.off'),
        os.path.join(dir_top, 'misc_shapes', 'bunny.off'),
        os.path.join(dir_top, 'misc_shapes', 'hand.off')
    ]
    
    fn = None
    for test_file in test_files:
        if os.path.exists(test_file):
            fn = test_file
            break
    
    if fn is None:
        print("Warning: No test mesh files found")
        print("Creating a test sphere mesh instead...")
        X, F = surface_mesh.sphere_mesh_gen(3)
        m = surface_mesh(X, F)
    else:
        print(f"Reading mesh from {fn}")
        X, F = readoff(fn)
        m = surface_mesh(X, F)
    
    print(f"Mesh loaded: {len(m.X)} vertices, {len(m.F)} faces")
    
    # Optimize mesh
    dim = 150
    nseeds = 20
    
    print("Remeshing...")
    m.meshresample_keepratio = 0.5
    m = m.optimize_mesh()
    
    print(f"Optimized mesh: {len(m.X)} vertices, {len(m.F)} faces")
    
    # Mesh segmentation
    print(f"\nPerforming mesh segmentation with {nseeds} seeds...")
    try:
        m, L, slix, P, Pconn = mesh_segmentation_rw(m, nseeds)
        print(f"Segmentation completed: {len(np.unique(L))} patches")
        
        # Generate patch info
        m, PM, Pconn = patch_info_gen(m, P, Pconn)
        print("Patch information generated")
        
        # Plot segmented mesh
        plot_mesh_pyvista(m, f"Segmented Mesh ({len(np.unique(L))} patches)")
        
    except Exception as e:
        print(f"Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: simple labels
        m.face_labels = np.ones(len(m.F), dtype=int)
        PM = {'pm': m, 'P': []}
    
    # Spherical parameterization
    print("\nPerforming spherical parameterization...")
    initial_b2sopts = {
        'plot_flag': 0,
        'lambdaA': 1.0,
        'lambda': 1e2,
        'lambda1': 1e-3,
        'lambda2': 1e-1,
        'maxiter': 1000,
        'equal_area': 0
    }
    
    try:
        mp, PM, failed_patches = spherical_parameterization(m, nseeds, initial_b2sopts, PM)
        if failed_patches:
            print(f"Failed patches: {failed_patches}")
        mp.needs_map2sphere = False
        print("Parameterization completed")
        
        # Plot parameterized mesh
        plot_mesh_pyvista(mp, "Parameterized Mesh")
        
    except Exception as e:
        print(f"Parameterization failed: {e}")
        import traceback
        traceback.print_exc()
        mp = m
    
    # Spherical Harmonics projection
    L_max = 8
    gdim = 120
    
    print(f"\nComputing spherical harmonics with L_max={L_max}, gdim={gdim}")
    
    try:
        b = sh_basis(L_max, gdim)
        s = shp_surface(L_max, b, mp)
        
        from pySHP.sh_surface import sh_surface
        sh_x = sh_surface(s.xc)
        mshp, X_mesh, C_mesh, Y_LK, t_mesh, p_mesh = sh_surface.get_mesh(sh_x, 6)
        print(f"SH mesh: {len(X_mesh)} vertices, {len(C_mesh)} faces")
        
        plot_mesh_pyvista(mshp, f"Spherical Harmonics (L={L_max})")
        
    except Exception as e:
        print(f"SH projection failed: {e}")
        import traceback
        traceback.print_exc()
    
    toc = time.time()
    print(f"\nTotal time: {toc - tic:.2f} seconds")

if __name__ == "__main__":
    main()
