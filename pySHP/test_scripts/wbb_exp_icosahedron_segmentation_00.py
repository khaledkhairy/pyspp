"""
Test script for icosahedron-based segmentation
Translated from MATLAB wbb_exp_icosahedron_segmentation_00.m
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
            # Add face labels if available
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
    print("=== Icosahedron Segmentation Test ===")
    tic = time.time()
    
    # Read mesh file
    dir_top = os.path.join(os.path.dirname(__file__), '..', '..', 'Matlab', 
                           'shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off')
    
    fn = os.path.join(dir_top, 'misc_shapes', 'bunny.off')
    
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
    dim = 100
    nseeds = 12
    
    print("Remeshing...")
    m.meshresample_keepratio = 0.5
    m = m.optimize_mesh()
    
    print(f"Optimized mesh: {len(m.X)} vertices, {len(m.F)} faces")
    
    # Mesh segmentation (simplified - full implementation would use mesh_segmentation_rw)
    print(f"\nPerforming mesh segmentation with {nseeds} seeds...")
    # For now, create simple labels based on vertex positions
    # Full implementation would use random walk segmentation
    m.edge_info()
    m.props()
    
    # Simple segmentation based on vertex clustering
    from sklearn.cluster import KMeans
    try:
        kmeans = KMeans(n_clusters=nseeds, random_state=0, n_init=10)
        face_centers = np.array([m.X[m.F[i]].mean(axis=0) for i in range(len(m.F))])
        labels = kmeans.fit_predict(face_centers) + 1
        m.face_labels = labels
        print(f"Segmentation completed: {len(np.unique(labels))} patches")
    except ImportError:
        print("sklearn not available, skipping segmentation")
        m.face_labels = np.ones(len(m.F), dtype=int)
    
    # Plot segmented mesh
    plot_mesh_pyvista(m, f"Segmented Mesh ({nseeds} patches)")
    
    # Spherical Harmonics projection
    L_max = 8
    gdim = 120
    
    print(f"\nComputing spherical harmonics with L_max={L_max}, gdim={gdim}")
    
    b = sh_basis(L_max, gdim)
    s = shp_surface(L_max, b)
    
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
