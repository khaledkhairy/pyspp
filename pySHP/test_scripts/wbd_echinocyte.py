"""
Test script for echinocyte mesh parameterization
Translated from MATLAB wbd_echinocyte.m
"""

import numpy as np
import sys
import os

# Add parent directory to path
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
        mesh_pv = pv.PolyData(mesh.X, mesh.F)
        plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True)
        plotter.add_text(title, font_size=12)
        plotter.show()
    except ImportError:
        print("PyVista not available, using matplotlib fallback")
        plot_mesh_matplotlib(mesh, title)

def plot_mesh_matplotlib(mesh, title="Mesh"):
    """Plot mesh using matplotlib"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create poly collection
    verts = mesh.X[mesh.F]
    collection = Poly3DCollection(verts, alpha=0.7, facecolor='lightblue', edgecolor='black')
    ax.add_collection3d(collection)
    
    # Set limits
    ax.set_xlim(mesh.X[:, 0].min(), mesh.X[:, 0].max())
    ax.set_ylim(mesh.X[:, 1].min(), mesh.X[:, 1].max())
    ax.set_zlim(mesh.X[:, 2].min(), mesh.X[:, 2].max())
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.show()

def main():
    print("=== Echinocyte Mesh Parameterization Test ===")
    tic = time.time()
    
    # Read mesh file
    # Note: Update this path to point to your .off file
    dir_top = os.path.join(os.path.dirname(__file__), '..', '..', 'Matlab', 
                           'shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off')
    
    fn = os.path.join(dir_top, 'scientific', 'echinocyte.off')
    
    if not os.path.exists(fn):
        print(f"Warning: Mesh file not found at {fn}")
        print("Creating a test sphere mesh instead...")
        # Create a test sphere
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
    # Simplified remeshing - would use trimesh in full implementation
    m.meshresample_keepratio = 0.5
    m = m.optimize_mesh()
    
    print(f"Optimized mesh: {len(m.X)} vertices, {len(m.F)} faces")
    
    # Plot original mesh
    plot_mesh_pyvista(m, "Original Mesh")
    
    # Spherical Harmonics projection
    L_max = 24
    gdim = 320
    
    print(f"\nComputing spherical harmonics with L_max={L_max}, gdim={gdim}")
    
    b = sh_basis(L_max, gdim)
    print("Basis computed")
    
    # For now, create a simple shp_surface from the mesh
    # Full implementation would do proper parameterization
    s = shp_surface(L_max, b)
    print("SHP surface created")
    
    # Try to fit mesh to SH (simplified)
    try:
        # Map to sphere coordinates
        from pySHP.utils import kk_cart2sph
        t, p, r = kk_cart2sph(m.X[:, 0], m.X[:, 1], m.X[:, 2])
        m.t = t
        m.p = p
        
        # Perform SH analysis
        s.mesh2shp(m, L_max)
        print("Mesh fitted to spherical harmonics")
        
        # Get mesh representation
        mshp, X_mesh, C_mesh, Y_LK, t_mesh, p_mesh = sh_surface.get_mesh(
            sh_surface(s.xc), 5
        )
        print(f"SH mesh: {len(X_mesh)} vertices, {len(C_mesh)} faces")
        
        # Plot SH representation
        plot_mesh_pyvista(mshp, f"Spherical Harmonics (L={L_max})")
        
    except Exception as e:
        print(f"Error in SH fitting: {e}")
        print("Plotting basis sphere instead...")
        s.update()
        plot_mesh_pyvista(s, "SH Basis Sphere")
    
    toc = time.time()
    print(f"\nTotal time: {toc - tic:.2f} seconds")

if __name__ == "__main__":
    main()
