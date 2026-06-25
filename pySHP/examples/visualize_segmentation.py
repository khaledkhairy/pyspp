"""
Example script demonstrating visualization of mesh segmentation

This script shows how to:
1. Generate/load a mesh
2. Segment the mesh into patches
3. Visualize the segmented mesh with different colors for each patch
4. Visualize key vertices and patch structure

Author: Khaled Khairy
"""

import numpy as np
import sys
import os

# Add parent directories to path for proper imports
code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, code_dir)

from pySHP.surface_mesh import surface_mesh
from pySHP.level1.mesh_segmentation_rw import mesh_segmentation_rw
from pySHP.level1.patch_info_gen import patch_info_gen


def visualize_segmentation_example():
    """
    Demonstrates mesh segmentation and visualization.
    """
    print("="*60)
    print("Mesh Segmentation Visualization Example")
    print("="*60)
    
    # =========================================================================
    # Step 1: Generate a sphere mesh (or load your own mesh)
    # =========================================================================
    print("\n[1] Generating sphere mesh...")
    n_subdivisions = 2  # Higher = more vertices
    X, F = surface_mesh.sphere_mesh_gen(n_subdivisions)
    m = surface_mesh(X, F)
    m.props()
    m.edge_info()
    print(f"    Mesh created: {len(m.X)} vertices, {len(m.F)} faces")
    
    # =========================================================================
    # Step 2: Segment the mesh
    # =========================================================================
    print("\n[2] Segmenting mesh...")
    nseeds = 8  # Number of patches
    sigma = 1.0  # Segmentation parameter
    
    ms, L, slix, P, Pconn = mesh_segmentation_rw(m, nseeds, sigma)
    
    n_patches = len(np.unique(L))
    print(f"    Created {n_patches} patches")
    print(f"    Label distribution: {dict(zip(*np.unique(L, return_counts=True)))}")
    
    # =========================================================================
    # Step 3: Generate patch info (for advanced visualization)
    # =========================================================================
    print("\n[3] Generating patch info structure...")
    m_with_patches, PM, Pconn = patch_info_gen(ms, P, Pconn)
    print(f"    Patch mesh has {PM['npatches']} patches")
    print(f"    Number of edges between patches: {len(PM['Edges'])}")
    
    # =========================================================================
    # Step 4: Visualize
    # =========================================================================
    print("\n[4] Visualizing...")
    
    # Method 1: Simple patch visualization using plot_labels()
    # This colors each face by its patch label
    print("\n    Opening visualization window...")
    print("    (Close the window to continue)")
    
    m_with_patches.plot_labels(flag=1)  # flag=1 shows edges, flag=0 hides edges
    
    # Method 2: Show patches with key vertices and center vertices
    # This shows additional structural information
    print("\n    Showing patch structure with key vertices...")
    m_with_patches.plot_patches(PM, pflag=1)
    
    # Method 3: Show border vertices (useful for open meshes/patches)
    print("\n    Showing border information...")
    m_with_patches.plot_border()
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


def visualize_with_pyvista_example():
    """
    Higher quality visualization using PyVista (if available).
    PyVista provides interactive 3D rendering.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista")
        print("Falling back to matplotlib visualization.")
        visualize_segmentation_example()
        return
    
    print("="*60)
    print("Mesh Segmentation Visualization (PyVista)")
    print("="*60)
    
    # Generate and segment mesh
    print("\n[1] Generating and segmenting mesh...")
    X, F = surface_mesh.sphere_mesh_gen(2)
    m = surface_mesh(X, F)
    m.props()
    m.edge_info()
    
    nseeds = 8
    ms, L, slix, P, Pconn = mesh_segmentation_rw(m, nseeds, sigma=1.0)
    m_seg, PM, Pconn = patch_info_gen(ms, P, Pconn)
    
    print(f"    Created mesh with {len(np.unique(L))} patches")
    
    # Create PyVista mesh
    print("\n[2] Creating PyVista visualization...")
    faces_pv = np.hstack([[3] + list(face) for face in m_seg.F])
    mesh_pv = pv.PolyData(m_seg.X, faces_pv)
    
    # Add face labels as cell data for coloring
    mesh_pv.cell_data['patch_label'] = m_seg.face_labels
    
    # Plot
    print("\n[3] Rendering...")
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_pv, 
                     scalars='patch_label',
                     cmap='tab10',  # Categorical colormap
                     show_edges=True,
                     edge_color='black',
                     line_width=0.5)
    plotter.add_title("Mesh Segmentation (colored by patch)")
    plotter.background_color = 'white'
    plotter.show()
    
    print("\nVisualization complete!")


def quick_plot_patches(m, face_labels=None):
    """
    Quick function to plot a mesh with patch colors.
    
    Parameters:
    -----------
    m : surface_mesh
        The mesh to plot
    face_labels : array, optional
        Face labels (patch indices). If None, uses m.face_labels
    """
    if face_labels is not None:
        m.face_labels = face_labels
    
    if m.face_labels is None:
        print("No face labels available. Running plot() instead.")
        m.plot()
    else:
        m.plot_labels(flag=1)


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("\nChoose visualization method:")
    print("  1 - Matplotlib (basic, built-in)")
    print("  2 - PyVista (advanced, interactive)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '2':
        visualize_with_pyvista_example()
    else:
        visualize_segmentation_example()
