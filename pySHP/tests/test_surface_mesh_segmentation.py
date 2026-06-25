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
from pySHP.level1.mesh_segmentation_rw import mesh_segmentation_rw
from pySHP.level1.patch_info_gen import patch_info_gen


"""Test spherical parameterization"""
# read off file
topic = 'misc_shapes' #'scientific' #'misc_shapes'
file_name = 'mushroom.off' #'planula_01.off' #'mushroom.off' #'test_mud_01.off' #'MaxPlankHead.off'
fn_shape = os.path.join('C:\\Users\\Khaled Khairy\\Dropbox\\Projects\\hot\\Project_spherical_parameterization\\code', 
'Matlab','shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off', 
topic,
file_name 
)
X, F = readoff(fn_shape)
m = surface_mesh(X, F)
m.repair_mesh()
m.info()  # Check initial topology
m.subdivide(1)  # 4x more faces
#m.remesh(method='isotropic', n_iterations=2, smooth_iterations=2)
#m.info()  # Verify topology preserved
m.check_mesh_integrity()  # Check for problems
m.print_mesh_quality()  # See triangle quality
#m.plot()

# Generate mesh segmentation
nseeds = 3  # Use 6 seeds for testing
sigma = 1.0
curvature_weight = -1e-6  # 0.0 = no curvature, 1.0 = full curvature weighting
print("\n" + "="*60)
print(f"Calling mesh_segmentation_rw with verbose=True, plot_intermediate=True")
print(f"  sigma={sigma}, curvature_weight={curvature_weight}")
print("="*60)
ms, L, slix, P, Pconn = mesh_segmentation_rw(m, nseeds, sigma, curvature_weight=curvature_weight, verbose=True, plot_intermediate=True)

# report the results of the mesh segmentation
print("\n" + "="*60)
print("Verification: ms should be the SAME mesh as m, with face_labels added")
print("="*60)
print(f"m has {len(m.X)} vertices, {len(m.F)} faces")
print(f"ms has {len(ms.X)} vertices, {len(ms.F)} faces")
print(f"Are they the same object? {m is ms}")
print(f"Does ms have face_labels? {hasattr(ms, 'face_labels') and ms.face_labels is not None}")
if hasattr(ms, 'face_labels') and ms.face_labels is not None:
    print(f"Number of unique labels: {len(np.unique(ms.face_labels))}")
    print(f"Label range: [{ms.face_labels.min()}, {ms.face_labels.max()}]")

# Plot the segmented mesh to confirm it's the same size
print("\nPlotting segmented mesh (should be same size as input)...")
ms.plot_labels()

# generate patches and patch mesh
m_seg, PM, Pconn = patch_info_gen(ms, P, Pconn)

# report the results of the patch generation
print("\n" + "="*60)
print(f"\nSegmentation complete:")
print(f"  Number of patches: {len(np.unique(L))}")
print(f"  Mesh has {len(m_seg.F)} faces")
if 'pm' in PM:
    print(f"  Simplified mesh has {len(PM['pm'].F)} faces")

# [1] Plot the simplified mesh
print("\n[1] Plotting simplified mesh...")
surface_mesh.plot_simplified_mesh(PM, show_keys=True, show_cv=True)

'''
# [2] Visualize patches with colors on full mesh
print("\n[2] Plotting full mesh with patch labels...")
m_seg.plot_labels()  # Each patch gets a different color!

# [3] Visualize with key vertices and structure
print("\n[3] Plotting full mesh with patches and key vertices...")
m_seg.plot_patches(PM)

# [4] Show border vertices on the full mesh
print("\n[4] Plotting full mesh with border vertices...")
m_seg.plot_border()
'''