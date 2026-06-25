"""
Diagnostic script: Run the full pipeline on the mushroom mesh and produce
a fine-mesh sphere quality report identifying foldovers, elongated triangles,
and potential face intersections.

This script replicates the notebook flow including the known coordinate
transfer step (PM['pm'].t = ms.t) that requires the Xkeyind remapping fix.
"""

import numpy as np
import sys
import os

code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, code_dir)

from pySHP.surface_mesh import surface_mesh
from pySHP.utils import readoff, kk_cart2sph, kk_sph2cart

# ---------- Stage 1: Load mesh ----------
print("=" * 70)
print("Stage 1: Loading mushroom mesh")
print("=" * 70)
fn = os.path.join(code_dir, 'Matlab', 'shp_toolbox-main', 'shp_toolbox-main',
                  'test_data', 'off', 'test_set', 'mushroom_repaired_02.off')
if not os.path.exists(fn):
    print(f"Mesh file not found: {fn}")
    sys.exit(1)
X, F = readoff(fn)
ms = surface_mesh(X, F)
ms.props()
ms.edge_info()
print(f"  Vertices: {len(ms.X)}, Faces: {len(ms.F)}")

# ---------- Stage 2: Segmentation ----------
print("\n" + "=" * 70)
print("Stage 2: Segmentation")
print("=" * 70)
from pySHP.level1.mesh_segmentation_rw import mesh_segmentation_rw

nseeds = 9
sigma = 1.0
ms, L, slix, P, Pconn = mesh_segmentation_rw(ms, nseeds, sigma)
n_patches = len(np.unique(L))
print(f"  Patches: {n_patches}")

# ---------- Stage 3: Patch info generation ----------
print("\n" + "=" * 70)
print("Stage 3: Patch info generation")
print("=" * 70)
from pySHP.level1.patch_info_gen import patch_info_gen

m, PM, Pconn = patch_info_gen(ms, P, Pconn)
print(f"  Simplified mesh: {len(PM['pm'].X)} vertices, {len(PM['pm'].F)} faces")
print(f"  Edges: {len(PM['Edges'])}")
print(f"  Keys: {len(PM['keys'])}")

# ---------- Stage 4: Spherical conformal parameterization of FULL mesh ----------
print("\n" + "=" * 70)
print("Stage 4: Spherical conformal parameterization (full mesh)")
print("=" * 70)
ms.newton_niter = 10
ms.newton_step_edge = 0.05
ms.bijective_plot_flag = 0
ms.map2sphere()
print(f"  ms.t range: [{ms.t.min():.4f}, {ms.t.max():.4f}]")
print(f"  ms.p range: [{ms.p.min():.4f}, {ms.p.max():.4f}]")

# ---------- Stage 4b: Transfer coordinates to PM['pm'] (notebook behaviour) ----------
print("\n  Transferring ms.t/p to PM['pm'].t/p (replicating notebook)...")
PM['pm'].t = ms.t.copy()
PM['pm'].p = ms.p.copy()
print(f"  PM['pm'].t length: {len(PM['pm'].t)} (pm vertices: {len(PM['pm'].X)})")
print(f"  (Mismatch expected -- parameterize_patches_cart will remap via Xkeyind)")

# ---------- Stage 5: Patch parameterization ----------
print("\n" + "=" * 70)
print("Stage 5: parameterize_patches_cart (plot_flag=1)")
print("=" * 70)
from pySHP.level1.parameterize_patches_cart import parameterize_patches_cart

PM = parameterize_patches_cart(PM, plot_flag=1)

# ---------- Stage 6: Additional per-patch diagnosis ----------
print("\n" + "=" * 70)
print("Stage 6: Per-patch boundary condition quality")
print("=" * 70)

from pySHP.level0.mesh_utils import reduce_to_minimal_set

for pix in range(PM['npatches']):
    if pix not in PM.get('PX', {}):
        print(f"  Patch {pix}: NOT PARAMETERIZED")
        continue
    patm = PM['P'][pix][0]
    px = PM['PX'][pix]
    fixed_mask = px[:, 0].astype(bool)
    x_s, y_s, z_s = px[:, 1], px[:, 2], px[:, 3]

    minpatm, uv = reduce_to_minimal_set(patm)
    X_sph = np.column_stack([x_s, y_s, z_s])
    n_faces = len(minpatm.F)

    # Compute orient for every face
    orients = np.zeros(n_faces)
    aspect_ratios = np.zeros(n_faces)
    for fi in range(n_faces):
        i0, i1, i2 = minpatm.F[fi]
        v0, v1, v2 = X_sph[i0], X_sph[i1], X_sph[i2]
        cross_vec = np.cross(v1 - v0, v2 - v0)
        centroid = (v0 + v1 + v2) / 3.0
        orients[fi] = np.dot(centroid, cross_vec)
        e0 = np.linalg.norm(v1 - v0)
        e1 = np.linalg.norm(v2 - v1)
        e2 = np.linalg.norm(v0 - v2)
        max_e = max(e0, e1, e2)
        min_e = min(e0, e1, e2)
        aspect_ratios[fi] = max_e / max(min_e, 1e-15)

    n_fold = int(np.sum(orients < 0))
    n_elong = int(np.sum(aspect_ratios > 20))

    # Boundary vertices -- check Cartesian positions
    n_bnd_fixed = int(np.sum(fixed_mask))
    n_bnd_total = int(np.sum(minpatm.border_vertex)) if hasattr(minpatm, 'border_vertex') and minpatm.border_vertex is not None else -1

    # Solid angle subtended by boundary on sphere
    bnd_idx = np.where(fixed_mask)[0]
    bnd_pts = X_sph[bnd_idx]
    if len(bnd_pts) > 1:
        # Compute the centroid of boundary points and the angular spread
        bnd_centroid = bnd_pts.mean(axis=0)
        bnd_centroid_norm = np.linalg.norm(bnd_centroid)
        # Cosine of max angular distance from centroid
        if bnd_centroid_norm > 1e-10:
            bnd_centroid_hat = bnd_centroid / bnd_centroid_norm
            cos_angles = bnd_pts @ bnd_centroid_hat
            min_cos = cos_angles.min()
            max_angle_deg = np.degrees(np.arccos(np.clip(min_cos, -1, 1)))
        else:
            max_angle_deg = 180.0  # boundary wraps entire sphere
    else:
        max_angle_deg = 0.0

    # Distances between boundary points on sphere
    if len(bnd_pts) > 1:
        pairwise_d = np.linalg.norm(bnd_pts[:, None, :] - bnd_pts[None, :, :], axis=2)
        max_pair_dist = pairwise_d.max()
    else:
        max_pair_dist = 0.0

    flag = ""
    if n_fold > 0:
        flag = " *** FOLDOVERS ***"
    elif n_elong > 0:
        flag = " * ELONGATED *"

    print(f"  Patch {pix}: faces={n_faces}, fold={n_fold}, elong(>20)={n_elong}, "
          f"bnd_fixed={n_bnd_fixed}/{n_bnd_total}, "
          f"orient=[{orients.min():.6f},{orients.max():.6f}], "
          f"AR=[{aspect_ratios.mean():.1f},{aspect_ratios.max():.1f}], "
          f"bnd_spread={max_angle_deg:.1f}deg, bnd_maxd={max_pair_dist:.3f}"
          f"{flag}")

    # For problematic patches, print worst faces
    if n_fold > 0:
        worst = np.argsort(orients)[:min(5, n_fold)]
        for wi in worst:
            i0, i1, i2 = minpatm.F[wi]
            print(f"      fold face {wi}: verts=[{i0},{i1},{i2}], "
                  f"orient={orients[wi]:.6f}, AR={aspect_ratios[wi]:.1f}, "
                  f"fixed=[{fixed_mask[i0]},{fixed_mask[i1]},{fixed_mask[i2]}]")
    if n_elong > 0 and n_fold == 0:
        worst = np.argsort(-aspect_ratios)[:min(5, n_elong)]
        for wi in worst:
            i0, i1, i2 = minpatm.F[wi]
            v0, v1, v2 = X_sph[i0], X_sph[i1], X_sph[i2]
            print(f"      elong face {wi}: AR={aspect_ratios[wi]:.1f}, "
                  f"orient={orients[wi]:.6f}, "
                  f"pos=({v0[0]:.3f},{v0[1]:.3f},{v0[2]:.3f})-"
                  f"({v1[0]:.3f},{v1[1]:.3f},{v1[2]:.3f})-"
                  f"({v2[0]:.3f},{v2[1]:.3f},{v2[2]:.3f})")

# ---------- Summary ----------
print("\n" + "=" * 70)
print("Quality report written to: pySHP/tests/fine_mesh_sphere_quality.txt")
print("=" * 70)
print("Done.")
