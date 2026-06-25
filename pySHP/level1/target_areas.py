"""
Per-face target spherical areas from mesh curvature.

Target area for face i is proportional to:
    face_area_3d[i] * mean(|H| at vertices of face i])^2
normalized to sum to 4*pi.
"""

import numpy as np


def compute_curvature_target_areas(mesh, smooth_passes=0, min_weight=1e-12,
                                   area_exponent=2.0, equal_area_blend=0.0):
    """Compute per-face target areas on the unit sphere from curvature.

    The curvature target gives high-curvature faces more sphere area
    (``weight = face_area * |H|**area_exponent``). Because that *compresses*
    low-curvature regions (e.g. a stalk) and forces highly sheared / thin
    triangles, the target can be **blended toward equal area** to relax shear at
    the cost of the curvature-area correspondence (goal (a) vs goal (c)).

    Parameters
    ----------
    mesh : surface_mesh
        Mesh with ``.F`` and mean curvature ``.H`` (``props()`` called if needed).
    smooth_passes : int
        Optional Laplacian smoothing passes on face weights before normalization.
    min_weight : float
        Floor on face weights to avoid zero targets.
    area_exponent : float
        Exponent on ``|H|`` (default 2.0 = classic). Smaller values flatten the
        curvature emphasis; ``0`` makes the curvature term uniform.
    equal_area_blend : float in [0, 1]
        Linear blend of the (normalised) curvature target with an equal-area
        target (``4*pi / n_faces`` per face). ``0`` = pure curvature (default),
        ``1`` = equal area. Use to relax shear in compressed regions.

    Returns
    -------
    Ao : ndarray, shape (n_faces,)
        Target signed spherical area per face, summing to 4*pi.
    face_weights : ndarray, shape (n_faces,)
        Unnormalized curvature weights before scaling to 4*pi.
    """
    if mesh.H is None or mesh.F_areas is None:
        mesh.props()

    F = np.asarray(mesh.F, dtype=int)
    H = np.asarray(mesh.H, dtype=float)
    F_areas = np.asarray(mesh.F_areas, dtype=float)
    n_faces = len(F)

    abs_H = np.abs(np.where(np.isfinite(H), H, 0.0))
    face_H = np.mean(abs_H[F], axis=1)
    face_weights = F_areas * face_H ** float(area_exponent)
    face_weights = np.maximum(face_weights, min_weight)

    if smooth_passes > 0 and hasattr(mesh, 'L') and mesh.L:
        face_weights = _smooth_face_field(mesh, face_weights, smooth_passes)

    total = np.sum(face_weights)
    if total <= 0:
        Ao = np.full(n_faces, 4.0 * np.pi / max(n_faces, 1))
        return Ao, face_weights

    Ao_curv = face_weights / total * (4.0 * np.pi)

    b = float(np.clip(equal_area_blend, 0.0, 1.0))
    if b > 0.0:
        Ao_equal = np.full(n_faces, 4.0 * np.pi / max(n_faces, 1))
        Ao = (1.0 - b) * Ao_curv + b * Ao_equal
        Ao = Ao / np.sum(Ao) * (4.0 * np.pi)
    else:
        Ao = Ao_curv

    return Ao, face_weights


def _smooth_face_field(mesh, face_values, n_passes):
    """Laplacian smooth a per-face scalar field via vertex adjacency."""
    F = np.asarray(mesh.F, dtype=int)
    nv = len(mesh.X)
    n_faces = len(F)

    vert_val = np.zeros(nv)
    vert_count = np.zeros(nv)
    for fi in range(n_faces):
        for vi in F[fi]:
            vert_val[vi] += face_values[fi]
            vert_count[vi] += 1
    vert_count = np.maximum(vert_count, 1)
    vert_val /= vert_count

    for _ in range(n_passes):
        vert_new = vert_val.copy()
        for vi in range(nv):
            nbrs = mesh.L.get(vi, [])
            if len(nbrs) > 0:
                nbr_list = list(nbrs)
                vert_new[vi] = 0.5 * vert_val[vi] + 0.5 * np.mean(vert_val[nbr_list])
        vert_val = vert_new

    smoothed = np.zeros(n_faces)
    for fi in range(n_faces):
        smoothed[fi] = np.mean(vert_val[F[fi]])
    return smoothed
