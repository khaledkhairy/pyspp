"""
Interpolate spherical parameterization from a decimated mesh onto the
full fine mesh using barycentric coordinates.

Given a decimated mesh M_dec with a valid spherical parameterization
(theta, phi per vertex), this module computes (theta, phi) for every
vertex of the original fine mesh M_orig by:

1. Copying (theta, phi) directly for vertices that survived decimation.
2. For removed vertices, finding the enclosing triangle in M_dec (in 3D)
   and computing barycentric interpolation of (theta, phi).

The interpolation operates in Cartesian coordinates on the unit sphere
(converting theta/phi -> xyz, interpolating, then converting back) to
avoid discontinuity artefacts at the phi=0/2pi wrap-around.
"""

import numpy as np


def _barycentric_coords(p, a, b, c):
    """Compute barycentric coordinates of point *p* w.r.t. triangle (a, b, c).

    Returns (u, v, w) such that p ~ u*a + v*b + w*c.
    If the point is inside the triangle, all three are in [0, 1].
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return -1.0, -1.0, -1.0
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def interpolate_fine_mesh_parameterization(
        X_fine, F_fine,
        X_dec, F_dec, t_dec, p_dec,
        vert_map_dec,
        verbose=True):
    """Interpolate (theta, phi) from decimated mesh to full fine mesh.

    Parameters
    ----------
    X_fine : (N_fine, 3) array
        Vertex positions of the original fine mesh.
    F_fine : (M_fine, 3) array
        Face connectivity of the original fine mesh.
    X_dec : (N_dec, 3) array
        Vertex positions of the decimated mesh.
    F_dec : (M_dec, 3) array
        Face connectivity of the decimated mesh.
    t_dec : (N_dec,) array
        Theta (co-latitude) values on the sphere for each decimated vertex.
    p_dec : (N_dec,) array
        Phi (longitude) values on the sphere for each decimated vertex.
    vert_map_dec : (N_dec,) array of int
        Mapping from decimated vertex index to original fine-mesh vertex index.
        ``vert_map_dec[i]`` is the fine-mesh index of decimated vertex ``i``.
    verbose : bool
        Print progress information.

    Returns
    -------
    t_full : (N_fine,) array
        Theta values for every fine-mesh vertex.
    p_full : (N_fine,) array
        Phi values for every fine-mesh vertex.
    report : dict
        Keys: 'n_survived', 'n_interpolated', 'n_failed', 'failed_indices'.
    """
    from scipy.spatial import cKDTree

    N_fine = len(X_fine)
    N_dec = len(X_dec)

    t_full = np.zeros(N_fine, dtype=float)
    p_full = np.zeros(N_fine, dtype=float)

    # Convert decimated mesh (theta, phi) to Cartesian on unit sphere for
    # safe interpolation (avoids phi wrap-around issues).
    sph_x = np.sin(t_dec) * np.cos(p_dec)
    sph_y = np.sin(t_dec) * np.sin(p_dec)
    sph_z = np.cos(t_dec)

    # Step 1: Direct copy for survived vertices
    survived_set = set()
    for dec_i, orig_i in enumerate(vert_map_dec):
        orig_i = int(orig_i)
        if 0 <= orig_i < N_fine:
            t_full[orig_i] = t_dec[dec_i]
            p_full[orig_i] = p_dec[dec_i]
            survived_set.add(orig_i)

    removed_verts = np.array(
        [i for i in range(N_fine) if i not in survived_set], dtype=int)

    if verbose:
        print(f'Survived vertices (direct copy): {len(survived_set)}')
        print(f'Vertices to interpolate: {len(removed_verts)}')

    if len(removed_verts) == 0:
        return t_full, p_full, {
            'n_survived': len(survived_set),
            'n_interpolated': 0,
            'n_failed': 0,
            'failed_indices': np.array([], dtype=int),
        }

    # Step 2: Build spatial acceleration for finding enclosing triangles.
    # Use face centroids + KD-tree for candidate faces, then check
    # barycentric coordinates for the actual enclosing face.
    F_dec = np.asarray(F_dec, dtype=int)
    centroids = (X_dec[F_dec[:, 0]] + X_dec[F_dec[:, 1]] +
                 X_dec[F_dec[:, 2]]) / 3.0

    # Compute max edge length per face for search radius
    face_radii = np.zeros(len(F_dec))
    for i in range(3):
        j = (i + 1) % 3
        edge_len = np.linalg.norm(
            X_dec[F_dec[:, i]] - X_dec[F_dec[:, j]], axis=1)
        face_radii = np.maximum(face_radii, edge_len)

    tree_centroids = cKDTree(centroids)
    max_search_radius = np.max(face_radii) * 2.0

    # Also build vertex-based face adjacency for fallback nearest-vertex search
    tree_dec_verts = cKDTree(X_dec)

    # Vertex -> faces adjacency
    vert_to_faces = [[] for _ in range(N_dec)]
    for fi in range(len(F_dec)):
        for vi in F_dec[fi]:
            vert_to_faces[int(vi)].append(fi)

    n_interpolated = 0
    n_failed = 0
    failed_indices = []

    bary_tol = -0.05  # allow slight extrapolation

    for idx_count, orig_i in enumerate(removed_verts):
        pt = X_fine[orig_i]

        # Strategy 1: search by face centroid proximity
        found = False
        candidate_faces = tree_centroids.query_ball_point(
            pt, max_search_radius)

        best_fi = -1
        best_bary = None
        best_dist = np.inf

        for fi in candidate_faces:
            a = X_dec[F_dec[fi, 0]]
            b = X_dec[F_dec[fi, 1]]
            c = X_dec[F_dec[fi, 2]]
            u, v, w = _barycentric_coords(pt, a, b, c)
            if u >= bary_tol and v >= bary_tol and w >= bary_tol:
                proj = u * a + v * b + w * c
                d = np.linalg.norm(pt - proj)
                if d < best_dist:
                    best_dist = d
                    best_fi = fi
                    best_bary = (u, v, w)
                    found = True

        # Strategy 2: nearest vertex fallback
        if not found:
            _, nearest_dec_vi = tree_dec_verts.query(pt)
            for fi in vert_to_faces[nearest_dec_vi]:
                a = X_dec[F_dec[fi, 0]]
                b = X_dec[F_dec[fi, 1]]
                c = X_dec[F_dec[fi, 2]]
                u, v, w = _barycentric_coords(pt, a, b, c)
                # Slightly more relaxed tolerance for nearest-vertex fallback
                if u >= -0.15 and v >= -0.15 and w >= -0.15:
                    proj = u * a + v * b + w * c
                    d = np.linalg.norm(pt - proj)
                    if d < best_dist:
                        best_dist = d
                        best_fi = fi
                        best_bary = (u, v, w)
                        found = True

        if found and best_fi >= 0:
            u, v, w = best_bary
            # Clamp to valid range and renormalize
            u = max(u, 0.0)
            v = max(v, 0.0)
            w = max(w, 0.0)
            s = u + v + w
            if s > 0:
                u /= s
                v /= s
                w /= s

            i0, i1, i2 = F_dec[best_fi]

            # Interpolate in Cartesian on unit sphere
            sx = u * sph_x[i0] + v * sph_x[i1] + w * sph_x[i2]
            sy = u * sph_y[i0] + v * sph_y[i1] + w * sph_y[i2]
            sz = u * sph_z[i0] + v * sph_z[i1] + w * sph_z[i2]

            # Renormalize to unit sphere
            r = np.sqrt(sx**2 + sy**2 + sz**2)
            if r > 1e-15:
                sx /= r
                sy /= r
                sz /= r

            # Convert back to (theta, phi)
            theta = np.arccos(np.clip(sz, -1.0, 1.0))
            phi = np.arctan2(sy, sx) % (2 * np.pi)

            t_full[orig_i] = theta
            p_full[orig_i] = phi
            n_interpolated += 1
        else:
            # Last resort: use nearest decimated vertex's (theta, phi)
            _, nearest_dec_vi = tree_dec_verts.query(pt)
            t_full[orig_i] = t_dec[nearest_dec_vi]
            p_full[orig_i] = p_dec[nearest_dec_vi]
            n_failed += 1
            failed_indices.append(orig_i)

        if verbose and (idx_count + 1) % 500 == 0:
            print(f'  Processed {idx_count + 1}/{len(removed_verts)} vertices '
                  f'({n_interpolated} interpolated, {n_failed} fallback)')

    if verbose:
        print(f'Interpolation complete: {n_interpolated} interpolated, '
              f'{n_failed} fallback (nearest vertex)')

    return t_full, p_full, {
        'n_survived': len(survived_set),
        'n_interpolated': n_interpolated,
        'n_failed': n_failed,
        'failed_indices': np.array(failed_indices, dtype=int),
    }
