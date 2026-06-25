"""
Multi-objective Newton optimization for spherical parameterization.

Port of MATLAB ``newton_optimization.m`` objective with per-face target areas Ao,
flip penalty, angle deformation, and edge-length equalization.
Uses Cartesian-coordinate sparse Jacobian (same pattern as newton_steps_cart).
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres, spsolve

from ..utils import kk_sph2cart, kk_cart2sph
from ..surface_mesh import surface_mesh


def build_face_membership(F, nvert):
    """For each vertex, list of incident face indices."""
    face_memb = [[] for _ in range(nvert)]
    for fi in range(len(F)):
        for vi in F[fi]:
            face_memb[int(vi)].append(fi)
    return face_memb


def default_multi_objective_opts(**overrides):
    """Default optimization hyperparameters (MATLAB ``bin2shp``-aligned)."""
    opts = {
        'lambdaA': 1.0,
        'lambda_flip': 1e3,
        'lambda1': 1e-4,
        'lambda2': 1e-2,
        'maxiter': 300,
        'min_step': 1e-6,
        'stepfac': 0.1,
        'dchg': 10.0 * np.sqrt(np.finfo(float).eps),
        'total_area': 4.0 * np.pi,
        'fixed': None,
        'prevent_flip': True,
        'verbose': 0,
    }
    opts.update(overrides)
    return opts


def _face_metrics(t, p, F, face_indices):
    """Compute area, shear, edge variance, flip flag for faces."""
    nfaces = len(F)
    if face_indices is None:
        face_indices = np.arange(nfaces)
    else:
        face_indices = np.asarray(face_indices, dtype=int)

    A = np.zeros(nfaces, dtype=float)
    shear = np.zeros(nfaces, dtype=float)
    edge_def = np.zeros(nfaces, dtype=float)
    flipped = np.zeros(nfaces, dtype=bool)

    u, v, w = kk_sph2cart(t, p, np.ones(len(t)))
    X_sph = np.column_stack([u, v, w])

    for fi in face_indices:
        v1, v2, v3 = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        A[fi] = surface_mesh.spherical_triangle_area(
            t[v1], p[v1], t[v2], p[v2], t[v3], p[v3])

        p0, p1, p2 = X_sph[v1], X_sph[v2], X_sph[v3]
        flipped[fi] = np.dot(p0, np.cross(p1, p2)) <= 0

        _, sh, _ = surface_mesh.spherical_triangle_angles_and_shear(
            t[v1], p[v1], t[v2], p[v2], t[v3], p[v3])
        shear[fi] = sh

        ea = surface_mesh.spherical_edge_length(t[v2], p[v2], t[v3], p[v3])
        eb = surface_mesh.spherical_edge_length(t[v3], p[v3], t[v1], p[v1])
        ec = surface_mesh.spherical_edge_length(t[v1], p[v1], t[v2], p[v2])
        edges = np.array([ea, eb, ec])
        edge_def[fi] = np.sum((edges - edges.mean()) ** 2)

    return A, shear, edge_def, flipped


def multi_objective_energy(t, p, F, Ao, opts=None, face_indices=None, A_cache=None):
    """Per-face energy contributions for the multi-objective function."""
    if opts is None:
        opts = default_multi_objective_opts()

    Ao = np.asarray(Ao, dtype=float).reshape(-1)
    nfaces = len(F)
    lambdaA = opts.get('lambdaA', 1.0)
    lambda_flip = opts.get('lambda_flip', opts.get('lambda', 1e3))
    lambda1 = opts.get('lambda1', 1e-4)
    lambda2 = opts.get('lambda2', 1e-2)
    total_area = opts.get('total_area', 4.0 * np.pi)

    if A_cache is None:
        A, shear, edge_def, flipped = _face_metrics(t, p, F, face_indices)
    else:
        A = A_cache
        _, shear, edge_def, flipped = _face_metrics(t, p, F, face_indices)

    dA = (A - Ao) ** 2
    global_area_pen = lambda_flip * (np.sum(A) - total_area) ** 2
    E = (lambdaA * dA + lambda1 * shear + lambda2 * edge_def
         + lambda_flip * flipped.astype(float) + global_area_pen)
    aux = {'dA': dA, 'shear': shear, 'edge_def': edge_def,
           'global_area_pen': global_area_pen}
    return E, A, flipped, aux


def _cartesian_to_tp(X_cart):
    norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xn = X_cart / norms
    t, p, _ = kk_cart2sph(Xn[:, 0], Xn[:, 1], Xn[:, 2])
    return np.mod(t, np.pi), np.mod(p, 2.0 * np.pi)


def newton_multi_objective(t, p, F, Ao, opts=None, filename='multi_obj'):
    """Optimize (t, p) using multi-objective Newton with sparse Cartesian Jacobian."""
    if opts is None:
        opts = default_multi_objective_opts()

    t = np.asarray(t, dtype=float).copy()
    p = np.mod(np.asarray(p, dtype=float), 2.0 * np.pi).copy()
    F = np.asarray(F, dtype=int)
    Ao = np.asarray(Ao, dtype=float).reshape(-1)
    nvert = len(t)
    nfaces = len(F)

    maxiter = int(opts.get('maxiter', 300))
    min_step = float(opts.get('min_step', 1e-6))
    stepfac = float(opts.get('stepfac', 0.1))
    verbose = int(opts.get('verbose', 0))
    prevent_flip = bool(opts.get('prevent_flip', True))
    seps = np.sqrt(np.finfo(float).eps)

    u, v, w = kk_sph2cart(t, p, np.ones(nvert))
    X_cart = np.column_stack([u, v, w])

    JacPat = lil_matrix((nfaces, nvert * 3))
    for ix in range(nfaces):
        for vert in F[ix]:
            JacPat[ix, int(vert) * 3] = 1
            JacPat[ix, int(vert) * 3 + 1] = 1
            JacPat[ix, int(vert) * 3 + 2] = 1
    JacPat = JacPat.tocsr()
    indrow, indcol = JacPat.nonzero()
    indJ = np.arange(len(indrow))

    orient_ref = None
    if prevent_flip:
        _, orient_ref = surface_mesh.spherical_triangles_valid_orientation(
            X_cart, F, None, min_orient=1e-8)

    fixed = opts.get('fixed', None)
    if fixed is not None:
        fixed = np.asarray(fixed, dtype=int)

    residuals = []

    for iteration in range(maxiter):
        norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
        X_cart = X_cart / np.maximum(norms, 1e-12)
        t_cur, p_cur = _cartesian_to_tp(X_cart)

        E, A, _, _ = multi_objective_energy(t_cur, p_cur, F, Ao, opts)
        X_flat = X_cart.flatten()
        CHG_vec = np.maximum(seps * np.abs(X_flat), 1e-10)

        E_plus = np.zeros(len(indJ), dtype=float)
        for jx in range(len(indJ)):
            row, col = indrow[jx], indcol[jx]
            vert_idx = col // 3
            coord_idx = col % 3
            v1, v2, v3 = F[row, 0], F[row, 1], F[row, 2]
            if vert_idx not in (v1, v2, v3):
                E_plus[jx] = E[row]
                continue
            X_pert = X_cart.copy()
            X_pert[vert_idx, coord_idx] = X_flat[col] + CHG_vec[col]
            X_pert[vert_idx] /= max(np.linalg.norm(X_pert[vert_idx]), 1e-12)
            t_p, p_p = _cartesian_to_tp(X_pert)
            E_p, _, _, _ = multi_objective_energy(
                t_p, p_p, F, Ao, opts, face_indices=[row])
            E_plus[jx] = E_p[row]

        Jvals = (E_plus - E[indrow]) / CHG_vec[indcol]
        Jvals = np.nan_to_num(Jvals, nan=0.0, posinf=0.0, neginf=0.0)
        J = csr_matrix((Jvals, (indrow, indcol)), shape=(nfaces, nvert * 3))

        A_mat = J @ J.T
        b = -E
        try:
            dv, info = gmres(A_mat, b, restart=50, maxiter=100, rtol=1e-5, atol=1e-7)
            if info != 0:
                dv = spsolve(A_mat, b)
        except Exception:
            dv = np.zeros(nfaces, dtype=float)

        dX_flat = stepfac * J.T.dot(dv)
        dX = dX_flat.reshape((nvert, 3))
        step_norm = np.linalg.norm(dX)

        X_trial = X_cart + dX
        norms = np.linalg.norm(X_trial, axis=1, keepdims=True)
        X_trial = X_trial / np.maximum(norms, 1e-12)

        accepted = True
        if prevent_flip:
            ok, _ = surface_mesh.spherical_triangles_valid_orientation(
                X_trial, F, orient_ref, min_orient=1e-8)
            if not ok:
                alpha = stepfac
                accepted = False
                for _ in range(15):
                    alpha *= 0.5
                    X_try = X_cart + alpha * dX
                    X_try = X_try / np.maximum(
                        np.linalg.norm(X_try, axis=1, keepdims=True), 1e-12)
                    ok, _ = surface_mesh.spherical_triangles_valid_orientation(
                        X_try, F, orient_ref, min_orient=1e-8)
                    if ok:
                        X_trial = X_try
                        accepted = True
                        break

        if accepted:
            X_cart = X_trial

        if fixed is not None and len(fixed):
            u_f, v_f, w_f = kk_sph2cart(t[fixed], p[fixed], np.ones(len(fixed)))
            X_cart[fixed] = np.column_stack([u_f, v_f, w_f])

        residuals.append(float(np.linalg.norm(E)))
        if verbose >= 1 and (iteration < 3 or iteration % 10 == 0):
            print(f"  iter {iteration}: ||E||={residuals[-1]:.4e}, "
                  f"sum(A)={np.sum(A):.4f}, step={step_norm:.4e}")

        if step_norm <= min_step:
            break

    t_out, p_out = _cartesian_to_tp(X_cart)
    E_final, A_final, flip_final, aux_final = multi_objective_energy(
        t_out, p_out, F, Ao, opts)
    report = {
        'residuals': residuals,
        'sum_area': float(np.sum(A_final)),
        'n_flipped': int(np.sum(flip_final)),
        'energy_norm': float(np.linalg.norm(E_final)),
        **aux_final,
    }
    return t_out, p_out, residuals, report
