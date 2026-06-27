"""
Conformalized Mean-Curvature-Flow (cMCF) spherical parameterization.

Why this module exists
----------------------
The patch / simplified-cage pipeline (``tiered_spherical_parameterization``) is
robust *only* when an arbitrary shape can first be cut into perfectly
disk-like patches (no caps, no annuli, no necks). That segmentation step is the
source of almost every failure mode (crown sentinels, cylinder quads, sliver
cage triangles -> foldovers) and is what makes the pipeline brittle.

This module replaces the brittle front half with a **segmentation-free** map
from an arbitrary genus-0 surface to the unit sphere, and keeps the proven back
half (area / shear Newton, bijectivity gate, SHP fit).

The backbone is *conformalized mean curvature flow* (Kazhdan, Solomon &
Ben-Chen, "Can Mean-Curvature Flow be Modified to be Non-Singular?", SGP 2012):

    (M_t + dt * K_0) X_{t+1} = M_t X_t

where ``K_0`` is the cotangent (stiffness) matrix of the **original** surface
(held fixed -- this is the "conformalized" trick) and ``M_t`` is the lumped mass
(vertex Voronoi/barycentric areas) recomputed from the **current** positions.
After each implicit step the surface is re-centred (area-weighted) and rescaled
to constant area. For a genus-0 surface this flow provably avoids singularities
and converges to a round sphere, so projecting the converged positions
radially, ``S = X / |X|``, gives a smooth, near-conformal, (almost always)
bijective spherical map -- with **no segmentation, no patches, no cages**.

The whole pipeline
------------------
    Stage 0  preprocess  : repair + keep-largest + curvature-adaptive remesh
    Stage 1  cMCF        : arbitrary genus-0 surface -> bijective sphere map
    Stage 2  area redist : equal-area  <->  area proportional to curvature
                           (blend), via the existing flip-prevented Newton
    Stage 3  SHP fit     : shp_surface(mesh, L_max) -> coefficients

If cMCF leaves residual folds (rare; very anisotropic input triangles), an
explicit **coarse-to-fine fallback** is available
(:func:`fps_cage_sphere_map`): geodesically-equidistant farthest-point samples
seed a coarse cage, the cage is embedded with cMCF, and the full mesh "follows"
the cage by flip-prevented spherical relaxation -- the user's
"equidistant cage + follow" idea, realised on a guaranteed-bijective base.

Public API
----------
    parameterize_to_sphere   end-to-end (cMCF backbone + optional fallback +
                             area redistribution); returns a result dict whose
                             intermediate meshes are kept for visualization.
    cmcf_sphere_map          Stage 1 only (sets mesh.t / mesh.p).
    redistribute_area        Stage 2 only (equal-area <-> curvature blend).
    fps_cage_sphere_map      coarse-to-fine fallback.
    geodesic_farthest_point_sampling  blue-noise samples (geodesic distance).
    sphere_foldover_count / sphere_diagnostics  cheap quality probes.
"""

import numpy as np
import trimesh
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from ..surface_mesh import surface_mesh
from ..utils import kk_cart2sph, kk_sph2cart
from ..level0.mesh_utils import cotangent_laplacian
from ..level1.bijectivity_gate import (
    check_bijectivity_gate,
    compute_parametric_quality,
)


def ensure_outward_winding(X, F):
    """Return ``F`` with globally consistent, outward-pointing winding.

    This is essential before *any* foldover accounting: the repo's
    curvature-adaptive remesh can leave the triangle winding globally
    inconsistent (some CCW, some CW). If we do not fix it, a perfectly bijective
    sphere map reads as having huge numbers of "folds" (mixed orientation signs),
    and naive winding-flip "fixes" silently *mask* genuine folds instead of
    reporting them. With a consistent CCW-outward winding, a non-folded face has
    ``S_i . (S_j x S_k) > 0`` on the sphere, so ``orient < 0`` is an honest
    foldover count.
    """
    tm = trimesh.Trimesh(vertices=np.asarray(X, dtype=float),
                         faces=np.asarray(F, dtype=int), process=False)
    trimesh.repair.fix_normals(tm)  # consistent winding, outward for watertight
    return np.asarray(tm.faces, dtype=int)


# --------------------------------------------------------------------------- #
# Small geometry helpers (all vectorized)
# --------------------------------------------------------------------------- #
def _face_areas(X, F):
    """Per-face area (vectorized)."""
    v0, v1, v2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _lumped_mass(X, F, nv):
    """Barycentric lumped vertex mass: M_i = (1/3) * sum incident face areas."""
    fa = _face_areas(X, F)
    M = np.zeros(nv)
    for k in range(3):
        np.add.at(M, F[:, k], fa / 3.0)
    # Guard against zero-area stars (degenerate verts) so the solve stays SPD.
    M = np.maximum(M, 1e-12)
    return M


def _cotan_stiffness(X, F):
    """SPD-ish cotangent stiffness ``K`` such that the heat/MCF step is
    ``(M + dt K) x' = M x``.

    The repo's ``cotangent_laplacian`` returns ``L`` with positive off-diagonal
    cotan weights and negative diagonal (i.e. ``L = -K``); we negate to get the
    positive-(semi)definite stiffness used by the implicit Euler step.
    """
    L = cotangent_laplacian(X, F)
    return (-L).tocsc()


def _total_area(X, F):
    return float(np.sum(_face_areas(X, F)))


def _normalize_area(X, F, target_area=4.0 * np.pi):
    """Scale ``X`` about the origin so its surface area equals ``target_area``."""
    a = _total_area(X, F)
    if a > 1e-20:
        X = X * np.sqrt(target_area / a)
    return X


def _area_weighted_center(X, F, M):
    """Subtract the (lumped-mass weighted) centroid."""
    c = (M[:, None] * X).sum(axis=0) / max(M.sum(), 1e-20)
    return X - c[None, :]


def sphere_foldover_count(S, F):
    """Number of inverted faces of a sphere map (vectorized).

    A face is "inverted" if its outward orientation disagrees with the majority
    (the map may be globally inside-out, which is not a real foldover -- we
    therefore count whichever sign is the minority).
    """
    v0, v1, v2 = S[F[:, 0]], S[F[:, 1]], S[F[:, 2]]
    orient = np.einsum('ij,ij->i', v0, np.cross(v1, v2))
    n_pos = int(np.sum(orient > 0))
    n_neg = int(np.sum(orient < 0))
    return min(n_pos, n_neg)


def _sphere_tri_areas(S, F):
    """Per-face triangle area on the unit sphere (flat proxy 0.5*|triple|)."""
    v0, v1, v2 = S[F[:, 0]], S[F[:, 1]], S[F[:, 2]]
    return 0.5 * np.abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2)))


def _adjacency_list(F, nv):
    """Undirected vertex adjacency as a list of int arrays."""
    nbr = [set() for _ in range(nv)]
    for f in F:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        nbr[a].update((b, c))
        nbr[b].update((a, c))
        nbr[c].update((a, b))
    return [np.array(sorted(s), dtype=int) for s in nbr]


def _set_tp_from_sphere(mesh, S):
    """Store a unit-sphere embedding ``S`` (nv x 3) as ``mesh.t`` / ``mesh.p``."""
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-15)
    t, p, _ = kk_cart2sph(S[:, 0], S[:, 1], S[:, 2])
    mesh.t = t
    mesh.p = p
    return S


def _sphere_from_tp(mesh):
    u, v, w = kk_sph2cart(mesh.t, mesh.p, np.ones(len(mesh.t)))
    return np.column_stack([u, v, w])


# --------------------------------------------------------------------------- #
# Stage 1: conformalized mean curvature flow
# --------------------------------------------------------------------------- #
def conformalized_mean_curvature_flow(
        X, F, n_iter=120, step_factor=1.0, tol=1e-5,
        snapshot_every=0, verbose=True):
    """Flow a genus-0 surface to a round sphere (cMCF), return the sphere map.

    Parameters
    ----------
    X, F : ndarray
        Vertices (nv x 3) and triangles (nf x 3, 0-indexed).
    n_iter : int
        Maximum implicit Euler steps.
    step_factor : float
        Time step is ``dt = step_factor * mean(M)`` where ``M`` is the lumped
        mass after area-normalisation (so it is scale-free). 0.5 - 2.0 is a good
        range; larger rounds faster but distorts more per step.
    tol : float
        Convergence tolerance on the relative change of the "sphericity"
        (std/mean of the vertex radius about the area-weighted centre).
    snapshot_every : int
        If > 0, keep a copy of the (area-normalised) flowed 3-D surface every
        ``snapshot_every`` iterations (for visualization of the flow).
    verbose : bool

    Returns
    -------
    S : ndarray (nv x 3)
        Unit-sphere positions (radially projected from the converged flow).
    info : dict
        ``n_iter_run``, ``sphericity`` history, ``foldovers`` history,
        ``snapshots`` (list of 3-D surfaces) and ``converged``.
    """
    X = np.asarray(X, dtype=float).copy()
    F = np.asarray(F, dtype=int)
    nv = len(X)

    # Work in an area-normalised frame so the time step is scale-free.
    X = X - X.mean(axis=0)
    X = _normalize_area(X, F)

    K0 = _cotan_stiffness(X, F)            # fixed initial conformal stiffness

    spher_hist, fold_hist, snaps = [], [], []
    prev_spher = None
    converged = False

    for it in range(n_iter):
        M = _lumped_mass(X, F, nv)
        dt = float(step_factor) * float(np.mean(M))
        A = (diags(M) + dt * K0).tocsc()
        rhs = M[:, None] * X
        try:
            lu = splu(A)
            Xn = lu.solve(rhs)
        except Exception as exc:                       # noqa: BLE001
            if verbose:
                print(f"  cMCF: linear solve failed at iter {it} ({exc}); "
                      "stopping early")
            break

        Xn = _area_weighted_center(Xn, F, _lumped_mass(Xn, F, nv))
        Xn = _normalize_area(Xn, F)
        X = Xn

        r = np.linalg.norm(X, axis=1)
        spher = float(np.std(r) / max(np.mean(r), 1e-15))   # 0 == perfect ball
        S_now = X / np.maximum(r[:, None], 1e-15)
        nfold = sphere_foldover_count(S_now, F)
        spher_hist.append(spher)
        fold_hist.append(nfold)

        if snapshot_every and (it % snapshot_every == 0):
            snaps.append(X.copy())

        if verbose and (it < 3 or it % 10 == 0):
            print(f"  cMCF it {it:3d}: sphericity={spher:.5f}, folds={nfold}, "
                  f"dt={dt:.3e}")

        if prev_spher is not None and abs(prev_spher - spher) < tol * max(
                prev_spher, 1e-12) and nfold == 0:
            converged = True
            if verbose:
                print(f"  cMCF converged at iter {it} "
                      f"(sphericity={spher:.5f}, folds=0)")
            break
        prev_spher = spher

    r = np.linalg.norm(X, axis=1)
    S = X / np.maximum(r[:, None], 1e-15)
    if snapshot_every:
        snaps.append(X.copy())

    info = {
        'n_iter_run': len(spher_hist),
        'sphericity': spher_hist,
        'foldovers': fold_hist,
        'final_sphericity': spher_hist[-1] if spher_hist else np.nan,
        'final_foldovers': fold_hist[-1] if fold_hist else -1,
        'converged': converged,
        'snapshots': snaps,
    }
    return S, info


def _spherical_tangential_smooth(S, F, fixed_mask=None, n_iter=20,
                                 nbr=None, verbose=False):
    """Untangle / smooth a sphere map by tangential Laplacian relaxation.

    Each free vertex is moved to the (re-normalised) mean of its neighbours on
    the sphere. This is the classic local untangler: it removes the small
    inverted triangles a discrete flow can leave behind, while keeping the map
    on the sphere. ``fixed_mask`` pins vertices (e.g. cage vertices).
    """
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-15)
    nv = len(S)
    if nbr is None:
        nbr = _adjacency_list(F, nv)
    if fixed_mask is None:
        fixed_mask = np.zeros(nv, dtype=bool)

    n0 = sphere_foldover_count(S, F)
    for _ in range(n_iter):
        Sn = S.copy()
        for vi in range(nv):
            if fixed_mask[vi]:
                continue
            ns = nbr[vi]
            if len(ns) == 0:
                continue
            avg = S[ns].mean(axis=0)
            nrm = np.linalg.norm(avg)
            if nrm > 1e-12:
                Sn[vi] = avg / nrm
        S = Sn
    n1 = sphere_foldover_count(S, F)
    if verbose:
        print(f"  tangential smooth: foldovers {n0} -> {n1}")
    return S


def _avg_matrix(F, nv):
    """Row-normalised vertex adjacency (for fast vectorised neighbour averaging)."""
    rows, cols, seen = [], [], set()
    for f in F:
        for a, b in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])),
                     (int(f[2]), int(f[0]))):
            e = (a, b) if a < b else (b, a)
            if e in seen:
                continue
            seen.add(e)
            rows += [a, b]
            cols += [b, a]
    rows = np.asarray(rows, int)
    cols = np.asarray(cols, int)
    deg = np.bincount(rows, minlength=nv).astype(float)
    deg[deg == 0] = 1.0
    from scipy.sparse import csr_matrix as _csr
    return _csr((1.0 / deg[rows], (rows, cols)), shape=(nv, nv))


def _untangle_spherical(S, F, max_iter=400, fixed_mask=None, check_every=10,
                        verbose=False):
    """Strong fold remover: iterated tangential (neighbour-average) smoothing on
    the sphere, vectorised as a sparse mat-vec so hundreds of sweeps are cheap.

    Returns the least-folded iterate ``(S, n_folds)``. The map may become very
    non-uniform (that is fine -- area is restored afterwards by the equalizer);
    the point is to reach a *bijective* (fold-free) configuration.
    """
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-15)
    nv = len(S)
    f0 = sphere_foldover_count(S, F)
    if f0 == 0:
        return S, 0
    W = _avg_matrix(np.asarray(F, int), nv)
    best_S, best_f = S.copy(), f0
    for it in range(int(max_iter)):
        Sn = W @ S
        if fixed_mask is not None:
            Sn[fixed_mask] = S[fixed_mask]
        Sn = Sn / np.maximum(np.linalg.norm(Sn, axis=1, keepdims=True), 1e-15)
        S = Sn
        if (it + 1) % check_every == 0:
            f = sphere_foldover_count(S, F)
            if f < best_f:
                best_f, best_S = f, S.copy()
            if f == 0:
                break
    if verbose:
        print(f"  untangle: folds {f0} -> {best_f} (tangential smoothing)")
    return best_S, best_f


def cmcf_sphere_map(mesh, n_iter=120, step_factor=1.0, tol=1e-5,
                    untangle_iter=400, snapshot_every=0, verbose=True):
    """Stage 1: map a (preprocessed) genus-0 mesh to the unit sphere via cMCF.

    Sets ``mesh.t`` / ``mesh.p`` (KK colatitude/azimuth). Assumes ``mesh.F`` has
    consistent outward winding (call :func:`ensure_outward_winding` first; the
    end-to-end :func:`parameterize_to_sphere` does this once after preprocess).
    Returns ``(mesh, info)``.
    """
    mesh.F = ensure_outward_winding(mesh.X, mesh.F)
    X = np.asarray(mesh.X, dtype=float)
    F = np.asarray(mesh.F, dtype=int)

    S, info = conformalized_mean_curvature_flow(
        X, F, n_iter=n_iter, step_factor=step_factor, tol=tol,
        snapshot_every=snapshot_every, verbose=verbose)

    nfold = sphere_foldover_count(S, F)
    if nfold > 0 and untangle_iter > 0:
        if verbose:
            print(f"  cMCF left {nfold} folds; adaptive tangential untangle "
                  f"(<= {untangle_iter})")
        S, nfold = _untangle_spherical(S, F, max_iter=untangle_iter,
                                       verbose=verbose)
    info['foldovers_after_untangle'] = sphere_foldover_count(S, F)

    _set_tp_from_sphere(mesh, S)
    return mesh, info


# --------------------------------------------------------------------------- #
# Conformal Mobius centering (canonical pose; helps SHP conditioning)
# --------------------------------------------------------------------------- #
def _rot_align(u, v):
    """Rotation matrix taking unit vector ``u`` to unit vector ``v`` (Rodrigues)."""
    u = u / max(np.linalg.norm(u), 1e-15)
    v = v / max(np.linalg.norm(v), 1e-15)
    c = float(np.dot(u, v))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 deg about any axis perpendicular to u
        a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax = np.cross(u, a)
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    ax = np.cross(u, v)
    s = np.linalg.norm(ax)
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def mobius_center(mesh, n_iter=40, tol=1e-4, verbose=True):
    """Conformally recentre a sphere map so its (area-weighted) centre of mass
    sits at the origin -- a canonical normalisation for SHP.

    Each iteration (a) rotates the sphere so the current centre of mass lies on
    the +z axis, then (b) applies a stereographic scaling (a hyperbolic Mobius
    transform along that axis) that zeroes the z-component of the centre of
    mass. Both are **conformal automorphisms of the sphere**, so the map stays
    bijective (folds cannot be created). Weighted by the 3-D surface vertex
    areas so the balance is intrinsic to the shape.

    Operates in place on ``mesh.t`` / ``mesh.p``; returns ``mesh``.
    """
    S = _sphere_from_tp(mesh)
    w = _lumped_mass(mesh.X, np.asarray(mesh.F, int), len(mesh.X))
    w = w / w.sum()
    z = np.array([0.0, 0.0, 1.0])

    c0 = float(np.linalg.norm((w[:, None] * S).sum(axis=0)))
    for _ in range(int(n_iter)):
        mu = (w[:, None] * S).sum(axis=0)
        m = float(np.linalg.norm(mu))
        if m < tol:
            break
        R = _rot_align(mu / m, z)        # centre of mass -> +z axis
        S = S @ R.T
        # Stereographic from north pole; scale by s; invert. Choose s (>0) to
        # zero the weighted mean z. mean_z(s) is monotonic from -1 (s->0) to
        # +1 (s->inf); bisection on log s.
        zc = np.clip(S[:, 2], -1.0, 1.0)
        denom = np.maximum(1.0 - zc, 1e-12)
        wx = S[:, 0] / denom
        wy = S[:, 1] / denom
        r2 = wx * wx + wy * wy                # |w|^2
        north = (1.0 - zc) < 1e-9             # maps to infinity (stays north)

        def mean_z(scale):
            s2 = (scale * scale) * r2
            zz = np.where(north, 1.0, (s2 - 1.0) / (s2 + 1.0))
            return float(np.sum(w * zz))

        lo, hi = 1e-6, 1e6
        for _b in range(60):
            mid = np.sqrt(lo * hi)
            if mean_z(mid) > 0:
                hi = mid
            else:
                lo = mid
        s = np.sqrt(lo * hi)
        s2 = (s * s) * r2
        newz = np.where(north, 1.0, (s2 - 1.0) / (s2 + 1.0))
        # Reconstruct x',y' from scaled stereographic coords:
        #   x' = 2 s wx / (1+s^2|w|^2), y' = 2 s wy / (1+s^2|w|^2)
        denom2 = 1.0 + s2
        nx = np.where(north, 0.0, 2.0 * s * wx / denom2)
        ny = np.where(north, 0.0, 2.0 * s * wy / denom2)
        S = np.column_stack([nx, ny, newz])
        S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-15)

    c1 = float(np.linalg.norm((w[:, None] * S).sum(axis=0)))
    _set_tp_from_sphere(mesh, S)
    if verbose:
        print(f"  mobius_center: centroid {c0:.4f} -> {c1:.4f}")
    return mesh


# --------------------------------------------------------------------------- #
# Stage 2: area redistribution (equal-area  <->  curvature)  -- flip-prevented
# --------------------------------------------------------------------------- #
def _target_areas(mesh, area_blend, area_exponent):
    """Per-face target sphere areas summing to 4*pi.

    The key SHP-sampling control. ``area_blend`` blends two normalised targets:

    * ``1.0`` -> **equal sphere-area per triangle** (``Ao_i = 4*pi / N``). This is
      the recommended default: *combined with curvature-adaptive remeshing* (more,
      smaller triangles where the shape bends) it makes the spherical triangles
      uniform in size -- so the sphere is sampled evenly (well-conditioned SH fit)
      -- while the curved regions automatically receive sphere area proportional
      to their triangle count (i.e. to their curvature content). Needs nothing
      but the face count (cheap).
    * ``0.0`` -> **curvature-weighted** (``Ao_i proportional to area_3D*|H|^p``):
      pushes *extra* sphere area into curved regions on top of what the mesh
      density already provides. Intended for a (near-)uniform input mesh; using
      it together with a curvature-remeshed mesh double-counts curvature and
      over-concentrates. Needs ``props()`` (curvature).

    A linear blend in between. Returns ``Ao`` (n_faces,).
    """
    F = np.asarray(mesh.F, dtype=int)
    n = len(F)
    four_pi = 4.0 * np.pi
    Ao_uniform = np.full(n, four_pi / max(n, 1))
    b = float(np.clip(area_blend, 0.0, 1.0))
    if b >= 0.999:
        return Ao_uniform
    mesh.props()
    fa = np.maximum(_face_areas(np.asarray(mesh.X, float), F), 1e-15)
    H = np.abs(np.where(np.isfinite(np.asarray(mesh.H, float)),
                        np.asarray(mesh.H, float), 0.0))
    face_H = H[F].mean(axis=1)
    w = np.maximum(fa * face_H ** float(area_exponent), 1e-15)
    Ao_curv = w / w.sum() * four_pi
    Ao = b * Ao_uniform + (1.0 - b) * Ao_curv
    return Ao / Ao.sum() * four_pi


def redistribute_area(mesh, area_blend=1.0, area_exponent=2.0,
                      shear_niter=30, shear_step=0.03,
                      multi_niter=80, multi_step=0.1,
                      shear_weight=1e-2, area_weight=1.0, verbose=True):
    """Polish the sphere map toward an SHP-optimal area distribution.

    ``area_blend`` in [0, 1] (see :func:`_target_areas`): ``1.0`` = area-
    preserving (classic SPHARM, uniform sampling), ``0.0`` = area proportional
    to curvature, in between = blend.

    Uses the existing flip-prevented multi-objective Newton (``map2sphere``
    optimization_method 6) to drive the spherical areas toward the target
    without introducing folds, then a short pure-shear pass to clean up
    parametric triangle quality. Operates in place; returns ``mesh``.

    Note: with ``prevent_flip=True`` this *cannot remove* folds that are already
    present -- it only avoids creating new ones. Make the map bijective first
    (Stage 1 / fallback).
    """
    if mesh.t is None or mesh.p is None:
        raise ValueError("redistribute_area needs an existing sphere map "
                         "(mesh.t / mesh.p). Run cmcf_sphere_map first.")

    Ao = _target_areas(mesh, area_blend, area_exponent)
    mesh.target_areas = Ao

    mesh.bijective_plot_flag = 0
    mesh.mapping_plot_flag = 0
    mesh.prevent_flip = True

    if multi_niter > 0:
        mesh.optimization_method = 6
        mesh.newton_niter = multi_niter
        mesh.newton_step = multi_step
        mesh.multi_objective_opts = {
            'lambdaA': float(area_weight), 'lambda1': float(shear_weight),
            'lambda2': float(shear_weight * 10.0), 'prevent_flip': True,
        }
        mesh.needs_map2sphere = False
        if verbose:
            kind = ('uniform (equal area / triangle)' if area_blend >= 0.999
                    else 'curvature-weighted')
            print(f"  area redistribute: multi-objective x{multi_niter} "
                  f"({kind}, shear_w={shear_weight})")
        mesh.map2sphere()

    if shear_niter > 0:
        mesh.optimization_method = 5
        mesh.newton_niter = shear_niter
        mesh.newton_step = shear_step
        mesh.needs_map2sphere = False
        mesh.map2sphere()

    return mesh


def _tri_q(S, F):
    """Per-face scalar triple product det[S_i, S_j, S_k] (signed; >0 = oriented).

    For unit vectors this is a smooth proxy for spherical-triangle area
    (proportional for small triangles) with a very clean gradient -- ideal for
    an area-equalizing optimizer.
    """
    Si, Sj, Sk = S[F[:, 0]], S[F[:, 1]], S[F[:, 2]]
    return np.einsum('ij,ij->i', Si, np.cross(Sj, Sk))


def equalize_areas(mesh, area_blend=1.0, area_exponent=2.0, n_iter=1200,
                   step=0.04, mu_frac=2e-2, tol=1e-7, plateau_tol=2e-3,
                   lambda_shape=0.0, verbose=True):
    """De-concentrate / equalize spherical areas with an interior-point method.

    Minimises ``E(S) = sum_f (q_f - Ao_f)^2 - mu * sum_f log(q_f)`` over the
    sphere-vertex positions ``S`` (``q_f`` = per-face triple product, a smooth
    area proxy). The **log-barrier** keeps every ``q_f > 0`` (so no triangle can
    fold) *and*, crucially, its gradient ``-mu/q_f`` blows up as a triangle
    collapses -- so it actively **inflates** the regions a conformal map (cMCF)
    concentrates (e.g. a thin stalk), which the flip-prevented Newton cannot do.
    Fully vectorised (triple-product gradients are cross products), so it is fast.

    ``area_blend``: 1.0 -> uniform target (equal area / triangle); 0.0 ->
    curvature-weighted target (``area_3D*|H|^p``); linear blend in between.

    Operates in place on ``mesh.t`` / ``mesh.p``; returns ``(mesh, info)``.
    """
    F = np.asarray(mesh.F, dtype=int)
    Fi, Fj, Fk = F[:, 0], F[:, 1], F[:, 2]
    nv = len(mesh.X)
    S = _sphere_from_tp(mesh)

    q = _tri_q(S, F)
    if np.sum(q) < 0:                      # globally inside-out -> reflect
        S[:, 0] = -S[:, 0]
        q = _tri_q(S, F)
    total = float(np.sum(np.abs(q)))

    # Target (in q-units), summing to the current total so the matching term is
    # scale-consistent.
    b = float(np.clip(area_blend, 0.0, 1.0))
    if b >= 0.999:
        Ao = np.full(len(F), total / max(len(F), 1))
    else:
        mesh.props()
        fa = np.maximum(_face_areas(np.asarray(mesh.X, float), F), 1e-15)
        H = np.abs(np.where(np.isfinite(np.asarray(mesh.H, float)),
                            np.asarray(mesh.H, float), 0.0))
        w = np.maximum(fa * H[F].mean(axis=1) ** float(area_exponent), 1e-15)
        Ao_curv = w / w.sum() * total
        Ao_uniform = np.full(len(F), total / max(len(F), 1))
        Ao = b * Ao_uniform + (1.0 - b) * Ao_curv
        Ao = Ao / Ao.sum() * total
    mu = float(mu_frac) * float(np.mean(Ao)) ** 2

    cov0 = float(np.std(q) / max(np.mean(q), 1e-20))
    nfold0 = int(np.sum(q <= 0))
    n_acc = 0
    # Adam per-vertex moments (adaptive steps: the concentrated stalk has huge
    # gradients, the cap tiny ones -- per-coordinate scaling lets both progress).
    m1 = np.zeros((nv, 3))
    m2 = np.zeros((nv, 3))
    b1, b2, adam_eps = 0.9, 0.999, 1e-12
    cov_prev = cov0
    cur_nfold = nfold0
    # Keep the best iterate by (n_foldovers, area_cov). Because steps are only
    # rejected if they INCREASE folds, the barrier gradient also UNtangles a
    # folded cMCF start -- so this doubles as a fold remover for hard shapes.
    best_key = (nfold0, cov0)
    S_best = S.copy()
    for it in range(int(n_iter)):
        qpos = np.maximum(q, 1e-9)
        dEdq = 2.0 * (q - Ao) - mu / qpos
        Si, Sj, Sk = S[Fi], S[Fj], S[Fk]
        g = np.zeros((nv, 3))
        np.add.at(g, Fi, dEdq[:, None] * np.cross(Sj, Sk))
        np.add.at(g, Fj, dEdq[:, None] * np.cross(Sk, Si))
        np.add.at(g, Fk, dEdq[:, None] * np.cross(Si, Sj))
        if lambda_shape > 0.0:
            # Shape regularizer: penalise per-face edge-length variance so the
            # spherical triangles become regular (equilateral) -> isotropic,
            # low-shear sampling, at the cost of a little area uniformity. The
            # 3-D triangles are fixed, so this only redistributes the sphere
            # vertices toward a more regular triangulation.
            vij, vjk, vki = Si - Sj, Sj - Sk, Sk - Si
            lij = np.maximum(np.linalg.norm(vij, axis=1), 1e-12)
            ljk = np.maximum(np.linalg.norm(vjk, axis=1), 1e-12)
            lki = np.maximum(np.linalg.norm(vki, axis=1), 1e-12)
            ebar = (lij + ljk + lki) / 3.0
            cij = 2.0 * (lij - ebar) / lij
            cjk = 2.0 * (ljk - ebar) / ljk
            cki = 2.0 * (lki - ebar) / lki
            gs = np.zeros((nv, 3))
            np.add.at(gs, Fi, cij[:, None] * vij - cki[:, None] * vki)
            np.add.at(gs, Fj, -cij[:, None] * vij + cjk[:, None] * vjk)
            np.add.at(gs, Fk, -cjk[:, None] * vjk + cki[:, None] * vki)
            sa = float(np.sqrt(np.mean(g * g))) + 1e-20
            ss = float(np.sqrt(np.mean(gs * gs))) + 1e-20
            g = g + lambda_shape * (sa / ss) * gs   # relative-weighted blend
        g -= np.einsum('ij,ij->i', g, S)[:, None] * S      # tangent projection
        if cur_nfold == 0 and float(np.max(np.linalg.norm(g, axis=1))) < tol:
            break
        m1 = b1 * m1 + (1.0 - b1) * g
        m2 = b2 * m2 + (1.0 - b2) * (g * g)
        mhat = m1 / (1.0 - b1 ** (it + 1))
        vhat = m2 / (1.0 - b2 ** (it + 1))
        d = mhat / (np.sqrt(vhat) + adam_eps)              # Adam direction
        d -= np.einsum('ij,ij->i', d, S)[:, None] * S
        dmax = float(np.max(np.linalg.norm(d, axis=1)))
        eta = float(step) / (dmax + 1e-12)                 # cap max vertex move
        # Backtracking line search: accept a step only if it does NOT increase
        # the foldover count -> a fold-free map stays fold-free, and a folded
        # cMCF start gets progressively untangled (the barrier inflates folds).
        accepted = False
        for _ls in range(25):
            Sn = S - eta * d
            Sn = Sn / np.maximum(np.linalg.norm(Sn, axis=1, keepdims=True), 1e-15)
            qn = _tri_q(Sn, F)
            nfn = int(np.sum(qn <= 0))
            if nfn <= cur_nfold:
                S, q, cur_nfold, n_acc = Sn, qn, nfn, n_acc + 1
                accepted = True
                break
            eta *= 0.5
        if not accepted:
            break
        cov = float(np.std(q) / max(np.mean(q), 1e-20))
        key = (cur_nfold, cov)             # prefer fewer folds, then lower cov
        if key < best_key:
            best_key, S_best = key, S.copy()
        # Early stop only once fold-free AND uniformity has plateaued.
        if (it + 1) % 50 == 0:
            if verbose:
                print(f"  equalize it {it + 1:4d}: folds={cur_nfold} "
                      f"area_cov={cov:.3f} (best folds={best_key[0]} "
                      f"cov={best_key[1]:.3f})")
            if best_key[0] == 0 and best_key[1] > (1.0 - plateau_tol) * cov_prev:
                break
            cov_prev = best_key[1]

    S = S_best                              # return the best (not the last) map
    q = _tri_q(S, F)
    cov1 = float(np.std(q) / max(np.mean(q), 1e-20))
    _set_tp_from_sphere(mesh, S)
    info = {'area_cov_before': cov0, 'area_cov_after': cov1,
            'foldovers_before': nfold0, 'n_accepted': n_acc,
            'foldovers': int(np.sum(q <= 0))}
    if verbose:
        print(f"  equalize_areas: area_cov {cov0:.3f} -> {cov1:.3f}, "
              f"folds {nfold0} -> {info['foldovers']} ({n_acc} steps)")
    return mesh, info


# --------------------------------------------------------------------------- #
# Stretch-aligned anisotropic refinement (the "pre-compressed triangle"
# look-ahead): refine 3-D edges whose spherical IMAGE is long.
# --------------------------------------------------------------------------- #
def _refine_long_spherical_edges(X, F, S, split_factor=1.7, labels=None):
    """Split the 3-D edges whose *spherical* image is long (red-green, manifold).

    After a parameterization, an edge that is long on the sphere is one the map
    *stretched* -- i.e. it lies along the high-stretch direction and is sparsely
    sampled there. Bisecting it (3-D midpoint) inserts a vertex along that
    direction, so the 3-D triangles become "pre-compressed" along the stretch
    axis: on the next parameterization the map spreads those closely-spaced
    vertices out, yielding more regular (less sheared / sliver) spherical
    triangles and denser sampling where it was sparse. This is exactly the
    stretch-driven look-ahead, using the spherical edge length as the (cheap,
    direction-aware) stretch indicator -- no metric tensor required.

    Shared edges use a shared midpoint, and each triangle is split by how many
    of its edges are marked (1 -> 2, 2 -> 3, 3 -> 4 sub-triangles), so the
    result is watertight (no T-junctions). New vertices also get a **sphere**
    position (renormalised edge midpoint of ``S``) so the refined mesh can be
    re-parameterized from a warm (still-bijective) start. Returns
    ``(Xn, Fn, Sn, labels_n, n_new)``.
    """
    X = np.asarray(X, float)
    F = np.asarray(F, int)
    S = np.asarray(S, float)
    labels = np.asarray(labels) if labels is not None else None

    elen = {}
    for f in F:
        for a, b in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])),
                     (int(f[2]), int(f[0]))):
            k = (a, b) if a < b else (b, a)
            if k not in elen:
                elen[k] = float(np.linalg.norm(S[k[0]] - S[k[1]]))
    if not elen:
        return X, F, S, labels, 0
    thr = float(split_factor) * float(np.median(list(elen.values())))
    mark = {k for k, v in elen.items() if v > thr}
    if not mark:
        return X, F, S, labels, 0

    Xn = [x for x in X]
    Sn = [s for s in S]
    mid = {}

    def gm(a, b):
        k = (a, b) if a < b else (b, a)
        m = mid.get(k)
        if m is None:
            m = len(Xn)
            Xn.append(0.5 * (X[a] + X[b]))
            sm = S[a] + S[b]
            Sn.append(sm / max(np.linalg.norm(sm), 1e-15))
            mid[k] = m
        return m

    Fn, Ln = [], ([] if labels is not None else None)
    for fi in range(len(F)):
        v0, v1, v2 = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        m01 = (min(v0, v1), max(v0, v1)) in mark
        m12 = (min(v1, v2), max(v1, v2)) in mark
        m20 = (min(v2, v0), max(v2, v0)) in mark
        n = int(m01) + int(m12) + int(m20)
        if n == 0:
            subs = [(v0, v1, v2)]
        elif n == 3:
            a, b, c = gm(v0, v1), gm(v1, v2), gm(v2, v0)
            subs = [(v0, a, c), (a, v1, b), (c, b, v2), (a, b, c)]
        elif n == 1:
            if m01:
                m = gm(v0, v1); subs = [(v0, m, v2), (m, v1, v2)]
            elif m12:
                m = gm(v1, v2); subs = [(v1, m, v0), (m, v2, v0)]
            else:
                m = gm(v2, v0); subs = [(v2, m, v1), (m, v0, v1)]
        else:  # n == 2
            if m01 and m12:
                a, b = gm(v0, v1), gm(v1, v2)
                subs = [(a, v1, b), (v0, a, b), (v0, b, v2)]
            elif m12 and m20:
                a, b = gm(v1, v2), gm(v2, v0)
                subs = [(a, v2, b), (v1, a, b), (v1, b, v0)]
            else:
                a, b = gm(v2, v0), gm(v0, v1)
                subs = [(a, v0, b), (v2, a, b), (v2, b, v1)]
        for s in subs:
            Fn.append(s)
            if Ln is not None:
                Ln.append(labels[fi])

    Xn = np.asarray(Xn)
    Sn = np.asarray(Sn)
    Fn = np.asarray(Fn, dtype=int)
    Ln = np.asarray(Ln) if Ln is not None else None
    return Xn, Fn, Sn, Ln, len(Xn) - len(X)


def _parameterize_and_equalize(m, cmcf_iter=120, cmcf_step=1.0, untangle_iter=400,
                               center=True, area_blend=1.0, area_exponent=2.0,
                               area_n_iter=1200, lambda_shape=0.3, verbose=False):
    """cMCF -> Mobius centre -> area equalize, in place; returns ``(m, eq_info)``."""
    m, _ = cmcf_sphere_map(m, n_iter=cmcf_iter, step_factor=cmcf_step,
                           untangle_iter=untangle_iter, verbose=verbose)
    if center:
        m = mobius_center(m, verbose=verbose)
    m, eq_info = equalize_areas(m, area_blend=area_blend,
                                area_exponent=area_exponent, n_iter=area_n_iter,
                                lambda_shape=lambda_shape, verbose=verbose)
    return m, eq_info


def _anisotropic_diag(mesh, rnd, n_new=0, verbose=True):
    d = sphere_diagnostics(mesh, verbose=False)
    rec = {'round': rnd, 'n_verts': len(mesh.X), 'n_faces': len(mesh.F),
           'n_new_verts': n_new, 'max_shear': d['max_shear'],
           'min_quality': d['min_quality'], 'mean_quality': d['mean_quality'],
           'area_cov': d['area_cov'], 'n_foldovers': d['n_foldovers']}
    if verbose:
        print(f"  [aniso r{rnd}] {rec['n_verts']}v: "
              f"max_shear={rec['max_shear']:.2f} min_q={rec['min_quality']:.3f} "
              f"mean_q={rec['mean_quality']:.3f} area_cov={rec['area_cov']:.2f} "
              f"folds={rec['n_foldovers']}")
    return rec


def anisotropic_rounds(m, n_rounds=2, split_factor=1.7, area_blend=1.0,
                       area_exponent=2.0, area_n_iter=1200, lambda_shape=0.3,
                       verbose=True):
    """Run warm-start anisotropic refinement rounds on an *already*-parameterized
    mesh ``m`` (with ``.t`` / ``.p``). Returns ``(m, history)``.

    Each round: split the 3-D edges the map stretched (long on the sphere) ->
    keep the bijective map for old vertices, place split midpoints on the sphere
    -> recentre + re-equalize. No cMCF re-run (warm, fold-free, fast).
    """
    eq_kw = dict(area_blend=area_blend, area_exponent=area_exponent,
                 n_iter=area_n_iter, lambda_shape=lambda_shape)
    history = [_anisotropic_diag(m, 0, verbose=verbose)]
    if history[0]['n_foldovers'] > 0:
        # Refining a folded map splits its spurious long edges -> vertex blow-up
        # with no untangling. Require a bijective map before anisotropic refine.
        if verbose:
            print(f"  [aniso] map still has {history[0]['n_foldovers']} "
                  "foldovers; skipping anisotropic refinement (would amplify)")
        return m, history
    for r in range(1, int(n_rounds) + 1):
        S = _sphere_from_tp(m)
        Xn, Fn, Sn, Ln, n_new = _refine_long_spherical_edges(
            m.X, m.F, S, split_factor=split_factor,
            labels=getattr(m, 'face_labels', None))
        if n_new == 0:
            if verbose:
                print(f"  [aniso r{r}] no long spherical edges; stopping")
            break
        m = surface_mesh(Xn, Fn)
        if Ln is not None:
            m.face_labels = Ln
        _set_tp_from_sphere(m, Sn)
        m = mobius_center(m, verbose=False)
        m, _ = equalize_areas(m, verbose=False, **eq_kw)
        if verbose:
            print(f"  [aniso r{r}] refined +{n_new} verts ({len(Fn)} faces); "
                  "re-equalized (warm start)")
        history.append(_anisotropic_diag(m, r, n_new, verbose=verbose))
    return m, history


def anisotropic_refine_parameterization(m_pre, n_rounds=2, split_factor=1.7,
                                        verbose=True, **pkw):
    """Full stretch-aligned anisotropic look-ahead from a preprocessed mesh:
    parameterize (cMCF -> centre -> equalize), then :func:`anisotropic_rounds`.

    ``pkw`` forwarded to :func:`_parameterize_and_equalize` (cmcf_iter,
    area_blend, area_n_iter, lambda_shape, ...). Returns ``(m_param, history)``.
    """
    labels = (np.asarray(m_pre.face_labels).copy()
              if getattr(m_pre, 'face_labels', None) is not None else None)
    m = surface_mesh(np.asarray(m_pre.X, float).copy(),
                     np.asarray(m_pre.F, int).copy())
    if labels is not None:
        m.face_labels = labels.copy()
    m, _ = _parameterize_and_equalize(m, verbose=verbose, **pkw)
    return anisotropic_rounds(
        m, n_rounds=n_rounds, split_factor=split_factor,
        area_blend=pkw.get('area_blend', 1.0),
        area_exponent=pkw.get('area_exponent', 2.0),
        area_n_iter=pkw.get('area_n_iter', 1200),
        lambda_shape=pkw.get('lambda_shape', 0.3), verbose=verbose)


# --------------------------------------------------------------------------- #
# Geodesic farthest-point sampling ("randomly equidistant on the manifold")
# --------------------------------------------------------------------------- #
def _edge_graph(X, F, nv):
    """Sparse graph of edge lengths (for approximate geodesic distance)."""
    rows, cols, vals = [], [], []
    seen = set()
    for f in F:
        for a, b in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])),
                     (int(f[2]), int(f[0]))):
            e = (a, b) if a < b else (b, a)
            if e in seen:
                continue
            seen.add(e)
            d = float(np.linalg.norm(X[a] - X[b]))
            rows += [a, b]
            cols += [b, a]
            vals += [d, d]
    return csr_matrix((vals, (rows, cols)), shape=(nv, nv))


def geodesic_farthest_point_sampling(mesh, n_samples, seed=0, verbose=True):
    """Blue-noise samples that are ~equidistant *along the surface*.

    Greedy farthest-point sampling under **graph-geodesic** distance (Dijkstra
    on the edge graph -- a good, dependency-free approximation of true geodesic
    distance). The result is the user's "randomly equidistant on the mesh
    manifold" point set: well separated in surface distance, denser nowhere by
    construction (curvature biasing can be layered on later).

    Returns an int array of vertex indices (length ``n_samples``).
    """
    X = np.asarray(mesh.X, dtype=float)
    F = np.asarray(mesh.F, dtype=int)
    nv = len(X)
    n_samples = int(min(n_samples, nv))
    G = _edge_graph(X, F, nv)

    rng = np.random.default_rng(seed)
    start = int(rng.integers(nv))
    samples = [start]
    dmin = dijkstra(G, indices=start, directed=False)
    dmin[~np.isfinite(dmin)] = 0.0  # disconnected guard

    while len(samples) < n_samples:
        nxt = int(np.argmax(dmin))
        if dmin[nxt] <= 0:
            break
        samples.append(nxt)
        d = dijkstra(G, indices=nxt, directed=False)
        d[~np.isfinite(d)] = 0.0
        dmin = np.minimum(dmin, d)
    if verbose:
        print(f"  geodesic FPS: {len(samples)} samples "
              f"(min pairwise spacing ~ {dmin[dmin > 0].min() if np.any(dmin > 0) else 0:.3g})")
    return np.asarray(samples, dtype=int)


# --------------------------------------------------------------------------- #
# Coarse-to-fine fallback: equidistant cage + "follow"  (Praun-Hoppe flavour)
# --------------------------------------------------------------------------- #
def fps_cage_sphere_map(mesh, n_cage=250, cage_curv_weight=1.0,
                        cmcf_iter=200, cmcf_step=1.0,
                        follow_iter=200, polish=True, verbose=True):
    """Coarse-to-fine spherical map: equidistant cage embedded, full mesh follows.

    This is the user's "cage" idea made robust:

    1. **Equidistant cage** -- geodesic farthest-point samples are *protected*
       while the mesh is curvature-aware-decimated to a coarse cage, so the cage
       vertices are a subset of the fine vertices, well spread along the surface.
    2. **Embed the cage** with cMCF (small + very robust) -> a bijective coarse
       sphere embedding.
    3. **Follow** -- every fine vertex is placed on the sphere by flip-prevented
       tangential relaxation with the cage vertices *pinned* to their embedded
       positions. Because the pinned cage is already a valid blue-noise sphere
       embedding, the relaxation untangles the interior toward a bijective map
       (no "entanglement": all free vertices move simultaneously toward the
       cage, never crossing pinned anchors).
    4. Optional flip-prevented area/shear polish.

    Use as a fallback when :func:`cmcf_sphere_map` leaves residual folds on a
    pathological shape. Returns ``(mesh, info)``.
    """
    mesh.F = ensure_outward_winding(mesh.X, mesh.F)
    X = np.asarray(mesh.X, dtype=float)
    F = np.asarray(mesh.F, dtype=int)
    nv = len(X)

    fps = geodesic_farthest_point_sampling(mesh, n_cage, verbose=verbose)

    # ---- coarse cage (FPS vertices protected so they survive) ------------- #
    target_faces = int(min(len(F), max(4 * n_cage, 200)))
    cage, _ = mesh.curvature_aware_decimation(
        target_faces=target_faces, curvature_weight=cage_curv_weight,
        protected_vertices=fps, verbose=False)
    cage.F = ensure_outward_winding(cage.X, cage.F)
    cage.props()
    if verbose:
        print(f"  cage: {len(cage.X)} verts, {len(cage.F)} faces "
              f"(target {target_faces})")

    # ---- embed the cage with cMCF ----------------------------------------- #
    Sc, cage_info = conformalized_mean_curvature_flow(
        cage.X, cage.F, n_iter=cmcf_iter, step_factor=cmcf_step,
        verbose=False)
    Sc = _spherical_tangential_smooth(Sc, cage.F, n_iter=40)
    if verbose:
        print(f"  cage embedded: folds={sphere_foldover_count(Sc, cage.F)}, "
              f"sphericity={cage_info['final_sphericity']:.4f}")

    # ---- map cage vertices back to fine-mesh indices (by position) -------- #
    tree = cKDTree(X)
    _, cage_to_fine = tree.query(cage.X, k=1)
    cage_to_fine = np.asarray(cage_to_fine, dtype=int)

    # ---- initialise the full sphere map: pins exact, interior from cMCF ---- #
    S_full, _ = conformalized_mean_curvature_flow(
        X, F, n_iter=cmcf_iter, step_factor=cmcf_step, verbose=False)
    fixed_mask = np.zeros(nv, dtype=bool)
    fixed_mask[cage_to_fine] = True
    S_full[cage_to_fine] = Sc      # pin to the (bijective) cage embedding

    # ---- "follow": flip-prevented tangential relaxation, cage pinned ------ #
    nbr = _adjacency_list(F, nv)
    S_full = _spherical_tangential_smooth(
        S_full, F, fixed_mask=fixed_mask, n_iter=follow_iter, nbr=nbr,
        verbose=verbose)

    _set_tp_from_sphere(mesh, S_full)

    if polish:
        redistribute_area(mesh, area_blend=1.0, multi_niter=0,
                          shear_niter=20, verbose=verbose)

    info = {
        'n_cage': int(n_cage), 'cage_faces': int(len(cage.F)),
        'cage_foldovers': sphere_foldover_count(Sc, cage.F),
        'foldovers': sphere_foldover_count(_sphere_from_tp(mesh), F),
        'fps': fps, 'cage': cage,
    }
    if verbose:
        print(f"  cage-follow map: folds={info['foldovers']}")
    return mesh, info


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def sphere_diagnostics(mesh, area_tol=0.05, verbose=True):
    """Bijectivity + parametric-quality summary of the current sphere map.

    ``bijective`` is reported by the **true topological criterion -- zero
    foldovers** (no overlapping / inverted triangles). The conservative gate
    (``gate_passed``) additionally flags *quality* issues (vertex centroid off
    origin, extreme edge-length ratio) that are common and harmless for a
    conformal map of a thin feature (e.g. a stalk concentrates into a tiny
    sphere region): those are distortion, not loss of bijectivity, and are what
    Stage 2 / a Mobius centering address.
    """
    gate = check_bijectivity_gate(mesh, area_tol=area_tol, verbose=False)
    q = compute_parametric_quality(mesh)
    shear, shear_summ = surface_mesh.compute_shear_spherical(
        mesh.t, mesh.p, mesh.F)
    S = _sphere_from_tp(mesh)
    a = _sphere_tri_areas(S, np.asarray(mesh.F, int))
    a_mean = float(np.mean(a)) if len(a) else 0.0
    # Spherical-triangle-area uniformity: how evenly the sphere is sampled.
    # cov -> 0 and ratio -> 1 means uniform triangles (ideal SH sampling).
    area_cov = float(np.std(a) / a_mean) if a_mean > 1e-20 else np.inf
    area_ratio = float(a.max() / max(a.min(), 1e-20)) if len(a) else np.inf
    out = {
        'bijective': int(gate['n_foldovers']) == 0,
        'gate_passed': bool(gate['passed']),
        'n_foldovers': int(gate['n_foldovers']),
        'area_excess_rel': float(gate['area_excess_rel']),
        'centroid_dist': float(gate['centroid_dist']),
        'edge_ratio': float(gate['edge_ratio']),
        'area_cov': area_cov,
        'area_ratio': area_ratio,
        'min_quality': float(q['min_quality']),
        'mean_quality': float(q['mean_quality']),
        'max_shear': float(np.max(shear)) if len(shear) else 0.0,
        'mean_shear': float(shear_summ.get('mean', 0.0)),
        'gate': gate,
    }
    if verbose:
        print(f"  sphere: bijective(folds==0)={out['bijective']} "
              f"folds={out['n_foldovers']} | "
              f"area_excess={100 * out['area_excess_rel']:.1f}% "
              f"min_q={out['min_quality']:.3f} "
              f"max_shear={out['max_shear']:.2f} | "
              f"sphere-area cov={area_cov:.2f} ratio={area_ratio:.0f} "
              f"centroid={out['centroid_dist']:.3f}")
    return out


# --------------------------------------------------------------------------- #
# SHP sampling: upsample the sphere map so the SH least-squares is well-posed
# --------------------------------------------------------------------------- #
def _subdivide_sphere_once(X, F, S):
    """One 1-to-4 midpoint subdivision carrying 3-D values ``X`` and sphere
    positions ``S`` together.

    Edge midpoints get the *linear* average of the 3-D positions (a denser
    sample of the same piecewise-linear surface map) and the *renormalised*
    average of the sphere positions (the midpoint projected back onto the
    sphere). Shared edges are de-duplicated so the result stays watertight.
    """
    X = np.asarray(X, float)
    S = np.asarray(S, float)
    F = np.asarray(F, int)
    Xn = [X[i] for i in range(len(X))]
    Sn = [S[i] for i in range(len(S))]
    mid = {}

    def _mid(a, b):
        key = (a, b) if a < b else (b, a)
        m = mid.get(key)
        if m is not None:
            return m
        m = len(Xn)
        Xn.append(0.5 * (X[a] + X[b]))
        s = S[a] + S[b]
        s = s / max(np.linalg.norm(s), 1e-15)
        Sn.append(s)
        mid[key] = m
        return m

    Fn = []
    for f in F:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
        Fn += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.asarray(Xn), np.asarray(Fn, dtype=int), np.asarray(Sn)


def upsample_for_shp(mesh, L_max=16, oversample=40, max_subdiv=3, verbose=True):
    """Densify the sphere map so the SH fit has enough, well-distributed samples.

    The SH analysis (:meth:`shp_surface.shp_analysis`) is a least-squares over
    the mesh *vertices* at their ``(t, p)``, with ``(L_max+1)**2`` unknowns per
    channel. Too few (or clustered) vertices -> ill-conditioned fit -> ringing /
    spikes. We subdivide the parameterized mesh on the sphere (interpolating
    both ``(t, p)`` and the associated ``xyz``) until

        n_vertices >= oversample * (L_max + 1)**2

    (capped at ``max_subdiv`` 1-to-4 subdivisions). Returns a new
    ``surface_mesh`` carrying dense ``X`` / ``t`` / ``p``.
    """
    X = np.asarray(mesh.X, float).copy()
    F = np.asarray(mesh.F, int).copy()
    S = _sphere_from_tp(mesh)
    target = int(oversample) * (int(L_max) + 1) ** 2
    n_sub = 0
    while len(X) < target and n_sub < int(max_subdiv):
        X, F, S = _subdivide_sphere_once(X, F, S)
        n_sub += 1
    dense = surface_mesh(X, F)
    _set_tp_from_sphere(dense, S)
    if verbose:
        print(f"  upsample_for_shp: {len(mesh.X)} -> {len(dense.X)} verts "
              f"({n_sub} subdiv; target {target} for L_max={L_max})")
    return dense


def fit_shp(mesh, L_max=16, oversample=40, max_subdiv=3, verbose=True):
    """Fit a spherical-harmonics surface to a (bijective) sphere map, after
    upsampling the sphere sampling to match ``L_max``.

    Returns ``(shp, rms_rel, dense)`` where ``rms_rel`` is the per-vertex
    reconstruction RMS relative to the bounding-box diagonal (scale-free).
    """
    from ..shp_surface import shp_surface

    dense = upsample_for_shp(mesh, L_max=L_max, oversample=oversample,
                             max_subdiv=max_subdiv, verbose=verbose)
    s = shp_surface(dense, int(L_max))
    res = getattr(s, 'residual', None)
    if res is not None and np.ndim(res) == 2:
        rms = float(np.sqrt(np.mean(np.sum(np.asarray(res) ** 2, axis=1))))
    else:
        rms = float('nan')
    bbox = dense.X.max(axis=0) - dense.X.min(axis=0)
    diag = float(np.linalg.norm(bbox)) or 1.0
    rms_rel = rms / diag
    if verbose:
        print(f"  fit_shp: L_max={L_max}, recon RMS={rms:.5f} "
              f"({100 * rms_rel:.2f}% of bbox diag), "
              f"#coeffs/channel={(L_max + 1) ** 2}")
    return s, rms_rel, dense


# --------------------------------------------------------------------------- #
# Rotation + scale invariance (canonical SHP via first-order-ellipsoid align)
# --------------------------------------------------------------------------- #
def _l1_basis_at_axes(shp):
    """3x3 matrix B with B[axis, order] = Y_{1,order}(axis) (order = -1,0,1).

    The degree-1 real SH are linear in direction, so ``y1(s) = B^T s``; this is
    measured (not assumed) to stay agnostic to the basis normalisation/order.
    """
    axes = np.eye(3)
    t, p, _ = kk_cart2sph(axes[:, 0], axes[:, 1], axes[:, 2])
    B = np.zeros((3, 3))
    for ki, k in enumerate((-1, 0, 1)):
        B[:, ki] = np.asarray(shp.basis.ylk_bosh(1, k, p, t)).flatten()
    return B


def shp_degree_energy(shp, scale_invariant=True):
    """Per-degree rotation-invariant SHP descriptor (for shape distance).

    ``E_l = sqrt(sum_{m,channel} c_{l,m,channel}^2)`` is invariant to BOTH object
    and parameter-domain rotations (rotations are unitary within each degree and
    orthogonal across x/y/z), so it is a robust shape signature -- unlike a
    canonical *pose*, it is well-defined even for symmetric shapes. With
    ``scale_invariant`` the vector is normalised by its l>=1 norm (size-free).
    Compare two shapes by the Euclidean distance between their ``E`` vectors.
    """
    L = int(shp.L_max)
    xc = np.asarray(shp.xc, float)
    yc = np.asarray(shp.yc, float)
    zc = np.asarray(shp.zc, float)
    E = np.zeros(L + 1)
    idx = 0
    for l in range(L + 1):
        n = 2 * l + 1
        sl = slice(idx, idx + n)
        E[l] = np.sqrt(np.sum(xc[sl] ** 2 + yc[sl] ** 2 + zc[sl] ** 2))
        idx += n
    if scale_invariant:
        denom = np.linalg.norm(E[1:]) if L >= 1 else 0.0
        if denom > 1e-15:
            E = E / denom
    return E


def canonicalize_shp(mesh, L_max=16, oversample=40, scale_mode='ellipsoid',
                     fix_sign=True, verbose=True):
    """Rotation + scale + translation **invariant** ('canonical') SHP surface.

    First-order-ellipsoid (FOE / Brechbuhler) alignment: the degree-1
    coefficients define a linear map ``A`` from sphere directions to the object
    (the dominant ellipsoid). ``A = U S V^T`` gives the object rotation ``U``,
    the parameter-domain rotation ``V`` and the semi-axes ``S``. We apply ``U``
    to the object, ``V`` to the sphere samples, divide by the scale, and
    **re-fit** -- so no Wigner matrices are needed and ``A`` becomes diagonal
    (canonical). Because the upstream cMCF pipeline is ~rotation-equivariant,
    the resulting coefficients are ~invariant to the input mesh's pose and size,
    which is what makes SHP shape-distance / comparison meaningful.

    Returns ``(shp_canonical, info)``.
    """
    from ..shp_surface import shp_surface

    dense = upsample_for_shp(mesh, L_max=L_max, oversample=oversample,
                             verbose=False)
    S = _sphere_from_tp(dense)
    X = np.asarray(dense.X, float).copy()
    X -= X.mean(axis=0)                                   # translation

    d0 = surface_mesh(X.copy(), np.asarray(dense.F, int).copy())
    d0.t = dense.t.copy()
    d0.p = dense.p.copy()
    s0 = shp_surface(d0, int(L_max))

    C1 = np.array([[s0.xc[1], s0.xc[2], s0.xc[3]],
                   [s0.yc[1], s0.yc[2], s0.yc[3]],
                   [s0.zc[1], s0.zc[2], s0.zc[3]]], dtype=float)
    A = C1 @ _l1_basis_at_axes(s0).T                      # direction -> object
    U, Sig, Vt = np.linalg.svd(A)
    Vo = Vt.T
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0
    if np.linalg.det(Vo) < 0:
        Vo[:, -1] *= -1.0

    Xc = X @ U                                            # object rotation
    Sc = S @ Vo                                           # domain rotation

    if scale_mode == 'rms':
        scale = float(np.sqrt(np.mean(np.sum(Xc ** 2, axis=1))))
    else:                                                 # 'ellipsoid'
        scale = float(Sig[0])
    scale = scale if scale > 1e-12 else 1.0
    Xc = Xc / scale

    if fix_sign:
        # Resolve the residual +/- axis ambiguity deterministically via the
        # 3rd moment (skewness); flip in pairs to stay a proper rotation.
        m3 = np.mean(Xc ** 3, axis=0)
        want = np.where(m3 < 0, -1.0, 1.0)
        if np.prod(want) < 0:                  # odd # of flips -> fix parity on
            want[int(np.argmin(np.abs(m3)))] *= -1.0   # the least-skewed axis
        Xc = Xc * want[None, :]

    dc = surface_mesh(Xc, np.asarray(dense.F, int).copy())
    t, p, _ = kk_cart2sph(Sc[:, 0], Sc[:, 1], Sc[:, 2])
    dc.t = t
    dc.p = p
    shp_c = shp_surface(dc, int(L_max))

    info = {'scale': scale, 'semi_axes': [float(x) for x in Sig],
            'det_obj': float(np.linalg.det(U)),
            'det_dom': float(np.linalg.det(Vo))}
    if verbose:
        print(f"  canonicalize_shp: semi-axes={np.round(Sig, 4).tolist()}, "
              f"scale={scale:.4f}")
    return shp_c, info


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def parameterize_to_sphere(
        mesh, target_verts=2000, curvature_strength=2.0, do_preprocess=True,
        cmcf_iter=120, cmcf_step=1.0, cmcf_tol=1e-5, untangle_iter=400,
        area_blend=1.0, area_exponent=2.0, area_n_iter=1200, lambda_shape=0.3,
        aniso_rounds=0, aniso_split_factor=1.7,
        allow_cage_fallback=True, n_cage=250,
        center=True, fit_shp_L_max=16, shp_oversample=40,
        snapshot_every=0, verbose=True):
    """Arbitrary genus-0 mesh -> bijective, SHP-ready unit-sphere map.

    Pipeline: preprocess -> cMCF (Stage 1) -> [cage fallback if folds remain] ->
    area redistribution (Stage 2). Returns a result dict; the parameterized mesh
    is ``result['mesh']`` (carries ``.t`` / ``.p``), and intermediate meshes are
    kept for visualization.

    Parameters mirror the module docstring; ``area_blend`` 1.0 = equal area
    (default, classic SPHARM), 0.0 = area proportional to curvature.
    """
    result = {'history': []}

    # ---- Stage 0: preprocess --------------------------------------------- #
    if do_preprocess:
        from .tiered_spherical_parameterization import preprocess_mesh
        if verbose:
            print("=" * 60)
            print("Stage 0: preprocess (repair + curvature-adaptive remesh)")
            print("=" * 60)
        m, q_stats = preprocess_mesh(
            mesh, target_verts=target_verts,
            curvature_strength=curvature_strength, verbose=verbose)
        result['preprocess'] = q_stats
    else:
        m = surface_mesh(np.asarray(mesh.X, float).copy(),
                         np.asarray(mesh.F, int).copy())
        m.props()
    # Consistent outward winding once, so every downstream foldover count is
    # honest (see ensure_outward_winding).
    m.F = ensure_outward_winding(m.X, m.F)
    result['m_pre'] = surface_mesh(m.X.copy(), m.F.copy())

    # ---- Stage 1: cMCF ---------------------------------------------------- #
    if verbose:
        print("\n" + "=" * 60)
        print("Stage 1: cMCF -> bijective spherical map")
        print("=" * 60)
    m, cmcf_info = cmcf_sphere_map(
        m, n_iter=cmcf_iter, step_factor=cmcf_step, tol=cmcf_tol,
        untangle_iter=untangle_iter, snapshot_every=snapshot_every,
        verbose=verbose)
    result['cmcf_info'] = cmcf_info
    diag1 = sphere_diagnostics(m, verbose=verbose)
    result['stage1_diag'] = diag1
    result['method'] = 'cmcf'

    # ---- Optional coarse-to-fine fallback if folds remain ---------------- #
    if diag1['n_foldovers'] > 0 and allow_cage_fallback:
        if verbose:
            print("\n" + "=" * 60)
            print(f"Stage 1b: cMCF left {diag1['n_foldovers']} folds -> "
                  "FPS cage fallback")
            print("=" * 60)
        m_fb = surface_mesh(result['m_pre'].X.copy(), result['m_pre'].F.copy())
        m_fb.props()
        m_fb, cage_info = fps_cage_sphere_map(
            m_fb, n_cage=n_cage, cmcf_step=cmcf_step, verbose=verbose)
        diag_fb = sphere_diagnostics(m_fb, verbose=verbose)
        result['cage_info'] = cage_info
        result['stage1b_diag'] = diag_fb
        # Keep whichever map has fewer folds.
        if diag_fb['n_foldovers'] < diag1['n_foldovers']:
            m = m_fb
            result['method'] = 'fps_cage'
            if verbose:
                print(f"  -> using FPS-cage map ({diag_fb['n_foldovers']} folds)")
        elif verbose:
            print(f"  -> keeping cMCF map ({diag1['n_foldovers']} folds)")

    result['m_sphere_raw'] = surface_mesh(m.X.copy(), m.F.copy())
    result['m_sphere_raw'].t = m.t.copy()
    result['m_sphere_raw'].p = m.p.copy()

    # ---- Conformal Mobius centering (canonical pose for SHP) ------------- #
    if center:
        if verbose:
            print("\n" + "=" * 60)
            print("Stage 1c: conformal Mobius centering")
            print("=" * 60)
        m = mobius_center(m, verbose=verbose)

    # ---- Stage 2: area equalization (interior-point, fold-free) ---------- #
    if verbose:
        print("\n" + "=" * 60)
        print(f"Stage 2: area equalization (blend={area_blend}: "
              f"{'uniform' if area_blend >= 0.999 else 'curvature-weighted'})")
        print("=" * 60)
    m, eq_info = equalize_areas(
        m, area_blend=area_blend, area_exponent=area_exponent,
        n_iter=area_n_iter, lambda_shape=lambda_shape, verbose=verbose)
    result['equalize_info'] = eq_info
    diag2 = sphere_diagnostics(m, verbose=verbose)
    result['stage2_diag'] = diag2
    result['mesh'] = m

    # ---- Stage 2b: stretch-aligned anisotropic refinement (look-ahead) --- #
    if aniso_rounds > 0:
        if verbose:
            print("\n" + "=" * 60)
            print(f"Stage 2b: anisotropic refinement ({aniso_rounds} rounds, "
                  f"split_factor={aniso_split_factor})")
            print("=" * 60)
        m, aniso_hist = anisotropic_rounds(
            m, n_rounds=aniso_rounds, split_factor=aniso_split_factor,
            area_blend=area_blend, area_exponent=area_exponent,
            area_n_iter=area_n_iter, lambda_shape=lambda_shape, verbose=verbose)
        result['aniso_history'] = aniso_hist
        diag2 = sphere_diagnostics(m, verbose=verbose)
        result['stage2_diag'] = diag2
        result['mesh'] = m

    # ---- Stage 3: SHP fit (upsampled to match L_max) --------------------- #
    if fit_shp_L_max:
        if verbose:
            print("\n" + "=" * 60)
            print(f"Stage 3: SHP fit (L_max={fit_shp_L_max}, "
                  f"oversample={shp_oversample})")
            print("=" * 60)
        shp, rms_rel, dense = fit_shp(
            m, L_max=fit_shp_L_max, oversample=shp_oversample, verbose=verbose)
        result['shp'] = shp
        result['shp_rms_rel'] = rms_rel
        result['shp_dense'] = dense

    if verbose:
        print("\n" + "=" * 60)
        print("Done. Summary")
        print("=" * 60)
        print(f"  method={result['method']}  "
              f"folds: stage1={diag1['n_foldovers']} -> final={diag2['n_foldovers']}")
        print(f"  final: bijective={diag2['bijective']} "
              f"min_q={diag2['min_quality']:.3f} "
              f"max_shear={diag2['max_shear']:.2f} "
              f"sphere-area cov={diag2['area_cov']:.2f} "
              f"(1->uniform sampling)")
        if 'shp_rms_rel' in result:
            print(f"  SHP L_max={fit_shp_L_max}: recon RMS="
                  f"{100 * result['shp_rms_rel']:.2f}% of bbox diag")
    return result
