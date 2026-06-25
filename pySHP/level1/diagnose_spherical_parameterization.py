"""
Diagnostic for the spherical parameterization of a simplified mesh.

Run AFTER map2sphere() (bijective mapping) to inspect the quality of the
initial theta/phi assignment on the unit sphere.  Produces a detailed text
report that helps debug scrambled / self-intersecting parameterizations.
"""

import numpy as np
import datetime
from collections import defaultdict


def _sph2cart(t, p):
    """Spherical (theta, phi) -> Cartesian on the unit sphere."""
    x = np.sin(t) * np.cos(p)
    y = np.sin(t) * np.sin(p)
    z = np.cos(t)
    return np.column_stack([x, y, z])


def _spherical_triangle_signed_area(v1, v2, v3):
    """Signed spherical area of a triangle on the unit sphere.

    Positive when vertices wind counter-clockwise as seen from outside.
    Uses the formula:  orient = dot(v1, cross(v2, v3))
    which equals 6 * signed volume of the tetrahedron (O, v1, v2, v3).
    """
    return np.dot(v1, np.cross(v2, v3))


def diagnose_spherical_parameterization(ms, PM=None, verbose=True,
                                         output_file=None):
    """Comprehensive diagnostic of the simplified mesh after map2sphere().

    Parameters
    ----------
    ms : surface_mesh
        The simplified mesh **after** ``map2sphere()`` has been called.
        Must have ``.t`` (theta) and ``.p`` (phi) arrays set, as well as
        ``.X``, ``.F``, and optionally ``.ixN``, ``.ixS``.
    PM : dict, optional
        The patch-mesh structure.  If supplied, per-patch analysis is
        performed and the report includes patch-level statistics.
    verbose : bool
        Print the report to stdout as it is generated.
    output_file : str, optional
        Write the complete report to this text file.

    Returns
    -------
    report : dict
        Keys include ``valid`` (bool), ``issues`` (list[str]), and all
        intermediate statistics.
    """
    lines = []
    def log(msg=""):
        lines.append(msg)
        if verbose:
            print(msg)

    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log("=" * 70)
    log(f"SPHERICAL PARAMETERIZATION DIAGNOSTIC  |  {ts}")
    log("=" * 70)

    issues = []

    X = np.asarray(ms.X, dtype=float)
    F = np.asarray(ms.F, dtype=int)
    nV, nF = len(X), len(F)

    t = np.asarray(ms.t, dtype=float) if ms.t is not None else None
    p = np.asarray(ms.p, dtype=float) if ms.p is not None else None

    if t is None or p is None:
        log("  ERROR: ms.t or ms.p is None -- map2sphere() has not been called.")
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as fout:
                fout.write('\n'.join(lines))
        return {'valid': False, 'issues': ['t or p is None']}

    log(f"  Mesh: {nV} vertices, {nF} faces")

    # ------------------------------------------------------------------
    # 1. Pole detection
    # ------------------------------------------------------------------
    ixN = getattr(ms, 'ixN', None)
    ixS = getattr(ms, 'ixS', None)
    log(f"\n--- Poles ---")
    if ixN is not None:
        log(f"  North pole (ixN={ixN}): X={X[ixN].round(4)}, theta={t[ixN]:.6f}, phi={p[ixN]:.6f}")
    else:
        log(f"  North pole: not set")
    if ixS is not None:
        log(f"  South pole (ixS={ixS}): X={X[ixS].round(4)}, theta={t[ixS]:.6f}, phi={p[ixS]:.6f}")
    else:
        log(f"  South pole: not set")

    # ------------------------------------------------------------------
    # 2. Theta / phi global statistics
    # ------------------------------------------------------------------
    log(f"\n--- Theta (latitude) statistics ---")
    log(f"  Range: [{t.min():.6f}, {t.max():.6f}]  (expect [0, pi])")
    log(f"  Mean: {t.mean():.6f},  Std: {t.std():.6f}")
    n_zero_t = int(np.sum(np.abs(t) < 1e-12))
    n_pi_t = int(np.sum(np.abs(t - np.pi) < 1e-12))
    n_out_t = int(np.sum((t < -1e-6) | (t > np.pi + 1e-6)))
    log(f"  Vertices at theta~0 (north): {n_zero_t}")
    log(f"  Vertices at theta~pi (south): {n_pi_t}")
    if n_out_t > 0:
        log(f"  WARNING: {n_out_t} vertices outside [0, pi]")
        issues.append(f"{n_out_t} theta values outside [0, pi]")

    log(f"\n--- Phi (longitude) statistics ---")
    log(f"  Range: [{p.min():.6f}, {p.max():.6f}]  (expect [0, 2*pi])")
    log(f"  Mean: {p.mean():.6f},  Std: {p.std():.6f}")
    n_zero_p = int(np.sum(np.abs(p) < 1e-12))
    log(f"  Vertices at phi~0: {n_zero_p}")
    n_out_p = int(np.sum((p < -1e-6) | (p > 2 * np.pi + 1e-6)))
    if n_out_p > 0:
        log(f"  WARNING: {n_out_p} vertices outside [0, 2*pi]")
        issues.append(f"{n_out_p} phi values outside [0, 2*pi]")

    # Histogram of theta (10 bins)
    log(f"\n--- Theta histogram (10 equal bins over [0, pi]) ---")
    bin_edges = np.linspace(0, np.pi, 11)
    counts, _ = np.histogram(t, bins=bin_edges)
    for i in range(10):
        bar = '#' * min(counts[i], 60)
        log(f"  [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {counts[i]:>4d}  {bar}")

    # ------------------------------------------------------------------
    # 3. Cartesian positions on the unit sphere
    # ------------------------------------------------------------------
    Xsph = _sph2cart(t, p)
    radii = np.linalg.norm(Xsph, axis=1)
    log(f"\n--- Unit-sphere Cartesian positions ---")
    log(f"  Radius range: [{radii.min():.8f}, {radii.max():.8f}] (expect ~1)")

    # ------------------------------------------------------------------
    # 4. Per-face spherical triangle analysis
    # ------------------------------------------------------------------
    log(f"\n--- Spherical triangle orientation (per face) ---")
    orients = np.zeros(nF, dtype=float)
    sph_areas = np.zeros(nF, dtype=float)
    for fi in range(nF):
        v0, v1, v2 = Xsph[F[fi][0]], Xsph[F[fi][1]], Xsph[F[fi][2]]
        orients[fi] = _spherical_triangle_signed_area(v0, v1, v2)

    sph_areas = np.abs(orients)
    n_pos = int(np.sum(orients > 0))
    n_neg = int(np.sum(orients < 0))
    n_degen = int(np.sum(np.abs(orients) < 1e-15))
    log(f"  Positive orientation (CCW from outside): {n_pos}")
    log(f"  Negative orientation (CW  from outside): {n_neg}")
    log(f"  Degenerate (|orient| < 1e-15):           {n_degen}")
    if n_neg > 0 and n_pos > 0:
        log(f"  >>> MIXED ORIENTATIONS: {n_neg} inverted faces out of {nF}")
        issues.append(f"{n_neg}/{nF} faces inverted on sphere")
    elif n_neg == nF:
        log(f"  >>> ALL faces inverted (all CW) -- global winding flip needed")
        issues.append("all faces inverted on sphere")
    elif n_pos == nF:
        log(f"  All faces consistently oriented (CCW) -- GOOD")

    # Identify the worst inverted faces
    if n_neg > 0:
        neg_idx = np.where(orients < 0)[0]
        worst = neg_idx[np.argsort(orients[neg_idx])[:min(10, len(neg_idx))]]
        log(f"\n  Worst inverted faces (most negative orient):")
        for fi in worst:
            v0i, v1i, v2i = F[fi]
            log(f"    Face {fi}: verts=[{v0i},{v1i},{v2i}], "
                f"theta=[{t[v0i]:.4f},{t[v1i]:.4f},{t[v2i]:.4f}], "
                f"phi=[{p[v0i]:.4f},{p[v1i]:.4f},{p[v2i]:.4f}], "
                f"orient={orients[fi]:.6e}")

    # Area statistics
    log(f"\n--- Spherical triangle area statistics ---")
    log(f"  Total |area|: {sph_areas.sum():.6f}  (expect ~4*pi = {4*np.pi:.6f})")
    log(f"  Min |area|: {sph_areas.min():.6e}")
    log(f"  Max |area|: {sph_areas.max():.6e}")
    log(f"  Mean |area|: {sph_areas.mean():.6e}")
    log(f"  Std  |area|: {sph_areas.std():.6e}")
    area_ratio = sph_areas.max() / max(sph_areas.min(), 1e-30)
    log(f"  Max/Min ratio: {area_ratio:.2f}")
    if sph_areas.sum() > 6 * np.pi:
        overlap_est = sph_areas.sum() - 4 * np.pi
        log(f"  >>> Total area exceeds 4*pi by {overlap_est:.4f} "
            f"({100 * overlap_est / (4 * np.pi):.1f}%) -- indicates overlap")
        issues.append(f"total spherical area exceeds 4*pi by {overlap_est:.4f}")

    # ------------------------------------------------------------------
    # 5. Edge-length analysis on the sphere
    # ------------------------------------------------------------------
    log(f"\n--- Spherical edge lengths ---")
    edge_lens = []
    edge_set = set()
    for fi in range(nF):
        for i in range(3):
            a, b = int(F[fi][i]), int(F[fi][(i + 1) % 3])
            ekey = (min(a, b), max(a, b))
            if ekey not in edge_set:
                edge_set.add(ekey)
                elen = np.linalg.norm(Xsph[a] - Xsph[b])
                edge_lens.append(elen)
    edge_lens = np.array(edge_lens)
    log(f"  Number of edges: {len(edge_lens)}")
    log(f"  Min: {edge_lens.min():.6e}")
    log(f"  Max: {edge_lens.max():.6e}")
    log(f"  Mean: {edge_lens.mean():.6e}")
    log(f"  Max/Min ratio: {edge_lens.max() / max(edge_lens.min(), 1e-30):.2f}")
    n_tiny = int(np.sum(edge_lens < 1e-6))
    if n_tiny > 0:
        log(f"  WARNING: {n_tiny} edges shorter than 1e-6 (near-degenerate)")
        issues.append(f"{n_tiny} near-degenerate edges on sphere")

    # ------------------------------------------------------------------
    # 6. Vertex spread analysis (are vertices clustered or spread?)
    # ------------------------------------------------------------------
    log(f"\n--- Vertex spread on sphere ---")
    centroid_sph = Xsph.mean(axis=0)
    centroid_sph_norm = np.linalg.norm(centroid_sph)
    log(f"  Centroid of sphere points: [{centroid_sph[0]:.4f}, "
        f"{centroid_sph[1]:.4f}, {centroid_sph[2]:.4f}]")
    log(f"  Centroid distance from origin: {centroid_sph_norm:.6f} "
        f"(expect ~0 for good spread)")
    if centroid_sph_norm > 0.3:
        log(f"  >>> Vertices clustered in one hemisphere (centroid bias {centroid_sph_norm:.4f})")
        issues.append(f"vertex centroid far from origin ({centroid_sph_norm:.4f})")

    # ------------------------------------------------------------------
    # 7. Per-vertex: list the first N vertices with theta, phi, Cartesian
    # ------------------------------------------------------------------
    log(f"\n--- Per-vertex theta/phi (first 30 and poles) ---")
    show_verts = list(range(min(30, nV)))
    if ixN is not None and ixN not in show_verts:
        show_verts.append(ixN)
    if ixS is not None and ixS not in show_verts:
        show_verts.append(ixS)
    for vi in show_verts:
        label = ""
        if vi == ixN:
            label = " [NORTH POLE]"
        elif vi == ixS:
            label = " [SOUTH POLE]"
        log(f"  v{vi}: theta={t[vi]:.6f}, phi={p[vi]:.6f}, "
            f"X_orig={X[vi].round(4)}, X_sph={Xsph[vi].round(4)}{label}")

    # ------------------------------------------------------------------
    # 8. Per-patch analysis (if PM supplied)
    # ------------------------------------------------------------------
    if PM is not None:
        pm = PM.get('pm')
        fl = getattr(pm, 'face_labels', None) if pm is not None else None
        if fl is None and pm is not None:
            fl = getattr(ms, 'face_labels', None)
        npatches = PM.get('npatches', 0)

        if fl is not None and npatches > 0:
            fl = np.asarray(fl, dtype=int)
            log(f"\n--- Per-patch spherical parameterization ---")
            for pix in range(npatches):
                mask = (fl == pix)
                pf = np.where(mask)[0]
                if len(pf) == 0:
                    log(f"  Patch {pix}: NO FACES")
                    continue
                p_orient = orients[pf]
                n_pp = int(np.sum(p_orient > 0))
                n_np = int(np.sum(p_orient < 0))
                p_area = sph_areas[pf]
                # Theta range of patch vertices
                pverts = np.unique(F[pf].flatten())
                t_range = (t[pverts].min(), t[pverts].max())
                p_range = (p[pverts].min(), p[pverts].max())
                status = "OK" if n_np == 0 else f"MIXED ({n_np}/{len(pf)} inverted)"
                log(f"  Patch {pix}: {len(pf)} faces, orient {status}, "
                    f"area_sum={p_area.sum():.5f}, "
                    f"theta=[{t_range[0]:.4f},{t_range[1]:.4f}], "
                    f"phi=[{p_range[0]:.4f},{p_range[1]:.4f}]")

    # ------------------------------------------------------------------
    # 9. Face winding consistency check (3D mesh vs sphere)
    # ------------------------------------------------------------------
    log(f"\n--- 3D mesh face winding vs sphere winding ---")
    n_3d_pos = 0
    n_3d_neg = 0
    n_mismatch = 0
    mismatch_faces = []
    for fi in range(nF):
        v0, v1, v2 = X[F[fi][0]], X[F[fi][1]], X[F[fi][2]]
        centroid_3d = (v0 + v1 + v2) / 3.0
        normal_3d = np.cross(v1 - v0, v2 - v0)
        orient_3d = np.dot(normal_3d, centroid_3d)
        if orient_3d > 0:
            n_3d_pos += 1
        else:
            n_3d_neg += 1
        # Check if 3D winding matches sphere winding
        if (orient_3d > 0) != (orients[fi] > 0):
            n_mismatch += 1
            if len(mismatch_faces) < 20:
                mismatch_faces.append(fi)

    log(f"  3D mesh: {n_3d_pos} outward, {n_3d_neg} inward")
    log(f"  Winding mismatches (3D vs sphere): {n_mismatch}/{nF}")
    if n_mismatch > 0:
        log(f"  First mismatched faces: {mismatch_faces}")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    valid = len(issues) == 0
    log(f"\n{'=' * 70}")
    if valid:
        log("RESULT: GOOD -- consistent orientations, no obvious issues")
    else:
        log(f"RESULT: ISSUES FOUND -- {len(issues)} problem(s):")
        for iss in issues:
            log(f"  - {iss}")
    log("=" * 70)

    report = {
        'valid': valid,
        'issues': issues,
        'nV': nV, 'nF': nF,
        'ixN': ixN, 'ixS': ixS,
        'theta_range': (float(t.min()), float(t.max())),
        'phi_range': (float(p.min()), float(p.max())),
        'n_positive_orient': n_pos,
        'n_negative_orient': n_neg,
        'n_degenerate': n_degen,
        'total_sph_area': float(sph_areas.sum()),
        'n_3d_outward': n_3d_pos,
        'n_3d_inward': n_3d_neg,
        'n_winding_mismatch': n_mismatch,
        'centroid_bias': float(centroid_sph_norm),
        'orientations': orients,
        'spherical_areas': sph_areas,
    }

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(lines))
        if verbose:
            print(f"\nDiagnostic written to: {output_file}")

    return report
