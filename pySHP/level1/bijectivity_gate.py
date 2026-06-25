"""
Bijectivity quality gate for spherical parameterizations.
"""

import numpy as np


def _sph2cart(t, p):
    x = np.sin(t) * np.cos(p)
    y = np.sin(t) * np.sin(p)
    z = np.cos(t)
    return np.column_stack([x, y, z])


def check_bijectivity_gate(ms, area_tol=0.01, centroid_tol=0.15,
                           max_edge_ratio=50.0, verbose=False):
    """Check whether a spherical parameterization is approximately bijective.

    Parameters
    ----------
    ms : surface_mesh
        Mesh with ``.t``, ``.p``, ``.F`` set.
    area_tol : float
        Relative tolerance on |sum(signed areas) - 4*pi| / 4*pi.
    centroid_tol : float
        Maximum allowed distance of vertex centroid from origin.
    max_edge_ratio : float
        Maximum allowed ratio of longest to shortest spherical edge.
    verbose : bool
        Print summary.

    Returns
    -------
    report : dict
        Keys: ``passed``, ``issues``, ``signed_total_area``, ``area_excess_rel``,
        ``n_foldovers``, ``centroid_dist``, ``edge_ratio``.
    """
    t = np.asarray(ms.t, dtype=float)
    p = np.mod(np.asarray(ms.p, dtype=float), 2.0 * np.pi)
    F = np.asarray(ms.F, dtype=int)
    X_sph = _sph2cart(t, p)

    signed_areas = np.zeros(len(F))
    n_foldovers = 0
    for fi in range(len(F)):
        v0, v1, v2 = X_sph[F[fi, 0]], X_sph[F[fi, 1]], X_sph[F[fi, 2]]
        orient = np.dot(v0, np.cross(v1, v2))
        signed_areas[fi] = 0.5 * orient
        if orient < 0:
            n_foldovers += 1

    four_pi = 4.0 * np.pi
    signed_total = np.sum(signed_areas)
    area_excess_rel = abs(signed_total - four_pi) / four_pi

    centroid = np.mean(X_sph, axis=0)
    centroid_dist = float(np.linalg.norm(centroid))

    edge_lengths = []
    edges = set()
    for fi in range(len(F)):
        for i in range(3):
            a, b = sorted((int(F[fi, i]), int(F[fi, (i + 1) % 3])))
            if (a, b) not in edges:
                edges.add((a, b))
                va, vb = X_sph[a], X_sph[b]
                cos_a = np.clip(np.dot(va, vb), -1.0, 1.0)
                edge_lengths.append(np.arccos(cos_a))
    edge_lengths = np.asarray(edge_lengths) if edge_lengths else np.array([1.0])
    edge_ratio = float(edge_lengths.max() / max(edge_lengths.min(), 1e-12))

    issues = []
    if area_excess_rel > area_tol:
        issues.append(
            f"signed total area deviates by {100 * area_excess_rel:.1f}% from 4*pi"
        )
    if n_foldovers > 0:
        issues.append(f"{n_foldovers} foldover faces (negative orientation)")
    if centroid_dist > centroid_tol:
        issues.append(f"vertex centroid far from origin ({centroid_dist:.4f})")
    if edge_ratio > max_edge_ratio:
        issues.append(f"edge length ratio {edge_ratio:.1f} exceeds {max_edge_ratio}")

    passed = len(issues) == 0
    report = {
        'passed': passed,
        'issues': issues,
        'signed_total_area': float(signed_total),
        'area_excess_rel': float(area_excess_rel),
        'n_foldovers': int(n_foldovers),
        'centroid_dist': centroid_dist,
        'edge_ratio': edge_ratio,
        'abs_total_area': float(np.sum(np.abs(signed_areas))),
    }

    if verbose:
        status = 'PASSED' if passed else 'FAILED'
        print(f"Bijectivity gate: {status}")
        print(f"  signed total area: {signed_total:.4f} (4*pi={four_pi:.4f})")
        print(f"  area excess rel:   {100 * area_excess_rel:.2f}%")
        print(f"  foldovers:         {n_foldovers}")
        print(f"  centroid dist:     {centroid_dist:.4f}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")

    return report


def compute_achieved_spherical_areas(ms):
    """Return per-face signed and absolute spherical triangle areas."""
    from ..utils import kk_sph2cart

    t = np.asarray(ms.t, dtype=float)
    p = np.mod(np.asarray(ms.p, dtype=float), 2.0 * np.pi)
    F = np.asarray(ms.F, dtype=int)
    u, v, w = kk_sph2cart(t, p, np.ones(len(t)))
    X_sph = np.column_stack([u, v, w])

    signed = np.zeros(len(F))
    absolute = np.zeros(len(F))
    for fi in range(len(F)):
        v0, v1, v2 = X_sph[F[fi, 0]], X_sph[F[fi, 1]], X_sph[F[fi, 2]]
        orient = np.dot(v0, np.cross(v1, v2))
        signed[fi] = 0.5 * orient
        absolute[fi] = abs(orient) * 0.5
    return signed, absolute


def compute_parametric_quality(ms):
    """Compute parametric triangle quality on the sphere (equilateral metric)."""
    from ..utils import kk_sph2cart

    t = np.asarray(ms.t, dtype=float)
    p = np.mod(np.asarray(ms.p, dtype=float), 2.0 * np.pi)
    F = np.asarray(ms.F, dtype=int)
    u, v, w = kk_sph2cart(t, p, np.ones(len(t)))
    X_sph = np.column_stack([u, v, w])

    qualities = []
    aspect_ratios = []
    for fi in range(len(F)):
        verts = X_sph[F[fi]]
        edges = []
        for i in range(3):
            a, b = verts[i], verts[(i + 1) % 3]
            edges.append(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
        edges = np.asarray(edges)
        area = abs(np.dot(verts[0], np.cross(verts[1], verts[2]))) * 0.5
        denom = np.sum(edges ** 2)
        q = 4.0 * area * np.sqrt(3.0) / denom if denom > 1e-15 else 0.0
        qualities.append(q)
        aspect_ratios.append(edges.max() / max(edges.min(), 1e-12))

    qualities = np.asarray(qualities)
    aspect_ratios = np.asarray(aspect_ratios)
    return {
        'mean_quality': float(np.mean(qualities)),
        'min_quality': float(np.min(qualities)),
        'mean_aspect_ratio': float(np.mean(aspect_ratios)),
        'max_aspect_ratio': float(np.max(aspect_ratios)),
        'qualities': qualities,
        'aspect_ratios': aspect_ratios,
    }
