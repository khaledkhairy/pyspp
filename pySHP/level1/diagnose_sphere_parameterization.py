"""
Comprehensive diagnostic for the spherical parameterization (result of parameterize_patches_cart).

Writes detailed structure of the mesh mapped onto the sphere: vertex positions,
per-patch faces, edge analysis, orientation checks. Run after parameterize_patches_cart()
to diagnose scrambled or invalid parameterization.
"""

import numpy as np
import datetime
from collections import defaultdict


def write_sphere_parameterization_diagnostic(PM, output_path=None):
    """
    Write a compact diagnostic of the sphere parameterization to a text file.
    Mirrors the format of simplified_mesh_diagnostic.txt for comparison.

    Parameters
    ----------
    PM : dict
        Patch structure with 'spm' (sphere mesh), 'Xkeyind', 'npatches'
    output_path : str, optional
        Path to write. If None, uses pySHP/tests/sphere_parameterization_diagnostic.txt
    """
    import os
    spm = PM.get('spm')
    if spm is None or spm.X is None or spm.F is None:
        return
    Xkeyind = np.asarray(PM.get('Xkeyind', [])).ravel()
    # nkeys = count of key+sentinel vertices (before first fictitious/center with Xkeyind=-1)
    if len(Xkeyind) > 0:
        fict = np.where(Xkeyind < 0)[0]
        nkeys = int(PM.get('nkeys', fict[0] if len(fict) > 0 else len(Xkeyind)))
    else:
        nkeys = 0
    npatches = PM.get('npatches', 0)
    fl = np.asarray(spm.face_labels).flatten() if spm.face_labels is not None else np.zeros(len(spm.F), dtype=int)

    if output_path is None:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests')
        os.makedirs(base, exist_ok=True)
        output_path = os.path.join(base, 'sphere_parameterization_diagnostic.txt')

    run_id = PM.get('run_id', 'unknown')
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append(f"Run ID: {run_id}  |  Timestamp: {ts}")
    lines.append("=" * 70)
    lines.append("SPHERE PARAMETERIZATION DIAGNOSTIC (initial from parameterize_patches_cart)")
    lines.append("=" * 70)
    lines.append(f"Vertices: {len(spm.X)}, Faces: {len(spm.F)}, nkeys: {nkeys}")
    lines.append(f"Xkeyind (first 40): {Xkeyind[:min(40, len(Xkeyind))].tolist()}")
    lines.append("")

    # Vertex positions on sphere (cartesian + theta, phi if available)
    lines.append("Vertex positions on sphere (simpl_idx: mX_idx -> [x, y, z], (theta, phi)):")
    for vi in range(min(len(spm.X), 50)):
        mx = int(Xkeyind[vi]) if vi < len(Xkeyind) and int(Xkeyind[vi]) >= 0 else 'fict'
        pos = spm.X[vi]
        r = np.linalg.norm(pos)
        label = 'key' if vi < nkeys else 'center/fict'
        t_str, p_str = '', ''
        if hasattr(spm, 't') and spm.t is not None and vi < len(spm.t):
            t_str = f", t={spm.t[vi]:.4f}"
        if hasattr(spm, 'p') and spm.p is not None and vi < len(spm.p):
            p_str = f", p={spm.p[vi]:.4f}"
        lines.append(f"  v{vi} (mX={mx}, {label}): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] r={r:.4f}{t_str}{p_str}")
    lines.append("")

    # Per-patch face listing
    for pix in range(npatches):
        p_faces = np.where(fl == pix)[0]
        lines.append(f"--- Patch {pix}: {len(p_faces)} faces ---")
        if len(PM.get('keys', [])) > 0:
            pk = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int).tolist()
            lines.append(f"  Keys (mX): {pk}")
        normals = []
        for fi in p_faces:
            f = spm.F[fi]
            mX = []
            for v in f:
                v = int(v)
                if v < len(Xkeyind) and int(Xkeyind[v]) >= 0:
                    mX.append(int(Xkeyind[v]))
                else:
                    mX.append(f'fict_{v}')
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            v1, v2, v3 = spm.X[a], spm.X[b], spm.X[c]
            n = np.cross(v2 - v1, v3 - v1)
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                n = n / nn
            normals.append(n)
            orient = np.dot(v1, np.cross(v2, v3))
            lines.append(f"  Face {fi}: simpl={f.tolist()}, mX={mX}, normal=[{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}], orient={orient:.4f}")
        if len(normals) >= 2:
            ref = normals[0]
            flipped = [i for i, n in enumerate(normals) if np.dot(n, ref) < 0]
            if flipped:
                lines.append(f"  WARNING: {len(flipped)} face normals flipped vs face 0: indices {flipped}")
            else:
                lines.append(f"  All {len(normals)} face normals consistent")
        lines.append("")

    # Per-patch angular span (geodesic distance between furthest key pair)
    lines.append("--- Per-patch angular span on sphere ---")
    any_overspread = False
    for pix in range(npatches):
        if len(PM.get('keys', [])) > 0:
            pk_mX = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int)
        else:
            pk_mX = np.array([], dtype=int)
        pk_simpl = []
        for ki in range(min(nkeys, len(Xkeyind))):
            if int(Xkeyind[ki]) in pk_mX:
                pk_simpl.append(ki)
        if len(pk_simpl) < 2:
            continue
        max_arc = 0.0
        worst_pair = (0, 0)
        positions = spm.X[pk_simpl]
        for i in range(len(pk_simpl)):
            for j in range(i + 1, len(pk_simpl)):
                pi = positions[i] / max(np.linalg.norm(positions[i]), 1e-15)
                pj = positions[j] / max(np.linalg.norm(positions[j]), 1e-15)
                arc = np.arccos(np.clip(np.dot(pi, pj), -1.0, 1.0))
                if arc > max_arc:
                    max_arc = arc
                    worst_pair = (pk_simpl[i], pk_simpl[j])
        status = ""
        if max_arc > np.pi * 0.75:
            status = " *** SEVERELY OVERSPREAD (>135°) ***"
            any_overspread = True
        elif max_arc > np.pi * 0.5:
            status = " ** OVERSPREAD (>90°) **"
            any_overspread = True
        lines.append(
            f"  Patch {pix}: max_arc={np.degrees(max_arc):.1f}° "
            f"(v{worst_pair[0]}–v{worst_pair[1]}){status}")
    if any_overspread:
        lines.append("  WARNING: Overspread patches will produce boundary "
                      "self-intersections and foldovers in fine mesh.")
        lines.append("  Consider: different segmentation, avoid_topological_annuli=True, "
                      "or more optimization iterations.")
    lines.append("")

    # Edge analysis
    edge_faces = defaultdict(list)
    for fi in range(len(spm.F)):
        f = spm.F[fi]
        for i in range(3):
            u, v = int(f[i]), int(f[(i + 1) % 3])
            e = (min(u, v), max(u, v))
            edge_faces[e].append(fi)
    nm = {e: flist for e, flist in edge_faces.items() if len(flist) > 2}
    boundary = {e: flist for e, flist in edge_faces.items() if len(flist) == 1}
    lines.append(f"Non-manifold edges: {len(nm)}")
    for e, flist in sorted(nm.items()):
        labels = [int(fl[fi]) for fi in flist]
        lines.append(f"  Edge {e}: {len(flist)} faces (patches {labels})")
    lines.append(f"Boundary edges: {len(boundary)}")
    for e, flist in sorted(boundary.items()):
        labels = [int(fl[fi]) for fi in flist]
        lines.append(f"  Edge {e}: patch {labels}")

    # Spherical orientation (all faces should point outward: orient > 0)
    orient_out = []
    degenerate = []
    for fi, f in enumerate(spm.F):
        v1, v2, v3 = spm.X[f[0]], spm.X[f[1]], spm.X[f[2]]
        o = np.dot(v1, np.cross(v2, v3))
        orient_out.append(o)
        if np.abs(o) < 1e-10:
            degenerate.append(fi)
    n_outward = sum(1 for o in orient_out if o > 1e-10)
    n_inward = sum(1 for o in orient_out if o < -1e-10)
    lines.append("")
    lines.append("Spherical orientation (orient = dot(v1, cross(v2,v3)); outward > 0):")
    lines.append(f"  Outward: {n_outward}, Inward: {n_inward}, Degenerate: {len(degenerate)}")
    if degenerate:
        lines.append(f"  Degenerate face indices: {degenerate[:20]}{'...' if len(degenerate) > 20 else ''}")

    lines.append("")
    lines.append("=" * 70)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return output_path


def diagnose_sphere_parameterization_full(PM, verbose=True, output_file=None):
    """
    Full diagnostic of the sphere parameterization topology and geometry.
    Mirrors diagnose_simplified_mesh_full format.

    Parameters
    ----------
    PM : dict
        Patch structure with 'spm'
    verbose : bool
        Print to stdout
    output_file : str, optional
        If set, write diagnostic to this file

    Returns
    -------
    report : dict
        Diagnostic report
    """
    lines = []

    def log(msg):
        lines.append(msg)
        if verbose:
            print(msg)

    spm = PM.get('spm')
    if spm is None or spm.X is None or spm.F is None:
        log("ERROR: PM['spm'] is None or missing X/F (run parameterize_patches_cart first)")
        return {'valid': False, 'error': 'no spm'}

    X, F = spm.X, spm.F
    nV, nF = len(X), len(F)
    fl = getattr(spm, 'face_labels', None)
    if fl is None:
        fl = np.zeros(nF, dtype=int)
    else:
        fl = np.asarray(fl).flatten()
    npatches = PM.get('npatches', int(np.max(fl)) + 1 if len(fl) > 0 else 0)

    log("=" * 70)
    log("SPHERE PARAMETERIZATION FULL DIAGNOSTIC")
    log("=" * 70)
    log(f"  Vertices: {nV}")
    log(f"  Faces: {nF}")
    log(f"  Patches: {npatches}")
    log(f"  Face labels range: {int(np.min(fl))} to {int(np.max(fl))}")

    # Check radius (all vertices should be on unit sphere)
    radii = np.linalg.norm(X, axis=1)
    r_min, r_max = float(np.min(radii)), float(np.max(radii))
    log(f"\n--- Radius check (should be 1.0) ---")
    log(f"  Min radius: {r_min:.6f}, Max radius: {r_max:.6f}")
    off_sphere = np.where((radii < 0.99) | (radii > 1.01))[0]
    if len(off_sphere) > 0:
        log(f"  WARNING: {len(off_sphere)} vertices off unit sphere: {off_sphere[:10].tolist()}...")

    # Spherical orientation (outward = orient > 0)
    orient_all = []
    for fi, f in enumerate(F):
        v1, v2, v3 = X[f[0]], X[f[1]], X[f[2]]
        o = np.dot(v1, np.cross(v2, v3))
        orient_all.append(o)
    orient_all = np.array(orient_all)
    n_outward = np.sum(orient_all > 1e-10)
    n_inward = np.sum(orient_all < -1e-10)
    n_degenerate = np.sum(np.abs(orient_all) <= 1e-10)
    log(f"\n--- Spherical orientation ---")
    log(f"  Outward (orient > 0): {n_outward}")
    log(f"  Inward (orient < 0): {n_inward}")
    log(f"  Degenerate (|orient| ~ 0): {n_degenerate}")
    if n_inward > 0:
        inward_fi = np.where(orient_all < -1e-10)[0]
        log(f"  Inward face indices (sample): {inward_fi[:15].tolist()}...")

    # Edge analysis
    edge_faces = defaultdict(list)
    for fix in range(nF):
        f = F[fix]
        pix = int(fl[fix])
        for i in range(3):
            v1, v2 = int(f[i]), int(f[(i + 1) % 3])
            e = (min(v1, v2), max(v1, v2))
            edge_faces[e].append((fix, pix))
    nE = len(edge_faces)
    boundary_edges = [(e, flist) for e, flist in edge_faces.items() if len(flist) == 1]
    non_manifold = [(e, flist) for e, flist in edge_faces.items() if len(flist) != 2]

    log(f"\n--- Edge analysis ---")
    log(f"  Total edges: {nE}")
    log(f"  Boundary edges: {len(boundary_edges)}")
    log(f"  Non-manifold edges: {len(non_manifold)}")
    if boundary_edges:
        for e, flist in boundary_edges[:10]:
            fix, pix = flist[0]
            log(f"    Edge {e} -> face {fix}, patch {pix}")
    if non_manifold:
        nm_gt2 = [(e, flist) for e, flist in non_manifold if len(flist) > 2]
        if nm_gt2:
            log(f"  Non-manifold (>2 faces):")
            for e, flist in nm_gt2[:10]:
                patches = [p for _, p in flist]
                log(f"    Edge {e} -> {len(flist)} faces, patches {patches}")

    # Euler
    chi = nV - nE + nF
    log(f"\n--- Topology ---")
    log(f"  Euler characteristic: {chi} (expect 2 for closed sphere)")
    if chi != 2:
        log(f"  WARNING: Not a topological sphere (boundary/non-manifold present)")

    # Per-patch orientation consistency
    log(f"\n--- Per-patch orientation ---")
    for pix in range(npatches):
        patch_fi = np.where(fl == pix)[0]
        if len(patch_fi) == 0:
            log(f"  Patch {pix}: NO FACES")
            continue
        o_patch = orient_all[patch_fi]
        n_out = np.sum(o_patch > 1e-10)
        n_in = np.sum(o_patch < -1e-10)
        n_deg = np.sum(np.abs(o_patch) <= 1e-10)
        if n_in > 0 or n_deg > 0:
            log(f"  Patch {pix}: {len(patch_fi)} faces, outward={n_out}, inward={n_in}, degenerate={n_deg}")
        else:
            log(f"  Patch {pix}: {len(patch_fi)} faces, all outward")

    # Summary
    issues = []
    if r_min < 0.9 or r_max > 1.1:
        issues.append("Vertices off unit sphere")
    if n_inward > 0:
        issues.append(f"{n_inward} inward-facing faces")
    if n_degenerate > 0:
        issues.append(f"{n_degenerate} degenerate faces")
    if len(boundary_edges) > 0:
        issues.append(f"{len(boundary_edges)} boundary edges")
    if len(non_manifold) > 0:
        issues.append(f"{len(non_manifold)} non-manifold edges")
    if chi != 2:
        issues.append(f"Euler χ={chi} (need 2)")

    valid = len(issues) == 0
    log(f"\n{'=' * 70}")
    if valid:
        log("RESULT: VALID — sphere parameterization ready for optimization")
    else:
        log(f"RESULT: INVALID — {len(issues)} issue(s):")
        for iss in issues:
            log(f"  - {iss}")
    log("=" * 70)

    report = {
        'valid': valid,
        'nV': nV, 'nE': nE, 'nF': nF,
        'euler_characteristic': chi,
        'n_outward': n_outward, 'n_inward': n_inward, 'n_degenerate': n_degenerate,
        'n_boundary_edges': len(boundary_edges),
        'n_non_manifold': len(non_manifold),
        'r_min': r_min, 'r_max': r_max,
        'issues': issues,
    }

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        if verbose:
            print(f"\nSphere parameterization diagnostic written to: {output_file}")

    return report
