"""
Comprehensive diagnostic for simplified mesh: finds boundary edges (gaps), non-manifold edges,
genus issues, and identifies which patches are responsible.

Run after generate_simplified_mesh() and before map2sphere() to catch all topology issues.
"""

import numpy as np
import datetime
from collections import defaultdict


def diagnose_simplified_mesh_full(PM, verbose=True, output_file=None):
    """
    Full diagnostic of the simplified mesh topology.
    
    Checks:
    1. Every patch has faces
    2. No floating vertices
    3. Every edge shared by exactly 2 faces (manifold)
    4. No boundary edges (closed mesh)
    5. Genus = 0 (topological sphere)
    6. Single connected component
    7. Identifies which patches contribute to boundary edges (gaps)
    
    Parameters
    ----------
    PM : dict
        Patch mesh structure with 'pm'.
    verbose : bool
        Print to stdout.
    output_file : str, optional
        If set, write diagnostic to this file.
    
    Returns
    -------
    report : dict
        Full diagnostic report.
    """
    lines = []
    def log(msg):
        lines.append(msg)
        if verbose:
            print(msg)
    
    pm = PM.get('pm')
    if pm is None or pm.X is None or pm.F is None:
        log("ERROR: PM['pm'] is None or missing X/F")
        return {'valid': False, 'error': 'no mesh'}
    
    X, F = pm.X, pm.F
    nV, nF = len(X), len(F)
    fl = getattr(pm, 'face_labels', None)
    if fl is None:
        fl = np.zeros(nF, dtype=int)
    else:
        fl = np.asarray(fl).flatten()
    npatches = PM.get('npatches', int(np.max(fl)) + 1 if len(fl) > 0 else 0)
    
    run_id = PM.get('run_id', 'unknown')
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log(f"Run ID: {run_id}  |  Timestamp: {ts}")
    log("=" * 70)
    log("SIMPLIFIED MESH FULL DIAGNOSTIC")
    log("=" * 70)
    log(f"  Vertices: {nV}")
    log(f"  Faces: {nF}")
    log(f"  Patches: {npatches}")
    log(f"  Face labels range: {int(np.min(fl))} to {int(np.max(fl))}")
    
    # ---- Check 1: Every patch has faces ----
    patches_with_faces = set(fl.tolist())
    patches_missing = sorted(set(range(npatches)) - patches_with_faces)
    faces_per_patch = {pix: int(np.sum(fl == pix)) for pix in range(npatches)}
    log(f"\n--- Patch face counts ---")
    for pix in range(npatches):
        log(f"  Patch {pix}: {faces_per_patch.get(pix, 0)} faces")
    if patches_missing:
        log(f"  WARNING: Patches with NO faces: {patches_missing}")
    
    # ---- Check 2: Floating vertices ----
    vertex_in_face = np.zeros(nV, dtype=bool)
    for f in F:
        for v in f:
            if 0 <= v < nV:
                vertex_in_face[int(v)] = True
    floating = np.where(~vertex_in_face)[0]
    if len(floating) > 0:
        log(f"\n  WARNING: {len(floating)} floating vertices: {floating.tolist()[:20]}{'...' if len(floating) > 20 else ''}")
    else:
        log(f"\n  OK: No floating vertices")
    
    # ---- Check 3 & 4: Edge analysis (manifold, boundary) ----
    edge_faces = defaultdict(list)  # edge -> list of (face_idx, patch_idx)
    for fix in range(nF):
        f = F[fix]
        pix = int(fl[fix])
        for i in range(3):
            v1, v2 = int(f[i]), int(f[(i + 1) % 3])
            e = (min(v1, v2), max(v1, v2))
            edge_faces[e].append((fix, pix))
    
    nE = len(edge_faces)
    boundary_edges = []  # edges with exactly 1 face
    non_manifold_edges = []  # edges with != 2 faces
    for e, flist in edge_faces.items():
        if len(flist) == 1:
            boundary_edges.append((e, flist))
        if len(flist) != 2:
            non_manifold_edges.append((e, flist))
    
    log(f"\n--- Edge analysis ---")
    log(f"  Total edges: {nE}")
    log(f"  Boundary edges (1 face): {len(boundary_edges)}")
    log(f"  Non-manifold edges (!=2 faces): {len(non_manifold_edges)}")
    
    if boundary_edges:
        # Identify which patches contribute to boundary edges
        boundary_patch_counts = defaultdict(int)
        log(f"\n  Boundary edges detail:")
        for e, flist in boundary_edges[:30]:
            fix, pix = flist[0]
            boundary_patch_counts[pix] += 1
            v1, v2 = e
            pos1 = ', '.join(f'{c:.4f}' for c in X[v1])
            pos2 = ', '.join(f'{c:.4f}' for c in X[v2])
            log(f"    Edge ({v1}, {v2}) -> face {fix}, patch {pix}, verts=[{pos1}], [{pos2}]")
        if len(boundary_edges) > 30:
            log(f"    ... and {len(boundary_edges) - 30} more")
        log(f"  Boundary edges per patch:")
        for pix in sorted(boundary_patch_counts.keys()):
            log(f"    Patch {pix}: {boundary_patch_counts[pix]} boundary edges")
    
    if non_manifold_edges:
        nm_not_boundary = [(e, flist) for e, flist in non_manifold_edges if len(flist) > 2]
        if nm_not_boundary:
            log(f"\n  Non-manifold edges (>2 faces) detail:")
            for e, flist in nm_not_boundary[:20]:
                patches = [pix for _, pix in flist]
                log(f"    Edge ({e[0]}, {e[1]}) -> {len(flist)} faces, patches {patches}")
    
    # ---- Check 5: Euler characteristic / genus ----
    chi = nV - nE + nF
    genus = max(0, (2 - chi) // 2)
    log(f"\n--- Topology ---")
    log(f"  Euler characteristic: {chi} (should be 2 for genus-0)")
    log(f"  Genus: {genus}")
    if chi != 2:
        log(f"  WARNING: Topology is NOT genus-0 sphere")
        log(f"  Need: V - E + F = 2, got: {nV} - {nE} + {nF} = {chi}")
        if len(boundary_edges) > 0:
            log(f"  Likely cause: {len(boundary_edges)} boundary edges (gaps in mesh)")
    
    # ---- Check 6: Connected components ----
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import connected_components
    adj = lil_matrix((nV, nV), dtype=bool)
    for f in F:
        for i in range(3):
            v1, v2 = int(f[i]), int(f[(i + 1) % 3])
            adj[v1, v2] = True
            adj[v2, v1] = True
    n_components, comp_labels = connected_components(adj.tocsr(), directed=False, return_labels=True)
    log(f"\n--- Connectivity ---")
    log(f"  Connected components: {n_components}")
    if n_components > 1:
        for ci in range(n_components):
            verts_in_comp = np.where(comp_labels == ci)[0]
            log(f"    Component {ci}: {len(verts_in_comp)} vertices")
    
    # ---- Per-patch detailed check ----
    log(f"\n--- Per-patch face winding check ---")
    for pix in range(npatches):
        patch_faces = np.where(fl == pix)[0]
        if len(patch_faces) == 0:
            log(f"  Patch {pix}: NO FACES")
            continue
        # Check face normals consistency
        normals = []
        for fix in patch_faces:
            f = F[fix]
            v0, v1, v2 = X[f[0]], X[f[1]], X[f[2]]
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                normals.append(n / norm)
        if len(normals) >= 2:
            ref = normals[0]
            flipped = sum(1 for n in normals[1:] if np.dot(n, ref) < 0)
            if flipped > 0:
                log(f"  Patch {pix}: {len(normals)} faces, {flipped} normals inconsistent with face 0")
            else:
                log(f"  Patch {pix}: {len(normals)} faces, normals consistent")
        else:
            log(f"  Patch {pix}: {len(patch_faces)} face(s)")
    
    # ---- Summary ----
    issues = []
    if patches_missing:
        issues.append(f"Missing faces for patches: {patches_missing}")
    if len(floating) > 0:
        issues.append(f"{len(floating)} floating vertices")
    if len(boundary_edges) > 0:
        issues.append(f"{len(boundary_edges)} boundary edges (gaps)")
    if len(non_manifold_edges) > 0:
        issues.append(f"{len(non_manifold_edges)} non-manifold edges")
    if chi != 2:
        issues.append(f"Genus {genus} (need 0)")
    if n_components > 1:
        issues.append(f"{n_components} disconnected components")
    
    valid = len(issues) == 0
    log(f"\n{'=' * 70}")
    if valid:
        log("RESULT: VALID — manifold, closed, genus-0, ready for bijective mapping")
    else:
        log(f"RESULT: INVALID — {len(issues)} issue(s):")
        for iss in issues:
            log(f"  - {iss}")
    log("=" * 70)
    
    report = {
        'valid': valid,
        'nV': nV, 'nE': nE, 'nF': nF,
        'euler_characteristic': chi,
        'genus': genus,
        'n_boundary_edges': len(boundary_edges),
        'n_non_manifold_edges': len(non_manifold_edges),
        'n_floating_vertices': len(floating),
        'n_components': n_components,
        'patches_missing_faces': patches_missing,
        'boundary_edges': boundary_edges,
        'non_manifold_edges': non_manifold_edges,
        'faces_per_patch': faces_per_patch,
        'issues': issues,
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(lines))
        if verbose:
            print(f"\nDiagnostic written to: {output_file}")
    
    return report
