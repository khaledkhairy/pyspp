"""
Pre-map2sphere diagnostic for the simplified mesh.

Run BEFORE map2sphere() to verify that the mesh handed to the bijective
mapping is topologically and geometrically sound.  Catches winding
inconsistencies, degenerate faces, neighbour-list errors, and pole
selection issues that would produce a scrambled parameterization.
"""

import numpy as np
import datetime
from collections import defaultdict, deque


def _edge_key(a, b):
    return (min(int(a), int(b)), max(int(a), int(b)))


def diagnose_pre_map2sphere(ms, PM=None, verbose=True, output_file=None):
    """Comprehensive pre-bijective-mapping diagnostic.

    Parameters
    ----------
    ms : surface_mesh
        The simplified mesh that is about to be passed to ``map2sphere()``.
        Does **not** need ``.t`` or ``.p`` set yet.
    PM : dict, optional
        Patch-mesh structure (for per-patch reporting).
    verbose : bool
        Print the report to stdout as it is generated.
    output_file : str, optional
        Write the full report to this text file.

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
    log(f"PRE-MAP2SPHERE DIAGNOSTIC  |  {ts}")
    log("=" * 70)

    issues = []

    X = np.asarray(ms.X, dtype=float)
    F = np.asarray(ms.F, dtype=int)
    nV, nF = len(X), len(F)
    log(f"  Mesh: {nV} vertices, {nF} faces")

    # ==================================================================
    # 1. Basic topology
    # ==================================================================
    log(f"\n--- 1. Topology ---")

    edge_faces = defaultdict(list)
    for fi in range(nF):
        for i in range(3):
            e = _edge_key(F[fi][i], F[fi][(i + 1) % 3])
            edge_faces[e].append(fi)

    nE = len(edge_faces)
    n_boundary = sum(1 for flist in edge_faces.values() if len(flist) == 1)
    n_nonmanifold = sum(1 for flist in edge_faces.values() if len(flist) > 2)
    chi = nV - nE + nF
    genus = max(0, (2 - chi) // 2)

    log(f"  Edges: {nE}")
    log(f"  Euler char: {chi}  (expect 2 for genus-0)")
    log(f"  Genus: {genus}")
    log(f"  Boundary edges: {n_boundary}")
    log(f"  Non-manifold edges (>2 faces): {n_nonmanifold}")

    if chi != 2:
        issues.append(f"Euler characteristic {chi} (expect 2)")
    if n_boundary > 0:
        issues.append(f"{n_boundary} boundary edges")
    if n_nonmanifold > 0:
        issues.append(f"{n_nonmanifold} non-manifold edges")

    # Closed-mesh check
    expected_faces = 2 * nV - 4
    log(f"  Expected faces for closed genus-0: {expected_faces}, actual: {nF}")
    if nF != expected_faces:
        log(f"    (mismatch may be fine if genus != 0 or mesh not closed)")

    # ==================================================================
    # 2. Winding consistency (BFS)
    # ==================================================================
    log(f"\n--- 2. Face winding consistency (BFS propagation) ---")

    e2f = defaultdict(list)
    for fi in range(nF):
        for i in range(3):
            e = _edge_key(F[fi][i], F[fi][(i + 1) % 3])
            e2f[e].append(fi)

    visited = np.zeros(nF, dtype=bool)
    inconsistent_edges = []
    n_components = 0

    for seed in range(nF):
        if visited[seed]:
            continue
        n_components += 1
        queue = deque([seed])
        visited[seed] = True
        while queue:
            fi = queue.popleft()
            for i in range(3):
                a, b = int(F[fi][i]), int(F[fi][(i + 1) % 3])
                ekey = _edge_key(a, b)
                for fj in e2f[ekey]:
                    if fj == fi:
                        continue
                    if visited[fj]:
                        # Already visited: just check consistency
                        vj = list(F[fj])
                        if a in vj and b in vj:
                            ia, ib = vj.index(a), vj.index(b)
                            # Consistent: (a,b) in fi means (b,a) in fj
                            if (ib + 1) % 3 != ia:
                                inconsistent_edges.append((ekey, fi, fj))
                        continue
                    visited[fj] = True
                    vj = list(F[fj])
                    if a in vj and b in vj:
                        ia, ib = vj.index(a), vj.index(b)
                        if (ib + 1) % 3 != ia:
                            inconsistent_edges.append((ekey, fi, fj))
                    queue.append(fj)

    n_unvisited = int(np.sum(~visited))
    # Deduplicate (each inconsistent edge may be found from both sides)
    seen_pairs = set()
    unique_inconsistent = []
    for ekey, fi, fj in inconsistent_edges:
        pair = (min(fi, fj), max(fi, fj))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_inconsistent.append((ekey, fi, fj))

    log(f"  Face components: {n_components}")
    log(f"  Unvisited faces: {n_unvisited}")
    log(f"  Inconsistent winding edges: {len(unique_inconsistent)}")

    if len(unique_inconsistent) > 0:
        issues.append(f"{len(unique_inconsistent)} edges with inconsistent winding")
        log(f"  First 20 inconsistent edges:")
        for ekey, fi, fj in unique_inconsistent[:20]:
            log(f"    Edge {ekey}: faces {fi} & {fj}")

    # Global outward-normal check (signed volume)
    signed_vol = 0.0
    for fi in range(nF):
        v0, v1, v2 = X[F[fi][0]], X[F[fi][1]], X[F[fi][2]]
        signed_vol += np.dot(v0, np.cross(v1, v2))
    signed_vol_6 = signed_vol / 6.0
    orient_str = "outward" if signed_vol > 0 else "inward"
    log(f"  Signed volume: {signed_vol_6:.6f} -> normals point {orient_str}")
    if signed_vol < 0:
        issues.append("normals point inward (negative signed volume)")

    # ==================================================================
    # 3. Degenerate / near-degenerate faces
    # ==================================================================
    log(f"\n--- 3. Degenerate faces ---")
    areas = np.zeros(nF)
    for fi in range(nF):
        v0, v1, v2 = X[F[fi][0]], X[F[fi][1]], X[F[fi][2]]
        areas[fi] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    n_zero_area = int(np.sum(areas < 1e-15))
    n_tiny_area = int(np.sum(areas < 1e-8))
    log(f"  Zero-area faces (< 1e-15): {n_zero_area}")
    log(f"  Tiny-area faces (< 1e-8):  {n_tiny_area}")
    log(f"  Area range: [{areas.min():.6e}, {areas.max():.6e}]")
    log(f"  Area mean: {areas.mean():.6e}, std: {areas.std():.6e}")
    if n_zero_area > 0:
        issues.append(f"{n_zero_area} zero-area faces")

    # Duplicate vertices
    dists = np.zeros(nV)
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    pairs = tree.query_pairs(r=1e-10)
    if pairs:
        log(f"  WARNING: {len(pairs)} duplicate vertex pairs (dist < 1e-10)")
        issues.append(f"{len(pairs)} duplicate vertex pairs")
    else:
        log(f"  No duplicate vertices")

    # ==================================================================
    # 4. Neighbor list L
    # ==================================================================
    log(f"\n--- 4. Neighbor list (L) ---")
    # Compute L the same way edge_info() does
    L_sets = {ix: set() for ix in range(nV)}
    for fi in range(nF):
        v0, v1, v2 = int(F[fi][0]), int(F[fi][1]), int(F[fi][2])
        L_sets[v0].update((v1, v2))
        L_sets[v1].update((v0, v2))
        L_sets[v2].update((v0, v1))
    L = {k: list(v) for k, v in L_sets.items()}

    valences = np.array([len(L[i]) for i in range(nV)])
    log(f"  Valence range: [{valences.min()}, {valences.max()}]")
    log(f"  Valence mean: {valences.mean():.2f}")
    n_low_val = int(np.sum(valences < 3))
    n_high_val = int(np.sum(valences > 12))
    if n_low_val > 0:
        low_verts = np.where(valences < 3)[0]
        log(f"  WARNING: {n_low_val} vertices with valence < 3: {low_verts[:20].tolist()}")
        issues.append(f"{n_low_val} vertices with valence < 3")
    if n_high_val > 0:
        log(f"  {n_high_val} vertices with valence > 12")

    # Symmetry check
    n_asym = 0
    for v in range(nV):
        for u in L[v]:
            if v not in L[u]:
                n_asym += 1
    if n_asym > 0:
        log(f"  WARNING: {n_asym} asymmetric neighbor pairs")
        issues.append(f"{n_asym} asymmetric neighbor pairs in L")
    else:
        log(f"  Neighbor list is symmetric: OK")

    # Valence histogram
    log(f"\n  Valence histogram:")
    for val in range(int(valences.min()), min(int(valences.max()) + 1, 20)):
        cnt = int(np.sum(valences == val))
        if cnt > 0:
            bar = '#' * min(cnt, 50)
            log(f"    valence {val:>2d}: {cnt:>4d}  {bar}")

    # ==================================================================
    # 5. Pole selection preview
    # ==================================================================
    log(f"\n--- 5. Pole selection (get_graph preview) ---")
    try:
        ms_tmp = ms.__class__(X.copy(), F.copy())
        ms_tmp.edge_info()
        _, _, ixN, ixS, weights, ixN2, ixS2, _ = ms_tmp.get_graph()

        dist_3d = np.linalg.norm(X[ixN] - X[ixS])
        log(f"  Weighted poles:   ixN={ixN}, ixS={ixS}")
        log(f"    N position: {X[ixN].round(4)}")
        log(f"    S position: {X[ixS].round(4)}")
        log(f"    3D distance N-S: {dist_3d:.4f}")
        log(f"    N valence: {valences[ixN]}, S valence: {valences[ixS]}")

        dist_3d_uw = np.linalg.norm(X[ixN2] - X[ixS2])
        log(f"  Unweighted poles: ixN2={ixN2}, ixS2={ixS2}")
        log(f"    N2 position: {X[ixN2].round(4)}")
        log(f"    S2 position: {X[ixS2].round(4)}")
        log(f"    3D distance N2-S2: {dist_3d_uw:.4f}")

        # Check: are poles connected? (should have neighbors)
        if valences[ixN] < 3:
            log(f"  WARNING: North pole valence {valences[ixN]} < 3")
            issues.append(f"North pole has low valence ({valences[ixN]})")
        if valences[ixS] < 3:
            log(f"  WARNING: South pole valence {valences[ixS]} < 3")
            issues.append(f"South pole has low valence ({valences[ixS]})")

        # Simulate date-line path (steepest theta ascent from N to S)
        log(f"\n--- 5b. Date-line feasibility check ---")
        # Run latitude_calc
        t, _, _ = ms_tmp.__class__.latitude_calc(ms_tmp.L, ixN, ixS)
        log(f"  Theta computed: range [{t.min():.6f}, {t.max():.6f}]")
        log(f"  Theta at N (ixN={ixN}): {t[ixN]:.6f}")
        log(f"  Theta at S (ixS={ixS}): {t[ixS]:.6f}")

        # Trace date-line
        nbrs = np.asarray(ms_tmp.L[ixN])
        if len(nbrs) > 0:
            here = int(nbrs[-1])
            dtline = []
            maximum = 0.0
            counter = 0
            stuck = False
            while here != ixS:
                counter += 1
                if counter > nV:
                    stuck = True
                    break
                dtline.append(here)
                nbrs_h = np.asarray(ms_tmp.L[here])
                nextpos = None
                for ix in range(len(nbrs_h)):
                    if t[nbrs_h[ix]] > maximum:
                        maximum = t[nbrs_h[ix]]
                        nextpos = ix
                if nextpos is not None:
                    here = int(nbrs_h[nextpos])
                else:
                    stuck = True
                    break

            log(f"  Date-line path length: {len(dtline)} vertices")
            if stuck:
                log(f"  WARNING: Date-line got STUCK after {counter} steps "
                    f"(did not reach south pole)")
                issues.append("date-line path did not reach south pole")
            else:
                log(f"  Date-line reached south pole: OK")
                # Theta along date line should be monotonically increasing
                dt_thetas = t[dtline]
                mono_breaks = int(np.sum(np.diff(dt_thetas) < -1e-10))
                if mono_breaks > 0:
                    log(f"  WARNING: {mono_breaks} monotonicity breaks in "
                        f"date-line theta")
                else:
                    log(f"  Date-line theta monotonically increasing: OK")

            if len(dtline) > 0:
                log(f"  Date-line vertices (first 20): {dtline[:20]}")
                log(f"  Date-line theta (first 20): "
                    f"{[f'{t[v]:.4f}' for v in dtline[:20]]}")
        else:
            log(f"  WARNING: North pole has no neighbors")
            issues.append("North pole has no neighbors")

    except Exception as e:
        log(f"  Pole selection FAILED: {e}")
        issues.append(f"pole selection failed: {e}")

    # ==================================================================
    # 6. Per-patch face counts and winding (if PM available)
    # ==================================================================
    if PM is not None:
        pm_obj = PM.get('pm')
        fl = getattr(pm_obj, 'face_labels', None) if pm_obj is not None else None
        npatches = PM.get('npatches', 0)

        if fl is not None and npatches > 0:
            fl = np.asarray(fl, dtype=int)
            log(f"\n--- 6. Per-patch analysis ---")
            log(f"  Patches: {npatches}, face_labels unique: "
                f"{len(np.unique(fl))}")
            for pix in range(npatches):
                mask = (fl == pix)
                pf = np.where(mask)[0]
                if len(pf) == 0:
                    log(f"  Patch {pix}: NO FACES")
                    continue
                # Outward normal test per patch
                n_out = 0
                n_in = 0
                for fi in pf:
                    v0, v1, v2 = X[F[fi][0]], X[F[fi][1]], X[F[fi][2]]
                    c = (v0 + v1 + v2) / 3.0
                    nrm = np.cross(v1 - v0, v2 - v0)
                    if np.dot(nrm, c) > 0:
                        n_out += 1
                    else:
                        n_in += 1
                status = "OK" if n_in == 0 else f"MIXED ({n_in}/{len(pf)} inward)"
                log(f"  Patch {pix}: {len(pf)} faces, winding {status}")

    # ==================================================================
    # 7. Summary
    # ==================================================================
    valid = len(issues) == 0
    log(f"\n{'=' * 70}")
    if valid:
        log("RESULT: READY -- mesh appears suitable for bijective mapping")
    else:
        log(f"RESULT: ISSUES FOUND -- {len(issues)} problem(s):")
        for iss in issues:
            log(f"  - {iss}")
    log("=" * 70)

    report = {
        'valid': valid,
        'issues': issues,
        'nV': nV, 'nF': nF, 'nE': nE,
        'euler_characteristic': chi,
        'genus': genus,
        'n_boundary': n_boundary,
        'n_nonmanifold': n_nonmanifold,
        'n_inconsistent_winding': len(unique_inconsistent),
        'signed_volume': float(signed_vol_6),
        'n_zero_area_faces': n_zero_area,
        'n_face_components': n_components,
        'valence_range': (int(valences.min()), int(valences.max())),
    }

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(lines))
        if verbose:
            print(f"\nDiagnostic written to: {output_file}")

    return report
