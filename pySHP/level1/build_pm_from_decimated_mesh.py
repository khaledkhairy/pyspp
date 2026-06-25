"""
Build a PM data structure using curvature-aware mesh decimation instead of
fan-triangulated simplified meshes.

The decimated mesh retains only original vertices (no interpolation) and
preserves more detail in high-curvature regions.  It replaces PM['pm']
produced by ``generate_simplified_mesh()`` while keeping the rest of PM
(keys, edges, sentinels, edge_dat, P, OUT_chain, CV) intact.

Typical usage
-------------
>>> from pySHP.level1.build_pm_from_decimated_mesh import build_pm_from_decimated_mesh
>>> PM = build_pm_from_decimated_mesh(m_seg, PM, target_ratio=0.08, verbose=True)
"""

import numpy as np
from collections import defaultdict, deque


def _ensure_manifold(m_seg, verbose=False):
    """Repair non-manifold / non-closed mesh so it can be decimated.

    Checks for:
    - Non-manifold edges (shared by >2 faces)
    - Boundary edges (shared by 1 face -- mesh not closed)

    Uses PyVista ``clean`` + ``fill_holes`` to resolve these.

    Returns the (possibly repaired) mesh.  ``m_seg`` is modified
    in-place and also returned for convenience.
    """
    F = np.asarray(m_seg.F, dtype=int)
    nf = len(F)

    ef = defaultdict(int)
    for fi in range(nf):
        for i in range(3):
            a, b = int(F[fi][i]), int(F[fi][(i + 1) % 3])
            ef[(min(a, b), max(a, b))] += 1
    n_nm = sum(1 for cnt in ef.values() if cnt > 2)
    n_bnd = sum(1 for cnt in ef.values() if cnt == 1)

    if n_nm == 0 and n_bnd == 0:
        if verbose:
            print("  Manifold check: OK (closed, 0 non-manifold edges)")
        return m_seg

    if verbose:
        print(f"  Manifold repair needed: {n_nm} non-manifold edges, "
              f"{n_bnd} boundary edges")

    try:
        import pyvista as pv

        saved_fl = None
        if m_seg.face_labels is not None:
            saved_fl = np.asarray(m_seg.face_labels).copy()

        faces_pv = np.hstack(
            [np.full((nf, 1), 3, dtype=int), F]).ravel()
        mesh = pv.PolyData(np.asarray(m_seg.X, dtype=float), faces_pv)

        if n_nm > 0:
            mesh = mesh.clean(
                tolerance=0.0,
                remove_unused_points=True,
                produce_merge_map=False,
            )

        for _fill_pass in range(5):
            raw_f = mesh.faces
            if raw_f is None or len(raw_f) == 0:
                break
            _ef_tmp = defaultdict(int)
            _F_tmp = raw_f.reshape(-1, 4)[:, 1:4]
            for _fi in range(len(_F_tmp)):
                for _i in range(3):
                    _a = int(_F_tmp[_fi][_i])
                    _b = int(_F_tmp[_fi][(_i + 1) % 3])
                    _ef_tmp[(min(_a, _b), max(_a, _b))] += 1
            _n_bnd_tmp = sum(1 for _c in _ef_tmp.values() if _c == 1)
            if _n_bnd_tmp == 0:
                break
            mesh = mesh.fill_holes(hole_size=1e6)

        new_X = np.array(mesh.points)
        raw_faces = mesh.faces
        if raw_faces is not None and len(raw_faces) > 0:
            new_F = np.array(raw_faces.reshape(-1, 4)[:, 1:4])
        else:
            new_F = F

        if verbose:
            print(f"    Before: {len(m_seg.X)} verts, {nf} faces")
            print(f"    After:  {len(new_X)} verts, {len(new_F)} faces")

        m_seg.X = new_X
        m_seg.F = new_F

        if saved_fl is not None:
            if len(new_F) == nf:
                m_seg.face_labels = saved_fl
            else:
                m_seg.face_labels = _remap_face_labels_after_clean(
                    F, new_F, saved_fl, m_seg.X)
                if verbose:
                    n_ul = len(np.unique(m_seg.face_labels))
                    print(f"    Face labels remapped: {n_ul} unique labels")

        m_seg.needs_edge_info = True
        m_seg.H = None

        ef2 = defaultdict(int)
        for fi in range(len(new_F)):
            for i in range(3):
                a, b = int(new_F[fi][i]), int(new_F[fi][(i + 1) % 3])
                ef2[(min(a, b), max(a, b))] += 1
        n_nm2 = sum(1 for cnt in ef2.values() if cnt > 2)
        n_bnd2 = sum(1 for cnt in ef2.values() if cnt == 1)
        if verbose:
            print(f"    After repair: {n_nm2} non-manifold, "
                  f"{n_bnd2} boundary edges")

    except ImportError:
        if verbose:
            print("  WARNING: pyvista not available, skipping manifold repair")

    return m_seg


def _remap_face_labels_after_clean(old_F, new_F, old_labels, new_X):
    """Map face labels from old to new faces after PyVista clean.

    Matches each new face to the old face with the same vertex set
    (allowing for vertex index remapping by clean's merge step).
    Falls back to centroid matching for unmatched faces.
    """
    n_new = len(new_F)
    new_labels = np.zeros(n_new, dtype=int)

    old_face_sets = {}
    for fi in range(len(old_F)):
        key = tuple(sorted(int(v) for v in old_F[fi]))
        old_face_sets[key] = int(old_labels[fi])

    unmatched = []
    for fi in range(n_new):
        key = tuple(sorted(int(v) for v in new_F[fi]))
        if key in old_face_sets:
            new_labels[fi] = old_face_sets[key]
        else:
            unmatched.append(fi)

    if unmatched:
        from scipy.spatial import cKDTree
        old_centroids = np.array([
            new_X[old_F[i]].mean(axis=0) if old_F[i].max() < len(new_X)
            else np.zeros(3)
            for i in range(len(old_F))
        ])
        tree = cKDTree(old_centroids)
        for fi in unmatched:
            c = new_X[new_F[fi]].mean(axis=0)
            _, idx = tree.query(c)
            new_labels[fi] = int(old_labels[idx])

    return new_labels


def _equalize_valences(mesh, target_valence=6, max_passes=20, verbose=False):
    """Improve mesh quality by flipping edges to move valences toward a target.

    For each interior edge shared by exactly two triangles, a flip is
    performed when it strictly reduces the sum of squared valence
    deviations from ``target_valence`` for the four affected vertices.

    Constraints
    -----------
    * Patch boundaries are never flipped (edges between faces with
      different ``face_labels``).
    * Flips that would create degenerate triangles (duplicate vertices
      in a face) are rejected.
    * Each pass iterates over all candidate edges; passes repeat until
      no more improvements are found or ``max_passes`` is reached.

    Parameters
    ----------
    mesh : surface_mesh
        Modified in place (``mesh.F`` is updated).
    target_valence : int
        Ideal vertex valence (6 for equilateral triangle meshes).
    max_passes : int
        Maximum number of full sweeps.
    verbose : bool
        Print progress.

    Returns
    -------
    total_flips : int
        Total number of edges flipped across all passes.
    """
    F = np.asarray(mesh.F, dtype=int)
    nf = len(F)
    fl = getattr(mesh, 'face_labels', None)
    if fl is not None:
        fl = np.asarray(fl, dtype=int)

    # Build edge -> face pair mapping (only interior edges)
    edge2pair = {}
    for fi in range(nf):
        for i in range(3):
            a, b = int(F[fi][i]), int(F[fi][(i + 1) % 3])
            ekey = (min(a, b), max(a, b))
            if ekey not in edge2pair:
                edge2pair[ekey] = [fi]
            else:
                edge2pair[ekey].append(fi)

    # Keep only interior edges (exactly 2 faces)
    interior_edges = {e: flist for e, flist in edge2pair.items()
                      if len(flist) == 2}

    # Compute initial valences
    nv = len(mesh.X)
    valence = np.zeros(nv, dtype=int)
    for fi in range(nf):
        for v in F[fi]:
            valence[int(v)] += 1

    def _dev(v):
        return (valence[v] - target_valence) ** 2

    total_flips = 0

    for pass_num in range(max_passes):
        flips_this_pass = 0

        # Rebuild interior edges each pass (connectivity changes)
        edge2pair.clear()
        for fi in range(nf):
            for i in range(3):
                a, b = int(F[fi][i]), int(F[fi][(i + 1) % 3])
                ekey = (min(a, b), max(a, b))
                if ekey not in edge2pair:
                    edge2pair[ekey] = [fi]
                else:
                    edge2pair[ekey].append(fi)
        interior_edges = [(e, flist) for e, flist in edge2pair.items()
                          if len(flist) == 2]

        for (a, b), (f0, f1) in interior_edges:
            # Skip patch-boundary edges
            if fl is not None and fl[f0] != fl[f1]:
                continue

            # Identify the four vertices of the diamond: a-b is the
            # shared edge, c and d are the opposite vertices.
            vf0 = [int(v) for v in F[f0]]
            vf1 = [int(v) for v in F[f1]]
            c = [v for v in vf0 if v != a and v != b]
            d = [v for v in vf1 if v != a and v != b]
            if len(c) != 1 or len(d) != 1:
                continue
            c, d = c[0], d[0]

            # Don't create an edge that already exists (would make
            # non-manifold) or a degenerate triangle
            if c == d:
                continue
            ek_new = (min(c, d), max(c, d))
            if ek_new in edge2pair and len(edge2pair[ek_new]) >= 2:
                continue

            # Cost before flip
            cost_before = _dev(a) + _dev(b) + _dev(c) + _dev(d)

            # Cost after flip: a and b each lose 1, c and d each gain 1
            va_new = valence[a] - 1
            vb_new = valence[b] - 1
            vc_new = valence[c] + 1
            vd_new = valence[d] + 1

            # Reject if it would create valence < 3
            if va_new < 3 or vb_new < 3:
                continue

            cost_after = ((va_new - target_valence) ** 2 +
                          (vb_new - target_valence) ** 2 +
                          (vc_new - target_valence) ** 2 +
                          (vd_new - target_valence) ** 2)

            if cost_after >= cost_before:
                continue

            # Perform the flip: replace edge (a,b) with (c,d)
            # f0: (a, b, c) -> (c, d, a)
            # f1: (a, d, b) -> (d, c, b)   (but vertex order varies)
            # We need to preserve winding.  In f0, the edge a->b is
            # followed by b->c->a.  After flip, we want c->d->a with
            # the same winding sense.
            ia0 = vf0.index(a)
            ib0 = vf0.index(b)
            ic0 = vf0.index(c)
            # In f0, replace b with d (keeping winding a->?->c)
            F[f0][ib0] = d

            ia1 = vf1.index(a)
            ib1 = vf1.index(b)
            id1 = vf1.index(d)
            # In f1, replace a with c
            F[f1][ia1] = c

            # Update valences
            valence[a] -= 1
            valence[b] -= 1
            valence[c] += 1
            valence[d] += 1

            flips_this_pass += 1

        total_flips += flips_this_pass
        if verbose and flips_this_pass > 0:
            dev_sum = sum((valence[v] - target_valence) ** 2
                          for v in range(nv))
            print(f"    Pass {pass_num}: {flips_this_pass} flips, "
                  f"valence dev^2 = {dev_sum}")
        if flips_this_pass == 0:
            break

    mesh.F = F

    if verbose:
        v3 = int(np.sum(valence == 3))
        v4 = int(np.sum(valence == 4))
        v56 = int(np.sum((valence >= 5) & (valence <= 7)))
        vhi = int(np.sum(valence > 7))
        print(f"  Valence equalization: {total_flips} total flips over "
              f"{min(pass_num + 1, max_passes)} passes")
        print(f"    val=3: {v3}, val=4: {v4}, val=5-7: {v56}, val>7: {vhi}")

    return total_flips


def _fix_winding_consistency(mesh, verbose=False):
    """Make all face windings consistent using BFS propagation.

    Two adjacent faces sharing edge (a, b) are consistently wound when
    the edge appears as (a, b) in one face and (b, a) in the other.
    Starting from face 0, BFS flips any neighbor whose shared edge
    runs in the same direction.  After propagation, a signed-volume
    test determines whether normals point outward; if not, all faces
    are flipped.

    Parameters
    ----------
    mesh : surface_mesh
        Modified in place (``mesh.F`` rows may be reversed).
    verbose : bool
        Print summary.

    Returns
    -------
    n_flipped : int
        Number of faces whose winding was reversed.
    """
    F = np.asarray(mesh.F)
    nf = len(F)
    if nf == 0:
        return 0

    edge2faces = defaultdict(list)
    for fi in range(nf):
        for i in range(3):
            a, b = int(F[fi][i]), int(F[fi][(i + 1) % 3])
            edge2faces[(min(a, b), max(a, b))].append(fi)

    visited = np.zeros(nf, dtype=bool)
    flipped = np.zeros(nf, dtype=bool)

    # BFS from every unvisited face (handles disconnected components)
    for seed in range(nf):
        if visited[seed]:
            continue
        queue = deque([seed])
        visited[seed] = True
        while queue:
            fi = queue.popleft()
            for i in range(3):
                a, b = int(F[fi][i]), int(F[fi][(i + 1) % 3])
                ekey = (min(a, b), max(a, b))
                for fj in edge2faces[ekey]:
                    if visited[fj]:
                        continue
                    visited[fj] = True
                    vj = list(F[fj])
                    ia = vj.index(a) if a in vj else -1
                    ib = vj.index(b) if b in vj else -1
                    if ia < 0 or ib < 0:
                        queue.append(fj)
                        continue
                    # Consistent: edge (a,b) in fi means (b,a) in fj
                    if (ib + 1) % 3 == ia:
                        pass  # already consistent
                    else:
                        F[fj] = [F[fj][0], F[fj][2], F[fj][1]]
                        flipped[fj] = True
                    queue.append(fj)

    n_flipped = int(flipped.sum())

    # Signed-volume test: if total signed volume is negative, normals
    # point inward -> flip everything.
    X = np.asarray(mesh.X)
    signed_vol = 0.0
    for fi in range(nf):
        v0, v1, v2 = X[F[fi][0]], X[F[fi][1]], X[F[fi][2]]
        signed_vol += np.dot(v0, np.cross(v1, v2))
    if signed_vol < 0:
        for fi in range(nf):
            F[fi] = [F[fi][0], F[fi][2], F[fi][1]]
        flipped = ~flipped
        n_flipped = nf - n_flipped

    mesh.F = F

    if verbose:
        print(f"  Winding fix: {n_flipped} faces flipped "
              f"(signed_vol={'positive' if signed_vol >= 0 else 'negative'})")

    return n_flipped


def build_pm_from_decimated_mesh(m_seg, PM, target_faces=None, target_ratio=None,
                                 curvature_weight=1.0, verbose=True):
    """
    Replace PM['pm'] with a curvature-aware decimated version of the original mesh.

    The function:
    1. Collects all key vertices and center vertices from PM (they must survive).
    2. Decimates *m_seg* with curvature-aware half-edge collapse, protecting
       those critical vertices.
    3. Transfers face labels from the surviving original faces.
    4. Rebuilds the PM mapping fields (Xkeyind, Keyind, CVind, nkeys).
    5. Validates the result (manifold, closed, genus-0).

    Parameters
    ----------
    m_seg : surface_mesh
        Segmented mesh (original resolution) with *face_labels* set.
        Must be the same mesh from which PM was constructed.
    PM : dict
        Existing PM structure from ``patch_info_gen`` (or partial -- at
        minimum needs 'keys', 'CV', 'npatches').
    target_faces : int, optional
        Target face count for the decimated mesh.
    target_ratio : float, optional
        Fraction of original faces to keep (default 0.08 ≈ 8 %).
    curvature_weight : float
        Curvature preservation weight (default 1.0).
    verbose : bool
        Print progress.

    Returns
    -------
    PM : dict
        The *same* PM dict, updated in-place with:
        - PM['pm']       : the decimated surface_mesh
        - PM['Xkeyind']  : vertex mapping (decimated → original)
        - PM['Keyind']    : key vertex indices in decimated mesh
        - PM['CVind']     : center vertex indices in decimated mesh
        - PM['nkeys']     : number of key+sentinel vertices
    """
    from ..surface_mesh import surface_mesh
    from .validate_simplified_mesh import validate_simplified_mesh

    if target_faces is None and target_ratio is None:
        target_ratio = 0.08

    # ----------------------------------------------------------------
    # 1.  Collect protected vertices (keys + sentinels + centers +
    #     per-patch anchors so no patch vanishes during decimation)
    # ----------------------------------------------------------------
    protected = set()

    if 'keys' in PM and len(PM['keys']) > 0:
        for row in PM['keys']:
            protected.add(int(row[1]))

    if 'sentinels' in PM and PM['sentinels'] is not None:
        for s1, s2 in PM['sentinels']:
            if int(s1) >= 0:
                protected.add(int(s1))
            if int(s2) >= 0:
                protected.add(int(s2))

    if 'CV' in PM:
        for cv in PM['CV']:
            protected.add(int(cv))

    # Protect vertices in small patches so they don't vanish entirely
    # during aggressive decimation.  For patches below a face-count
    # threshold, protect ALL their vertices; for others, protect enough
    # to ensure at least a few triangles survive.
    npatches = PM.get('npatches', 0)
    if 'P' in PM and m_seg.face_labels is not None:
        fl = np.asarray(m_seg.face_labels)
        uL_orig = np.sort(np.unique(fl))
        F_seg = np.asarray(m_seg.F, dtype=int)
        avg_faces_per_patch = len(F_seg) / max(npatches, 1)
        small_threshold = max(int(avg_faces_per_patch * 0.5), 30)
        for pix in range(npatches):
            label = uL_orig[pix] if pix < len(uL_orig) else pix
            patch_faces = np.where(fl == label)[0]
            if len(patch_faces) == 0:
                continue
            patch_verts = np.unique(F_seg[patch_faces].ravel())
            if len(patch_faces) < small_threshold:
                for v in patch_verts:
                    protected.add(int(v))
            else:
                already = sum(1 for v in patch_verts if v in protected)
                need = max(0, 6 - already)
                if need > 0:
                    candidates = [v for v in patch_verts if v not in protected]
                    np.random.seed(pix)
                    chosen = candidates[:need] if len(candidates) <= need else \
                        list(np.random.choice(candidates, need, replace=False))
                    for v in chosen:
                        protected.add(int(v))

    if verbose:
        fl_check = np.asarray(m_seg.face_labels)
        print(f"build_pm_from_decimated_mesh: "
              f"{len(protected)} protected vertices "
              f"(keys + sentinels + centers + per-patch anchors)")
        print(f"  m_seg: {len(m_seg.X)} verts, {len(m_seg.F)} faces, "
              f"face_labels: {len(fl_check)} entries, "
              f"{len(np.unique(fl_check))} unique, "
              f"range [{fl_check.min()}, {fl_check.max()}]")

    # ----------------------------------------------------------------
    # 1b. Pre-check: if input has non-manifold edges, clean the mesh
    #     via PyVista before decimation (remesh_by_curvature can
    #     sometimes leave non-manifold edges).
    # ----------------------------------------------------------------
    m_seg = _ensure_manifold(m_seg, verbose=verbose)

    # ----------------------------------------------------------------
    # 2.  Decimate
    # ----------------------------------------------------------------
    m_dec, vert_map = m_seg.curvature_aware_decimation(
        target_faces=target_faces,
        target_ratio=target_ratio,
        curvature_weight=curvature_weight,
        protected_vertices=list(protected),
        verbose=verbose,
    )

    # ----------------------------------------------------------------
    # 2b. Remap face labels from original (possibly 1-indexed) to
    #     0-based patch indices expected by validate_simplified_mesh
    #     and the rest of the pipeline.
    # ----------------------------------------------------------------
    if m_dec.face_labels is not None:
        fl_orig = np.asarray(m_seg.face_labels)
        uL = np.sort(np.unique(fl_orig))
        n_before = len(np.unique(m_dec.face_labels))
        label2idx = {int(lab): idx for idx, lab in enumerate(uL)}
        m_dec.face_labels = np.array(
            [label2idx.get(int(fl), int(fl)) for fl in m_dec.face_labels],
            dtype=int)
        n_after = len(np.unique(m_dec.face_labels))
        if verbose:
            print(f"  Face labels: {n_before} unique before remap, "
                  f"{n_after} unique after remap to 0-based")
            if n_after < npatches:
                missing = set(range(npatches)) - set(np.unique(m_dec.face_labels))
                print(f"  WARNING: patches missing after decimation: {sorted(missing)}")
    else:
        if verbose:
            print("  WARNING: m_dec has no face_labels (decimation did not "
                  "transfer them)")

    # ----------------------------------------------------------------
    # 2c. Fix face winding so all normals point consistently outward.
    #     Mixed winding causes self-intersecting triangles after
    #     spherical parameterization.
    # ----------------------------------------------------------------
    _fix_winding_consistency(m_dec, verbose=verbose)

    # ----------------------------------------------------------------
    # 2d. Equalize vertex valences via edge flips.
    #     Curvature-aware decimation tends to produce many low-valence
    #     vertices (val 3-4) which make the Brechbühler diffusion
    #     ill-conditioned.  Edge flips move valences toward 6 without
    #     changing the vertex set or face count.
    # ----------------------------------------------------------------
    _equalize_valences(m_dec, target_valence=6, max_passes=20,
                       verbose=verbose)

    # ----------------------------------------------------------------
    # 3.  Build reverse lookup: original vertex index → decimated index
    # ----------------------------------------------------------------
    orig2dec = {}
    for dec_i, orig_i in enumerate(vert_map):
        orig2dec[int(orig_i)] = dec_i

    missing_keys = [v for v in protected if int(v) not in orig2dec]
    if missing_keys and verbose:
        print(f"  WARNING: {len(missing_keys)} protected vertices not in "
              f"decimated mesh (mesh may have been too aggressively decimated)")

    # ----------------------------------------------------------------
    # 4.  Initialise spherical-coordinate arrays
    # ----------------------------------------------------------------
    nv_dec = len(m_dec.X)
    m_dec.t = np.zeros(nv_dec, dtype=float)
    m_dec.p = np.zeros(nv_dec, dtype=float)

    if m_seg.border_vertex is not None:
        m_dec.border_vertex = np.zeros(nv_dec, dtype=int)
        for i, orig_i in enumerate(vert_map):
            if orig_i < len(m_seg.border_vertex):
                m_dec.border_vertex[i] = m_seg.border_vertex[int(orig_i)]

    # ----------------------------------------------------------------
    # 5.  Build Xkeyind, Keyind, CVind
    # ----------------------------------------------------------------
    Xkeyind = vert_map.copy()

    key_orig_indices = set()
    if 'keys' in PM and len(PM['keys']) > 0:
        key_orig_indices = set(int(k) for k in PM['keys'][:, 1])
    if 'sentinels' in PM and PM['sentinels'] is not None:
        for s1, s2 in PM['sentinels']:
            if int(s1) >= 0:
                key_orig_indices.add(int(s1))
            if int(s2) >= 0:
                key_orig_indices.add(int(s2))

    Keyind = np.array(
        [dec_i for dec_i, orig_i in enumerate(vert_map)
         if int(orig_i) in key_orig_indices],
        dtype=int,
    )

    cv_orig = set(int(cv) for cv in PM.get('CV', []))
    CVind = np.array(
        [dec_i for dec_i, orig_i in enumerate(vert_map)
         if int(orig_i) in cv_orig],
        dtype=int,
    )

    nkeys = len(Keyind)

    # ----------------------------------------------------------------
    # 6.  Store into PM
    # ----------------------------------------------------------------
    PM['pm'] = m_dec
    PM['Xkeyind'] = Xkeyind
    PM['Keyind'] = Keyind
    PM['CVind'] = CVind
    PM['nkeys'] = nkeys
    PM['_decimation_method'] = 'curvature_aware'
    PM['_decimation_vert_map'] = vert_map
    PM['_decimation_orig2dec'] = orig2dec

    if verbose:
        print(f"  PM updated: pm has {len(m_dec.X)} verts, {len(m_dec.F)} faces, "
              f"nkeys={nkeys}, CVind={len(CVind)}")

    # ----------------------------------------------------------------
    # 7.  Validate
    # ----------------------------------------------------------------
    is_valid, issues = validate_simplified_mesh(m_dec, PM, verbose=verbose)
    if not is_valid:
        if verbose:
            print(f"  Decimated mesh validation issues: {list(issues.keys())}")
            for k, v in issues.items():
                print(f"    {k}: {v}")
    else:
        if verbose:
            print("  Decimated mesh PASSED all validation checks "
                  "(manifold, closed, genus-0, connected)")

    return PM


def find_valid_segmentation_with_decimated_mesh(
        m, nseeds_range=None, min_neighbors=3, sig=1.0,
        curvature_weight_seg=0.0, curvature_weight_dec=1.0,
        target_faces=None, target_ratio=None,
        verbose=True, plot_intermediate=False,
        allow_annular=True, failure_log_path=None):
    """
    End-to-end alternative to ``find_valid_segmentation_with_simplified_mesh``.

    For each seed count in *nseeds_range*:
    1. Segment the mesh (random-walk).
    2. Check patch-neighbor validity.
    3. Build PM via ``patch_info_gen`` (which internally builds a fan-triangulated
       simplified mesh -- we ignore it).
    4. Replace PM['pm'] with a curvature-aware decimated mesh.
    5. Validate the result.
    6. Return the first valid configuration, or None.

    This avoids all the topological annuli / scrambled-boundary issues of
    fan-triangulated simplified meshes because the decimated mesh inherits
    the original mesh's clean topology.

    Parameters
    ----------
    m : surface_mesh
        Input mesh.
    nseeds_range : tuple (lo, hi) or list of int
        Seed counts to try.  Default (8, 25).
    min_neighbors, sig, curvature_weight_seg
        Parameters for ``mesh_segmentation_rw``.
    curvature_weight_dec : float
        Curvature weight for decimation (default 1.0).
    target_faces, target_ratio
        Decimation target (default ratio 0.08).
    verbose, plot_intermediate, allow_annular, failure_log_path
        Same semantics as the original ``find_valid_segmentation_with_simplified_mesh``.

    Returns
    -------
    result : dict or None
        Keys: 'nseeds', 'ms', 'L', 'slix', 'P', 'Pconn', 'm_seg', 'PM',
        'neighbor_report', 'failed_patterns'.
    """
    from .mesh_segmentation_rw import mesh_segmentation_rw
    from .patch_info_gen import patch_info_gen
    from .find_valid_segmentation import (
        compute_vertex_based_patch_connectivity,
        check_patch_neighbors_valid,
        _append_failure_log,
    )
    from .validate_simplified_mesh import validate_simplified_mesh
    import uuid, datetime

    run_id = uuid.uuid4().hex[:8]

    if nseeds_range is None:
        nseeds_range = (8, 25)
    if isinstance(nseeds_range, (list, np.ndarray)):
        seeds_to_try = list(nseeds_range)
    else:
        lo, hi = int(nseeds_range[0]), int(nseeds_range[1])
        seeds_to_try = list(range(lo, hi + 1))

    failed_patterns = []

    if verbose:
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("=" * 60)
        print("find_valid_segmentation_with_decimated_mesh")
        print(f"  Run ID: {run_id}  |  Timestamp: {ts}")
        print(f"  Trying nseeds in {seeds_to_try}")
        print(f"  Decimation: curvature_weight={curvature_weight_dec}, "
              f"target_faces={target_faces}, target_ratio={target_ratio}")
        print("=" * 60)

    for nseeds in seeds_to_try:
        if verbose:
            print(f"\n--- nseeds = {nseeds} ---")

        # [1] Segment
        try:
            ms, L, slix, P, Pconn = mesh_segmentation_rw(
                m, nseeds, sig=sig, curvature_weight=curvature_weight_seg,
                verbose=False, plot_intermediate=False,
            )
        except Exception as e:
            if verbose:
                print(f"  Segmentation failed: {e}")
            failed_patterns.append({'nseeds': nseeds, 'stage': 'segmentation',
                                    'error': str(e)})
            _append_failure_log(failure_log_path, nseeds, 'segmentation',
                                {'error': str(e)})
            continue

        # [2] Neighbor check
        Pconn_v = compute_vertex_based_patch_connectivity(ms)
        ok, report = check_patch_neighbors_valid(
            Pconn_v, min_neighbors=min_neighbors, verbose=verbose,
            allow_annular=allow_annular,
        )
        if not ok:
            if verbose:
                print(f"  Invalid patches: {report['invalid_patches']}")
            failed_patterns.append({'nseeds': nseeds, 'stage': 'neighbor_check',
                                    'invalid_patches': report['invalid_patches']})
            _append_failure_log(failure_log_path, nseeds, 'neighbor_check',
                                {'invalid_patches': report['invalid_patches']})
            continue

        if verbose:
            print(f"  Neighbor check PASS ({report['n_patches']} patches)")

        # [3] Build PM (patch_info_gen builds its own simplified mesh internally;
        #     we will overwrite PM['pm'] in the next step)
        #
        # IMPORTANT: patch_info_gen internally calls get_border() which
        # overwrites m.face_labels with a binary border indicator.  We
        # must preserve the segmentation labels for decimation.
        saved_face_labels = ms.face_labels.copy()

        try:
            m_seg, PM, Pconn_out = patch_info_gen(
                ms, P, Pconn,
                validate_segmentation=False,
                raise_on_invalid_segmentation=False,
                strict_simplified_mesh=False,
            )
        except Exception as e:
            if verbose:
                print(f"  patch_info_gen failed: {e}")
            failed_patterns.append({'nseeds': nseeds, 'stage': 'patch_info_gen',
                                    'error': str(e)})
            _append_failure_log(failure_log_path, nseeds, 'patch_info_gen',
                                {'error': str(e)})
            ms.face_labels = saved_face_labels
            continue

        # Restore segmentation labels (get_border overwrites them)
        m_seg.face_labels = saved_face_labels

        # [4] Replace simplified mesh with decimated mesh
        try:
            PM = build_pm_from_decimated_mesh(
                m_seg, PM,
                target_faces=target_faces,
                target_ratio=target_ratio,
                curvature_weight=curvature_weight_dec,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  Decimation failed: {e}")
            failed_patterns.append({'nseeds': nseeds, 'stage': 'decimation',
                                    'error': str(e)})
            _append_failure_log(failure_log_path, nseeds, 'decimation',
                                {'error': str(e)})
            continue

        PM['run_id'] = run_id

        # [5] Validate
        is_valid, issues = validate_simplified_mesh(PM['pm'], PM, verbose=False)
        if not is_valid:
            if verbose:
                print(f"  Decimated mesh INVALID: {list(issues.keys())}")
            failed_patterns.append({'nseeds': nseeds,
                                    'stage': 'decimated_mesh_validation',
                                    'issues': issues})
            _append_failure_log(failure_log_path, nseeds,
                                'decimated_mesh_validation', issues)
            continue

        if verbose:
            pm = PM['pm']
            print(f"\n  *** SUCCESS: nseeds={nseeds}, {report['n_patches']} patches, "
                  f"decimated mesh: {len(pm.X)} verts, {len(pm.F)} faces")

        if plot_intermediate:
            try:
                ms.plot_segmentation_with_seeds(
                    slix, verbose=verbose,
                    title=f'Valid (nseeds={nseeds}, decimated mesh)')
            except Exception as e:
                if verbose:
                    print(f"  Plot failed: {e}")

        return {
            'nseeds': nseeds,
            'ms': ms,
            'L': L,
            'slix': slix,
            'P': P,
            'Pconn': Pconn,
            'm_seg': m_seg,
            'PM': PM,
            'neighbor_report': report,
            'failed_patterns': failed_patterns,
        }

    if verbose:
        print(f"\nNo valid segmentation+decimated_mesh in range {seeds_to_try}")
    return None
