"""
Find a mesh segmentation where every patch is well-behaved for simplified mesh construction.

A patch is well-behaved if it has either:
- Exactly 1 neighbor (cap patch): handled with fictitious keys and fan triangulation
- 2 neighbors (annular/band): invalid by default (allow_annular=False) - often causes foldovers
- >= 3 neighbors (regular patch): has enough key vertices for a proper boundary cycle

This module provides:
- check_patch_neighbors_valid: check if a segmentation satisfies the neighbor constraint
- find_valid_segmentation: iterate over nseeds range to find valid neighbor structure
- find_valid_segmentation_with_simplified_mesh: full flow - valid segmentation AND valid
  simplified mesh; records failed patterns and moves on to next nseeds
"""

import numpy as np
import uuid
import datetime
from scipy.sparse import csr_matrix


def check_patch_neighbors_valid(Pconn, min_neighbors=3, verbose=False, allow_annular=True):
    """
    Check whether every patch has a valid number of neighbors for simplified mesh construction.

    Valid neighbor counts (with allow_annular=True, default):
    - 1 neighbor: cap patch (handled with fictitious keys during simplified mesh construction)
    - 2 neighbors: annular/band patch (now allowed; we handle it)
    - >= min_neighbors: regular patch with enough key vertices

    Invalid:
    - 0 neighbors: isolated patch (should not happen with proper segmentation)
    - 2 neighbors: only when allow_annular=False (legacy strict mode)

    Parameters
    ----------
    Pconn : sparse matrix or ndarray
        Patch connectivity matrix (symmetric, Pconn[i,j] > 0 iff patches i and j are neighbors).
    min_neighbors : int
        Minimum required neighbors for non-cap, non-annular patches (default 3).
    verbose : bool
        Print per-patch neighbor counts.
    allow_annular : bool
        If True (default), patches with 2 neighbors (annular/band) are valid.
        If False, 2 neighbors is invalid (legacy strict mode).
    """
    if hasattr(Pconn, 'toarray'):
        A = (Pconn.toarray() > 0).astype(int)
    else:
        A = (np.asarray(Pconn) > 0).astype(int)
    # Make symmetric
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    n_patches = A.shape[0]
    neighbors = A.sum(axis=1).astype(int)
    if hasattr(neighbors, 'A1'):
        neighbors = np.asarray(neighbors).flatten()
    else:
        neighbors = np.asarray(neighbors).flatten()
    
    cap_patches = [int(i) for i in range(n_patches) if neighbors[i] == 1]
    annular_patches = [int(i) for i in range(n_patches) if neighbors[i] == 2]
    if allow_annular:
        invalid_patches = [int(i) for i in range(n_patches) if neighbors[i] == 0]
    else:
        invalid_patches = [int(i) for i in range(n_patches)
                           if neighbors[i] == 0 or (2 <= neighbors[i] < min_neighbors)]
    is_valid = len(invalid_patches) == 0

    report = {
        'n_patches': n_patches,
        'neighbors_per_patch': neighbors,
        'cap_patches': cap_patches,
        'annular_patches': annular_patches,
        'invalid_patches': invalid_patches,
        'min_neighbor_count': int(neighbors.min()) if n_patches > 0 else 0,
    }

    if verbose:
        mode = "allow annular" if allow_annular else "strict (no annular)"
        print(f"check_patch_neighbors_valid (min_regular={min_neighbors}, {mode}):")
        print(f"  Patches: {n_patches}")
        for i in range(n_patches):
            if neighbors[i] == 1:
                flag = " (cap)"
            elif neighbors[i] == 2 and allow_annular:
                flag = " (annular)"
            elif neighbors[i] == 0 or (not allow_annular and 2 <= neighbors[i] < min_neighbors):
                flag = " *** INVALID"
            else:
                flag = ""
            print(f"  Patch {i}: {neighbors[i]} neighbors{flag}")
        if is_valid:
            parts = []
            if cap_patches:
                parts.append(f"caps={cap_patches}")
            if allow_annular and annular_patches:
                parts.append(f"annular={annular_patches}")
            msg = f"  PASS: all patches valid" + (f" ({', '.join(parts)})" if parts else "")
            print(msg)
        else:
            print(f"  FAIL: invalid patches: {invalid_patches}")

    return is_valid, report


# Backward-compatible alias
def check_min_neighbors(Pconn, min_neighbors=3, verbose=False):
    """Backward-compatible wrapper for check_patch_neighbors_valid."""
    return check_patch_neighbors_valid(Pconn, min_neighbors=min_neighbors, verbose=verbose)


def compute_vertex_based_patch_connectivity(ms):
    """
    Compute patch connectivity based on shared vertices (not just face adjacency).

    Two patches are neighbors if they share at least one mesh vertex.
    This catches connections through triple-junction key vertices that face-based
    Pconn misses (e.g. a cap patch touching two other patches through a single vertex).

    Parameters
    ----------
    ms : surface_mesh
        Segmented mesh with face_labels set.

    Returns
    -------
    Pconn_vertex : sparse matrix
        Patch connectivity matrix (symmetric).
    """
    from collections import defaultdict
    from scipy.sparse import lil_matrix

    fl = ms.face_labels
    uL = np.unique(fl)
    npatches = len(uL)
    label_to_idx = {int(lab): i for i, lab in enumerate(uL)}

    # Find which patches each vertex belongs to
    vertex_patches = defaultdict(set)
    for fi in range(len(ms.F)):
        pidx = label_to_idx.get(int(fl[fi]))
        if pidx is None:
            continue
        for v in ms.F[fi]:
            vertex_patches[int(v)].add(pidx)

    # Two patches are neighbors if they share at least one vertex
    Pconn_vertex = lil_matrix((npatches, npatches), dtype=int)
    for v, patches in vertex_patches.items():
        patches_list = list(patches)
        for i in range(len(patches_list)):
            for j in range(i + 1, len(patches_list)):
                p1, p2 = patches_list[i], patches_list[j]
                Pconn_vertex[p1, p2] = 1
                Pconn_vertex[p2, p1] = 1

    return Pconn_vertex.tocsr()


def find_valid_segmentation(m, nseeds_range=None, min_neighbors=3, sig=1.0,
                             curvature_weight=0.0, verbose=True, plot_intermediate=False, allow_annular=True):
    """
    Iterate over a range of seed counts to find a segmentation where every patch
    has a valid neighbor count for simplified mesh construction.

    Valid: 1 neighbor (cap, handled with fictitious keys) or >= min_neighbors.
    Invalid: 0 or 2 neighbors (neck/bridge, causes topology issues).

    Uses vertex-based connectivity (not just face adjacency) to accurately
    detect connections through triple-junction key vertices.

    Parameters
    ----------
    m : surface_mesh
        Input mesh (vertices and faces; will be modified with face_labels).
    nseeds_range : tuple or list of int, optional
        (min_seeds, max_seeds) range to try. Default (8, 20).
        Can also be a list of specific seed counts to try.
    min_neighbors : int
        Minimum neighbors for non-cap patches (default 3).
    sig : float
        Segmentation sigma parameter (default 1.0).
    curvature_weight : float
        Curvature weight for segmentation (default 0.0).
    verbose : bool
        Print progress.
    plot_intermediate : bool
        If True, plot the resulting segmentation when a valid one is found
        (similar to mesh_segmentation_rw with plot_intermediate=True).
    allow_annular : bool
        If True (default), patches with 2 neighbors (annular/band) are valid.
        If False, 2 neighbors is invalid (legacy strict mode).

    Returns
    -------
    result : dict or None
        If a valid segmentation is found:
        - 'nseeds': number of seeds used
        - 'ms': segmented mesh (surface_mesh with face_labels)
        - 'L': face labels array
        - 'slix': seed face indices
        - 'P': patch structures
        - 'Pconn': patch connectivity matrix (vertex-based)
        - 'neighbor_report': neighbor check report
        If no valid segmentation found in range, returns None.
    """
    from .mesh_segmentation_rw import mesh_segmentation_rw

    if nseeds_range is None:
        nseeds_range = (8, 20)

    if isinstance(nseeds_range, (list, np.ndarray)):
        seeds_to_try = list(nseeds_range)
    else:
        lo, hi = int(nseeds_range[0]), int(nseeds_range[1])
        seeds_to_try = list(range(lo, hi + 1))

    if verbose:
        print("=" * 60)
        print(f"find_valid_segmentation: trying nseeds in {seeds_to_try}")
        print(f"  min_neighbors = {min_neighbors}, sig = {sig}, curvature_weight = {curvature_weight}")
        print("=" * 60)

    for nseeds in seeds_to_try:
        if verbose:
            print(f"\n--- Trying nseeds = {nseeds} ---")

        try:
            ms, L, slix, P, Pconn = mesh_segmentation_rw(
                m, nseeds, sig=sig, curvature_weight=curvature_weight,
                verbose=False, plot_intermediate=False
            )
        except Exception as e:
            if verbose:
                print(f"  Segmentation failed for nseeds={nseeds}: {e}")
            continue

        # Use vertex-based connectivity (more accurate than face-based Pconn)
        Pconn_vertex = compute_vertex_based_patch_connectivity(ms)
        is_valid, report = check_patch_neighbors_valid(Pconn_vertex, min_neighbors=min_neighbors,
                                                       verbose=verbose, allow_annular=allow_annular)

        if is_valid:
            if verbose:
                caps = report.get('cap_patches', [])
                annular = report.get('annular_patches', [])
                cap_msg = f", caps={caps}" if caps else ""
                ann_msg = f", annular={annular}" if annular else ""
                print(f"\n  FOUND valid segmentation: nseeds = {nseeds}, "
                      f"{report['n_patches']} patches, min neighbors = {report['min_neighbor_count']}{cap_msg}{ann_msg}")
            if plot_intermediate:
                try:
                    ms.plot_segmentation_with_seeds(slix, verbose=verbose,
                        title=f'Valid segmentation (nseeds={nseeds}, {report["n_patches"]} patches)')
                    if verbose:
                        print("Plotted resulting segmentation with seed faces highlighted")
                except Exception as e:
                    if verbose:
                        print(f"plot_segmentation_with_seeds failed: {e}")
                    try:
                        ms.plot_labels()
                        if verbose:
                            print("Plotted resulting segmentation (plot_labels fallback)")
                    except Exception as e2:
                        if verbose:
                            print(f"Could not plot segmentation: {e2}")
            return {
                'nseeds': nseeds,
                'ms': ms,
                'L': L,
                'slix': slix,
                'P': P,
                'Pconn': Pconn,
                'neighbor_report': report,
            }
        else:
            if verbose:
                print(f"  nseeds={nseeds}: {report['n_patches']} patches, "
                      f"invalid patches={report['invalid_patches']}")

    if verbose:
        print(f"\nNo valid segmentation found in range {seeds_to_try}")
        print(f"Consider increasing the range or reducing min_neighbors.")

    return None


def _has_topological_annuli(PM, m=None):
    """Return (has_annuli, list of patch indices with 2 boundary components).
    
    When m is provided, re-runs analyze_patch_types(m, PM) to get correct boundary_components
    (the stored report may have been computed before border_vertex was set on patch meshes).
    """
    if m is not None:
        try:
            from .patch_type_analysis import analyze_patch_types
            ptr = analyze_patch_types(m, PM, verbose=False)
        except Exception:
            ptr = None
    else:
        ptr = (PM.get('patch_structure_report') or {}).get('patch_type_report')
    if ptr is None:
        return False, []
    boundary_comps = ptr.get('boundary_components', [])
    annular_patch_indices = []
    for pix, comps in enumerate(boundary_comps):
        if len(comps) == 2:
            annular_patch_indices.append(pix)
    return len(annular_patch_indices) > 0, annular_patch_indices


def _has_zero_key_patches(PM):
    """Return (has_zero_key, list of patch indices with 0 key vertices before synthetic keys)."""
    report = PM.get('patch_structure_report') or {}
    zk = report.get('zero_key_patch_indices', np.array([], dtype=int))
    zero_key = np.asarray(zk).flatten().tolist() if zk is not None else []
    return len(zero_key) > 0, zero_key


def _simplified_mesh_face_quality_issues(pm, min_angle_deg=5.0, max_aspect_ratio=50.0):
    """Check for degenerate/tapered faces. Return (has_issues, list of bad face indices)."""
    if pm is None or pm.X is None or pm.F is None:
        return False, []
    X, F = np.asarray(pm.X), np.asarray(pm.F)
    bad_faces = []
    min_rad = np.radians(min_angle_deg)
    for fix, f in enumerate(F):
        v0, v1, v2 = X[int(f[0])], X[int(f[1])], X[int(f[2])]
        e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
        len0 = np.linalg.norm(e0)
        len1 = np.linalg.norm(e1)
        len2 = np.linalg.norm(e2)
        if len0 < 1e-14 or len1 < 1e-14 or len2 < 1e-14:
            bad_faces.append((fix, 'degenerate'))
            continue
        # Min angle via cross product
        angles = [
            np.arcsin(np.clip(np.linalg.norm(np.cross(e0, -e2)) / (len0 * len2), 0, 1)),
            np.arcsin(np.clip(np.linalg.norm(np.cross(e1, -e0)) / (len1 * len0), 0, 1)),
            np.arcsin(np.clip(np.linalg.norm(np.cross(e2, -e1)) / (len2 * len1), 0, 1)),
        ]
        min_ang = min(angles)
        if min_ang < min_rad:
            bad_faces.append((fix, f'min_angle={np.degrees(min_ang):.2f}deg'))
            continue
        # Aspect ratio: longest_edge / shortest_edge
        lengths = sorted([len0, len1, len2])
        if lengths[0] > 1e-14:
            ar = lengths[2] / lengths[0]
            if ar > max_aspect_ratio:
                bad_faces.append((fix, f'aspect_ratio={ar:.1f}'))
    return len(bad_faces) > 0, bad_faces


def find_valid_segmentation_with_simplified_mesh(m, nseeds_range=None, min_neighbors=3, sig=1.0,
                                                 curvature_weight=0.0, verbose=True, plot_intermediate=False,
                                                 allow_annular=False, avoid_topological_annuli=True,
                                                 avoid_zero_key_patches=True,
                                                 min_face_angle_deg=5.0, max_face_aspect_ratio=50.0,
                                                 failure_log_path=None):
    """
    Find nseeds that yields both valid patch neighbor structure AND valid simplified mesh.

    Iterates over nseeds in range; for each:
    1. Run mesh segmentation
    2. Check patch neighbors valid (no annular by default)
    3. Run patch_info_gen to build simplified mesh
    4. Reject if avoid_topological_annuli and any patch has 2 boundary components
    5. Reject if avoid_zero_key_patches and any patch has 0 key vertices
    6. Reject if simplified mesh has degenerate/tapered faces (bad quality)
    7. Validate simplified mesh (closed, manifold, genus-zero)
    8. If invalid: record failure pattern, try next nseeds; if valid: return and stop

    Only proceeds with spherical parameterization when both checks pass. Failed patterns
    are recorded for future analysis.

    Parameters
    ----------
    m : surface_mesh
        Input mesh (vertices and faces).
    nseeds_range : tuple or list of int, optional
        (min_seeds, max_seeds) or list of values to try. Default (8, 25).
    min_neighbors, sig, curvature_weight, verbose, plot_intermediate
        As in find_valid_segmentation.
    allow_annular : bool
        If False (default), patches with 2 neighbors are invalid; skips and tries next nseeds.
        Set True to allow annular patches (may cause foldovers).
    avoid_topological_annuli : bool
        If True (default), reject segmentations where any patch has 2 boundary components
        (cap-enclosing annulus or cylinder). These cause severe foldovers during parameterization.
    avoid_zero_key_patches : bool
        If True (default), reject when any patch has 0 key vertices (super-tapered cap).
    min_face_angle_deg : float
        Minimum face angle in degrees for simplified mesh (default 5). Reject if any face is thinner.
    max_face_aspect_ratio : float
        Max edge-aspect-ratio for simplified mesh faces (default 50). Reject if any face is too elongated.
    failure_log_path : str, optional
        If provided, append failed patterns to this file (JSON lines format).

    Returns
    -------
    result : dict or None
        If found: dict with 'nseeds', 'ms', 'L', 'slix', 'P', 'Pconn', 'm_seg', 'PM',
        'neighbor_report', 'failed_patterns' (list of {nseeds, reason, issues}).
        If not found: None.
    """
    from .mesh_segmentation_rw import mesh_segmentation_rw
    from .patch_info_gen import patch_info_gen
    from .validate_simplified_mesh import validate_simplified_mesh
    import json
    import os

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
        print(f"find_valid_segmentation_with_simplified_mesh")
        print(f"  Run ID: {run_id}  |  Timestamp: {ts}")
        print(f"  Trying nseeds in {seeds_to_try}")
        print(f"  Require: valid neighbors, no annuli, no zero-key, good face quality, valid simplified mesh")
        print(f"  avoid_topological_annuli={avoid_topological_annuli}, avoid_zero_key_patches={avoid_zero_key_patches}")
        print(f"  failure_log_path = {failure_log_path}")
        print("=" * 60)

    for nseeds in seeds_to_try:
        if verbose:
            print(f"\n--- Trying nseeds = {nseeds} ---")

        try:
            ms, L, slix, P, Pconn = mesh_segmentation_rw(
                m, nseeds, sig=sig, curvature_weight=curvature_weight,
                verbose=False, plot_intermediate=False
            )
        except Exception as e:
            if verbose:
                print(f"  Segmentation failed: {e}")
            failed_patterns.append({'nseeds': nseeds, 'stage': 'segmentation', 'error': str(e)})
            _append_failure_log(failure_log_path, nseeds, 'segmentation', {'error': str(e)})
            continue

        Pconn_vertex = compute_vertex_based_patch_connectivity(ms)
        is_valid, report = check_patch_neighbors_valid(Pconn_vertex, min_neighbors=min_neighbors,
                                                       verbose=verbose, allow_annular=allow_annular)

        if not is_valid:
            if verbose:
                print(f"  nseeds={nseeds}: invalid patches={report['invalid_patches']}")
            failed_patterns.append({
                'nseeds': nseeds, 'stage': 'neighbor_check',
                'invalid_patches': report['invalid_patches'], 'report': report
            })
            _append_failure_log(failure_log_path, nseeds, 'neighbor_check',
                               {'invalid_patches': report['invalid_patches']})
            continue

        if verbose:
            caps = report.get('cap_patches', [])
            annular = report.get('annular_patches', [])
            parts = [f"caps={caps}" if caps else "", f"annular={annular}" if annular else ""]
            parts = [p for p in parts if p]
            print(f"  Neighbor check PASS ({report['n_patches']} patches" +
                  (f", {', '.join(parts)}" if parts else "") + ")")

        try:
            m_seg, PM, Pconn_out = patch_info_gen(
                ms, P, Pconn,
                validate_segmentation=False,
                raise_on_invalid_segmentation=False,
                strict_simplified_mesh=False
            )
            PM['run_id'] = run_id
        except Exception as e:
            if verbose:
                print(f"  patch_info_gen failed: {e}")
            failed_patterns.append({'nseeds': nseeds, 'stage': 'patch_info_gen', 'error': str(e)})
            _append_failure_log(failure_log_path, nseeds, 'patch_info_gen', {'error': str(e)})
            continue

        pm = PM.get('pm')
        if pm is None:
            if verbose:
                print(f"  nseeds={nseeds}: PM has no 'pm' (simplified mesh)")
            failed_patterns.append({'nseeds': nseeds, 'stage': 'simplified_mesh', 'error': 'PM["pm"] is None'})
            _append_failure_log(failure_log_path, nseeds, 'simplified_mesh', {'error': 'PM["pm"] is None'})
            continue

        if avoid_topological_annuli:
            has_annuli, annular_patches = _has_topological_annuli(PM, m=m_seg)
            if has_annuli:
                if verbose:
                    print(f"  nseeds={nseeds}: topological annuli in patches {annular_patches} "
                          f"(2 boundary components) – skipping")
                failed_patterns.append({
                    'nseeds': nseeds, 'stage': 'topological_annuli',
                    'annular_patches': annular_patches,
                    'reason': 'Patch(es) with 2 boundary components (cap-enclosing or cylinder) cause foldovers',
                })
                _append_failure_log(failure_log_path, nseeds, 'topological_annuli',
                                   {'annular_patches': annular_patches})
                continue

        if avoid_zero_key_patches:
            has_zk, zero_key_patches = _has_zero_key_patches(PM)
            if has_zk:
                if verbose:
                    print(f"  nseeds={nseeds}: zero-key patches {zero_key_patches} "
                          f"(super-tapered cap – no triple junctions)")
                failed_patterns.append({
                    'nseeds': nseeds, 'stage': 'zero_key_patches',
                    'zero_key_patches': zero_key_patches,
                    'reason': 'Patch(es) with 0 key vertices cause degenerate simplified mesh geometry',
                })
                _append_failure_log(failure_log_path, nseeds, 'zero_key_patches',
                                   {'zero_key_patches': zero_key_patches})
                continue

        has_bad_faces, bad_faces = _simplified_mesh_face_quality_issues(
            pm, min_angle_deg=min_face_angle_deg, max_aspect_ratio=max_face_aspect_ratio)
        if has_bad_faces:
            if verbose:
                worst = bad_faces[:5]
                print(f"  nseeds={nseeds}: degenerate/tapered faces in simplified mesh: {worst} (and {len(bad_faces)-5} more)" if len(bad_faces) > 5 else f"  nseeds={nseeds}: degenerate/tapered faces: {bad_faces}")
            failed_patterns.append({
                'nseeds': nseeds, 'stage': 'face_quality',
                'bad_faces': bad_faces[:20],
                'reason': f'Simplified mesh has faces with min_angle<{min_face_angle_deg}deg or aspect_ratio>{max_face_aspect_ratio}',
            })
            _append_failure_log(failure_log_path, nseeds, 'face_quality',
                               {'n_bad_faces': len(bad_faces), 'sample': bad_faces[:5]})
            continue

        is_simpl_valid, issues = validate_simplified_mesh(pm, PM, verbose=verbose)
        if not is_simpl_valid:
            if verbose:
                print(f"  nseeds={nseeds}: simplified mesh INVALID - {list(issues.keys())}")
            failed_patterns.append({
                'nseeds': nseeds, 'stage': 'simplified_mesh_validation',
                'issues': issues
            })
            _append_failure_log(failure_log_path, nseeds, 'simplified_mesh_validation', issues)
            continue

        if verbose:
            caps = report.get('cap_patches', [])
            annular = report.get('annular_patches', [])
            cap_msg = f", caps={caps}" if caps else ""
            ann_msg = f", annular={annular}" if annular else ""
            print(f"\n  *** SUCCESS: nseeds = {nseeds}, {report['n_patches']} patches, "
                  f"valid simplified mesh ({len(pm.X)} verts, {len(pm.F)} faces){cap_msg}{ann_msg}")

        if plot_intermediate:
            try:
                ms.plot_segmentation_with_seeds(slix, verbose=verbose,
                    title=f'Valid (nseeds={nseeds}, {report["n_patches"]} patches)')
                if verbose:
                    print("Plotted resulting segmentation")
            except Exception as e:
                if verbose:
                    print(f"Plot failed: {e}")

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
        print(f"\nNo valid segmentation+simplified_mesh found in range {seeds_to_try}")
        print(f"  Failed {len(failed_patterns)} attempt(s). Consider increasing range or inspect failure log.")
    return None


def _append_failure_log(path, nseeds, stage, data):
    """Append a failure record to a JSON-lines log file."""
    if path is None or path == '':
        return
    try:
        import json
        record = {'nseeds': nseeds, 'stage': stage, **data}
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str) + '\n')
    except Exception:
        pass
