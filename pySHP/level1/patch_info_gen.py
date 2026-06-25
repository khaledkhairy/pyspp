"""
Generate patch information from segmented mesh
Translated from MATLAB level1/patch_info_gen.m

This module generates the complete patch data structure including:
- Patch outer edges
- Patch connectivity
- Key vertices (vertices at patch junctions)
- Sentinel vertices (key vertices delimiting edges)
- Edge chains between patches
- Simplified patch-level mesh

TERMINOLOGY (key vs sentinel)
-----------------------------
- Key vertices: Vertices where 3 or more patches meet (triple+ junctions). They are
  stored in PM['keys'] as [patch_index, vertex_index_in_mesh, face_index]. Key vertices
  are the primary anchors for the simplified mesh; every patch with multiple neighbors
  has at least two keys on each shared boundary.

- Sentinel vertices: For each patch-patch edge, the two vertices that delimit the
  boundary chain between the two patches. Stored in PM['sentinels'] as one row per
  edge: [s1, s2] (mesh indices). Sentinels mark the start and end of the vertex chain
  along that edge. They may be key vertices (if they are triple+ junctions) or plain
  boundary vertices (e.g. on a cap with only one neighbor). For the simplified mesh
  we need keys/sentinels evenly distributed along each boundary so that consecutive
  key/sentinel vertices have roughly equal numbers of border edges between them.

- On a cap patch (single neighbor, disk-like): The whole boundary is one chain. We
  place synthetic keys evenly along the full boundary (equal border-edge spacing) so
  the distribution matches multi-neighbor patches.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from ..surface_mesh import surface_mesh
from .get_border import get_border, compute_face_neighbors
from .get_center_vert import get_center_vert
from .fix_flipped_faces import fix_flipped_faces


def patch_info_gen(m, P=None, Pconn=None, validate_segmentation=True,
                   raise_on_invalid_segmentation=False, strict_simplified_mesh=False):
    """
    Generate patch information structure from segmented mesh.

    PM is the single source of truth; the simplified mesh PM['pm'] is derived
    from it. See SIMPLIFIED_MESH_ARCHITECTURE.md for invariants and compatibility.
    
    Parameters:
    -----------
    m : surface_mesh
        Mesh with face_labels populated
    P : list, optional
        Precomputed patch structures
    Pconn : sparse matrix, optional
        Precomputed patch connectivity
    validate_segmentation : bool, default True
        Check that segmentation is well-behaved (all patches have 1 or >=3 neighbors).
        Uses vertex-based connectivity. If invalid, warn or raise per
        raise_on_invalid_segmentation.
    raise_on_invalid_segmentation : bool, default False
        If True and segmentation is invalid, raise ValueError instead of warning.
    strict_simplified_mesh : bool, default False
        If True and simplified mesh fails topology validation (manifold, genus-0,
        closed), raise ValueError instead of continuing.
        
    Returns:
    --------
    m : surface_mesh
        Mesh with updated information
    PM : dict
        Patch mesh structure containing:
        - pm: simplified patch-level mesh
        - P: list of patch data [patch_mesh, face_labels, edge_face_indices]
        - Edges: patch-level edges (pairs of patch indices)
        - Xkeyind: indices of key/center vertices into m.X (-1 = fictitious)
        - Keyind: indices into PM.pm.X for key vertices
        - CVind: indices into PM.pm.X for center vertices
        - sentinels: start/end key vertices for each edge
        - edge_dat: vertex chains for each edge
        - OUT: outline vertices for each patch
        - CV: center vertex for each patch
        - keys: [patch_index, vertex_index, face_index] for key vertices
    Pconn : sparse matrix
        Patch connectivity matrix
    """
    plot_flag = False
    verbose_key_debug = True  # Debug key vertex and face generation
    
    if not hasattr(m, 'face_labels') or m.face_labels is None:
        raise ValueError("Mesh must have face_labels for patch generation")
    
    L = m.face_labels
    m.needs_edge_info = True
    m.edge_info()
    
    # Compute face neighbors if not available
    if not hasattr(m, 'face_nbrs') or m.face_nbrs is None:
        m.face_nbrs = compute_face_neighbors(m)
    
    # Ensure border_vertex is initialized and has correct size
    # border_vertex should be same length as m.X (number of vertices)
    if not hasattr(m, 'border_vertex') or m.border_vertex is None:
        m.border_vertex = np.zeros(len(m.X), dtype=int)
    elif len(m.border_vertex) != len(m.X):
        # border_vertex has wrong size - recompute it
        m, _, _ = get_border(m)
    
    # [1] and [2] Build patches if not provided
    if P is None or len(P) == 0 or Pconn is None:
        P, Pconn = build_patch_data_structure(m, L)
    
    # Get unique labels
    uL = np.unique(L)
    numpatches = len(uL)
    
    # [2a] Segmentation validation gate (see SIMPLIFIED_MESH_ARCHITECTURE.md)
    if validate_segmentation and numpatches > 0:
        from .find_valid_segmentation import (
            compute_vertex_based_patch_connectivity,
            check_patch_neighbors_valid,
        )
        Pconn_vertex = compute_vertex_based_patch_connectivity(m)
        is_valid, seg_report = check_patch_neighbors_valid(
            Pconn_vertex, min_neighbors=3, verbose=False
        )
        if not is_valid:
            inv = seg_report.get('invalid_patches', [])
            msg = (f"Segmentation has invalid patches (0 or 2 neighbors): {inv}. "
                   "Use find_valid_segmentation() to obtain a well-behaved segmentation, "
                   "or set validate_segmentation=False to skip this check.")
            if raise_on_invalid_segmentation:
                raise ValueError(msg)
            print(f'patch_info_gen: WARNING - {msg}')
    
    # [3] Detect patches with only one neighbor and build patch structure report
    if Pconn is None:
        from scipy.sparse import csr_matrix
        Pconn = csr_matrix((numpatches, numpatches))
    
    row_sums = np.array(Pconn.sum(axis=1)).flatten()
    # Patches with exactly one neighboring patch (row_sum == 1)
    single_neighbor_ind = np.where(row_sums == 1)[0]
    # Report structure; single_neighbor_edges filled after PM['Edges'] exists (in add_fictitious_keys_for_single_neighbor_edges)
    PM_report = {
        'single_neighbor_patch_indices': single_neighbor_ind,
        'single_neighbor_edges': [],
        'warnings': [],
        'n_neighbors_per_patch': row_sums,
    }
    if len(single_neighbor_ind) > 0 and numpatches > 2:
        PM_report['warnings'].append(
            f"Patches with only one neighbor: {single_neighbor_ind.tolist()}. "
            "This can cause poor initial spherical parameterization and flipped faces. "
            "Fictitious key vertices will be added along the common edge to stabilize the simplified mesh."
        )
        print('patch_info_gen: WARNING - Found patches with only one neighbor.')
        print(f'  Patches: {single_neighbor_ind.tolist()}')
        print('  Adding fictitious key vertices along common edges to improve initial parameterization.')
    
    # [4] Generate P{pix}{4-7} and pev (patch edge vertex information)
    key_ix_all = []  # Will store [patch_ix, vertex_ix, face_ix]
    
    if verbose_key_debug:
        print(f"\n{'='*60}")
        print(f"KEY VERTEX DETECTION DEBUG:")
        print(f"  Number of patches: {numpatches}")
    
    for pix in range(numpatches):
        p_edge_faces = P[pix][2]  # Patch edge faces (indices into m.F)
        pev = []  # Patch edge vertex data structure
        
        if verbose_key_debug and pix == 0:
            print(f"  Patch {pix}: {len(p_edge_faces)} edge faces")
        
        for pfix in range(len(p_edge_faces)):
            # Get vertices of this face
            vert_ix = m.F[p_edge_faces[pfix]]
            
            for vix in range(len(vert_ix)):
                # Get face members for this vertex
                fm = m.face_memb.get(vert_ix[vix], [])
                
                for fix in range(len(fm)):
                    for p2ix in range(numpatches):
                        # Check if this face is an edge face of patch p2ix
                        # IMPORTANT: Use numpy isin() for reliable array membership test
                        edge_faces_p2 = P[p2ix][2]
                        if len(edge_faces_p2) > 0 and np.isin(fm[fix], edge_faces_p2):
                            pev.append([pix, p_edge_faces[pfix], vert_ix[vix], fm[fix], p2ix])
                    
                    # Handle border vertices
                    # Check bounds before accessing border_vertex
                    vert_idx = vert_ix[vix]
                    if vert_idx < len(m.border_vertex) and m.border_vertex[vert_idx]:
                        pev.append([pix, p_edge_faces[pfix], vert_ix[vix], fm[fix], -1])  # -1 for border
        
        pev = np.array(pev) if pev else np.array([]).reshape(0, 5)
        
        if verbose_key_debug and pix == 0:
            print(f"  Patch {pix}: pev has {len(pev)} entries")
        
        # Define outline vertices
        if len(pev) > 0:
            face_edge_vert, ia, ic = np.unique(pev[:, 2], return_index=True, return_inverse=True)
            
            if verbose_key_debug and pix == 0:
                print(f"  Patch {pix}: {len(face_edge_vert)} unique edge vertices")
            
            # Find key vertices (vertices touching more than 2 patches)
            outvert = np.zeros(len(face_edge_vert), dtype=int)
            key_ix = []
            
            for outix in range(len(face_edge_vert)):
                vert = face_edge_vert[outix]
                # Get patch membership
                mask = pev[:, 2] == vert
                if np.sum(mask) > 0:
                    patch_members = np.unique(pev[mask, 4])
                    patch_members = patch_members[patch_members >= 0]  # Exclude border (-1)
                    
                    if verbose_key_debug and pix == 0 and outix < 3:
                        print(f"    Vertex {vert}: touches {len(patch_members)} patches: {patch_members}")
                    
                    if len(patch_members) > 2:
                        outvert[outix] = 1
                        face_ix = pev[ia[outix], 3] if ia[outix] < len(pev) else 0
                        key_ix.append([pix, vert, face_ix])
            
            if verbose_key_debug:
                print(f"  Patch {pix}: Found {len(key_ix)} key vertices")
            
            # P[pix] already has 3 elements [mp, mpL, mpLindx] at indices 0, 1, 2
            P[pix].append({})  # P[pix][3] - unused
            P[pix].append({'key_ix': np.array(key_ix) if key_ix else np.array([]).reshape(0, 3)})  # P[pix][4] - KEY_IX IS HERE!
            P[pix].append({})  # P[pix][5] - unused
            P[pix].append(pev)  # P[pix][6] - pev data
        else:
            if verbose_key_debug:
                print(f"  Patch {pix}: No pev entries - skipping key detection")
            P[pix].extend([{}, {'key_ix': np.array([]).reshape(0, 3)}, {}, np.array([]).reshape(0, 5)])
    
    if verbose_key_debug:
        print(f"{'='*60}\n")
    
    # [5] Generate PM.OUT: outline vertices per patch
    PM = {'OUT': {}}
    PM['patch_structure_report'] = PM_report
    for pix in range(numpatches):
        pev = P[pix][6] if len(P[pix]) > 6 else np.array([]).reshape(0, 5)
        
        if len(pev) > 0:
            # Find vertices belonging to this patch's edge faces
            all_out_verts = np.unique(pev[pev[:, 4] == pix, 2])
            outline_vert = []
            
            for vix in range(len(all_out_verts)):
                vert = all_out_verts[vix]
                vpev = pev[pev[:, 2] == vert]
                if len(np.unique(vpev[:, 4])) > 1:
                    outline_vert.append(vert)
            
            PM['OUT'][pix] = np.array(outline_vert)
        else:
            PM['OUT'][pix] = np.array([])
    
    # [6] Build PM.P, PM.Pconn and PM.keys
    PM['P'] = P
    PM['npatches'] = numpatches
    PM['Pconn'] = Pconn.copy()
    # Remove self-connections
    for i in range(numpatches):
        PM['Pconn'][i, i] = 0
    
    # Accumulate keys from all patches
    # P[pix][4] contains {'key_ix': ...} - see structure above
    keys = []
    for pix in range(numpatches):
        if len(P[pix]) > 4 and 'key_ix' in P[pix][4]:
            patch_keys = P[pix][4]['key_ix']
            if len(patch_keys) > 0:
                keys.extend(patch_keys.tolist())
    
    # Consolidate keys (remove duplicates)
    if keys:
        keys = np.array(keys)
        _, ia = np.unique(keys, axis=0, return_index=True)
        keys = keys[sorted(ia)]
    
    if verbose_key_debug:
        print(f"\n{'='*60}")
        print(f"KEY ACCUMULATION RESULT:")
        print(f"  Total keys found: {len(keys) if len(keys) > 0 else 0}")
        if len(keys) > 0:
            unique_key_verts = np.unique(keys[:, 1])
            print(f"  Unique key vertices: {len(unique_key_verts)}")
            print(f"  Key vertices: {unique_key_verts[:min(10, len(unique_key_verts))]}...")
        print(f"{'='*60}\n")
    else:
        keys = np.array([]).reshape(0, 3)
    
    PM['keys'] = keys
    
    # [6b] Record patches with zero key vertices (needed for fictitious keys and manifold)
    nkeys_per_patch = np.zeros(numpatches, dtype=int)
    if len(PM['keys']) > 0:
        for pix in range(numpatches):
            nkeys_per_patch[pix] = np.sum(PM['keys'][:, 0] == pix)
    PM_report['zero_key_patch_indices'] = np.where(nkeys_per_patch == 0)[0]
    if len(PM_report['zero_key_patch_indices']) > 0:
        PM_report['warnings'].append(
            f"Patches with zero key vertices: {PM_report['zero_key_patch_indices'].tolist()}. "
            "Fictitious keys will be added along their incident edges for a valid simplified mesh."
        )
        if verbose_key_debug:
            print(f"  Zero-key patches: {PM_report['zero_key_patch_indices'].tolist()}")
    
    # [7] Build PM.Edges: patch-level edges
    edges = []
    for rpix in range(numpatches):
        for cpix in range(rpix + 1, numpatches):
            if Pconn[rpix, cpix] > 0 or Pconn[cpix, rpix] > 0:
                edges.append([rpix, cpix])
    
    PM['Edges'] = np.array(edges) if edges else np.array([]).reshape(0, 2)
    
    # [8] Determine sentinels (start/end key vertices per edge)
    C = PM['keys']
    sentinels = []
    
    for eix in range(len(PM['Edges'])):
        pix1 = PM['Edges'][eix, 0]
        pix2 = PM['Edges'][eix, 1]
        
        # Find keys belonging to both patches
        if len(C) > 0:
            c1 = C[C[:, 0] == pix1, 1]
            c2 = C[C[:, 0] == pix2, 1]
            keys_shared = np.intersect1d(c1, c2)
            
            if len(keys_shared) >= 2:
                sentinels.append([keys_shared[0], keys_shared[1]])
            elif len(keys_shared) == 1:
                # Only one shared key - use outline vertices
                out1 = PM['OUT'].get(pix1, np.array([]))
                out2 = PM['OUT'].get(pix2, np.array([]))
                shared_out = np.intersect1d(out1, out2)
                if len(shared_out) >= 2:
                    sentinels.append([shared_out[0], shared_out[-1]])
                else:
                    sentinels.append([keys_shared[0], keys_shared[0]])
            else:
                # No shared keys - use outline vertices
                out1 = PM['OUT'].get(pix1, np.array([]))
                out2 = PM['OUT'].get(pix2, np.array([]))
                shared_out = np.intersect1d(out1, out2)
                if len(shared_out) >= 2:
                    sentinels.append([shared_out[0], shared_out[-1]])
                else:
                    sentinels.append([0, 0])
        else:
            sentinels.append([0, 0])
    
    PM['sentinels'] = np.array(sentinels) if sentinels else np.array([]).reshape(0, 2)
    
    # [8'] Handle border vertices for patch refinement
    if hasattr(m, 'border_vertex') and m.border_vertex is not None and len(m.border_vertex) > 0 and np.any(m.border_vertex):
        for pix in range(numpatches):
            # Find sentinels involving this patch that are on the border
            sntl = []
            for eix in range(len(PM['Edges'])):
                if PM['Edges'][eix, 0] == pix or PM['Edges'][eix, 1] == pix:
                    sntl.extend(PM['sentinels'][eix].tolist())
            
            sntl = np.unique(sntl)
            # Check bounds before accessing border_vertex
            valid_mask = (sntl >= 0) & (sntl < len(m.border_vertex))
            if np.any(valid_mask):
                border_sntl = sntl[valid_mask & (m.border_vertex[sntl[valid_mask].astype(int)] == 1)]
            else:
                border_sntl = np.array([])
            
            if len(border_sntl) == 2:
                PM['Edges'] = np.vstack([PM['Edges'], [pix, -1]])  # -1 for border
                PM['sentinels'] = np.vstack([PM['sentinels'], border_sntl[:2]])
    
    # [9] Generate edge chains
    PM['edge_dat'] = generate_edge_chains(m, PM)
    
    # [9a] Fix degenerate sentinels using edge chains (for edges with no shared keys / outline)
    for eix in range(len(PM['Edges'])):
        if PM['Edges'][eix, 1] < 0:
            continue
        s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
        if s1 >= 0 and s2 >= 0 and s1 != s2:
            continue
        chain = PM['edge_dat'][eix]
        if hasattr(chain, '__len__') and len(chain) >= 2:
            PM['sentinels'][eix, 0] = int(chain[0])
            PM['sentinels'][eix, 1] = int(chain[-1])
        else:
            # Build ordered shared boundary and set sentinels from it
            pix1, pix2 = int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1])
            chain = _ordered_shared_boundary_chain(m, PM, pix1, pix2)
            if len(chain) >= 2:
                PM['sentinels'][eix, 0] = int(chain[0])
                PM['sentinels'][eix, 1] = int(chain[-1])
                PM['edge_dat'][eix] = chain
    
    # [9b] Detect "neck" edges: an edge (pix1, pix2) where every vertex on the boundary
    #      between the two patches belongs only to those two patches (no triple/quadruple
    #      key vertices on that edge). Such edges need fictitious key vertices like
    #      single-neighbor edges to get a valid simplified mesh.
    neck_edges = detect_neck_edges(m, PM, P)
    PM['patch_structure_report']['neck_edges'] = neck_edges
    if len(neck_edges) > 0:
        PM['patch_structure_report']['warnings'].append(
            f"Neck edges detected (boundary exclusively between two patches): {neck_edges}. "
            "Fictitious key vertices will be added along these edges for a valid simplified mesh."
        )
        print('patch_info_gen: Found neck edges (no key vertices on boundary).')
        print(f'  Neck edge indices: {neck_edges}')
    
    # [9a'] Add synthetic shared keys for edges with no shared keys (cap/neck boundaries).
    # - Only add key (pix, v) if v is on patch pix's border (legal key).
    # - Cap edges: use FULL boundary chain of the cap so keys have equal numbers of border
    #   edges between them (same distribution as multi-neighbor patches).
    # - Non-cap: evenly distribute by arc length along the existing chain.
    zero_key_set = set(PM_report.get('zero_key_patch_indices', np.array([], dtype=int)).tolist())
    neck_set = set(neck_edges)
    new_key_rows = []
    cap_edge_full_chains = {}  # eix -> full_chain (for updating edge_dat after adding keys)
    for eix in range(len(PM['Edges'])):
        if PM['Edges'][eix, 1] < 0:
            continue
        pix1, pix2 = int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1])
        s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
        if s1 == s2 or s1 < 0 or s2 < 0:
            continue
        if len(PM['keys']) > 0:
            c1 = PM['keys'][PM['keys'][:, 0] == pix1, 1]
            c2 = PM['keys'][PM['keys'][:, 0] == pix2, 1]
            keys_shared = np.intersect1d(c1, c2)
            if len(keys_shared) >= 3:
                continue
        chain = PM['edge_dat'][eix] if eix < len(PM['edge_dat']) else np.array([])
        if not hasattr(chain, '__len__') or len(chain) < 2:
            chain = np.array([s1, s2])
        chain = np.asarray(chain).flatten()
        L = len(chain)
        is_cap_edge = pix1 in zero_key_set or pix2 in zero_key_set
        is_neck = eix in neck_set
        if is_cap_edge:
            # Use full boundary chain of the cap so keys are evenly spaced (equal border edges between them)
            cap_pix = pix1 if pix1 in zero_key_set else pix2
            full_chain = _full_boundary_chain_cap_edge(m, PM, eix, pix1, pix2, s1, s2, cap_pix)
            if len(full_chain) >= 4:
                chain = full_chain
                L = len(chain)
                cap_edge_full_chains[eix] = full_chain
            n_want = 4 if L >= 4 else 3
        elif is_neck:
            n_want = 3
        else:
            n_want = 2
        # Equal distribution: for cap use index spacing (equal border edges); else arc-length
        if L >= 2:
            if is_cap_edge and L >= 4:
                # Equal numbers of border edges between consecutive keys: indices 0, L/4, 2L/4, 3L/4
                idx = [0, L // 4, (2 * L) // 4, (3 * L) // 4]
                idx = sorted(set(idx))
                if len(idx) < n_want and L >= 3:
                    idx = [0, L // 3, (2 * L) // 3, L - 1][:n_want]
                    idx = sorted(set(idx))
            elif is_cap_edge and L >= 3:
                idx = [0, L // 2, L - 1]
                idx = sorted(set(idx))
            else:
                chain_pts = m.X[chain.astype(int)]
                arc_lengths = np.zeros(L)
                for i in range(1, L):
                    arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(chain_pts[i] - chain_pts[i-1])
                total_len = arc_lengths[-1]
                if total_len > 1e-10:
                    if n_want >= 4 and L >= 4:
                        target_lengths = [0.0, total_len / 3.0, 2.0 * total_len / 3.0, total_len]
                    elif n_want >= 3 and L >= 3:
                        target_lengths = [0.0, total_len / 2.0, total_len]
                    else:
                        target_lengths = [0.0, total_len]
                    idx = []
                    for tlen in target_lengths:
                        i_closest = np.argmin(np.abs(arc_lengths - tlen))
                        idx.append(i_closest)
                    idx = sorted(set(idx))
                else:
                    if n_want >= 4 and L >= 4:
                        idx = [0, L // 3, 2 * L // 3, L - 1]
                    elif n_want >= 3 and L >= 3:
                        idx = [0, L // 2, L - 1]
                    else:
                        idx = [0, L - 1]
                    idx = sorted(set(idx))
        else:
            idx = [0, L - 1] if L >= 2 else [0]
        n_verts = len(PM['P'][pix1][0].X) if PM['P'][pix1][0].X is not None else 0
        verts_candidates = [int(chain[i]) for i in idx if 0 <= i < L and 0 <= chain[i] < n_verts]
        # Only add a vertex if it is on BOTH patches' borders (so both patches share the same key set on this edge)
        for v in verts_candidates:
            if v < 0:
                continue
            on_pix1 = False
            on_pix2 = False
            for pix, flag in ((pix1, 'on_pix1'), (pix2, 'on_pix2')):
                patm = PM['P'][pix][0]
                if hasattr(patm, 'border_vertex') and patm.border_vertex is not None and v < len(patm.border_vertex) and patm.border_vertex[v] == 1:
                    if pix == pix1:
                        on_pix1 = True
                    else:
                        on_pix2 = True
            if on_pix1 and on_pix2:
                new_key_rows.append([pix1, v, -1])
                new_key_rows.append([pix2, v, -1])
    if new_key_rows:
        new_key_rows = np.array(new_key_rows)
        PM['keys'] = np.vstack([PM['keys'], new_key_rows])
        _, ia = np.unique(PM['keys'][:, :2], axis=0, return_index=True)
        PM['keys'] = PM['keys'][sorted(ia)]
        if verbose_key_debug:
            print(f"  Added synthetic shared keys (legal border only, evenly spaced) for edges with <3 shared keys.")
        # So that E and boundary cycles use the same order: set edge_dat for cap edges to full chain
        for eix, full_chain in cap_edge_full_chains.items():
            if eix < len(PM['edge_dat']):
                PM['edge_dat'][eix] = full_chain
    
    # [9c] Run patch type analysis (categorizes patches: cap, cylinder, multi, etc.)
    # This enables detect_cap_and_cylinder_patches to use accurate categorization (handles caps with 2 neighbors)
    try:
        from .patch_type_analysis import analyze_patch_types
        patch_type_report = analyze_patch_types(m, PM, verbose=False)
        PM['patch_structure_report']['patch_type_report'] = patch_type_report
    except Exception as e:
        if verbose_key_debug:
            print(f"  Warning: patch type analysis failed: {e}. Using fallback detection.")
        PM['patch_structure_report']['patch_type_report'] = None
    
    # [9c'] Classify cap (disk, 1-2 neighbors) and cylinder (two boundary loops) patches
    cap_patches, cylinder_patches = detect_cap_and_cylinder_patches(PM)
    PM['patch_structure_report']['cap_patches'] = cap_patches
    PM['patch_structure_report']['cylinder_patches'] = cylinder_patches
    if len(cap_patches) > 0 or len(cylinder_patches) > 0:
        if len(cap_patches) > 0:
            print(f'patch_info_gen: Cap patches (disk, 1-2 neighbors): {cap_patches}')
        if len(cylinder_patches) > 0:
            print(f'patch_info_gen: Cylinder patches (two boundary loops): {cylinder_patches}')
    
    # [10] Generate center vertex for each patch
    # Center vertices should be in the interior of patches, not on borders
    cv = np.zeros(numpatches, dtype=int)
    for pix in range(numpatches):
        p = PM['P'][pix][0]  # Patch mesh (uses same vertex indices as original mesh m)
        t = PM['OUT'].get(pix, np.array([]))  # Outline vertices (indices into original mesh m)
        cv[pix] = get_center_vert(p, t, plot_flag)  # Returns index into p.X (same as m.X)
    PM['CV'] = cv
    
    # [13] and [14] must run BEFORE [11] so that OUT_chain exists when building the simplified mesh.
    # Otherwise _patch_boundary_cycle falls back to edge walk / geometric order, which can produce
    # wrong cyclic order (e.g. patch 2 crossed triangles) when the patch has multiple neighbors.
    # [13] Update fine patches with border_vertex information
    for pix in range(numpatches):
        PM['P'][pix][0].border_vertex = np.zeros(len(PM['P'][pix][0].X), dtype=int)
        PM['P'][pix][0], _, _ = get_border(PM['P'][pix][0])
        border_verts = np.where(PM['P'][pix][0].border_vertex)[0]
        for bv in border_verts:
            if bv < len(m.border_vertex):
                m.border_vertex[bv] = 1
    
    # [14] Generate outline chains based on border information for each patch
    # This is what ultimately counts for parameterization (MATLAB lines 980-991)
    PM['OUT_chain'] = {}
    if hasattr(m, 'border_vertex') and m.border_vertex is not None and np.any(m.border_vertex):
        from .border2chain import border2chain
        for pix in range(numpatches):
            ppm = PM['P'][pix][0]  # Get the patch
            PM['OUT_chain'][pix] = border2chain(ppm)

    # [11] Generate simplified mesh
    PM = generate_simplified_mesh(m, PM)
    
    # [12] Fix flipped faces
    PM['pm'], Fnc = fix_flipped_faces(PM['pm'], verbose=False)
    
    # [12b] Add fictitious key vertices along single-neighbor and neck edges for robust initial parameterization
    PM = add_fictitious_keys_for_single_neighbor_edges(m, PM, n_fictitious=3)
    
    # [12c] Ensure simplified mesh is genus zero (required for spherical parameterization)
    PM = ensure_genus_zero_simplified_mesh(m, PM)
    
    # [12d] Validate simplified mesh structure (check for missing faces, floating vertices, etc.)
    from .validate_simplified_mesh import (
        validate_simplified_mesh,
        diagnose_patch_boundary_issues,
        verify_pm_vertex_compatibility,
    )
    is_valid, issues = validate_simplified_mesh(PM['pm'], PM, verbose=True)
    verify_pm_vertex_compatibility(PM, len(m.X), verbose=True)
    if not is_valid:
        # Diagnose patch boundary issues to help identify root causes
        diagnostics = diagnose_patch_boundary_issues(PM, verbose=True)
        if strict_simplified_mesh:
            raise ValueError(
                f"Simplified mesh validation failed: {issues}. "
                "Fix segmentation or mesh construction before map2sphere."
            )
    
    # [13] and [14] are now run before [11] (see above) so OUT_chain exists when building simplified mesh.
    
    # [15] Use outline chain to define new more accurate edge_dat entities
    # This is the refined edge chain generation (MATLAB lines 992-1054)
    # This is MORE RELIABLE than the initial edge_dat from generate_edge_chains
    # It uses border2chain to get accurate border chains, then extracts edge segments
    if hasattr(m, 'border_vertex') and m.border_vertex is not None and np.any(m.border_vertex):
        PM['patch'] = {}
        for pix in range(numpatches):
            ppm = PM['P'][pix][0]
            pmout = PM['OUT_chain'].get(pix, np.array([]))
            
            if len(pmout) == 0:
                continue
            
            # CRITICAL: Verify that pmout contains ALL border vertices
            # Get actual border vertices for this patch
            border_vertices = np.where(ppm.border_vertex)[0] if hasattr(ppm, 'border_vertex') and ppm.border_vertex is not None else np.array([])
            
            # Check if border2chain returned all border vertices
            pmout_set = set(pmout)
            border_set = set(border_vertices)
            missing_from_chain = border_set - pmout_set
            
            if len(missing_from_chain) > 0:
                # border2chain didn't return all vertices - try to fix it
                print(f'Patch {pix}: WARNING - border2chain missed {len(missing_from_chain)} border vertices')
                print(f'  Attempting to fix by including missing vertices...')
                
                # Try to insert missing vertices into the chain at appropriate positions
                # This is a fallback - ideally border2chain should work correctly
                # For now, we'll add missing vertices to the end and let the extraction handle it
                # OR: regenerate the chain with a different starting point
                from .border2chain import border2chain
                # Try a few different starting points
                for trial_start in range(min(5, len(border_vertices))):
                    pmout_try = border2chain(ppm, bix=trial_start)
                    pmout_try_set = set(pmout_try)
                    missing_try = border_set - pmout_try_set
                    if len(missing_try) < len(missing_from_chain):
                        pmout = pmout_try
                        pmout_set = set(pmout)
                        missing_from_chain = border_set - pmout_set
                        if len(missing_from_chain) == 0:
                            break
                
                # If still missing, append them (not ideal but ensures completeness)
                if len(missing_from_chain) > 0:
                    print(f'  Still missing {len(missing_from_chain)} vertices - appending to chain')
                    # Append missing vertices (they'll be included in edge chains)
                    pmout = np.concatenate([pmout, np.array(list(missing_from_chain))])
                    PM['OUT_chain'][pix] = pmout  # Update the stored chain
            
            # Get keys for this patch
            if len(PM['keys']) > 0:
                keys_o = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int)
            else:
                keys_o = np.array([], dtype=int)
            
            # Sort keys as they appear in the chain (MATLAB lines 1004-1009)
            keys = []
            for cix in range(len(pmout)):
                if pmout[cix] in keys_o:
                    keys.append(pmout[cix])
            keys = np.array(keys)
            
            if len(keys) == 0:
                # No keys found - create a single chain with all border vertices
                # This ensures all border vertices are included even without keys
                print(f'Patch {pix}: No keys found - creating single chain with all border vertices')
                PM['patch'][pix] = {
                    'key_dat': np.array([]).reshape(0, 2),
                    'edge_dat': [pmout]  # Single chain with all vertices
                }
                continue
            
            key_dat = []
            edge_dat = []
            vertices_in_chains = set()
            
            # Extract edge chains between consecutive keys (MATLAB lines 1013-1051)
            for eix in range(len(keys)):
                ed = []
                currkey = keys[eix]
                
                if eix + 1 >= len(keys):
                    nextkey = keys[0]  # Loop back to first key (MATLAB line 1017)
                else:
                    nextkey = keys[eix + 1]
                
                # Find position of currkey in pmout (MATLAB line 1021)
                pos = np.where(pmout == currkey)[0]
                if len(pos) == 0:
                    continue
                pos = pos[0]
                
                # Start building edge chain (MATLAB lines 1022-1023)
                ed.append(pmout[pos])
                vertices_in_chains.add(pmout[pos])
                chain_count = 1
                
                cont_edge = True
                while cont_edge:
                    chain_count += 1
                    pos = pos + 1
                    if pos >= len(pmout):  # Loop back (MATLAB line 1029-1030)
                        pos = 0
                    
                    ed.append(pmout[pos])
                    vertices_in_chains.add(pmout[pos])
                    
                    # Stop when we reach nextkey (MATLAB line 1045)
                    if pmout[pos] == nextkey:
                        cont_edge = False
                
                key_dat.append([currkey, nextkey])
                edge_dat.append(np.array(ed))
            
            # CRITICAL: Ensure ALL border vertices are included
            # Check if any border vertices are missing from chains
            missing_vertices = border_set - vertices_in_chains
            if len(missing_vertices) > 0:
                print(f'Patch {pix}: WARNING - {len(missing_vertices)} border vertices not in any edge chain')
                print(f'  Missing vertices (first 10): {list(missing_vertices)[:10]}')
                # Add missing vertices to the first edge chain (or create a new one)
                # This ensures they get boundary values during parameterization
                if len(edge_dat) > 0:
                    # Append missing vertices to the first chain
                    missing_arr = np.array(list(missing_vertices))
                    edge_dat[0] = np.concatenate([edge_dat[0], missing_arr])
                    print(f'  Added missing vertices to first edge chain')
                else:
                    # No chains exist - create one with all missing vertices
                    edge_dat.append(np.array(list(missing_vertices)))
                    key_dat = np.array([]).reshape(0, 2)
                    print(f'  Created new edge chain with missing vertices')
            
            if len(edge_dat) > 0:
                PM['patch'][pix] = {
                    'key_dat': np.array(key_dat),
                    'edge_dat': edge_dat
                }
    
    # Print patch structure report when single-neighbor, neck, or zero-key patches were detected
    report = PM.get('patch_structure_report')
    zk = report.get('zero_key_patch_indices', []) if report else []
    n_zk = len(zk) if hasattr(zk, '__len__') else 0
    if report and (len(report.get('single_neighbor_patch_indices', [])) > 0
                   or len(report.get('neck_edges', [])) > 0 or n_zk > 0):
        print(f"\n{'='*60}")
        print("Patch structure report (simplified mesh)")
        print(f"  Neighbors per patch: {report['n_neighbors_per_patch'].tolist()}")
        print(f"  Single-neighbor patches: {report.get('single_neighbor_patch_indices', []).tolist()}")
        print(f"  Neck edges (boundary only between two patches): {report.get('neck_edges', [])}")
        print(f"  Edges with fictitious keys: {report.get('edges_with_fictitious_keys', report.get('single_neighbor_edges', []))}")
        for w in report.get('warnings', []):
            print(f"  WARNING: {w}")
        print(f"{'='*60}\n")
    
    return m, PM, Pconn


def build_patch_data_structure(m, L):
    """
    Build patch data structures from face labels.
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh
    L : array
        Face labels
        
    Returns:
    --------
    P : list
        List of patch data structures
    Pconn : sparse matrix
        Patch connectivity
    """
    uL = np.unique(L)
    numpatches = len(uL)
    P = []
    
    # Build patches
    for lix in range(numpatches):
        label = uL[lix]
        
        # Find faces belonging to this patch
        patch_face_mask = L == label
        patch_faces = np.where(patch_face_mask)[0]
        
        # Check patch connectivity using original mesh face neighbors
        # A patch should be connected - all faces should be reachable from each other
        if hasattr(m, 'face_nbrs') and m.face_nbrs is not None and len(patch_faces) > 1:
            # Build connectivity graph for this patch
            from scipy.sparse.csgraph import connected_components
            patch_nbrs = {}
            for fix in patch_faces:
                if isinstance(m.face_nbrs, dict):
                    all_nbrs = m.face_nbrs.get(fix, [])
                else:
                    all_nbrs = m.face_nbrs[fix, :].nonzero()[1]
                # Only keep neighbors that are also in this patch
                patch_nbrs[fix] = [nbr for nbr in all_nbrs if nbr in patch_faces]
            
            # Check if patch is connected
            if len(patch_faces) > 0:
                # Build adjacency matrix for patch
                face_to_idx = {f: i for i, f in enumerate(patch_faces)}
                n_patch_faces = len(patch_faces)
                patch_adj = lil_matrix((n_patch_faces, n_patch_faces), dtype=bool)
                for fix, nbrs in patch_nbrs.items():
                    i = face_to_idx[fix]
                    for nbr in nbrs:
                        if nbr in face_to_idx:
                            j = face_to_idx[nbr]
                            patch_adj[i, j] = True
                            patch_adj[j, i] = True  # Make symmetric
                
                n_components, labels = connected_components(patch_adj.tocsr(), directed=False, return_labels=True)
                if n_components > 1:
                    print(f"WARNING: Patch {lix} (label {label}) is NOT CONNECTED! "
                          f"It has {n_components} disconnected components with sizes: "
                          f"{[np.sum(labels == i) for i in range(n_components)]}")
        
        # Create patch mesh with only this patch's faces
        mp = surface_mesh(m.X.copy(), m.F[patch_face_mask].copy())
        
        # Get border information for patch
        mp, mpL, _ = get_border(mp)
        
        # Original face indices and edge face indices
        Loindx = patch_faces
        
        # Edge faces: faces with fewer than 3 neighbors within the patch
        mpLindx = []
        if hasattr(mp, 'face_nbrs') and mp.face_nbrs is not None:
            for i in range(len(mp.F)):
                # Handle both dict and sparse matrix formats
                if isinstance(mp.face_nbrs, dict):
                    nbr_count = len(mp.face_nbrs.get(i, []))
                else:
                    # Sparse matrix format
                    nbr_count = mp.face_nbrs[i, :].nnz
                if nbr_count < 3:
                    mpLindx.append(Loindx[i])
        
        P.append([mp, mpL, np.array(mpLindx)])
    
    # Build connectivity
    Pconn = lil_matrix((numpatches, numpatches), dtype=int)
    
    for lix in range(numpatches):
        mpLindx = P[lix][2]
        
        for fix in mpLindx:
            if hasattr(m, 'face_nbrs') and m.face_nbrs is not None:
                # Handle both dict and sparse matrix formats
                if isinstance(m.face_nbrs, dict):
                    nbrs = m.face_nbrs.get(fix, [])
                else:
                    # Sparse matrix format
                    nbrs = m.face_nbrs[fix, :].nonzero()[1]
                for nbr in nbrs:
                    nbr_label = L[nbr]
                    nbr_idx = np.where(uL == nbr_label)[0]
                    if len(nbr_idx) > 0:
                        Pconn[lix, nbr_idx[0]] = 1
    
    return P, csr_matrix(Pconn)


def get_vertex_patch_membership(m, P, numpatches):
    """
    Return a mapping: vertex_index -> set of patch indices that touch that vertex
    (excluding border -1). Uses pev from each patch.
    """
    v_to_patches = {}
    for pix in range(numpatches):
        pev = P[pix][6] if len(P[pix]) > 6 else np.array([]).reshape(0, 5)
        if len(pev) == 0:
            continue
        for v in np.unique(pev[:, 2].astype(int)):
            if v not in v_to_patches:
                v_to_patches[v] = set()
            patches = np.unique(pev[pev[:, 2] == v, 4].astype(int))
            for p in patches:
                if p >= 0:
                    v_to_patches[v].add(int(p))
    return v_to_patches


def _patch_boundary_component_count(patm):
    """
    Return the number of boundary loops (connected components of the boundary) for a patch mesh.
    A patch with 2 boundary components has cylinder topology (e.g. band around a neck).
    """
    if not hasattr(patm, 'border_vertex') or patm.border_vertex is None:
        return 1
    border = np.where(patm.border_vertex)[0]
    if len(border) < 2:
        return 1 if len(border) <= 1 else 0
    border_set = set(border)
    # Boundary edges: edges of faces that appear in exactly one face (patch boundary)
    edge_count = {}
    for f in patm.F:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in [(a, b), (b, c), (c, a)]:
            if u > v:
                u, v = v, u
            edge_count[(u, v)] = edge_count.get((u, v), 0) + 1
    boundary_edges = [uv for uv, c in edge_count.items() if c == 1 and uv[0] in border_set and uv[1] in border_set]
    if len(boundary_edges) == 0:
        return 1
    # Build graph on border vertices
    adj = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    # Connected components
    visited = set()
    n_components = 0
    for start in border:
        if start in visited:
            continue
        n_components += 1
        stack = [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            for w in adj.get(u, []):
                if w not in visited:
                    visited.add(w)
                    stack.append(w)
    return n_components


def detect_cap_and_cylinder_patches(PM):
    """
    Classify patches into cap (disk, 1-2 neighbors) and cylinder (two boundary loops).
    Uses patch type analysis if available; otherwise falls back to zero-key detection.
    Returns (cap_patches, cylinder_patches) as lists of patch indices.
    """
    report = PM.get('patch_structure_report')
    if report is None:
        return [], []
    
    # Try to use patch type analysis if available (more accurate, handles caps with 2 neighbors)
    patch_type_report = report.get('patch_type_report')
    if patch_type_report is not None and 'patch_type' in patch_type_report:
        from .patch_type_analysis import PATCH_TYPE_CAP, PATCH_TYPE_CYLINDER
        patch_types = patch_type_report['patch_type']
        cap_patches = [pix for pix in range(len(patch_types)) if patch_types[pix] == PATCH_TYPE_CAP]
        cylinder_patches = [pix for pix in range(len(patch_types)) if patch_types[pix] == PATCH_TYPE_CYLINDER]
        return cap_patches, cylinder_patches
    
    # Fallback: use zero-key detection (old method, only catches caps with 0 keys)
    zero_key = report.get('zero_key_patch_indices', np.array([], dtype=int))
    cap_patches = list(zero_key.flatten()) if hasattr(zero_key, 'flatten') else list(zero_key)
    cylinder_patches = []
    for pix in range(PM['npatches']):
        patm = PM['P'][pix][0]
        n_comp = _patch_boundary_component_count(patm)
        if n_comp == 2:
            cylinder_patches.append(pix)
    return cap_patches, cylinder_patches


def detect_neck_edges(m, PM, P):
    """
    Detect "neck" edges: an edge (pix1, pix2) where every vertex on the boundary
    between the two patches belongs only to those two patches (no triple or
    quadruple key vertices on that edge).
    
    Such edges are the generalization of the single-neighbor case: the entire
    border segment is exclusive to the two patches. They require fictitious
    key vertices (2 or 3) for a valid simplified mesh, like single-neighbor edges.
    
    Parameters:
    -----------
    m : surface_mesh
        Full mesh
    PM : dict
        Patch mesh structure with Edges, edge_dat
    P : list
        Patch data (P[pix][6] = pev)
        
    Returns:
    --------
    neck_edges : list of int
        Edge indices eix for which the edge is a neck (all boundary vertices
        belong only to the two adjacent patches).
    """
    numpatches = PM['npatches']
    v_to_patches = get_vertex_patch_membership(m, P, numpatches)
    Edges = PM['Edges']
    edge_dat = PM['edge_dat']
    neck_edges = []
    
    for eix in range(len(Edges)):
        pix1, pix2 = int(Edges[eix, 0]), int(Edges[eix, 1])
        if pix2 < 0:
            continue
        chain = edge_dat[eix] if eix < len(edge_dat) else np.array([])
        if len(chain) < 2:
            continue
        allowed = {pix1, pix2}
        is_neck = True
        for v in chain:
            v = int(v)
            patches = v_to_patches.get(v, set())
            if patches != allowed:
                is_neck = False
                break
        if is_neck:
            neck_edges.append(eix)
    
    return neck_edges


def generate_edge_chains(m, PM):
    """
    Generate vertex chains for each edge between patches.
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh
    PM : dict
        Patch mesh structure
        
    Returns:
    --------
    edge_dat : list
        List of vertex chains for each edge
    """
    edge_dat = []
    
    m.edge_info()
    
    for eix in range(len(PM['Edges'])):
        pix1 = PM['Edges'][eix, 0]
        pix2 = PM['Edges'][eix, 1]
        
        if pix2 < 0:  # Border edge
            edge_dat.append(np.array([]))
            continue
        
        out = PM['OUT'].get(pix1, np.array([]))
        s1 = int(PM['sentinels'][eix, 0])
        s2 = int(PM['sentinels'][eix, 1])
        
        # Find chain from s1 to s2
        chain = find_edge_chain(m, out, s1, s2, PM, pix1, pix2)
        edge_dat.append(chain)
    
    return edge_dat


def find_edge_chain(m, out, s1, s2, PM, pix1, pix2):
    """
    Find the vertex chain between two sentinel vertices along a patch edge.
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh
    out : array
        Outline vertices
    s1, s2 : int
        Sentinel vertex indices
    PM : dict
        Patch mesh structure
    pix1, pix2 : int
        Patch indices
        
    Returns:
    --------
    chain : array
        Ordered vertex indices from s1 to s2
    """
    if s1 == s2:
        return np.array([s1])
    
    if s1 not in out or s2 not in out:
        # Try to find path through neighbors
        return np.array([s1, s2])
    
    # Try walking from s1 to s2
    chain = [s1]
    currvert = s1
    prevvert = -1
    
    max_iter = len(out) * 2
    for _ in range(max_iter):
        if currvert == s2:
            break
        
        # Get neighbors
        L = m.L.get(currvert, [])
        
        # Find next vertex that is an outline vertex
        next_vert = None
        for nbr in L:
            if nbr != prevvert and nbr in out and nbr not in chain:
                # Check if edge is on boundary between patches
                is_boundary = is_patch_boundary_edge(m, currvert, nbr, pix1, pix2)
                if is_boundary:
                    next_vert = nbr
                    break
        
        if next_vert is None:
            # Try any outline neighbor
            for nbr in L:
                if nbr != prevvert and nbr in out and nbr not in chain:
                    next_vert = nbr
                    break
        
        if next_vert is None:
            break
        
        chain.append(next_vert)
        prevvert = currvert
        currvert = next_vert
    
    if chain[-1] != s2:
        chain.append(s2)
    
    return np.array(chain)


def _full_boundary_chain_cap_edge(m, PM, eix, pix1, pix2, s1, s2, cap_pix):
    """
    For a cap edge (incident to a zero-key patch), build the full boundary chain
    so that keys can be placed with equal numbers of border edges between them.
    
    find_edge_chain(m, out, s1, s2) returns one path from s1 to s2 (often the
    shorter arc). This function returns the full closed loop: s1 -> s2 -> ... -> s1
    by concatenating the s1->s2 path and the s2->s1 path (other arc).
    
    cap_pix : int
        The patch index of the cap (zero-key patch) so we use its OUT for the loop.
    
    Returns:
    --------
    full_chain : ndarray
        Vertex indices in order along the full boundary (length = len(chain1) + len(chain2) - 2).
    """
    out = PM['OUT'].get(cap_pix, np.array([]))
    if len(out) < 2:
        return np.array([s1, s2])
    chain_s1_s2 = find_edge_chain(m, out, s1, s2, PM, pix1, pix2)
    chain_s2_s1 = find_edge_chain(m, out, s2, s1, PM, pix1, pix2)
    if len(chain_s1_s2) < 2 or len(chain_s2_s1) < 2:
        return chain_s1_s2 if len(chain_s1_s2) >= 2 else chain_s2_s1
    # Full loop: s1 ... s2 ... s1 (no duplicate at end)
    full = list(chain_s1_s2) + list(chain_s2_s1)[1:-1]
    return np.array(full, dtype=int)


def _ordered_shared_boundary_chain(m, PM, pix1, pix2):
    """
    Build an ordered vertex chain along the boundary between two patches when
    sentinels are degenerate. Uses shared outline vertices and walks the mesh.
    """
    out1 = PM['OUT'].get(pix1, np.array([]))
    out2 = PM['OUT'].get(pix2, np.array([]))
    shared = np.intersect1d(out1, out2)
    if len(shared) < 2:
        return np.array([])
    shared_set = set(shared.tolist())
    start = int(shared[0])
    chain = [start]
    curr = start
    prev = -1
    for _ in range(len(shared) + 2):
        L = m.L.get(curr, [])
        next_v = None
        for nbr in L:
            if nbr != prev and nbr in shared_set and nbr not in chain:
                if is_patch_boundary_edge(m, curr, nbr, pix1, pix2):
                    next_v = nbr
                    break
        if next_v is None:
            for nbr in L:
                if nbr != prev and nbr in shared_set and nbr not in chain:
                    next_v = nbr
                    break
        if next_v is None:
            break
        chain.append(next_v)
        prev, curr = curr, next_v
    return np.array(chain)


def is_patch_boundary_edge(m, v1, v2, pix1, pix2):
    """
    Check if an edge is on the boundary between two patches.
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh
    v1, v2 : int
        Vertex indices
    pix1, pix2 : int
        Patch indices
        
    Returns:
    --------
    is_boundary : bool
        True if edge is on patch boundary
    """
    # Get faces containing both vertices
    f1 = set(m.face_memb.get(v1, []))
    f2 = set(m.face_memb.get(v2, []))
    common_faces = list(f1.intersection(f2))
    
    if len(common_faces) != 2:
        return True  # Edge on mesh boundary
    
    # Check if the two faces belong to different patches
    labels = [m.face_labels[f] for f in common_faces]
    
    return labels[0] != labels[1]


def _patch_boundary_vertex_count(pix, PM, sentinels_arr):
    """
    Count how many boundary vertices patch pix would have in the simplified mesh
    if sentinels were sentinels_arr. (Used to avoid collapsing a neck edge if it
    would leave a patch with < 2 boundary vertices.)
    """
    boundary_mX = set()
    if len(PM['keys']) > 0:
        patch_keys = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int)
        boundary_mX.update(patch_keys.tolist())
    for eix in range(len(PM['Edges'])):
        if PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix:
            continue
        if PM['Edges'][eix, 1] < 0:
            continue
        s1 = int(sentinels_arr[eix, 0])
        s2 = int(sentinels_arr[eix, 1])
        if s1 >= 0:
            boundary_mX.add(s1)
        if s2 >= 0:
            boundary_mX.add(s2)
    # Vertices that will be in the simplified mesh = keys + unique sentinels
    in_mesh = set(PM['keys'][:, 1].astype(int).tolist()) if len(PM['keys']) > 0 else set()
    for eix in range(len(sentinels_arr)):
        if PM['Edges'][eix, 1] < 0:
            continue
        for j in (0, 1):
            v = int(sentinels_arr[eix, j])
            if v >= 0:
                in_mesh.add(v)
    return len(boundary_mX & in_mesh)


def _simplified_mesh_genus(pm):
    """Return genus of simplified mesh pm (0 = sphere). Uses Euler: chi = V - E + F, genus = (2 - chi) / 2 for closed."""
    if pm.X is None or pm.F is None or len(pm.F) == 0:
        return 0
    pm.needs_edge_info = True
    pm.edge_info()
    nV = len(pm.X)
    nF = len(pm.F)
    nE = len(pm.E) if pm.E is not None else 0
    chi = nV - nE + nF
    return max(0, (2 - chi) // 2)


def ensure_genus_zero_simplified_mesh(m, PM):
    """
    Ensure the simplified patch mesh has genus zero (sphere topology) so that
    spherical parameterization is possible. If the mesh has genus > 0 (e.g. due
    to neck patches), try collapsing one neck edge's sentinels to a single vertex
    (skip that edge) to remove a handle.
    
    Parameters:
    -----------
    m : surface_mesh
        Full-resolution mesh
    PM : dict
        Patch mesh structure with pm, sentinels, Edges, neck_edges, etc.
        
    Returns:
    --------
    PM : dict
        Updated PM with genus-zero simplified mesh if possible.
    """
    pm = PM.get('pm')
    if pm is None:
        return PM
    genus = _simplified_mesh_genus(pm)
    if genus == 0:
        return PM
    
    report = PM.get('patch_structure_report')
    neck_edges = list(report.get('neck_edges', [])) if report else []
    cap_patches = set(report.get('cap_patches', []))
    if len(neck_edges) == 0:
        if report is not None and 'warnings' in report:
            report['warnings'].append(
                "Simplified mesh has genus > 0 (not a topological sphere). "
                "Spherical parameterization may fail. Consider mesh segmentation or add sentinels."
            )
        print('patch_info_gen: WARNING - Simplified mesh genus > 0 and no neck edges to collapse.')
        return PM
    
    # Save original sentinels so we can try collapsing one neck edge at a time
    sentinels_orig = np.array(PM['sentinels'])
    Edges = PM['Edges']
    for eix in neck_edges:
        if eix >= len(sentinels_orig):
            continue
        s1 = int(sentinels_orig[eix, 0])
        s2 = int(sentinels_orig[eix, 1])
        if s1 == s2:
            continue
        # Simulate collapse: would both incident patches still have >= 2 boundary vertices?
        sentinels_try = np.array(sentinels_orig, copy=True)
        sentinels_try[eix, 1] = s1  # [s1, s1]
        pix1 = int(Edges[eix, 0])
        pix2 = int(Edges[eix, 1])
        if pix2 < 0:
            continue
        # Do not collapse neck edges incident to cap patches: caps need two distinct
        # sentinels on their single edge to form a proper boundary cycle; collapsing
        # would leave 1 sentinel and break the cap's topology.
        if pix1 in cap_patches or pix2 in cap_patches:
            continue
        n1 = _patch_boundary_vertex_count(pix1, PM, sentinels_try)
        n2 = _patch_boundary_vertex_count(pix2, PM, sentinels_try)
        if n1 < 2 or n2 < 2:
            # Collapsing would leave a patch with < 2 boundary vertices (would create floating vertices)
            continue
        # Collapse this edge's sentinels to one vertex (skip edge in simplified mesh)
        PM['sentinels'] = np.array(sentinels_orig, copy=True)
        PM['sentinels'][eix, 1] = s1  # [s1, s1]
        try:
            PM = generate_simplified_mesh(m, PM)
            PM['pm'], _ = fix_flipped_faces(PM['pm'], verbose=False)
            PM = add_fictitious_keys_for_single_neighbor_edges(m, PM, n_fictitious=3)
            genus_new = _simplified_mesh_genus(PM['pm'])
            if genus_new == 0:
                if report is not None and 'warnings' in report:
                    report['warnings'].append(
                        f"Genus reduced to 0 by collapsing sentinel for neck edge {eix} (genus-zero fix)."
                    )
                print(f'patch_info_gen: Genus reduced to 0 by collapsing neck edge {eix}.')
                return PM
        except Exception:
            pass
        # Restore and try next neck edge
        PM['sentinels'] = np.array(sentinels_orig)
    
    if report is not None and 'warnings' in report:
        report['warnings'].append(
            "Simplified mesh still has genus > 0 after trying to collapse neck edges. "
            "Spherical parameterization may fail."
        )
    print('patch_info_gen: WARNING - Could not reduce simplified mesh to genus 0.')
    PM['sentinels'] = np.array(sentinels_orig)
    # Restore mesh state (re-run with original sentinels so we don't leave PM in half-fixed state)
    if 'fictitious_per_edge' in PM:
        del PM['fictitious_per_edge']  # Force clean face generation; add_fictitious will repopulate
    PM = generate_simplified_mesh(m, PM)
    PM['pm'], _ = fix_flipped_faces(PM['pm'], verbose=False)
    PM = add_fictitious_keys_for_single_neighbor_edges(m, PM, n_fictitious=3)
    return PM


def _boundary_cycle_from_patch_faces(X, F, face_mask):
    """
    Get the boundary cycle (ordered vertex indices) for a patch from its face list.
    For a fan patch: center appears in every face; boundary = the cycle of non-center
    vertices that forms the outline. We find the center as the vertex that appears in
    ALL faces, then boundary edges = single-face edges not incident to center.
    Returns list of vertex indices in cycle order (global mesh indices), or None.
    """
    from collections import defaultdict, Counter
    face_indices = np.where(face_mask)[0]
    nf = len(face_indices)
    if nf == 0:
        return None
    edge_to_faces = defaultdict(list)
    vertex_face_count = Counter()
    for fi in face_indices:
        tri = F[fi]
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        vertex_face_count[a] += 1
        vertex_face_count[b] += 1
        vertex_face_count[c] += 1
        for u, v in [(a, b), (b, c), (c, a)]:
            if u > v:
                u, v = v, u
            edge_to_faces[(u, v)].append(fi)
    # Center = vertex that appears in every (or nearly every) face of the patch
    # For a fan of n triangles, center appears in all n faces; boundary verts appear in 1 or 2.
    center_candidates = [v for v, cnt in vertex_face_count.items() if cnt == nf]
    if len(center_candidates) == 0:
        # Fallback: vertex with highest face count
        center_candidates = [max(vertex_face_count, key=vertex_face_count.get)]
    center_set = set(center_candidates)
    # Boundary edges: appear in exactly one face and don't touch any center vertex
    single_face_edges = [e for e, flist in edge_to_faces.items() if len(flist) == 1]
    boundary_edges = [e for e in single_face_edges if not (e[0] in center_set or e[1] in center_set)]
    if len(boundary_edges) < 2:
        return None
    # Build adjacency: vertex -> list of neighbors on boundary
    adj = defaultdict(list)
    for u, v in boundary_edges:
        adj[u].append(v)
        adj[v].append(u)
    # Walk cycle from first edge
    start, next_v = boundary_edges[0]
    cycle = [start, next_v]
    used_edges = {(min(start, next_v), max(start, next_v))}
    while len(cycle) < len(boundary_edges) + 1:
        cur = cycle[-1]
        nbrs = [w for w in adj[cur] if (min(cur, w), max(cur, w)) not in used_edges]
        if not nbrs:
            if cur == start and len(cycle) >= 3:
                break
            return None
        nxt = nbrs[0]
        used_edges.add((min(cur, nxt), max(cur, nxt)))
        if nxt == start:
            break
        cycle.append(nxt)
    return cycle if cycle[0] == cycle[-1] or (len(cycle) >= 3 and cycle[-1] in adj[start]) else None


def _segment_intersect_2d(p0, p1, q0, q1):
    """True if open segment (p0,p1) and (q0,q1) intersect in 2D (xy)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    def on_segment(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))
    c1 = cross(p0, p1, q0) * cross(p0, p1, q1)
    c2 = cross(q0, q1, p0) * cross(q0, q1, p1)
    if c1 > 0 or c2 > 0:
        return False
    if c1 == 0 and c2 == 0:
        return (on_segment(p0, p1, q0) or on_segment(p0, p1, q1) or
                on_segment(q0, q1, p0) or on_segment(q0, q1, p1))
    return True


def diagnose_simplified_patch_self_intersection(PM):
    """
    For each patch, check if its simplified-mesh boundary polygon is self-intersecting
    (scrambled). Uses the patch's faces to extract the boundary cycle, projects to
    best-fit plane, and tests non-adjacent edge pairs for intersection.

    Returns
    -------
    dict
        pix -> {'is_self_intersecting': bool, 'boundary_cycle': list, 'intersecting_pairs': list,
                'n_boundary': int, 'message': str}
    """
    pm = PM.get('pm')
    if pm is None or pm.X is None or pm.F is None:
        return {}
    fl = getattr(pm, 'face_labels', None)
    if fl is None:
        fl = np.zeros(len(pm.F), dtype=int)
    else:
        fl = np.asarray(fl).flatten()
    X, F = np.asarray(pm.X), np.asarray(pm.F)
    cylinder_patches = set(PM.get('patch_structure_report', {}).get('cylinder_patches', []))
    result = {}
    for pix in range(PM['npatches']):
        face_mask = (fl == pix)
        if not np.any(face_mask):
            result[pix] = {'is_self_intersecting': False, 'boundary_cycle': [], 'intersecting_pairs': [],
                           'n_boundary': 0, 'message': 'no faces'}
            continue
        if pix in cylinder_patches:
            result[pix] = {'is_self_intersecting': False, 'boundary_cycle': [], 'intersecting_pairs': [],
                           'n_boundary': 0, 'message': 'cylinder (two rings, skip)'}
            continue
        cycle = _boundary_cycle_from_patch_faces(X, F, face_mask)
        if cycle is None or len(cycle) < 3:
            result[pix] = {'is_self_intersecting': False, 'boundary_cycle': cycle or [], 'intersecting_pairs': [],
                           'n_boundary': len(cycle) if cycle else 0, 'message': 'no closed boundary'}
            continue
        # Remove duplicate closing vertex if present
        if cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        n = len(cycle)
        # Project to best-fit plane (PCA: two principal components)
        pts = X[cycle]
        centroid = np.mean(pts, axis=0)
        C = np.cov((pts - centroid).T)
        try:
            w, v = np.linalg.eigh(C)
            # Two axes spanning the plane (largest two eigenvalues)
            idx = np.argsort(w)[::-1]
            u1, u2 = v[:, idx[0]], v[:, idx[1]]
            pts_2d = np.column_stack([(pts - centroid) @ u1, (pts - centroid) @ u2])
        except Exception:
            pts_2d = pts[:, :2]
        # Check non-adjacent edges for intersection
        intersecting_pairs = []
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = pts_2d[i], pts_2d[(i + 1) % n]
                c, d = pts_2d[j], pts_2d[(j + 1) % n]
                if _segment_intersect_2d(a, b, c, d):
                    intersecting_pairs.append((int(i), int(j)))
        is_bad = len(intersecting_pairs) > 0
        result[pix] = {
            'is_self_intersecting': is_bad,
            'boundary_cycle': cycle,
            'intersecting_pairs': intersecting_pairs,
            'n_boundary': n,
            'message': f'self-intersecting ({len(intersecting_pairs)} crossing pairs)' if is_bad else 'OK',
        }
    return result


def _print_patch_self_intersection_report(report):
    """Print a short summary of patch self-intersection diagnostic."""
    if not report:
        return
    print("\n" + "=" * 60)
    print("Patch self-intersection diagnostic (simplified mesh):")
    for pix in sorted(report.keys()):
        r = report[pix]
        status = "SCRAMBLED" if r['is_self_intersecting'] else "OK"
        print(f"  Patch {pix}: {status}  (n_boundary={r['n_boundary']}, {r['message']})")
        if r.get('intersecting_pairs'):
            print(f"    Intersecting edge pairs (cycle indices): {r['intersecting_pairs'][:10]}")
    print("=" * 60 + "\n")


def add_fictitious_keys_for_single_neighbor_edges(m, PM, n_fictitious=3):
    """
    For patches that have only one neighbor, subdivide the common edge in the simplified
    mesh by inserting fictitious key vertices (equally spaced along the edge chain in 3D).
    This yields a more robust initial spherical parameterization and helps avoid flipped faces.
    
    Parameters:
    -----------
    m : surface_mesh
        Full-resolution mesh (for 3D positions of edge chain)
    PM : dict
        Patch mesh structure with pm, Edges, sentinels, edge_dat, Xkeyind, patch_structure_report
    n_fictitious : int
        Number of fictitious vertices to insert per single-neighbor edge (default 3)
        
    Returns:
    --------
    PM : dict
        Updated PM with modified pm (and Xkeyind); report updated with single_neighbor_edges list.
    """
    report = PM.get('patch_structure_report')
    if report is None:
        return PM
    
    single_ind = report.get('single_neighbor_patch_indices', np.array([], dtype=int))
    single_set = set(single_ind.tolist()) if len(single_ind) > 0 else set()
    zk = report.get('zero_key_patch_indices', np.array([], dtype=int))
    zero_key_set = set(zk.tolist()) if hasattr(zk, 'tolist') else set(zk) if zk is not None else set()
    Edges = PM['Edges']
    sentinels = PM['sentinels']
    edge_dat = PM['edge_dat']
    pm = PM['pm']
    Xkeyind = PM['Xkeyind']
    npatches = PM['npatches']
    
    # Identify edges where at least one patch has only one neighbor (exclude border edges pix2 < 0)
    single_neighbor_edges = []
    for eix in range(len(Edges)):
        pix1, pix2 = int(Edges[eix, 0]), int(Edges[eix, 1])
        if pix2 < 0:
            continue
        if pix1 in single_set or pix2 in single_set:
            single_neighbor_edges.append(eix)
    
    # Also include "neck" edges and edges incident to cap patches (zero-key or low-neighbor caps)
    # Exclude cap patches with many neighbors (>= 5): they use cap_crown_data and already
    # have proper face generation. Adding fictitious keys would conflict with the crown faces.
    neck_edges = report.get('neck_edges', [])
    cap_edges = []
    cap_patches = set(report.get('cap_patches', []))
    n_neighbors_report = report.get('n_neighbors_per_patch', np.zeros(PM['npatches'], dtype=int))
    # Only target edges for caps with few neighbors (< 5); caps with >= 5 use crown centers
    cap_patches_low_neighbor = {pix for pix in cap_patches if n_neighbors_report[pix] < 5}
    for eix in range(len(Edges)):
        pix1, pix2 = int(Edges[eix, 0]), int(Edges[eix, 1])
        if pix2 < 0:
            continue
        if pix1 in cap_patches_low_neighbor or pix2 in cap_patches_low_neighbor:
            cap_edges.append(eix)
    neck_set = set(neck_edges)
    # Skip edges that already have 3+ shared keys (e.g. cap edge after synthetic keys)
    edges_with_enough_keys = set()
    if len(PM['keys']) > 0:
        for eix in range(len(Edges)):
            pix1, pix2 = int(Edges[eix, 0]), int(Edges[eix, 1])
            if pix2 < 0:
                continue
            c1 = set(PM['keys'][PM['keys'][:, 0] == pix1, 1].astype(int).tolist())
            c2 = set(PM['keys'][PM['keys'][:, 0] == pix2, 1].astype(int).tolist())
            if len(c1 & c2) >= 3:
                edges_with_enough_keys.add(eix)
    # Cap patches: disk-like patches (zero-key or 2-neighbor caps); we need at least 3 fictitious on their edges.
    # Cylinder patches: ensure the "other" border (non-neck) has >= 3 keys by adding fictitious on one non-neck edge per cylinder.
    cylinder_patches = set(report.get('cylinder_patches', []))
    cylinder_other_edges = []
    for pix in cylinder_patches:
        for eix in range(len(Edges)):
            pix1, pix2 = int(Edges[eix, 0]), int(Edges[eix, 1])
            if pix2 < 0 or (pix1 != pix and pix2 != pix):
                continue
            if eix in neck_set:
                continue
            cylinder_other_edges.append(eix)
            break  # one edge per cylinder for the "other" border
    edges_to_subdivide = sorted(
        (set(single_neighbor_edges) | set(neck_edges) | set(cap_edges) | set(cylinder_other_edges))
        - edges_with_enough_keys
    )
    # Process neck edges first so the crown is built before other edges (manifold genus-zero)
    def edge_priority(eix):
        return (0 if eix in neck_set else 1, eix)
    edges_ordered = sorted(edges_to_subdivide, key=edge_priority)
    report['single_neighbor_edges'] = single_neighbor_edges
    report['edges_with_fictitious_keys'] = edges_to_subdivide  # single-neighbor, neck, cap, and cylinder edges
    
    if len(edges_to_subdivide) == 0:
        return PM
    
    # Map (i1, i2) -> list of (face_index, cv, order) where order is 'i1_i2' or 'i2_i1'
    def find_faces_with_edge(F, i1, i2):
        out = []
        for fix in range(len(F)):
            f = F[fix]
            if i1 in f and i2 in f:
                cv = f[0] if f[0] != i1 and f[0] != i2 else (f[1] if f[1] != i1 and f[1] != i2 else f[2])
                order = 'i1_i2' if (f[0] == i1 and f[1] == i2) or (f[1] == i1 and f[2] == i2) or (f[2] == i1 and f[0] == i2) else 'i2_i1'
                out.append((fix, cv, order))
        return out
    
    new_X = list(pm.X)
    new_Xkeyind = list(Xkeyind)
    new_F = list(pm.F)
    new_face_labels = list(pm.face_labels) if hasattr(pm, 'face_labels') and pm.face_labels is not None else []
    if len(new_face_labels) != len(new_F):
        new_face_labels = [0] * len(new_F)
    
    # Store fictitious vertex indices per edge (for boundary cycle building)
    PM['fictitious_per_edge'] = {}  # eix -> [simplified_mesh_indices] in order s1 -> ... -> s2
    
    # Process each edge: neck edges get 4 crown sentinels (cylinder-like), others get n_fictitious (default 3)
    for eix in edges_ordered:
        s1, s2 = int(sentinels[eix, 0]), int(sentinels[eix, 1])
        if s1 == s2:
            continue
        chain = edge_dat[eix]
        if len(chain) < 2:
            continue
        i1_cands = np.where(np.array(new_Xkeyind) == s1)[0]
        i2_cands = np.where(np.array(new_Xkeyind) == s2)[0]
        if len(i1_cands) == 0 or len(i2_cands) == 0:
            continue
        i1, i2 = int(i1_cands[0]), int(i2_cands[0])
        
        # Neck edges: 4 crown sentinels for a proper cylindrical band. Cap patches: at least 3 fictitious. Others: n_fictitious
        is_cap_edge = pix1 in cap_patches or pix2 in cap_patches
        n_crown = 4 if eix in neck_set else max(3, n_fictitious) if is_cap_edge else n_fictitious
        
        chain_pts = m.X[chain]
        n_chain = len(chain_pts)
        if n_chain < 2:
            continue
        ts = np.linspace(1.0 / (n_crown + 1), n_crown / (n_crown + 1), n_crown)
        fict_pts = []
        for t in ts:
            idx_float = t * (n_chain - 1)
            i_low = int(np.floor(idx_float))
            i_high = min(i_low + 1, n_chain - 1)
            u = idx_float - i_low
            pt = (1 - u) * chain_pts[i_low] + u * chain_pts[i_high]
            fict_pts.append(pt)
        
        n_prev = len(new_X)
        for pt in fict_pts:
            new_X.append(pt)
            new_Xkeyind.append(-1)  # Fictitious vertex
        new_inds = list(range(n_prev, n_prev + n_crown))
        
        # Store fictitious vertex indices for this edge (in order s1 -> crown -> s2)
        PM['fictitious_per_edge'][eix] = new_inds.copy()
        
        # Find faces that contain edge (i1, i2). Allow 1 or 2 (crown even if one patch missing)
        face_entries = find_faces_with_edge(new_F, i1, i2)
        if len(face_entries) < 1:
            continue
        
        # Replace each face with (n_crown+1) triangular faces (crown/cylinder band)
        for fix, cv, order in face_entries:
            fl = new_face_labels[fix] if fix < len(new_face_labels) else 0
            if order == 'i1_i2':
                seq = [i1] + new_inds + [i2]   # s1 -> f1 -> f2 -> ... -> fN -> s2
            else:
                seq = [i2] + list(reversed(new_inds)) + [i1]   # s2 -> fN -> ... -> f1 -> s1
            new_faces = []
            for k in range(len(seq) - 1):
                new_faces.append([cv, seq[k], seq[k + 1]])  # fan: center to crown segment
            new_F[fix] = None
            new_face_labels[fix] = None
            for nf in new_faces:
                new_F.append(nf)
                new_face_labels.append(fl)
        
        # Compact so next edge sees updated face list
        new_F = [f for f in new_F if f is not None]
        new_face_labels = [l for l in new_face_labels if l is not None]
    
    pm.X = np.array(new_X)
    pm.F = np.array(new_F)
    if hasattr(pm, 'face_labels'):
        pm.face_labels = np.array(new_face_labels)
    PM['Xkeyind'] = np.array(new_Xkeyind)
    pm.t = np.zeros(len(pm.X))
    pm.p = np.zeros(len(pm.X))
    if hasattr(pm, 'border_vertex') and pm.border_vertex is not None:
        n_new = len(pm.X) - len(pm.border_vertex)
        pm.border_vertex = np.concatenate([pm.border_vertex, np.zeros(n_new, dtype=int)])
    pm.needs_edge_info = True
    pm.needs_map2sphere = True
    return PM


def _insert_crown_sentinels_in_cycle(cycle, pix, PM, Xkeyind, nkeys):
    """
    Insert crown sentinels (fictitious vertices) into the boundary cycle for patch pix.
    For each edge incident to pix that has fictitious vertices, insert them between
    the two sentinels in the correct order (s1 -> crown -> s2).
    """
    if 'fictitious_per_edge' not in PM or len(PM['fictitious_per_edge']) == 0:
        return cycle
    if len(cycle) < 2:
        return cycle
    
    # Build mapping: edge -> (s1_simpl, s2_simpl, fict_inds) for edges incident to pix with crown
    edges_with_crown = {}
    for eix in range(len(PM['Edges'])):
        if PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix:
            continue
        if PM['Edges'][eix, 1] < 0:
            continue
        if eix not in PM.get('fictitious_per_edge', {}):
            continue
        s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
        s1_idx = np.where(Xkeyind == s1)[0]
        s2_idx = np.where(Xkeyind == s2)[0]
        s1_simpl = int(s1_idx[0]) if len(s1_idx) > 0 and s1_idx[0] < nkeys else None
        s2_simpl = int(s2_idx[0]) if len(s2_idx) > 0 and s2_idx[0] < nkeys else None
        if s1_simpl is not None and s2_simpl is not None:
            # Only include fictitious vertices that exist in the current mesh (check Xkeyind length)
            max_valid_idx = len(Xkeyind) - 1
            fict_inds = PM['fictitious_per_edge'][eix]
            valid_fict_inds = [fi for fi in fict_inds if fi <= max_valid_idx]
            if len(valid_fict_inds) > 0:
                edges_with_crown[eix] = (s1_simpl, s2_simpl, valid_fict_inds)
    
    if len(edges_with_crown) == 0:
        return cycle
    
    # Build new cycle by inserting crowns: iterate through original cycle, insert crown when we see s1->s2 or s2->s1
    cycle_new = []
    inserted_edges = set()  # Track which edges we've inserted crown for
    i = 0
    while i < len(cycle):
        v = cycle[i]
        cycle_new.append(v)
        # Check if this vertex and the next form a sentinel pair for an edge with crown
        next_i = (i + 1) % len(cycle)
        next_v = cycle[next_i]
        # Check all edges with crown to see if (v, next_v) matches (s1, s2) or (s2, s1)
        for eix, (s1_simpl, s2_simpl, fict_inds) in edges_with_crown.items():
            if eix in inserted_edges:
                continue
            if (v == s1_simpl and next_v == s2_simpl) or (i == len(cycle) - 1 and v == s1_simpl and cycle[0] == s2_simpl):
                # Forward: s1 -> crown -> s2
                cycle_new.extend(fict_inds)
                inserted_edges.add(eix)
                break
            elif (v == s2_simpl and next_v == s1_simpl) or (i == len(cycle) - 1 and v == s2_simpl and cycle[0] == s1_simpl):
                # Reverse: s2 -> crown -> s1
                cycle_new.extend(reversed(fict_inds))
                inserted_edges.add(eix)
                break
        i += 1
    
    return cycle_new


def sync_cylinder_neighbor_boundary_cycles(PM):
    """
    After cylinder (and any patch) key vertices are finalized, build an authoritative
    boundary cycle for every patch that shares an edge with a cylinder. Uses edge_dat
    as the single source of truth for vertex order along each edge, and OUT_chain for
    the cyclic order of edges around the patch. This avoids ripple effects (patch 2, 4,
    gaps) when cylinder keys were added.

    Stores PM['boundary_cycle_mX'][pix] = list of m.X vertex indices in cycle order
    (keys+sentinels only). _patch_boundary_cycle uses this when present.

    Returns:
        PM with boundary_cycle_mX populated for cylinder neighbors.
    """
    report = PM.get('patch_structure_report')
    if report is None:
        return PM
    cylinder_patches = set(report.get('cylinder_patches', []))
    if not cylinder_patches:
        return PM
    Edges = PM.get('Edges', np.array([]).reshape(0, 2))
    edge_dat = PM.get('edge_dat', [])
    if len(edge_dat) == 0:
        return PM
    neighbors_cylinder = set()
    for eix in range(len(Edges)):
        p1, p2 = int(Edges[eix, 0]), int(Edges[eix, 1])
        if p2 < 0:
            continue
        if p1 in cylinder_patches:
            neighbors_cylinder.add(p2)
        if p2 in cylinder_patches:
            neighbors_cylinder.add(p1)

    def edge_for_pair(va, vb, pix):
        va, vb = int(va), int(vb)
        for eix in range(len(Edges)):
            if (Edges[eix, 0] != pix and Edges[eix, 1] != pix) or Edges[eix, 1] < 0:
                continue
            ch = edge_dat[eix] if eix < len(edge_dat) else []
            if not hasattr(ch, '__len__') or len(ch) < 2:
                continue
            ch = np.asarray(ch).flatten().tolist()
            try:
                ia = ch.index(va)
                ib = ch.index(vb)
                if abs(ia - ib) == 1 or (ia == 0 and ib == len(ch) - 1) or (ib == 0 and ia == len(ch) - 1):
                    return eix
            except (ValueError, TypeError):
                pass
        return None

    if 'boundary_cycle_mX' not in PM:
        PM['boundary_cycle_mX'] = {}

    for pix in neighbors_cylinder:
        boundary_mX = set()
        if len(PM.get('keys', [])) > 0:
            patch_keys = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int)
            boundary_mX.update(patch_keys.tolist())
        for eix in range(len(Edges)):
            if Edges[eix, 0] != pix and Edges[eix, 1] != pix or Edges[eix, 1] < 0:
                continue
            s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
            if s1 >= 0:
                boundary_mX.add(s1)
            if s2 >= 0:
                boundary_mX.add(s2)

        chain = PM.get('OUT_chain') and PM['OUT_chain'].get(pix)
        if chain is None or len(chain) < 2:
            continue
        chain = np.asarray(chain).flatten()
        segments = []
        cur_eix = None
        cur_list = []
        for i in range(len(chain)):
            v = int(chain[i])
            v_next = int(chain[(i + 1) % len(chain)])
            e = edge_for_pair(v, v_next, pix)
            if e is not None:
                if e != cur_eix:
                    if cur_eix is not None and len(cur_list) > 0:
                        segments.append((cur_eix, list(cur_list)))
                    cur_eix = e
                    cur_list = [v] if v in boundary_mX else []
                if v in boundary_mX and (not cur_list or cur_list[-1] != v):
                    cur_list.append(v)
        if cur_eix is not None and len(cur_list) > 0:
            segments.append((cur_eix, list(cur_list)))
        if len(segments) == 0:
            continue

        cycle_mX = []
        for eix, _ in segments:
            ch = edge_dat[eix] if eix < len(edge_dat) else []
            if not hasattr(ch, '__len__') or len(ch) < 2:
                continue
            ch = np.asarray(ch).flatten().tolist()
            # Use full set of boundary vertices on this edge (edge_dat order) so we never
            # split an edge into two segments when OUT_chain transition isn't recognized.
            ordered = [int(v) for v in ch if int(v) in boundary_mX]
            if len(ordered) == 0:
                continue
            if cycle_mX and ordered[0] == cycle_mX[-1]:
                ordered = ordered[1:]
            cycle_mX.extend(ordered)

        if len(cycle_mX) >= 2 and set(cycle_mX) == boundary_mX and len(cycle_mX) == len(boundary_mX):
            PM['boundary_cycle_mX'][pix] = cycle_mX

    return PM


def _boundary_cycle_from_edge_chains(pix, nkeys, Xkeyind, PM, boundary_mX, mX_to_simpl, boundary_simpl_set):
    """
    Build boundary cycle for patch pix so key/sentinel order along each edge
    matches edge_dat (same order as used by neighbors, e.g. cylinder patch rings).
    Uses OUT_chain to get cyclic order of edges, then for each edge segment
    uses edge_dat order so the interface with cylinder/other patches is consistent.

    Returns:
        list of simplified indices (keys+sentinels only, no crown yet), or None if not buildable.
    """
    chain = PM.get('OUT_chain') and PM['OUT_chain'].get(pix)
    edge_dat = PM.get('edge_dat', [])
    if chain is None or len(chain) < 2 or len(edge_dat) == 0:
        return None
    chain = np.asarray(chain).flatten()
    # Which edge does each consecutive pair (chain[i], chain[i+1]) belong to?
    # eix has chain; chain has mesh indices. So for (v_a, v_b) find eix where both in edge_dat[eix] and consecutive.
    def edge_for_pair(va, vb):
        va, vb = int(va), int(vb)
        for eix in range(len(PM['Edges'])):
            if (PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix) or PM['Edges'][eix, 1] < 0:
                continue
            ch = edge_dat[eix] if eix < len(edge_dat) else []
            if not hasattr(ch, '__len__') or len(ch) < 2:
                continue
            ch = np.asarray(ch).flatten().tolist()
            try:
                ia = ch.index(va)
                ib = ch.index(vb)
                if abs(ia - ib) == 1 or (ia == 0 and ib == len(ch) - 1) or (ib == 0 and ia == len(ch) - 1):
                    return eix
            except (ValueError, TypeError):
                pass
        return None
    # Segment OUT_chain into runs by edge
    segments = []  # list of (eix, list of m.X indices in OUT_chain order)
    cur_eix = None
    cur_list = []
    for i in range(len(chain)):
        v = int(chain[i])
        v_next = int(chain[(i + 1) % len(chain)])
        e = edge_for_pair(v, v_next)
        if e is not None:
            if e != cur_eix:
                if cur_eix is not None and len(cur_list) > 0:
                    segments.append((cur_eix, list(cur_list)))
                cur_eix = e
                cur_list = [v] if v in boundary_mX else []
            if v in boundary_mX and (not cur_list or cur_list[-1] != v):
                cur_list.append(v)
    if cur_eix is not None and len(cur_list) > 0:
        segments.append((cur_eix, list(cur_list)))
    if len(segments) == 0:
        return None
    # For each segment, use full set of boundary vertices on this edge in edge_dat order
    # (so we never split an edge when OUT_chain transition isn't recognized).
    cycle_mX = []
    for eix, _ in segments:
        ch = edge_dat[eix] if eix < len(edge_dat) else []
        if not hasattr(ch, '__len__') or len(ch) < 2:
            continue
        ch = np.asarray(ch).flatten().tolist()
        ordered = [int(v) for v in ch if int(v) in boundary_mX]
        if len(ordered) == 0:
            continue
        if cycle_mX and ordered[0] == cycle_mX[-1]:
            ordered = ordered[1:]
        cycle_mX.extend(ordered)
    # Map to simplified indices
    cycle_simpl = []
    for v in cycle_mX:
        v = int(v)
        si = mX_to_simpl.get(v)
        if si is not None and (not cycle_simpl or cycle_simpl[-1] != si):
            cycle_simpl.append(si)
    if len(cycle_simpl) < 2:
        return None
    # Ensure we have all boundary vertices (set equality)
    if set(cycle_simpl) != boundary_simpl_set:
        return None
    return cycle_simpl


def _resample_ring_for_cylinder(ring, target_len):
    """
    Resample a ring to target_len vertices for cylinder band generation.
    Avoids consecutive duplicates (which would create degenerate quads).
    When upsampling L->L+k, appends ring[1], ring[2], ... so the repeat never
    falls on (i, i+1); when downsampling, uses uniform sampling.
    """
    L = len(ring)
    if L == target_len:
        return list(ring)
    if L == 0:
        return [ring[0]] * target_len if ring else []
    if target_len < L:
        # Downsample: uniform sampling, no consecutive dup possible
        return [ring[int(round(i * L / target_len)) % L] for i in range(target_len)]
    # Upsample L -> target_len (target_len > L). Avoid ring[i]==ring[i+1].
    # Append ring[1], ring[2], ... for each extra slot so repeat is never consecutive.
    k = target_len - L
    extra = [ring[(1 + j) % L] for j in range(k)]
    return list(ring) + extra


def _zipper_triangulate_rings(r1, r2, X):
    """
    Triangulate the annular band between two cyclic vertex rings of (potentially)
    different sizes using a zipper / ear-clipping approach.

    Produces exactly len(r1) + len(r2) triangles that:
      - cover every edge on both rings exactly once,
      - use no center vertex,
      - introduce no duplicate vertices,
      - form a watertight, manifold band with χ = 0.

    At each step the algorithm must advance one pointer around r1 or r2.
    It picks whichever triangle has the shorter diagonal (greedy minimum-weight).

    Parameters
    ----------
    r1 : list[int]
        Outer ring vertex indices (simplified mesh), length n1 >= 3.
    r2 : list[int]
        Inner ring vertex indices (simplified mesh), length n2 >= 2.
    X : ndarray, shape (V, 3)
        Vertex positions for distance computation.

    Returns
    -------
    tris : list[list[int]]
        Triangle list [[a, b, c], ...], len = n1 + n2.
    """
    n1, n2 = len(r1), len(r2)
    tris = []
    i, j = 0, 0          # current position on each ring
    adv_i, adv_j = 0, 0  # how many steps we have advanced on each ring

    while adv_i < n1 or adv_j < n2:
        i_curr = i % n1
        j_curr = j % n2
        i_next = (i + 1) % n1
        j_next = (j + 1) % n2

        can_i = adv_i < n1
        can_j = adv_j < n2

        if can_i and can_j:
            # Choose shorter diagonal
            d_i = np.linalg.norm(X[r1[i_next]] - X[r2[j_curr]])
            d_j = np.linalg.norm(X[r1[i_curr]] - X[r2[j_next]])
            if d_i <= d_j:
                tris.append([r1[i_curr], r1[i_next], r2[j_curr]])
                i += 1; adv_i += 1
            else:
                tris.append([r1[i_curr], r2[j_next], r2[j_curr]])
                j += 1; adv_j += 1
        elif can_i:
            tris.append([r1[i_curr], r1[i_next], r2[j_curr]])
            i += 1; adv_i += 1
        else:
            tris.append([r1[i_curr], r2[j_next], r2[j_curr]])
            j += 1; adv_j += 1

    return tris


def _patch_boundary_cycle(pix, nkeys, Xkeyind, E, PM, X_verts=None, verbose_fail=False):
    """
    Build the ordered boundary cycle for patch pix in simplified mesh indices.
    Includes both key vertices and sentinel vertices so neck/single-neighbor edges
    are correctly represented (sentinels lie between keys along the boundary).
    Also includes crown sentinels (fictitious vertices) for neck edges to form a proper cylinder.

    Prefer PM['OUT_chain'][pix] when available: it gives the correct cyclic order
    of border vertices on the patch (in m.X indices). We filter to keys+sentinels
    and map to simplified indices so neck/single-neighbor patches get the right order.
    Fall back to edge-based walk when OUT_chain is missing or insufficient.
    Final fallback: order boundary vertices by angle around patch center (geometric).

    Returns:
    --------
    cycle : list of int
        Simplified mesh indices in boundary order (cycle), or empty list if cannot build.
    """
    # Boundary vertices for this patch: keys + sentinels from incident edges (m.X indices)
    boundary_mX = set()
    if len(PM['keys']) > 0:
        patch_keys = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int)
        boundary_mX.update(patch_keys.tolist())
    for eix in range(len(PM['Edges'])):
        if PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix:
            continue
        if PM['Edges'][eix, 1] < 0:
            continue
        s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
        if s1 >= 0:
            boundary_mX.add(s1)
        if s2 >= 0:
            boundary_mX.add(s2)
    # Map boundary m.X indices to simplified mesh indices (only those that exist in simplified mesh)
    mX_to_simpl = {}
    boundary_simpl_set = set()
    for v in boundary_mX:
        idx = np.where(Xkeyind == v)[0]
        if len(idx) > 0 and idx[0] < nkeys:
            si = int(idx[0])
            mX_to_simpl[v] = si
            boundary_simpl_set.add(si)
    # Also include fictitious vertices (crown sentinels) for edges incident to this patch
    # Only include them if they exist in the current mesh (check X_verts length if provided, or Xkeyind length)
    max_vert_idx = len(Xkeyind) - 1  # Fictitious vertices are appended, so check Xkeyind length
    if 'fictitious_per_edge' in PM:
        for eix in range(len(PM['Edges'])):
            if PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix:
                continue
            if PM['Edges'][eix, 1] < 0:
                continue
            if eix in PM['fictitious_per_edge']:
                fict_inds = PM['fictitious_per_edge'][eix]
                # Only include fictitious vertices that exist in the current mesh
                valid_fict = [fi for fi in fict_inds if fi <= max_vert_idx]
                boundary_simpl_set.update(valid_fict)
    if len(boundary_simpl_set) < 2:
        if verbose_fail:
            print(f"    _patch_boundary_cycle(patch {pix}): boundary_simpl_set has {len(boundary_simpl_set)} vertices (need >= 2); boundary_mX size = {len(boundary_mX)}")
        return []

    # Single-edge patches (caps): derive boundary directly from the edge_dat chain so
    # the vertex order matches the neighbor's boundary cycle (prevents edge mismatch).
    incident_edges = [(eix, int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1]))
                      for eix in range(len(PM['Edges']))
                      if (int(PM['Edges'][eix, 0]) == pix or int(PM['Edges'][eix, 1]) == pix)
                         and int(PM['Edges'][eix, 1]) >= 0]
    if len(incident_edges) == 1:
        eix_only, _, _ = incident_edges[0]
        chain_dat = PM['edge_dat'][eix_only] if eix_only < len(PM['edge_dat']) else np.array([])
        if hasattr(chain_dat, '__len__') and len(chain_dat) >= 2:
            chain_dat = np.asarray(chain_dat).flatten()
            cycle_from_chain = []
            seen_chain = set()
            for v in chain_dat:
                v = int(v)
                si = mX_to_simpl.get(v)
                if si is not None and si not in seen_chain:
                    cycle_from_chain.append(si)
                    seen_chain.add(si)
            if len(cycle_from_chain) >= 3:
                cycle_with_crown = _insert_crown_sentinels_in_cycle(cycle_from_chain, pix, PM, Xkeyind, nkeys)
                if verbose_fail:
                    print(f"    _patch_boundary_cycle(patch {pix}): single-edge cap, using edge_dat chain order (len={len(cycle_from_chain)} -> {len(cycle_with_crown)} with crown)")
                return cycle_with_crown

    # Prefer OUT_chain: ordered boundary outline for this patch (vertex indices in m.X / patch mesh)
    chain = PM.get('OUT_chain') and PM['OUT_chain'].get(pix)
    if chain is not None and len(chain) > 0:
        # Filter chain to keys+sentinels only, preserving order; map to simplified indices
        cycle_ordered = []
        seen_simpl = set()
        for v in chain:
            v = int(v) if hasattr(v, '__int__') else v
            if v not in boundary_mX:
                continue
            si = mX_to_simpl.get(v)
            if si is None:
                continue
            # Avoid duplicate consecutive entries (chain can repeat at closure)
            if cycle_ordered and cycle_ordered[-1] == si:
                continue
            cycle_ordered.append(si)
            seen_simpl.add(si)
        # Use OUT_chain result only if it includes all key/sentinel vertices (correct cyclic order).
        # Fictitious vertices (crown sentinels, indices >= nkeys) are inserted later by
        # _insert_crown_sentinels_in_cycle, so they should NOT cause a fallback.
        key_sentinel_simpl = {si for si in boundary_simpl_set if si < nkeys}
        missing = key_sentinel_simpl - seen_simpl
        if len(missing) == 0 and len(cycle_ordered) >= 2:
            # Insert crown sentinels into the cycle
            cycle_with_crown = _insert_crown_sentinels_in_cycle(cycle_ordered, pix, PM, Xkeyind, nkeys)
            if verbose_fail:
                print(f"    _patch_boundary_cycle(patch {pix}): using OUT_chain (len={len(cycle_ordered)} keys+sentinels -> {len(cycle_with_crown)} with crown)")
            return cycle_with_crown
        if verbose_fail:
            print(f"    _patch_boundary_cycle(patch {pix}): OUT_chain missed {len(missing)} key/sentinel vertices (missing simpl: {sorted(missing)[:10]}); falling back to edge walk")
        # If chain missed some key/sentinel vertices, do not guess order; fall back to edge walk
    # Fallback: build cycle from edge connectivity
    boundary_simpl = list(boundary_simpl_set)
    E_bound = []
    E_arr = np.asarray(E)
    if E_arr.ndim == 1 and E_arr.size == 2:
        E_arr = E_arr.reshape(1, 2)
    elif E_arr.ndim != 2 or (E_arr.shape[0] > 0 and E_arr.shape[1] != 2):
        E_arr = np.array([]).reshape(0, 2)
    for i, j in E_arr:
        i, j = int(i), int(j)
        if i < nkeys and j < nkeys and i in boundary_simpl_set and j in boundary_simpl_set:
            E_bound.append((i, j))
    # Ensure every incident edge's sentinel pair is in E_bound (E may have skipped s1==s2 or border edges).
    # This fixes cases where we have n boundary vertices but only n-1 edges from E (e.g. one edge not in E).
    for eix in range(len(PM['Edges'])):
        if PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix:
            continue
        if PM['Edges'][eix, 1] < 0:
            continue
        s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
        if s1 == s2:
            continue
        si1 = mX_to_simpl.get(s1)
        si2 = mX_to_simpl.get(s2)
        if si1 is not None and si2 is not None and si1 in boundary_simpl_set and si2 in boundary_simpl_set:
            if (si1, si2) not in E_bound and (si2, si1) not in E_bound:
                E_bound.append((si1, si2))
    # If OUT_chain exists (e.g. on re-run), add consecutive boundary pairs to E_bound as well.
    chain = PM.get('OUT_chain') and PM['OUT_chain'].get(pix)
    if chain is not None and len(chain) >= 2:
        for idx in range(len(chain)):
            v_a = int(chain[idx]) if hasattr(chain[idx], '__int__') else chain[idx]
            v_b = int(chain[(idx + 1) % len(chain)]) if hasattr(chain[(idx + 1) % len(chain)], '__int__') else chain[(idx + 1) % len(chain)]
            if v_a not in boundary_mX or v_b not in boundary_mX:
                continue
            si_a = mX_to_simpl.get(v_a)
            si_b = mX_to_simpl.get(v_b)
            if si_a is not None and si_b is not None and si_a != si_b:
                if (si_a, si_b) not in E_bound and (si_b, si_a) not in E_bound:
                    E_bound.append((si_a, si_b))
    adj = {}
    for i, j in E_bound:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    n_bound = len(boundary_simpl_set)
    # Two boundary components: return two rings.  This applies to pre-classified
    # cylinders AND to any patch that encloses a cap/neck (annular topology).
    # We detect components from the adjacency graph built above.
    if n_bound >= 4:  # need at least 2+2 for two valid loops
        comp_sets = []
        visited = set()
        for v in boundary_simpl:
            if v in visited:
                continue
            comp = set()
            stack = [v]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for w in adj.get(u, []):
                    if w not in visited:
                        stack.append(w)
            comp_sets.append(comp)
        if len(comp_sets) == 2:
            # Build one cycle per component (close path if needed)
            def _walk_cycle(comp_set):
                comp_list = list(comp_set)
                if len(comp_list) < 2:
                    return comp_list
                start = comp_list[0]
                cy = [start]
                used = {start}
                for _ in range(len(comp_list) - 1):
                    cur = cy[-1]
                    next_cands = [n for n in adj.get(cur, []) if n in comp_set and n not in used]
                    if not next_cands:
                        break
                    nxt = next_cands[0]
                    cy.append(nxt)
                    used.add(nxt)
                if len(cy) == len(comp_set) and len(cy) >= 2 and cy[-1] in adj.get(cy[0], []):
                    return cy
                if len(cy) == len(comp_set) and len(cy) >= 2:
                    return cy  # still closed by convention
                return cy
            c1 = _walk_cycle(comp_sets[0])
            c2 = _walk_cycle(comp_sets[1])
            if len(c1) >= 2 and len(c2) >= 2:
                # Insert crown sentinels into each ring
                r1 = _insert_crown_sentinels_in_cycle(c1, pix, PM, Xkeyind, nkeys)
                r2 = _insert_crown_sentinels_in_cycle(c2, pix, PM, Xkeyind, nkeys)
                return [r1, r2]  # list of two cycles = cylinder
    # If we have exactly n-1 edges for n boundary vertices, the graph is a path or has one isolated;
    # add one edge so the walk can form a full cycle (e.g. n vertices, n-1 edges).
    if n_bound >= 3 and len(E_bound) == n_bound - 1:
        degree = {v: len(adj.get(v, [])) for v in boundary_simpl}
        deg1 = [v for v in boundary_simpl if degree.get(v, 0) == 1]
        deg0 = [v for v in boundary_simpl if degree.get(v, 0) == 0]
        if verbose_fail:
            print(f"    _patch_boundary_cycle(patch {pix}): n_bound={n_bound}, E_bound={len(E_bound)}, deg1 count={len(deg1)}, deg0 count={len(deg0)}")
        added = False
        if len(deg1) == 2:
            a, b = int(deg1[0]), int(deg1[1])
            pair_ab = (a, b)
            pair_ba = (b, a)
            if pair_ab not in E_bound and pair_ba not in E_bound:
                # Path: two endpoints not connected; add closing edge
                E_bound.append(pair_ab)
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
                added = True
                if verbose_fail:
                    print(f"    _patch_boundary_cycle(patch {pix}): added closing edge ({a},{b}) for path ({n_bound} vertices, {n_bound-1} edges)")
            else:
                # Two components: edge (a,b) + rest (e.g. 6-cycle). Add (a,u) and (b,v) and remove (u,v) so we get one 8-cycle
                other = [int(x) for x in boundary_simpl if x != a and x != b]
                for (u, v) in list(E_bound):
                    u, v = int(u), int(v)
                    if u in other and v in other:
                        if (a, u) not in E_bound and (u, a) not in E_bound and (b, v) not in E_bound and (v, b) not in E_bound:
                            E_bound.remove((u, v)) if (u, v) in E_bound else E_bound.remove((v, u))
                            adj[u].remove(v)
                            adj[v].remove(u)
                            E_bound.append((a, u))
                            E_bound.append((b, v))
                            adj.setdefault(a, []).append(u)
                            adj.setdefault(u, []).append(a)
                            adj.setdefault(b, []).append(v)
                            adj.setdefault(v, []).append(b)
                            added = True
                            if verbose_fail:
                                print(f"    _patch_boundary_cycle(patch {pix}): connected components with edges ({a},{u}) and ({b},{v}), removed ({u},{v}) ({n_bound} vertices)")
                            break
                    if added:
                        break
                if not added:
                    # Fallback: add (a, x) for any x in other to at least merge components
                    for x in other:
                        if (a, x) not in E_bound and (x, a) not in E_bound:
                            E_bound.append((a, x))
                            adj.setdefault(a, []).append(x)
                            adj.setdefault(x, []).append(a)
                            added = True
                            if verbose_fail:
                                print(f"    _patch_boundary_cycle(patch {pix}): connected components with edge ({a},{x}) (fallback)")
                            break
        elif len(deg0) == 1 and len(deg1) == 0:
            # One isolated vertex + 7-cycle: connect isolated to any vertex on the cycle
            iso = int(deg0[0])
            other = int(next((v for v in boundary_simpl if v != iso), iso))
            if (iso, other) not in E_bound and (other, iso) not in E_bound:
                E_bound.append((iso, other))
                adj.setdefault(iso, []).append(other)
                adj.setdefault(other, []).append(iso)
                added = True
                if verbose_fail:
                    print(f"    _patch_boundary_cycle(patch {pix}): connected isolated vertex {iso} to {other} ({n_bound} vertices, {n_bound-1} edges)")
        if verbose_fail and not added:
            print(f"    _patch_boundary_cycle(patch {pix}): no closing edge added (deg1={len(deg1)}, deg0={len(deg0)})")
    if len(adj) > 0:
        # Prefer starting from a degree-1 vertex so the walk traces a full path (for 8 nodes, 7 edges)
        def walk_from(start):
            cy = [start]
            used = {start}
            for _ in range(len(boundary_simpl) - 1):
                cur = cy[-1]
                next_cands = [n for n in adj.get(cur, []) if n not in used]
                if not next_cands:
                    next_cands = [n for n in adj.get(cur, []) if n in boundary_simpl_set]
                if not next_cands:
                    break
                nxt = next_cands[0]
                if nxt in used and nxt != start:
                    break
                cy.append(nxt)
                used.add(nxt)
            return cy
        # Try start from vertex with smallest degree first (path endpoints have degree 1)
        degree = {v: len(adj.get(v, [])) for v in boundary_simpl}
        starts = sorted(boundary_simpl, key=lambda v: degree.get(v, 0))
        cycle = None
        for start in starts:
            cycle = walk_from(start)
            if len(cycle) >= len(boundary_simpl) - 1:
                break
        if cycle is None:
            cycle = walk_from(boundary_simpl[0])
        # If walk closed the cycle (returned to start), we have [start, ..., last, start]; drop duplicate
        if len(cycle) == len(boundary_simpl) + 1 and cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        if len(cycle) == len(boundary_simpl):
            # Check if cycle closes: last vertex should be adjacent to start
            last_s, start_s = cycle[-1], cycle[0]
            if start_s in adj.get(last_s, []):
                cycle_with_crown = _insert_crown_sentinels_in_cycle(cycle, pix, PM, Xkeyind, nkeys)
                return cycle_with_crown
            # Cycle has all vertices but missing closing edge (e.g. one sentinel pair not in E).
            # Add closing edge and use this cycle.
            E_bound.append((last_s, start_s))
            adj.setdefault(last_s, []).append(start_s)
            adj.setdefault(start_s, []).append(last_s)
            cycle_with_crown = _insert_crown_sentinels_in_cycle(cycle, pix, PM, Xkeyind, nkeys)
            return cycle_with_crown
        # Path has all but one vertex: add missing vertex and the two edges to close the cycle
        if len(cycle) == len(boundary_simpl) - 1:
            missing_set = boundary_simpl_set - set(cycle)
            if len(missing_set) == 1:
                v_m = list(missing_set)[0]
                last_s, start_s = cycle[-1], cycle[0]
                E_bound.append((last_s, v_m))
                E_bound.append((v_m, start_s))
                adj.setdefault(last_s, []).append(v_m)
                adj.setdefault(v_m, []).append(last_s)
                adj.setdefault(v_m, []).append(start_s)
                adj.setdefault(start_s, []).append(v_m)
                full_cycle = cycle + [v_m]
                cycle_with_crown = _insert_crown_sentinels_in_cycle(full_cycle, pix, PM, Xkeyind, nkeys)
                return cycle_with_crown
    if verbose_fail:
        print(f"    _patch_boundary_cycle(patch {pix}): edge walk failed; boundary_simpl_set size = {len(boundary_simpl_set)}, E_bound = {len(E_bound)}, adj size = {len(adj)}")
    # Geometric fallback: order boundary vertices by angle around patch center
    if X_verts is not None and len(X_verts) > nkeys + pix:
        center = np.array(X_verts[nkeys + pix], dtype=float)
        # Filter boundary_simpl to only include indices that exist in X_verts
        boundary_simpl_valid = [si for si in boundary_simpl if si < len(X_verts)]
        if len(boundary_simpl_valid) < 2:
            return []
        pts = np.array([X_verts[si] for si in boundary_simpl_valid], dtype=float)
        vecs = pts - center
        # Project to plane (use PCA: dominant 2 axes)
        if vecs.shape[0] >= 2:
            try:
                from numpy.linalg import svd
                U, _, _ = svd(vecs.T @ vecs)
                if U.shape[1] >= 2:
                    proj = vecs @ U[:, :2]
                    angles = np.arctan2(proj[:, 1], proj[:, 0])
                    order = np.argsort(angles)
                    cycle_geom = [boundary_simpl_valid[i] for i in order]
                    # Insert crown sentinels into the geometric cycle (only if they exist in mesh)
                    cycle_with_crown = _insert_crown_sentinels_in_cycle(cycle_geom, pix, PM, Xkeyind, nkeys)
                    if verbose_fail:
                        print(f"    _patch_boundary_cycle(patch {pix}): using geometric (angle) fallback, cycle length = {len(cycle_with_crown)}")
                    return cycle_with_crown
            except Exception:
                pass
    return []


def _cylinder_non_cap_ring_from_edge_chains(pix, PM, Xkeyind, nkeys, edge_dat, cap_neighbor):
    """
    Build the ordered non-cap ring for a cylinder patch from edge chains,
    so the ring order matches the shared boundary order used by neighbors (avoids criss-crossing).

    Parameters:
    -----------
    pix : int
        Cylinder patch index
    PM : dict
        Patch mesh structure (Edges, edge_dat)
    Xkeyind : array
        Simplified mesh vertex -> mesh index (first nkeys are key/sentinel)
    nkeys : int
        Number of key/sentinel vertices
    edge_dat : list of arrays
        Vertex chain per edge (mesh indices)
    cap_neighbor : int
        Patch index of the cap (to exclude from non-cap ring chains)

    Returns:
    --------
    cycle : list of int or None
        Ordered simplified indices along the non-cap ring, or None if not buildable.
    """
    mX_to_simpl = {}
    for i in range(nkeys):
        mX_to_simpl[Xkeyind[i]] = i

    # Non-cap edges: incident to pix, other != cap_neighbor
    non_cap_edges = []
    for eix in range(len(PM['Edges'])):
        p1, p2 = int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1])
        if p2 < 0:
            continue
        if (p1 != pix and p2 != pix):
            continue
        other = p2 if p1 == pix else p1
        if other == cap_neighbor:
            continue
        non_cap_edges.append((eix, other))

    if not non_cap_edges:
        return None

    # Build simplified-index chains for each non-cap edge (preserve chain order)
    # Track all key/sentinel vertices on the non-cap ring so we can validate full coverage
    non_cap_vertex_set = set()
    chains_simpl = []
    for eix, _ in non_cap_edges:
        chain = edge_dat[eix] if eix < len(edge_dat) else []
        if not hasattr(chain, '__len__') or len(chain) < 2:
            continue
        simpl = []
        for v in chain:
            v = int(v)
            if v in mX_to_simpl:
                si = mX_to_simpl[v]
                simpl.append(si)
                non_cap_vertex_set.add(si)
        if len(simpl) >= 2:
            chains_simpl.append(simpl)

    if not chains_simpl:
        return None

    # Adjacency from consecutive pairs in chains (undirected)
    adj = {}
    for seg in chains_simpl:
        for i in range(len(seg) - 1):
            a, b = seg[i], seg[i + 1]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

    # Walk cycle starting from first vertex of first chain; close when we hit start again
    start = chains_simpl[0][0]
    cycle = [start]
    used = {start}
    max_steps = sum(len(s) for s in chains_simpl) + 2
    for _ in range(max_steps):
        cur = cycle[-1]
        nbrs = [n for n in adj.get(cur, []) if n not in used]
        if not nbrs:
            # Try to close: if start is neighbor, we're done
            if len(cycle) >= 2 and start in adj.get(cur, []):
                break
            # Back to previous (allow revisiting to close)
            nbrs = [n for n in adj.get(cur, []) if n != (cycle[-2] if len(cycle) > 1 else -1)]
        if not nbrs:
            break
        nxt = nbrs[0]
        if nxt == start:
            break
        cycle.append(nxt)
        used.add(nxt)

    # Validate: cycle must be closed and contain all non-cap key/sentinel vertices (no gap)
    if len(cycle) < 2:
        return None
    if start not in adj.get(cycle[-1], []):
        return None  # not closed
    if non_cap_vertex_set and set(cycle) != non_cap_vertex_set:
        return None  # missing vertices -> would cause gap; fall back to graph order

    return cycle


def _get_boundary_cycle(pix, nkeys, Xkeyind, E, PM, X, m, use_global_boundary_graph):
    """
    Get boundary cycle for patch pix. Tries global boundary graph first if enabled,
    or for complex patches (>= 6 neighbors) which often have self-intersecting cycles
    with the legacy path. Returns same format as _patch_boundary_cycle.

    For patches with two fine-mesh boundary components (annular / one_side_one_other_multi),
    always use _patch_boundary_cycle which correctly detects two components and returns
    [ring_outer, ring_inner].
    """
    preport = PM.get('patch_structure_report', {})
    n_neighbors = preport.get('n_neighbors_per_patch', np.zeros(PM.get('npatches', 0), dtype=int))

    # Check if patch type analysis found two boundary components for this patch.
    # If so, prefer _patch_boundary_cycle which now detects two components and
    # returns [ring1, ring2] — the graph may merge them into one bad cycle.
    ptr = preport.get('patch_type_report') or {}
    boundary_comps = ptr.get('boundary_components', [])
    patch_has_two_loops = (pix < len(boundary_comps) and len(boundary_comps[pix]) == 2)
    if patch_has_two_loops:
        return _patch_boundary_cycle(pix, nkeys, Xkeyind, E, PM, X_verts=X, verbose_fail=False)

    use_graph = (use_global_boundary_graph or
                 (pix < len(n_neighbors) and n_neighbors[pix] >= 6))
    if use_graph and PM.get('_global_boundary_graph') is not None:
        try:
            from .global_boundary_graph import get_patch_boundary_cycles_from_graph_v2
            result = get_patch_boundary_cycles_from_graph_v2(
                PM['_global_boundary_graph'], PM, pix, Xkeyind, nkeys, X, m
            )
        except NameError:
            result = []
        except Exception:
            result = []
        if result:
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                # Cylinder: [c1, c2] - need each ring to have >= 2 vertices
                if all(len(c) >= 2 for c in result):
                    return [_insert_crown_sentinels_in_cycle(c, pix, PM, Xkeyind, nkeys) for c in result]
            elif isinstance(result, list) and len(result) >= 3 and not isinstance(result[0], list):
                # Single cycle - need >= 3 for a valid fan
                return _insert_crown_sentinels_in_cycle(result, pix, PM, Xkeyind, nkeys)
    return _patch_boundary_cycle(pix, nkeys, Xkeyind, E, PM, X_verts=X, verbose_fail=False)


def generate_simplified_mesh(m, PM, use_global_boundary_graph=False):
    """
    Generate simplified patch-level mesh.
    
    Parameters:
    -----------
    m : surface_mesh
        Full resolution mesh
    PM : dict
        Patch mesh structure
    use_global_boundary_graph : bool, default False
        If True, use global boundary graph for boundary cycles (POC).
        Falls back to _patch_boundary_cycle on failure or weak result.
        
    Returns:
    --------
    PM : dict
        Updated PM with pm, Xkeyind, Keyind, CVind
    """
    # Clear fictitious_per_edge from any previous pass so that _patch_boundary_cycle
    # doesn't include stale crown indices in boundary_simpl_set (they'll be recreated
    # by add_fictitious_keys_for_single_neighbor_edges after this function returns).
    if 'fictitious_per_edge' in PM:
        del PM['fictitious_per_edge']
    
    # Build global boundary graph for consistent boundary cycles. Required when
    # use_global_boundary_graph is True, or when any patch has >= 6 neighbors
    # (complex patches often get self-intersecting cycles from OUT_chain/edge walk).
    preport = PM.get('patch_structure_report', {})
    n_neighbors = preport.get('n_neighbors_per_patch', np.zeros(PM['npatches'], dtype=int))
    needs_graph = use_global_boundary_graph or np.any(n_neighbors >= 6)
    if needs_graph:
        try:
            from .global_boundary_graph import build_global_boundary_graph
            PM['_global_boundary_graph'] = build_global_boundary_graph(m, PM)
        except Exception:
            PM['_global_boundary_graph'] = None
    
    # Get unique key vertices
    if len(PM['keys']) > 0:
        indx = np.unique(PM['keys'][:, 1].astype(int))
    else:
        indx = np.array([], dtype=int)
    
    # Also include sentinel vertices if they're not already keys
    # Sentinels are used for edges, so they must be in the simplified mesh
    sentinel_vertices = []
    if len(PM['sentinels']) > 0:
        for eix in range(len(PM['sentinels'])):
            s1 = int(PM['sentinels'][eix, 0])
            s2 = int(PM['sentinels'][eix, 1])
            if s1 > 0 and s1 not in indx:
                sentinel_vertices.append(s1)
            if s2 > 0 and s2 not in indx:
                sentinel_vertices.append(s2)
        sentinel_vertices = np.unique(sentinel_vertices)
    else:
        sentinel_vertices = np.array([], dtype=int)
    
    # Combine: key vertices + sentinel vertices (if not keys) + center/crown vertices
    all_key_verts = np.unique(np.concatenate([indx, sentinel_vertices]) if len(sentinel_vertices) > 0 else indx)
    nkeys = len(all_key_verts)
    cylinder_patches = set(PM.get('patch_structure_report', {}).get('cylinder_patches', []))
    npatches = PM['npatches']
    verbose_debug = False  # Set True for detailed debugging output

    # Pass 1: Build temporary X with one center per patch (so we can get boundary_cycle for cylinders)
    X_center_temp = []
    Xkeyind_center_temp = []
    for pix in range(npatches):
        cv_m = int(PM['CV'][pix])
        X_center_temp.append(m.X[cv_m])
        Xkeyind_center_temp.append(cv_m)
    X_center_temp = np.array(X_center_temp)
    Xkeyind_center_temp = np.array(Xkeyind_center_temp, dtype=int)
    if len(all_key_verts) > 0:
        X_temp = np.vstack([m.X[all_key_verts], X_center_temp])
        Xkeyind_temp = np.concatenate([all_key_verts, Xkeyind_center_temp])
    else:
        X_temp = X_center_temp
        Xkeyind_temp = Xkeyind_center_temp
    # Temporary center_offset: one per patch
    center_offset_temp = list(range(npatches))

    # Generate edges (use temp Xkeyind; center_offset_temp for cv_idx)
    center_offset = center_offset_temp
    X = X_temp
    Xkeyind = Xkeyind_temp
    X_center_list = list(X_center_temp)
    Xkeyind_center_list = list(Xkeyind_center_temp)
    
    # Generate edges
    # Base edges: for each patch-patch edge, connect consecutive key/sentinel vertices along the chain
    # so that edges with 3--4 keys (e.g. cap-cylinder) get a proper chain of edges in the simplified mesh.
    all_key_set = set(all_key_verts.tolist())
    E = []
    for eix in range(len(PM['Edges'])):
        if PM['Edges'][eix, 1] < 0:
            continue
        s1 = int(PM['sentinels'][eix, 0])
        s2 = int(PM['sentinels'][eix, 1])
        if s1 == s2 or s1 < 0 or s2 < 0:
            continue
        chain = PM['edge_dat'][eix] if eix < len(PM['edge_dat']) else np.array([])
        if hasattr(chain, '__len__') and len(chain) >= 2:
            # Keep vertices that are in the simplified mesh, in chain order
            ordered = []
            for v in chain:
                v = int(v)
                if v in all_key_set:
                    ordered.append(v)
            if len(ordered) >= 2:
                for k in range(len(ordered) - 1):
                    a, b = ordered[k], ordered[k + 1]
                    ia = np.where(Xkeyind == a)[0]
                    ib = np.where(Xkeyind == b)[0]
                    if len(ia) > 0 and len(ib) > 0:
                        E.append([int(ia[0]), int(ib[0])])
                continue
        # Fallback: single edge from s1 to s2
        i1 = np.where(Xkeyind == s1)[0]
        i2 = np.where(Xkeyind == s2)[0]
        if len(i1) > 0 and len(i2) > 0:
            E.append([int(i1[0]), int(i2[0])])
        elif len(i1) > 0 or len(i2) > 0:
            import warnings
            warnings.warn(f'patch_info_gen: Step 11: edges based on sentinels not consistent for edge {eix}. '
                         f'Sentinels: [{s1}, {s2}], Found indices: [{i1}, {i2}]')
            if len(i1) > 0 and len(i2) == 0:
                E.append([int(i1[0]), int(i1[0])])
            elif len(i1) == 0 and len(i2) > 0:
                E.append([int(i2[0]), int(i2[0])])
    
    # Add edges from center vertices to key vertices (skip cylinder: they use crown and faces only)
    # Also add border edges between key vertices on border (if exactly 2)
    # Note: nkeys was already computed above; center_offset gives start index per patch
    for pix in range(PM['npatches']):
        if pix in cylinder_patches:
            continue  # Cylinder: no single center; edges come from crown-based face generation
        cv_idx = nkeys + center_offset[pix]  # Index of center vertex in X
        
        # Get keys for this patch
        if len(PM['keys']) > 0:
            patch_keys = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int)
            
            # [1] Add patch center connections to key vertices
            for key in patch_keys:
                key_idx = np.where(Xkeyind == key)[0]
                if len(key_idx) > 0:
                    E.append([cv_idx, key_idx[0]])
            
            # [2] Add border edges (if present)
            # Find key vertices for this patch that are on the border
            if hasattr(m, 'border_vertex') and m.border_vertex is not None:
                # Check which patch keys are on the border
                isborder = []
                for key in patch_keys:
                    if key < len(m.border_vertex) and m.border_vertex[key] == 1:
                        isborder.append(key)
                
                # If exactly 2 border key vertices, add edge between them
                if len(isborder) == 2:
                    idx1 = np.where(Xkeyind == isborder[0])[0]
                    idx2 = np.where(Xkeyind == isborder[1])[0]
                    if len(idx1) > 0 and len(idx2) > 0:
                        E.append([idx1[0], idx2[0]])
    
    E = np.array(E, dtype=np.int64) if E else np.array([]).reshape(0, 2)
    if E.ndim == 1:
        E = E.reshape(-1, 2)  # single edge [a,b] -> shape (1, 2)

    # Cap patches with many neighbors: two-layer center structure
    # Layer 1: crown ring of centers between boundary and interior (n vertices)
    # Layer 2: single inner center that fans to the crown ring (1 vertex)
    # This avoids the single-center fan that crosses itself on curved surfaces.
    # Total centers per cap crown patch: n_crown + 1
    cap_crown_data = {}  # pix -> {boundary_ring, n, crown_positions, inner_center_position}
    preport = PM.get('patch_structure_report', {})
    cap_patches_list = preport.get('cap_patches', [])
    n_neighbors = preport.get('n_neighbors_per_patch', np.zeros(PM['npatches'], dtype=int))
    
    for pix in cap_patches_list:
        # Cap patches with many neighbors (>= 5) need crown + inner center
        if n_neighbors[pix] >= 5:
            boundary_cycle = _get_boundary_cycle(pix, nkeys, Xkeyind, E, PM, X, m, use_global_boundary_graph)
            if isinstance(boundary_cycle, list) and len(boundary_cycle) > 0:
                if isinstance(boundary_cycle[0], list):
                    continue  # Cylinder-like: skip
                boundary_ring = list(boundary_cycle)
                if len(boundary_ring) >= 3:
                    n = len(boundary_ring)
                    X_pts = X
                    boundary_pts = X_pts[boundary_ring]
                    boundary_centroid = np.mean(boundary_pts, axis=0)
                    # Crown ring: blend boundary vertices toward centroid (30% inward)
                    crown_positions = []
                    offset_factor = 0.3
                    for i in range(n):
                        center_pt = (1 - offset_factor) * boundary_pts[i] + offset_factor * boundary_centroid
                        dists = np.linalg.norm(m.X - center_pt, axis=1)
                        nearest_vi = np.argmin(dists)
                        center_pt = m.X[nearest_vi].copy()
                        crown_positions.append(center_pt)
                    # Inner center: at the boundary centroid, projected onto mesh surface
                    dists = np.linalg.norm(m.X - boundary_centroid, axis=1)
                    nearest_vi = np.argmin(dists)
                    inner_center = m.X[nearest_vi].copy()
                    cap_crown_data[pix] = {
                        'boundary_ring': boundary_ring,
                        'n': n,
                        'crown_positions': np.array(crown_positions),
                        'inner_center_position': inner_center,
                    }
                    if verbose_debug:
                        print(f"  Patch {pix}: Cap with {n_neighbors[pix]} neighbors -> crown of {n} + 1 inner center")
    
    # Store cap crown counts in report for diagnostic access
    PM['patch_structure_report']['cap_crown_n'] = {pix: data['n'] for pix, data in cap_crown_data.items()}
    
    # Cylinder patches: get boundary cycles, then one center per quad (centroid of quad).
    # Use the two rings from _patch_boundary_cycle as-is; resample to same n; align r2 with r1 by best cyclic shift.
    # Special handling (cap vs non-cap ring, edge-chain ordering) only when we add it back carefully.
    cylinder_ring_data = {}  # pix -> {r1, r2, n, center_positions} (r1,r2 = simplified indices; n = len(r1))
    for pix in cylinder_patches:
        boundary_cycle = _get_boundary_cycle(pix, nkeys, Xkeyind, E, PM, X, m, use_global_boundary_graph)
        if not (isinstance(boundary_cycle, list) and len(boundary_cycle) == 2 and
                isinstance(boundary_cycle[0], list) and isinstance(boundary_cycle[1], list)):
            continue
        r1, r2 = boundary_cycle[0], boundary_cycle[1]
        if len(r1) < 2 or len(r2) < 2:
            continue
        n1, n2 = len(r1), len(r2)
        # MUST use the larger ring size so no vertex is dropped (dropping creates gaps).
        # The shorter ring is upsampled. Use resampling that avoids consecutive duplicates
        # (which would create degenerate quads). See _resample_ring_for_cylinder.
        n = max(n1, n2, 3)
        r1 = _resample_ring_for_cylinder(r1, n)
        r2 = _resample_ring_for_cylinder(r2, n)
        X_pts = X
        # Align r2 to r1: try both r2 and reversed(r2) since rings may have opposite winding
        best_sum = np.inf
        r2_shifted = list(r2)  # default
        for r2_try in [r2, list(reversed(r2))]:
            for shift in range(n):
                s = sum(np.linalg.norm(X_pts[r1[i]] - X_pts[r2_try[(i + shift) % n]]) for i in range(n))
                if s < best_sum:
                    best_sum = s
                    r2_shifted = [r2_try[(i + shift) % n] for i in range(n)]
        center_positions = []
        used_vi = set()
        for i in range(n):
            a, b = r1[i], r1[(i + 1) % n]
            c, d = r2_shifted[(i + 1) % n], r2_shifted[i]
            cen = (X_pts[a] + X_pts[b] + X_pts[c] + X_pts[d]) / 4.0
            # Project quad center onto the mesh surface (nearest original mesh vertex)
            # so cylinder faces don't penetrate through neighboring patches.
            # Avoid reusing the same vertex to prevent degenerate/overlapping centers.
            dists = np.linalg.norm(m.X - cen, axis=1)
            order = np.argsort(dists)
            nearest_vi = None
            for vi in order:
                if vi not in used_vi:
                    nearest_vi = vi
                    break
            if nearest_vi is None:
                nearest_vi = int(order[0])
            used_vi.add(nearest_vi)
            center_positions.append(m.X[nearest_vi].copy())
        cylinder_ring_data[pix] = {
            'r1': r1, 'r2': r2_shifted, 'n': n,
            'center_positions': np.array(center_positions),
        }

    # Detect annular patches (2 boundary components but not in cylinder_patches).
    # These arise when a patch encloses a cap/neck.
    # Use ZIPPER triangulation (no center vertex, no resampling) to produce a
    # manifold band of n1+n2 triangles between rings of size n1 and n2.
    # This avoids the non-manifold edges caused by resampling a shorter ring.
    zipper_data = {}  # pix -> {'r1': [...], 'r2': [...]}
    cap_patches = set(preport.get('cap_patches', []))
    ptr = preport.get('patch_type_report') or {}
    boundary_comps = ptr.get('boundary_components', [])
    for pix in range(npatches):
        if pix in cylinder_ring_data or pix in cap_crown_data or pix in cylinder_patches:
            continue
        if pix >= len(boundary_comps) or len(boundary_comps[pix]) != 2:
            continue
        # This patch has 2 boundary components → annular topology
        boundary_cycle = _get_boundary_cycle(pix, nkeys, Xkeyind, E, PM, X, m, use_global_boundary_graph)
        if not (isinstance(boundary_cycle, list) and len(boundary_cycle) == 2 and
                isinstance(boundary_cycle[0], list) and isinstance(boundary_cycle[1], list)):
            continue
        r1, r2 = boundary_cycle[0], boundary_cycle[1]
        if len(r1) < 2 or len(r2) < 2:
            continue

        # Cap-ring alignment: if one ring is shared with a cap, use the cap's edge_dat chain
        # order so the annulus and cap boundary match (avoids non-manifold and boundary edges).
        # Case 1: Single-edge cap (1 neighbor on one ring) - well-handled.
        # Case 2: Two neighbors on one ring, several on the other - need alignment from the
        # 2-neighbor ring's edge chains so rings match geometric order (avoids wrong vertex pairing).
        cap_ring_reordered = None
        cap_tag = None
        # Try Case 1 first: single-edge cap neighbor
        for eix in range(len(PM['Edges'])):
            if PM['Edges'][eix, 1] < 0:
                continue
            p1, p2 = int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1])
            other = p2 if p1 == pix else (p1 if p2 == pix else -1)
            if other < 0 or other not in cap_patches:
                continue
            # Check if neighbor is a single-edge cap (so it uses edge_dat for its boundary)
            incident = sum(1 for j in range(len(PM['Edges'])) if PM['Edges'][j, 1] >= 0 and
                          (int(PM['Edges'][j, 0]) == other or int(PM['Edges'][j, 1]) == other))
            if incident != 1:
                continue
            chain = PM['edge_dat'][eix] if eix < len(PM['edge_dat']) else []
            if not hasattr(chain, '__len__') or len(chain) < 2:
                continue
            chain = np.asarray(chain).flatten()
            # Map chain (m.X indices) to simplified indices
            cycle_from_chain = []
            for v in chain:
                v = int(v)
                idx = np.where(Xkeyind == v)[0]
                if len(idx) > 0 and idx[0] < nkeys:
                    cycle_from_chain.append(int(idx[0]))
            if len(cycle_from_chain) < 2:
                continue
            chain_set = set(cycle_from_chain)
            for ring, tag in [(r1, 'r1'), (r2, 'r2')]:
                if set(ring) == chain_set and len(ring) == len(cycle_from_chain):
                    cap_ring_reordered = cycle_from_chain
                    cap_tag = tag
                    break
            if cap_ring_reordered is not None:
                break

        # Case 2: No single-edge cap found. For (a=2, b>=2) — two neighbors on one ring, several
        # on the other — ensure consistent outer/inner convention: larger ring = outer (r1),
        # smaller = inner (r2). This reduces wrong vertex pairing when cap_ring_reordered is N/A.
        if cap_ring_reordered is None and len(r1) >= 2 and len(r2) >= 2:
            if len(r1) < len(r2):
                r1, r2 = r2, r1  # swap so r1 is outer (larger), r2 is inner (smaller)

        if cap_ring_reordered is not None and cap_tag is not None:
            if cap_tag == 'r1':
                r1 = cap_ring_reordered
            else:
                r2 = cap_ring_reordered

        # Align r2 to r1 (best cyclic shift + optional reversal) — NO resampling.
        # Metric: sum of distances between corresponding pairs, wrapping each ring.
        n1, n2 = len(r1), len(r2)
        X_pts = X
        best_sum = np.inf
        r2_aligned = list(r2)
        for r2_try in [list(r2), list(reversed(r2))]:
            n2_try = len(r2_try)
            for shift in range(n2_try):
                s = sum(np.linalg.norm(X_pts[r1[k % n1]] - X_pts[r2_try[(k + shift) % n2_try]])
                        for k in range(max(n1, n2_try)))
                if s < best_sum:
                    best_sum = s
                    r2_aligned = [r2_try[(k + shift) % n2_try] for k in range(n2_try)]

        zipper_data[pix] = {'r1': list(r1), 'r2': r2_aligned}
        if verbose_debug:
            print(f"  Patch {pix}: annular -> zipper triangulation (outer={n1}, inner={n2})")

    # Store cylinder ring counts and zipper patches in report for diagnostic access
    PM['patch_structure_report']['cylinder_ring_n'] = {pix: data['n'] for pix, data in cylinder_ring_data.items()}
    PM['patch_structure_report']['zipper_patches'] = list(zipper_data.keys())
    
    # Rebuild X and Xkeyind: cylinder → n quad centers; cap crown → n+1 centers;
    # zipper → 0 centers (band triangulated directly from rings); else → 1 CV per patch.
    X_center_list = []
    Xkeyind_center_list = []
    center_offset = [0] * npatches
    off = 0
    for pix in range(npatches):
        center_offset[pix] = off
        if pix in cylinder_ring_data:
            data = cylinder_ring_data[pix]
            for k in range(data['n']):
                X_center_list.append(data['center_positions'][k])
                Xkeyind_center_list.append(-1)
            off += data['n']
        elif pix in cap_crown_data:
            data = cap_crown_data[pix]
            # First n entries: crown ring centers
            for k in range(data['n']):
                X_center_list.append(data['crown_positions'][k])
                Xkeyind_center_list.append(-1)
            # Last entry: inner center
            X_center_list.append(data['inner_center_position'])
            Xkeyind_center_list.append(-1)
            off += data['n'] + 1  # n crown + 1 inner center
        elif pix in zipper_data:
            # Zipper patches: NO center vertex (band triangulated from rings)
            pass  # off stays the same
        else:
            cv_m = int(PM['CV'][pix])
            X_center_list.append(m.X[cv_m])
            Xkeyind_center_list.append(cv_m)
            off += 1
    X_center = np.array(X_center_list)
    Xkeyind_center = np.array(Xkeyind_center_list, dtype=int)
    if len(all_key_verts) > 0:
        X = np.vstack([m.X[all_key_verts], X_center])
        Xkeyind = np.concatenate([all_key_verts, Xkeyind_center])
    else:
        X = X_center
        Xkeyind = Xkeyind_center
    
    # Generate faces for simplified mesh
    # Match MATLAB algorithm exactly: use edge connectivity to walk around center vertex
    F = []
    face_labels = []
    
    # Debug: Print diagnostic information
    if verbose_debug:
        print(f"\n{'='*60}")
        print(f"generate_simplified_mesh DEBUG:")
        print(f"  Unique keys (indx): {len(indx)}")
        print(f"  Sentinel vertices added: {len(sentinel_vertices)}")
        print(f"  Total key vertices in mesh (nkeys): {nkeys}")
        print(f"  npatches: {PM['npatches']}")
        print(f"  Keys shape: {PM['keys'].shape if len(PM['keys']) > 0 else 'empty'}")
        if len(PM['keys']) > 0:
            print(f"  Keys (first 5): {PM['keys'][:min(5, len(PM['keys']))]}")
        if len(PM['sentinels']) > 0:
            print(f"  Sentinels shape: {PM['sentinels'].shape}")
            print(f"  Sentinels (first 5): {PM['sentinels'][:min(5, len(PM['sentinels']))]}")
        print(f"  Edges created: {len(E)}")
        if len(E) > 0:
            print(f"  Edges (first 5): {E[:min(5, len(E))]}")
        print(f"  X shape: {X.shape}, Xkeyind shape: {Xkeyind.shape}")
        print(f"  Center vertices (CV): {PM['CV']}")
        print(f"{'='*60}\n")
    
    # Patch structure summary (for debugging problematic patches)
    # Run diagnostics for patches with potential issues (multi-neighbor patches, caps, etc.)
    if verbose_debug and PM['npatches'] > 0:
        # Diagnose first few patches or patches with many neighbors (common sources of issues)
        patches_to_diagnose = []
        preport = PM.get('patch_structure_report', {})
        n_neighbors = preport.get('n_neighbors_per_patch', np.zeros(PM['npatches'], dtype=int))
        # Include patches with many neighbors (likely to have issues) or first few patches
        for pix in range(min(3, PM['npatches'])):
            patches_to_diagnose.append(pix)
        for pix in range(PM['npatches']):
            if n_neighbors[pix] >= 5 and pix not in patches_to_diagnose:
                patches_to_diagnose.append(pix)
                if len(patches_to_diagnose) >= 5:
                    break
        
        preport = PM.get('patch_structure_report', {})
        ptr = preport.get('patch_type_report') or {}
        ptype = ptr.get('patch_type', [None] * PM['npatches'])
        if hasattr(ptype, '__iter__') and not isinstance(ptype, (str, dict)):
            ptype = list(ptype) if ptype is not None else []
        n_neighbors = preport.get('n_neighbors_per_patch', np.zeros(PM['npatches'], dtype=int))
        comps = ptr.get('boundary_components', [])
        for pix_diag in patches_to_diagnose:
            print(f"\n  --- Patch {pix_diag} structure ---")
            print(f"  Patch {pix_diag} type (from patch_type_analysis): {ptype[pix_diag] if pix_diag < len(ptype) else 'unknown'}")
            print(f"  Patch {pix_diag} neighbor count: {n_neighbors[pix_diag] if pix_diag < len(n_neighbors) else '?'}")
            if pix_diag < len(comps):
                for cix, (vset, nset) in enumerate(comps[pix_diag]):
                    print(f"  Boundary component {cix}: {len(vset)} vertices, neighbors {sorted(nset)}")
            # Incident edges for this patch
            incident = []
            for eix in range(len(PM['Edges'])):
                p1, p2 = int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1])
                if p2 < 0:
                    continue
                if p1 == pix_diag or p2 == pix_diag:
                    other = p2 if p1 == pix_diag else p1
                    chain = PM.get('edge_dat', [])
                    chain_len = len(chain[eix]) if eix < len(chain) and hasattr(chain[eix], '__len__') else 0
                    incident.append((eix, other, chain_len))
            print(f"  Incident edges (eix, neighbor, chain_len): {incident}")
            # Keys for this patch
            if len(PM['keys']) > 0:
                keys_p = PM['keys'][PM['keys'][:, 0] == pix_diag, 1].astype(int).tolist()
                print(f"  Patch {pix_diag} keys (m.X indices): {keys_p[:15]}{'...' if len(keys_p) > 15 else ''} (total {len(keys_p)})")
            out_chain_p = PM.get('OUT_chain') and PM['OUT_chain'].get(pix_diag)
            print(f"  OUT_chain for patch {pix_diag}: present={out_chain_p is not None}, len={len(out_chain_p) if out_chain_p is not None else 0}")
            if out_chain_p is not None and len(out_chain_p) > 0:
                boundary_mX_p = set()
                if len(PM['keys']) > 0:
                    boundary_mX_p.update(PM['keys'][PM['keys'][:, 0] == pix_diag, 1].astype(int).tolist())
                for eix in range(len(PM['Edges'])):
                    if PM['Edges'][eix, 0] != pix_diag and PM['Edges'][eix, 1] != pix_diag or PM['Edges'][eix, 1] < 0:
                        continue
                    s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
                    boundary_mX_p.add(s1)
                    boundary_mX_p.add(s2)
                in_chain = sum(1 for v in out_chain_p if int(v) in boundary_mX_p)
                print(f"  OUT_chain: {in_chain} of {len(boundary_mX_p)} patch-{pix_diag} boundary vertices appear in chain")
            print(f"  --- end Patch {pix_diag} structure ---\n")
    
    # Generate faces for each patch using boundary cycle (keys + sentinels in order).
    # This correctly handles single-neighbor patches (0 keys, only sentinels), neck
    # patches (sentinels between keys along the boundary), and cylinder patches (two rings).
    # STRATEGY: Never skip a patch - ensure every patch gets at least one face.
    for pix in range(PM['npatches']):
        cv_idx = nkeys + center_offset[pix]  # Single center for non-cylinder; first crown index for cylinder
        boundary_cycle = _get_boundary_cycle(pix, nkeys, Xkeyind, E, PM, X, m, use_global_boundary_graph)

        # Build set of all edges that are part of crown segments (already have faces).
        # Needed by annular and single-cycle paths to avoid duplicate non-manifold faces.
        crown_edges = set()
        if 'fictitious_per_edge' in PM:
            for eix in range(len(PM['Edges'])):
                if PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix:
                    continue
                if PM['Edges'][eix, 1] < 0:
                    continue
                if eix in PM['fictitious_per_edge']:
                    s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
                    s1_idx = np.where(Xkeyind == s1)[0]
                    s2_idx = np.where(Xkeyind == s2)[0]
                    if len(s1_idx) > 0 and len(s2_idx) > 0:
                        s1_simpl = int(s1_idx[0]) if s1_idx[0] < nkeys else None
                        s2_simpl = int(s2_idx[0]) if s2_idx[0] < nkeys else None
                        if s1_simpl is not None and s2_simpl is not None:
                            max_valid_idx = len(Xkeyind) - 1
                            fict_inds = [fi for fi in PM['fictitious_per_edge'][eix] if fi <= max_valid_idx]
                            if len(fict_inds) > 0:
                                crown_edges.add((s1_simpl, fict_inds[0]))
                                for k in range(len(fict_inds) - 1):
                                    crown_edges.add((fict_inds[k], fict_inds[k + 1]))
                                crown_edges.add((fict_inds[-1], s2_simpl))
                                crown_edges.add((s2_simpl, fict_inds[-1]))
                                for k in range(len(fict_inds) - 1, 0, -1):
                                    crown_edges.add((fict_inds[k], fict_inds[k - 1]))
                                crown_edges.add((fict_inds[0], s1_simpl))

        # Cap patches with many neighbors: two-layer triangulation
        # Layer 1 (outer band): boundary ring to crown ring (2 triangles per sector)
        # Layer 2 (inner fan): inner center to crown ring (1 triangle per sector)
        # This creates a fully closed patch with no gaps.
        if pix in cap_crown_data:
            data = cap_crown_data[pix]
            boundary_ring, n = data['boundary_ring'], data['n']
            crown_start = nkeys + center_offset[pix]
            inner_center_idx = crown_start + n  # Inner center is the last vertex
            f_cap_crown = []
            n_degenerate = 0
            for i in range(n):
                c_idx = crown_start + i
                c_next = crown_start + ((i + 1) % n)
                b1 = boundary_ring[i]
                b2 = boundary_ring[(i + 1) % n]
                # Outer band triangle 1: crown_i -> boundary_i -> boundary_{i+1}
                tri1 = [c_idx, b1, b2]
                if tri1[0] != tri1[1] and tri1[1] != tri1[2] and tri1[0] != tri1[2]:
                    f_cap_crown.append(tri1)
                else:
                    n_degenerate += 1
                # Outer band triangle 2: crown_i -> boundary_{i+1} -> crown_{i+1}
                tri2 = [c_idx, b2, c_next]
                if tri2[0] != tri2[1] and tri2[1] != tri2[2] and tri2[0] != tri2[2]:
                    f_cap_crown.append(tri2)
                else:
                    n_degenerate += 1
                # Inner fan triangle: inner_center -> crown_i -> crown_{i+1}
                tri3 = [inner_center_idx, c_idx, c_next]
                if tri3[0] != tri3[1] and tri3[1] != tri3[2] and tri3[0] != tri3[2]:
                    f_cap_crown.append(tri3)
                else:
                    n_degenerate += 1
            F.extend(f_cap_crown)
            face_labels.extend([pix] * len(f_cap_crown))
            if verbose_debug:
                msg = f"  Patch {pix}: Generated {len(f_cap_crown)} faces from cap crown (n={n} crown + 1 inner, {n} boundary)"
                if n_degenerate > 0:
                    msg += f" [WARNING: skipped {n_degenerate} degenerate faces]"
                print(msg)
            continue
        
        # Cylinder (neck): one center per quad (centroid of A,B,E,D). Quad i = (r1[i], r1[i+1], r2[i+1], r2[i]).
        # Four triangles per quad: (center_i, A, B), (center_i, B, E), (center_i, E, D), (center_i, D, A).
        if pix in cylinder_ring_data:
            data = cylinder_ring_data[pix]
            r1, r2, n = data['r1'], data['r2'], data['n']
            crown_start = nkeys + center_offset[pix]
            f_cyl = []
            n_degenerate = 0
            for i in range(n):
                c_idx = crown_start + i
                A, B = r1[i], r1[(i + 1) % n]
                E, D = r2[(i + 1) % n], r2[i]
                # Skip degenerate faces (duplicate vertices in a triangle)
                for tri in [[c_idx, A, B], [c_idx, B, E], [c_idx, E, D], [c_idx, D, A]]:
                    if tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2]:
                        f_cyl.append(tri)
                    else:
                        n_degenerate += 1
            F.extend(f_cyl)
            face_labels.extend([pix] * len(f_cyl))
            if verbose_debug:
                msg = f"  Patch {pix}: Generated {len(f_cyl)} faces from cylinder (n={n} quads)"
                if n_degenerate > 0:
                    msg += f" [WARNING: skipped {n_degenerate} degenerate faces]"
                print(msg)
            continue

        # Zipper-triangulated annular patch (two rings of different sizes).
        # _zipper_triangulate_rings produces n1+n2 triangles with no center vertex.
        if pix in zipper_data:
            data = zipper_data[pix]
            r1_z, r2_z = data['r1'], data['r2']
            f_zip = _zipper_triangulate_rings(r1_z, r2_z, X)
            # Filter degenerate triangles (shouldn't happen but be safe)
            f_zip = [t for t in f_zip if t[0] != t[1] and t[1] != t[2] and t[0] != t[2]]
            F.extend(f_zip)
            face_labels.extend([pix] * len(f_zip))
            if verbose_debug:
                print(f"  Patch {pix}: Generated {len(f_zip)} faces from zipper "
                      f"(outer={len(r1_z)}, inner={len(r2_z)})")
            continue

        # Annular patch (two boundary loops, e.g. patch that encloses a cap).
        # _patch_boundary_cycle returns [ring_outer, ring_inner] for these.
        # Generate a separate fan from the center to each ring.
        if (isinstance(boundary_cycle, list) and len(boundary_cycle) == 2 and
                isinstance(boundary_cycle[0], list) and isinstance(boundary_cycle[1], list)):
            ring_a, ring_b = boundary_cycle
            if len(ring_a) >= 2 and len(ring_b) >= 2:
                f_annular = []
                n_degenerate = 0
                for ring in [ring_a, ring_b]:
                    for i in range(len(ring)):
                        a = ring[i]
                        b = ring[(i + 1) % len(ring)]
                        if a == b or a == cv_idx or b == cv_idx:
                            n_degenerate += 1
                            continue
                        # Skip crown edges (already have faces from fictitious key insertion)
                        if (a, b) not in crown_edges:
                            f_annular.append([cv_idx, a, b])
                        else:
                            n_degenerate += 1
                if len(f_annular) > 0:
                    F.extend(f_annular)
                    face_labels.extend([pix] * len(f_annular))
                    if verbose_debug:
                        msg = f"  Patch {pix}: Generated {len(f_annular)} faces from annular patch (rings: {len(ring_a)}+{len(ring_b)})"
                        if n_degenerate > 0:
                            msg += f" [skipped {n_degenerate} degenerate/crown]"
                        print(msg)
                continue

        # Single cycle: boundary_cycle is a list of simplified indices
        if not isinstance(boundary_cycle, list):
            boundary_cycle = list(boundary_cycle) if isinstance(boundary_cycle, (tuple, np.ndarray)) else []
        
        # If boundary cycle failed, try to create a minimal cycle from available vertices
        if len(boundary_cycle) < 2:
            if verbose_debug:
                print(f"  Patch {pix}: boundary cycle has < 2 vertices, attempting recovery...")
            # Try to find any boundary vertices for this patch
            recovery_vertices = []
            # Check for keys
            if len(PM['keys']) > 0:
                patch_keys = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int)
                for key in patch_keys:
                    idx = np.where(Xkeyind == key)[0]
                    if len(idx) > 0 and idx[0] < nkeys:
                        recovery_vertices.append(int(idx[0]))
            # Check for sentinels on edges incident to this patch
            for eix in range(len(PM['Edges'])):
                if (PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix) or PM['Edges'][eix, 1] < 0:
                    continue
                s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
                for s in [s1, s2]:
                    if s >= 0:
                        idx = np.where(Xkeyind == s)[0]
                        if len(idx) > 0 and idx[0] < nkeys:
                            si = int(idx[0])
                            if si not in recovery_vertices:
                                recovery_vertices.append(si)
            
            if len(recovery_vertices) >= 3:
                # Order by angle around patch center for a valid fan
                cv_idx = nkeys + center_offset[pix] if pix < len(center_offset) else nkeys
                cpt = X[cv_idx] if cv_idx < len(X) else np.mean(X[recovery_vertices], axis=0)
                pts = X[recovery_vertices]
                centroid = np.mean(pts, axis=0)
                u1 = np.array([1, 0, 0]) if np.linalg.norm(np.cross(pts[0] - centroid, [0, 1, 0])) < 1e-10 else np.cross(pts[0] - centroid, [0, 1, 0])
                u1 = u1 / (np.linalg.norm(u1) + 1e-12)
                u2 = np.cross(pts[0] - centroid, u1)
                u2 = u2 / (np.linalg.norm(u2) + 1e-12)
                angles = [np.arctan2((p - centroid) @ u2, (p - centroid) @ u1) for p in pts]
                order = np.argsort(angles)
                boundary_cycle = [recovery_vertices[i] for i in order]
                if verbose_debug:
                    print(f"    Recovery: Ordered {len(boundary_cycle)} vertices by angle for patch {pix}")
            elif len(recovery_vertices) >= 2:
                boundary_cycle = recovery_vertices[:2]
                if verbose_debug:
                    print(f"    Recovery: Using {len(boundary_cycle)} vertices (minimal) for patch {pix}")
            elif len(recovery_vertices) == 1:
                # Only one vertex - duplicate it to form degenerate edge (better than nothing)
                boundary_cycle = [recovery_vertices[0], recovery_vertices[0]]
                if verbose_debug:
                    print(f"    Recovery: Using single vertex (duplicated) for patch {pix}")
            else:
                # No boundary vertices at all - this patch cannot have faces
                if verbose_debug:
                    print(f"  Patch {pix}: No boundary vertices found, cannot generate faces")
                continue
        
        # Keep the original center vertex (on the mesh surface from get_center_vert).
        # Only replace with boundary centroid if the fan is invalid (center outside polygon).
        # The on-surface center produces better visual quality on curved surfaces.
        if len(boundary_cycle) >= 3:
            bpts = X[boundary_cycle]
            centroid_b = np.mean(bpts, axis=0)
            cpt = X[cv_idx]
            # PCA: project boundary + center to best-fit 2D plane
            C_cov = np.cov((bpts - centroid_b).T)
            try:
                w, v = np.linalg.eigh(C_cov)
                idx_sort = np.argsort(w)[::-1]
                u1, u2 = v[:, idx_sort[0]], v[:, idx_sort[1]]
            except Exception:
                u1, u2 = np.array([1, 0, 0]), np.array([0, 1, 0])
            bpts_2d = np.column_stack([(bpts - centroid_b) @ u1, (bpts - centroid_b) @ u2])
            cpt_2d = np.array([(cpt - centroid_b) @ u1, (cpt - centroid_b) @ u2])
            # Check fan validity: all signed areas must have the same sign
            n_bc = len(boundary_cycle)
            signed_areas = []
            for i in range(n_bc):
                a2, b2 = bpts_2d[i], bpts_2d[(i + 1) % n_bc]
                sa = (a2[0] - cpt_2d[0]) * (b2[1] - cpt_2d[1]) - (b2[0] - cpt_2d[0]) * (a2[1] - cpt_2d[1])
                signed_areas.append(sa)
            n_pos = sum(1 for s in signed_areas if s > 1e-12)
            n_neg = sum(1 for s in signed_areas if s < -1e-12)
            fan_valid = (n_pos == 0 or n_neg == 0)
            if not fan_valid:
                # Center is outside polygon in 2D projection → replace with centroid
                X[cv_idx] = centroid_b
                if verbose_debug:
                    print(f"  Patch {pix}: center produces invalid fan ({n_pos} pos, {n_neg} neg). Moved to boundary centroid.")
                # Recompute signed areas for orientation
                cpt_2d = np.array([0.0, 0.0])  # centroid projects to origin
                signed_areas = []
                for i in range(n_bc):
                    a2, b2 = bpts_2d[i], bpts_2d[(i + 1) % n_bc]
                    sa = a2[0] * b2[1] - b2[0] * a2[1]
                    signed_areas.append(sa)
                n_pos = sum(1 for s in signed_areas if s > 0)
                n_neg = sum(1 for s in signed_areas if s < 0)
            # Orient: ensure consistent winding (reverse if majority negative)
            if n_neg > n_pos:
                boundary_cycle = list(reversed(boundary_cycle))
                if verbose_debug:
                    print(f"  Patch {pix}: reversed boundary cycle for consistent winding")
        
        # Patch diagnostic: boundary cycle order and which patch-patch edge each segment belongs to
        if len(boundary_cycle) >= 2 and verbose_debug:
            edge_dat = PM.get('edge_dat', [])
            cycle_mX = [int(Xkeyind[si]) if si < len(Xkeyind) else -1 for si in boundary_cycle]
            print(f"\n  --- Patch {pix} diagnostic (boundary cycle analysis) ---")
            print(f"  Boundary cycle length: {len(boundary_cycle)}")
            print(f"  Cycle (simpl indices): {boundary_cycle[:20]}{'...' if len(boundary_cycle) > 20 else ''}")
            print(f"  Cycle (m.X indices):   {cycle_mX[:20]}{'...' if len(cycle_mX) > 20 else ''}")
            # For each consecutive pair (a,b), find which eix has both in chain (in that order or reverse)
            seg_edges = []
            for i in range(len(boundary_cycle)):
                a, b = boundary_cycle[i], boundary_cycle[(i + 1) % len(boundary_cycle)]
                va, vb = int(Xkeyind[a]) if a < len(Xkeyind) else -1, int(Xkeyind[b]) if b < len(Xkeyind) else -1
                found = None
                for eix in range(len(PM['Edges'])):
                    if (PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix):
                        continue
                    if PM['Edges'][eix, 1] < 0:
                        continue
                    ch = edge_dat[eix] if eix < len(edge_dat) else []
                    if not hasattr(ch, '__len__') or len(ch) < 2:
                        continue
                    ch = np.asarray(ch).flatten().tolist()
                    try:
                        ia = ch.index(va)
                        ib = ch.index(vb)
                        if abs(ia - ib) == 1 or (ia == 0 and ib == len(ch)-1) or (ib == 0 and ia == len(ch)-1):
                            found = (eix, int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1]))
                            break
                    except (ValueError, TypeError):
                        pass
                seg_edges.append((i, va, vb, found))
            print(f"  Consecutive segments (cycle index, mX_a, mX_b, edge (eix, p1, p2)):")
            for i, va, vb, fe in seg_edges[:min(15, len(seg_edges))]:
                print(f"    [{i}] {va} -> {vb}  edge={fe}")
            if len(seg_edges) > 15:
                print(f"    ... and {len(seg_edges)-15} more")
            # Investigation: compare edge order in edge_dat vs boundary_cycle (fixes may have changed perceived order)
            # Find edges incident to this patch for comparison
            for eix in range(len(PM['Edges'])):
                p1, p2 = int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1])
                if p2 >= 0 and (p1 == pix or p2 == pix):
                    ch = edge_dat[eix] if eix < len(edge_dat) else []
                    if hasattr(ch, '__len__') and len(ch) >= 2:
                        ch_flat = np.asarray(ch).flatten().tolist()
                        ch_set = set(int(x) for x in ch_flat)
                        # How many of this patch's keys are in this edge chain?
                        keys_in_chain = [int(v) for v in ch_flat if int(v) in set(cycle_mX)]
                        neighbor_pix = p2 if p1 == pix else p1
                        print(f"  Patch {pix} edge chain analysis (eix={eix}, neighbor={neighbor_pix}):")
                        print(f"    edge_dat[{eix}] total length: {len(ch_flat)}  (OUT_chain length: {len(PM.get('OUT_chain', {}).get(pix, []))})")
                        print(f"    Patch {pix} keys in this chain: {keys_in_chain} ({len(keys_in_chain)} of {len(cycle_mX)})")
                        if len(keys_in_chain) > 2:
                            print(f"    WARNING: edge_dat for ({pix},{neighbor_pix}) contains {len(keys_in_chain)} keys — chain may be bloated (find_edge_chain walked the wrong way)")
                    break
            print(f"  --- end Patch {pix} diagnostic ---\n")
        
        # Generate faces from boundary cycle (crown_edges already built at top of loop)
        f = []
        if len(boundary_cycle) == 2:
            # Single edge: only generate if it doesn't have crown (crown faces already created)
            a, b = boundary_cycle[0], boundary_cycle[1]
            if (a, b) not in crown_edges and a != b and a != cv_idx and b != cv_idx:
                f.append([cv_idx, a, b])
        else:
            # For each edge in the cycle, skip if it's part of a crown segment
            for i in range(len(boundary_cycle)):
                a = boundary_cycle[i]
                b = boundary_cycle[(i + 1) % len(boundary_cycle)]
                # Skip degenerate faces and crown edges
                if a == b or a == cv_idx or b == cv_idx:
                    continue
                if (a, b) not in crown_edges:
                    f.append([cv_idx, a, b])
        
        if len(f) > 0:
            F.extend(f)
            face_labels.extend([pix] * len(f))
            if verbose_debug:
                skipped = len(boundary_cycle) - len(f) if len(boundary_cycle) > 2 else (1 - len(f))
                if skipped > 0:
                    print(f"  Patch {pix}: Generated {len(f)} faces from boundary cycle (len={len(boundary_cycle)}, skipped {skipped} crown edges)")
                else:
                    print(f"  Patch {pix}: Generated {len(f)} faces from boundary cycle (len={len(boundary_cycle)})")
    
    F = np.array(F) if F else np.array([]).reshape(0, 3)
    face_labels = np.array(face_labels) if face_labels else np.array([], dtype=int)
    
    if verbose_debug:
        print(f"\n{'='*60}")
        print(f"Face generation summary:")
        print(f"  Total faces generated: {len(F)}")
        print(f"  Total face labels: {len(face_labels)}")
        if len(F) == 0:
            print(f"  WARNING: No faces generated!")
            print(f"    - Keys available: {len(PM['keys']) > 0}")
            print(f"    - Edges available: {len(E) > 0}")
            print(f"    - If both are True, face generation logic may have failed")
        print(f"{'='*60}\n")
    
    # Create simplified mesh
    pm = surface_mesh(X, F)
    pm.face_labels = face_labels
    pm.t = np.zeros(len(X))
    pm.p = np.zeros(len(X))
    
    # Transfer border info if present (crown vertices have Xkeyind=-1, leave border_vertex 0)
    pm.border_vertex = np.zeros(len(X), dtype=int)
    for vix in range(len(X)):
        mx = int(Xkeyind[vix])
        if 0 <= mx < len(m.border_vertex):
            pm.border_vertex[vix] = m.border_vertex[mx]
    
    # Transfer t, p values if present (crown: leave 0)
    if hasattr(m, 't') and m.t is not None:
        for vix in range(len(X)):
            mx = int(Xkeyind[vix])
            if 0 <= mx < len(m.t):
                pm.t[vix] = m.t[mx]
                pm.p[vix] = m.p[mx]
    
    PM['pm'] = pm
    PM['Xkeyind'] = Xkeyind
    # Keyind should only include actual keys (not sentinel vertices that aren't keys)
    # The first len(indx) entries in X are the unique keys
    n_actual_keys = len(indx)  # Original unique keys (before adding sentinels)
    PM['Keyind'] = np.arange(n_actual_keys)
    PM['CVind'] = np.arange(nkeys, len(X))
    PM['nkeys'] = nkeys  # explicit count of key+sentinel vertices for diagnostics

    # Independent diagnostic: detect scrambled/self-intersecting simplified patch meshes
    PM['patch_self_intersection'] = diagnose_simplified_patch_self_intersection(PM)
    scrambled = [pix for pix, r in PM['patch_self_intersection'].items() if r.get('is_self_intersecting')]
    if scrambled or verbose_debug:
        _print_patch_self_intersection_report(PM['patch_self_intersection'])

    # Write comprehensive diagnostic to file for debugging
    _write_simplified_mesh_diagnostic(PM, Xkeyind, nkeys, m)
    
    return PM


def _write_simplified_mesh_diagnostic(PM, Xkeyind, nkeys, m):
    """Write a comprehensive diagnostic of the simplified mesh to a text file."""
    import os, datetime
    from collections import defaultdict
    try:
        diag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests')
        os.makedirs(diag_dir, exist_ok=True)
        diag_path = os.path.join(diag_dir, 'simplified_mesh_diagnostic.txt')
    except Exception:
        return
    try:
        pm = PM['pm']
        fl = np.asarray(pm.face_labels).flatten()
        run_id = PM.get('run_id', 'unknown')
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = []
        lines.append(f"Run ID: {run_id}  |  Timestamp: {ts}")
        lines.append("=" * 70)
        lines.append("SIMPLIFIED MESH COMPREHENSIVE DIAGNOSTIC")
        lines.append("=" * 70)
        lines.append(f"Vertices: {len(pm.X)}, Faces: {len(pm.F)}, nkeys: {nkeys}")
        lines.append(f"Xkeyind (first 40): {Xkeyind[:min(40, len(Xkeyind))].tolist()}")
        lines.append("")

        # Vertex positions (key/sentinel + centers)
        lines.append("Vertex positions (simpl_idx: mX_idx -> [x, y, z]):")
        for vi in range(min(len(pm.X), 50)):
            mx = int(Xkeyind[vi]) if vi < len(Xkeyind) and int(Xkeyind[vi]) >= 0 else f'fict'
            pos = pm.X[vi]
            label = 'key' if vi < nkeys else 'center/fict'
            lines.append(f"  v{vi} (mX={mx}, {label}): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        lines.append("")

        # Per-patch face listing
        for pix in range(PM['npatches']):
            p_faces = np.where(fl == pix)[0]
            lines.append(f"--- Patch {pix}: {len(p_faces)} faces ---")
            # Keys and sentinels
            if len(PM['keys']) > 0:
                pk = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int).tolist()
                lines.append(f"  Keys (mX): {pk}")
            # Center position
            cylinder_patches = set(PM.get('patch_structure_report', {}).get('cylinder_patches', []))
            cylinder_ring_n = PM.get('patch_structure_report', {}).get('cylinder_ring_n', {})
            cap_crown_n = PM.get('patch_structure_report', {}).get('cap_crown_n', {})
            zipper_pix_set = set(PM.get('patch_structure_report', {}).get('zipper_patches', []))
            if pix not in cylinder_patches and pix not in zipper_pix_set:
                cv_off = 0
                for pp in range(pix):
                    if pp in cylinder_patches:
                        cv_off += cylinder_ring_n.get(pp, 1)
                    elif pp in cap_crown_n:
                        cv_off += cap_crown_n[pp] + 1  # n crown + 1 inner
                    elif pp in zipper_pix_set:
                        pass  # zipper patches: 0 center vertices
                    else:
                        cv_off += 1
                cv_vi = nkeys + cv_off
                if cv_vi < len(pm.X):
                    pos = pm.X[cv_vi]
                    lines.append(f"  Center: simpl={cv_vi}, pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            # Incident edges
            for eix in range(len(PM['Edges'])):
                p1, p2 = int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1])
                if p2 < 0:
                    continue
                if p1 == pix or p2 == pix:
                    other = p2 if p1 == pix else p1
                    s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
                    lines.append(f"  Edge eix={eix}: neighbor={other}, sentinels=({s1},{s2})")
            # Faces with normal analysis
            normals = []
            for fi in p_faces:
                f = pm.F[fi]
                mX = []
                for v in f:
                    v = int(v)
                    if v < len(Xkeyind) and int(Xkeyind[v]) >= 0:
                        mX.append(int(Xkeyind[v]))
                    else:
                        mX.append(f'fict_{v}')
                # Compute face normal
                a, b, c = int(f[0]), int(f[1]), int(f[2])
                e1 = pm.X[b] - pm.X[a]
                e2 = pm.X[c] - pm.X[a]
                n = np.cross(e1, e2)
                nn = np.linalg.norm(n)
                if nn > 1e-12:
                    n = n / nn
                normals.append(n)
                lines.append(f"  Face {fi}: simpl={f.tolist()}, mX={mX}, normal=[{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}]")
            # Check normal consistency (all normals should point roughly the same way)
            if len(normals) >= 2:
                ref = normals[0]
                flipped = [i for i, n in enumerate(normals) if np.dot(n, ref) < 0]
                if flipped:
                    lines.append(f"  WARNING: {len(flipped)} face normals flipped vs face 0: indices {flipped}")
                else:
                    lines.append(f"  All {len(normals)} face normals consistent")
            lines.append("")

        # Edge analysis
        edge_faces = defaultdict(list)
        for fi in range(len(pm.F)):
            f = pm.F[fi]
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            for u, v in [(a, b), (b, c), (c, a)]:
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

        # Cross-patch shared edges
        lines.append("")
        lines.append("Shared edges between patch pairs:")
        pair_edges = defaultdict(list)
        for e, flist in edge_faces.items():
            labels = sorted(set(int(fl[fi]) for fi in flist))
            if len(labels) == 2:
                pair_edges[tuple(labels)].append(e)
        for pair, edges in sorted(pair_edges.items()):
            lines.append(f"  Patches {pair}: {len(edges)} shared edges: {edges[:10]}{'...' if len(edges) > 10 else ''}")

        # Fictitious per edge
        fpe = PM.get('fictitious_per_edge', {})
        if fpe:
            lines.append(f"\nFictitious per edge: {len(fpe)} edges")
            for eix, inds in sorted(fpe.items()):
                p1, p2 = int(PM['Edges'][eix, 0]), int(PM['Edges'][eix, 1])
                lines.append(f"  eix={eix} ({p1},{p2}): fictitious indices = {inds}")

        lines.append("\n" + "=" * 70)
        with open(diag_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"  Diagnostic written to: {diag_path}")
    except Exception as ex:
        print(f"  Diagnostic write failed: {ex}")
