"""
Analyze the "type" of each patch for simplified-mesh setup.

Classifies patches into:
- cap: one boundary component, only one patch neighbor (end/cap of shape).
- cylinder / neck: two boundary components, one neighbor per side (band around neck).
- one_side_one_other_multi: one boundary component with 1 neighbor, another with 2+ neighbors
  (e.g. foot of mushroom: one border to neck, one border to multiple patches).
- multi: 3+ neighbors with no single "one side" (regular junction patch).

This classification drives how we build the simplified mesh (key/sentinel placement,
face generation: fan vs roundabout cylinder, etc.) so the result is well-behaved.
"""

import numpy as np
from scipy.sparse import csr_matrix


# Patch type constants for use in code
PATCH_TYPE_CAP = 'cap'
PATCH_TYPE_CYLINDER = 'cylinder'
PATCH_TYPE_ONE_SIDE_ONE_OTHER_MULTI = 'one_side_one_other_multi'
PATCH_TYPE_SINGLE_NEIGHBOR = 'single_neighbor'  # 1 neighbor, may overlap with cap
PATCH_TYPE_MULTI = 'multi'


def _boundary_components_and_neighbors(m, PM, pix):
    """
    For patch pix, return a list of (component_vertex_set, neighbor_patch_set).
    Each component is one boundary loop; neighbor_patch_set is the set of patch
    indices that share that boundary. Uses PM['Edges'] and PM['edge_dat'] to
    assign neighbors to components (patch meshes may use local indices).
    """
    patm = PM['P'][pix][0]
    if not hasattr(patm, 'border_vertex') or patm.border_vertex is None:
        return []
    border = np.where(patm.border_vertex)[0]
    if len(border) < 2:
        return [((set(border.tolist()) if len(border) > 0 else set()), set())] if len(border) > 0 else []
    border_set = set(border.tolist())
    # Boundary edges: edges on patch boundary (only one face of patch)
    edge_to_faces = {}
    for fi, f in enumerate(patm.F):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in [(a, b), (b, c), (c, a)]:
            if u > v:
                u, v = v, u
            edge_to_faces.setdefault((u, v), []).append(fi)
    boundary_edges = [
        uv for uv, faces in edge_to_faces.items()
        if len(faces) == 1 and uv[0] in border_set and uv[1] in border_set
    ]
    # Build graph on border vertices
    adj = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    # Connected components of boundary (vertex sets)
    visited = set()
    comp_verts_list = []
    for start in border:
        if start in visited:
            continue
        comp_verts = set()
        stack = [start]
        visited.add(start)
        comp_verts.add(start)
        while stack:
            u = stack.pop()
            for w in adj.get(u, []):
                if w not in visited:
                    visited.add(w)
                    comp_verts.add(w)
                    stack.append(w)
        comp_verts_list.append(comp_verts)
    # Assign neighbor patches to each component via PM['Edges'] and PM['edge_dat']
    Edges = PM.get('Edges', np.array([]).reshape(0, 2))
    edge_dat = PM.get('edge_dat', [])
    components = []
    for comp_verts in comp_verts_list:
        neighbor_patches = set()
        for eix in range(len(Edges)):
            if Edges[eix, 1] < 0:
                continue
            p1, p2 = int(Edges[eix, 0]), int(Edges[eix, 1])
            if p1 != pix and p2 != pix:
                continue
            other = p2 if p1 == pix else p1
            chain = edge_dat[eix] if eix < len(edge_dat) else np.array([])
            if hasattr(chain, '__len__') and len(chain) > 0:
                chain_set = set(np.asarray(chain).flatten().tolist())
                if chain_set & comp_verts:
                    neighbor_patches.add(other)
            else:
                s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
                if s1 in comp_verts or s2 in comp_verts:
                    neighbor_patches.add(other)
        components.append((comp_verts, neighbor_patches))
    return components


def analyze_patch_types(m, PM, verbose=True):
    """
    Classify each patch into a type for simplified-mesh handling.

    Parameters
    ----------
    m : surface_mesh
        Full mesh with face_labels.
    PM : dict
        Patch mesh structure (from patch_info_gen) with P, Edges, patch_structure_report.
    verbose : bool
        If True, print a summary.

    Returns
    -------
    report : dict
        - patch_type: list of str, one per patch (cap, cylinder, one_side_one_other_multi, multi).
        - n_neighbors_per_patch: array (from PM report).
        - boundary_components: list of list of (vertex_set, neighbor_set) per patch.
        - handling_scheme: dict mapping type -> short description of how we handle it.
    """
    npatches = PM['npatches']
    preport = PM.get('patch_structure_report', {})
    n_neighbors = preport.get('n_neighbors_per_patch', np.zeros(npatches, dtype=int))
    zero_key = set(preport.get('zero_key_patch_indices', np.array([], dtype=int)).flatten().tolist())
    single_neighbor = set(preport.get('single_neighbor_patch_indices', np.array([], dtype=int)).flatten().tolist())
    cylinder_patches = set(preport.get('cylinder_patches', []))

    patch_type = [None] * npatches
    boundary_components = []

    for pix in range(npatches):
        comps = _boundary_components_and_neighbors(m, PM, pix)
        boundary_components.append(comps)
        n_comp = len(comps)
        neighbor_counts_per_side = [len(nset) for (_, nset) in comps]

        if n_comp == 0:
            patch_type[pix] = PATCH_TYPE_MULTI
            continue
        if n_comp == 1:
            n_nbr = neighbor_counts_per_side[0]
            # Cap patches: disk-like with 1 or 2 neighbors (zero-key or low-key patches)
            # Single-neighbor patches: 1 neighbor but may have keys (not zero-key)
            if n_nbr <= 2:
                patch_type[pix] = PATCH_TYPE_CAP if pix in zero_key or n_nbr == 2 else PATCH_TYPE_SINGLE_NEIGHBOR
            else:
                patch_type[pix] = PATCH_TYPE_MULTI
            continue
        if n_comp == 2:
            a, b = neighbor_counts_per_side[0], neighbor_counts_per_side[1]
            if a == 1 and b == 1:
                patch_type[pix] = PATCH_TYPE_CYLINDER
            elif (a == 1 and b >= 2) or (b == 1 and a >= 2):
                patch_type[pix] = PATCH_TYPE_ONE_SIDE_ONE_OTHER_MULTI
            else:
                patch_type[pix] = PATCH_TYPE_MULTI
            continue
        # 3+ boundary components: treat as multi
        patch_type[pix] = PATCH_TYPE_MULTI

    handling_scheme = {
        PATCH_TYPE_CAP: 'One fan from center to single boundary (keys/sentinels on full loop). Cap patches may have 1 or 2 neighbors. Fictitious keys added on incident edges.',
        PATCH_TYPE_SINGLE_NEIGHBOR: 'Same as cap: one fan; fictitious keys on incident edge.',
        PATCH_TYPE_CYLINDER: 'Roundabout: two fans from center to each ring; no band between rings (avoids crossing).',
        PATCH_TYPE_ONE_SIDE_ONE_OTHER_MULTI: 'One boundary has 1 neighbor, other has 2+. Fan from center; keys on both borders.',
        PATCH_TYPE_MULTI: 'Fan from center to single boundary cycle (keys/sentinels along each edge).',
    }

    report = {
        'patch_type': patch_type,
        'n_neighbors_per_patch': n_neighbors,
        'boundary_components': boundary_components,
        'handling_scheme': handling_scheme,
        'zero_key_patches': zero_key,
        'cylinder_patches': cylinder_patches,
        'single_neighbor_patches': single_neighbor,
    }

    if verbose:
        print('Patch type analysis (simplified-mesh setup):')
        print('  Types: cap (1-2 neighbors), cylinder, one_side_one_other_multi, multi')
        for pix in range(npatches):
            comp_summary = []
            for (vs, ns) in boundary_components[pix]:
                comp_summary.append(f'{len(ns)}nbr')
            print(f'  Patch {pix}: {patch_type[pix]} (neighbors={n_neighbors[pix]}, components={comp_summary})')
        print('  Handling: see report["handling_scheme"]')

    return report


def get_patch_type_report(m, PM):
    """
    Convenience: return the same report as analyze_patch_types with verbose=False.
    """
    return analyze_patch_types(m, PM, verbose=False)
