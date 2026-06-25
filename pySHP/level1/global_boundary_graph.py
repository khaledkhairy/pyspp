"""
Global Boundary Graph for consistent patch boundary cycles.

Instead of per-patch heuristics (OUT_chain, edge walk, geometric fallback), we build
a global graph once: nodes = keys + sentinels, arcs = boundary chains between them.
Each patch's boundary is a closed walk in this graph. This guarantees:
- Same chain appears once; both patches see it in opposite order (consistent orientation).
- No merging conflicts from OUT_chain vs edge_dat.
- Cyclic order at each node derived from geometry.

See SIMPLIFIED_MESH_ARCHITECTURE.md for context.
"""

import numpy as np
from collections import defaultdict


def _is_boundary_between_patches(m, v1, v2, pix1, pix2):
    """True if mesh edge (v1,v2) lies on boundary between patches pix1 and pix2."""
    f1 = set(m.face_memb.get(v1, []))
    f2 = set(m.face_memb.get(v2, []))
    common = list(f1.intersection(f2))
    if len(common) != 2:
        return False  # mesh boundary or non-manifold
    labels = [int(m.face_labels[f]) for f in common]
    return {labels[0], labels[1]} == {pix1, pix2}


def _trace_chain_on_mesh(m, pix1, pix2, s1, s2):
    """
    Trace vertex chain from s1 to s2 along the boundary between patches pix1 and pix2.
    Uses mesh connectivity only (no OUT, no OUT_chain).

    Returns:
        chain : ndarray of mesh vertex indices, ordered s1 -> s2, or empty on failure.
    """
    if s1 == s2:
        return np.array([s1], dtype=int)
    m.edge_info()
    # Build adjacency of boundary vertices (only edges on pix1-pix2 boundary)
    adj = defaultdict(list)
    for v in range(len(m.X)):
        for nbr in m.L.get(v, []):
            if nbr > v and _is_boundary_between_patches(m, v, nbr, pix1, pix2):
                adj[v].append(nbr)
                adj[nbr].append(v)
    # BFS from s1 to s2
    from collections import deque
    parent = {s1: -1}
    q = deque([s1])
    found = False
    while q:
        u = q.popleft()
        if u == s2:
            found = True
            break
        for w in adj[u]:
            if w not in parent:
                parent[w] = u
                q.append(w)
    if not found:
        return np.array([s1, s2], dtype=int)  # fallback
    # Reconstruct path
    path = []
    u = s2
    while u >= 0:
        path.append(u)
        u = parent.get(u, -1)
    return np.array(path[::-1], dtype=int)


def build_global_boundary_graph(m, PM):
    """
    Build the global boundary graph from mesh and PM.

    Nodes = keys + sentinels (mesh indices).
    Arcs = chains between nodes. Each chain stores (node_a, node_b, vertices, patches).

    Parameters
    ----------
    m : surface_mesh
        Full mesh with face_labels.
    PM : dict
        Must have: Edges, sentinels, keys, edge_dat (used only if trace fails).

    Returns
    -------
    graph : dict
        - 'nodes': set of mesh vertex indices
        - 'chains': list of dicts with keys:
            'nodes': (m_a, m_b) - endpoints
            'vertices': ndarray - full chain m_a -> m_b (including endpoints)
            'patches': (pix1, pix2)
            'eix': edge index in PM['Edges']
        - 'node_to_chains': dict node -> list of (chain_idx, is_from_node)
          where is_from_node True means we leave this node along the chain (traverse a->b)
    """
    Edges = PM['Edges']
    sentinels = PM['sentinels']
    keys_set = set()
    if len(PM.get('keys', [])) > 0:
        keys_set = set(PM['keys'][:, 1].astype(int).tolist())

    nodes = set(keys_set)
    for eix in range(len(Edges)):
        if Edges[eix, 1] < 0:
            continue
        s1, s2 = int(sentinels[eix, 0]), int(sentinels[eix, 1])
        if s1 >= 0:
            nodes.add(s1)
        if s2 >= 0:
            nodes.add(s2)
    nodes = frozenset(nodes)

    chains = []
    node_to_chains = defaultdict(list)

    for eix in range(len(Edges)):
        if Edges[eix, 1] < 0:
            continue
        pix1, pix2 = int(Edges[eix, 0]), int(Edges[eix, 1])
        s1, s2 = int(sentinels[eix, 0]), int(sentinels[eix, 1])
        if s1 < 0 or s2 < 0:
            continue

        # Trace chain on mesh
        chain = _trace_chain_on_mesh(m, pix1, pix2, s1, s2)
        if len(chain) < 2:
            chain = np.array([s1, s2], dtype=int)

        # Extract nodes along chain (keys + sentinels that appear in chain)
        chain_list = chain.tolist()
        node_positions = [(i, int(chain_list[i])) for i in range(len(chain_list))
                         if int(chain_list[i]) in nodes]
        if len(node_positions) < 2:
            node_positions = [(0, s1), (len(chain_list) - 1, s2)]

        # Create arc segments between consecutive nodes
        for k in range(len(node_positions) - 1):
            i_a, n_a = node_positions[k]
            i_b, n_b = node_positions[k + 1]
            seg_verts = chain[i_a:i_b + 1]
            ch = {
                'nodes': (n_a, n_b),
                'vertices': seg_verts,
                'patches': (pix1, pix2),
                'eix': eix,
            }
            idx = len(chains)
            chains.append(ch)
            node_to_chains[n_a].append((idx, True))   # leave n_a toward n_b
            node_to_chains[n_b].append((idx, False))  # arrive at n_b from n_a

    return {
        'nodes': nodes,
        'chains': chains,
        'node_to_chains': dict(node_to_chains),
        'Edges': Edges,
    }


def get_patch_boundary_cycles_from_graph_v2(graph, PM, pix, Xkeyind, nkeys, X, m):
    """
    Get boundary cycle(s) for patch pix from the global graph.
    At each node, picks the next chain in consistent angular order (continue along boundary).
    """
    chains = graph['chains']
    mX = m.X

    mX_to_simpl = {}
    for si in range(min(nkeys, len(Xkeyind))):
        mx = int(Xkeyind[si])
        if mx >= 0:
            mX_to_simpl[mx] = si

    # For each node, list (chain_idx, neighbor_node) for chains that border pix
    patch_node_neighbors = defaultdict(list)
    for idx, ch in enumerate(chains):
        p1, p2 = ch['patches']
        n_a, n_b = ch['nodes']
        if pix == p1:
            patch_node_neighbors[n_a].append((idx, n_b))
            patch_node_neighbors[n_b].append((idx, n_a))
        elif pix == p2:
            patch_node_neighbors[n_b].append((idx, n_a))
            patch_node_neighbors[n_a].append((idx, n_b))

    used_chains = set()
    all_cycles = []
    cylinder_patches = set(PM.get('patch_structure_report', {}).get('cylinder_patches', []))

    def angle_at_node(from_mx, at_mx, to_mx):
        """Angle from (at-from) to (to-at) in the xy plane, for cyclic ordering."""
        if at_mx >= len(mX) or from_mx >= len(mX) or to_mx >= len(mX):
            return 0.0
        v_in = mX[at_mx] - mX[from_mx]
        v_out = mX[to_mx] - mX[at_mx]
        if np.linalg.norm(v_in) < 1e-12 or np.linalg.norm(v_out) < 1e-12:
            return 0.0
        a_in = np.arctan2(v_in[1], v_in[0])
        a_out = np.arctan2(v_out[1], v_out[0])
        return (a_out - a_in + 4 * np.pi) % (2 * np.pi)

    def trace_from(start_node):
        cyc = []
        cur = start_node
        prev_node = -1
        max_steps = len(chains) * 2
        for _ in range(max_steps):
            si = mX_to_simpl.get(cur)
            if si is not None and (not cyc or cyc[-1] != si):
                cyc.append(si)
            neighbors = [(idx, nb) for idx, nb in patch_node_neighbors[cur]
                        if idx not in used_chains and nb != prev_node]
            if not neighbors:
                break
            # Pick the neighbor that continues the boundary (next in cyclic order after prev)
            if prev_node >= 0:
                # Sort by angle: we want the one that is "next" after prev when going around
                neighbors.sort(key=lambda c: angle_at_node(prev_node, cur, c[1]))
            else:
                # First step: sort by angle from cur to neighbor (arbitrary but consistent)
                def first_angle(c):
                    nb = c[1]
                    if cur >= len(mX) or nb >= len(mX):
                        return 0.0
                    d = mX[nb] - mX[cur]
                    return np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 1e-12 else 0.0
                neighbors.sort(key=first_angle)
            idx, nb = neighbors[0]
            used_chains.add(idx)
            prev_node = cur
            cur = nb
            if cur == start_node:
                break
        return cyc

    for node in list(patch_node_neighbors.keys()):
        if node not in mX_to_simpl:
            continue
        if any(idx not in used_chains for idx, _ in patch_node_neighbors[node]):
            c = trace_from(node)
            if len(c) >= 2:
                all_cycles.append(c)
        if pix in cylinder_patches and len(all_cycles) >= 2:
            break
        if pix not in cylinder_patches and len(all_cycles) >= 1:
            break

    if len(all_cycles) == 2 and pix in cylinder_patches:
        return all_cycles
    return all_cycles[0] if len(all_cycles) == 1 else []
