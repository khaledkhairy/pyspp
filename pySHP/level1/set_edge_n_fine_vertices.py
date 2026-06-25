"""
Set per-edge fine-vertex counts for a simplified mesh using the PM (patch) structure.

Uses the same border-chain logic as border2chain / PM['patch'][pix]['edge_dat'] and
PM['edge_dat'] to count how many fine-mesh vertices lie along each simplified-mesh
edge. This supports optimization_method == 3 (alternating area and edge-length
Newton steps) on the simplified mesh.
"""

import numpy as np


def set_edge_n_fine_vertices_from_PM(m, PM):
    """
    Set m.edge_n_fine_vertices from the PM structure using border chains and edge_dat.

    For key–key edges (between two key/sentinel vertices): uses PM['edge_dat'][eix]
    when the simplified edge corresponds to inter-patch edge eix, or
    PM['patch'][pix]['edge_dat'] when the edge lies on a single-patch border chain.

    For key–center edges: uses a heuristic (mean chain length for that patch if
    available, else 1).

    Parameters
    ----------
    m : surface_mesh
        Simplified mesh (same topology as PM['pm']), e.g. ms = surface_mesh(pm.X, pm.F).
    PM : dict
        Patch structure from patch_info_gen. Must contain:
        - 'Xkeyind': (nvert,) index from simplified vertex -> fine vertex
        - 'edge_dat': list of vertex chains per PM['Edges'] edge
        - 'Edges': (n_edges_pm, 2) patch index pairs
        - 'sentinels': (n_edges_pm, 2) fine-mesh vertex indices per edge
        - 'npatches': int
        - 'patch': dict pix -> {'edge_dat': [arrays of fine vertex indices per chain]}

    Returns
    -------
    m : surface_mesh
        m with m.edge_n_fine_vertices set (length = len(m.E)).
    """
    if 'Xkeyind' not in PM or PM['Xkeyind'] is None:
        raise ValueError("PM must contain 'Xkeyind' (simplified vertex -> fine vertex)")
    if m.X.shape[0] != len(PM['Xkeyind']):
        raise ValueError("Simplified mesh vertex count must match len(PM['Xkeyind'])")

    m.edge_info()
    n_edges = len(m.E)
    nvert = m.X.shape[0]
    npatches = PM['npatches']
    Xkeyind = np.asarray(PM['Xkeyind']).ravel()
    nkeys = len(Xkeyind) - npatches  # first nkeys are keys/sentinels, rest are CV

    # Inter-patch edges: PM['Edges'], PM['sentinels'], PM['edge_dat']
    Edges = PM.get('Edges', np.array([]).reshape(0, 2))
    sentinels = PM.get('sentinels', np.array([]).reshape(0, 2))
    edge_dat = PM.get('edge_dat', [])

    # Build lookup: (fine_v1, fine_v2) -> (eix, len(chain)) for key-key inter-patch edges
    def _norm_pair(a, b):
        return (min(a, b), max(a, b))

    sentinel_to_eix = {}
    for eix in range(len(Edges)):
        if eix >= len(sentinels) or Edges[eix, 1] < 0:
            continue
        s1, s2 = int(sentinels[eix, 0]), int(sentinels[eix, 1])
        key = _norm_pair(s1, s2)
        chain = edge_dat[eix] if eix < len(edge_dat) else np.array([])
        sentinel_to_eix[key] = (eix, len(chain) if chain is not None else 0)

    # Per-patch chains for key-key edges on one patch (border chains between consecutive keys)
    # PM['patch'][pix]['edge_dat'] is list of arrays; each array is fine vertices between two consecutive keys
    patch_chains = PM.get('patch', {})

    counts = np.zeros(n_edges, dtype=np.float64)
    for eix in range(n_edges):
        v1, v2 = int(m.E[eix, 0]), int(m.E[eix, 1])
        f1, f2 = Xkeyind[v1], Xkeyind[v2]

        if v1 < nkeys and v2 < nkeys:
            # Key–key edge: inter-patch or same-patch border chain
            key = _norm_pair(f1, f2)
            if key in sentinel_to_eix:
                _, cnt = sentinel_to_eix[key]
                counts[eix] = max(cnt, 1)
            else:
                # Same-patch chain between consecutive keys: find chain in PM['patch'][pix]['edge_dat']
                found = False
                for pix in range(npatches):
                    if pix not in patch_chains or 'edge_dat' not in patch_chains[pix]:
                        continue
                    for chain in patch_chains[pix]['edge_dat']:
                        if chain is None or len(chain) == 0:
                            continue
                        if f1 in chain and f2 in chain:
                            i1, i2 = np.where(chain == f1)[0][0], np.where(chain == f2)[0][0]
                            cnt = abs(i2 - i1) + 1
                            counts[eix] = max(cnt, 1)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    counts[eix] = 1.0
        else:
            # Key–center edge: one of v1,v2 is center (>= nkeys)
            if v1 >= nkeys:
                pix = v1 - nkeys
            else:
                pix = v2 - nkeys
            if pix in patch_chains and 'edge_dat' in patch_chains[pix]:
                chains = patch_chains[pix]['edge_dat']
                lens = [len(c) for c in chains if c is not None and len(c) > 0]
                counts[eix] = np.mean(lens) if lens else 1.0
            else:
                counts[eix] = 1.0

    m.edge_n_fine_vertices = np.asarray(counts, dtype=np.float64)
    return m
