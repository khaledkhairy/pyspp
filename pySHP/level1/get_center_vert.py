"""
Get center vertex for a patch
Translated from MATLAB level1/get_center_vert.m
"""

import numpy as np
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


def get_center_vert(p, t=None, plot_flag=False):
    """
    Determine a central vertex from a patch p.
    The center vertex is the one with minimum squared difference from 
    median distance to outline vertices.
    
    Parameters:
    -----------
    p : surface_mesh
        Input patch mesh
    t : array, optional
        Indices of outline vertices
    plot_flag : bool
        Whether to plot for debugging
        
    Returns:
    --------
    vert : int
        Index of center vertex
    """
    # Ensure edge info is computed
    p.needs_edge_info = True
    p.edge_info()
    p.needs_updating = True
    
    # Build graph and compute distances
    G, weights = get_graph(p)
    
    # Get all unique vertices in patch faces
    s = np.unique(p.F.flatten())
    
    # Match Matlab implementation: compute distances from ALL vertices to outline vertices
    # (Matlab doesn't exclude outline vertices from being candidates)
    if t is not None and len(t) > 0:
        # Filter t to only include vertices that exist in the patch mesh
        t_valid = np.array([v for v in t if v < len(p.X)])
        
        # Compute distances from all vertices in s to outline vertices t
        d = compute_distances(G, s, t_valid, len(p.X))
    else:
        # No outline vertices provided, compute distances from all vertices to all vertices
        d = compute_distances(G, s, s, len(p.X))
    
    # Find vertex with minimum squared difference from median
    d_flat = d.flatten()
    d_flat = d_flat[~np.isinf(d_flat)]
    if len(d_flat) > 0:
        median_d = np.median(d_flat)
    else:
        median_d = 0
    
    dmeddiff = np.sum((d - median_d)**2, axis=1)
    
    # Find vertex with minimum difference (match Matlab: find first vertex with min value)
    min_val = np.min(dmeddiff)
    min_indices = np.where(dmeddiff == min_val)[0]
    vert = s[min_indices[0]]  # Take first vertex with minimum value, matching Matlab
    
    return vert


def get_graph(p):
    """
    Build a graph from mesh with curvature-based edge weights.
    
    Parameters:
    -----------
    p : surface_mesh
        Input mesh
        
    Returns:
    --------
    G : sparse matrix or networkx graph
        Graph representation
    weights : array
        Edge weights
    """
    nvert = len(p.X)
    
    # Compute curvature if not available
    if not hasattr(p, 'H') or p.H is None:
        p.props()
    
    # Build adjacency matrix with weights
    rows = []
    cols = []
    data = []
    
    if p.E is not None and len(p.E) > 0:
        for edge in p.E:
            v1, v2 = int(edge[0]), int(edge[1])
            
            # Edge weight based on distance
            weight = np.linalg.norm(p.X[v1] - p.X[v2])
            
            # Optionally add curvature penalty
            if hasattr(p, 'H') and p.H is not None:
                h1 = p.H[v1] if v1 < len(p.H) else 0
                h2 = p.H[v2] if v2 < len(p.H) else 0
                weight *= (1 + abs(h1) + abs(h2))
            
            rows.extend([v1, v2])
            cols.extend([v2, v1])
            data.extend([weight, weight])
    
    G = csr_matrix((data, (rows, cols)), shape=(nvert, nvert))
    weights = np.array(data)
    
    return G, weights


def compute_distances(G, sources, targets, nvert):
    """
    Compute shortest path distances from sources to targets.
    
    Parameters:
    -----------
    G : sparse matrix
        Graph adjacency matrix with weights
    sources : array
        Source vertex indices
    targets : array
        Target vertex indices
    nvert : int
        Total number of vertices
        
    Returns:
    --------
    d : array (len(sources) x len(targets))
        Distance matrix
    """
    # Use scipy's dijkstra
    dist_matrix = dijkstra(G, directed=False, indices=sources)
    
    # Extract distances to targets
    d = dist_matrix[:, targets]
    
    return d
