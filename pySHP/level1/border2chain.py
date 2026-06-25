"""
Convert border vertices to ordered chain
Translated from MATLAB level1/border2chain.m
"""

import numpy as np


def border2chain(m, bix=0):
    """
    Detect border vertices and generate a chain sequence by walking along 
    border vertices from neighbor to neighbor.
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh with border_vertex populated
    bix : int
        Starting border vertex index (into the border vertex array)
        
    Returns:
    --------
    chain : array
        Ordered array of vertex indices forming the border chain
    """
    # Ensure edge info is computed
    m.edge_info()
    
    # Get border vertices
    bv = np.where(m.border_vertex)[0]
    
    if len(bv) == 0:
        return np.array([])
    
    chain = np.zeros(len(bv), dtype=int)
    cix = 0
    
    # Start with the specified border vertex
    if bix >= len(bv):
        bix = 0
    chain[cix] = bv[bix]
    
    prevprevvert = -1
    prevvert = -1
    oix = 0
    currvert = chain[cix]
    full_circle = False
    
    while not full_circle:
        oix += 1
        
        # Get face membership of current vertex
        fmcv = m.face_memb.get(currvert, []) if isinstance(m.face_memb, dict) else []
        
        # Get neighbor vertices
        L = m.L.get(currvert, []) if isinstance(m.L, dict) else []
        
        # Handle special case for pioneer vertex (only 2 links)
        if len(L) == 2:
            ispioneer = True
        else:
            ispioneer = False
        
        # Handle special case for pioneer at step 2
        if ispioneer and oix == 1:
            if len(L) >= 2:
                prevprevvert = L[1]
                prevvert = L[1]
        
        # Find valid neighbors (border vertices not in chain)
        l = np.zeros(len(L), dtype=int)
        for lix, neighbor in enumerate(L):
            if neighbor not in chain[:oix]:  # Don't go back
                if neighbor in bv:  # Must be a border vertex
                    l[lix] = 1
        
        # If multiple valid neighbors, prioritize pioneer vertices
        if np.sum(l) > 1:
            lp = np.zeros(len(l), dtype=int)
            for lix in range(len(l)):
                if l[lix]:
                    potvl = L[lix]
                    potL = m.L.get(potvl, []) if isinstance(m.L, dict) else []
                    lp[lix] = len(potL)
            
            # Prefer vertices with only 2 links (pioneers)
            if np.any(lp == 2):
                l[:] = 0
                l[lp == 2] = 1
        
        if not full_circle:
            # Get candidate vertices
            vl = [L[i] for i in range(len(L)) if l[i]]
            vix = []
            
            for v in vl:
                if v not in chain[:oix]:
                    vix.append(v)
            
            if len(vix) > 0:
                chain[oix] = vix[0]
                prevprevvert = prevvert
                prevvert = currvert
                currvert = vix[0]
            else:
                full_circle = True
        
        # Check if we've looped back
        if currvert == chain[0] and oix > 2:
            full_circle = True
        
        # Safety check: don't exceed array bounds
        if oix >= len(bv) - 1:
            full_circle = True
    
    # Remove trailing zeros
    chain = chain[chain != 0]
    
    # If first vertex was removed, add it back
    if len(chain) == 0 or (bv[bix] not in chain):
        chain = np.concatenate([[bv[bix]], chain])
    
    # Sanity check: try other starting points if needed
    if np.any(chain == 0) or len(chain) < len(bv):
        # Try different starting points
        ntries = min(len(bv) // 2, 5)
        for trial in range(ntries):
            rbvix = np.random.randint(0, len(bv))
            chain_try = _border2chain_simple(m, bv, rbvix)
            if not np.any(chain_try == 0) and len(chain_try) == len(bv):
                return chain_try
    
    return chain


def _border2chain_simple(m, bv, start_idx):
    """
    Simplified border to chain conversion using edge walking.
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh
    bv : array
        Border vertex indices
    start_idx : int
        Starting index into bv array
        
    Returns:
    --------
    chain : array
        Ordered chain of vertex indices
    """
    if len(bv) == 0:
        return np.array([])
    
    chain = []
    visited = set()
    current = bv[start_idx]
    chain.append(current)
    visited.add(current)
    
    bv_set = set(bv)
    
    while True:
        # Get neighbors of current vertex that are also border vertices
        neighbors = m.L.get(current, []) if isinstance(m.L, dict) else []
        next_vert = None
        
        for nbr in neighbors:
            if nbr in bv_set and nbr not in visited:
                next_vert = nbr
                break
        
        if next_vert is None:
            break
        
        chain.append(next_vert)
        visited.add(next_vert)
        current = next_vert
        
        if len(chain) >= len(bv):
            break
    
    return np.array(chain)


def get_border_chain(m, be_start_ix=None):
    """
    Generate chain starting with a specific boundary edge.
    Alternative implementation based on boundary edges.
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh
    be_start_ix : int, optional
        Starting boundary edge index
        
    Returns:
    --------
    chain : array
        Ordered chain of vertex indices
    boundary_edges : array
        Logical array indicating boundary edges
    """
    from .get_border import get_border
    
    m, mpl, boundary_edges = get_border(m)
    
    ind_be = np.where(boundary_edges)[0]
    
    if len(ind_be) == 0:
        return np.array([]), boundary_edges
    
    if be_start_ix is None:
        be_start_ix = np.random.randint(0, len(ind_be))
    
    be_curr = be_start_ix
    count = 0
    chain = []
    edge_chain = []
    
    edge_curr = ind_be[be_start_ix]
    chain.append(m.E[edge_curr, 0])
    edge_chain.append(be_curr)
    
    full_circle = False
    while not full_circle:
        count += 1
        
        # Find candidate edges connected via current vertex
        ind_edge = []
        for i, be_idx in enumerate(ind_be):
            if m.E[be_idx, 0] == chain[-1] or m.E[be_idx, 1] == chain[-1]:
                ind_edge.append(i)
        
        # Remove current edge
        ind_edge = [i for i in ind_edge if i != edge_chain[-1]]
        
        if len(ind_edge) != 1:
            break
        
        # Get next vertex
        edge_idx = ind_be[ind_edge[0]]
        vert = list(m.E[edge_idx])
        vert.remove(chain[-1])
        
        if len(vert) == 0 or vert[0] in chain:
            full_circle = True
        else:
            chain.append(vert[0])
            edge_chain.append(ind_edge[0])
    
    return np.array(chain), boundary_edges
