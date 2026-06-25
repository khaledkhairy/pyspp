"""
Border detection for mesh patches
Translated from MATLAB level1/get_border.m
"""

import numpy as np


def get_border(mp):
    """
    Determine border vertices of shape, based on its faces.
    mp is assumed to be an open shape that has at least one mesh-level edge.
    If not, border_vertex will be all zeros.
    
    Parameters:
    -----------
    mp : surface_mesh
        Input mesh (can be a patch)
        
    Returns:
    --------
    mp : surface_mesh
        Mesh with updated border_vertex (logical vector same length as mp.X)
        and updated face_labels: logical vector for faces either
        border (1) or not (0)
    mpL : array
        Face labels (1 = border face, 0 = interior)
    be : array
        Logical of length mp.E with value 1 if border edge, 0 otherwise
    """
    # Ensure edge info is computed
    mp.needs_edge_info = True
    mpL = np.zeros(len(mp.F), dtype=int)  # Will be 1 for edge faces
    mp.edge_info()  # Generate edge information
    
    # Determine faces that are on the edge using face neighbors
    # A face on the edge has fewer than 3 neighbors
    if hasattr(mp, 'face_nbrs') and mp.face_nbrs is not None:
        for i in range(len(mp.F)):
            if isinstance(mp.face_nbrs, dict):
                nbr_count = len(mp.face_nbrs.get(i, []))
            else:
                nbr_count = np.sum(mp.face_nbrs[i] != 0) if hasattr(mp.face_nbrs, '__getitem__') else 0
            if nbr_count < 3:
                mpL[i] = 1
    
    mp.face_labels = mpL
    
    # Determine boundary edges
    # A boundary edge is one that belongs to only one face
    be = np.zeros(len(mp.E), dtype=int)
    
    for eix in range(len(mp.E)):
        v1, v2 = mp.E[eix]
        
        # Find all faces that these vertices belong to
        f1 = mp.face_memb.get(v1, []) if isinstance(mp.face_memb, dict) else []
        f2 = mp.face_memb.get(v2, []) if isinstance(mp.face_memb, dict) else []
        
        # Find common faces (both vertices must be in the same face for it to be an edge)
        common = set(f1).intersection(set(f2))
        
        if len(common) == 1:
            be[eix] = 1
    
    # Determine border vertices from boundary edges
    nbv = np.sum(be)  # Total number of boundary edges
    beix = np.where(be)[0]  # Indices of boundary edges
    
    # Initialize border_vertex to zeros
    mp.border_vertex = np.zeros(len(mp.X), dtype=int)
    
    for bvix in range(nbv):
        vpair = mp.E[beix[bvix]]
        mp.border_vertex[vpair[0]] = 1
        mp.border_vertex[vpair[1]] = 1
    
    return mp, mpL, be


def compute_face_neighbors(mp):
    """
    Compute face neighbors for a mesh
    
    Parameters:
    -----------
    mp : surface_mesh
        Input mesh
        
    Returns:
    --------
    face_nbrs : dict
        Dictionary mapping face index to list of neighboring face indices
    """
    nfaces = len(mp.F)
    face_nbrs = {i: [] for i in range(nfaces)}
    
    # Build edge-to-face map
    edge_to_faces = {}
    for i in range(nfaces):
        face = mp.F[i]
        edges = [
            tuple(sorted([int(face[0]), int(face[1])])),
            tuple(sorted([int(face[1]), int(face[2])])),
            tuple(sorted([int(face[2]), int(face[0])]))
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(i)
    
    # Build neighbor list
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            face_nbrs[faces[0]].append(faces[1])
            face_nbrs[faces[1]].append(faces[0])
    
    return face_nbrs
