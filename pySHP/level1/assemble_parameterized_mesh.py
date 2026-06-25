"""
Assemble full parameterized mesh from all patches
Translated from MATLAB workflow
"""

import numpy as np
from ..surface_mesh import surface_mesh


def assemble_parameterized_mesh(m_original, PM):
    """
    Assemble full parameterized mesh from all patches.
    
    This function collects theta and phi values from all parameterized patches
    and assigns them to the original mesh vertices.
    
    Parameters:
    -----------
    m_original : surface_mesh
        Original mesh (before segmentation)
    PM : dict
        Patch mesh structure with parameterized patches
        
    Returns:
    --------
    m_param : surface_mesh
        Full mesh with t and p values from all patches
    """
    # Create a copy of the original mesh
    m_param = surface_mesh(m_original.X.copy(), m_original.F.copy())
    
    # Copy face labels if present
    if hasattr(m_original, 'face_labels') and m_original.face_labels is not None:
        m_param.face_labels = m_original.face_labels.copy()
    
    # Initialize t and p arrays
    m_param.t = np.zeros(len(m_param.X))
    m_param.p = np.zeros(len(m_param.X))
    
    # Track which vertices have been assigned
    vertices_assigned = np.zeros(len(m_param.X), dtype=bool)
    
    # Copy t and p values from each parameterized patch to the full mesh
    for pix in range(PM['npatches']):
        pat = PM['P'][pix][0]  # Get the parameterized patch
        
        # Get all vertex indices used in this patch
        # Patches use the same vertex indices as the original mesh
        Vix = np.unique(pat.F.flatten())
        
        # Copy t and p values from patch to full mesh
        for v in Vix:
            if v < len(m_param.t) and v < len(pat.t):
                # Only assign if the patch has a valid value (non-zero)
                # Zero values indicate unparameterized vertices
                if pat.t[v] != 0 or pat.p[v] != 0:
                    # If vertex already assigned, prefer non-zero values
                    if not vertices_assigned[v] or (m_param.t[v] == 0 and m_param.p[v] == 0):
                        m_param.t[v] = pat.t[v]
                        m_param.p[v] = pat.p[v]
                        vertices_assigned[v] = True
    
    # Report statistics
    num_assigned = np.sum(vertices_assigned)
    num_with_t = np.sum(m_param.t != 0)
    num_with_p = np.sum(m_param.p != 0)
    
    print(f"Assembled full parameterized mesh:")
    print(f"  Total vertices: {len(m_param.X)}")
    print(f"  Vertices with t values: {num_with_t} ({100*num_with_t/len(m_param.X):.1f}%)")
    print(f"  Vertices with p values: {num_with_p} ({100*num_with_p/len(m_param.X):.1f}%)")
    
    if num_with_t > 0:
        valid_t = m_param.t[m_param.t != 0]
        print(f"  Theta range: [{valid_t.min():.4f}, {valid_t.max():.4f}]")
    
    if num_with_p > 0:
        valid_p = m_param.p[m_param.p != 0]
        print(f"  Phi range: [{valid_p.min():.4f}, {valid_p.max():.4f}]")
    
    # Check for unassigned vertices
    num_unassigned = len(m_param.X) - num_assigned
    if num_unassigned > 0:
        print(f"  WARNING: {num_unassigned} vertices not assigned from any patch")
        print(f"           These vertices will have t=0, p=0")
    
    return m_param
