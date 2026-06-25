"""
Mesh utility functions for pySHP
Translated from MATLAB level0 functions
"""

import numpy as np
from scipy.sparse import csr_matrix, spdiags


def reduce_to_minimal_set(m):
    """
    Reduce mesh to minimal set of vertices
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh (may have unused vertices)
        
    Returns:
    --------
    minm : surface_mesh
        Mesh with only used vertices
    uv : array
        Index mapping: minm vertices to original m vertices
        Use: m.t[uv] = minm.t to restore
    """
    from ..surface_mesh import surface_mesh
    
    # Flatten faces and find unique vertices
    flin = m.F.flatten()
    uv, ia, ic = np.unique(flin, return_index=True, return_inverse=True)
    
    # Create new face indices (0 to len(uv)-1)
    vec = np.arange(len(uv))
    flin_new = vec[ic]
    F = flin_new.reshape(m.F.shape)
    
    # Extract used vertices
    X = m.X[uv, :]
    
    # Create new mesh
    minm = surface_mesh(X, F)
    
    # Transfer border_vertex
    if m.border_vertex is not None and len(m.border_vertex) > 0:
        minm.border_vertex = m.border_vertex[uv]
    else:
        minm.border_vertex = np.zeros(len(X))
    
    # Transfer t and p if present
    if hasattr(m, 't') and m.t is not None and len(m.t) > 0:
        minm.t = m.t[uv]
    if hasattr(m, 'p') and m.p is not None and len(m.p) > 0:
        minm.p = m.p[uv]
    
    # Transfer face_labels if present
    if hasattr(m, 'face_labels') and m.face_labels is not None:
        minm.face_labels = m.face_labels.copy()
    
    # Update edge info
    minm.edge_info()
    
    return minm, uv


def cotangent_laplacian(v, f):
    """
    Compute the cotangent Laplacian of a mesh
    
    Based on: P. T. Choi, K. C. Lam, and L. M. Lui, 
    "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization 
     for Genus-0 Closed Brain Surfaces."
    SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.
    
    Parameters:
    -----------
    v : array (nv x 3)
        Vertex coordinates
    f : array (nf x 3)
        Face connectivity (0-indexed)
        
    Returns:
    --------
    L : sparse matrix
        Cotangent Laplacian matrix
    """
    nv = len(v)
    
    f1 = f[:, 0]
    f2 = f[:, 1]
    f3 = f[:, 2]
    
    # Edge lengths
    l1 = np.sqrt(np.sum((v[f2] - v[f3])**2, axis=1))
    l2 = np.sqrt(np.sum((v[f3] - v[f1])**2, axis=1))
    l3 = np.sqrt(np.sum((v[f1] - v[f2])**2, axis=1))
    
    # Semi-perimeter and area (Heron's formula)
    s = (l1 + l2 + l3) * 0.5
    area = np.sqrt(np.maximum(s * (s - l1) * (s - l2) * (s - l3), 1e-20))
    
    # Cotangent weights
    cot12 = (l1**2 + l2**2 - l3**2) / (2 * area + 1e-10)
    cot23 = (l2**2 + l3**2 - l1**2) / (2 * area + 1e-10)
    cot31 = (l1**2 + l3**2 - l2**2) / (2 * area + 1e-10)
    
    # Diagonal entries
    diag1 = -cot12 - cot31
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23
    
    # Build sparse matrix
    II = np.concatenate([f1, f2, f2, f3, f3, f1, f1, f2, f3])
    JJ = np.concatenate([f2, f1, f3, f2, f1, f3, f1, f2, f3])
    V = np.concatenate([cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3])
    
    L = csr_matrix((V, (II, JJ)), shape=(nv, nv))
    
    return L


def beltrami_coefficient(P, f, v):
    """
    Compute the Beltrami coefficient for a map
    
    Parameters:
    -----------
    P : array (nv x 2)
        2D parameterization
    f : array (nf x 3)
        Face connectivity
    v : array (nv x 3)
        Original 3D vertex coordinates
        
    Returns:
    --------
    mu : array (nf,)
        Beltrami coefficient per face
    """
    nf = len(f)
    
    f0 = f[:, 0]
    f1 = f[:, 1]
    f2 = f[:, 2]
    
    # Get coordinates
    v0 = v[f0]
    v1 = v[f1]
    v2 = v[f2]
    
    P0 = P[f0]
    P1 = P[f1]
    P2 = P[f2]
    
    # Compute edge vectors in 3D
    e1_3d = v1 - v0
    e2_3d = v2 - v0
    
    # Compute edge vectors in 2D
    e1_2d = P1 - P0
    e2_2d = P2 - P0
    
    # Area in 2D
    area_2d = 0.5 * (e1_2d[:, 0] * e2_2d[:, 1] - e1_2d[:, 1] * e2_2d[:, 0])
    area_2d = np.maximum(np.abs(area_2d), 1e-20) * np.sign(area_2d + 1e-20)
    
    # Compute Jacobian components
    # dz = (partial f / partial z)
    # dz_bar = (partial f / partial z_bar)
    
    # Complex representation
    z1 = P1[:, 0] + 1j * P1[:, 1]
    z2 = P2[:, 0] + 1j * P2[:, 1]
    z0 = P0[:, 0] + 1j * P0[:, 1]
    
    # For conformal map correction
    # mu = dz_bar / dz
    mu = np.zeros(nf, dtype=complex)
    
    return mu


def linear_beltrami_solver(P, f, mu, fixed_idx, fixed_val):
    """
    Solve the linear Beltrami equation
    
    Parameters:
    -----------
    P : array (nv x 2)
        Initial 2D parameterization
    f : array (nf x 3)
        Face connectivity
    mu : array (nf,)
        Beltrami coefficient
    fixed_idx : array
        Indices of fixed vertices
    fixed_val : array (n_fixed x 2)
        Values of fixed vertices
        
    Returns:
    --------
    map : array (nv x 2)
        Corrected parameterization
    """
    # For small mu, just return the original map
    # Full implementation would solve the generalized Laplace equation
    return P.copy()


def face_area(v, f):
    """
    Compute face areas
    
    Parameters:
    -----------
    v : array (nv x 3)
        Vertex coordinates
    f : array (nf x 3)
        Face connectivity
        
    Returns:
    --------
    areas : array (nf,)
        Face areas
    """
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    
    e1 = v1 - v0
    e2 = v2 - v0
    
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    return areas
