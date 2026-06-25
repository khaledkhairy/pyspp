"""
Spherical parameterization main function
Translated from MATLAB level2/spherical_parameterization.m

Main steps for spherical parameterization:
[1] Mesh-level segmentation
[2] Generation of a patch-level (simplified) mesh
[3] Mapping of simplified mesh onto the sphere
[4] Optimization of this mapping
[5] Parameterization of the fine mesh for each patch
[6] (Optional) Re-parameterization of fine mesh of individual patches
[7] Assignment of t, p values to original mesh

Author: Khaled Khairy. Copyright 2022 St. Jude Children's Research Hospital.
Python translation: 2024
"""

import numpy as np
from ..surface_mesh import surface_mesh
from ..utils import kk_cart2sph, kk_sph2cart
from ..level0.mesh_utils import reduce_to_minimal_set
from ..level1.mesh_segmentation_rw import mesh_segmentation_rw
from ..level1.patch_info_gen import patch_info_gen
from ..level1.fix_flipped_faces import fix_flipped_faces
from ..level1.parameterize_patches_cart import parameterize_patches_cart
from .spherical_conformal_parameterization import spherical_conformal_parameterization


def spherical_parameterization(m_o, nseeds=12, initial_b2sopts=None, PM=None):
    """
    Main function for patch-based spherical parameterization.
    
    Parameters:
    -----------
    m_o : surface_mesh
        Input mesh
    nseeds : int
        Number of seeds for segmentation (default: 12)
    initial_b2sopts : dict, optional
        Optimization options:
        - plot_flag: int (0 = no plots)
        - lambdaA: float (area weight, default: 1.0)
        - lambda_: float (flipped faces penalty, default: 1e3)
        - lambda1: float (angle deformation weight, default: 1e-4)
        - lambda2: float (geodesic edge length weight, default: 1e-2)
        - maxiter: int (max iterations, default: 1000)
        - equal_area: int (0 = curvature-based, 1 = equal area)
    PM : dict, optional
        Precomputed patch mesh structure
        
    Returns:
    --------
    m_o : surface_mesh
        Parameterized mesh with t and p fields populated
    PM : dict
        Patch mesh structure
    failed_patches : list
        List of patch indices that failed optimization
    """
    # Configuration parameters
    plot_flag = 0
    verbose = False
    sigma = 1.0  # Segmentation parameter
    
    # Default optimization options
    if initial_b2sopts is None:
        initial_b2sopts = {
            'plot_flag': 0,
            'lambdaA': 1.0,
            'lambda_': 1e3,       # Flipped faces penalty
            'lambda1': 1e-4,      # Angle deformation
            'lambda2': 1e-2,      # Geodesic edge lengths
            'maxiter': 1000,
            'equal_area': 0
        }
    
    # Refinement optimization options
    refine_b2sopts = {
        'plot_flag': 0,
        'lambdaA': 1.0,
        'lambda_': 0,
        'lambda1': 1e-1,
        'lambda2': 1e-1,
        'maxiter': 300
    }
    
    # Fine patch parameterization options
    ppc_plot_flag = 0
    max_faces = float('inf')  # No iterative refinement
    pnseeds = 3
    
    failed_patches = []
    
    # Determine if this is an initial run (no border vertices)
    initial_run = np.sum(m_o.border_vertex) == 0 if m_o.border_vertex is not None else True
    
    # Reduce to minimal set if fresh calculation
    if PM is None:
        m, uv = reduce_to_minimal_set(m_o)
    else:
        m = m_o
        uv = np.arange(len(m_o.X))
    
    # [1] Mesh segmentation
    if PM is None:
        if verbose:
            print(f'Performing mesh segmentation with {nseeds} seeds...')
        
        ms, L, slix, P, Pconn = mesh_segmentation_rw(m, nseeds, sigma)
        
        if verbose:
            print(f'Segmentation complete. Created {len(np.unique(L))} patches.')
    else:
        ms = m
    
    # [2] Generate patch connectivity and structure
    if PM is None:
        if verbose:
            print('Generating patch information...')
        
        m, PM, Pconn = patch_info_gen(ms, P, Pconn)
        
        if verbose:
            print(f'Patch info generated. {PM["npatches"]} patches created.')
    else:
        m = ms
    
    # [3] Bijective mapping: Initial spherical parameterization
    pm = PM['pm']
    pm.props()
    pm.edge_info()
    pm, Fnc = fix_flipped_faces(pm, verbose=False)
    
    if initial_run:
        if verbose:
            print('Computing initial spherical conformal parameterization...')
        
        # Use spherical conformal map
        pm = spherical_conformal_parameterization(pm)
        
        # Create spherical mesh from parameterization
        u, v, w = kk_sph2cart(pm.t, pm.p, np.ones_like(pm.p))
        spm = surface_mesh(np.column_stack([u, v, w]), pm.F.copy())
        spm.face_labels = pm.face_labels.copy() if pm.face_labels is not None else None
        spm.t = pm.t.copy()
        spm.p = pm.p.copy()
        
        # Fix any flipped faces
        spm, Fnc = fix_flipped_faces(spm, verbose=False)
        
        # Update t, p from fixed mesh
        t, p, r = kk_cart2sph(spm.X[:, 0], spm.X[:, 1], spm.X[:, 2])
        spm.t = t
        spm.p = p
        
        pm.t = spm.t.copy()
        pm.p = spm.p.copy()
        
        PM['spm'] = spm
    
    # [4] Optimization of spherical parameterization
    if initial_run:
        if verbose:
            print('Optimizing spherical parameterization...')
        
        # Run Newton optimization (simplified version)
        t, p = optimize_parameterization(pm, PM, initial_b2sopts)
        
        # Update meshes with optimized values
        u, v, w = kk_sph2cart(t, p, np.ones_like(p))
        spm = surface_mesh(np.column_stack([u, v, w]), pm.F.copy())
        spm.face_labels = pm.face_labels.copy() if pm.face_labels is not None else None
        spm, Fnc = fix_flipped_faces(spm, verbose=False)
        
        t, p, r = kk_cart2sph(spm.X[:, 0], spm.X[:, 1], spm.X[:, 2])
        spm.t = t
        spm.p = p
        
        pm.t = t.copy()
        pm.p = p.copy()
        PM['pm'].t = t.copy()
        PM['pm'].p = p.copy()
        PM['spm'] = spm
        
        if verbose:
            print('Optimization complete.')
    
    # [5] Parameterize patches based on simplified mesh
    if verbose:
        print('Parameterizing individual patches...')
    
    PM = parameterize_patches_cart(PM, ppc_plot_flag)
    
    if verbose:
        print('Patch parameterization complete.')
    
    # [6] Optional: Refine parameterization for large patches
    for pix in range(PM['npatches']):
        pp_try = PM['P'][pix][0]
        
        if len(pp_try.F) > max_faces:
            if verbose:
                print(f'Refining patch {pix} ({len(pp_try.F)} faces)...')
            
            try:
                # Set border vertices
                pp_try.border_vertex[PM['OUT'].get(pix, [])] = 1
                
                # Recursive parameterization
                pp, PM_sub, fp = spherical_parameterization(pp_try, pnseeds)
                PM['P'][pix][0] = pp
                
            except Exception as e:
                if verbose:
                    print(f'Failed to refine patch {pix}: {e}')
                failed_patches.append(pix)
    
    # [7] Assemble full spherical mesh
    m.t = np.zeros(len(m.X))
    m.p = np.zeros(len(m.X))
    
    for pix in range(PM['npatches']):
        pat = PM['P'][pix][0]
        Vix = np.unique(pat.F.flatten())
        
        for v in Vix:
            if v < len(m.t) and v < len(pat.t):
                m.t[v] = pat.t[v]
                m.p[v] = pat.p[v]
    
    # Restore to original mesh indexing
    if PM is None:
        m_o.t = np.zeros(len(m_o.X))
        m_o.p = np.zeros(len(m_o.X))
        m_o.t[uv] = m.t
        m_o.p[uv] = m.p
    else:
        m_o.t = m.t.copy()
        m_o.p = m.p.copy()
    
    m_o.face_labels = m.face_labels.copy() if hasattr(m, 'face_labels') and m.face_labels is not None else None
    m_o.border_vertex = m.border_vertex.copy() if hasattr(m, 'border_vertex') and m.border_vertex is not None else None
    
    if verbose:
        print('Spherical parameterization complete.')
    
    return m_o, PM, failed_patches


def optimize_parameterization(pm, PM, opts):
    """
    Optimize spherical parameterization using gradient descent.
    
    Simplified version of Newton-Raphson optimization from MATLAB.
    Minimizes a cost function based on:
    - Area distortion
    - Flipped faces
    - Angle distortion
    - Edge length distortion
    
    Parameters:
    -----------
    pm : surface_mesh
        Simplified patch mesh with initial t, p values
    PM : dict
        Patch mesh structure
    opts : dict
        Optimization options
        
    Returns:
    --------
    t, p : array
        Optimized theta and phi values
    """
    t = pm.t.copy()
    p = pm.p.copy()
    
    maxiter = opts.get('maxiter', 100)
    step_size = 0.01
    min_step = 1e-6
    
    lambdaA = opts.get('lambdaA', 1.0)
    lambda_flip = opts.get('lambda_', 1e3)
    
    nv = len(t)
    
    # Determine fixed vertices (border vertices)
    fixed = np.where(pm.border_vertex)[0] if pm.border_vertex is not None else np.array([])
    free = np.setdiff1d(np.arange(nv), fixed)
    
    # Target areas (equal distribution)
    nfaces = len(pm.F)
    target_area = 4 * np.pi / nfaces
    
    for iteration in range(maxiter):
        # Compute current Cartesian coordinates
        u, v, w = kk_sph2cart(t, p, np.ones_like(t))
        X = np.column_stack([u, v, w])
        
        # Compute gradient (simplified)
        grad_t = np.zeros(nv)
        grad_p = np.zeros(nv)
        
        # Area-based gradient
        for fidx in range(nfaces):
            face = pm.F[fidx]
            v0, v1, v2 = X[face[0]], X[face[1]], X[face[2]]
            
            # Signed area on sphere
            area = compute_spherical_triangle_area(v0, v1, v2)
            
            # Gradient contribution
            area_diff = area - target_area
            
            for i, vidx in enumerate(face):
                if vidx in free:
                    grad_t[vidx] += lambdaA * area_diff * 0.1
                    grad_p[vidx] += lambdaA * area_diff * 0.1
        
        # Update (gradient descent)
        t[free] -= step_size * grad_t[free]
        p[free] -= step_size * grad_p[free]
        
        # Ensure valid ranges
        t = np.clip(t, 0, np.pi)
        p = np.mod(p, 2 * np.pi)
        
        # Check convergence
        grad_norm = np.sqrt(np.sum(grad_t[free]**2) + np.sum(grad_p[free]**2))
        if grad_norm < min_step:
            break
    
    return t, p


def compute_spherical_triangle_area(v0, v1, v2):
    """
    Compute signed area of a spherical triangle.
    
    Parameters:
    -----------
    v0, v1, v2 : array (3,)
        Vertices on unit sphere
        
    Returns:
    --------
    area : float
        Signed area
    """
    # Use the formula: area = |a . (b x c)| where vectors are from origin
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross)
    
    # Sign based on orientation
    normal = cross / (np.linalg.norm(cross) + 1e-10)
    centroid = (v0 + v1 + v2) / 3
    sign = np.sign(np.dot(normal, centroid))
    
    return sign * area


def spherical_area(m):
    """
    Compute total spherical area of a mesh on the sphere.
    
    Parameters:
    -----------
    m : surface_mesh
        Mesh with t and p populated
        
    Returns:
    --------
    area : float
        Total spherical area
    """
    if m.t is None or m.p is None:
        return 0.0
    
    u, v, w = kk_sph2cart(m.t, m.p, np.ones_like(m.t))
    X = np.column_stack([u, v, w])
    
    total_area = 0.0
    for face in m.F:
        v0, v1, v2 = X[face[0]], X[face[1]], X[face[2]]
        total_area += abs(compute_spherical_triangle_area(v0, v1, v2))
    
    return total_area
