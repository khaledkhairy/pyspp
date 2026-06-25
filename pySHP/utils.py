"""
Utility functions for pySHP
Translated from MATLAB level0 functions
"""

import numpy as np
from scipy.spatial.transform import Rotation


def kk_cart2sph(u, v, w):
    """
    Convert Cartesian to spherical coordinates (KK convention)
    
    Parameters:
    -----------
    u, v, w : array
        Cartesian coordinates
        
    Returns:
    --------
    t, p, r : array
        Spherical coordinates (theta, phi, radius)
        t is colatitude, p is azimuth
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    r = np.sqrt(u**2 + v**2 + w**2)
    # Avoid division by zero for points at/near origin (causes invalid value in arcsin)
    r_safe = np.maximum(r, 1e-15)

    # Use scipy's cart2sph (returns azimuth, elevation, radius)
    # azimuth is angle from x-axis, elevation is angle from xy-plane
    theta = np.arctan2(v, u)  # azimuth
    phi = np.arcsin(np.clip(w / r_safe, -1.0, 1.0))  # elevation (clip for numerics)
    
    # KK convention: p = theta (azimuth), t = pi/2 - phi (colatitude)
    p = theta
    t = np.pi/2 - phi
    
    return t, p, r


def kk_sph2cart(t, p, r):
    """
    Convert spherical to Cartesian coordinates (KK convention)
    
    Parameters:
    -----------
    t, p, r : array
        Spherical coordinates (theta=colatitude, phi=azimuth, radius)
        
    Returns:
    --------
    u, v, w : array
        Cartesian coordinates
    """
    t = np.asarray(t)
    p = np.asarray(p)
    r = np.asarray(r)
    
    # KK convention: phi = pi/2 - t, theta = p
    phi = np.pi/2 - t
    theta = p
    
    # Standard sph2cart conversion
    u = r * np.cos(phi) * np.cos(theta)
    v = r * np.cos(phi) * np.sin(theta)
    w = r * np.sin(phi)
    
    return u, v, w


def kk_cross(u, v):
    """
    Cross product of 3-vectors
    
    Parameters:
    -----------
    u, v : array
        3-vectors (N x 3)
        
    Returns:
    --------
    rvec : array
        Cross product (N x 3)
    """
    u = np.asarray(u)
    v = np.asarray(v)
    
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    
    x = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    y = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    z = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    
    return np.column_stack([x, y, z])


def kk_dot(u, v):
    """
    Dot product of 3-vectors
    
    Parameters:
    -----------
    u, v : array
        3-vectors (N x 3)
        
    Returns:
    --------
    rvec : array
        Dot product (N,)
    """
    u = np.asarray(u)
    v = np.asarray(v)
    
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    
    return u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1] + u[:, 2] * v[:, 2]


def kk_iseven(n):
    """
    Check if number is even
    """
    return (n % 2) == 0


def indices_gen(c):
    """
    Generate l, m indices for spherical harmonics coefficients
    
    Parameters:
    -----------
    c : array
        Coefficient vector
        
    Returns:
    --------
    l, m : array
        Degree and order indices
    flags : array
        Flags indicating non-zero coefficients
    """
    c = np.asarray(c)
    dim = len(c)
    
    l = np.zeros(dim, dtype=int)
    m = np.zeros(dim, dtype=int)
    
    counter = 0
    lval = 0
    
    while counter < dim:
        mval = -lval
        for i in range(2 * lval + 1):
            if counter >= dim:
                break
            l[counter] = lval
            m[counter] = mval
            mval += 1
            counter += 1
        lval += 1
    
    RNDOFF = 1e-10
    flags = np.ones(len(c), dtype=int)
    for i in range(len(c)):
        if abs(c[i]) < RNDOFF:
            flags[i] = 0
    
    return l, m, flags


def readoff(fname):
    """
    Read Geomview Object File Format (OFF)
    
    Parameters:
    -----------
    fname : str
        Filename
        
    Returns:
    --------
    node : array
        Vertex coordinates (N x 3)
    elem : array
        Face indices (M x 3 or M x 4)
    """
    import meshio
    
    mesh = meshio.read(fname)
    return mesh.points, mesh.cells[0].data


def writeoff(fname, vertices, faces):
    """
    Write Geomview Object File Format (OFF)
    
    Parameters:
    -----------
    fname : str
        Filename
    vertices : array
        Vertex coordinates (N x 3)
    faces : array
        Face indices (M x 3)
    """
    import meshio
    
    # Convert to 0-indexed if needed
    faces = np.asarray(faces)
    if faces.min() > 0:
        faces = faces - 1
    
    cells = [("triangle", faces)]
    mesh = meshio.Mesh(vertices, cells)
    mesh.write(fname)
