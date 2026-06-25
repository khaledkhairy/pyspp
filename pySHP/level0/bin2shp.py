"""
Binary image to spherical harmonics conversion
Translated from MATLAB level0/bin2shp.m
Simplified version - focuses on mesh input
"""

import numpy as np
from ..surface_mesh import surface_mesh
from ..shp_surface import shp_surface
from ..sh_basis import sh_basis
from ..utils import readoff


def bin2shp(mask, bin2shp_options=None):
    """
    Convert binary image or mesh to spherical harmonics
    
    Parameters:
    -----------
    mask : str or surface_mesh
        Either a filename (.off) or surface_mesh object
    bin2shp_options : dict, optional
        Options dictionary
        
    Returns:
    --------
    s : shp_surface
        Spherical harmonics surface
    m : surface_mesh
        Mesh object
    co_out : dict
        Additional output (placeholder)
    """
    # Default options
    if bin2shp_options is None:
        bin2shp_options = {
            'L_max': 12,
            'gdim': 60,
            'dim': 64,
            'verbose': True,
            'plot_flag': False,
            'meshresample_keepratio': 0.7
        }
    
    # Handle input
    if isinstance(mask, str):
        # File input
        if mask.endswith('.off'):
            # Read mesh file
            X, F = readoff(mask)
            m = surface_mesh(X, F)
        else:
            raise ValueError(f"Unsupported file format: {mask}")
    elif isinstance(mask, surface_mesh):
        # Direct mesh input
        m = mask
    else:
        raise ValueError("Input must be filename (.off) or surface_mesh object")
    
    # Optimize mesh
    if bin2shp_options.get('meshresample_keepratio', 1.0) < 1.0:
        m.meshresample_keepratio = bin2shp_options['meshresample_keepratio']
        m = m.optimize_mesh()
    
    # Create spherical harmonics representation
    L_max = bin2shp_options.get('L_max', 12)
    gdim = bin2shp_options.get('gdim', 60)
    
    b = sh_basis(L_max, gdim)
    s = shp_surface(L_max, b, m)
    
    co_out = {}
    
    return s, m, co_out


def bin2shp_options_template():
    """
    Return template options dictionary
    """
    return {
        'L_max': 12,
        'gdim': 60,
        'dim': 64,
        'verbose': True,
        'plot_flag': False,
        'meshresample_keepratio': 0.7,
        'fn_out': 'shp3_out.shp3'
    }
