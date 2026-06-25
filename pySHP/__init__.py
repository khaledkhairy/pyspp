"""
pySHP - Python Spherical Harmonics Parameterization
A Python translation of the MATLAB shp_toolbox for spherical parameterization
and spherical harmonics analysis of 3D meshes.
"""

__version__ = "0.1.0"

from .sh_basis import sh_basis
from .sh_surface import sh_surface
from .shp_surface import shp_surface
from .surface_mesh import surface_mesh

# Level functions
from .level0 import bin2shp
from .level1 import mesh_segmentation_rw, patch_info_gen
from .level2 import spherical_parameterization, build_patch_data_structure

__all__ = [
    'sh_basis', 'sh_surface', 'shp_surface', 'surface_mesh',
    'bin2shp', 'mesh_segmentation_rw', 'patch_info_gen',
    'spherical_parameterization', 'build_patch_data_structure'
]
