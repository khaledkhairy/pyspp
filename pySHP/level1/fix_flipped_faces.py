"""
Fix flipped faces in a mesh
Translated from MATLAB level1/fix_flipped_faces.m
"""

import numpy as np

try:
    from ..utils import kk_sph2cart
except ImportError:
    from pySHP.utils import kk_sph2cart


def fix_flipped_faces(pm, verbose=False):
    """
    Fix faces whose normals point inward (for a mesh centered at origin).
    
    Parameters:
    -----------
    pm : surface_mesh
        Input mesh
    verbose : bool
        Whether to print information about flipped faces
        
    Returns:
    --------
    pm : surface_mesh
        Mesh with fixed face orientations
    Fnc : array
        Boolean array indicating which faces were correctly oriented
        (1 = correct, 0 = was flipped)
    """
    nfaces = len(pm.F)
    Fnc = np.ones(nfaces, dtype=int)
    
    # Compute face normals
    Fn = compute_face_normals(pm)
    
    for ix in range(nfaces):
        # Get centroid of face
        face_verts = pm.F[ix]
        centroid = np.mean(pm.X[face_verts], axis=0)
        
        # Check if normal points outward (same direction as centroid from origin)
        # This assumes mesh is roughly centered at origin
        dot_product = np.dot(Fn[ix], centroid)
        
        if dot_product < 0:
            # Normal points inward, flip the face
            if verbose:
                print(f'Flipping face: {ix}')
            pm.F[ix] = [pm.F[ix, 0], pm.F[ix, 2], pm.F[ix, 1]]
            Fnc[ix] = 0
    
    if np.any(Fnc == 0) and verbose:
        print(f'Fixed {np.sum(Fnc == 0)} flipped faces')
    
    return pm, Fnc


def compute_face_normals(pm):
    """
    Compute face normals for a mesh.
    
    Parameters:
    -----------
    pm : surface_mesh
        Input mesh
        
    Returns:
    --------
    Fn : array (nfaces x 3)
        Face normals
    """
    nfaces = len(pm.F)
    Fn = np.zeros((nfaces, 3))
    
    for i in range(nfaces):
        v0 = pm.X[pm.F[i, 0]]
        v1 = pm.X[pm.F[i, 1]]
        v2 = pm.X[pm.F[i, 2]]
        
        # Compute cross product
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        
        Fn[i] = normal
    
    return Fn


def check_face_orientations(pm, reference_point=None):
    """
    Check face orientations relative to a reference point.
    
    Parameters:
    -----------
    pm : surface_mesh
        Input mesh
    reference_point : array (3,), optional
        Point assumed to be inside the mesh. If None, uses origin.
        
    Returns:
    --------
    correct : array (nfaces,)
        Boolean array indicating correctly oriented faces
    n_flipped : int
        Number of incorrectly oriented faces
    """
    if reference_point is None:
        reference_point = np.zeros(3)
    
    Fn = compute_face_normals(pm)
    nfaces = len(pm.F)
    correct = np.ones(nfaces, dtype=bool)
    
    for i in range(nfaces):
        centroid = np.mean(pm.X[pm.F[i]], axis=0)
        direction = centroid - reference_point
        
        if np.dot(Fn[i], direction) < 0:
            correct[i] = False
    
    n_flipped = np.sum(~correct)
    
    return correct, n_flipped


def check_spherical_parameterization_normals(mesh, min_orient=1e-8, verbose=True):
    """
    Check the fully parameterized mesh (with t, p on unit sphere) for flipped faces
    or inconsistent normals: i.e. some normals pointing inward and some outward.
    
    Uses spherical positions from mesh.t and mesh.p: orient = dot(v1, cross(v2, v3))
    for each face. Positive orient = outward, negative = inward. If signs are
    mixed or any face is degenerate (|orient| < min_orient), the mesh is invalid
    for spherical harmonics.
    
    Parameters:
    -----------
    mesh : surface_mesh
        Must have .t and .p (theta, phi) set (e.g. after assemble_parameterized_mesh).
    min_orient : float
        Minimum |orientation| to consider a face non-degenerate.
    verbose : bool
        If True, print a summary and warnings.
        
    Returns:
    --------
    result : dict
        - all_consistent (bool): True iff all checked faces have same sign and are non-degenerate.
        - n_outward (int): number of faces with orient > 0.
        - n_inward (int): number of faces with orient < 0.
        - n_degenerate (int): number of faces with |orient| < min_orient.
        - n_skipped (int): faces skipped (vertex with t=0 and p=0).
        - n_checked (int): total faces checked.
        - flipped_face_indices (ndarray): face indices with orient < 0.
        - degenerate_face_indices (ndarray): face indices with |orient| < min_orient.
    """
    if not hasattr(mesh, 't') or not hasattr(mesh, 'p'):
        out = {
            'all_consistent': False,
            'n_outward': 0, 'n_inward': 0, 'n_degenerate': 0,
            'n_skipped': len(mesh.F), 'n_checked': 0,
            'flipped_face_indices': np.array([], dtype=int),
            'degenerate_face_indices': np.array([], dtype=int),
        }
        if verbose:
            print("check_spherical_parameterization_normals: mesh has no .t or .p; cannot check.")
        return out

    t = np.asarray(mesh.t)
    p = np.asarray(mesh.p)
    nvert = len(t)
    if len(p) != nvert:
        if verbose:
            print("check_spherical_parameterization_normals: len(t) != len(p); cannot check.")
        return {
            'all_consistent': False,
            'n_outward': 0, 'n_inward': 0, 'n_degenerate': 0,
            'n_skipped': len(mesh.F), 'n_checked': 0,
            'flipped_face_indices': np.array([], dtype=int),
            'degenerate_face_indices': np.array([], dtype=int),
        }

    # Vertex valid if at least one of t or p is non-zero (parameterized)
    valid_vertex = (t != 0) | (p != 0)
    # Spherical Cartesian on unit sphere (unused for invalid verts but we need no NaNs for indexing)
    u, v, w = kk_sph2cart(t, p, np.ones(nvert))
    X_sphere = np.column_stack([u, v, w])

    nfaces = len(mesh.F)
    orient_out = np.full(nfaces, np.nan)

    for ix in range(nfaces):
        i, j, k = mesh.F[ix, 0], mesh.F[ix, 1], mesh.F[ix, 2]
        if not (valid_vertex[i] and valid_vertex[j] and valid_vertex[k]):
            continue
        v1 = X_sphere[i]
        v2 = X_sphere[j]
        v3 = X_sphere[k]
        orient_out[ix] = np.dot(v1, np.cross(v2, v3))

    checked = ~np.isnan(orient_out)
    n_checked = int(np.sum(checked))
    n_skipped = nfaces - n_checked

    if n_checked == 0:
        out = {
            'all_consistent': False,
            'n_outward': 0, 'n_inward': 0, 'n_degenerate': 0,
            'n_skipped': n_skipped, 'n_checked': 0,
            'flipped_face_indices': np.array([], dtype=int),
            'degenerate_face_indices': np.array([], dtype=int),
        }
        if verbose:
            print("check_spherical_parameterization_normals: no faces with all vertices parameterized; cannot check.")
        return out

    o = orient_out[checked]
    n_outward = int(np.sum(o > 0))
    n_inward = int(np.sum(o < 0))
    degenerate_face_indices = np.where(checked & (np.abs(orient_out) < min_orient))[0]
    n_degenerate = len(degenerate_face_indices)
    flipped_face_indices = np.where(checked & (orient_out < -min_orient))[0]

    all_same_sign = (n_outward == 0 or n_inward == 0) and n_degenerate == 0
    all_consistent = bool(all_same_sign)

    result = {
        'all_consistent': all_consistent,
        'n_outward': n_outward,
        'n_inward': n_inward,
        'n_degenerate': n_degenerate,
        'n_skipped': n_skipped,
        'n_checked': n_checked,
        'flipped_face_indices': flipped_face_indices,
        'degenerate_face_indices': degenerate_face_indices,
    }

    if verbose:
        print("\n--- Full parameterized mesh: spherical normals check ---")
        print(f"  Faces checked (all vertices parameterized): {n_checked} / {nfaces}")
        if n_skipped:
            print(f"  Faces skipped (missing t,p): {n_skipped}")
        print(f"  Outward (orient > 0): {n_outward}")
        print(f"  Inward (orient < 0):  {n_inward}")
        print(f"  Degenerate (|orient| < {min_orient:.0e}): {n_degenerate}")
        if not all_consistent:
            print("  >>> INCONSISTENT: some normals inward, some outward (or degenerate).")
            print("      Spherical parameterization optimization / spherical harmonics may be invalid.")
            if len(flipped_face_indices) <= 20:
                print(f"      Flipped face indices: {flipped_face_indices.tolist()}")
            else:
                print(f"      Flipped face indices (first 20): {flipped_face_indices[:20].tolist()} ...")
        else:
            print("  All checked faces have consistent outward normals.")
        print("---\n")

    return result


def fix_spherical_parameterization_normals(mesh, min_orient=1e-8, verbose=True):
    """
    Fix inconsistent normals on the full parameterized mesh (with t, p on unit sphere)
    by flipping face winding for faces that point inward (orient < 0).
    
    Call this after assembling the full mesh from patches. Modifies mesh.F in place.
    
    Parameters:
    -----------
    mesh : surface_mesh
        Must have .t and .p set (e.g. after assemble_parameterized_mesh).
    min_orient : float
        Faces with orient < -min_orient are considered inward and flipped.
    verbose : bool
        If True, print how many faces were flipped.
        
    Returns:
    --------
    mesh : surface_mesh
        The same mesh with face windings fixed (in place).
    n_fixed : int
        Number of faces that were flipped.
    """
    result = check_spherical_parameterization_normals(mesh, min_orient=min_orient, verbose=False)
    flipped = result['flipped_face_indices']
    n_fixed = len(flipped)
    
    for ix in flipped:
        # Flip face winding: (0,1,2) -> (0,2,1) so normal reverses
        mesh.F[ix] = [mesh.F[ix, 0], mesh.F[ix, 2], mesh.F[ix, 1]]
    
    if verbose and n_fixed > 0:
        print(f"fix_spherical_parameterization_normals: flipped {n_fixed} inward-facing faces.")
    
    return mesh, n_fixed


def ensure_outward_normals(mesh, use_spherical=True, verbose=True):
    """
    Ensure all face normals point outward. Chooses the appropriate fix based on
    whether the mesh has spherical parameterization (t, p) or is a 3D mesh.
    
    Parameters
    ----------
    mesh : surface_mesh
        Mesh to fix (modified in place).
    use_spherical : bool
        If True and mesh has .t and .p, use fix_spherical_parameterization_normals
        (for parameterized meshes on the sphere). If False or no t/p, use
        fix_flipped_faces (for 3D meshes centered at origin).
    verbose : bool
        Print fix summary.
        
    Returns
    -------
    mesh : surface_mesh
        Mesh with outward normals (modified in place).
    n_fixed : int
        Number of faces that were flipped.
    """
    if use_spherical and hasattr(mesh, 't') and hasattr(mesh, 'p'):
        mesh, n_fixed = fix_spherical_parameterization_normals(mesh, verbose=verbose)
    else:
        _, Fnc = fix_flipped_faces(mesh, verbose=verbose)
        n_fixed = int(np.sum(Fnc == 0))
    return mesh, n_fixed
