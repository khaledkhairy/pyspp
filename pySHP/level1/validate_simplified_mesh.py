"""
Validation and repair functions for simplified patch mesh to ensure well-behaved topology.

A well-behaved simplified mesh must satisfy:
1. Every patch has at least one face (no missing patches)
2. Every vertex is part of at least one face (no floating vertices)
3. Every edge is shared by exactly 2 faces (manifold, no non-manifold edges)
4. Mesh is closed (no boundary edges)
5. Genus is zero (topological sphere)
6. All patches are connected (single connected component)
"""

import numpy as np
from scipy.sparse import csr_matrix


def validate_simplified_mesh(pm, PM, verbose=True):
    """
    Validate that the simplified mesh is well-behaved (manifold, genus-zero, closed).
    
    Parameters:
    -----------
    pm : surface_mesh
        Simplified patch mesh
    PM : dict
        Patch mesh structure
    verbose : bool
        Print detailed validation results
        
    Returns:
    --------
    is_valid : bool
        True if mesh passes all checks
    issues : dict
        Dictionary of issues found (empty if valid)
    """
    issues = {}
    
    if pm is None or pm.X is None or pm.F is None:
        issues['no_mesh'] = "Simplified mesh is None or missing X/F"
        return False, issues
    
    nV = len(pm.X)
    nF = len(pm.F)
    
    if nF == 0:
        issues['no_faces'] = "Simplified mesh has 0 faces"
        return False, issues
    
    # Check 1: Every patch has at least one face
    if hasattr(pm, 'face_labels') and pm.face_labels is not None:
        unique_labels = np.unique(pm.face_labels)
        npatches = PM.get('npatches', len(unique_labels))
        patches_with_faces = set(unique_labels.tolist())
        patches_missing = set(range(npatches)) - patches_with_faces
        if len(patches_missing) > 0:
            issues['missing_patches'] = {
                'patches': sorted(patches_missing),
                'count': len(patches_missing)
            }
    
    # Check 2: Every vertex is part of at least one face (no floating vertices)
    vertex_in_face = np.zeros(nV, dtype=bool)
    for f in pm.F:
        for v in f:
            if 0 <= v < nV:
                vertex_in_face[v] = True
    floating_vertices = np.where(~vertex_in_face)[0]
    if len(floating_vertices) > 0:
        issues['floating_vertices'] = {
            'indices': floating_vertices.tolist(),
            'count': len(floating_vertices)
        }
    
    # Check 3: Manifold edges (every edge shared by exactly 2 faces)
    pm.needs_edge_info = True
    pm.edge_info()
    if pm.E is not None and len(pm.E) > 0:
        edge_face_count = {}
        for fix, f in enumerate(pm.F):
            for i in range(3):
                v1, v2 = f[i], f[(i + 1) % 3]
                e = tuple(sorted([v1, v2]))
                edge_face_count[e] = edge_face_count.get(e, 0) + 1
        non_manifold_edges = [e for e, count in edge_face_count.items() if count != 2]
        if len(non_manifold_edges) > 0:
            issues['non_manifold_edges'] = {
                'edges': non_manifold_edges[:10],  # First 10
                'count': len(non_manifold_edges)
            }
    
    # Check 4: Closed mesh (no boundary edges)
    # Compute boundary from actual edge usage (authoritative), not border_vertex which can be stale
    edge_face_count = {}
    for fix, f in enumerate(pm.F):
        for i in range(3):
            v1, v2 = int(f[i]), int(f[(i + 1) % 3])
            e = tuple(sorted([v1, v2]))
            edge_face_count[e] = edge_face_count.get(e, 0) + 1
    n_boundary_edges = sum(1 for c in edge_face_count.values() if c == 1)
    if n_boundary_edges > 0:
        issues['has_boundary'] = {
            'boundary_edges': n_boundary_edges,
            'message': f"Mesh has {n_boundary_edges} boundary edges (should be closed)"
        }
    
    # Check 5: Genus zero
    nE = len(pm.E) if pm.E is not None else 0
    chi = nV - nE + nF
    genus = max(0, (2 - chi) // 2)
    if genus > 0:
        issues['genus'] = {
            'genus': genus,
            'euler_characteristic': chi,
            'message': f"Mesh has genus {genus} (not a sphere, need genus 0)"
        }
    
    # Check 6: Connected (single component)
    if pm.F is not None and len(pm.F) > 0:
        # Build vertex adjacency
        from scipy.sparse import lil_matrix
        adj = lil_matrix((nV, nV), dtype=bool)
        for f in pm.F:
            for i in range(3):
                v1, v2 = f[i], f[(i + 1) % 3]
                adj[v1, v2] = True
                adj[v2, v1] = True
        adj = adj.tocsr()
        # Find connected components
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(adj, directed=False, return_labels=True)
        if n_components > 1:
            component_sizes = [np.sum(labels == i) for i in range(n_components)]
            issues['disconnected'] = {
                'n_components': n_components,
                'component_sizes': component_sizes,
                'message': f"Mesh has {n_components} disconnected components"
            }
    
    is_valid = len(issues) == 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Simplified Mesh Validation:")
        print(f"  Vertices: {nV}")
        print(f"  Faces: {nF}")
        print(f"  Edges: {nE}")
        print(f"  Euler characteristic: {chi}")
        print(f"  Genus: {genus}")
        if is_valid:
            print(f"  Status: VALID (manifold, closed, genus-zero)")
        else:
            print(f"  Status: INVALID - {len(issues)} issue(s) found")
            for issue_name, issue_data in issues.items():
                print(f"    - {issue_name}: {issue_data}")
        print(f"{'='*60}\n")
    
    return is_valid, issues


def verify_pm_vertex_compatibility(PM, n_fine_vertices, verbose=True):
    """
    Verify that PM['Xkeyind'] is consistent with the simplified mesh and fine mesh.

    Every simplified vertex i must have Xkeyind[i] either:
    - >= 0 and < n_fine_vertices: corresponds to fine-mesh vertex (used in parameterization)
    - -1: fictitious vertex (topology only, excluded from fine-to-sphere mapping)

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm', 'Xkeyind'
    n_fine_vertices : int
        Number of vertices in the original fine mesh
    verbose : bool
        Print result

    Returns
    -------
    is_consistent : bool
        True if Xkeyind is valid
    """
    pm = PM.get('pm')
    Xkeyind = PM.get('Xkeyind')
    if pm is None or Xkeyind is None:
        return True  # Nothing to verify
    Xkeyind = np.asarray(Xkeyind).ravel()
    n_simpl = len(pm.X)
    if len(Xkeyind) != n_simpl:
        if verbose:
            print(f"  Vertex compatibility: MISMATCH - len(Xkeyind)={len(Xkeyind)} != n_simpl={n_simpl}")
        return False
    invalid = (Xkeyind >= 0) & (Xkeyind >= n_fine_vertices)
    n_invalid = np.sum(invalid)
    n_fictitious = np.sum(Xkeyind == -1)
    if n_invalid > 0:
        if verbose:
            bad = np.where(invalid)[0]
            print(f"  Vertex compatibility: INVALID - {n_invalid} vertices have Xkeyind >= n_fine={n_fine_vertices}: {bad[:5].tolist()}...")
        return False
    if verbose:
        print(f"  Vertex compatibility: OK ({n_simpl - n_fictitious} fine-mesh, {n_fictitious} fictitious)")
    return True


def diagnose_patch_boundary_issues(PM, verbose=True):
    """
    Diagnose why patches might have missing faces or incomplete boundary cycles.
    
    Parameters:
    -----------
    PM : dict
        Patch mesh structure
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    diagnostics : dict
        Per-patch diagnostics
    """
    diagnostics = {}
    npatches = PM.get('npatches', 0)
    
    for pix in range(npatches):
        diag = {
            'has_keys': False,
            'n_keys': 0,
            'n_incident_edges': 0,
            'n_sentinels': 0,
            'sentinels': [],
            'missing_sentinels': [],
            'potential_issues': []
        }
        
        # Check keys
        if len(PM['keys']) > 0:
            patch_keys = PM['keys'][PM['keys'][:, 0] == pix, 1].astype(int)
            diag['has_keys'] = len(patch_keys) > 0
            diag['n_keys'] = len(patch_keys)
        
        # Check incident edges and sentinels
        for eix in range(len(PM['Edges'])):
            if PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix:
                continue
            if PM['Edges'][eix, 1] < 0:
                continue
            diag['n_incident_edges'] += 1
            s1, s2 = int(PM['sentinels'][eix, 0]), int(PM['sentinels'][eix, 1])
            if s1 >= 0:
                diag['n_sentinels'] += 1
                diag['sentinels'].append(s1)
            if s2 >= 0 and s2 != s1:
                diag['n_sentinels'] += 1
                diag['sentinels'].append(s2)
            if s1 < 0 or s2 < 0:
                diag['missing_sentinels'].append(eix)
        
        diag['sentinels'] = list(set(diag['sentinels']))
        diag['n_sentinels'] = len(diag['sentinels'])
        
        # Identify potential issues
        if diag['n_keys'] == 0 and diag['n_sentinels'] < 2:
            diag['potential_issues'].append(f"Zero-key patch with < 2 sentinels (need fictitious keys)")
        if diag['n_sentinels'] < 2:
            diag['potential_issues'].append(f"Only {diag['n_sentinels']} sentinel(s) - cannot form boundary cycle")
        if len(diag['missing_sentinels']) > 0:
            diag['potential_issues'].append(f"Edges with missing sentinels: {diag['missing_sentinels']}")
        
        diagnostics[pix] = diag
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Patch Boundary Diagnostics:")
        for pix, diag in diagnostics.items():
            if len(diag['potential_issues']) > 0 or diag['n_keys'] == 0:
                print(f"  Patch {pix}:")
                print(f"    Keys: {diag['n_keys']}, Sentinels: {diag['n_sentinels']}, Incident edges: {diag['n_incident_edges']}")
                if len(diag['potential_issues']) > 0:
                    for issue in diag['potential_issues']:
                        print(f"    WARNING: {issue}")
        print(f"{'='*60}\n")
    
    return diagnostics
