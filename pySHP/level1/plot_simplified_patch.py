"""
Extract and plot isolated parts of the simplified mesh (one patch at a time).

Useful for inspecting individual patches (e.g. cap patches, cylinder patches) without the rest of the mesh.
Export to HTML for reliable interactive viewing when the Jupyter "client" backend is flaky.
"""

import numpy as np


def extract_simplified_submesh_for_patch(PM, pix):
    """
    Extract the submesh of the simplified mesh that contains only vertices
    and faces belonging to the given patch.

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm' (simplified mesh with .X, .F, .face_labels),
        and optionally 'Xkeyind', 'keys', 'CV'.
    pix : int
        Patch index.

    Returns
    -------
    result : dict
        - X : (n_verts, 3) vertex positions
        - F : (n_faces, 3) face indices (reindexed 0..n_verts-1)
        - face_labels : (n_faces,) all equal to pix
        - orig_to_sub : dict mapping original simplified-mesh vertex index -> submesh index
        - n_verts_orig : number of vertices in submesh (from original)
        - n_faces : number of faces
    """
    pm = PM.get('pm')
    if pm is None or pm.X is None or pm.F is None:
        return None
    fl = getattr(pm, 'face_labels', None)
    if fl is None:
        fl = np.zeros(len(pm.F), dtype=int)
    else:
        fl = np.asarray(fl).flatten()
    face_ix = np.where(fl == pix)[0]
    if len(face_ix) == 0:
        verts_used = set()
    else:
        verts_used = set(pm.F[face_ix].ravel().tolist())
    verts_used = np.array(sorted(verts_used))
    nv = len(verts_used)
    orig_to_sub = {int(v): i for i, v in enumerate(verts_used)}
    X_sub = pm.X[verts_used]
    F_orig = pm.F[face_ix]
    F_sub = np.array([[orig_to_sub[int(a)], orig_to_sub[int(b)], orig_to_sub[int(c)]]
                      for a, b, c in F_orig], dtype=np.int64)
    return {
        'X': X_sub,
        'F': F_sub,
        'face_labels': np.full(len(F_sub), pix, dtype=int),
        'orig_to_sub': orig_to_sub,
        'n_verts_orig': nv,
        'n_faces': len(F_sub),
        'patch_index': pix,
    }


def _cells_from_F(F):
    """PyVista cell array: [3, a, b, c, 3, ...] for triangles."""
    n = F.shape[0]
    return np.hstack([np.full((n, 1), 3), F]).flatten()


def _boundary_cycle_and_sequence_for_patch(PM, pix, sub):
    """
    Get boundary cycle (global vertex indices) and vertex -> sequence number map
    for patch pix. Uses PM['patch_self_intersection'] if available; else derives
    from submesh (boundary edges -> walk cycle).
    Returns (cycle_list, vertex_to_seq) where vertex_to_seq maps global simpl index -> int or str.
    """
    # Prefer stored diagnostic (run after generate_simplified_mesh)
    inter = PM.get('patch_self_intersection') or {}
    r = inter.get(pix, {})
    cycle = r.get('boundary_cycle')
    if cycle is not None and len(cycle) >= 2:
        vertex_to_seq = {}
        for k, v in enumerate(cycle):
            vertex_to_seq[int(v)] = k
        # Center and others not in cycle get no label or 'C'
        return list(cycle), vertex_to_seq
    # Fallback: derive from submesh (boundary edges, walk cycle)
    X, F = sub['X'], sub['F']
    orig_to_sub = sub['orig_to_sub']
    sub_to_orig = {j: i for i, j in orig_to_sub.items()}
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    vertex_degree = defaultdict(int)
    for fi in range(len(F)):
        tri = F[fi]
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in [(a, b), (b, c), (c, a)]:
            vertex_degree[u] += 1
            if u > v:
                u, v = v, u
            edge_to_faces[(u, v)].append(fi)
    single = [e for e, flist in edge_to_faces.items() if len(flist) == 1]
    if not single:
        return [], {}
    patch_verts = set()
    for e in single:
        patch_verts.update(e)
    center = max(patch_verts, key=lambda v: vertex_degree[v]) if patch_verts else None
    boundary_edges = [e for e in single if center not in e]
    if len(boundary_edges) < 2:
        return [], {}
    adj = defaultdict(list)
    for u, v in boundary_edges:
        adj[u].append(v)
        adj[v].append(u)
    start, next_v = boundary_edges[0]
    cycle_sub = [start, next_v]
    used = {(min(start, next_v), max(start, next_v))}
    while len(cycle_sub) < len(boundary_edges) + 1:
        cur = cycle_sub[-1]
        nbrs = [w for w in adj[cur] if (min(cur, w), max(cur, w)) not in used]
        if not nbrs:
            break
        nxt = nbrs[0]
        used.add((min(cur, nxt), max(cur, nxt)))
        if nxt == start:
            break
        cycle_sub.append(nxt)
    # Map submesh indices to global
    cycle_global = [sub_to_orig.get(s, s) for s in cycle_sub]
    vertex_to_seq = {}
    for k, v in enumerate(cycle_global):
        vertex_to_seq[int(v)] = k
    return cycle_global, vertex_to_seq


def plot_simplified_patch_isolated(PM, pix, title=None, show_edges=True, color=None,
                                   export_html_path=None, return_plotter=False,
                                   show_vertex_sequence=False, color_faces_individually=False):
    """
    Plot only the part of the simplified mesh that belongs to patch `pix`.

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm'.
    pix : int
        Patch index.
    title : str, optional
        Window/title text.
    show_edges : bool
        Draw edges.
    color : tuple or str, optional
        Face color (e.g. (0.45, 0.75, 1.0) for light blue). Ignored if color_faces_individually=True.
    export_html_path : str, optional
        If set, export an HTML file to this path so you can open it in a browser
        for reliable interactive rotation/zoom (avoids Jupyter client issues).
    return_plotter : bool
        If True, return the plotter without showing (so caller can add more or show).
    show_vertex_sequence : bool
        If True, label each vertex with its boundary cycle sequence number (0, 1, 2, ...).
    color_faces_individually : bool
        If True, color each face differently (by face index) to identify intersecting faces.

    Returns
    -------
    plotter : pyvista.Plotter or None
        If return_plotter=True, the plotter; else None after .show() or .export_html().
    """
    try:
        import pyvista as pv
    except ImportError:
        print('plot_simplified_patch_isolated: PyVista not available')
        return None

    sub = extract_simplified_submesh_for_patch(PM, pix)
    if sub is None or sub['n_faces'] == 0:
        print(f'No faces for patch {pix} in simplified mesh')
        return None

    cells = _cells_from_F(sub['F'])
    mesh = pv.PolyData(sub['X'], cells)
    plotter = pv.Plotter()

    if color_faces_individually:
        # Per-face scalar -> distinct color (helps identify which faces intersect)
        n_cells = sub['F'].shape[0]
        mesh.cell_data['face_id'] = np.arange(n_cells)
        plotter.add_mesh(mesh, scalars='face_id', cmap='tab20', show_edges=show_edges,
                        edge_color='black', line_width=1.0, scalar_bar_args={'title': 'Face'})
    else:
        if color is None:
            # Default color: cycle through a palette for different patches
            palette = [
                (0.45, 0.75, 1.0), (0.45, 1.0, 0.75), (1.0, 0.85, 0.45), (1.0, 0.65, 0.75),
                (0.65, 0.85, 1.0), (0.75, 1.0, 0.65), (1.0, 0.9, 0.6), (0.9, 0.7, 0.85),
            ]
            color = palette[pix % len(palette)]
        plotter.add_mesh(mesh, color=color, show_edges=show_edges, edge_color='black', line_width=1.0)

    if show_vertex_sequence:
        cycle_global, vertex_to_seq = _boundary_cycle_and_sequence_for_patch(PM, pix, sub)
        # Submesh vertex i corresponds to global index (keys of orig_to_sub are in sorted order)
        global_indices = sorted(sub['orig_to_sub'].keys())
        points = sub['X']
        labels = []
        for i in range(len(points)):
            global_idx = int(global_indices[i])
            seq = vertex_to_seq.get(global_idx)
            labels.append(str(seq) if seq is not None else 'C')
        plotter.add_point_labels(points, labels, font_size=12, point_size=15, render_points_as_spheres=True,
                                 always_visible=True)

    if title is None:
        title = f'Simplified mesh — Patch {pix} only ({sub["n_verts_orig"]} verts, {sub["n_faces"]} faces)'
    plotter.add_text(title, font_size=10)
    plotter.background_color = 'white'

    if export_html_path:
        try:
            plotter.export_html(export_html_path)
            print(f'Exported interactive plot to: {export_html_path}')
        except Exception as e:
            print(f'export_html failed ({e}); try: pip install trame. Opening window instead.')
            plotter.show()
        plotter.close()
        return None
    if return_plotter:
        return plotter
    plotter.show()
    return None


def plot_two_simplified_patches_isolated(PM, pix1, pix2, export_html_dir=None):
    """
    Create two separate plots (or two HTML files) for patches pix1 and pix2.
    If export_html_dir is set, write patch_<pix1>.html and patch_<pix2>.html there
    for reliable interactive viewing in the browser.

    Parameters
    ----------
    PM : dict
        Patch structure.
    pix1, pix2 : int
        Patch indices.
    export_html_dir : str, optional
        Directory path for HTML exports. If None, opens interactive windows instead.
    """
    try:
        import pyvista as pv
    except ImportError:
        print('PyVista not available')
        return

    if export_html_dir:
        import os
        os.makedirs(export_html_dir, exist_ok=True)
        path1 = os.path.join(export_html_dir, f'simplified_patch_{pix1}.html')
        path2 = os.path.join(export_html_dir, f'simplified_patch_{pix2}.html')
        plot_simplified_patch_isolated(PM, pix1, title=f'Patch {pix1} only', export_html_path=path1)
        plot_simplified_patch_isolated(PM, pix2, title=f'Patch {pix2} only', export_html_path=path2)
        print(f'Open in browser for interaction: {path1}, {path2}')
        return

    # Two separate plotters (two windows or two inline frames)
    plot_simplified_patch_isolated(PM, pix1, title=f'Simplified mesh — Patch {pix1} only')
    plot_simplified_patch_isolated(PM, pix2, title=f'Simplified mesh — Patch {pix2} only')


def extract_simplified_submesh_for_patches(PM, pix_list):
    """
    Extract the submesh of the simplified mesh that contains vertices and faces
    belonging to any of the given patches (e.g. two patches to inspect shared boundary).

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm' (simplified mesh with .X, .F, .face_labels).
    pix_list : list of int
        Patch indices.

    Returns
    -------
    result : dict or None
        - X : (n_verts, 3) vertex positions
        - F : (n_faces, 3) face indices (reindexed)
        - face_labels : (n_faces,) patch index per face
        - orig_to_sub : dict original vertex index -> submesh index
        - n_faces : int
    """
    pm = PM.get('pm')
    if pm is None or pm.X is None or pm.F is None:
        return None
    fl = getattr(pm, 'face_labels', None)
    if fl is None:
        fl = np.zeros(len(pm.F), dtype=int)
    else:
        fl = np.asarray(fl).flatten()
    pix_set = set(pix_list)
    face_ix = np.where(np.isin(fl, list(pix_set)))[0]
    if len(face_ix) == 0:
        return None
    verts_used = set(pm.F[face_ix].ravel().tolist())
    verts_used = np.array(sorted(verts_used))
    nv = len(verts_used)
    orig_to_sub = {int(v): i for i, v in enumerate(verts_used)}
    X_sub = pm.X[verts_used]
    F_orig = pm.F[face_ix]
    F_sub = np.array([[orig_to_sub[int(a)], orig_to_sub[int(b)], orig_to_sub[int(c)]]
                      for a, b, c in F_orig], dtype=np.int64)
    fl_sub = fl[face_ix]
    return {
        'X': X_sub,
        'F': F_sub,
        'face_labels': fl_sub,
        'orig_to_sub': orig_to_sub,
        'n_verts_orig': nv,
        'n_faces': len(F_sub),
        'patch_indices': pix_list,
    }


def plot_two_patches_combined(PM, pix1, pix2, export_html_path=None, title=None):
    """
    Plot the simplified mesh for both patches in one view (different colors per patch)
    to inspect the shared boundary. Exports to HTML if export_html_path is set.

    Parameters
    ----------
    PM : dict
        Patch structure.
    pix1, pix2 : int
        Patch indices.
    export_html_path : str, optional
        If set, export this combined view to HTML.
    title : str, optional
        Plot title.
    """
    try:
        import pyvista as pv
    except ImportError:
        print('PyVista not available')
        return None
    sub = extract_simplified_submesh_for_patches(PM, [pix1, pix2])
    if sub is None or sub['n_faces'] == 0:
        print(f'No faces for patches {pix1}, {pix2} in simplified mesh')
        return None
    fl = sub['face_labels']
    # Two meshes by patch for different colors
    cells = _cells_from_F(sub['F'])
    mesh = pv.PolyData(sub['X'], cells)
    if title is None:
        title = f'Simplified mesh — Patches {pix1} & {pix2} ({sub["n_verts_orig"]} verts, {sub["n_faces"]} faces)'
    plotter = pv.Plotter()
    mask1 = (fl == pix1)
    mask2 = (fl == pix2)
    if np.any(mask1):
        m1 = mesh.extract_cells(np.where(mask1)[0])
        plotter.add_mesh(m1, color=(0.45, 0.75, 1.0), show_edges=True, edge_color='black', line_width=1.0)
    if np.any(mask2):
        m2 = mesh.extract_cells(np.where(mask2)[0])
        plotter.add_mesh(m2, color=(0.45, 1.0, 0.75), show_edges=True, edge_color='black', line_width=1.0)
    plotter.add_text(title, font_size=10)
    plotter.background_color = 'white'
    if export_html_path:
        try:
            plotter.export_html(export_html_path)
            print(f'Exported pair plot to: {export_html_path}')
        except Exception as e:
            print(f'export_html failed: {e}')
        plotter.close()
        return None
    return plotter


def export_simplified_mesh_full_html(PM, export_html_path, title=None):
    """
    Export the full simplified mesh (all patches) to a single HTML file for
    reliable interactive inspection (rotate, pan, zoom) in the browser.

    Patches are colored distinctly so patch boundaries are visible.

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm' (simplified mesh with .X, .F, .face_labels).
    export_html_path : str
        Path for the output HTML file.
    title : str, optional
        Title shown in the plot. Default: "Simplified mesh (full)".

    Returns
    -------
    bool
        True if export succeeded, False otherwise.
    """
    try:
        import pyvista as pv
    except ImportError:
        print('export_simplified_mesh_full_html: PyVista not available')
        return False
    pm = PM.get('pm')
    if pm is None or pm.X is None or pm.F is None:
        print('export_simplified_mesh_full_html: PM["pm"] has no X or F')
        return False
    fl = getattr(pm, 'face_labels', None)
    if fl is None:
        fl = np.zeros(len(pm.F), dtype=int)
    else:
        fl = np.asarray(fl).flatten()
    cells = _cells_from_F(pm.F)
    mesh = pv.PolyData(pm.X, cells)
    if title is None:
        title = f'Simplified mesh (full) — {len(pm.X)} verts, {len(pm.F)} faces'
    plotter = pv.Plotter()
    # Distinct colors per patch (cycle through a palette so many patches remain distinguishable)
    palette = [
        (0.45, 0.75, 1.0), (0.45, 1.0, 0.75), (1.0, 0.85, 0.45), (1.0, 0.65, 0.75),
        (0.65, 0.85, 1.0), (0.75, 1.0, 0.65), (1.0, 0.9, 0.6), (0.9, 0.7, 0.85),
        (0.7, 0.9, 1.0), (0.8, 1.0, 0.8), (1.0, 0.95, 0.7), (0.95, 0.8, 0.9),
    ]
    patch_ids = np.unique(fl)
    for i, pix in enumerate(patch_ids):
        mask = (fl == pix)
        if not np.any(mask):
            continue
        m = mesh.extract_cells(np.where(mask)[0])
        color = palette[i % len(palette)]
        plotter.add_mesh(m, color=color, show_edges=True, edge_color='black', line_width=1.0)
    plotter.add_text(title, font_size=10)
    plotter.background_color = 'white'
    try:
        plotter.export_html(export_html_path)
        print(f'Exported full simplified mesh to: {export_html_path}')
        plotter.close()
        return True
    except Exception as e:
        print(f'export_simplified_mesh_full_html failed: {e}')
        plotter.close()
        return False


def export_cylinder_and_neighbors_simplified_meshes(PM, cylinder_pix=None, export_dir=None,
                                                     show_vertex_sequence=True, color_faces_individually=True):
    """
    Export simplified meshes for a cylinder patch and all its neighbors for debugging:
    - One HTML per patch (cylinder and each neighbor): simplified_patch_<pix>.html
    - One HTML per pair (cylinder, neighbor): simplified_pair_<cyl>_<neighbor>.html

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm', 'Edges'.
    cylinder_pix : int, optional
        Cylinder patch index. If None, auto-detect first cylinder patch from PM.
    export_dir : str, optional
        Directory for HTML files. If None, uses 'simplified_patch_html' under cwd.
    show_vertex_sequence : bool
        If True, label vertices with boundary cycle sequence (0, 1, 2, ...) in patch HTMLs.
    color_faces_individually : bool
        If True, color each face differently in patch HTMLs to identify intersecting faces.

    Returns
    -------
    list of str
        Paths to exported HTML files.
    """
    import os
    if export_dir is None:
        export_dir = os.path.join(os.getcwd(), 'simplified_patch_html')
    os.makedirs(export_dir, exist_ok=True)
    
    # Auto-detect cylinder patch if not provided
    if cylinder_pix is None:
        report = PM.get('patch_structure_report', {})
        cylinder_patches = report.get('cylinder_patches', [])
        if len(cylinder_patches) > 0:
            cylinder_pix = cylinder_patches[0]
        else:
            print('export_cylinder_and_neighbors_simplified_meshes: No cylinder patch found. Provide cylinder_pix or run patch_info_gen first.')
            return []
    
    Edges = PM.get('Edges', np.array([]).reshape(0, 2))
    neighbors = []
    for eix in range(len(Edges)):
        p1, p2 = int(Edges[eix, 0]), int(Edges[eix, 1])
        if p2 < 0:
            continue
        if p1 == cylinder_pix:
            neighbors.append(p2)
        elif p2 == cylinder_pix:
            neighbors.append(p1)
    neighbors = sorted(set(neighbors))
    exported = []
    # Individual patches: cylinder first, then each neighbor (with sequence numbers and per-face colors)
    for pix in [cylinder_pix] + neighbors:
        path = os.path.join(export_dir, f'simplified_patch_{pix}.html')
        plot_simplified_patch_isolated(PM, pix, title=f'Patch {pix} only', export_html_path=path,
                                      show_vertex_sequence=show_vertex_sequence,
                                      color_faces_individually=color_faces_individually)
        exported.append(path)
    # Pairs: (cylinder, neighbor) for each neighbor
    for other in neighbors:
        path = os.path.join(export_dir, f'simplified_pair_{cylinder_pix}_{other}.html')
        plot_two_patches_combined(PM, cylinder_pix, other, export_html_path=path,
                                  title=f'Patches {cylinder_pix} & {other} (shared boundary)')
        exported.append(path)
    print(f'Exported {len(exported)} HTML files to {export_dir}')
    return exported


def export_simplified_mesh_html_for_inspection(PM, export_dir=None, cylinder_pix=None):
    """
    Export all HTML views needed for reliable interactive inspection (rotate, pan, zoom)
    when PyVista in Jupyter is unreliable.

    Writes to export_dir:
    - simplified_mesh_full.html — full simplified mesh (all patches, colored by patch)
    - simplified_pair_<cylinder_pix>_<neighbor>.html — one file per cylinder–neighbor pair (if cylinder_pix provided)
    - simplified_patch_<pix>.html — one file per patch (cylinder + its neighbors, if cylinder_pix provided)

    Open the HTML files in a browser for stable interaction.

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm', 'Edges'.
    export_dir : str, optional
        Directory for HTML files. If None, uses 'simplified_patch_html' under cwd.
    cylinder_pix : int, optional
        Cylinder patch index for pair exports. If None, auto-detects first cylinder patch.

    Returns
    -------
    list of str
        Paths to all exported HTML files.
    """
    import os
    if export_dir is None:
        export_dir = os.path.join(os.getcwd(), 'simplified_patch_html')
    os.makedirs(export_dir, exist_ok=True)
    exported = []

    # 1) Full simplified mesh
    full_path = os.path.join(export_dir, 'simplified_mesh_full.html')
    if export_simplified_mesh_full_html(PM, full_path):
        exported.append(full_path)

    # 2) Cylinder patch + each neighbor (pair meshes), if cylinder_pix is provided or auto-detected
    if cylinder_pix is not None:
        exported_pairs = export_cylinder_and_neighbors_simplified_meshes(
            PM, cylinder_pix=cylinder_pix, export_dir=export_dir
        )
        exported.extend(exported_pairs)
        print(f'Inspection HTMLs: open in browser — full: {full_path}')
        print(f'  Pairs (patch {cylinder_pix} + neighbor): simplified_pair_{cylinder_pix}_*.html')
    else:
        print(f'Inspection HTMLs: open in browser — full: {full_path}')
    return exported


def _sphere_points_from_tp(t, p, radius=1.0):
    """
    Convert spherical coordinates (theta, phi) to Cartesian points on the unit sphere.

    Parameters
    ----------
    t, p : array-like
        Theta (colatitude) and phi (azimuth) in the KK convention.
    radius : float
        Sphere radius (default 1.0).

    Returns
    -------
    X : (n, 3) ndarray
        Cartesian coordinates on the sphere.
    """
    from ..utils import kk_sph2cart
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    r = np.full_like(t, float(radius))
    u, v, w = kk_sph2cart(t, p, r)
    return np.column_stack([u, v, w])


def export_simplified_mesh_spherical_parameterization_html(PM, export_html_path, t, p,
                                                           title=None, show_reference_sphere=True):
    """
    Export the simplified mesh drawn on the unit sphere (using given theta, phi)
    to an interactive HTML file.

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm' (simplified mesh with .F, .face_labels).
    export_html_path : str
        Path for the output HTML file.
    t, p : array-like
        Theta and phi for each vertex of the simplified mesh (same length as pm.X).
    title : str, optional
        Title shown in the plot.
    show_reference_sphere : bool
        If True, draw a semi-transparent reference sphere (default True).

    Returns
    -------
    bool
        True if export succeeded, False otherwise.
    """
    try:
        import pyvista as pv
    except ImportError:
        print('export_simplified_mesh_spherical_parameterization_html: PyVista not available')
        return False
    pm = PM.get('pm')
    if pm is None or pm.F is None:
        print('export_simplified_mesh_spherical_parameterization_html: PM["pm"] missing or no F')
        return False
    t = np.asarray(t).flatten()
    p = np.asarray(p).flatten()
    if len(t) != len(pm.X) or len(p) != len(pm.X):
        print(f'export_simplified_mesh_spherical_parameterization_html: t/p length {len(t)}/{len(p)} != n_verts {len(pm.X)}')
        return False
    X_sph = _sphere_points_from_tp(t, p, radius=1.0)
    cells = _cells_from_F(pm.F)
    mesh = pv.PolyData(X_sph, cells)
    fl = getattr(pm, 'face_labels', None)
    if fl is None:
        fl = np.zeros(len(pm.F), dtype=int)
    else:
        fl = np.asarray(fl).flatten()
    plotter = pv.Plotter()
    if show_reference_sphere:
        sphere = pv.Sphere(radius=0.98, theta_resolution=30, phi_resolution=30)
        plotter.add_mesh(sphere, color='cyan', opacity=0.25, show_edges=False)
    palette = [
        (0.45, 0.75, 1.0), (0.45, 1.0, 0.75), (1.0, 0.85, 0.45), (1.0, 0.65, 0.75),
        (0.65, 0.85, 1.0), (0.75, 1.0, 0.65), (1.0, 0.9, 0.6), (0.9, 0.7, 0.85),
        (0.7, 0.9, 1.0), (0.8, 1.0, 0.8), (1.0, 0.95, 0.7), (0.95, 0.8, 0.9),
    ]
    patch_ids = np.unique(fl)
    for i, pix in enumerate(patch_ids):
        mask = (fl == pix)
        if not np.any(mask):
            continue
        m = mesh.extract_cells(np.where(mask)[0])
        color = palette[i % len(palette)]
        plotter.add_mesh(m, color=color, show_edges=True, edge_color='black', line_width=0.5, opacity=0.95)
    if title is None:
        title = 'Simplified mesh on sphere'
    plotter.add_text(title, font_size=10)
    plotter.background_color = 'white'
    try:
        plotter.export_html(export_html_path)
        print(f'Exported spherical parameterization to: {export_html_path}')
        plotter.close()
        return True
    except Exception as e:
        print(f'export_simplified_mesh_spherical_parameterization_html failed: {e}')
        plotter.close()
        return False


def export_simplified_mesh_initial_spherical_parameterization_html(PM, export_html_path,
                                                                     title=None, show_reference_sphere=True):
    """
    Compute the initial spherical parameterization of the simplified mesh (bijective map only,
    no Newton optimization) and export it to an interactive HTML file.

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm' (simplified mesh with .X, .F).
    export_html_path : str
        Path for the output HTML file.
    title : str, optional
        Title shown in the plot. Default: "Simplified mesh — initial spherical parameterization".
    show_reference_sphere : bool
        If True, draw a semi-transparent reference sphere (default True).

    Returns
    -------
    bool
        True if export succeeded, False otherwise.
    """
    from ..surface_mesh import surface_mesh
    pm = PM.get('pm')
    if pm is None or pm.X is None or pm.F is None:
        print('export_simplified_mesh_initial_spherical_parameterization_html: PM["pm"] missing X or F')
        return False
    # Build temporary mesh to get L (neighbor list) and run bijective_map_gen
    m = surface_mesh(pm.X.copy(), pm.F.copy())
    m.props()
    if m.L is None or len(m.L) == 0:
        print('export_simplified_mesh_initial_spherical_parameterization_html: could not compute neighbor list')
        return False
    try:
        t, p, dtline, W, A, b, ixN, ixS = surface_mesh.bijective_map_gen(
            m.X, m.F, m.L, plotflag=0, ixN=None, ixS=None)
    except Exception as e:
        print(f'export_simplified_mesh_initial_spherical_parameterization_html: bijective_map_gen failed: {e}')
        return False
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(p)):
        print('export_simplified_mesh_initial_spherical_parameterization_html: initial t/p contain NaN/Inf')
        return False
    if title is None:
        title = 'Simplified mesh — initial spherical parameterization'
    return export_simplified_mesh_spherical_parameterization_html(
        PM, export_html_path, t, p, title=title, show_reference_sphere=show_reference_sphere)


def export_simplified_mesh_final_spherical_parameterization_html(PM, export_html_path,
                                                                 optimization_method=2, newton_niter=100,
                                                                 newton_step=0.2, prevent_flip=True,
                                                                 title=None, show_reference_sphere=True):
    """
    Compute the final spherical parameterization of the simplified mesh (initial bijective map
    plus Newton optimization) and export it to an interactive HTML file.

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm' (simplified mesh with .X, .F, .face_labels).
    export_html_path : str
        Path for the output HTML file.
    optimization_method : int
        surface_mesh map2sphere method (default 2 = area/shear correction).
    newton_niter : int
        Number of Newton iterations (default 100).
    newton_step : float
        Step factor for Newton (default 0.2).
    prevent_flip : bool
        Whether to prevent face flips during optimization (default True).
    title : str, optional
        Title shown in the plot. Default: "Simplified mesh — final spherical parameterization".
    show_reference_sphere : bool
        If True, draw a semi-transparent reference sphere (default True).

    Returns
    -------
    bool
        True if export succeeded, False otherwise.
    """
    from ..surface_mesh import surface_mesh
    pm = PM.get('pm')
    if pm is None or pm.X is None or pm.F is None:
        print('export_simplified_mesh_final_spherical_parameterization_html: PM["pm"] missing X or F')
        return False
    # Work on a copy so we do not mutate PM['pm']
    m = surface_mesh(pm.X.copy(), pm.F.copy())
    if hasattr(pm, 'face_labels') and pm.face_labels is not None:
        m.face_labels = np.asarray(pm.face_labels).flatten().copy()
    m.optimization_method = optimization_method
    m.newton_niter = newton_niter
    m.newton_step = newton_step
    m.prevent_flip = prevent_flip
    m.bijective_plot_flag = 0
    try:
        m.map2sphere()
    except Exception as e:
        print(f'export_simplified_mesh_final_spherical_parameterization_html: map2sphere failed: {e}')
        return False
    if m.t is None or m.p is None or np.any(~np.isfinite(m.t)) or np.any(~np.isfinite(m.p)):
        print('export_simplified_mesh_final_spherical_parameterization_html: final t/p invalid or NaN/Inf')
        return False
    if title is None:
        title = 'Simplified mesh — final spherical parameterization'
    return export_simplified_mesh_spherical_parameterization_html(
        PM, export_html_path, m.t, m.p, title=title, show_reference_sphere=show_reference_sphere)


def export_simplified_mesh_spherical_parameterization_html_both(PM, export_dir=None,
                                                                optimization_method=2, newton_niter=100,
                                                                newton_step=0.2, prevent_flip=True):
    """
    Export both initial and final spherical parameterization of the simplified mesh to HTML.

    Writes:
    - simplified_mesh_sphere_initial.html — initial (bijective map only)
    - simplified_mesh_sphere_final.html   — final (after Newton optimization)

    Parameters
    ----------
    PM : dict
        Patch structure with 'pm'.
    export_dir : str, optional
        Directory for HTML files. If None, uses 'simplified_patch_html' under cwd.
    optimization_method : int
        Passed to final parameterization (default 2).
    newton_niter : int
        Passed to final parameterization (default 100).
    newton_step : float
        Passed to final parameterization (default 0.2).
    prevent_flip : bool
        Passed to final parameterization (default True).

    Returns
    -------
    list of str
        Paths to exported HTML files (initial, final).
    """
    import os
    if export_dir is None:
        export_dir = os.path.join(os.getcwd(), 'simplified_patch_html')
    os.makedirs(export_dir, exist_ok=True)
    paths = []
    initial_path = os.path.join(export_dir, 'simplified_mesh_sphere_initial.html')
    if export_simplified_mesh_initial_spherical_parameterization_html(PM, initial_path):
        paths.append(initial_path)
    final_path = os.path.join(export_dir, 'simplified_mesh_sphere_final.html')
    if export_simplified_mesh_final_spherical_parameterization_html(
            PM, final_path,
            optimization_method=optimization_method, newton_niter=newton_niter,
            newton_step=newton_step, prevent_flip=prevent_flip):
        paths.append(final_path)
    if paths:
        print(f'Spherical parameterization HTMLs: {paths}')
    return paths
