"""
Surface mesh class for 3D mesh operations
Translated from MATLAB @surface_mesh class

NOTE (PyVista in Jupyter): Plotting automatically tries backends in order: client, trame, panel, static.
If you see "Loading..." and no plot, force a working backend at the start of your notebook:
  import pyvista as pv
  pv.set_jupyter_backend('static')   # non-interactive but always shows an image
  # or try: pv.set_jupyter_backend('client')  # interactive, works in many environments
Then re-run the plotting cell. You can also call surface_mesh.fix_pyvista_backend() before plotting.
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy import sparse
from scipy.sparse.linalg import gmres, spsolve, lsqr
from scipy.linalg import solve
from scipy.sparse.csgraph import shortest_path
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
from .utils import kk_cart2sph, kk_sph2cart, kk_cross, kk_dot, writeoff


def _ensure_pyvista_jupyter_backend():
    """
    Ensure PyVista's Jupyter backend (trame) is properly initialized.
    
    This fixes the "Loading..." spinner issue that occurs when PyVista's
    trame server isn't initialized or gets disconnected after code changes.
    
    The function:
    1. Detects if running in Jupyter
    2. Sets PyVista's backend to 'trame' (or fallback to 'panel'/'static')
    3. Ensures the backend is initialized before creating plotters
    
    Returns
    -------
    in_jupyter : bool
        True if running in Jupyter, False otherwise
    """
    try:
        import sys
        in_jupyter = 'ipykernel' in sys.modules or 'IPython' in sys.modules
    except:
        in_jupyter = False
    
    if in_jupyter:
        try:
            import pyvista as pv
            # If the user already set a backend (e.g. in notebook: pv.set_jupyter_backend('static')),
            # do not override it so figures are not forced to a backend that shows empty.
            current = getattr(pv.global_theme, 'jupyter_backend', None)
            if current and str(current).strip().lower() not in ('none', ''):
                pass  # keep current backend
            else:
                # Prefer 'client' for interactive (was working); fallback to static if needed.
                for backend in ('client', 'trame', 'panel', 'static'):
                    try:
                        pv.set_jupyter_backend(backend)
                        break
                    except Exception:
                        continue
        except Exception:
            # If PyVista isn't available, continue anyway
            pass
    
    return in_jupyter


class surface_mesh:
    """
    Surface mesh class for triangular/quad meshes
    """
    
    def __init__(self, X=None, F=None):
        """
        Initialize surface mesh
        
        Parameters:
        -----------
        X : array, optional
            Vertex coordinates (N x 3)
        F : array, optional
            Face connectivity (M x 3 or M x 4)
        """
        self.id = None  # Unique ID
        self.X = None
        self.F = None
        self.E = None
        self.L = None  # Vertex neighbors
        self.euler = None
        self.face_memb = None
        self.face_nbrs = None
        self.face_labels = None
        self.border_vertex = None
        self.isclosed_shape = True
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
        self.euler_violation = False
        
        # Geometric properties
        self.A = None
        self.V = None
        self.v = None
        self.F_areas = None
        self.h = None
        self.H = None
        self.Eb = None
        self.da = None
        self.dA = None
        self.quality = None
        self.M = None
        
        # Scalar fields
        self.sf = []
        
        # Configuration
        self.laplacian_smooth_iter = 10
        self.laplacian_smooth_beta = 0.001
        self.meshresample_keepratio = 0.7
        self.ixN = 0
        self.ixS = 0
        self.bijective_plot_flag = 1
        self.mapping_plot_flag = 2
        self.newton_niter = 200
        self.newton_step = 0.05
        self.newton_step_edge = 0.05  # Step factor for edge-length optimization (method 3)
        self.optimization_method = 1
        self.edge_n_fine_vertices = None  # Per-edge count of fine-mesh vertices (method 3); length n_edges
        self.edge_length_relative_error = False  # If True, method 4 minimizes relative error (L_e/T_e - 1)^2
        self.prevent_flip = True  # If True, use backtracking line search to avoid triangle flip (methods 2, 3, 4, 5)
        self.target_areas = None  # Per-face Ao for method 6 (multi-objective)
        self.multi_objective_opts = None  # Dict override for method 6
        self.t = None
        self.p = None
        self.objfun = None
        self.prep = None
        self.use_camorbit = True
        
        if X is not None and F is not None:
            X = np.asarray(X)
            F = np.asarray(F)
            
            if X.shape[0] == 3 and X.shape[1] != 3:
                X = X.T
            if F.shape[0] == 3 and F.shape[1] != 3:
                F = F.T
            
            self.X = X
            self.F = F
            self.border_vertex = np.zeros(X.shape[0])
    
    @staticmethod
    def sphere_mesh_gen(n):
        """
        Generate sphere mesh using icosahedron subdivision
        
        Parameters:
        -----------
        n : int
            Number of subdivisions
            
        Returns:
        --------
        X : array
            Vertex coordinates (N x 3)
        F : array
            Face connectivity (M x 3)
        """
        # Icosahedron vertices
        tau = 0.8506508084
        one = 0.5257311121
        
        p = np.array([
            [tau, one, 0],
            [-tau, one, 0],
            [-tau, -one, 0],
            [tau, -one, 0],
            [one, 0, tau],
            [one, 0, -tau],
            [-one, 0, -tau],
            [-one, 0, tau],
            [0, tau, one],
            [0, -tau, one],
            [0, -tau, -one],
            [0, tau, -one]
        ])
        
        # Icosahedron faces (1-indexed, will convert to 0-indexed)
        t = np.array([
            [5, 8, 9], [5, 10, 8], [6, 12, 7], [6, 7, 11],
            [1, 4, 5], [1, 6, 4], [3, 2, 8], [3, 7, 2],
            [9, 12, 1], [9, 2, 12], [10, 4, 11], [10, 11, 3],
            [9, 1, 5], [12, 6, 1], [5, 4, 10], [6, 11, 4],
            [8, 2, 9], [7, 12, 2], [8, 10, 3], [7, 3, 11]
        ]) - 1  # Convert to 0-indexed
        
        nt = 20
        np_count = 12
        
        # Subdivide
        for i in range(n):
            told = t.copy()
            t = np.zeros((nt * 4, 3), dtype=int)
            peMap = {}  # Point-edge map
            ct = 0
            
            # Use lists to allow modification in nested function
            np_count_list = [np_count]
            p_list = [p]
            
            for j in range(nt):
                p1, p2, p3 = told[j]
                
                # Get edge midpoints
                def get_midpoint(p1_idx, p2_idx):
                    key = tuple(sorted([p1_idx, p2_idx]))
                    if key in peMap:
                        return peMap[key]
                    else:
                        np_count_list[0] += 1
                        mid = (p_list[0][p1_idx] + p_list[0][p2_idx]) / 2
                        mid = mid / np.linalg.norm(mid)
                        if np_count_list[0] > len(p_list[0]):
                            p_list[0] = np.vstack([p_list[0], mid])
                        else:
                            p_list[0][np_count_list[0] - 1] = mid
                        peMap[key] = np_count_list[0] - 1
                        return np_count_list[0] - 1
                
                p4 = get_midpoint(p1, p2)
                p5 = get_midpoint(p2, p3)
                p6 = get_midpoint(p1, p3)
                
                # Create 4 new triangles
                t[ct] = [p1, p4, p6]
                ct += 1
                t[ct] = [p4, p5, p6]
                ct += 1
                t[ct] = [p4, p2, p5]
                ct += 1
                t[ct] = [p6, p5, p3]
                ct += 1
            
            nt = ct
            np_count = np_count_list[0]
            p = p_list[0][:np_count]
        
        return p, t
    
    def edge_info(self):
        """Compute edge information and vertex neighbors"""
        if self.needs_edge_info:
            nvert = len(self.X)
            nfaces = len(self.F)
            
            if self.F.shape[0] == 3:
                self.F = self.F.T
            if self.X.shape[0] == 3:
                self.X = self.X.T
            
            # Check if closed
            if nvert * 2 - 4 != nfaces:
                self.isclosed_shape = False
            
            # Single O(F) pass to build face membership, neighbors, and edges
            face_memb = {ix: [] for ix in range(nvert)}
            L_sets = {ix: set() for ix in range(nvert)}
            edge_set = set()

            F_int = self.F.astype(int)
            for f_idx in range(nfaces):
                v0, v1, v2 = int(F_int[f_idx, 0]), int(F_int[f_idx, 1]), int(F_int[f_idx, 2])
                face_memb[v0].append(f_idx)
                face_memb[v1].append(f_idx)
                face_memb[v2].append(f_idx)
                L_sets[v0].update((v1, v2))
                L_sets[v1].update((v0, v2))
                L_sets[v2].update((v0, v1))
                edge_set.add((min(v0, v1), max(v0, v1)))
                edge_set.add((min(v1, v2), max(v1, v2)))
                edge_set.add((min(v0, v2), max(v0, v2)))

            E = sorted(edge_set)
            self.E = np.array(E, dtype=int) if E else np.array([], dtype=int).reshape(0, 2)
            self.L = {k: list(v) for k, v in L_sets.items()}
            self.face_memb = face_memb
            
            # Face neighbors - compute using edge-to-face mapping (like MATLAB triangulation/neighbors)
            # This is critical for mesh segmentation to work correctly
            # Store as sparse matrix (like MATLAB) for consistency
            from .level1.get_border import compute_face_neighbors
            from scipy.sparse import lil_matrix, csr_matrix
            face_nbrs_dict = compute_face_neighbors(self)
            # Convert dict to sparse matrix (like MATLAB's triangulation/neighbors)
            self.face_nbrs = lil_matrix((nfaces, nfaces), dtype=bool)
            for i, nbrs in face_nbrs_dict.items():
                for nbr in nbrs:
                    self.face_nbrs[i, nbr] = True
            self.face_nbrs = self.face_nbrs.tocsr()
            
            # Check Euler characteristic
            if len(self.X) - len(self.E) + len(self.F) != 2:
                self.euler_violation = True
            else:
                self.euler_violation = False
            
            self.euler = len(np.unique(self.F.flatten())) + len(self.F) - len(self.E)
            self.needs_edge_info = False
    
    def info(self, verbose=True):
        """
        Get mesh information including topology.
        
        Parameters:
        -----------
        verbose : bool
            If True, print the information. Default: True
            
        Returns:
        --------
        info_dict : dict
            Dictionary containing mesh information:
            - n_vertices: number of vertices
            - n_faces: number of faces
            - n_edges: number of edges
            - euler_characteristic: V - E + F
            - genus: topological genus (number of handles)
            - is_closed: whether mesh is closed (watertight)
            - is_manifold: whether mesh is manifold
            - n_boundary_edges: number of boundary edges (0 for closed mesh)
            - n_components: number of connected components
        """
        if self.needs_edge_info:
            self.edge_info()
        
        n_vertices = len(self.X)
        n_faces = len(self.F)
        n_edges = len(self.E) if self.E is not None else 0
        
        # Euler characteristic: V - E + F
        # For closed orientable surface: chi = 2 - 2g where g is genus
        euler_char = n_vertices - n_edges + n_faces
        
        # Genus calculation: g = (2 - chi) / 2 for closed orientable surfaces
        # genus = 0 for sphere, 1 for torus, etc.
        genus = (2 - euler_char) // 2 if euler_char <= 2 else 0
        
        # Check for boundary edges (edges belonging to only one face)
        edge_face_count = {}
        for f_idx, face in enumerate(self.F):
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge_key = tuple(sorted([v1, v2]))
                edge_face_count[edge_key] = edge_face_count.get(edge_key, 0) + 1
        
        boundary_edges = [e for e, count in edge_face_count.items() if count == 1]
        n_boundary_edges = len(boundary_edges)
        is_closed = (n_boundary_edges == 0)
        
        # Check manifold: each edge should have exactly 1 or 2 adjacent faces
        non_manifold_edges = [e for e, count in edge_face_count.items() if count > 2]
        is_manifold = (len(non_manifold_edges) == 0)
        
        # Count connected components using union-find
        parent = list(range(n_vertices))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for face in self.F:
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            union(v0, v1)
            union(v1, v2)
        
        n_components = len(set(find(i) for i in range(n_vertices)))
        
        # For open surfaces, adjust genus calculation
        # chi = 2 - 2g - b where b is number of boundary loops
        if not is_closed:
            # Count boundary loops
            n_boundary_loops = self._count_boundary_loops(boundary_edges)
            # Adjusted genus: g = (2 - chi - b) / 2
            genus = max(0, (2 - euler_char - n_boundary_loops) // 2)
        else:
            n_boundary_loops = 0
        
        info_dict = {
            'n_vertices': n_vertices,
            'n_faces': n_faces,
            'n_edges': n_edges,
            'euler_characteristic': euler_char,
            'genus': genus,
            'is_closed': is_closed,
            'is_manifold': is_manifold,
            'n_boundary_edges': n_boundary_edges,
            'n_boundary_loops': n_boundary_loops,
            'n_components': n_components,
        }
        
        if verbose:
            print("=" * 50)
            print("Mesh Information")
            print("=" * 50)
            print(f"Vertices:            {n_vertices}")
            print(f"Faces:               {n_faces}")
            print(f"Edges:               {n_edges}")
            print("-" * 50)
            print(f"Euler characteristic: {euler_char}  (V - E + F)")
            print(f"Genus:               {genus}  (0=sphere, 1=torus, ...)")
            print("-" * 50)
            print(f"Is closed:           {is_closed}")
            print(f"Is manifold:         {is_manifold}")
            print(f"Boundary edges:      {n_boundary_edges}")
            print(f"Boundary loops:      {n_boundary_loops}")
            print(f"Connected components: {n_components}")
            print("=" * 50)
        
        return info_dict
    
    def disp(self, verbose=True):
        """
        Display surface mesh information (compact summary).
        
        Calls info() to show mesh topology and geometric properties
        (area, volume) if computed. Ensures props() is run so area/volume
        are shown when possible.
        
        Parameters:
        -----------
        verbose : bool
            If True, print the information. Default: True
            
        Returns:
        --------
        info_dict : dict
            Same as info(): mesh information dictionary.
        """
        if self.A is None or self.V is None:
            self.props()
        info_dict = self.info(verbose=verbose)
        if verbose and info_dict is not None:
            # Optionally append area/volume if already computed
            if self.A is not None:
                print(f"Area:                {self.A:.6g}")
            if self.V is not None:
                print(f"Volume:              {self.V:.6g}")
            if self.A is not None or self.V is not None:
                print("=" * 50)
        return info_dict
    
    def write_off(self, filename, renormalize=False):
        """
        Write mesh to a file in Geomview OFF format (same as MATLAB @surface_mesh/write_off).
        
        Parameters
        ----------
        filename : str
            Output path (e.g. 'mesh.off').
        renormalize : bool, optional
            If True, center vertices and scale by standard deviation before writing
            (default: False).
            
        Returns
        -------
        None
        """
        if self.X is None or self.F is None:
            raise ValueError("Mesh has no vertices or faces to write")
        vertex = np.asarray(self.X, dtype=np.float64)
        face = np.asarray(self.F, dtype=np.int64)
        if vertex.shape[0] == 3 and vertex.shape[1] != 3:
            vertex = vertex.T
        if vertex.shape[1] != 3:
            raise ValueError("Vertices must be N x 3")
        if face.shape[0] == 3 and face.shape[1] != 3:
            face = face.T
        if face.shape[1] != 3:
            raise ValueError("Faces must be M x 3 (triangles only for OFF)")
        if renormalize:
            m = np.mean(vertex, axis=0)
            s = np.std(vertex, axis=0)
            s = np.where(s > 1e-12, s, 1.0)
            vertex = (vertex - m) / s
        writeoff(filename, vertex, face)
    
    @staticmethod
    def fix_pyvista_backend():
        """
        Explicitly reinitialize PyVista's Jupyter backend to fix "Loading..." spinner issues.
        
        Call this method if interactive PyVista figures show "Loading..." instead of rendering.
        This can happen after code changes, kernel restarts, or when PyVista's trame server
        gets disconnected.
        
        Usage:
        ------
        >>> from pySHP.surface_mesh import surface_mesh
        >>> surface_mesh.fix_pyvista_backend()  # Fix backend before plotting
        >>> m.plot()  # Now plots should work
        
        Returns
        -------
        bool
            True if running in Jupyter and backend was set, False otherwise
        """
        return _ensure_pyvista_jupyter_backend()
    
    def _count_boundary_loops(self, boundary_edges):
        """Count the number of boundary loops from boundary edges"""
        if len(boundary_edges) == 0:
            return 0
        
        # Build adjacency for boundary vertices
        boundary_adj = {}
        for v1, v2 in boundary_edges:
            if v1 not in boundary_adj:
                boundary_adj[v1] = []
            if v2 not in boundary_adj:
                boundary_adj[v2] = []
            boundary_adj[v1].append(v2)
            boundary_adj[v2].append(v1)
        
        # Count loops by traversing
        visited = set()
        n_loops = 0
        
        for start in boundary_adj:
            if start in visited:
                continue
            
            # Traverse this loop
            current = start
            while current not in visited:
                visited.add(current)
                neighbors = boundary_adj.get(current, [])
                # Find unvisited neighbor
                next_v = None
                for n in neighbors:
                    if n not in visited:
                        next_v = n
                        break
                if next_v is None:
                    break
                current = next_v
            
            n_loops += 1
        
        return n_loops
    
    def check_mesh_integrity(self, verbose=True):
        """
        Check mesh for common problems.
        
        Returns:
        --------
        problems : list
            List of problem descriptions (empty if mesh is valid)
        """
        problems = []
        
        if self.needs_edge_info:
            self.edge_info()
        
        # Check for degenerate faces (duplicate vertices)
        for f_idx, face in enumerate(self.F):
            if len(np.unique(face)) != 3:
                problems.append(f"Degenerate face {f_idx}: {face}")
        
        # Check for unreferenced vertices
        used_vertices = np.unique(self.F.flatten())
        if len(used_vertices) != len(self.X):
            n_unused = len(self.X) - len(used_vertices)
            problems.append(f"{n_unused} unreferenced vertices")
        
        # Check for non-manifold edges
        edge_face_count = {}
        for f_idx, face in enumerate(self.F):
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge_key = tuple(sorted([v1, v2]))
                edge_face_count[edge_key] = edge_face_count.get(edge_key, 0) + 1
        
        non_manifold = [(e, c) for e, c in edge_face_count.items() if c > 2]
        if non_manifold:
            problems.append(f"{len(non_manifold)} non-manifold edges (shared by >2 faces)")
        
        # Check for holes (boundary edges)
        boundary = [e for e, c in edge_face_count.items() if c == 1]
        if boundary:
            problems.append(f"{len(boundary)} boundary edges (holes in mesh)")
        
        # Check for zero-area faces
        if self.F_areas is None:
            self.props()
        
        zero_area = np.sum(self.F_areas < 1e-10)
        if zero_area > 0:
            problems.append(f"{zero_area} zero-area faces")
        
        if verbose:
            if problems:
                print("Mesh integrity issues found:")
                for p in problems:
                    print(f"  - {p}")
            else:
                print("Mesh integrity check passed (no issues found)")
        
        return problems
    
    def mesh_check_repair(self, mode='duplicated', verbose=True):
        """
        Check and repair mesh issues (mimics MATLAB meshcheckrepair)
        
        Parameters:
        -----------
        mode : str
            Repair mode: 'duplicated', 'dup', 'isolated', 'deep', 'meshfix', 
                         'intersect', 'open', 'normals'
        verbose : bool
            Print diagnostic information
        
        Returns:
        --------
        self : surface_mesh
            Returns self for method chaining
        """
        try:
            import trimesh
            from trimesh import repair
            
            # Store initial counts for reporting
            initial_vertices = len(self.X)
            initial_faces = len(self.F)
            
            # Convert to trimesh format
            mesh_trimesh = trimesh.Trimesh(vertices=self.X, faces=self.F)
            
            # Helper function to remove duplicate faces manually
            def remove_duplicate_faces(mesh):
                """Remove duplicate faces from mesh, returns (cleaned_mesh, n_removed)"""
                initial_faces = len(mesh.faces)
                # Sort face vertices to normalize orientation
                faces_sorted = np.sort(mesh.faces, axis=1)
                # Find unique faces
                _, unique_idx = np.unique(faces_sorted, axis=0, return_index=True)
                n_removed = initial_faces - len(unique_idx)
                cleaned_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[unique_idx])
                return cleaned_mesh, n_removed
            
            if mode in ['duplicated', 'dup']:
                # Remove duplicate vertices and faces
                if verbose:
                    print("Removing duplicate vertices and faces...")
                # Count duplicate vertices before merging
                # (trimesh doesn't report this, so we estimate by checking if count changes)
                n_vertices_before = len(mesh_trimesh.vertices)
                # Merge vertices (consolidates duplicate vertices)
                mesh_trimesh.merge_vertices(merge_tex=True, merge_norm=True)
                n_vertices_after = len(mesh_trimesh.vertices)
                n_duplicate_vertices = n_vertices_before - n_vertices_after
                
                # Remove duplicate faces
                mesh_trimesh, n_duplicate_faces = remove_duplicate_faces(mesh_trimesh)
                mesh_trimesh.remove_unreferenced_vertices()
                
                if verbose:
                    if n_duplicate_vertices > 0:
                        print(f"  Removed {n_duplicate_vertices} duplicate vertices")
                    if n_duplicate_faces > 0:
                        print(f"  Removed {n_duplicate_faces} duplicate faces")
                    if n_duplicate_vertices == 0 and n_duplicate_faces == 0:
                        print("  No duplicate vertices or faces found")
                
            elif mode == 'isolated':
                # Remove isolated vertices (not referenced by any face)
                if verbose:
                    print("Removing isolated vertices...")
                n_vertices_before = len(mesh_trimesh.vertices)
                mesh_trimesh.remove_unreferenced_vertices()
                n_vertices_after = len(mesh_trimesh.vertices)
                n_isolated_vertices = n_vertices_before - n_vertices_after
                if verbose:
                    if n_isolated_vertices > 0:
                        print(f"  Removed {n_isolated_vertices} isolated vertices")
                    else:
                        print("  No isolated vertices found")
                
            elif mode == 'deep':
                # Deep cleaning: multiple repair operations
                if verbose:
                    print("Performing deep mesh cleaning...")
                # Remove duplicate vertices
                n_vertices_before = len(mesh_trimesh.vertices)
                mesh_trimesh.merge_vertices(merge_tex=True, merge_norm=True)
                n_duplicate_vertices = n_vertices_before - len(mesh_trimesh.vertices)
                
                # Use trimesh.repair module for comprehensive repair
                # Note: These functions may modify in place or return the mesh
                result = repair.fix_normals(mesh_trimesh)
                if result is not None:
                    mesh_trimesh = result
                result = repair.fix_winding(mesh_trimesh)
                if result is not None:
                    mesh_trimesh = result
                
                # Remove duplicate faces
                mesh_trimesh, n_duplicate_faces = remove_duplicate_faces(mesh_trimesh)
                
                # Remove degenerate faces (zero area) - trimesh doesn't have this method, so we implement it
                n_faces_before = len(mesh_trimesh.faces)
                # Compute face areas and filter out zero-area faces
                face_areas = mesh_trimesh.area_faces
                valid_faces = face_areas > 1e-10  # Keep faces with area > threshold
                if np.any(~valid_faces):
                    mesh_trimesh = trimesh.Trimesh(vertices=mesh_trimesh.vertices, 
                                                   faces=mesh_trimesh.faces[valid_faces])
                n_degenerate_faces = n_faces_before - len(mesh_trimesh.faces)
                
                # Remove unreferenced vertices
                n_vertices_before_iso = len(mesh_trimesh.vertices)
                mesh_trimesh.remove_unreferenced_vertices()
                n_isolated_vertices = n_vertices_before_iso - len(mesh_trimesh.vertices)
                
                # Fill holes if possible
                try:
                    mesh_trimesh.fill_holes()
                except:
                    pass  # May fail for complex meshes
                
                if verbose:
                    if n_duplicate_vertices > 0:
                        print(f"  Removed {n_duplicate_vertices} duplicate vertices")
                    if n_duplicate_faces > 0:
                        print(f"  Removed {n_duplicate_faces} duplicate faces")
                    if n_degenerate_faces > 0:
                        print(f"  Removed {n_degenerate_faces} degenerate faces")
                    if n_isolated_vertices > 0:
                        print(f"  Removed {n_isolated_vertices} isolated vertices")
                
            elif mode == 'meshfix':
                # General mesh fixing
                if verbose:
                    print("Performing general mesh fixing...")
                # Use trimesh.repair for comprehensive fixing
                # Note: These functions may modify in place or return the mesh
                result = repair.fix_normals(mesh_trimesh)
                if result is not None:
                    mesh_trimesh = result
                result = repair.fix_winding(mesh_trimesh)
                if result is not None:
                    mesh_trimesh = result
                # Merge vertices
                n_vertices_before = len(mesh_trimesh.vertices)
                mesh_trimesh.merge_vertices(merge_tex=True, merge_norm=True)
                n_duplicate_vertices = n_vertices_before - len(mesh_trimesh.vertices)
                # Remove duplicate faces
                mesh_trimesh, n_duplicate_faces = remove_duplicate_faces(mesh_trimesh)
                # Remove degenerate faces (zero area) - trimesh doesn't have this method, so we implement it
                n_faces_before = len(mesh_trimesh.faces)
                # Compute face areas and filter out zero-area faces
                face_areas = mesh_trimesh.area_faces
                valid_faces = face_areas > 1e-10  # Keep faces with area > threshold
                if np.any(~valid_faces):
                    mesh_trimesh = trimesh.Trimesh(vertices=mesh_trimesh.vertices, 
                                                   faces=mesh_trimesh.faces[valid_faces])
                n_degenerate_faces = n_faces_before - len(mesh_trimesh.faces)
                # Remove unreferenced vertices
                n_vertices_before_iso = len(mesh_trimesh.vertices)
                mesh_trimesh.remove_unreferenced_vertices()
                n_isolated_vertices = n_vertices_before_iso - len(mesh_trimesh.vertices)
                
                if verbose:
                    if n_duplicate_vertices > 0:
                        print(f"  Removed {n_duplicate_vertices} duplicate vertices")
                    if n_duplicate_faces > 0:
                        print(f"  Removed {n_duplicate_faces} duplicate faces")
                    if n_degenerate_faces > 0:
                        print(f"  Removed {n_degenerate_faces} degenerate faces")
                    if n_isolated_vertices > 0:
                        print(f"  Removed {n_isolated_vertices} isolated vertices")
                
            elif mode == 'intersect':
                # Fix self-intersections (this is complex, trimesh doesn't have direct support)
                # We'll do what we can: remove degenerate faces and fix normals
                if verbose:
                    print("Attempting to fix self-intersections...")
                result = repair.fix_normals(mesh_trimesh)
                if result is not None:
                    mesh_trimesh = result
                result = repair.fix_winding(mesh_trimesh)
                if result is not None:
                    mesh_trimesh = result
                # Remove degenerate faces (zero area)
                face_areas = mesh_trimesh.area_faces
                valid_faces = face_areas > 1e-10
                if np.any(~valid_faces):
                    mesh_trimesh = trimesh.Trimesh(vertices=mesh_trimesh.vertices, 
                                                   faces=mesh_trimesh.faces[valid_faces])
                
            elif mode == 'open':
                # Handle open meshes (meshes with boundaries)
                if verbose:
                    print("Processing open mesh (with boundaries)...")
                # Just ensure it's clean
                mesh_trimesh.remove_unreferenced_vertices()
                mesh_trimesh, n_duplicate_faces = remove_duplicate_faces(mesh_trimesh)
                # Remove degenerate faces (zero area)
                face_areas = mesh_trimesh.area_faces
                valid_faces = face_areas > 1e-10
                if np.any(~valid_faces):
                    mesh_trimesh = trimesh.Trimesh(vertices=mesh_trimesh.vertices, 
                                                   faces=mesh_trimesh.faces[valid_faces])
                
                if verbose and n_duplicate_faces > 0:
                    print(f"  Removed {n_duplicate_faces} duplicate faces")
            
            elif mode == 'normals':
                # Fix normal/winding consistency (ensure all normals point consistently)
                if verbose:
                    print("Fixing normal/winding consistency...")
                # Use trimesh repair functions (they may modify in place or return mesh)
                try:
                    result = repair.fix_normals(mesh_trimesh)
                    if result is not None:
                        mesh_trimesh = result
                except Exception as e:
                    if verbose:
                        print(f"  Warning: fix_normals failed: {e}")
                
                try:
                    result = repair.fix_winding(mesh_trimesh)
                    if result is not None:
                        mesh_trimesh = result
                except Exception as e:
                    if verbose:
                        print(f"  Warning: fix_winding failed: {e}")
                
                # Also use our custom fix_flipped_faces for closed meshes
                try:
                    from .level1.fix_flipped_faces import fix_flipped_faces
                    # Temporarily update self to use in fix_flipped_faces
                    self.X = np.array(mesh_trimesh.vertices)
                    self.F = np.array(mesh_trimesh.faces)
                    self, Fnc = fix_flipped_faces(self, verbose=False)
                    n_flipped = np.sum(Fnc == 0)
                    if verbose:
                        if n_flipped > 0:
                            print(f"  Fixed {n_flipped} flipped faces (normals now point consistently outward)")
                        else:
                            print("  All face normals are already consistent")
                    # Update mesh_trimesh with fixed faces
                    mesh_trimesh = trimesh.Trimesh(vertices=self.X, faces=self.F)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: fix_flipped_faces failed: {e}")
                
            else:
                if verbose:
                    print(f"Unknown repair mode '{mode}', performing basic cleaning...")
                mesh_trimesh.remove_unreferenced_vertices()
                mesh_trimesh, _ = remove_duplicate_faces(mesh_trimesh)
            
            # Update mesh
            self.X = np.array(mesh_trimesh.vertices)
            self.F = np.array(mesh_trimesh.faces)
            
            # Reset flags since mesh changed
            self.needs_updating = True
            self.needs_edge_info = True
            self.needs_map2sphere = True
            
            # Report final counts
            final_vertices = len(self.X)
            final_faces = len(self.F)
            vertices_removed = initial_vertices - final_vertices
            faces_removed = initial_faces - final_faces
            
            if verbose:
                print(f"Mesh repair complete: {final_vertices} vertices, {final_faces} faces")
                if vertices_removed > 0 or faces_removed > 0:
                    print(f"  Total removed: {vertices_removed} vertices, {faces_removed} faces")
                
        except ImportError:
            if verbose:
                print("Warning: trimesh not available, performing basic repair...")
            # Fallback: basic repair without trimesh
            self._basic_mesh_repair(mode, verbose)
        except Exception as e:
            if verbose:
                print(f"Error in mesh_check_repair: {e}")
                print("Attempting basic repair...")
            # Fallback: basic repair
            self._basic_mesh_repair(mode, verbose)
        
        return self
    
    def _basic_mesh_repair(self, mode, verbose):
        """Basic mesh repair without trimesh dependency"""
        if mode in ['duplicated', 'dup', 'isolated']:
            # Remove unreferenced vertices
            used_vertices = np.unique(self.F.flatten())
            if len(used_vertices) < len(self.X):
                # Remap face indices
                vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
                self.F = np.array([[vertex_map[int(v)] for v in face] for face in self.F])
                self.X = self.X[used_vertices]
                if verbose:
                    print(f"Removed {len(self.X) - len(used_vertices)} unreferenced vertices")
        
        # Remove degenerate faces (faces with duplicate vertices)
        valid_faces = []
        for face in self.F:
            if len(np.unique(face)) == 3:
                valid_faces.append(face)
        if len(valid_faces) < len(self.F):
            self.F = np.array(valid_faces)
            if verbose:
                print(f"Removed {len(self.F) - len(valid_faces)} degenerate faces")
        
        # Reset flags
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
    
    def find_disconnected_surfaces(self, verbose=True):
        """
        Find disconnected surface fragments (mimics MATLAB finddisconnsurf)
        
        Returns:
        --------
        face_groups : list
            List of face index arrays, each representing a connected component
        """
        if self.needs_edge_info:
            self.edge_info()
        
        # Build face adjacency graph
        nfaces = len(self.F)
        face_adj = {i: [] for i in range(nfaces)}
        
        # Build edge-to-face mapping
        edge_to_faces = {}
        for i, face in enumerate(self.F):
            edges = [
                tuple(sorted([int(face[0]), int(face[1])])),
                tuple(sorted([int(face[1]), int(face[2])])),
                tuple(sorted([int(face[2]), int(face[0])]))
            ]
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(i)
        
        # Build face adjacency
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                f1, f2 = faces[0], faces[1]
                face_adj[f1].append(f2)
                face_adj[f2].append(f1)
        
        # Find connected components using BFS
        visited = set()
        components = []
        
        for start_face in range(nfaces):
            if start_face in visited:
                continue
            
            # BFS to find all connected faces
            component = []
            queue = [start_face]
            visited.add(start_face)
            
            while queue:
                current = queue.pop(0)
                component.append(current)
                
                for neighbor in face_adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(np.array(component))
        
        if verbose:
            print(f"Found {len(components)} disconnected surface components")
            for i, comp in enumerate(components):
                print(f"  Component {i}: {len(comp)} faces")
        
        return components
    
    def keep_largest_surface(self, verbose=True):
        """
        Keep only the largest connected surface component (mimics MATLAB maxsurf)
        
        Returns:
        --------
        self : surface_mesh
            Returns self for method chaining
        """
        components = self.find_disconnected_surfaces(verbose=False)
        
        if len(components) <= 1:
            if verbose:
                print("Mesh is already a single connected component")
            return self
        
        # Find largest component
        largest_idx = np.argmax([len(comp) for comp in components])
        largest_faces = components[largest_idx]
        
        if verbose:
            print(f"Keeping largest component: {len(largest_faces)} faces (out of {len(self.F)} total)")
            print(f"Removing {len(components) - 1} smaller components")
        
        # Update faces
        self.F = self.F[largest_faces]
        
        # Remove unreferenced vertices
        used_vertices = np.unique(self.F.flatten())
        if len(used_vertices) < len(self.X):
            # Remap face indices
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
            self.F = np.array([[vertex_map[int(v)] for v in face] for face in self.F])
            self.X = self.X[used_vertices]
        
        # Reset flags
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
        
        return self
    
    def repair_mesh(self, verbose=True):
        """
        Apply full mesh repair sequence (mimics MATLAB meshcheckrepair workflow)
        This applies the standard repair sequence used in optimize_mesh
        
        Parameters:
        -----------
        verbose : bool
            Print diagnostic information
        
        Returns:
        --------
        self : surface_mesh
            Returns self for method chaining
        """
        if verbose:
            print("="*60)
            print("Starting mesh repair sequence...")
            print("="*60)
            print(f"Initial mesh: {len(self.X)} vertices, {len(self.F)} faces")
        
        # Standard repair sequence (matching MATLAB optimize_mesh)
        self.mesh_check_repair('duplicated', verbose=verbose)
        self.mesh_check_repair('isolated', verbose=verbose)
        self.mesh_check_repair('deep', verbose=verbose)
        
        # Check for disconnected surfaces
        components = self.find_disconnected_surfaces(verbose=False)
        n_components = len(components)
        if verbose:
            if n_components > 1:
                print(f"\nWARNING: Found {n_components} disconnected surface components!")
                for i, comp in enumerate(components):
                    print(f"  Component {i}: {len(comp)} faces")
            else:
                print(f"\nMesh connectivity: Single connected component ({len(components[0])} faces)")
        
        if n_components > 1:
            if verbose:
                print(f"Keeping largest component ({len(components[0])} faces)...")
            self.keep_largest_surface(verbose=False)  # Already reported above
            # Clean again after keeping largest
            self.mesh_check_repair('duplicated', verbose=verbose)
            try:
                self.mesh_check_repair('isolated', verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"Warning: Error in isolated vertex removal: {e}")
            try:
                self.mesh_check_repair('deep', verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"Warning: Error in deep cleaning: {e}")
        
        # Fix normal/winding consistency (ensure all normals point consistently)
        if verbose:
            print("\n" + "-"*60)
            print("Checking and fixing normal/winding consistency...")
            print("-"*60)
        self.mesh_check_repair('normals', verbose=verbose)
        
        # Final check
        if verbose:
            print("\n" + "="*60)
            print("Mesh repair complete!")
            print("="*60)
            print(f"Final mesh: {len(self.X)} vertices, {len(self.F)} faces")
            # Check Euler characteristic for closed meshes
            if len(self.X) * 2 - 4 == len(self.F):
                print("Mesh appears to be a closed shape (Euler characteristic = 2)")
            else:
                print(f"Mesh appears to be open (Euler characteristic != 2)")
                print(f"  Expected faces for closed mesh: {len(self.X) * 2 - 4}")
                print(f"  Actual faces: {len(self.F)}")
        
        return self
    
    def props(self):
        """Calculate mesh properties (area, volume, curvature)"""
        if self.F.shape[0] == 3:
            self.F = self.F.T
        
        X = self.X
        C = self.F
        
        # Calculate face areas and normals
        u = X[:, 0]
        v = X[:, 1]
        w = X[:, 2]
        
        x1 = u[C[:, 0]]
        y1 = v[C[:, 0]]
        z1 = w[C[:, 0]]
        x2 = u[C[:, 1]]
        y2 = v[C[:, 1]]
        z2 = w[C[:, 1]]
        x3 = u[C[:, 2]]
        y3 = v[C[:, 2]]
        z3 = w[C[:, 2]]
        
        q = np.column_stack([x2 - x1, y2 - y1, z2 - z1])
        r = np.column_stack([x3 - x1, y3 - y1, z3 - z1])
        
        crossqpr = kk_cross(q, r)
        twoA = np.linalg.norm(crossqpr, axis=1)
        A = np.sum(twoA) / 2
        F_areas = twoA / 2
        
        n = crossqpr / (twoA.reshape(-1, 1) + 1e-10)
        
        # Volume
        V = abs(np.sum(1/3 * kk_dot(n, np.column_stack([x1, y1, z1])) * twoA / 2))
        
        # Reduced volume
        Vo = 4/3 * np.pi * (A / 4 / np.pi)**(3/2)
        v = V / Vo
        
        # Triangle quality
        p_vec = np.column_stack([x3 - x2, y3 - y2, z3 - z2])
        d1 = np.linalg.norm(q, axis=1)
        d2 = np.linalg.norm(r, axis=1)
        d3 = np.linalg.norm(p_vec, axis=1)
        quality = 4 * F_areas * np.sqrt(3) / (d1**2 + d2**2 + d3**2)
        
        # Mean curvature calculation using dihedral angles
        # Based on MATLAB @surface_mesh/props.m
        # Remember: Gauss curvature is the angle defect at a vertex
        #           Mean curvature is edge length x dihedral angle at edge
        H = np.zeros(len(X))
        M = np.zeros(len(X))
        dA = np.zeros(len(X))
        
        for ix in range(len(X)):
            V_a = X[ix, :]
            
            # Find triangles containing this vertex
            # In MATLAB: [r c] = ind2sub(size(C),find(C==ix))
            # r contains row indices (face indices), c contains column indices (vertex position in face)
            face_mask = (C == ix)
            r = np.where(face_mask)[0]  # face indices containing vertex ix
            
            if len(r) == 0:
                dA[ix] = 0
                M[ix] = 0
                continue
            
            # Calculate dihedral angles of all unique pairs of triangles
            # In MATLAB: [I2 I1] = ind2sub([length(r) length(r)],find( tril(ones(length(r)),-1)~=0))
            # This creates all pairs (i,j) where i < j
            nr = len(r)
            if nr > 1:
                # Create lower triangular matrix (excluding diagonal)
                I1, I2 = np.tril_indices(nr, -1)
                # I2, I1 are pairs of indices into r
            else:
                I1 = np.array([], dtype=int)
                I2 = np.array([], dtype=int)
            
            H[ix] = 0
            for I in range(len(I2)):
                # Each permutation selects a triangle pair
                r1_idx = I2[I]  # index into r
                r2_idx = I1[I]  # index into r
                r1 = r[r1_idx]  # face index in C
                r2 = r[r2_idx]  # face index in C
                
                tr1 = C[r1, :]  # first triangle (vertex indices)
                tr2 = C[r2, :]  # second triangle (vertex indices)
                
                # Find the shared edge (the vertex that's not ix but is in both triangles)
                # In MATLAB: tr1r = tr1((tr1~=ix)); tr2r = tr2((tr2~=ix));
                tr1r = tr1[tr1 != ix]  # vertices in tr1 excluding ix
                tr2r = tr2[tr2 != ix]  # vertices in tr2 excluding ix
                
                # Find common vertex (the other vertex on the shared edge)
                # In MATLAB: rvrs = (length(tr1)-1):-1:1;
                #           indx = max(tr1r.*(tr1r==tr2r) + tr1r(rvrs).*(tr1r(rvrs)==tr2r));
                # This finds the vertex that appears in both tr1r and tr2r
                common_vertices = np.intersect1d(tr1r, tr2r)
                
                if len(common_vertices) > 0:
                    indx = common_vertices[0]  # shared edge vertex
                    V_e = X[indx, :]
                    
                    # Find the far vertex in tr2 (not ix, not indx)
                    # In MATLAB: V_far = X(tr2(tr2~=ix & tr2 ~=indx),:);
                    far_mask = (tr2 != ix) & (tr2 != indx)
                    if np.any(far_mask):
                        V_far = X[tr2[far_mask][0], :]
                        
                        # Calculate edge length
                        # In MATLAB: Lij = sqrt((V_a(1)-V_e(1))^2 + (V_a(2)-V_e(2))^2+(V_a(3)-V_e(3))^2);
                        Lij = np.linalg.norm(V_a - V_e)
                        
                        # Surface normals
                        n1 = n[r1, :]
                        n2 = n[r2, :]
                        
                        # Dihedral angle
                        # In MATLAB: theta(I) = acos(dot(n1,n2));
                        cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
                        theta = np.arccos(cos_theta)
                        
                        # Determine if convex or concave
                        # Complete the Hessian normal form of the planes
                        # In MATLAB: P1 = -(n1(1)*V_a(1) + n1(2)*V_a(2) + n1(3)*V_a(3));
                        P1 = -np.dot(n1, V_a)
                        P2 = -np.dot(n2, V_a)
                        
                        # Calculate whether the far point lies in the half-space of the normal direction
                        # In MATLAB: s = sign(dot(n1,V_far)+P1);
                        s = np.sign(np.dot(n1, V_far) + P1)
                        
                        # Accumulate curvature
                        # In MATLAB: H(ix) = H(ix) + Lij * real(theta(I))/4 * (s);
                        H[ix] = H[ix] + Lij * np.real(theta) / 4 * s
            
            # The average area of triangles around the vertex V_a
            # In MATLAB: dA(ix) = sum(F_areas(r))/3;
            dA[ix] = np.sum(F_areas[r]) / 3
            
            if dA[ix] == 0:
                M[ix] = 0
            else:
                # In MATLAB: H(ix) = H(ix)./dA(ix);
                H[ix] = H[ix] / dA[ix]
                # In MATLAB: M(ix) = H(ix).*dA(ix);
                M[ix] = H[ix] * dA[ix]
        
        # Total mean curvature
        # In MATLAB: h = sum(real(-M))./A;
        h = np.sum(np.real(-M)) / A if A > 0 else 0
        
        # Bending energy
        # In MATLAB: Eb = 2* sum(M.^2./dA)/8/pi;
        valid_dA = dA > 0
        Eb = 2 * np.sum(M[valid_dA]**2 / dA[valid_dA]) / (8 * np.pi) if np.any(valid_dA) else 0
        
        # Reduced area
        # In MATLAB: r = sqrt(A/4/pi); dAo = 8 * pi * r; da = 2 *sum(M)/dAo;
        r = np.sqrt(A / (4 * np.pi))
        dAo = 8 * np.pi * r
        da = 2 * np.sum(M) / dAo if dAo > 0 else 0
        
        self.A = A
        self.V = V
        self.v = v
        self.F_areas = F_areas
        self.quality = quality
        self.H = H
        self.h = h
        self.M = M
        self.dA = dA
        self.Eb = Eb
        self.da = da
        
        # Calculate median vertex curvature if L (vertex neighbors) exists
        if self.L is not None and len(self.L) > 0:
            H_median = H.copy()
            for vix in range(len(self.L)):
                l = self.L[vix]
                if l is not None and len(l) > 0:
                    h_vals = H[l]
                    h_valid = h_vals[(h_vals != 0) & ~np.isnan(h_vals)]
                    if len(h_valid) > 0:
                        mval = np.median(h_valid)
                        H_median[vix] = mval
                    else:
                        H_median[vix] = 0
                else:
                    H_median[vix] = 0
            self.H = H_median
        
        self.needs_updating = False
        
        return self
    
    def vertex_prop_to_edge_prop(self, v_prop):
        """
        Convert a vertex property to an edge property
        
        Parameters:
        -----------
        v_prop : array
            Vertex property (e.g., curvature H)
            
        Returns:
        --------
        edge_prop : array
            Edge property (average of vertex properties at edge endpoints)
        """
        assert len(v_prop) == len(self.X), 'property not of correct size'
        if self.needs_edge_info:
            self.edge_info()
        
        edge_prop = np.zeros(len(self.E))
        for eix in range(len(self.E)):
            v1 = self.E[eix, 0]
            v2 = self.E[eix, 1]
            edge_prop[eix] = (v_prop[v2] + v_prop[v1]) / 2  # average
        
        return edge_prop
    
    def get_graph(self):
        """
        Build graph from mesh and find north/south pole vertices
        
        Returns:
        --------
        G : networkx.Graph or None
            Graph representation (if networkx available)
        d : array
            Distance matrix
        ixN : int
            North pole vertex index
        ixS : int
            South pole vertex index
        weights : array
            Edge weights based on curvature
        ixN2 : int, optional
            Alternative north pole (unweighted graph)
        ixS2 : int, optional
            Alternative south pole (unweighted graph)
        g : networkx.Graph or None, optional
            Unweighted graph
        """
        if self.needs_edge_info:
            self.edge_info()
        
        # Calculate local curvature and assign edge weights
        self.props()
        
        # Convert vertex curvature to edge weights
        weights = self.vertex_prop_to_edge_prop(self.H)
        weights = weights - np.min(weights)
        weights = weights + np.median(weights)  # reasonable shift
        weights = 1.0 / (weights + 1e-10)  # avoid division by zero
        
        if HAS_NETWORKX:
            # Build weighted graph
            G = nx.Graph()
            for eix in range(len(self.E)):
                v1, v2 = self.E[eix, 0], self.E[eix, 1]
                G.add_edge(v1, v2, weight=weights[eix])
            
            # Calculate distances
            d = nx.floyd_warshall_numpy(G, weight='weight')
            d = np.array(d)
            
            # Find maximum distance pair (north and south poles)
            max_idx = np.unravel_index(np.argmax(d), d.shape)
            ixN = int(max_idx[0])
            ixS = int(max_idx[1])
            
            # Also compute unweighted graph distances
            g = nx.Graph()
            for eix in range(len(self.E)):
                v1, v2 = self.E[eix, 0], self.E[eix, 1]
                g.add_edge(v1, v2)
            
            d2 = nx.floyd_warshall_numpy(g)
            d2 = np.array(d2)
            max_idx2 = np.unravel_index(np.argmax(d2), d2.shape)
            ixN2 = int(max_idx2[0])
            ixS2 = int(max_idx2[1])
            
            return G, d, ixN, ixS, weights, ixN2, ixS2, g
        else:
            # Fallback: use scipy sparse matrices
            nvert = len(self.X)
            # Build sparse adjacency matrix with weights
            rows_orig = self.E[:, 0]
            cols_orig = self.E[:, 1]
            data = weights
            # Make symmetric
            rows = np.concatenate([rows_orig, cols_orig])
            cols = np.concatenate([cols_orig, rows_orig])
            data = np.concatenate([data, data])
            
            A = sparse.csr_matrix((data, (rows, cols)), shape=(nvert, nvert))
            
            # Use shortest path (Dijkstra) for weighted distances
            d = shortest_path(A, directed=False, method='D')
            d[np.isinf(d)] = 0  # Handle disconnected components
            
            # Find maximum distance pair
            max_idx = np.unravel_index(np.argmax(d), d.shape)
            ixN = int(max_idx[0])
            ixS = int(max_idx[1])
            
            # Unweighted graph
            A_unweighted = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nvert, nvert))
            d2 = shortest_path(A_unweighted, directed=False, method='D')
            d2[np.isinf(d2)] = 0
            max_idx2 = np.unravel_index(np.argmax(d2), d2.shape)
            ixN2 = int(max_idx2[0])
            ixS2 = int(max_idx2[1])
            
            return None, d, ixN, ixS, weights, ixN2, ixS2, None
    
    def optimize_mesh(self):
        """Optimize mesh using trimesh"""
        # Use trimesh for mesh operations
        mesh = trimesh.Trimesh(vertices=self.X, faces=self.F)
        
        # Simplify
        if self.meshresample_keepratio < 1.0:
            target_faces = int(len(mesh.faces) * self.meshresample_keepratio)
            mesh = mesh.simplify_quadric_decimation(target_faces)
        
        # Clean
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        self.X = mesh.vertices
        self.F = mesh.faces
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
        
        return self
    
    def map2sphere(self):
        """
        Map mesh to sphere parameterization using bijective mapping
        
        This computes spherical coordinates (t, p) for each vertex using
        heat diffusion equations (Brechbuehler 1995).
        """
        if self.needs_map2sphere:
            self.props()
            
            if self.F.shape[1] == 3:
                # Ensure edge info is computed
                if self.needs_edge_info:
                    self.edge_info()
                
                # Use bijective_map_gen for proper spherical parameterization
                if self.t is None or self.p is None:
                    t, p, dtline, W, A, b, ixN, ixS = surface_mesh.bijective_map_gen(
                        self.X, self.F, self.L, 
                        plotflag=self.bijective_plot_flag,
                        ixN=self.ixN if self.ixN > 0 else None,
                        ixS=self.ixS if self.ixS > 0 else None)
                    
                    self.t = t
                    self.p = p
                    self.ixN = ixN
                    self.ixS = ixS
                
                # Apply Newton steps for area-preserving optimization if requested
                if self.optimization_method == 1 and self.newton_niter > 0:
                    # Use bijective_plot_flag for visualization (2 = plot each iteration)
                    verbose_level = 3 if self.bijective_plot_flag else 0
                    # Use newton_steps_cart: Cartesian coordinates for Jacobian, spherical coordinates for area
                    self.t, self.p, newton_residuals = surface_mesh.newton_steps_cart(
                        self.t, self.p, self.F, 
                        self.newton_step, self.newton_niter,
                        flag=1, filename='surface_mesh', verbose=verbose_level)
                    # Store residuals for potential later use
                    self.newton_residuals = newton_residuals
                
                # Apply Newton steps with alternating area and shear correction if requested
                elif self.optimization_method == 2 and self.newton_niter > 0:
                    # Use bijective_plot_flag for visualization (2 = plot each iteration)
                    verbose_level = 3 if self.bijective_plot_flag else 0
                    # Use newton_shear_correction: alternates between area and shear minimization
                    self.t, self.p, newton_residuals = surface_mesh.newton_shear_correction(
                        self.t, self.p, self.F, 
                        self.newton_step, self.newton_niter,
                        flag=1, filename='surface_mesh', verbose=verbose_level,
                        prevent_flip=getattr(self, 'prevent_flip', getattr(self, 'shear_prevent_flip', True)))
                    # Store residuals for potential later use
                    self.newton_residuals = newton_residuals
                
                # Apply Newton steps with alternating area and edge-length correction (simplified mesh)
                elif self.optimization_method == 3 and self.newton_niter > 0:
                    if self.needs_edge_info:
                        self.edge_info()
                    verbose_level = 3 if self.bijective_plot_flag else 0
                    self.t, self.p, newton_residuals = surface_mesh.newton_edge_length_correction(
                        self.t, self.p, self.F, self.E,
                        self.edge_n_fine_vertices,
                        self.newton_step, self.newton_step_edge, self.newton_niter,
                        flag=1, filename='surface_mesh', verbose=verbose_level,
                        prevent_flip=getattr(self, 'prevent_flip', getattr(self, 'shear_prevent_flip', True)))
                    self.newton_residuals = newton_residuals
                
                # Apply Newton steps for edge-length minimization only (simplified mesh)
                elif self.optimization_method == 4 and self.newton_niter > 0:
                    if self.needs_edge_info:
                        self.edge_info()
                    # Always use verbose >= 1 for edge-length optimization to show progress
                    verbose_level = 3 if self.bijective_plot_flag >= 2 else (1 if self.bijective_plot_flag >= 1 else 1)
                    self.t, self.p, newton_residuals = surface_mesh.newton_edge_length_only(
                        self.t, self.p, self.F, self.E,
                        self.edge_n_fine_vertices,
                        self.newton_step_edge, self.newton_niter,
                        flag=1, filename='surface_mesh', verbose=verbose_level,
                        use_relative_error=getattr(self, 'edge_length_relative_error', False),
                        prevent_flip=getattr(self, 'prevent_flip', getattr(self, 'shear_prevent_flip', True)))
                    self.newton_residuals = newton_residuals
                # Apply Newton steps for pure shear correction only (no area alternation)
                elif self.optimization_method == 5 and self.newton_niter > 0:
                    verbose_level = 3 if self.bijective_plot_flag else 0
                    self.t, self.p, newton_residuals = surface_mesh.newton_shear_correction(
                        self.t, self.p, self.F,
                        self.newton_step, self.newton_niter,
                        flag=1, filename='surface_mesh', verbose=verbose_level, shear_only=True,
                        prevent_flip=getattr(self, 'prevent_flip', getattr(self, 'shear_prevent_flip', True)))
                    self.newton_residuals = newton_residuals
                elif self.optimization_method == 6 and self.newton_niter > 0:
                    self.t, self.p, newton_residuals, mo_report = self._run_multi_objective_optimization()
                    self.newton_residuals = newton_residuals
                    self.multi_objective_report = mo_report
                
                self.needs_map2sphere = False
            elif self.F.shape[1] == 4:
                # Quad mesh - use direct mapping for now
                if self.needs_edge_info:
                    self.edge_info()
                if self.t is None or self.p is None:
                    t, p, r = kk_cart2sph(self.X[:, 0], self.X[:, 1], self.X[:, 2])
                    self.t = t
                    self.p = p
                self.needs_map2sphere = False
            else:
                # No optimization
                if self.t is None or self.p is None:
                    t, p, r = kk_cart2sph(self.X[:, 0], self.X[:, 1], self.X[:, 2])
                    self.t = t
                    self.p = p
                self.needs_map2sphere = False
        # Re-run optimization when t,p already exist (e.g. switch from method 1 to method 4 or 5)
        elif (self.F.shape[1] == 3 and self.t is not None and self.p is not None and
              self.optimization_method in (2, 3, 4, 5, 6) and self.newton_niter > 0):
            if self.needs_edge_info:
                self.edge_info()
            if self.optimization_method == 2:
                verbose_level = 3 if self.bijective_plot_flag else 0
                self.t, self.p, newton_residuals = surface_mesh.newton_shear_correction(
                    self.t, self.p, self.F,
                    self.newton_step, self.newton_niter,
                    flag=1, filename='surface_mesh', verbose=verbose_level,
                    prevent_flip=getattr(self, 'prevent_flip', getattr(self, 'shear_prevent_flip', True)))
                self.newton_residuals = newton_residuals
            elif self.optimization_method == 3:
                verbose_level = 3 if self.bijective_plot_flag else 0
                self.t, self.p, newton_residuals = surface_mesh.newton_edge_length_correction(
                    self.t, self.p, self.F, self.E,
                    self.edge_n_fine_vertices,
                    self.newton_step, self.newton_step_edge, self.newton_niter,
                    flag=1, filename='surface_mesh', verbose=verbose_level,
                    prevent_flip=getattr(self, 'prevent_flip', getattr(self, 'shear_prevent_flip', True)))
                self.newton_residuals = newton_residuals
            elif self.optimization_method == 4:
                verbose_level = 3 if self.bijective_plot_flag >= 2 else (1 if self.bijective_plot_flag >= 1 else 1)
                self.t, self.p, newton_residuals = surface_mesh.newton_edge_length_only(
                    self.t, self.p, self.F, self.E,
                    self.edge_n_fine_vertices,
                    self.newton_step_edge, self.newton_niter,
                    flag=1, filename='surface_mesh', verbose=verbose_level,
                    use_relative_error=getattr(self, 'edge_length_relative_error', False),
                    prevent_flip=getattr(self, 'prevent_flip', getattr(self, 'shear_prevent_flip', True)))
                self.newton_residuals = newton_residuals
            elif self.optimization_method == 5:
                verbose_level = 3 if self.bijective_plot_flag else 0
                self.t, self.p, newton_residuals = surface_mesh.newton_shear_correction(
                    self.t, self.p, self.F,
                    self.newton_step, self.newton_niter,
                    flag=1, filename='surface_mesh', verbose=verbose_level, shear_only=True,
                    prevent_flip=getattr(self, 'prevent_flip', getattr(self, 'shear_prevent_flip', True)))
                self.newton_residuals = newton_residuals
            elif self.optimization_method == 6:
                self.t, self.p, newton_residuals, mo_report = self._run_multi_objective_optimization()
                self.newton_residuals = newton_residuals
                self.multi_objective_report = mo_report

        return self

    def _run_multi_objective_optimization(self):
        """Run curvature-aware multi-objective Newton (optimization_method 6)."""
        from .level1.target_areas import compute_curvature_target_areas
        from .level1.newton_multi_objective import (
            newton_multi_objective, default_multi_objective_opts)

        if self.target_areas is None:
            Ao, _ = compute_curvature_target_areas(self)
        else:
            Ao = np.asarray(self.target_areas, dtype=float).reshape(-1)

        opts = default_multi_objective_opts(
            maxiter=self.newton_niter,
            stepfac=self.newton_step,
            prevent_flip=getattr(self, 'prevent_flip', True),
            verbose=1 if self.bijective_plot_flag else 0,
        )
        if self.multi_objective_opts:
            opts.update(self.multi_objective_opts)
        opts['maxiter'] = self.newton_niter
        opts['stepfac'] = self.newton_step

        fixed = []
        if getattr(self, 'ixN', 0) and self.ixN > 0:
            fixed.append(int(self.ixN))
        if getattr(self, 'ixS', 0) and self.ixS > 0:
            fixed.append(int(self.ixS))
        if fixed:
            opts['fixed'] = np.array(fixed, dtype=int)

        t, p, residuals, report = newton_multi_objective(
            self.t, self.p, self.F, Ao, opts, filename='surface_mesh')
        self.target_areas = Ao
        return t, p, residuals, report
    
    def translate_to_center_of_mass(self):
        """
        Translate mesh to center of mass
        """
        if self.X is not None:
            self.X[:, 0] = self.X[:, 0] - np.mean(self.X[:, 0])
            self.X[:, 1] = self.X[:, 1] - np.mean(self.X[:, 1])
            self.X[:, 2] = self.X[:, 2] - np.mean(self.X[:, 2])
        return self
    
    def subdivide(self, nsub=1):
        """
        Subdivide mesh
        
        Parameters:
        -----------
        nsub : int
            Number of subdivisions
        """
        if nsub < 1:
            return self
        
        X = self.X
        F = self.F
        
        for _ in range(nsub):
            X, F = self.subdivide_XF(X, F, 1)
        
        self.X = X
        self.F = F
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
        
        return self
    
    @staticmethod
    def latitude_calc(L, ixN, ixS):
        """
        Calculate latitude from diffusion as in Brechbuehler 1995
        
        Parameters:
        -----------
        L : list of arrays
            Cell array of link arrays (vertex neighbors)
        ixN : int
            North pole vertex index
        ixS : int
            South pole vertex index
            
        Returns:
        --------
        t : array
            Theta (latitude) values for each vertex
        A : sparse matrix
            Matrix A as in Brechbuehler 1995
        b : sparse array
            Vector b
        """
        nvert = len(L)
        # Set up matrix A (as in Brechbuehler's paper appendix)
        A = sparse.lil_matrix((nvert, nvert))
        b = sparse.lil_matrix((nvert, 1))
        
        for iv in range(nvert):
            links = L[iv]  # array of indices of vertices linked to iv
            A[iv, iv] = len(links)
            for k in range(len(links)):
                A[iv, links[k]] = -1
        
        # Set boundary conditions
        A[ixN, :] = 0
        A[ixN, ixN] = 1
        A[ixS, :] = 0
        A[ixS, ixS] = 1
        
        Ap = A.tocsr()
        b = sparse.lil_matrix((nvert, 1))
        b[ixS, 0] = np.pi
        
        # Solve linear system
        t = sparse.linalg.spsolve(Ap, b.toarray().flatten())
        
        return t, Ap, b
    
    @staticmethod
    def longitude_calc(x, y, z, t, A, F, L, ixN, ixS):
        """
        Calculate longitude from diffusion as in Brechbuehler 1995
        
        Parameters:
        -----------
        x, y, z : array
            Vertex coordinates
        t : array
            Latitude (theta) values from latitude_calc
        A : sparse matrix
            Matrix A from latitude_calc
        F : array
            Face connectivity
        L : list of arrays
            Vertex neighbors
        ixN, ixS : int
            North and south pole indices
            
        Returns:
        --------
        p : array
            Phi (longitude) values for each vertex (mod 2π)
        A : sparse matrix
            Modified matrix A
        b : sparse array
            Vector b
        dtline : array
            Date line (path from N to S)
        W : array
            West vertices
        p_unwrapped : array
            Phi values before modulo (for visualization to show 2π jump)
        """
        # Modify matrix A: cut links with poles
        # Convert to LIL format for efficient modification (CSR modification is inefficient)
        A_lil = A.tolil()
        
        links = np.asarray(L[ixN])
        for ix in range(len(links)):
            link_idx = int(links[ix])
            A_lil[link_idx, link_idx] = A_lil[link_idx, link_idx] - 1
        
        links = np.asarray(L[ixS])
        for ix in range(len(links)):
            link_idx = int(links[ix])
            A_lil[link_idx, link_idx] = A_lil[link_idx, link_idx] - 1
        
        # Convert back to CSR for later use (but we'll convert again before solving)
        A = A_lil.tocsr()
        
        # Determine date line based on steepest ascent in theta
        dtline = []
        b = sparse.lil_matrix((len(L), 1))
        previous = ixN
        
        # Find path from north to south pole following steepest theta ascent
        nbrs = np.asarray(L[ixN])
        if len(nbrs) > 0:
            here = int(nbrs[-1])  # any neighbor of the north pole (last element, matching MATLAB nbrs(end))
            counter = 0
            maximum = 0  # Cumulative maximum - NOT reset inside loop (matching MATLAB)
            
            while here != ixS:
                counter += 1
                if counter > len(L):
                    raise ValueError('could not determine p')
                
                dtline.append(here)
                nbrs = np.asarray(L[here])  # get the direct neighbors of here
                
                nextpos = None
                # Find neighbor with highest theta that is greater than current maximum
                # This ensures we always move toward higher theta (toward south pole)
                for ix in range(len(nbrs)):
                    if t[nbrs[ix]] > maximum:
                        maximum = t[nbrs[ix]]
                        nextpos = ix
                
                previous = here
                if nextpos is not None:
                    here = int(nbrs[nextpos])
                else:
                    # If no neighbor has theta > maximum, we can't proceed
                    break
            
            # If date line is empty, try alternative path
            if len(dtline) == 0:
                nbrs = np.asarray(L[ixN])
                if len(nbrs) > 0:
                    here = int(nbrs[0])  # any neighbor of the north pole (first element)
                    counter = 0
                    maximum = 0  # Reset for alternative path
                    
                    while here != ixS:
                        counter += 1
                        if counter > len(L):
                            raise ValueError('could not determine p')
                        
                        dtline.append(here)
                        nbrs = np.asarray(L[here])  # get the direct neighbors of here
                        
                        nextpos = None
                        # Find neighbor with highest theta that is greater than current maximum
                        for ix in range(len(nbrs)):
                            if t[nbrs[ix]] > maximum:
                                maximum = t[nbrs[ix]]
                                nextpos = ix
                        
                        previous = here
                        if nextpos is not None:
                            here = int(nbrs[nextpos])
                        else:
                            break
        
        dtline = np.array(dtline)
        
        # Determine western links
        S = np.array([x[ixS], y[ixS], z[ixS]])
        N = np.array([x[ixN], y[ixN], z[ixN]])
        E = []
        W = []
        DL = np.zeros((len(dtline), 3), dtype=int)
        
        if len(dtline) > 0:
            here = dtline[0]
            Wprev = None  # Initialize Wprev outside loop (persists across iterations like MATLAB)
            for ix in range(len(dtline)):
                Wlinks = []
                W_count = 0
                
                if ix == 0:
                    prev = ixN
                else:
                    prev = int(dtline[ix - 1])  # Ensure integer indexing
                
                if ix == len(dtline) - 1:
                    next_vert = ixS
                else:
                    next_vert = int(dtline[ix + 1])  # Ensure integer indexing
                
                here_links = np.asarray(L[here])
                here_links = here_links[here_links != prev]
                here_links = here_links[here_links != next_vert]
                
                prev_links = np.asarray(L[prev])
                prev_links = prev_links[prev_links != here]
                
                next_links = np.asarray(L[next_vert])
                next_links = next_links[next_links != here]
                
                # Match MATLAB: Wprev is set only when ix == 1 (first iteration)
                # For subsequent iterations, Wprev persists from previous iteration's while loop
                if ix == 0:
                    common_links = np.intersect1d(here_links, prev_links)
                    if len(common_links) > 0:
                        Wprev = int(common_links[0])  # Match MATLAB: common_links(1)
                    else:
                        # Fallback if no common links (shouldn't happen in practice)
                        Wprev = int(here_links[0]) if len(here_links) > 0 else None
                # For ix > 0, Wprev should already be set from previous iteration
                # If not, use last element from previous Wlinks (though this shouldn't happen)
                if Wprev is None and ix > 0:
                    # This is a fallback - in practice Wprev should persist from previous iteration
                    Wprev = int(Wlinks[-1]) if len(Wlinks) > 0 else None
                
                if Wprev is not None:
                    Wlinks.append(Wprev)
                    W_count += 1
                    
                    # Match MATLAB: L{Wprev(1)} - Wprev is scalar here, but MATLAB uses (1) notation
                    Wprev_links = np.asarray(L[Wprev])
                    Wprev_links = Wprev_links[Wprev_links != here]
                    Wprev_links = Wprev_links[Wprev_links != prev]
                    
                    # Continue finding west links - match MATLAB while loop logic
                    while len(np.intersect1d(here_links, Wprev_links)) > 0:
                        # Match MATLAB: here_links(here_links ~=Wprev(1))
                        here_links = here_links[here_links != Wprev]
                        # Match MATLAB: Wprev = intersect(here_links,Wprev_links)
                        Wprev_intersect = np.intersect1d(here_links, Wprev_links)
                        if len(Wprev_intersect) > 0:
                            # Match MATLAB: Wprev(1) - take first element
                            Wprev = int(Wprev_intersect[0])
                            Wlinks.append(Wprev)
                            W_count += 1
                            
                            # Match MATLAB: L{Wprev(1)}
                            Wprev_links = np.asarray(L[Wprev])
                            Wprev_links = Wprev_links[Wprev_links != here]
                            Wprev_links = Wprev_links[Wprev_links != prev]
                            Wprev_links = Wprev_links[Wprev_links != next_vert]
                        else:
                            break
                
                W.extend(Wlinks)
                DL[ix, :] = [here, W_count, 0]
                prev = here
                here = int(next_vert)  # Ensure integer for next iteration
        
        # Modify b accordingly - match MATLAB exactly (lines 164-170)
        # NOTE: MATLAB does NOT modify matrix A for dtline vertices
        # The diffusion equation remains active for dtline vertices
        # The 2π jump is created solely through b vector modifications
        # The b vector modifications create the 2π jump across the date line
        
        # For each dtline vertex: b(dtline) = -W_count * 2π
        # This sets the fixed phi value at dtline vertices
        # The negative sign accounts for the fact that west vertices get +2π
        for ix in range(len(dtline)):
            dlarr = DL[ix, :]
            # dlarr[0] = here (vertex index), dlarr[1] = W_count, dlarr[2] = 0 (E_count, unused)
            # Set b for dtline vertex: -W_count * 2*pi
            b[dtline[ix], 0] = -dlarr[1] * 2 * np.pi
        
        # For each west vertex: b(W) += 2π
        # This adds 2π to all vertices on the "west" side of the date line
        # This creates the 2π offset for vertices crossing from east to west
        # When the diffusion equation solves, west vertices will have phi values that are 2π higher
        for ix in range(len(W)):
            b[W[ix], 0] = b[W[ix], 0] + 2 * np.pi
        
        # Solve linear system - match MATLAB: p = full(A\b)
        A_csr = A.tocsr()
        b_array = b.toarray().flatten()
        p = sparse.linalg.spsolve(A_csr, b_array)
        # Store p before modulo for visualization (to preserve 2π jump)
        p_unwrapped = p.copy()
        # Apply modulo 2π (MATLAB: p = mod(p, 2*pi))
        # This is for the return value, but we'll use unwrapped for visualization
        p = np.mod(p, 2 * np.pi)
        
        return p, A_csr, b, dtline, np.array(W), p_unwrapped
    
    @staticmethod
    def bijective_map_gen(X, F, L, plotflag=0, ixN=None, ixS=None):
        """
        Calculate bijective mapping of mesh to sphere
        
        Parameters:
        -----------
        X : array
            Vertex coordinates (N x 3)
        F : array
            Face connectivity
        L : list of arrays
            Vertex neighbors
        plotflag : int, optional
            Whether to plot intermediate results
        ixN, ixS : int, optional
            North and south pole indices (auto-detected if None)
            
        Returns:
        --------
        t : array
            Theta (latitude) values
        p : array
            Phi (longitude) values
        dtline : array
            Date line
        W : array
            West vertices
        A : sparse matrix
            Matrix A
        b : sparse array
            Vector b
        ixN, ixS : int
            North and south pole indices
        """
        if X.shape[0] == 3 and X.shape[1] != 3:
            X = X.T
        
        dtline = []
        W = []
        A = None
        b = None
        
        # Get north and south pole if not provided
        if ixN is None or ixS is None:
            m = surface_mesh(X, F)
            _, _, ixN, ixS, _, _, _, _ = m.get_graph()
        
        if plotflag:
            print('Calculating theta mapping')
        
        # Calculate theta (latitude)
        t, A, b = surface_mesh.latitude_calc(L, ixN, ixS)
        
        if plotflag:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # Create persistent figure for theta
                fig_theta = plt.figure('THETA (latitude)', figsize=(8, 6))
                ax = fig_theta.add_subplot(111, projection='3d')
                
                # Create vertex colors
                t_norm = t / np.max(t) if np.max(t) > 0 else t
                colors = plt.cm.viridis(t_norm)
                
                # Create face colors by averaging vertex colors
                face_colors = np.mean(colors[F], axis=1)
                
                # Create poly collection
                verts = X[F]
                collection = Poly3DCollection(verts, facecolors=face_colors, 
                                             edgecolors='k', linewidths=0.5)
                ax.add_collection3d(collection)
                
                # Set limits and aspect
                ax.set_xlim(X[:, 0].min(), X[:, 0].max())
                ax.set_ylim(X[:, 1].min(), X[:, 1].max())
                ax.set_zlim(X[:, 2].min(), X[:, 2].max())
                ax.set_aspect('equal')
                ax.set_title('THETA (latitude)')
                plt.show(block=False)
                plt.draw()
                plt.pause(0.1)  # Small pause to ensure figure is displayed
            except ImportError:
                pass
        
        if plotflag:
            print('Calculating phi mapping')
        
        # Calculate phi (longitude)
        p, A, b, dtline, W, p_unwrapped = surface_mesh.longitude_calc(
            X[:, 0], X[:, 1], X[:, 2], t, A, F, L, ixN, ixS)
        
        # Visualize date line (dtline) to verify it connects poles correctly
        if plotflag:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
                
                # Create figure for date line visualization
                fig_dtline = plt.figure('Date Line (dtline) - North to South Pole Path', figsize=(10, 8))
                ax = fig_dtline.add_subplot(111, projection='3d')
                
                # Plot the mesh without transparency
                verts = X[F]
                collection = Poly3DCollection(verts, facecolors='lightgray', 
                                             edgecolors='gray', linewidths=0.3, alpha=1.0)
                ax.add_collection3d(collection)
                
                # Highlight the date line path
                if len(dtline) > 0:
                    # Create path from north pole through dtline to south pole
                    dtline_path = np.vstack([
                        X[ixN:ixN+1, :],  # Start at north pole
                        X[dtline, :],     # Date line vertices
                        X[ixS:ixS+1, :]   # End at south pole
                    ])
                    
                    # Plot date line as a thick red line
                    ax.plot(dtline_path[:, 0], dtline_path[:, 1], dtline_path[:, 2], 
                           'r-', linewidth=3, label='Date Line (dtline)')
                    
                    # Mark date line vertices with red dots
                    ax.scatter(X[dtline, 0], X[dtline, 1], X[dtline, 2], 
                              c='red', s=50, marker='o', label='Date Line Vertices', zorder=5)
                
                # Mark north pole with blue sphere
                ax.scatter(X[ixN:ixN+1, 0], X[ixN:ixN+1, 1], X[ixN:ixN+1, 2], 
                          c='blue', s=200, marker='*', label='North Pole (ixN)', zorder=6)
                
                # Mark south pole with green sphere
                ax.scatter(X[ixS:ixS+1, 0], X[ixS:ixS+1, 1], X[ixS:ixS+1, 2], 
                          c='green', s=200, marker='*', label='South Pole (ixS)', zorder=6)
                
                # Set limits and aspect
                ax.set_xlim(X[:, 0].min(), X[:, 0].max())
                ax.set_ylim(X[:, 1].min(), X[:, 1].max())
                ax.set_zlim(X[:, 2].min(), X[:, 2].max())
                ax.set_aspect('equal')
                ax.set_title(f'Date Line Path: {len(dtline)} vertices connecting poles\n'
                           f'North Pole (idx {ixN}) -> South Pole (idx {ixS})')
                ax.legend()
                ax.view_init(elev=20, azim=45)
                
                plt.show(block=False)
                plt.draw()
                plt.pause(0.1)
                
                # Print diagnostic information
                print(f'\nDate Line (dtline) Information:')
                print(f'  North Pole index: {ixN}, coordinates: {X[ixN, :]}')
                print(f'  South Pole index: {ixS}, coordinates: {X[ixS, :]}')
                print(f'  Date line length: {len(dtline)} vertices')
                if len(dtline) > 0:
                    print(f'  First dtline vertex: {dtline[0]}, coordinates: {X[dtline[0], :]}')
                    print(f'  Last dtline vertex: {dtline[-1]}, coordinates: {X[dtline[-1], :]}')
                    print(f'  Theta values along dtline: min={t[dtline].min():.4f}, max={t[dtline].max():.4f}')
                    print(f'  Theta at North Pole: {t[ixN]:.4f}, Theta at South Pole: {t[ixS]:.4f}')
                else:
                    print('  WARNING: Date line is empty!')
            except ImportError:
                pass
        
        if plotflag:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                # Create persistent figure for phi (longitude)
                fig_phi = plt.figure('PHI (longitude)', figsize=(8, 6))
                ax = fig_phi.add_subplot(111, projection='3d')
                
                # Create vertex colors
                p_norm = p / (2 * np.pi) if np.max(p) > 0 else p
                colors = plt.cm.viridis(p_norm)
                
                # Create face colors by averaging vertex colors
                face_colors = np.mean(colors[F], axis=1)
                
                # Create poly collection
                verts = X[F]
                collection = Poly3DCollection(verts, facecolors=face_colors, 
                                             edgecolors='k', linewidths=0.5)
                ax.add_collection3d(collection)
                
                # Set limits and aspect
                ax.set_xlim(X[:, 0].min(), X[:, 0].max())
                ax.set_ylim(X[:, 1].min(), X[:, 1].max())
                ax.set_zlim(X[:, 2].min(), X[:, 2].max())
                ax.set_aspect('equal')
                ax.set_title('PHI (longitude)')
                plt.show(block=False)
                plt.draw()
                plt.pause(0.1)  # Small pause to ensure figure is displayed
                
                # Plot spherical parameterization state (persistent)
                # Use unwrapped phi values for visualization to show continuity across date line
                # The unwrapped values preserve the 2π jump, which creates proper continuity
                # when converted to cartesian coordinates (matching MATLAB behavior)
                u, v, w = kk_sph2cart(t, p_unwrapped, np.ones(len(p_unwrapped)))
                surface_mesh.plot_state(u, v, w, F, count='State of bijective mapping')
                
                # Create persistent histogram figure
                fig_hist = plt.figure('Theta and Phi Histograms', figsize=(12, 4))
                ax1, ax2 = fig_hist.subplots(1, 2)
                ax1.hist(t, bins=100, edgecolor='black')
                ax1.set_title('theta histogram')
                ax1.set_xlim([0, np.pi])
                ax1.set_xlabel('theta')
                ax1.set_ylabel('Frequency')
                
                ax2.hist(p, bins=100, edgecolor='black')
                ax2.set_title('phi histogram')
                ax2.set_xlim([0, 2 * np.pi])
                ax2.set_xlabel('phi')
                ax2.set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.show(block=False)
                plt.draw()
                plt.pause(0.1)  # Small pause to ensure figure is displayed
            except ImportError:
                pass
        
        return t, p, dtline, W, A, b, ixN, ixS
    
    @staticmethod
    def newton_steps(t, p, F, stepfac, maxiter, flag=0, filename='untitled', verbose=0, solver='gmres'):
        """
        Minimize triangle areas on sphere using Newton's method
        
        Parameters:
        -----------
        t : array
            Theta (latitude) values
        p : array
            Phi (longitude) values
        F : array
            Face connectivity
        stepfac : float
            Step factor for Newton iteration
        maxiter : int
            Maximum iterations
        flag : int, optional
            Whether to prepare Jacobian (1) or load from file (0)
        filename : str, optional
            Filename for saving/loading Jacobian pattern
        verbose : int, optional
            Verbosity level (0=none, 1=text, 2=plot)
        solver : str, optional
            Linear solver to use: 'gmres' (default, matches Matlab) or 'lu'
            
        Returns:
        --------
        t : array
            Updated theta values
        p : array
            Updated phi values
        residuals : array
            Residual norms for each iteration
        """
        nvert = len(t)
        nfaces = len(F)
        p = np.mod(p, 2 * np.pi)
        X = np.concatenate([t, p])
        
        u = np.zeros(nvert, dtype=np.float64)
        v = np.zeros(nvert, dtype=np.float64)
        w = np.zeros(nvert, dtype=np.float64)
        crossqpr = np.zeros((nfaces, 3), dtype=np.float64)
        
        # Prepare Jacobian sparsity pattern
        import os
        str_file = f'data_temp_newton_steps_{filename}.npz'
        if flag == 0:
            if not os.path.exists(str_file):
                flag = 1
        
        if flag == 1:
            # Build Jacobian pattern
            JacPat = sparse.lil_matrix((nfaces, nvert * 2))
            for ix in range(nfaces):
                verts = F[ix, :]
                for vert in verts:
                    JacPat[ix, vert] = 1
                    JacPat[ix, vert + nvert] = 1
            
            JacPat = JacPat.tocsr()
            i, j = JacPat.nonzero()
            indJ = np.ravel_multi_index((i, j), JacPat.shape)
            
            # Match MATLAB: JacInd = JacPat; JacInd(indJ) = indJ;
            # Create a matrix with same pattern as JacPat, filled with linear indices
            JacInd = JacPat.copy()
            JacInd = JacInd.tocoo()
            JacInd.data = indJ  # Fill with linear indices
            JacInd = JacInd.tocsr()
            
            indVertVal = np.zeros((len(indJ), 6), dtype=int)
            pos_vec = np.zeros(len(indJ), dtype=int)
            indcol = j.astype(np.uint16)
            indrow = i.astype(np.uint16)
            
            if verbose:
                print('Generating Jacobian sparsity pattern...')
            
            for ix in range(len(indJ)):
                if not (ix % 5000) and verbose:
                    print(f'{ix} of {len(indJ)}')
                
                row, col = indrow[ix], indcol[ix]
                # Match MATLAB: indvec = JacInd(row,:); indvec = indvec(indvec>0);
                indvec_row = JacInd[row, :].toarray().flatten()
                indvec = indvec_row[indvec_row > 0]  # Filter out zeros
                # Match MATLAB: pos_vec(ix) = find(indvec==indJ(ix));
                pos_vec[ix] = np.where(indvec == indJ[ix])[0][0] if len(np.where(indvec == indJ[ix])[0]) > 0 else 0
                # Match MATLAB: indVertVal(ix,:) = JacInd(indvec);
                # indvec contains linear indices, use them to index into JacInd
                # JacInd at position (r, c) contains the linear index for that position
                # So indVertVal should contain the linear indices from JacInd at positions indvec
                if len(indvec) > 0:
                    # Extract values from JacInd at linear indices indvec
                    # Convert linear indices to (row, col) pairs
                    rows_indvec, cols_indvec = np.unravel_index(indvec, JacInd.shape)
                    # Get values from JacInd (which are linear indices themselves)
                    vals = np.array([JacInd[r, c] for r, c in zip(rows_indvec, cols_indvec)])
                    # Pad to 6 elements if needed
                    if len(vals) >= 6:
                        indVertVal[ix, :] = vals[:6]
                    else:
                        indVertVal[ix, :len(vals)] = vals
                        indVertVal[ix, len(vals):] = 0  # Pad with zeros
            
            Cv = np.zeros((len(indJ), 6))
            indCv = np.ravel_multi_index((np.arange(len(indJ)), pos_vec), Cv.shape)
            J = sparse.lil_matrix((nfaces, nvert * 2))
            seps = np.sqrt(np.finfo(float).eps)
            
            if flag == 1:
                np.savez(str_file, J=J, seps=seps, JacPat=JacPat, JacInd=JacInd, indJ=indJ,
                        indcol=indcol, indVertVal=indVertVal, Cv=Cv, indCv=indCv,
                        indrow=indrow, pos_vec=pos_vec)
        else:
            data = np.load(str_file, allow_pickle=True)
            J = data['J'].item() if isinstance(data['J'], np.ndarray) else data['J']
            seps = float(data['seps'])
            JacPat = data['JacPat'].item() if isinstance(data['JacPat'], np.ndarray) else data['JacPat']
            if 'JacInd' in data:
                JacInd = data['JacInd'].item() if isinstance(data['JacInd'], np.ndarray) else data['JacInd']
            else:
                # Recreate JacInd if not saved (for backward compatibility)
                JacInd = JacPat.copy()
                JacInd = JacInd.tocoo()
                i, j = JacPat.nonzero()
                indJ_temp = np.ravel_multi_index((i, j), JacPat.shape)
                JacInd.data = indJ_temp
                JacInd = JacInd.tocsr()
            indJ = data['indJ']
            indcol = data['indcol']
            indVertVal = data['indVertVal']
            Cv = data['Cv']
            indCv = data['indCv']
            indrow = data['indrow']
            pos_vec = data['pos_vec']
            
            if verbose:
                print('Preallocating Jacobian...')
            J = sparse.lil_matrix((nfaces, nvert * 2))
            J[tuple(zip(*[(indrow[i], indcol[i]) for i in range(len(indJ))]))] = 1
            if verbose:
                print('Done!')
        
        # Initialize visualization figures if needed
        fig_state = None
        ax_state = None
        fig_hist = None
        ax_hist = None
        fig_residual = None
        ax_residual = None
        
        # Track residuals for each iteration
        residuals = []
        
        if verbose >= 2:
            try:
                import matplotlib.pyplot as plt
                # Create figure for state visualization (will be updated each iteration)
                # Use a unique name to distinguish from persistent figures
                fig_state = plt.figure('Newton Iteration Progress', figsize=(8, 6))
                ax_state = fig_state.add_subplot(111, projection='3d')
                
                # Create figure for residual progression
                fig_residual = plt.figure('Newton Residual Progression', figsize=(8, 5))
                ax_residual = fig_residual.add_subplot(111)
                ax_residual.set_xlabel('Iteration')
                ax_residual.set_ylabel('Residual Norm')
                ax_residual.set_title('Newton Steps Residual Progression')
                ax_residual.set_yscale('log')
                ax_residual.grid(True, alpha=0.3)
                
                if verbose > 2:
                    # Create figure for area histogram (will be updated each iteration)
                    fig_hist = plt.figure('Newton Area Distribution', figsize=(6, 4))
                    ax_hist = fig_hist.add_subplot(111)
            except ImportError:
                pass
        
        # Begin Newton iterations
        for iter_num in range(maxiter):
            if verbose:
                print(iter_num)
            
            t = np.asarray(X[:nvert]).copy()  # Ensure it's a proper array, not a view or scalar
            p = np.asarray(X[nvert:]).copy()  # Ensure it's a proper array, not a view or scalar
            p = np.mod(p, 2 * np.pi)
            # Ensure t and p are 1D arrays (not scalars) with correct shape
            t = np.atleast_1d(t).flatten()
            p = np.atleast_1d(p).flatten()
            # Ensure they have the correct length
            if len(t) != nvert:
                t = np.full(nvert, t[0] if len(t) > 0 else 0.0)
            if len(p) != nvert:
                p = np.full(nvert, p[0] if len(p) > 0 else 0.0)
            
            # Calculate triangle areas at current configuration
            # Compute values and assign directly (avoid in-place assignment issues)
            cos_t = np.cos(np.pi/2 - t)
            sin_t = np.sin(np.pi/2 - t)
            cos_p = np.cos(p)
            sin_p = np.sin(p)
            # Direct assignment to ensure arrays remain arrays
            u = np.asarray(cos_t * cos_p, dtype=np.float64).flatten()
            v = np.asarray(cos_t * sin_p, dtype=np.float64).flatten()
            w = np.asarray(sin_t, dtype=np.float64).flatten()
            # Ensure they're exactly nvert length (should already be, but be safe)
            if len(u) != nvert:
                u = np.resize(u, nvert)
            if len(v) != nvert:
                v = np.resize(v, nvert)
            if len(w) != nvert:
                w = np.resize(w, nvert)
            
            q = np.column_stack([u[F[:, 1]] - u[F[:, 0]], 
                               v[F[:, 1]] - v[F[:, 0]], 
                               w[F[:, 1]] - w[F[:, 0]]])
            r = np.column_stack([u[F[:, 2]] - u[F[:, 0]], 
                               v[F[:, 2]] - v[F[:, 0]], 
                               w[F[:, 2]] - w[F[:, 0]]])
            crossqpr = kk_cross(q, r)
            Areas = np.linalg.norm(crossqpr, axis=1) / 2
            
            # Calculate Jacobian using finite differences
            # CRITICAL: At poles (theta=0 or theta=π), X[0] ≈ 0, so seps*X[0] ≈ 0
            # This causes division by near-zero and huge Jacobian values
            # Use a minimum step size to avoid this numerical issue
            CHG_vec = seps * X
            
            # For theta values near 0 or π, use a minimum step size
            # This prevents division by near-zero at poles
            min_step_theta = 1e-3  # Minimum step for theta (colatitude)
            theta_mask = (X[:nvert] < min_step_theta) | (X[:nvert] > np.pi - min_step_theta)
            if np.any(theta_mask):
                CHG_vec_theta = CHG_vec[:nvert].copy()
                CHG_vec_theta[theta_mask] = np.maximum(np.abs(CHG_vec_theta[theta_mask]), min_step_theta) * np.sign(CHG_vec_theta[theta_mask])
                # For exactly zero, use positive step
                CHG_vec_theta[X[:nvert] == 0] = min_step_theta
                CHG_vec_theta[X[:nvert] == np.pi] = -min_step_theta
                CHG_vec[:nvert] = CHG_vec_theta
                
                # Diagnostic: show what we fixed
                if iter_num < 3 and verbose >= 1:
                    pole_vertices = np.where(theta_mask)[0]
                    if len(pole_vertices) > 0:
                        print(f'  FIXED: {len(pole_vertices)} pole vertex(ices) with near-zero theta step: {pole_vertices[:5]}')
                        for v in pole_vertices[:3]:
                            print(f'    Vertex {v}: theta={X[v]:.6e}, original CHG={seps*X[v]:.6e}, fixed CHG={CHG_vec[v]:.6e}')
            
            # For phi, also use minimum step if near zero
            min_step_phi = 1e-8
            phi_mask = np.abs(X[nvert:]) < min_step_phi
            if np.any(phi_mask):
                CHG_vec_phi = CHG_vec[nvert:].copy()
                CHG_vec_phi[phi_mask] = np.maximum(np.abs(CHG_vec_phi[phi_mask]), min_step_phi) * np.sign(CHG_vec_phi[phi_mask])
                # For exactly zero, use positive step
                CHG_vec_phi[X[nvert:] == 0] = min_step_phi
                CHG_vec[nvert:] = CHG_vec_phi
            # Match MATLAB: VAL = JacPat; VAL(indJ) = X(indcol);
            # Set values at linear indices indJ from X(indcol)
            VAL = JacPat.copy()
            VAL_full = VAL.toarray()  # Convert to dense for linear indexing
            # Set values at linear indices indJ
            VAL_full.flat[indJ] = X[indcol]
            
            # Match MATLAB: Cv(:) = full(VAL(indVertVal(:)));
            # indVertVal contains linear indices into VAL
            # Extract values using linear indexing (MATLAB uses column-major, same as numpy)
            indVertVal_flat = indVertVal.flatten()
            # Filter out zeros (padding) - zeros are not valid indices
            # In MATLAB, indexing with 0 would cause issues, so we only use non-zero indices
            valid_mask = indVertVal_flat > 0
            # Initialize Cv_flat with zeros
            Cv_flat = np.zeros(Cv.size, dtype=VAL_full.dtype)
            # Extract values only for valid (non-zero) indices
            # Note: indVertVal contains 0-based linear indices (from np.ravel_multi_index)
            # So we use them directly without subtracting 1
            if np.any(valid_mask):
                Cv_flat[valid_mask] = VAL_full.flat[indVertVal_flat[valid_mask]]
            Cv = Cv_flat.reshape(Cv.shape)
            
            # Match MATLAB: Cv(indCv) = X(indcol)+CHG_vec(indcol);
            # indCv are linear indices into Cv (1-based in MATLAB, 0-based in Python)
            # But indCv was created with np.ravel_multi_index which is 0-based, so it's correct
            Cv.flat[indCv] = X[indcol] + CHG_vec[indcol]
            
            # Calculate areas with perturbed values
            u1 = np.cos(np.pi/2 - Cv[:, 0]) * np.cos(Cv[:, 3])
            u2 = np.cos(np.pi/2 - Cv[:, 1]) * np.cos(Cv[:, 4])
            u3 = np.cos(np.pi/2 - Cv[:, 2]) * np.cos(Cv[:, 5])
            v1 = np.cos(np.pi/2 - Cv[:, 0]) * np.sin(Cv[:, 3])
            v2 = np.cos(np.pi/2 - Cv[:, 1]) * np.sin(Cv[:, 4])
            v3 = np.cos(np.pi/2 - Cv[:, 2]) * np.sin(Cv[:, 5])
            w1 = np.sin(np.pi/2 - Cv[:, 0])
            w2 = np.sin(np.pi/2 - Cv[:, 1])
            w3 = np.sin(np.pi/2 - Cv[:, 2])
            
            q_pert = np.column_stack([u2 - u1, v2 - v1, w2 - w1])
            r_pert = np.column_stack([u3 - u1, v3 - v1, w3 - w1])
            crossqpr_pert = kk_cross(q_pert, r_pert)
            areas_plus = np.linalg.norm(crossqpr_pert, axis=1) / 2
            
            # Compute Jacobian values
            # Note: CHG_vec should never be exactly zero now due to minimum step size above
            # But keep small epsilon for numerical safety
            Jvals = (areas_plus - Areas[indrow]) / (CHG_vec[indcol] + 1e-12)
            J = sparse.csr_matrix((Jvals, (indrow, indcol)), shape=(nfaces, nvert * 2))
            J.data[np.isnan(J.data)] = 0
            J.data[np.isinf(J.data)] = 100
            
            # Check for triangles with zero or very small Jacobian rows (stuck triangles)
            # These might need special handling
            if iter_num == 0:
                J_row_norms = np.array([np.linalg.norm(J[i, :].data) if J[i, :].nnz > 0 else 0.0 
                                       for i in range(nfaces)])
                small_jac_triangles = np.where(J_row_norms < 1e-10)[0]
                if len(small_jac_triangles) > 0 and verbose >= 1:
                    print(f'  Iter {iter_num}: Found {len(small_jac_triangles)} triangle(s) with very small Jacobian: {small_jac_triangles[:5]}')
                    for tri_idx in small_jac_triangles[:3]:
                        print(f'    Triangle {tri_idx}: area={Areas[tri_idx]:.6e}, ||J_row||={J_row_norms[tri_idx]:.6e}, vertices={F[tri_idx]}')
            
            # Solve linear system: J*J' * dv = -Areas
            # Match MATLAB exactly: gmres(J*J',-Areas,1, 5e0, 10)
            A = J * J.T
            b = -Areas
            
            # DETAILED DIAGNOSTIC for triangle 0: Why is it stuck?
            if iter_num < 3:
                tri0_idx = 0
                J0 = J[tri0_idx, :]
                J0_norm = np.linalg.norm(J0.data) if J0.nnz > 0 else 0.0
                A00 = A[tri0_idx, tri0_idx] if hasattr(A, 'diagonal') else A.toarray()[tri0_idx, tri0_idx]
                b0 = b[tri0_idx]
                tri0_vertices = F[tri0_idx]
                
                if verbose >= 1:
                    print(f'\n  === DETAILED DIAGNOSTIC: Triangle 0 ===')
                    print(f'  Area: {Areas[tri0_idx]:.6e}')
                    print(f'  Vertices: {tri0_vertices}')
                    print(f'  ||J[0,:]||: {J0_norm:.6e}')
                    print(f'  A[0,0] = ||J[0,:]||^2: {A00:.6e}')
                    print(f'  b[0] = -Area[0]: {b0:.6e}')
                    print(f'  Expected dv[0] ≈ b[0]/A[0,0] = {b0/A00:.6e}' if A00 > 0 else '  A[0,0] is zero!')
                    
                    # Check if connected to poles
                    pole_vertices = [0, 123]
                    pole_connections = [v for v in tri0_vertices if v in pole_vertices]
                    if pole_connections:
                        print(f'  WARNING: Connected to pole vertex(ices): {pole_connections}')
                    
                    # Check Jacobian entries for triangle 0
                    J0_dense = J0.toarray().flatten()
                    non_zero_J0 = np.where(np.abs(J0_dense) > 1e-10)[0]
                    if len(non_zero_J0) > 0:
                        print(f'  Non-zero J[0,:] entries at columns: {non_zero_J0[:10]}')
                        print(f'  Values: {[f"{J0_dense[i]:.2e}" for i in non_zero_J0[:10]]}')
                        # Check which vertices these correspond to
                        for col_idx in non_zero_J0[:6]:
                            if col_idx < nvert:
                                print(f'    Column {col_idx}: theta of vertex {col_idx}, value={J0_dense[col_idx]:.2e}')
                            elif col_idx < 2*nvert:
                                vtx_idx = col_idx - nvert
                                print(f'    Column {col_idx}: phi of vertex {vtx_idx}, value={J0_dense[col_idx]:.2e}')
                    
                    # Check row 0 of A matrix - which triangles interact with triangle 0?
                    A0_row = A[tri0_idx, :]
                    if hasattr(A0_row, 'toarray'):
                        A0_row_dense = A0_row.toarray().flatten()
                    else:
                        A0_row_dense = A0_row
                    large_A0_interactions = np.where(np.abs(A0_row_dense) > A00 * 1e-6)[0]
                    if len(large_A0_interactions) > 0:
                        print(f'  Triangles with significant interaction (A[0,j] > A[0,0]*1e-6): {large_A0_interactions[:10]}')
                        print(f'  A[0,0] = {A00:.2e}, max off-diagonal = {np.max(np.abs(A0_row_dense[np.arange(len(A0_row_dense)) != tri0_idx])):.2e}')
                    print(f'  === End Triangle 0 Diagnostic ===\n')
            
            # DIAGNOSTIC: Check which triangle has the largest area and analyze its Jacobian
            if iter_num == 0:
                largest_area_idx = np.argmax(Areas)
                largest_area = Areas[largest_area_idx]
                # Check Jacobian row for this triangle
                J_largest = J[largest_area_idx, :]
                J_largest_norm = np.linalg.norm(J_largest.data) if J_largest.nnz > 0 else 0.0
                # Check which vertices this triangle connects to
                tri_vertices = F[largest_area_idx]
                # Check Jacobian columns for these vertices (theta and phi)
                J_largest_dense = J_largest.toarray().flatten()
                J_cols_theta = [J_largest_dense[v] if v < len(J_largest_dense) else 0 for v in tri_vertices]
                J_cols_phi = [J_largest_dense[v + nvert] if v + nvert < len(J_largest_dense) else 0 for v in tri_vertices]
                
                if verbose >= 1:
                    print(f'  DIAGNOSTIC: Largest triangle is #{largest_area_idx}:')
                    print(f'    area={largest_area:.6e}, ||J_row||={J_largest_norm:.6e}')
                    print(f'    vertices={tri_vertices}')
                    print(f'    J columns (theta) for vertices: {[f"{x:.2e}" for x in J_cols_theta]}')
                    print(f'    J columns (phi) for vertices: {[f"{x:.2e}" for x in J_cols_phi]}')
                    # Check if any of these vertices are poles
                    pole_vertices = [0, 123]  # From the output: North Pole=0, South Pole=123
                    if any(v in tri_vertices for v in pole_vertices):
                        pole_vtx = [v for v in tri_vertices if v in pole_vertices]
                        print(f'    WARNING: This triangle is connected to pole vertex(ices): {pole_vtx}')
                        print(f'    This might explain why it cannot change!')
            
            # Add small regularization to help with stuck triangles
            # This ensures the system is well-conditioned and all triangles can move
            # Use a very small regularization based on the diagonal
            if iter_num == 0:
                # Check condition number on first iteration
                try:
                    A_dense = A.toarray()
                    cond_num = np.linalg.cond(A_dense)
                    if verbose >= 1:
                        print(f'  Iter {iter_num}: Condition number of J*J^T = {cond_num:.2e}')
                except:
                    pass
            
            # Add small regularization to diagonal to help with ill-conditioning
            # This is especially important for triangles with very small Jacobian entries
            # For sparse matrices, use diagonal sum instead of trace
            if hasattr(A, 'diagonal'):
                diag_A = A.diagonal()
                trace_A = diag_A.sum()
                min_diag = diag_A[diag_A > 0].min() if np.any(diag_A > 0) else 1e-10
                max_diag = diag_A.max() if len(diag_A) > 0 else 1e-6
            else:
                A_dense = A.toarray()
                trace_A = np.trace(A_dense) if A.shape[0] < 1000 else A.shape[0] * 1e-6
                diag_A = np.diag(A_dense)
                min_diag = diag_A[diag_A > 0].min() if np.any(diag_A > 0) else 1e-10
                max_diag = diag_A.max() if len(diag_A) > 0 else 1e-6
            
            # NO REGULARIZATION - user wants to diagnose the root cause
            # Just use A directly
            A_reg = A
            
            if verbose >= 1 and iter_num < 3:
                print(f'  Iter {iter_num}: No regularization applied. min_diag = {min_diag:.2e}, max_diag = {max_diag:.2e}')
                print(f'  Condition: max_diag/min_diag = {max_diag/min_diag:.2e}' if min_diag > 0 else '  min_diag is zero!')
            
            # Track residual: norm of Areas at current configuration (what we're minimizing)
            # This should decrease as optimization progresses
            residual_norm = np.linalg.norm(Areas)
            residuals.append(residual_norm)
            
            # Track area changes to identify stuck triangles
            if iter_num == 0:
                Areas_initial = Areas.copy()
            elif iter_num == 1:
                # After first iteration, identify triangles that didn't change
                area_changes = np.abs(Areas - Areas_initial)
                stuck_triangles = np.where(area_changes < 1e-10)[0]
                if len(stuck_triangles) > 0 and verbose >= 1:
                    print(f'  Iter {iter_num}: Found {len(stuck_triangles)} triangle(s) with minimal area change: {stuck_triangles[:5]}')
                    # Check Jacobian for stuck triangles
                    for tri_idx in stuck_triangles[:3]:  # Check first 3
                        jac_row = J[tri_idx, :]
                        jac_norm = np.linalg.norm(jac_row.data) if hasattr(jac_row, 'data') else np.linalg.norm(jac_row)
                        print(f'    Triangle {tri_idx}: area={Areas[tri_idx]:.6e}, ||J_row||={jac_norm:.6e}, vertices={F[tri_idx]}')
            
            # Debug: Check if X is changing (for first few iterations)
            if verbose >= 1 and iter_num < 3:
                X_norm = np.linalg.norm(X)
                Areas_norm = np.linalg.norm(Areas)
                print(f'  Iter {iter_num}: ||X||={X_norm:.6f}, ||Areas||={Areas_norm:.6e}, max(|Areas|)={np.max(np.abs(Areas)):.6e}, min(|Areas|)={np.min(np.abs(Areas)):.6e}')
            
            # Use GMRES as default (matching Matlab), or LU if requested
            if solver.lower() == 'lu':
                # Try LU decomposition with optional regularization
                try:
                    # Try direct sparse LU solve with regularized system
                    dv = spsolve(A_reg, b)
                    solver_used = 'spsolve (LU, regularized)'
                except:
                    # If sparse solve fails, try with regularization
                    try:
                        # Use the system directly (no regularization)
                        dv = spsolve(A_reg, b)
                        solver_used = 'spsolve (LU, no regularization)'
                    except:
                        # Fallback to GMRES if LU fails
                        if verbose:
                            print(f'Warning: LU solver failed at iteration {iter_num}, falling back to GMRES')
                        # MATLAB: [dv,flag4,relres4,iter4,resvec4] = gmres(J*J',-Areas,1, 5e0, 10)
                        # Use reasonable tolerance for GMRES with more iterations
                        # Use regularized system to help with stuck triangles
                        dv, info = gmres(A_reg, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
                        solver_used = 'gmres (fallback from LU, regularized)'
            else:
                # Default: GMRES matching MATLAB exactly
                # MATLAB: [dv,flag4,relres4,iter4,resvec4] = gmres(J*J',-Areas,1, 5e0, 10)
                # Parameters: restart=1, tol=5e0, maxiter=10
                # Note: MATLAB's tol=5e0 is very loose - likely means "don't enforce strict tolerance"
                # In scipy, we use a reasonable tolerance that allows progress
                # Use relative tolerance primarily, with loose absolute tolerance as fallback
                # Increase maxiter to allow more iterations for better convergence
                # Use regularized system to help with stuck triangles
                dv, info = gmres(A_reg, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
                solver_used = 'gmres (regularized)'
            
            # Note: We track the norm of Areas (the objective we're minimizing)
            # This is computed at the start of each iteration and should decrease over time
            
            # Handle solver results:
            # info = 0: successful convergence
            # info > 0: reached maxiter without converging, but solution may still be useful
            # info < 0: illegal input or breakdown (actual failure)
            if info < 0:
                # Actual failure - illegal input or breakdown
                if verbose:
                    print(f'Warning: Solver {solver_used} failed at iteration {iter_num} (info={info}), using zero step')
                dv = np.zeros_like(b)
                solver_used += ' (failed, using zero)'
            elif info > 0:
                # Reached maxiter without converging - solution may still be useful
                if verbose >= 1 and iter_num < 3:
                    print(f'  Iter {iter_num}: GMRES reached maxiter ({info}) without converging, but using solution')
            
            # Clean up any NaN/Inf values (but keep the solution even if info > 0)
            if np.any(np.isnan(dv)) or np.any(np.isinf(dv)):
                if verbose:
                    print(f'Warning: Solver {solver_used} returned NaN/Inf at iteration {iter_num}, using zero step')
                dv = np.zeros_like(b)
                solver_used += ' (NaN/Inf, using zero)'
            else:
                dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Debug: Check dv and solver convergence
            if verbose >= 1 and iter_num < 3:
                dv_norm = np.linalg.norm(dv)
                b_norm = np.linalg.norm(b)
                if dv_norm > 0:
                    # Check if solution actually satisfies the system
                    residual_check = np.linalg.norm(A.dot(dv) - b) / b_norm if b_norm > 0 else 0
                    print(f'  Iter {iter_num}: ||dv||={dv_norm:.6e}, max(|dv|)={np.max(np.abs(dv)):.6e}, solver_info={info}, rel_residual={residual_check:.6e}')
                else:
                    print(f'  Iter {iter_num}: WARNING - ||dv||=0! Solver returned zero solution. solver_info={info}')
            
            if verbose and iter_num == 0:
                print(f'Using solver: {solver_used}')
            
            # Take Newton step - match MATLAB exactly: X = X + stepfac.*J'*dv;
            # MATLAB uses: X = X + stepfac.*J'*dv;
            # J.T is (nvert*2, nfaces), dv is (nfaces,), so J.T * dv is (nvert*2,)
            dX = stepfac * J.T.dot(dv)
            
            # Check for triangles that might be stuck (very small dv values)
            # This can happen if the Jacobian row is nearly zero OR if the system solves in a way that keeps it constant
            if iter_num > 0 and iter_num < 5:
                dv_small = np.abs(dv) < 1e-12
                if np.any(dv_small) and verbose >= 1:
                    stuck_tri_dv = np.where(dv_small)[0]
                    print(f'  Iter {iter_num}: {len(stuck_tri_dv)} triangle(s) have very small dv: {stuck_tri_dv[:5]}')
                    # Check if these correspond to triangles with small areas or small Jacobian
                    for tri_idx in stuck_tri_dv[:5]:  # Check more triangles
                        jac_row = J[tri_idx, :]
                        jac_norm = np.linalg.norm(jac_row.data) if jac_row.nnz > 0 else 0.0
                        # Check A matrix diagonal for this triangle
                        A_diag = A[tri_idx, tri_idx] if hasattr(A, 'diagonal') else A.toarray()[tri_idx, tri_idx]
                        print(f'    Triangle {tri_idx}: area={Areas[tri_idx]:.6e}, dv={dv[tri_idx]:.6e}, ||J_row||={jac_norm:.6e}, A[{tri_idx},{tri_idx}]={A_diag:.2e}')
                        # Check if this is the largest triangle
                        if tri_idx == np.argmax(Areas):
                            print(f'      *** This is the LARGEST triangle - likely the stuck one! ***')
            
            # Debug: Check step size
            if verbose >= 1 and iter_num < 3:
                dX_norm = np.linalg.norm(dX)
                print(f'  Iter {iter_num}: ||dX||={dX_norm:.6e}, max(|dX|)={np.max(np.abs(dX)):.6e}')
            
            X_old = X.copy()  # Save old X for comparison
            X = X + dX
            
            # Apply constraints (matching MATLAB behavior)
            # MATLAB: p = mod(p,2*pi) is applied at start of each iteration
            # We don't clip t in MATLAB version, but we'll keep minimal constraints
            X[nvert:] = np.mod(X[nvert:], 2 * np.pi)  # p in [0, 2π]
            
            # Debug: Verify X actually changed
            if verbose >= 1 and iter_num < 3:
                X_change = np.linalg.norm(X - X_old)
                print(f'  Iter {iter_num}: ||X_new - X_old||={X_change:.6e}')
                
                # Recompute areas after step to see if they improved
                t_new = X[:nvert]
                p_new = X[nvert:]
                p_new = np.mod(p_new, 2 * np.pi)
                u_new = np.cos(np.pi/2 - t_new) * np.cos(p_new)
                v_new = np.cos(np.pi/2 - t_new) * np.sin(p_new)
                w_new = np.sin(np.pi/2 - t_new)
                q_new = np.column_stack([u_new[F[:, 1]] - u_new[F[:, 0]], 
                                        v_new[F[:, 1]] - v_new[F[:, 0]], 
                                        w_new[F[:, 1]] - w_new[F[:, 0]]])
                r_new = np.column_stack([u_new[F[:, 2]] - u_new[F[:, 0]], 
                                        v_new[F[:, 2]] - v_new[F[:, 0]], 
                                        w_new[F[:, 2]] - w_new[F[:, 0]]])
                crossqpr_new = kk_cross(q_new, r_new)
                Areas_new = np.linalg.norm(crossqpr_new, axis=1) / 2
                Areas_new_norm = np.linalg.norm(Areas_new)
                print(f'  Iter {iter_num}: ||Areas_old||={residual_norm:.6e}, ||Areas_new||={Areas_new_norm:.6e}, change={residual_norm - Areas_new_norm:.6e}')
            
            # Update visualization if verbose
            if verbose >= 2:
                t_plot = X[:nvert]
                p_plot = X[nvert:]
                p_plot = np.mod(p_plot, 2 * np.pi)
                u_plot = np.cos(np.pi/2 - t_plot) * np.cos(p_plot)
                v_plot = np.cos(np.pi/2 - t_plot) * np.sin(p_plot)
                w_plot = np.sin(np.pi/2 - t_plot)
                
                # Update existing figure instead of creating new one
                surface_mesh.plot_state(u_plot, v_plot, w_plot, F, 
                                       count=f'Iteration {iter_num + 1}/{maxiter}',
                                       fig=fig_state, ax=ax_state)
                
                # Update residual plot
                if fig_residual is not None and ax_residual is not None:
                    try:
                        import matplotlib.pyplot as plt
                        ax_residual.clear()
                        ax_residual.plot(range(1, len(residuals) + 1), residuals, 'b-o', markersize=4)
                        ax_residual.set_xlabel('Iteration')
                        ax_residual.set_ylabel('Residual Norm')
                        ax_residual.set_title('Newton Steps Residual Progression')
                        ax_residual.set_yscale('log')
                        ax_residual.grid(True, alpha=0.3)
                        ax_residual.set_xlim(0, maxiter + 1)
                        fig_residual.canvas.draw()
                        plt.pause(0.01)
                    except:
                        pass
                
                if verbose > 2 and fig_hist is not None and ax_hist is not None:
                    # Update area histogram in existing figure
                    try:
                        import matplotlib.pyplot as plt
                        ax_hist.clear()
                        ax_hist.hist(Areas, bins=50, edgecolor='black')
                        ax_hist.set_title(f'Triangle Areas - Iteration {iter_num + 1}')
                        ax_hist.set_xlabel('Area')
                        ax_hist.set_ylabel('Frequency')
                        fig_hist.canvas.draw()
                        plt.pause(0.01)
                    except:
                        pass
            
            # Print residual if verbose
            if verbose >= 1:
                print(f'Iteration {iter_num + 1}/{maxiter}: Residual norm = {residual_norm:.6e}')
        
        # Final values
        t = X[:nvert]
        p = X[nvert:]
        p = np.mod(p, 2 * np.pi)
        
        if verbose:
            u_final = np.cos(np.pi/2 - t) * np.cos(p)
            v_final = np.cos(np.pi/2 - t) * np.sin(p)
            w_final = np.sin(np.pi/2 - t)
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(u_final, v_final, w_final, triangles=F)
                plt.show()
            except ImportError:
                pass
        
        return t, p, np.array(residuals)
    
    @staticmethod
    def spherical_triangle_area(t1, p1, t2, p2, t3, p3):
        """
        Calculate the area of a spherical triangle using spherical polar coordinates.
        
        Uses the spherical excess formula: Area = A + B + C - π
        where A, B, C are the spherical angles at the vertices.
        
        Parameters:
        -----------
        t1, p1 : float
            Theta (colatitude) and phi (azimuth) of first vertex
        t2, p2 : float
            Theta and phi of second vertex
        t3, p3 : float
            Theta and phi of third vertex
            
        Returns:
        --------
        area : float
            Area of the spherical triangle on unit sphere
        """
        # Compute side lengths (central angles) using spherical law of cosines
        def central_angle(ta, pa, tb, pb):
            """Compute central angle between two points on sphere"""
            # Spherical law of cosines for sides
            cos_angle = (np.sin(ta) * np.sin(tb) * np.cos(pa - pb) + 
                        np.cos(ta) * np.cos(tb))
            return np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Side lengths (central angles)
        a = central_angle(t2, p2, t3, p3)  # side opposite vertex 1
        b = central_angle(t3, p3, t1, p1)  # side opposite vertex 2
        c = central_angle(t1, p1, t2, p2)  # side opposite vertex 3
        
        # Check for degenerate triangles (very small sides)
        eps = 1e-10
        if a < eps or b < eps or c < eps:
            # Degenerate triangle, return small area
            return eps
        
        # Compute spherical angles using spherical law of cosines for angles
        # cos(A) = (cos(a) - cos(b)cos(c)) / (sin(b)sin(c))
        sin_b = np.sin(b)
        sin_c = np.sin(c)
        sin_a = np.sin(a)
        
        # Avoid division by zero
        sin_b = np.maximum(np.abs(sin_b), eps)
        sin_c = np.maximum(np.abs(sin_c), eps)
        sin_a = np.maximum(np.abs(sin_a), eps)
        
        # Angle at vertex 1 (opposite side a)
        cos_A = (np.cos(a) - np.cos(b) * np.cos(c)) / (sin_b * sin_c)
        A = np.arccos(np.clip(cos_A, -1.0, 1.0))
        
        # Angle at vertex 2 (opposite side b)
        cos_B = (np.cos(b) - np.cos(c) * np.cos(a)) / (sin_c * sin_a)
        B = np.arccos(np.clip(cos_B, -1.0, 1.0))
        
        # Angle at vertex 3 (opposite side c)
        cos_C = (np.cos(c) - np.cos(a) * np.cos(b)) / (sin_a * sin_b)
        C = np.arccos(np.clip(cos_C, -1.0, 1.0))
        
        # Spherical excess
        excess = A + B + C - np.pi
        
        # Ensure non-negative area (should be, but handle numerical errors)
        area = max(excess, 0.0)
        
        return area
    
    @staticmethod
    def spherical_edge_length(t1, p1, t2, p2):
        """
        Geodesic (arc) length between two points on the unit sphere (central angle).
        
        Parameters:
        -----------
        t1, p1 : float
            Theta (colatitude) and phi (azimuth) of first point
        t2, p2 : float
            Theta and phi of second point
            
        Returns:
        --------
        length : float
            Central angle in radians (geodesic distance on unit sphere)
        """
        cos_angle = (np.sin(t1) * np.sin(t2) * np.cos(p1 - p2) + np.cos(t1) * np.cos(t2))
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    @staticmethod
    def spherical_triangle_angles_and_shear(t1, p1, t2, p2, t3, p3):
        """
        Calculate the spherical angles of a triangle and the shear (deviation from mean).
        
        Parameters:
        -----------
        t1, p1 : float
            Theta (colatitude) and phi (azimuth) of first vertex
        t2, p2 : float
            Theta and phi of second vertex
        t3, p3 : float
            Theta and phi of third vertex
            
        Returns:
        --------
        angles : array (3,)
            Spherical angles [A, B, C] at the three vertices
        shear : float
            Sum of squared deviations of angles from their mean
        mean_angle : float
            Mean of the three angles
        """
        # Compute side lengths (central angles) using spherical law of cosines
        def central_angle(ta, pa, tb, pb):
            """Compute central angle between two points on sphere"""
            cos_angle = (np.sin(ta) * np.sin(tb) * np.cos(pa - pb) + 
                        np.cos(ta) * np.cos(tb))
            return np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Side lengths (central angles)
        a = central_angle(t2, p2, t3, p3)  # side opposite vertex 1
        b = central_angle(t3, p3, t1, p1)  # side opposite vertex 2
        c = central_angle(t1, p1, t2, p2)  # side opposite vertex 3
        
        # Check for degenerate triangles
        eps = 1e-10
        if a < eps or b < eps or c < eps:
            # Degenerate triangle, return zero shear
            return np.array([np.pi/3, np.pi/3, np.pi/3]), 0.0, np.pi/3
        
        # Compute spherical angles using spherical law of cosines for angles
        sin_b = np.maximum(np.abs(np.sin(b)), eps)
        sin_c = np.maximum(np.abs(np.sin(c)), eps)
        sin_a = np.maximum(np.abs(np.sin(a)), eps)
        
        # Angle at vertex 1 (opposite side a)
        cos_A = (np.cos(a) - np.cos(b) * np.cos(c)) / (sin_b * sin_c)
        A = np.arccos(np.clip(cos_A, -1.0, 1.0))
        
        # Angle at vertex 2 (opposite side b)
        cos_B = (np.cos(b) - np.cos(c) * np.cos(a)) / (sin_c * sin_a)
        B = np.arccos(np.clip(cos_B, -1.0, 1.0))
        
        # Angle at vertex 3 (opposite side c)
        cos_C = (np.cos(c) - np.cos(a) * np.cos(b)) / (sin_a * sin_b)
        C = np.arccos(np.clip(cos_C, -1.0, 1.0))
        
        angles = np.array([A, B, C])
        mean_angle = np.mean(angles)
        
        # Shear: sum of squared deviations from mean
        shear = np.sum((angles - mean_angle)**2)
        
        return angles, shear, mean_angle
    
    @staticmethod
    def spherical_triangles_valid_orientation(X_cart, F, orient_ref=None, min_orient=1e-8):
        """
        Check that no spherical triangle has flipped or degenerated.
        Orientation per face: orient = dot(v1, cross(v2, v3)) for vertices on unit sphere.
        A flip or degenerate triangle would have orient sign change or |orient| < min_orient.
        
        Parameters:
        -----------
        X_cart : array (nvert, 3)
            Cartesian coordinates on unit sphere
        F : array (nfaces, 3)
            Face connectivity
        orient_ref : array (nfaces,) or None
            Reference orientations (one per face). If None, computed from X_cart and returned.
        min_orient : float
            Minimum allowed |orientation| (below = degenerate)
            
        Returns:
        --------
        valid : bool
            True if all triangles have valid orientation (no flip, not degenerate)
        orient_out : array (nfaces,), optional
            If orient_ref was None, the computed orientations (for use as orient_ref next time)
        """
        nfaces = F.shape[0]
        orient_out = np.zeros(nfaces, dtype=np.float64)
        for ix in range(nfaces):
            v1, v2, v3 = X_cart[F[ix, 0]], X_cart[F[ix, 1]], X_cart[F[ix, 2]]
            orient_out[ix] = np.dot(v1, np.cross(v2, v3))
        if orient_ref is None:
            # No reference: accept if all |orient| above threshold (consistent mesh)
            valid = np.all(np.abs(orient_out) >= min_orient)
            return valid, orient_out
        # Check same sign as reference (no flip) and not degenerate.
        # Use strict same sign: orient_out * orient_ref > 0 (rejects zero or opposite sign)
        same_sign = (orient_out * orient_ref) > 0
        above_min = np.abs(orient_out) >= min_orient
        valid = np.all(same_sign & above_min)
        return valid, orient_out
    
    @staticmethod
    def triangle_angles_and_shear_3d(p1, p2, p3):
        """
        Compute angles and shear for a Euclidean (3D) triangle.
        Shear = sum of squared deviations of angles from their mean (equilateral = 0).
        
        Parameters:
        -----------
        p1, p2, p3 : array-like, shape (3,)
            Vertex coordinates of the triangle
            
        Returns:
        --------
        angles : array (3,)
            Interior angles at the three vertices (radians)
        shear : float
            Sum of squared deviations of angles from mean
        mean_angle : float
            Mean of the three angles (radians)
        """
        p1 = np.asarray(p1, dtype=np.float64).ravel()
        p2 = np.asarray(p2, dtype=np.float64).ravel()
        p3 = np.asarray(p3, dtype=np.float64).ravel()
        a = np.linalg.norm(p2 - p3)  # side opposite p1
        b = np.linalg.norm(p3 - p1)  # side opposite p2
        c = np.linalg.norm(p1 - p2)  # side opposite p3
        eps = 1e-12
        if a < eps or b < eps or c < eps:
            return np.array([np.pi/3, np.pi/3, np.pi/3]), 0.0, np.pi/3
        # Law of cosines for angles
        cos_A = (b*b + c*c - a*a) / (2 * b * c + eps)
        cos_B = (a*a + c*c - b*b) / (2 * a * c + eps)
        cos_C = (a*a + b*b - c*c) / (2 * a * b + eps)
        A = np.arccos(np.clip(cos_A, -1.0, 1.0))
        B = np.arccos(np.clip(cos_B, -1.0, 1.0))
        C = np.arccos(np.clip(cos_C, -1.0, 1.0))
        angles = np.array([A, B, C])
        mean_angle = np.mean(angles)
        shear = np.sum((angles - mean_angle)**2)
        return angles, shear, mean_angle
    
    @staticmethod
    def compute_shear_spherical(t, p, F):
        """
        Compute per-face shear for a spherical parameterization (triangles on unit sphere).
        
        Parameters:
        -----------
        t, p : array (nvert,)
            Theta and phi of vertices on sphere
        F : array (nfaces, 3)
            Face connectivity
            
        Returns:
        --------
        shear_per_face : array (nfaces,)
            Shear value for each face (sum of squared angle deviations from mean)
        summary : dict
            Single-number metrics: 'mean', 'max', 'rms', 'total'
        """
        nfaces = F.shape[0]
        shear_per_face = np.zeros(nfaces, dtype=np.float64)
        for ix in range(nfaces):
            v1, v2, v3 = F[ix, 0], F[ix, 1], F[ix, 2]
            _, shear_per_face[ix], _ = surface_mesh.spherical_triangle_angles_and_shear(
                t[v1], p[v1], t[v2], p[v2], t[v3], p[v3])
        mean_s = np.mean(shear_per_face)
        max_s = np.max(shear_per_face)
        rms_s = np.sqrt(np.mean(shear_per_face**2))
        total_s = np.sum(shear_per_face)
        summary = {'mean': mean_s, 'max': max_s, 'rms': rms_s, 'total': total_s}
        return shear_per_face, summary
    
    @staticmethod
    def compute_shear_3d(X, F):
        """
        Compute per-face shear for a 3D mesh (Euclidean triangle angles).
        
        Parameters:
        -----------
        X : array (nvert, 3)
            Vertex coordinates
        F : array (nfaces, 3)
            Face connectivity
            
        Returns:
        --------
        shear_per_face : array (nfaces,)
            Shear value for each face
        summary : dict
            Single-number metrics: 'mean', 'max', 'rms', 'total'
        """
        nfaces = F.shape[0]
        shear_per_face = np.zeros(nfaces, dtype=np.float64)
        for ix in range(nfaces):
            v1, v2, v3 = F[ix, 0], F[ix, 1], F[ix, 2]
            _, shear_per_face[ix], _ = surface_mesh.triangle_angles_and_shear_3d(
                X[v1, :], X[v2, :], X[v3, :])
        mean_s = np.mean(shear_per_face)
        max_s = np.max(shear_per_face)
        rms_s = np.sqrt(np.mean(shear_per_face**2))
        total_s = np.sum(shear_per_face)
        summary = {'mean': mean_s, 'max': max_s, 'rms': rms_s, 'total': total_s}
        return shear_per_face, summary
    
    @staticmethod
    def newton_steps_cart(t, p, F, stepfac, maxiter, flag=0, filename='untitled', verbose=0, solver='gmres'):
        """
        Minimize triangle areas on sphere using Newton's method with Cartesian coordinates.
        
        This method uses Cartesian coordinates (u, v, w) for the Jacobian and Newton
        iterations, but calculates the spherical triangle surface area using spherical
        polar coordinates (theta, phi). This provides better numerical stability.
        
        Parameters:
        -----------
        t : array
            Theta (latitude) values
        p : array
            Phi (longitude) values
        F : array
            Face connectivity
        stepfac : float
            Step factor for Newton iteration
        maxiter : int
            Maximum iterations
        flag : int, optional
            Whether to prepare Jacobian (1) or load from file (0)
        filename : str, optional
            Filename for saving/loading Jacobian pattern
        verbose : int, optional
            Verbosity level (0=none, 1=text, 2=plot)
        solver : str, optional
            Linear solver to use: 'gmres' (default) or 'lu'
            
        Returns:
        --------
        t : array
            Updated theta values
        p : array
            Updated phi values
        residuals : array
            Residual norms for each iteration
        """
        nvert = len(t)
        nfaces = len(F)
        p = np.mod(p, 2 * np.pi)
        
        # Convert initial spherical coordinates to Cartesian on unit sphere
        u, v, w = kk_sph2cart(t, p, np.ones(nvert))
        X_cart = np.column_stack([u, v, w])  # (nvert, 3)
        
        # Prepare Jacobian sparsity pattern
        import os
        str_file = f'data_temp_newton_steps_cart_{filename}.npz'
        if flag == 0:
            if not os.path.exists(str_file):
                flag = 1
        
        if flag == 1:
            # Build Jacobian pattern: (nfaces, nvert * 3) for Cartesian coordinates
            JacPat = sparse.lil_matrix((nfaces, nvert * 3))
            for ix in range(nfaces):
                verts = F[ix, :]
                for vert in verts:
                    # Each vertex contributes 3 coordinates (u, v, w)
                    JacPat[ix, vert * 3] = 1      # u coordinate
                    JacPat[ix, vert * 3 + 1] = 1  # v coordinate
                    JacPat[ix, vert * 3 + 2] = 1  # w coordinate
            
            JacPat = JacPat.tocsr()
            i, j = JacPat.nonzero()
            indJ = np.ravel_multi_index((i, j), JacPat.shape)
            
            # Create JacInd with linear indices
            JacInd = JacPat.copy()
            JacInd = JacInd.tocoo()
            JacInd.data = indJ
            JacInd = JacInd.tocsr()
            
            indVertVal = np.zeros((len(indJ), 9), dtype=int)  # Up to 9 entries per row (3 vertices * 3 coords)
            pos_vec = np.zeros(len(indJ), dtype=int)
            indcol = j.astype(np.uint16)
            indrow = i.astype(np.uint16)
            
            if verbose:
                print('Generating Jacobian sparsity pattern...')
            
            for ix in range(len(indJ)):
                if not (ix % 5000) and verbose:
                    print(f'{ix} of {len(indJ)}')
                
                row, col = indrow[ix], indcol[ix]
                indvec_row = JacInd[row, :].toarray().flatten()
                indvec = indvec_row[indvec_row > 0]
                pos_vec[ix] = np.where(indvec == indJ[ix])[0][0] if len(np.where(indvec == indJ[ix])[0]) > 0 else 0
                
                if len(indvec) > 0:
                    rows_indvec, cols_indvec = np.unravel_index(indvec, JacInd.shape)
                    vals = np.array([JacInd[r, c] for r, c in zip(rows_indvec, cols_indvec)])
                    if len(vals) >= 9:
                        indVertVal[ix, :] = vals[:9]
                    else:
                        indVertVal[ix, :len(vals)] = vals
                        indVertVal[ix, len(vals):] = 0
            
            Cv = np.zeros((len(indJ), 9))
            indCv = np.ravel_multi_index((np.arange(len(indJ)), pos_vec), Cv.shape)
            J = sparse.lil_matrix((nfaces, nvert * 3))
            seps = np.sqrt(np.finfo(float).eps)
            
            if flag == 1:
                np.savez(str_file, J=J, seps=seps, JacPat=JacPat, JacInd=JacInd, indJ=indJ,
                        indcol=indcol, indVertVal=indVertVal, Cv=Cv, indCv=indCv,
                        indrow=indrow, pos_vec=pos_vec)
        else:
            data = np.load(str_file, allow_pickle=True)
            J = data['J'].item() if isinstance(data['J'], np.ndarray) else data['J']
            seps = float(data['seps'])
            JacPat = data['JacPat'].item() if isinstance(data['JacPat'], np.ndarray) else data['JacPat']
            if 'JacInd' in data:
                JacInd = data['JacInd'].item() if isinstance(data['JacInd'], np.ndarray) else data['JacInd']
            else:
                JacInd = JacPat.copy()
                JacInd = JacInd.tocoo()
                i, j = JacPat.nonzero()
                indJ_temp = np.ravel_multi_index((i, j), JacPat.shape)
                JacInd.data = indJ_temp
                JacInd = JacInd.tocsr()
            indJ = data['indJ']
            indcol = data['indcol']
            indVertVal = data['indVertVal']
            Cv = data['Cv']
            indCv = data['indCv']
            indrow = data['indrow']
            pos_vec = data['pos_vec']
            
            if verbose:
                print('Preallocating Jacobian...')
            J = sparse.lil_matrix((nfaces, nvert * 3))
            J[tuple(zip(*[(indrow[i], indcol[i]) for i in range(len(indJ))]))] = 1
            if verbose:
                print('Done!')
        
        # Initialize visualization figures if needed
        plotter_state = None
        fig_residual = None
        ax_residual = None
        
        residuals = []
        
        if verbose >= 2:
            try:
                # Use PyVista for 3D mesh visualization (works better in Jupyter)
                import pyvista as pv
                # Create plotter once - will be reused for all iterations
                plotter_state = None  # Will be created on first call to plot_state
            except ImportError:
                plotter_state = None
            
            try:
                import matplotlib.pyplot as plt
                fig_residual = plt.figure('Newton Residual Progression (Cartesian)', figsize=(8, 5))
                ax_residual = fig_residual.add_subplot(111)
                ax_residual.set_xlabel('Iteration')
                ax_residual.set_ylabel('Residual Norm')
                ax_residual.set_title('Newton Steps Residual Progression (Cartesian)')
                ax_residual.set_yscale('log')
                ax_residual.grid(True, alpha=0.3)
            except ImportError:
                pass
        
        # Begin Newton iterations
        for iter_num in range(maxiter):
            if verbose:
                print(iter_num)
            
            # Ensure X_cart is on unit sphere
            norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            X_cart = X_cart / norms
            
            # Convert to spherical coordinates for area calculation
            u_curr = X_cart[:, 0]
            v_curr = X_cart[:, 1]
            w_curr = X_cart[:, 2]
            t_curr, p_curr, r_curr = kk_cart2sph(u_curr, v_curr, w_curr)
            p_curr = np.mod(p_curr, 2 * np.pi)
            
            # Calculate triangle areas using spherical coordinates
            Areas = np.zeros(nfaces, dtype=np.float64)
            for ix in range(nfaces):
                v1, v2, v3 = F[ix, 0], F[ix, 1], F[ix, 2]
                area = surface_mesh.spherical_triangle_area(
                    t_curr[v1], p_curr[v1],
                    t_curr[v2], p_curr[v2],
                    t_curr[v3], p_curr[v3]
                )
                Areas[ix] = area
            
            # Calculate Jacobian using finite differences
            X_flat = X_cart.flatten()  # Flatten to (nvert * 3,)
            CHG_vec = seps * np.abs(X_flat)
            # Use minimum step size to avoid numerical issues
            min_step = 1e-8
            CHG_vec = np.maximum(CHG_vec, min_step)
            
            # Calculate areas with perturbed values using finite differences
            # For each Jacobian entry, we need to perturb one coordinate, project to sphere, and recompute area
            areas_plus = np.zeros(len(indJ), dtype=np.float64)
            
            # For each Jacobian entry, compute perturbed area
            for jx in range(len(indJ)):
                row = indrow[jx]
                col = indcol[jx]
                
                # Get the triangle vertices
                v1, v2, v3 = F[row, 0], F[row, 1], F[row, 2]
                
                # Determine which vertex and coordinate this entry corresponds to
                vert_idx = col // 3
                coord_idx = col % 3
                
                # Only compute if this vertex is part of the triangle
                if vert_idx in [v1, v2, v3]:
                    # Create perturbed Cartesian coordinates
                    X_pert = X_cart.copy()
                    
                    # Perturb the specific coordinate
                    X_pert[vert_idx, coord_idx] = X_flat[col] + CHG_vec[col]
                    
                    # Project the perturbed vertex back to unit sphere
                    # This is important: when we perturb one coordinate, we must renormalize
                    norm_pert = np.linalg.norm(X_pert[vert_idx, :])
                    norm_pert = np.maximum(norm_pert, 1e-12)
                    X_pert[vert_idx, :] = X_pert[vert_idx, :] / norm_pert
                    
                    # Convert to spherical and compute area
                    u_pert = X_pert[:, 0]
                    v_pert = X_pert[:, 1]
                    w_pert = X_pert[:, 2]
                    t_pert, p_pert, _ = kk_cart2sph(u_pert, v_pert, w_pert)
                    p_pert = np.mod(p_pert, 2 * np.pi)
                    
                    area_pert = surface_mesh.spherical_triangle_area(
                        t_pert[v1], p_pert[v1],
                        t_pert[v2], p_pert[v2],
                        t_pert[v3], p_pert[v3]
                    )
                    areas_plus[jx] = area_pert
                else:
                    # If vertex is not in triangle, area doesn't change
                    areas_plus[jx] = Areas[row]
            
            # Compute Jacobian values
            Jvals = (areas_plus - Areas[indrow]) / (CHG_vec[indcol] + 1e-12)
            J = sparse.csr_matrix((Jvals, (indrow, indcol)), shape=(nfaces, nvert * 3))
            J.data[np.isnan(J.data)] = 0
            J.data[np.isinf(J.data)] = 0
            
            # Solve linear system: J*J' * dv = -Areas
            A = J * J.T
            b = -Areas
            
            # Use GMRES as default
            if solver.lower() == 'lu':
                try:
                    dv = spsolve(A, b)
                    solver_used = 'spsolve (LU)'
                except:
                    if verbose:
                        print(f'Warning: LU solver failed at iteration {iter_num}, falling back to GMRES')
                    dv, info = gmres(A, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
                    solver_used = 'gmres (fallback from LU)'
            else:
                dv, info = gmres(A, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
                solver_used = 'gmres'
            
            # Handle solver results
            if info < 0:
                if verbose:
                    print(f'Warning: Solver {solver_used} failed at iteration {iter_num} (info={info}), using zero step')
                dv = np.zeros_like(b)
            elif info > 0:
                if verbose >= 1 and iter_num < 3:
                    print(f'  Iter {iter_num}: GMRES reached maxiter ({info}) without converging, but using solution')
            
            # Clean up any NaN/Inf values
            if np.any(np.isnan(dv)) or np.any(np.isinf(dv)):
                if verbose:
                    print(f'Warning: Solver {solver_used} returned NaN/Inf at iteration {iter_num}, using zero step')
                dv = np.zeros_like(b)
            else:
                dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Track residual
            residual_norm = np.linalg.norm(Areas)
            residuals.append(residual_norm)
            
            if verbose >= 1 and iter_num < 3:
                dv_norm = np.linalg.norm(dv)
                b_norm = np.linalg.norm(b)
                if dv_norm > 0:
                    residual_check = np.linalg.norm(A.dot(dv) - b) / b_norm if b_norm > 0 else 0
                    print(f'  Iter {iter_num}: ||dv||={dv_norm:.6e}, ||Areas||={residual_norm:.6e}, rel_residual={residual_check:.6e}')
            
            # Take Newton step: X = X + stepfac * J' * dv
            dX_flat = stepfac * J.T.dot(dv)  # (nvert * 3,)
            dX = dX_flat.reshape((nvert, 3))  # (nvert, 3)
            
            X_cart = X_cart + dX
            
            # Project back to unit sphere
            norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X_cart = X_cart / norms
            
            # Update visualization if verbose
            if verbose >= 2:
                try:
                    import sys
                    in_jupyter = 'ipykernel' in sys.modules
                    
                    # Update residual plot
                    if fig_residual is not None and ax_residual is not None:
                        import matplotlib.pyplot as plt
                        ax_residual.clear()
                        ax_residual.plot(range(1, len(residuals) + 1), residuals, 'b-o', markersize=4)
                        ax_residual.set_xlabel('Iteration')
                        ax_residual.set_ylabel('Residual Norm')
                        ax_residual.set_title('Newton Steps Residual Progression (Cartesian)')
                        ax_residual.set_yscale('log')
                        ax_residual.grid(True, alpha=0.3)
                        ax_residual.set_xlim(0, maxiter + 1)
                        fig_residual.canvas.draw()
                    
                    # Update 3D mesh plot (reuse same plotter)
                    # For Jupyter, get image data instead of displaying immediately
                    if in_jupyter:
                        plotter_state, image_3d = surface_mesh.plot_state(
                            X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F,
                            count=f'Iteration {iter_num + 1}/{maxiter}',
                            plotter=plotter_state, clear_display=False, return_image=True, two_views=True)
                    else:
                        plotter_state = surface_mesh.plot_state(
                            X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F,
                            count=f'Iteration {iter_num + 1}/{maxiter}',
                            plotter=plotter_state, clear_display=False, return_image=False, two_views=True)
                        image_3d = None
                    
                    # Display both figures together
                    if in_jupyter:
                        try:
                            from IPython.display import display, clear_output
                            # Clear output once and display both figures
                            clear_output(wait=True)
                            if fig_residual is not None:
                                display(fig_residual)
                            if image_3d is not None:
                                display(image_3d)
                            elif verbose > 2:
                                print("Warning: 3D plot image is None, screenshot may have failed")
                        except Exception as e:
                            if verbose > 2:
                                print(f"Warning: Display failed: {e}")
                            import traceback
                            if verbose > 3:
                                traceback.print_exc()
                    else:
                        import matplotlib.pyplot as plt
                        if fig_residual is not None:
                            plt.pause(0.01)
                except Exception as e:
                    if verbose > 2:
                        print(f"Warning: Plotting failed: {e}")
                    pass
            
            if verbose >= 1:
                print(f'Iteration {iter_num + 1}/{maxiter}: Residual norm = {residual_norm:.6e}')
        
        # Convert final Cartesian coordinates back to spherical
        u_final = X_cart[:, 0]
        v_final = X_cart[:, 1]
        w_final = X_cart[:, 2]
        t_final, p_final, r_final = kk_cart2sph(u_final, v_final, w_final)
        p_final = np.mod(p_final, 2 * np.pi)
        
        if verbose:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(u_final, v_final, w_final, triangles=F)
                plt.show()
            except ImportError:
                pass
        
        return t_final, p_final, np.array(residuals)
    
    @staticmethod
    def newton_shear_correction(t, p, F, stepfac, maxiter, flag=0, filename='untitled', verbose=0, solver='gmres', shear_only=False, prevent_flip=True):
        """
        Minimize triangle areas and shear on sphere using Newton's method, alternating between objectives.
        
        If shear_only=False (default): alternates between
        - Even iterations: minimizing spherical triangle areas (equalizing areas)
        - Odd iterations: minimizing spherical triangle shear (minimizing deviation of angles from mean)
        If shear_only=True: every iteration minimizes shear only (pure shear correction, like method 4 for edge length).
        
        If prevent_flip=True: before accepting a Newton step, a backtracking line search is used so that
        no spherical triangle flips or degenerates (orientation preserved). Can be turned off for speed.
        
        Uses Cartesian coordinates (u, v, w) for the Jacobian and Newton iterations, but calculates
        the spherical triangle properties using spherical polar coordinates (theta, phi).
        
        Parameters:
        -----------
        t : array
            Theta (latitude) values
        p : array
            Phi (longitude) values
        F : array
            Face connectivity
        stepfac : float
            Step factor for Newton iteration
        maxiter : int
            Maximum iterations
        flag : int, optional
            Whether to prepare Jacobian (1) or load from file (0)
        filename : str, optional
            Filename for saving/loading Jacobian pattern
        verbose : int, optional
            Verbosity level (0=none, 1=text, 2=plot)
        solver : str, optional
            Linear solver to use: 'gmres' (default) or 'lu'
        shear_only : bool, optional
            If True, minimize shear only every iteration (no area alternation).
        prevent_flip : bool, optional
            If True (default), use backtracking line search so no spherical triangle
            flips or degenerates. Set to False to disable (faster but may flip).
            
        Returns:
        --------
        t : array
            Updated theta values
        p : array
            Updated phi values
        residuals : array
            Residual norms for each iteration
        """
        nvert = len(t)
        nfaces = len(F)
        p = np.mod(p, 2 * np.pi)
        
        # Convert initial spherical coordinates to Cartesian on unit sphere
        u, v, w = kk_sph2cart(t, p, np.ones(nvert))
        X_cart = np.column_stack([u, v, w])  # (nvert, 3)
        
        # Prepare Jacobian sparsity pattern (same as newton_steps_cart)
        import os
        str_file = f'data_temp_newton_shear_correction_{filename}.npz'
        if flag == 0:
            if not os.path.exists(str_file):
                flag = 1
        
        if flag == 1:
            # Build Jacobian pattern: (nfaces, nvert * 3) for Cartesian coordinates
            JacPat = sparse.lil_matrix((nfaces, nvert * 3))
            for ix in range(nfaces):
                verts = F[ix, :]
                for vert in verts:
                    # Each vertex contributes 3 coordinates (u, v, w)
                    JacPat[ix, vert * 3] = 1      # u coordinate
                    JacPat[ix, vert * 3 + 1] = 1  # v coordinate
                    JacPat[ix, vert * 3 + 2] = 1  # w coordinate
            
            JacPat = JacPat.tocsr()
            i, j = JacPat.nonzero()
            indJ = np.ravel_multi_index((i, j), JacPat.shape)
            
            # Create JacInd with linear indices
            JacInd = JacPat.copy()
            JacInd = JacInd.tocoo()
            JacInd.data = indJ
            JacInd = JacInd.tocsr()
            
            indVertVal = np.zeros((len(indJ), 9), dtype=int)
            pos_vec = np.zeros(len(indJ), dtype=int)
            indcol = j.astype(np.uint16)
            indrow = i.astype(np.uint16)
            
            if verbose:
                print('Generating Jacobian sparsity pattern...')
            
            for ix in range(len(indJ)):
                if not (ix % 5000) and verbose:
                    print(f'{ix} of {len(indJ)}')
                
                row, col = indrow[ix], indcol[ix]
                indvec_row = JacInd[row, :].toarray().flatten()
                indvec = indvec_row[indvec_row > 0]
                pos_vec[ix] = np.where(indvec == indJ[ix])[0][0] if len(np.where(indvec == indJ[ix])[0]) > 0 else 0
                
                if len(indvec) > 0:
                    rows_indvec, cols_indvec = np.unravel_index(indvec, JacInd.shape)
                    vals = np.array([JacInd[r, c] for r, c in zip(rows_indvec, cols_indvec)])
                    if len(vals) >= 9:
                        indVertVal[ix, :] = vals[:9]
                    else:
                        indVertVal[ix, :len(vals)] = vals
                        indVertVal[ix, len(vals):] = 0
            
            Cv = np.zeros((len(indJ), 9))
            indCv = np.ravel_multi_index((np.arange(len(indJ)), pos_vec), Cv.shape)
            J = sparse.lil_matrix((nfaces, nvert * 3))
            seps = np.sqrt(np.finfo(float).eps)
            
            if flag == 1:
                np.savez(str_file, J=J, seps=seps, JacPat=JacPat, JacInd=JacInd, indJ=indJ,
                        indcol=indcol, indVertVal=indVertVal, Cv=Cv, indCv=indCv,
                        indrow=indrow, pos_vec=pos_vec)
        else:
            data = np.load(str_file, allow_pickle=True)
            J = data['J'].item() if isinstance(data['J'], np.ndarray) else data['J']
            seps = float(data['seps'])
            JacPat = data['JacPat'].item() if isinstance(data['JacPat'], np.ndarray) else data['JacPat']
            if 'JacInd' in data:
                JacInd = data['JacInd'].item() if isinstance(data['JacInd'], np.ndarray) else data['JacInd']
            else:
                JacInd = JacPat.copy()
                JacInd = JacInd.tocoo()
                i, j = JacPat.nonzero()
                indJ_temp = np.ravel_multi_index((i, j), JacPat.shape)
                JacInd.data = indJ_temp
                JacInd = JacInd.tocsr()
            indJ = data['indJ']
            indcol = data['indcol']
            indVertVal = data['indVertVal']
            Cv = data['Cv']
            indCv = data['indCv']
            indrow = data['indrow']
            pos_vec = data['pos_vec']
            
            if verbose:
                print('Preallocating Jacobian...')
            J = sparse.lil_matrix((nfaces, nvert * 3))
            J[tuple(zip(*[(indrow[i], indcol[i]) for i in range(len(indJ))]))] = 1
            if verbose:
                print('Done!')
        
        # Initialize visualization figures if needed
        plotter_state = None
        fig_residual = None
        ax_residual = None
        
        residuals = []
        
        if verbose >= 2:
            try:
                import pyvista as pv
                plotter_state = None
            except ImportError:
                plotter_state = None
            
            try:
                import matplotlib.pyplot as plt
                fig_residual = plt.figure('Newton Shear Correction Residual Progression', figsize=(8, 5))
                ax_residual = fig_residual.add_subplot(111)
                ax_residual.set_xlabel('Iteration')
                ax_residual.set_ylabel('Residual Norm')
                ax_residual.set_title('Newton Shear Correction Residual Progression')
                ax_residual.set_yscale('log')
                ax_residual.grid(True, alpha=0.3)
            except ImportError:
                pass
        
        # Initialize reference orientation for flip prevention (computed once from initial state)
        orient_ref = None
        if prevent_flip:
            _, orient_ref = surface_mesh.spherical_triangles_valid_orientation(X_cart, F, None, min_orient=1e-8)
            if verbose >= 1:
                print(f'Initialized reference orientation for flip prevention: {len(orient_ref)} faces')
        
        # Begin Newton iterations
        for iter_num in range(maxiter):
            # Determine objective: pure shear (shear_only) or alternate area/shear
            minimize_area = False if shear_only else (iter_num % 2 == 0)
            objective_name = 'area' if minimize_area else 'shear'
            
            if verbose:
                print(f'Iteration {iter_num + 1}/{maxiter}: Minimizing {objective_name}')
            
            # Ensure X_cart is on unit sphere
            norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X_cart = X_cart / norms
            
            # Convert to spherical coordinates
            u_curr = X_cart[:, 0]
            v_curr = X_cart[:, 1]
            w_curr = X_cart[:, 2]
            t_curr, p_curr, r_curr = kk_cart2sph(u_curr, v_curr, w_curr)
            p_curr = np.mod(p_curr, 2 * np.pi)
            
            # Calculate objective function values
            if minimize_area:
                # Calculate triangle areas
                Objectives = np.zeros(nfaces, dtype=np.float64)
                for ix in range(nfaces):
                    v1, v2, v3 = F[ix, 0], F[ix, 1], F[ix, 2]
                    area = surface_mesh.spherical_triangle_area(
                        t_curr[v1], p_curr[v1],
                        t_curr[v2], p_curr[v2],
                        t_curr[v3], p_curr[v3]
                    )
                    Objectives[ix] = area
            else:
                # Calculate triangle shear (deviation of angles from mean)
                Objectives = np.zeros(nfaces, dtype=np.float64)
                for ix in range(nfaces):
                    v1, v2, v3 = F[ix, 0], F[ix, 1], F[ix, 2]
                    _, shear, _ = surface_mesh.spherical_triangle_angles_and_shear(
                        t_curr[v1], p_curr[v1],
                        t_curr[v2], p_curr[v2],
                        t_curr[v3], p_curr[v3]
                    )
                    Objectives[ix] = shear
            
            # Calculate Jacobian using finite differences
            X_flat = X_cart.flatten()
            CHG_vec = seps * np.abs(X_flat)
            min_step = 1e-8
            CHG_vec = np.maximum(CHG_vec, min_step)
            
            # Calculate objectives with perturbed values
            objectives_plus = np.zeros(len(indJ), dtype=np.float64)
            
            for jx in range(len(indJ)):
                row = indrow[jx]
                col = indcol[jx]
                
                v1, v2, v3 = F[row, 0], F[row, 1], F[row, 2]
                vert_idx = col // 3
                coord_idx = col % 3
                
                if vert_idx in [v1, v2, v3]:
                    # Create perturbed Cartesian coordinates
                    X_pert = X_cart.copy()
                    X_pert[vert_idx, coord_idx] = X_flat[col] + CHG_vec[col]
                    
                    # Project the perturbed vertex back to unit sphere
                    norm_pert = np.linalg.norm(X_pert[vert_idx, :])
                    norm_pert = np.maximum(norm_pert, 1e-12)
                    X_pert[vert_idx, :] = X_pert[vert_idx, :] / norm_pert
                    
                    # Convert to spherical
                    u_pert = X_pert[:, 0]
                    v_pert = X_pert[:, 1]
                    w_pert = X_pert[:, 2]
                    t_pert, p_pert, _ = kk_cart2sph(u_pert, v_pert, w_pert)
                    p_pert = np.mod(p_pert, 2 * np.pi)
                    
                    # Compute objective
                    if minimize_area:
                        obj_pert = surface_mesh.spherical_triangle_area(
                            t_pert[v1], p_pert[v1],
                            t_pert[v2], p_pert[v2],
                            t_pert[v3], p_pert[v3]
                        )
                    else:
                        _, obj_pert, _ = surface_mesh.spherical_triangle_angles_and_shear(
                            t_pert[v1], p_pert[v1],
                            t_pert[v2], p_pert[v2],
                            t_pert[v3], p_pert[v3]
                        )
                    objectives_plus[jx] = obj_pert
                else:
                    objectives_plus[jx] = Objectives[row]
            
            # Compute Jacobian values
            Jvals = (objectives_plus - Objectives[indrow]) / (CHG_vec[indcol] + 1e-12)
            J = sparse.csr_matrix((Jvals, (indrow, indcol)), shape=(nfaces, nvert * 3))
            J.data[np.isnan(J.data)] = 0
            J.data[np.isinf(J.data)] = 0
            
            # Solve linear system: J*J' * dv = -Objectives
            A = J * J.T
            b = -Objectives
            
            # Use GMRES as default
            if solver.lower() == 'lu':
                try:
                    dv = spsolve(A, b)
                    solver_used = 'spsolve (LU)'
                except:
                    if verbose:
                        print(f'Warning: LU solver failed at iteration {iter_num}, falling back to GMRES')
                    dv, info = gmres(A, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
                    solver_used = 'gmres (fallback from LU)'
            else:
                dv, info = gmres(A, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
                solver_used = 'gmres'
            
            # Handle solver results
            if info < 0:
                if verbose:
                    print(f'Warning: Solver {solver_used} failed at iteration {iter_num} (info={info}), using zero step')
                dv = np.zeros_like(b)
            elif info > 0:
                if verbose >= 1 and iter_num < 3:
                    print(f'  Iter {iter_num}: GMRES reached maxiter ({info}) without converging, but using solution')
            
            # Clean up any NaN/Inf values
            if np.any(np.isnan(dv)) or np.any(np.isinf(dv)):
                if verbose:
                    print(f'Warning: Solver {solver_used} returned NaN/Inf at iteration {iter_num}, using zero step')
                dv = np.zeros_like(b)
            else:
                dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Track residual
            residual_norm = np.linalg.norm(Objectives)
            residuals.append(residual_norm)
            
            if verbose >= 1 and iter_num < 3:
                dv_norm = np.linalg.norm(dv)
                b_norm = np.linalg.norm(b)
                if dv_norm > 0:
                    residual_check = np.linalg.norm(A.dot(dv) - b) / b_norm if b_norm > 0 else 0
                    print(f'  Iter {iter_num}: ||dv||={dv_norm:.6e}, ||Objectives||={residual_norm:.6e}, rel_residual={residual_check:.6e}')
            
            # Take Newton step: X = X + stepfac * J' * dv
            dX_flat = stepfac * J.T.dot(dv)
            dX = dX_flat.reshape((nvert, 3))
            
            # Optional: backtracking line search to prevent triangle flip
            if prevent_flip:
                X_cart_old = X_cart.copy()
                # Use the reference orientation computed at the start (orient_ref is already set)
                alpha = 1.0
                beta = 0.5
                alpha_min = 1e-6
                max_ls = 40
                step_accept = False
                for _ in range(max_ls):
                    if alpha < alpha_min:
                        if verbose >= 1:
                            print(f'  Line search: no valid step >= alpha_min={alpha_min}; rejecting step')
                        break
                    X_trial = X_cart_old + alpha * dX
                    norms = np.linalg.norm(X_trial, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)
                    X_trial = X_trial / norms
                    valid, _ = surface_mesh.spherical_triangles_valid_orientation(X_trial, F, orient_ref, min_orient=1e-8)
                    if valid:
                        X_cart = X_trial
                        step_accept = True
                        if verbose >= 2:
                            print(f'  Line search: accepted alpha={alpha:.6e}')
                        break
                    alpha *= beta
                if not step_accept:
                    X_cart = X_cart_old
            else:
                X_cart = X_cart + dX
                norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                X_cart = X_cart / norms
            
            # Project back to unit sphere (when not using line search we already did it above)
            if not prevent_flip:
                norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                X_cart = X_cart / norms
            
            # Update visualization if verbose
            if verbose >= 2:
                try:
                    import sys
                    in_jupyter = 'ipykernel' in sys.modules
                    
                    # Update residual plot
                    if fig_residual is not None and ax_residual is not None:
                        import matplotlib.pyplot as plt
                        ax_residual.clear()
                        ax_residual.plot(range(1, len(residuals) + 1), residuals, 'b-o', markersize=4)
                        ax_residual.set_xlabel('Iteration')
                        ax_residual.set_ylabel('Residual Norm')
                        ax_residual.set_title(f'Newton Shear Correction ({objective_name} minimization)')
                        ax_residual.set_yscale('log')
                        ax_residual.grid(True, alpha=0.3)
                        ax_residual.set_xlim(0, maxiter + 1)
                        fig_residual.canvas.draw()
                    
                    # Update 3D mesh plot
                    if in_jupyter:
                        plotter_state, image_3d = surface_mesh.plot_state(
                            X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F,
                            count=f'Iter {iter_num + 1}/{maxiter} ({objective_name})',
                            plotter=plotter_state, clear_display=False, return_image=True, two_views=True)
                    else:
                        plotter_state = surface_mesh.plot_state(
                            X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F,
                            count=f'Iter {iter_num + 1}/{maxiter} ({objective_name})',
                            plotter=plotter_state, clear_display=False, return_image=False, two_views=True)
                        image_3d = None
                    
                    # Display both figures together
                    if in_jupyter:
                        try:
                            from IPython.display import display, clear_output
                            clear_output(wait=True)
                            if fig_residual is not None:
                                display(fig_residual)
                            if image_3d is not None:
                                display(image_3d)
                        except Exception as e:
                            if verbose > 2:
                                print(f"Warning: Display failed: {e}")
                except Exception as e:
                    if verbose > 2:
                        print(f"Warning: Plotting failed: {e}")
                    pass
            
            if verbose >= 1:
                print(f'Iteration {iter_num + 1}/{maxiter}: Residual norm = {residual_norm:.6e} ({objective_name})')
        
        # Convert final Cartesian coordinates back to spherical
        u_final = X_cart[:, 0]
        v_final = X_cart[:, 1]
        w_final = X_cart[:, 2]
        t_final, p_final, r_final = kk_cart2sph(u_final, v_final, w_final)
        p_final = np.mod(p_final, 2 * np.pi)
        
        if verbose:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(u_final, v_final, w_final, triangles=F)
                plt.show()
            except ImportError:
                pass
        
        return t_final, p_final, np.array(residuals)
    
    @staticmethod
    def newton_edge_length_correction(t, p, F, E, edge_target_vertex_count, stepfac_area, stepfac_edge, maxiter, flag=0, filename='untitled', verbose=0, solver='gmres', prevent_flip=False):
        """
        Minimize triangle areas and edge lengths on sphere, alternating between objectives.
        
        For simplified meshes: even iterations minimize spherical triangle area; odd iterations
        minimize spherical edge-length error. Target length for each edge is proportional to
        the number of fine-mesh vertices along that edge (edge_target_vertex_count), relative
        to total spherical edge length.
        
        Parameters:
        -----------
        t, p : array
            Theta and phi
        F : array
            Face connectivity
        E : array (n_edges, 2)
            Edge list (vertex index pairs)
        edge_target_vertex_count : array (n_edges,) or None
            Number of fine-mesh vertices along each simplified edge. If None, uses ones (uniform).
        stepfac_area : float
            Newton step factor for area minimization
        stepfac_edge : float
            Newton step factor for edge-length minimization
        maxiter : int
            Maximum iterations
        flag, filename, verbose, solver
            As in newton_steps_cart.
        prevent_flip : bool, optional
            If True (default), use backtracking line search to avoid triangle flip.
            
        Returns:
        --------
        t, p : array
            Updated spherical coordinates
        residuals : array
            Residual norms per iteration
        """
        import os
        nvert = len(t)
        nfaces = len(F)
        n_edges = len(E)
        p = np.mod(p, 2 * np.pi)
        
        # Target vertex counts: default uniform if not provided
        if edge_target_vertex_count is None:
            edge_target_vertex_count = np.ones(n_edges, dtype=np.float64)
        else:
            edge_target_vertex_count = np.asarray(edge_target_vertex_count, dtype=np.float64).ravel()
            if len(edge_target_vertex_count) != n_edges:
                raise ValueError('edge_target_vertex_count length must equal number of edges')
        
        count_sum = np.maximum(np.sum(edge_target_vertex_count), 1e-12)
        
        # Cartesian on unit sphere
        u, v, w = kk_sph2cart(t, p, np.ones(nvert))
        X_cart = np.column_stack([u, v, w])
        
        # Compute initial edge lengths to establish fixed target lengths for edge-length steps
        # Target lengths should be proportional to edge_target_vertex_count and fixed throughout optimization
        t_initial, p_initial = t.copy(), p.copy()
        p_initial = np.mod(p_initial, 2 * np.pi)
        L_e_initial = np.zeros(n_edges, dtype=np.float64)
        for ex in range(n_edges):
            va, vb = int(E[ex, 0]), int(E[ex, 1])
            L_e_initial[ex] = surface_mesh.spherical_edge_length(
                t_initial[va], p_initial[va], t_initial[vb], p_initial[vb])
        L_total_initial = np.sum(L_e_initial)
        
        # Fixed target lengths: proportional to fine-vertex counts, based on initial total length
        T_e_fixed = (edge_target_vertex_count / count_sum) * L_total_initial
        
        str_file = f'data_temp_newton_edge_length_correction_{filename}.npz'
        if flag == 0:
            if not os.path.exists(str_file):
                flag = 1
        
        seps = np.sqrt(np.finfo(float).eps)
        min_step = 1e-8
        
        # Area Jacobian pattern (nfaces x nvert*3)
        if flag == 1:
            JacPat_area = sparse.lil_matrix((nfaces, nvert * 3))
            for ix in range(nfaces):
                for vert in F[ix, :]:
                    JacPat_area[ix, vert * 3] = 1
                    JacPat_area[ix, vert * 3 + 1] = 1
                    JacPat_area[ix, vert * 3 + 2] = 1
            JacPat_area = JacPat_area.tocsr()
            i_a, j_a = JacPat_area.nonzero()
            indJ_area = np.ravel_multi_index((i_a, j_a), JacPat_area.shape)
            indrow_area = i_a.astype(np.uint16)
            indcol_area = j_a.astype(np.uint16)
            
            # Edge Jacobian pattern (n_edges x nvert*3)
            JacPat_edge = sparse.lil_matrix((n_edges, nvert * 3))
            for ex in range(n_edges):
                v1, v2 = int(E[ex, 0]), int(E[ex, 1])
                for vert in (v1, v2):
                    JacPat_edge[ex, vert * 3] = 1
                    JacPat_edge[ex, vert * 3 + 1] = 1
                    JacPat_edge[ex, vert * 3 + 2] = 1
            JacPat_edge = JacPat_edge.tocsr()
            i_e, j_e = JacPat_edge.nonzero()
            indJ_edge = np.ravel_multi_index((i_e, j_e), JacPat_edge.shape)
            indrow_edge = i_e.astype(np.uint16)
            indcol_edge = j_e.astype(np.uint16)
            
            np.savez(str_file, JacPat_area=JacPat_area, indJ_area=indJ_area, indrow_area=indrow_area, indcol_area=indcol_area,
                     JacPat_edge=JacPat_edge, indJ_edge=indJ_edge, indrow_edge=indrow_edge, indcol_edge=indcol_edge,
                     seps=seps)
        else:
            data = np.load(str_file, allow_pickle=True)
            JacPat_area = data['JacPat_area'].item() if hasattr(data['JacPat_area'], 'item') else data['JacPat_area']
            indJ_area = data['indJ_area']
            indrow_area = data['indrow_area']
            indcol_area = data['indcol_area']
            JacPat_edge = data['JacPat_edge'].item() if hasattr(data['JacPat_edge'], 'item') else data['JacPat_edge']
            indJ_edge = data['indJ_edge']
            indrow_edge = data['indrow_edge']
            indcol_edge = data['indcol_edge']
            seps = float(data['seps'])
        
        residuals = []
        plotter_state = None
        fig_residual = None
        ax_residual = None
        if verbose >= 2:
            try:
                import pyvista as pv
                plotter_state = None
            except ImportError:
                plotter_state = None
            try:
                import matplotlib.pyplot as plt
                fig_residual = plt.figure('Newton Edge-Length Correction Residual', figsize=(8, 5))
                ax_residual = fig_residual.add_subplot(111)
                ax_residual.set_xlabel('Iteration')
                ax_residual.set_ylabel('Residual Norm')
                ax_residual.set_yscale('log')
                ax_residual.grid(True, alpha=0.3)
            except ImportError:
                pass
        
        # Initialize reference orientation for flip prevention (computed once from initial state)
        orient_ref = None
        if prevent_flip:
            _, orient_ref = surface_mesh.spherical_triangles_valid_orientation(X_cart, F, None, min_orient=1e-8)
            if verbose >= 1:
                print(f'Initialized reference orientation for flip prevention: {len(orient_ref)} faces')
        
        for iter_num in range(maxiter):
            minimize_area = (iter_num % 2 == 0)
            stepfac = stepfac_area if minimize_area else stepfac_edge
            objective_name = 'area' if minimize_area else 'edge_length'
            
            if verbose:
                print(f'Iteration {iter_num + 1}/{maxiter}: Minimizing {objective_name}')
            
            norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X_cart = X_cart / norms
            
            u_curr = X_cart[:, 0]
            v_curr = X_cart[:, 1]
            w_curr = X_cart[:, 2]
            t_curr, p_curr, _ = kk_cart2sph(u_curr, v_curr, w_curr)
            p_curr = np.mod(p_curr, 2 * np.pi)
            X_flat = X_cart.flatten()
            CHG_vec = np.maximum(seps * np.abs(X_flat), min_step)
            
            if minimize_area:
                # Objectives = triangle areas
                Objectives = np.zeros(nfaces, dtype=np.float64)
                for ix in range(nfaces):
                    v1, v2, v3 = F[ix, 0], F[ix, 1], F[ix, 2]
                    Objectives[ix] = surface_mesh.spherical_triangle_area(
                        t_curr[v1], p_curr[v1], t_curr[v2], p_curr[v2], t_curr[v3], p_curr[v3])
                
                n_obj = nfaces
                indrow = indrow_area
                indcol = indcol_area
                indJ = indJ_area
            else:
                # Current spherical edge lengths
                L_e = np.zeros(n_edges, dtype=np.float64)
                for ex in range(n_edges):
                    va, vb = int(E[ex, 0]), int(E[ex, 1])
                    L_e[ex] = surface_mesh.spherical_edge_length(
                        t_curr[va], p_curr[va], t_curr[vb], p_curr[vb])
                
                # Use fixed target lengths (computed once at start, based on initial configuration)
                # This ensures targets don't shift during optimization
                Objectives = L_e - T_e_fixed  # residuals we want to drive to zero
                
                n_obj = n_edges
                indrow = indrow_edge
                indcol = indcol_edge
                indJ = indJ_edge
            
            # Finite-difference Jacobian
            obj_plus = np.zeros(len(indJ), dtype=np.float64)
            for jx in range(len(indJ)):
                row, col = indrow[jx], indcol[jx]
                vert_idx = col // 3
                coord_idx = col % 3
                
                if minimize_area:
                    v1, v2, v3 = F[row, 0], F[row, 1], F[row, 2]
                    if vert_idx not in [v1, v2, v3]:
                        obj_plus[jx] = Objectives[row]
                    else:
                        X_pert = X_cart.copy()
                        X_pert[vert_idx, coord_idx] = X_flat[col] + CHG_vec[col]
                        norm_pert = np.maximum(np.linalg.norm(X_pert[vert_idx, :]), 1e-12)
                        X_pert[vert_idx, :] /= norm_pert
                        u_pert, v_pert, w_pert = X_pert[:, 0], X_pert[:, 1], X_pert[:, 2]
                        t_pert, p_pert, _ = kk_cart2sph(u_pert, v_pert, w_pert)
                        p_pert = np.mod(p_pert, 2 * np.pi)
                        obj_plus[jx] = surface_mesh.spherical_triangle_area(
                            t_pert[v1], p_pert[v1], t_pert[v2], p_pert[v2], t_pert[v3], p_pert[v3])
                else:
                    va, vb = int(E[row, 0]), int(E[row, 1])
                    if vert_idx not in [va, vb]:
                        obj_plus[jx] = Objectives[row]
                    else:
                        X_pert = X_cart.copy()
                        X_pert[vert_idx, coord_idx] = X_flat[col] + CHG_vec[col]
                        norm_pert = np.maximum(np.linalg.norm(X_pert[vert_idx, :]), 1e-12)
                        X_pert[vert_idx, :] /= norm_pert
                        u_pert, v_pert, w_pert = X_pert[:, 0], X_pert[:, 1], X_pert[:, 2]
                        t_pert, p_pert, _ = kk_cart2sph(u_pert, v_pert, w_pert)
                    p_pert = np.mod(p_pert, 2 * np.pi)
                    obj_plus[jx] = surface_mesh.spherical_edge_length(
                        t_pert[va], p_pert[va], t_pert[vb], p_pert[vb]) - T_e_fixed[row]
            
            if minimize_area:
                Jvals = (obj_plus - Objectives[indrow]) / (CHG_vec[indcol] + 1e-12)
            else:
                Jvals = (obj_plus - Objectives[indrow]) / (CHG_vec[indcol] + 1e-12)
            
            J = sparse.csr_matrix((Jvals, (indrow, indcol)), shape=(n_obj, nvert * 3))
            J.data[np.isnan(J.data)] = 0
            J.data[np.isinf(J.data)] = 0
            
            A = J * J.T
            b = -Objectives
            
            if solver.lower() == 'lu':
                try:
                    dv = spsolve(A, b)
                except Exception:
                    dv, info = gmres(A, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
            else:
                dv, info = gmres(A, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
            
            if info != 0 and verbose and iter_num < 3:
                print(f'  Iter {iter_num}: solver info={info}')
            if np.any(np.isnan(dv)) or np.any(np.isinf(dv)):
                dv = np.zeros_like(b)
            dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
            
            residual_norm = np.linalg.norm(Objectives)
            residuals.append(residual_norm)
            
            dX_flat = stepfac * J.T.dot(dv)
            dX = dX_flat.reshape((nvert, 3))
            
            if prevent_flip:
                X_cart_old = X_cart.copy()
                # Use the reference orientation computed at the start (orient_ref is already set)
                alpha = 1.0
                beta = 0.5
                alpha_min = 1e-6
                max_ls = 40
                step_accept = False
                for _ in range(max_ls):
                    if alpha < alpha_min:
                        if verbose >= 1:
                            print(f'  Line search: no valid step >= alpha_min={alpha_min}; rejecting step')
                        break
                    X_trial = X_cart_old + alpha * dX
                    norms = np.linalg.norm(X_trial, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)
                    X_trial = X_trial / norms
                    valid, _ = surface_mesh.spherical_triangles_valid_orientation(X_trial, F, orient_ref, min_orient=1e-8)
                    if valid:
                        X_cart = X_trial
                        step_accept = True
                        if verbose >= 2:
                            print(f'  Line search: accepted alpha={alpha:.6e}')
                        break
                    alpha *= beta
                if not step_accept:
                    X_cart = X_cart_old
            else:
                X_cart = X_cart + dX
                norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                X_cart = X_cart / norms
            
            if verbose >= 2:
                try:
                    import sys
                    in_jupyter = 'ipykernel' in sys.modules
                    
                    if fig_residual is not None and ax_residual is not None:
                        import matplotlib.pyplot as plt
                        ax_residual.clear()
                        ax_residual.plot(range(1, len(residuals) + 1), residuals, 'b-o', markersize=4)
                        ax_residual.set_xlabel('Iteration')
                        ax_residual.set_ylabel('Residual Norm')
                        ax_residual.set_title(f'Newton Edge-Length Correction ({objective_name})')
                        ax_residual.set_yscale('log')
                        ax_residual.grid(True, alpha=0.3)
                        ax_residual.set_xlim(0, maxiter + 1)
                        fig_residual.canvas.draw()
                    
                    if in_jupyter:
                        plotter_state, image_3d = surface_mesh.plot_state(
                            X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F,
                            count=f'Iter {iter_num + 1}/{maxiter} ({objective_name})',
                            plotter=plotter_state, clear_display=False, return_image=True, two_views=True)
                    else:
                        plotter_state = surface_mesh.plot_state(
                            X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F,
                            count=f'Iter {iter_num + 1}/{maxiter} ({objective_name})',
                            plotter=plotter_state, clear_display=False, return_image=False, two_views=True)
                        image_3d = None
                    
                    if in_jupyter:
                        try:
                            from IPython.display import display, clear_output
                            clear_output(wait=True)
                            if fig_residual is not None:
                                display(fig_residual)
                            if image_3d is not None:
                                display(image_3d)
                        except Exception as e:
                            pass
                    else:
                        import matplotlib.pyplot as plt
                        if fig_residual is not None:
                            plt.pause(0.01)
                except Exception as e:
                    pass
            
            if verbose >= 1:
                print(f'  Residual norm = {residual_norm:.6e} ({objective_name})')
        
        u_final = X_cart[:, 0]
        v_final = X_cart[:, 1]
        w_final = X_cart[:, 2]
        t_final, p_final, _ = kk_cart2sph(u_final, v_final, w_final)
        p_final = np.mod(p_final, 2 * np.pi)
        return t_final, p_final, np.array(residuals)
    
    @staticmethod
    def plot_edges_with_vertex_counts(u, v, w, F, E, edge_target_vertex_count, L_e=None, T_e=None, title='Edges with Vertex Counts'):
        """
        Plot mesh with edges colored/thickness based on target vertex counts.
        Optionally show current vs target edge lengths.
        
        Parameters:
        -----------
        u, v, w : array
            Cartesian coordinates on sphere
        F : array
            Face connectivity
        E : array (n_edges, 2)
            Edge list
        edge_target_vertex_count : array (n_edges,)
            Target vertex counts for each edge
        L_e : array (n_edges,), optional
            Current edge lengths (for comparison plot)
        T_e : array (n_edges,), optional
            Target edge lengths (for comparison plot)
        title : str
            Plot title
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            has_comparison = L_e is not None and T_e is not None
            
            if has_comparison:
                fig = plt.figure(figsize=(16, 5))
            else:
                fig = plt.figure(figsize=(12, 5))
            
            X = np.column_stack([u, v, w])
            
            # Normalize vertex counts for visualization
            counts_norm = (edge_target_vertex_count - np.min(edge_target_vertex_count)) / (np.max(edge_target_vertex_count) - np.min(edge_target_vertex_count) + 1e-10)
            counts_norm = np.clip(counts_norm, 0, 1)
            
            # Left subplot: mesh with edge colors/thickness
            ax1 = fig.add_subplot(131 if has_comparison else 121, projection='3d')
            
            # Plot faces
            ax1.plot_trisurf(u, v, w, triangles=F, alpha=0.3, color='lightblue', edgecolor='none')
            
            # Plot edges with thickness/color based on vertex count
            for ex in range(len(E)):
                va, vb = int(E[ex, 0]), int(E[ex, 1])
                x_line = [X[va, 0], X[vb, 0]]
                y_line = [X[va, 1], X[vb, 1]]
                z_line = [X[va, 2], X[vb, 2]]
                count = edge_target_vertex_count[ex]
                # Color: red for high counts, blue for low counts
                color_val = counts_norm[ex]
                color = plt.cm.RdYlBu_r(color_val)  # Red=high, Blue=low
                linewidth = 1 + 3 * counts_norm[ex]  # Thicker for higher counts
                ax1.plot(x_line, y_line, z_line, color=color, linewidth=linewidth, alpha=0.8)
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(f'{title}\n(Red=high count, Blue=low count)')
            ax1.set_box_aspect([1,1,1])
            
            # Middle subplot: histogram of vertex counts
            ax2 = fig.add_subplot(132 if has_comparison else 122)
            ax2.hist(edge_target_vertex_count, bins=min(20, len(np.unique(edge_target_vertex_count))), edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Target Vertex Count')
            ax2.set_ylabel('Number of Edges')
            ax2.set_title('Distribution of Edge Vertex Counts')
            ax2.grid(True, alpha=0.3)
            
            # Right subplot: comparison of current vs target lengths (if provided)
            if has_comparison:
                ax3 = fig.add_subplot(133)
                length_ratios = L_e / (T_e + 1e-12)
                ax3.scatter(edge_target_vertex_count, length_ratios, alpha=0.6, s=30)
                ax3.axhline(y=1.0, color='r', linestyle='--', label='Target ratio = 1.0')
                ax3.set_xlabel('Target Vertex Count')
                ax3.set_ylabel('Current Length / Target Length')
                ax3.set_title('Length Ratio vs Vertex Count\n(Should be ~1.0 for all edges)')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                
                # Add correlation text
                correlation = np.corrcoef(length_ratios, edge_target_vertex_count / np.mean(edge_target_vertex_count))[0,1]
                ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Warning: Could not create edge visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def plot_shear_heatmap_spherical(t, p, F, shear_per_face=None, title='Shear on spherical parameterization', show_edges=True, cmap='hot_r', return_plotter=False):
        """
        Plot spherical parameterization with face colors = shear (heat map).
        If shear_per_face is None, computes it from (t, p, F).
        
        Parameters:
        -----------
        t, p : array (nvert,)
            Theta and phi on sphere
        F : array (nfaces, 3)
            Face connectivity
        shear_per_face : array (nfaces,), optional
            Per-face shear; if None, computed via compute_shear_spherical
        title : str
            Plot title
        show_edges : bool
            Whether to show mesh edges
        cmap : str
            Colormap name (default 'hot_r': low=dark, high=bright)
        return_plotter : bool
            If True, return (plotter, mesh) without showing
            
        Returns:
        --------
        plotter : pyvista.Plotter (or None)
        summary : dict (if shear_per_face was computed)
        """
        try:
            import pyvista as pv
            if shear_per_face is None:
                shear_per_face, summary = surface_mesh.compute_shear_spherical(t, p, F)
            else:
                summary = {'mean': np.mean(shear_per_face), 'max': np.max(shear_per_face),
                          'rms': np.sqrt(np.mean(shear_per_face**2)), 'total': np.sum(shear_per_face)}
            u, v, w = kk_sph2cart(t, p, np.ones(len(t)))
            X = np.column_stack([u, v, w])
            num_faces = F.shape[0]
            faces_with_n = np.hstack((np.full((num_faces, 1), 3), F))
            cells = faces_with_n.flatten()
            mesh = pv.PolyData(X, cells)
            mesh.cell_data['shear'] = shear_per_face
            in_jupyter = _ensure_pyvista_jupyter_backend()
            plotter = pv.Plotter(notebook=in_jupyter)
            plotter.add_mesh(mesh, scalars='shear', cmap=cmap, show_edges=show_edges, edge_color='black', scalar_bar_args={'title': 'Shear'})
            plotter.add_title(f'{title}\n(mean={summary["mean"]:.4e}, max={summary["max"]:.4e})')
            plotter.set_background('white')
            if return_plotter:
                return (plotter, summary)
            # Return show() result so Jupyter displays the interactive widget
            if in_jupyter:
                return plotter.show(return_viewer=True)
            plotter.show()
            return summary
        except ImportError:
            print("PyVista required for plot_shear_heatmap_spherical")
            return None
        except Exception as e:
            print(f"Warning: plot_shear_heatmap_spherical failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def plot_shear_heatmap_3d(X, F, shear_per_face=None, title='Shear on 3D mesh', show_edges=True, cmap='hot_r', return_plotter=False):
        """
        Plot 3D mesh with face colors = shear (heat map).
        If shear_per_face is None, computes it from (X, F) via compute_shear_3d.
        
        Parameters:
        -----------
        X : array (nvert, 3)
            Vertex coordinates
        F : array (nfaces, 3)
            Face connectivity
        shear_per_face : array (nfaces,), optional
            Per-face shear; if None, computed via compute_shear_3d
        title : str
            Plot title
        show_edges : bool
            Whether to show mesh edges
        cmap : str
            Colormap name (default 'hot_r')
        return_plotter : bool
            If True, return (plotter, mesh) without showing
            
        Returns:
        --------
        plotter : pyvista.Plotter (or None)
        summary : dict (if shear_per_face was computed)
        """
        try:
            import pyvista as pv
            if shear_per_face is None:
                shear_per_face, summary = surface_mesh.compute_shear_3d(X, F)
            else:
                summary = {'mean': np.mean(shear_per_face), 'max': np.max(shear_per_face),
                          'rms': np.sqrt(np.mean(shear_per_face**2)), 'total': np.sum(shear_per_face)}
            num_faces = F.shape[0]
            faces_with_n = np.hstack((np.full((num_faces, 1), 3), F))
            cells = faces_with_n.flatten()
            mesh = pv.PolyData(X, cells)
            mesh.cell_data['shear'] = shear_per_face
            in_jupyter = _ensure_pyvista_jupyter_backend()
            plotter = pv.Plotter(notebook=in_jupyter)
            plotter.add_mesh(mesh, scalars='shear', cmap=cmap, show_edges=show_edges, edge_color='black', scalar_bar_args={'title': 'Shear'})
            plotter.add_title(f'{title}\n(mean={summary["mean"]:.4e}, max={summary["max"]:.4e})')
            plotter.set_background('white')
            if return_plotter:
                return (plotter, summary)
            # Return show() result so Jupyter displays the interactive widget
            if in_jupyter:
                return plotter.show(return_viewer=True)
            plotter.show()
            return summary
        except ImportError:
            print("PyVista required for plot_shear_heatmap_3d")
            return None
        except Exception as e:
            print(f"Warning: plot_shear_heatmap_3d failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def newton_edge_length_only(t, p, F, E, edge_target_vertex_count, stepfac_edge, maxiter, flag=0, filename='untitled', verbose=0, solver='gmres', use_relative_error=False, prevent_flip=True):
        """
        Minimize edge lengths on sphere using Newton's method (edge-length only, no area minimization).
        
        Objective (why we need this after method 1):
        After method 1, spherical triangle areas are ~equal (good). But simplified-mesh edges
        represent different numbers of fine-mesh vertices (e.g. 10 vs 3). If two such edges
        have similar spherical length, placing 10 vs 3 vertices along them causes shear in
        patch-level parameterization. So we want: spherical edge length L_e proportional to
        the number of fine vertices n_e along that edge. Then vertex spacing is uniform.
        
        Target: T_e = (n_e / sum(n)) * L_total_initial so sum(T_e)=L_total_initial.
        We drive L_e -> T_e (absolute) or L_e/T_e -> 1 (relative, if use_relative_error=True).
        
        Correlation (length_ratio vs vertex_count_ratio): should be positive (edges with
        more vertices should be longer). Negative correlation suggests wrong sign, wrong
        alignment of E vs edge_target_vertex_count, or that absolute error dominates large
        edges; try use_relative_error=True.
        
        Parameters:
        -----------
        t, p : array
            Theta and phi
        F : array
            Face connectivity
        E : array (n_edges, 2)
            Edge list (vertex index pairs)
        edge_target_vertex_count : array (n_edges,) or None
            Number of fine-mesh vertices along each simplified edge. If None, uses ones (uniform).
        stepfac_edge : float
            Newton step factor for edge-length minimization
        maxiter : int
            Maximum iterations
        use_relative_error : bool
            If True, minimize sum (L_e/T_e - 1)^2 (relative error); else sum (L_e - T_e)^2.
        prevent_flip : bool, optional
            If True (default), use backtracking line search to avoid triangle flip.
        flag, filename, verbose, solver
            As in newton_steps_cart.
            
        Returns:
        --------
        t, p : array
            Updated spherical coordinates
        residuals : array
            Residual norms per iteration
        """
        import os
        nvert = len(t)
        n_edges = len(E)
        p = np.mod(p, 2 * np.pi)
        
        # Target vertex counts: default uniform if not provided
        if edge_target_vertex_count is None:
            edge_target_vertex_count = np.ones(n_edges, dtype=np.float64)
        else:
            edge_target_vertex_count = np.asarray(edge_target_vertex_count, dtype=np.float64).ravel()
            if len(edge_target_vertex_count) != n_edges:
                raise ValueError('edge_target_vertex_count length must equal number of edges')
        
        count_sum = np.maximum(np.sum(edge_target_vertex_count), 1e-12)
        
        # Verify edge_target_vertex_count values are meaningful
        if verbose >= 1:
            unique_counts = np.unique(edge_target_vertex_count)
            if len(unique_counts) == 1:
                print(f'Warning: All edges have same vertex count ({unique_counts[0]:.1f}). Edge-length optimization will try to equalize lengths.')
            else:
                print(f'Edge target vertex counts vary: min={np.min(edge_target_vertex_count):.1f}, max={np.max(edge_target_vertex_count):.1f}, std={np.std(edge_target_vertex_count):.2f}')
        
        # Cartesian on unit sphere
        u, v, w = kk_sph2cart(t, p, np.ones(nvert))
        X_cart = np.column_stack([u, v, w])
        
        # Compute initial edge lengths to establish fixed target lengths
        # Target lengths should be proportional to edge_target_vertex_count and fixed throughout optimization
        t_initial, p_initial = t.copy(), p.copy()
        p_initial = np.mod(p_initial, 2 * np.pi)
        L_e_initial = np.zeros(n_edges, dtype=np.float64)
        for ex in range(n_edges):
            va, vb = int(E[ex, 0]), int(E[ex, 1])
            L_e_initial[ex] = surface_mesh.spherical_edge_length(
                t_initial[va], p_initial[va], t_initial[vb], p_initial[vb])
        L_total_initial = np.sum(L_e_initial)
        
        # Fixed target lengths: proportional to fine-vertex counts, based on initial total length
        # Formula: T_e = (count_e / sum(counts)) * L_total_initial
        # This means: if count_e is 2x another edge's count, T_e should be 2x that edge's target
        T_e_fixed = (edge_target_vertex_count / count_sum) * L_total_initial
        
        if verbose >= 1:
            print(f'\n{"="*60}')
            print(f'Edge-Length Optimization Setup')
            print(f'{"="*60}')
            print(f'  Number of edges: {n_edges}')
            print(f'  Initial total edge length: {L_total_initial:.6e}')
            print(f'  Edge target vertex counts: min={np.min(edge_target_vertex_count):.1f}, max={np.max(edge_target_vertex_count):.1f}, mean={np.mean(edge_target_vertex_count):.1f}')
            print(f'  Target lengths: min={np.min(T_e_fixed):.6e}, max={np.max(T_e_fixed):.6e}, mean={np.mean(T_e_fixed):.6e}')
            print(f'  Initial edge lengths: min={np.min(L_e_initial):.6e}, max={np.max(L_e_initial):.6e}, mean={np.mean(L_e_initial):.6e}')
            
            # Show example: edges with min and max vertex counts
            min_count_idx = np.argmin(edge_target_vertex_count)
            max_count_idx = np.argmax(edge_target_vertex_count)
            print(f'\n  Example edges:')
            print(f'    Edge {min_count_idx}: vertex_count={edge_target_vertex_count[min_count_idx]:.1f}, target_length={T_e_fixed[min_count_idx]:.6e}, initial_length={L_e_initial[min_count_idx]:.6e}')
            print(f'    Edge {max_count_idx}: vertex_count={edge_target_vertex_count[max_count_idx]:.1f}, target_length={T_e_fixed[max_count_idx]:.6e}, initial_length={L_e_initial[max_count_idx]:.6e}')
            print(f'    Ratio (max/min): vertex_count={edge_target_vertex_count[max_count_idx]/edge_target_vertex_count[min_count_idx]:.2f}x, target_length={T_e_fixed[max_count_idx]/T_e_fixed[min_count_idx]:.2f}x')
            print(f'    Goal: After optimization, length ratio should match vertex_count ratio')
            # Alignment diagnostic: sample edges so (v1,v2,count,T_e,L_e,L_e/T_e) can be checked
            order_by_count = np.argsort(edge_target_vertex_count)
            print(f'\n  Alignment check (E vs edge_target_vertex_count): sample edges by vertex count:')
            for idx in [order_by_count[0], order_by_count[n_edges // 2], order_by_count[-1]]:
                va, vb = int(E[idx, 0]), int(E[idx, 1])
                ratio_i = L_e_initial[idx] / (T_e_fixed[idx] + 1e-12)
                print(f'    eix={idx} (v1={va}, v2={vb}) count={edge_target_vertex_count[idx]:.1f} T_e={T_e_fixed[idx]:.6e} L_e={L_e_initial[idx]:.6e} L_e/T_e={ratio_i:.3f}')
            print(f'  (If high-count edges have L_e/T_e < 1 and low-count > 1, targets are correct; optimization should push correlation positive)')
            if use_relative_error:
                print(f'  Using RELATIVE error objective: minimize sum (L_e/T_e - 1)^2')
            print(f'{"="*60}\n')
        
        str_file = f'data_temp_newton_edge_length_only_{filename}.npz'
        if flag == 0:
            if not os.path.exists(str_file):
                flag = 1
        
        seps = np.sqrt(np.finfo(float).eps)
        min_step = 1e-8
        
        # Edge Jacobian pattern (n_edges x nvert*3)
        if flag == 1:
            JacPat_edge = sparse.lil_matrix((n_edges, nvert * 3))
            for ex in range(n_edges):
                v1, v2 = int(E[ex, 0]), int(E[ex, 1])
                for vert in (v1, v2):
                    JacPat_edge[ex, vert * 3] = 1
                    JacPat_edge[ex, vert * 3 + 1] = 1
                    JacPat_edge[ex, vert * 3 + 2] = 1
            JacPat_edge = JacPat_edge.tocsr()
            i_e, j_e = JacPat_edge.nonzero()
            indJ_edge = np.ravel_multi_index((i_e, j_e), JacPat_edge.shape)
            indrow_edge = i_e.astype(np.uint16)
            indcol_edge = j_e.astype(np.uint16)
            
            np.savez(str_file, JacPat_edge=JacPat_edge, indJ_edge=indJ_edge, 
                     indrow_edge=indrow_edge, indcol_edge=indcol_edge, seps=seps)
        else:
            data = np.load(str_file, allow_pickle=True)
            JacPat_edge = data['JacPat_edge'].item() if hasattr(data['JacPat_edge'], 'item') else data['JacPat_edge']
            indJ_edge = data['indJ_edge']
            indrow_edge = data['indrow_edge']
            indcol_edge = data['indcol_edge']
            seps = float(data['seps'])
        
        # Initialize visualization
        residuals = []
        plotter_state = None
        fig_residual = None
        ax_residual = None
        
        if verbose >= 1:
            try:
                import sys
                in_jupyter = 'ipykernel' in sys.modules
                
                # Use PyVista for 3D mesh visualization (works better in Jupyter)
                if verbose >= 2:
                    try:
                        import pyvista as pv
                        plotter_state = None  # Will be created on first call to plot_state
                    except ImportError:
                        plotter_state = None
                
                # Always create residual plot for verbose >= 1
                import matplotlib.pyplot as plt
                
                # Create figure - plt.show() will work if %matplotlib inline is enabled
                fig_residual = plt.figure('Newton Edge-Length Only Residual Progression', figsize=(8, 5))
                ax_residual = fig_residual.add_subplot(111)
                ax_residual.set_xlabel('Iteration')
                ax_residual.set_ylabel('Residual Norm')
                ax_residual.set_title('Newton Edge-Length Only Residual Progression')
                ax_residual.set_yscale('log')
                ax_residual.grid(True, alpha=0.3)
                
                if verbose >= 1:
                    print('Residual progression plot will be displayed during optimization.')
                    if in_jupyter:
                        print('  (Ensure "%matplotlib inline" is enabled in your Jupyter session)')
            except ImportError:
                pass
        
        # Initialize reference orientation for flip prevention (computed once from initial state)
        orient_ref = None
        if prevent_flip:
            _, orient_ref = surface_mesh.spherical_triangles_valid_orientation(X_cart, F, None, min_orient=1e-8)
            if verbose >= 1:
                print(f'Initialized reference orientation for flip prevention: {len(orient_ref)} faces')
        
        # Begin Newton iterations
        for iter_num in range(maxiter):
            if verbose:
                print(f'Iteration {iter_num + 1}/{maxiter}: Minimizing edge_length')
            
            norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X_cart = X_cart / norms
            
            u_curr = X_cart[:, 0]
            v_curr = X_cart[:, 1]
            w_curr = X_cart[:, 2]
            t_curr, p_curr, _ = kk_cart2sph(u_curr, v_curr, w_curr)
            p_curr = np.mod(p_curr, 2 * np.pi)
            X_flat = X_cart.flatten()
            CHG_vec = np.maximum(seps * np.abs(X_flat), min_step)
            
            # Current spherical edge lengths
            L_e = np.zeros(n_edges, dtype=np.float64)
            for ex in range(n_edges):
                va, vb = int(E[ex, 0]), int(E[ex, 1])
                L_e[ex] = surface_mesh.spherical_edge_length(
                    t_curr[va], p_curr[va], t_curr[vb], p_curr[vb])
            
            # Use fixed target lengths (computed once at start, based on initial configuration)
            if use_relative_error:
                # Minimize sum (L_e/T_e - 1)^2 so all edges weighted equally in relative terms
                Objectives = (L_e / (T_e_fixed + 1e-12)) - 1.0
            else:
                Objectives = L_e - T_e_fixed  # residuals we want to drive to zero
            
            # Diagnostic output for first iteration
            if verbose >= 1 and iter_num == 0:
                L_total_curr = np.sum(L_e)
                # Compute ratios: actual length / target length (should approach 1.0)
                length_ratios = L_e / (T_e_fixed + 1e-12)
                vertex_count_ratios = edge_target_vertex_count / (np.mean(edge_target_vertex_count) + 1e-12)
                
                print(f'\n  {"="*50}')
                print(f'  Initial Edge-Length Statistics:')
                print(f'  {"="*50}')
                print(f'    Current total spherical edge length: {L_total_curr:.6e}')
                print(f'    Mean current edge length: {np.mean(L_e):.6e}')
                print(f'    Std current edge length: {np.std(L_e):.6e}')
                print(f'    Mean target length (fixed): {np.mean(T_e_fixed):.6e}')
                print(f'    Mean residual (L_e - T_e_fixed): {np.mean(Objectives):.6e}')
                print(f'    Max |residual|: {np.max(np.abs(Objectives)):.6e}')
                print(f'    Residual norm: {np.linalg.norm(Objectives):.6e}')
                print(f'    Relative error: {np.linalg.norm(Objectives) / (np.mean(T_e_fixed) * n_edges):.6e}')
                print(f'\n  Edge Length Ratios (L_e/T_e):')
                print(f'    min={np.min(length_ratios):.3f}, max={np.max(length_ratios):.3f}, mean={np.mean(length_ratios):.3f}, std={np.std(length_ratios):.3f}')
                print(f'    (Target: all ratios = 1.0, std = 0.0)')
                print(f'\n  Target Vertex Count Ratios:')
                print(f'    min={np.min(vertex_count_ratios):.3f}, max={np.max(vertex_count_ratios):.3f}')
                correlation = np.corrcoef(length_ratios, vertex_count_ratios)[0,1] if len(length_ratios) > 1 else 0.0
                print(f'\n  Correlation (length_ratio vs vertex_count_ratio): {correlation:.3f}')
                print(f'    (Target: ~1.0, meaning edges with more vertices are longer)')
                print(f'    (Current: {"GOOD" if correlation > 0.7 else "POOR - optimization should improve this"})')
                print(f'  {"="*50}\n')
            
            # Finite-difference Jacobian
            obj_plus = np.zeros(len(indJ_edge), dtype=np.float64)
            for jx in range(len(indJ_edge)):
                row, col = indrow_edge[jx], indcol_edge[jx]
                vert_idx = col // 3
                coord_idx = col % 3
                va, vb = int(E[row, 0]), int(E[row, 1])
                
                if vert_idx not in [va, vb]:
                    obj_plus[jx] = Objectives[row]
                else:
                    X_pert = X_cart.copy()
                    X_pert[vert_idx, coord_idx] = X_flat[col] + CHG_vec[col]
                    norm_pert = np.maximum(np.linalg.norm(X_pert[vert_idx, :]), 1e-12)
                    X_pert[vert_idx, :] /= norm_pert
                    u_pert, v_pert, w_pert = X_pert[:, 0], X_pert[:, 1], X_pert[:, 2]
                    t_pert, p_pert, _ = kk_cart2sph(u_pert, v_pert, w_pert)
                    p_pert = np.mod(p_pert, 2 * np.pi)
                    L_pert = surface_mesh.spherical_edge_length(
                        t_pert[va], p_pert[va], t_pert[vb], p_pert[vb])
                    if use_relative_error:
                        obj_plus[jx] = (L_pert / (T_e_fixed[row] + 1e-12)) - 1.0
                    else:
                        obj_plus[jx] = L_pert - T_e_fixed[row]
            
            Jvals = (obj_plus - Objectives[indrow_edge]) / (CHG_vec[indcol_edge] + 1e-12)
            
            J = sparse.csr_matrix((Jvals, (indrow_edge, indcol_edge)), shape=(n_edges, nvert * 3))
            J.data[np.isnan(J.data)] = 0
            J.data[np.isinf(J.data)] = 0
            
            # Verify Jacobian is non-zero
            if verbose >= 1 and iter_num < 3:
                jac_nnz = J.nnz
                jac_max = np.max(np.abs(J.data)) if len(J.data) > 0 else 0
                jac_mean = np.mean(np.abs(J.data)) if len(J.data) > 0 else 0
                print(f'  Jacobian: {jac_nnz} nonzeros, max|J|={jac_max:.6e}, mean|J|={jac_mean:.6e}')
                if jac_max < 1e-10:
                    print(f'  WARNING: Jacobian is nearly zero! Optimization may not work.')
            
            A = J * J.T
            b = -Objectives
            
            # Verify system is well-conditioned
            if verbose >= 1 and iter_num < 3:
                b_norm = np.linalg.norm(b)
                print(f'  System: ||b||={b_norm:.6e}, ||Objectives||={np.linalg.norm(Objectives):.6e}')
                if b_norm < 1e-10:
                    print(f'  WARNING: Right-hand side is nearly zero! Already converged or objective is wrong.')
            
            if solver.lower() == 'lu':
                try:
                    dv = spsolve(A, b)
                except Exception:
                    dv, info = gmres(A, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
            else:
                dv, info = gmres(A, b, restart=1, maxiter=50, rtol=1e-6, atol=1e-8)
            
            # Handle solver results
            if info < 0:
                if verbose:
                    print(f'Warning: Solver failed at iteration {iter_num} (info={info}), using zero step')
                dv = np.zeros_like(b)
            elif info > 0:
                if verbose >= 1 and iter_num < 3:
                    print(f'  Iter {iter_num}: GMRES reached maxiter ({info}) without converging, but using solution')
            
            # Clean up any NaN/Inf values
            if np.any(np.isnan(dv)) or np.any(np.isinf(dv)):
                if verbose:
                    print(f'Warning: Solver returned NaN/Inf at iteration {iter_num}, using zero step')
                dv = np.zeros_like(b)
            else:
                dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Track residual
            residual_norm = np.linalg.norm(Objectives)
            residuals.append(residual_norm)
            
            # Print residual at every iteration (similar to optimization_method 1)
            if verbose >= 1:
                print(f'  Iter {iter_num}: ||EdgeResiduals||={residual_norm:.6e}')
            
            if verbose >= 1 and iter_num < 3:
                dv_norm = np.linalg.norm(dv)
                b_norm = np.linalg.norm(b)
                if dv_norm > 0:
                    residual_check = np.linalg.norm(A.dot(dv) - b) / b_norm if b_norm > 0 else 0
                    print(f'  Iter {iter_num}: ||dv||={dv_norm:.6e}, ||EdgeResiduals||={residual_norm:.6e}, rel_residual={residual_check:.6e}')
                    print(f'  Jacobian: shape={J.shape}, nnz={J.nnz}, max|J|={np.max(np.abs(J.data)):.6e}')
                    print(f'  System: A shape={A.shape}, ||b||={b_norm:.6e}, ||A||_F={np.sqrt(np.sum(A.data**2)):.6e}')
                # Descent check: step should reduce objective; dv'*Objectives < 0 means descent
                descent = np.dot(dv, Objectives)
                if descent > 0:
                    print(f'  WARNING: Step is not a descent direction (dv''*Objectives={descent:.6e} > 0); check sign or formulation')
                else:
                    print(f'  Descent: dv''*Objectives={descent:.6e} (negative = good)')
            
            # Take Newton step: X = X + stepfac_edge * J' * dv
            dX_flat = stepfac_edge * J.T.dot(dv)
            dX = dX_flat.reshape((nvert, 3))
            dX_norm = np.linalg.norm(dX)
            
            # Diagnostic: verify step is being taken
            if verbose >= 1 and iter_num < 3:
                print(f'  Step size ||dX|| = {dX_norm:.6e}, stepfac_edge = {stepfac_edge}')
                max_dX = np.max(np.abs(dX))
                print(f'  Max |dX| component = {max_dX:.6e}')
            
            X_cart_old = X_cart.copy()
            
            if prevent_flip:
                # Use the reference orientation computed at the start (orient_ref is already set)
                alpha = 1.0
                beta = 0.5
                alpha_min = 1e-6
                max_ls = 40
                step_accept = False
                for _ in range(max_ls):
                    if alpha < alpha_min:
                        if verbose >= 1:
                            print(f'  Line search: no valid step >= alpha_min={alpha_min}; rejecting step')
                        break
                    X_trial = X_cart_old + alpha * dX
                    norms = np.linalg.norm(X_trial, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)
                    X_trial = X_trial / norms
                    valid, _ = surface_mesh.spherical_triangles_valid_orientation(X_trial, F, orient_ref, min_orient=1e-8)
                    if valid:
                        X_cart = X_trial
                        step_accept = True
                        if verbose >= 2:
                            print(f'  Line search: accepted alpha={alpha:.6e}')
                        break
                    alpha *= beta
                if not step_accept:
                    X_cart = X_cart_old
            else:
                X_cart = X_cart + dX
                norms = np.linalg.norm(X_cart, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                X_cart = X_cart / norms
            
            # Check if vertices actually moved
            vertex_displacement = np.linalg.norm(X_cart - X_cart_old, axis=1)
            max_displacement = np.max(vertex_displacement)
            if verbose >= 1 and iter_num < 3:
                print(f'  Max vertex displacement = {max_displacement:.6e}')
                if max_displacement < 1e-10:
                    print(f'  WARNING: Vertices not moving! Step may be too small or Jacobian may be wrong.')
            
            # Update visualization if verbose >= 1 (residual plot always, 3D plot if verbose >= 2)
            if verbose >= 1:
                try:
                    import sys
                    in_jupyter = 'ipykernel' in sys.modules
                    
                    # Always update residual plot for verbose >= 1
                    if fig_residual is not None and ax_residual is not None:
                        import matplotlib.pyplot as plt
                        ax_residual.clear()
                        ax_residual.plot(range(1, len(residuals) + 1), residuals, 'b-o', markersize=4, linewidth=2)
                        ax_residual.set_xlabel('Iteration', fontsize=12)
                        ax_residual.set_ylabel('Residual Norm ||L_e - T_e||', fontsize=12)
                        ax_residual.set_title(f'Newton Edge-Length Only Residual Progression (Iter {iter_num + 1}/{maxiter})', fontsize=12)
                        ax_residual.set_yscale('log')
                        ax_residual.grid(True, alpha=0.3)
                        ax_residual.set_xlim(0, maxiter + 1)
                        if len(residuals) > 0:
                            y_min = max(np.min(residuals) * 0.5, 1e-10)
                            y_max = max(np.max(residuals) * 1.5, 1e-6)
                            ax_residual.set_ylim(y_min, y_max)
                        fig_residual.canvas.draw()
                    
                    # Update 3D mesh plot with edge visualization at key iterations
                    fig_edges_iter = None
                    if verbose >= 1 and (iter_num == 0 or (iter_num + 1) % max(1, maxiter // 5) == 0 or iter_num == maxiter - 1):
                        # Plot edges with vertex counts for visual verification (with current vs target)
                        try:
                            import matplotlib.pyplot as plt
                            fig_edges_iter = surface_mesh.plot_edges_with_vertex_counts(
                                X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F, E, edge_target_vertex_count,
                                L_e=L_e, T_e=T_e_fixed,
                                title=f'Iter {iter_num + 1}/{maxiter}: Edges by Vertex Count')
                        except Exception as e:
                            if verbose >= 1:
                                pass  # Don't spam errors
                    
                    # Update 3D mesh plot only if verbose >= 2
                    image_3d = None
                    if verbose >= 2:
                        # For Jupyter, get image data instead of displaying immediately
                        if in_jupyter:
                            plotter_state, image_3d = surface_mesh.plot_state(
                                X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F,
                                count=f'Iteration {iter_num + 1}/{maxiter} (edge-length)',
                                plotter=plotter_state, clear_display=False, return_image=True, two_views=True)
                        else:
                            plotter_state = surface_mesh.plot_state(
                                X_cart[:, 0], X_cart[:, 1], X_cart[:, 2], F,
                                count=f'Iteration {iter_num + 1}/{maxiter} (edge-length)',
                                plotter=plotter_state, clear_display=False, return_image=False, two_views=True)
                            image_3d = None
                    
                    # Display plots in Jupyter using IPython.display (same as other methods)
                    if in_jupyter and (iter_num == 0 or (iter_num + 1) % max(1, maxiter // 5) == 0 or iter_num == maxiter - 1):
                        try:
                            from IPython.display import display, clear_output
                            clear_output(wait=True)
                            if fig_residual is not None:
                                display(fig_residual)
                            if fig_edges_iter is not None:
                                display(fig_edges_iter)
                            if image_3d is not None:
                                display(image_3d)
                        except Exception as e:
                            if verbose > 2:
                                print(f"Warning: Display failed: {e}")
                    elif not in_jupyter:
                        # Non-Jupyter: use plt.show() or plt.pause()
                        import matplotlib.pyplot as plt
                        if iter_num == 0 or (iter_num + 1) % max(1, maxiter // 5) == 0 or iter_num == maxiter - 1:
                            if fig_residual is not None:
                                plt.show(block=False)
                            if fig_edges_iter is not None:
                                plt.show(block=False)
                        else:
                            if fig_residual is not None:
                                plt.pause(0.01)
                except Exception as e:
                    if verbose >= 1:
                        print(f"Warning: Plotting failed: {e}")
                    import traceback
                    if verbose > 2:
                        traceback.print_exc()
            
            if verbose >= 1:
                # Show progress every iteration with edge-length correlation
                if iter_num == 0 or (iter_num + 1) % max(1, maxiter // 10) == 0 or iter_num == maxiter - 1:
                    # Compute current correlation between edge lengths and vertex counts
                    length_ratios = L_e / (T_e_fixed + 1e-12)
                    vertex_count_ratios = edge_target_vertex_count / (np.mean(edge_target_vertex_count) + 1e-12)
                    correlation = np.corrcoef(length_ratios, vertex_count_ratios)[0,1] if len(length_ratios) > 1 else 0.0
                    
                    print(f'Iteration {iter_num + 1}/{maxiter}: Residual norm = {residual_norm:.6e} (edge-length)')
                    if len(residuals) > 1:
                        reduction = residuals[0] / residual_norm if residual_norm > 0 else 1.0
                        print(f'  Reduction factor: {reduction:.2f}x from initial')
                    print(f'  Length/vertex_count correlation: {correlation:.3f} (target: ~1.0, higher is better)')
                    print(f'  Mean length ratio (L/T): {np.mean(length_ratios):.3f} (target: 1.0)')
        
        # Convert final Cartesian coordinates back to spherical
        u_final = X_cart[:, 0]
        v_final = X_cart[:, 1]
        w_final = X_cart[:, 2]
        t_final, p_final, r_final = kk_cart2sph(u_final, v_final, w_final)
        p_final = np.mod(p_final, 2 * np.pi)
        
        # Final summary and ensure residual plot is visible
        if verbose >= 1:
            # Compute final edge lengths and correlation
            u_final_temp = X_cart[:, 0]
            v_final_temp = X_cart[:, 1]
            w_final_temp = X_cart[:, 2]
            t_final_temp, p_final_temp, _ = kk_cart2sph(u_final_temp, v_final_temp, w_final_temp)
            p_final_temp = np.mod(p_final_temp, 2 * np.pi)
            L_e_final = np.zeros(n_edges, dtype=np.float64)
            for ex in range(n_edges):
                va, vb = int(E[ex, 0]), int(E[ex, 1])
                L_e_final[ex] = surface_mesh.spherical_edge_length(
                    t_final_temp[va], p_final_temp[va], t_final_temp[vb], p_final_temp[vb])
            length_ratios_final = L_e_final / (T_e_fixed + 1e-12)
            vertex_count_ratios = edge_target_vertex_count / (np.mean(edge_target_vertex_count) + 1e-12)
            correlation_final = np.corrcoef(length_ratios_final, vertex_count_ratios)[0,1] if len(length_ratios_final) > 1 else 0.0
            
            print(f'\n{"="*60}')
            print(f'Edge-Length Optimization Complete')
            print(f'{"="*60}')
            if len(residuals) > 0:
                print(f'  Initial residual: {residuals[0]:.6e}')
                print(f'  Final residual: {residuals[-1]:.6e}')
                if residuals[0] > 0:
                    reduction = residuals[0] / residuals[-1]
                    print(f'  Total reduction: {reduction:.2f}x')
                print(f'  Final length/vertex_count correlation: {correlation_final:.3f} (target: ~1.0)')
                print(f'  Final mean length ratio (L/T): {np.mean(length_ratios_final):.3f} (target: 1.0)')
                print(f'  Final std of length ratios: {np.std(length_ratios_final):.3f} (lower is better)')
                if len(residuals) <= 10:
                    print(f'  Residual progression: {residuals}')
                else:
                    print(f'  Residual progression (first 5): {residuals[:5]}')
                    print(f'  Residual progression (last 5): {residuals[-5:]}')
            
            # Ensure residual plot is displayed
            if fig_residual is not None and ax_residual is not None:
                try:
                    import matplotlib.pyplot as plt
                    import sys
                    in_jupyter = 'ipykernel' in sys.modules
                    
                    # Final update of plot
                    ax_residual.clear()
                    ax_residual.plot(range(1, len(residuals) + 1), residuals, 'b-o', markersize=4, linewidth=2)
                    ax_residual.set_xlabel('Iteration', fontsize=12)
                    ax_residual.set_ylabel('Residual Norm ||L_e - T_e||', fontsize=12)
                    ax_residual.set_title(f'Newton Edge-Length Only Residual Progression (Final)', fontsize=12)
                    ax_residual.set_yscale('log')
                    ax_residual.grid(True, alpha=0.3)
                    ax_residual.set_xlim(0, maxiter + 1)
                    if len(residuals) > 0:
                        y_min = max(np.min(residuals) * 0.5, 1e-10)
                        y_max = max(np.max(residuals) * 1.5, 1e-6)
                        ax_residual.set_ylim(y_min, y_max)
                    fig_residual.canvas.draw()
                    
                    # Plot edges with vertex counts for visual verification (with comparison)
                    fig_comparison = None
                    fig_edges = None
                    if verbose >= 1:
                        try:
                            # Compute final edge lengths for comparison
                            L_e_final_plot = np.zeros(n_edges, dtype=np.float64)
                            for ex in range(n_edges):
                                va, vb = int(E[ex, 0]), int(E[ex, 1])
                                L_e_final_plot[ex] = surface_mesh.spherical_edge_length(
                                    t_final[va], p_final[va], t_final[vb], p_final[vb])
                            
                            # Compute initial edge lengths for before/after comparison
                            u_initial, v_initial, w_initial = kk_sph2cart(t_initial, p_initial, np.ones(nvert))
                            L_e_initial_plot = L_e_initial.copy()
                            
                            # Create before/after comparison figure
                            fig_comparison = plt.figure(figsize=(16, 6))
                            
                            # Before plot
                            ax_before = fig_comparison.add_subplot(131, projection='3d')
                            X_before = np.column_stack([u_initial, v_initial, w_initial])
                            counts_norm = (edge_target_vertex_count - np.min(edge_target_vertex_count)) / (np.max(edge_target_vertex_count) - np.min(edge_target_vertex_count) + 1e-10)
                            counts_norm = np.clip(counts_norm, 0, 1)
                            ax_before.plot_trisurf(u_initial, v_initial, w_initial, triangles=F, alpha=0.3, color='lightblue', edgecolor='none')
                            for ex in range(len(E)):
                                va, vb = int(E[ex, 0]), int(E[ex, 1])
                                x_line = [X_before[va, 0], X_before[vb, 0]]
                                y_line = [X_before[va, 1], X_before[vb, 1]]
                                z_line = [X_before[va, 2], X_before[vb, 2]]
                                color_val = counts_norm[ex]
                                color = plt.cm.RdYlBu_r(color_val)
                                linewidth = 1 + 3 * counts_norm[ex]
                                ax_before.plot(x_line, y_line, z_line, color=color, linewidth=linewidth, alpha=0.8)
                            ax_before.set_title('BEFORE: Edges by Vertex Count', fontsize=11)
                            ax_before.set_box_aspect([1,1,1])
                            
                            # After plot
                            ax_after = fig_comparison.add_subplot(132, projection='3d')
                            X_after = np.column_stack([u_final, v_final, w_final])
                            ax_after.plot_trisurf(u_final, v_final, w_final, triangles=F, alpha=0.3, color='lightblue', edgecolor='none')
                            for ex in range(len(E)):
                                va, vb = int(E[ex, 0]), int(E[ex, 1])
                                x_line = [X_after[va, 0], X_after[vb, 0]]
                                y_line = [X_after[va, 1], X_after[vb, 1]]
                                z_line = [X_after[va, 2], X_after[vb, 2]]
                                color_val = counts_norm[ex]
                                color = plt.cm.RdYlBu_r(color_val)
                                linewidth = 1 + 3 * counts_norm[ex]
                                ax_after.plot(x_line, y_line, z_line, color=color, linewidth=linewidth, alpha=0.8)
                            ax_after.set_title('AFTER: Edges by Vertex Count', fontsize=11)
                            ax_after.set_box_aspect([1,1,1])
                            
                            # Comparison plot: length ratios
                            ax_comp = fig_comparison.add_subplot(133)
                            length_ratios_before = L_e_initial_plot / (T_e_fixed + 1e-12)
                            length_ratios_after = L_e_final_plot / (T_e_fixed + 1e-12)
                            ax_comp.scatter(edge_target_vertex_count, length_ratios_before, alpha=0.5, s=30, label='Before', color='blue')
                            ax_comp.scatter(edge_target_vertex_count, length_ratios_after, alpha=0.5, s=30, label='After', color='red')
                            ax_comp.axhline(y=1.0, color='k', linestyle='--', label='Target = 1.0')
                            ax_comp.set_xlabel('Target Vertex Count')
                            ax_comp.set_ylabel('Length Ratio (L/T)')
                            ax_comp.set_title('Before/After: Length Ratios\n(Should cluster around 1.0)', fontsize=11)
                            ax_comp.grid(True, alpha=0.3)
                            ax_comp.legend()
                            
                            corr_before = np.corrcoef(length_ratios_before, edge_target_vertex_count / np.mean(edge_target_vertex_count))[0,1]
                            corr_after = np.corrcoef(length_ratios_after, edge_target_vertex_count / np.mean(edge_target_vertex_count))[0,1]
                            ax_comp.text(0.05, 0.95, f'Corr Before: {corr_before:.3f}\nCorr After: {corr_after:.3f}', 
                                       transform=ax_comp.transAxes, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                            
                            plt.tight_layout()
                            
                            # Also create the detailed edge plot
                            fig_edges = surface_mesh.plot_edges_with_vertex_counts(
                                u_final, v_final, w_final, F, E, edge_target_vertex_count,
                                L_e=L_e_final_plot, T_e=T_e_fixed,
                                title='Final: Edges Colored by Vertex Count')
                        except Exception as e:
                            if verbose >= 1:
                                print(f'  Note: Edge visualization plot failed: {e}')
                                import traceback
                                if verbose > 2:
                                    traceback.print_exc()
                    
                    # Display all final plots using IPython.display for Jupyter (same as other methods)
                    if in_jupyter:
                        try:
                            from IPython.display import display, clear_output
                            clear_output(wait=True)
                            if fig_residual is not None:
                                display(fig_residual)
                            if fig_comparison is not None:
                                display(fig_comparison)
                            if fig_edges is not None:
                                display(fig_edges)
                            if verbose >= 1:
                                print(f'  All plots displayed above.')
                        except Exception as e:
                            if verbose >= 1:
                                print(f'  Warning: Display failed: {e}')
                    else:
                        # Non-Jupyter: use plt.show()
                        if fig_residual is not None:
                            plt.show(block=False)
                        if fig_comparison is not None:
                            plt.show(block=False)
                        if fig_edges is not None:
                            plt.show(block=False)
                except Exception as e:
                    if verbose >= 1:
                        print(f'  Warning: Final plot display failed: {e}')
                    import traceback
                    if verbose > 2:
                        traceback.print_exc()
                except Exception as e:
                    if verbose >= 1:
                        print(f"Warning: Final plot display failed: {e}")
                        import traceback
                        traceback.print_exc()
        
        if verbose:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(u_final, v_final, w_final, triangles=F)
                plt.show()
            except ImportError:
                pass
        
        return t_final, p_final, np.array(residuals)
    
    @staticmethod
    def plot_state(u, v, w, F, count=None, flag=2, az=None, el=None, fig=None, ax=None, plotter=None, clear_display=True, return_image=False, two_views=False):
        """
        Plot state of spherical parameterization
        
        Parameters:
        -----------
        u, v, w : array
            Cartesian coordinates on sphere
        F : array
            Face connectivity
        count : int, optional
            Iteration count for title
        flag : int, optional
            Zoom factor
        az, el : float, optional
            Azimuth and elevation for view
        fig : matplotlib figure, optional
            Existing figure to update (if None, creates new) - deprecated, kept for compatibility
        ax : matplotlib axis, optional
            Existing axis to update (if None, creates new) - deprecated, kept for compatibility
        plotter : pyvista.Plotter, optional
            Existing plotter to update (if None, creates new)
        clear_display : bool, optional
            Whether to clear output before displaying (default True). Set to False when displaying multiple figures.
        return_image : bool, optional
            If True, return the image data instead of displaying it (default False). Only works in Jupyter.
        two_views : bool, optional
            If True and return_image=True, show two views side-by-side (front and back of sphere).
            
        Returns:
        --------
        plotter : pyvista Plotter (or None if pyvista not available)
        image_data : IPython Image object (only if return_image=True and in Jupyter)
        """
        try:
            import pyvista as pv
            
            X = np.column_stack([u, v, w])
            t, p, r = kk_cart2sph(u, v, w)
            
            # Prepare faces with number of vertices
            num_faces = F.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), F))
            cells = faces_with_n_vertices.flatten()
            
            # Create mesh
            mesh = pv.PolyData(X, cells)
            
            # Create vertex colors based on theta
            t_norm = t / np.max(t) if np.max(t) > 0 else t
            # Store theta values for coloring
            mesh['theta'] = t_norm
            
            # Use existing plotter or create new one
            import matplotlib.cm as cm
            hsv_cmap = cm.get_cmap('hsv')
            
            # Ensure PyVista backend is initialized (fixes "Loading..." spinner issue)
            in_jupyter = _ensure_pyvista_jupyter_backend()
            
            if plotter is None:
                # Create new plotter
                if in_jupyter:
                    # For Jupyter, use inline plotter
                    # If return_image is True, we'll use off_screen for screenshots
                    # Otherwise, use interactive backend
                    if return_image:
                        plotter = pv.Plotter(off_screen=True)
                    else:
                        plotter = pv.Plotter(notebook=in_jupyter)
                else:
                    plotter = pv.Plotter(notebook=in_jupyter)
                
                plotter.add_mesh(mesh, scalars='theta', cmap=hsv_cmap, show_edges=True, edge_color='black', opacity=1.0)
                
                # Set camera position
                if az is not None and el is not None:
                    plotter.camera.elevation = el
                    plotter.camera.azimuth = az
                else:
                    plotter.camera.elevation = 10
                    plotter.camera.azimuth = 45
                
                # Set title
                if count is not None:
                    plotter.add_text(str(count), font_size=12)
                else:
                    plotter.add_text('State of bijective mapping', font_size=12)
                
                # Show plotter - for Jupyter, use interactive backend or static images
                if in_jupyter:
                    try:
                        from IPython.display import Image, display
                        import io
                        import PIL.Image
                        
                        # If return_image is True, we need a static image (screenshot)
                        # Otherwise, use interactive display
                        if return_image:
                            import time
                            
                            def _capture_view(az_deg, el_deg):
                                plotter.camera.azimuth = az_deg
                                plotter.camera.elevation = el_deg
                                plotter.render()
                                time.sleep(0.2)
                                try:
                                    arr = plotter.screenshot(transparent_background=False, window_size=[800, 600])
                                except Exception:
                                    arr = plotter.screenshot(transparent_background=False)
                                if arr is None or arr.size == 0:
                                    raise ValueError("Screenshot failed")
                                if len(arr.shape) == 3 and arr.shape[2] == 4:
                                    arr = arr[:, :, :3]
                                if arr.dtype != np.uint8:
                                    arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                                return arr
                            
                            if two_views:
                                img1 = _capture_view(45, 10)
                                img2 = _capture_view(225, 10)
                                img_array = np.hstack([img1, img2])
                            else:
                                plotter.render()
                                time.sleep(0.2)
                                try:
                                    img_array = plotter.screenshot(transparent_background=False, window_size=[800, 600])
                                except Exception as screenshot_error:
                                    try:
                                        img_array = plotter.screenshot(transparent_background=False)
                                    except:
                                        raise ValueError(f"Screenshot failed: {screenshot_error}")
                            
                            # Validate screenshot was captured
                            if img_array is None:
                                raise ValueError("Screenshot returned None")
                            
                            if img_array.size == 0:
                                raise ValueError("Screenshot returned empty array")
                            
                            # Check array shape
                            if not isinstance(img_array, np.ndarray):
                                raise ValueError(f"Screenshot returned non-array type: {type(img_array)}")
                            
                            # Ensure array is uint8 and has correct shape
                            if img_array.dtype != np.uint8:
                                if img_array.max() <= 1.0:
                                    img_array = (img_array * 255).astype(np.uint8)
                                else:
                                    img_array = img_array.astype(np.uint8)
                            
                            # Validate shape - should be (height, width, channels)
                            if len(img_array.shape) != 3:
                                raise ValueError(f"Invalid image shape: {img_array.shape}, expected 3D array (height, width, channels)")
                            
                            if img_array.shape[2] not in [3, 4]:
                                raise ValueError(f"Invalid number of channels: {img_array.shape[2]}, expected 3 (RGB) or 4 (RGBA)")
                            
                            # Ensure minimum dimensions
                            if img_array.shape[0] < 10 or img_array.shape[1] < 10:
                                raise ValueError(f"Image dimensions too small: {img_array.shape}")
                            
                            # Convert to RGB if RGBA
                            if img_array.shape[2] == 4:
                                img_array = img_array[:, :, :3]
                            
                            # Create PIL Image - this validates the array
                            try:
                                pil_img = PIL.Image.fromarray(img_array, mode='RGB')
                            except Exception as pil_error:
                                raise ValueError(f"Failed to create PIL Image: {pil_error}, array shape: {img_array.shape}, dtype: {img_array.dtype}")
                            
                            # Convert to bytes for IPython display
                            img_bytes = io.BytesIO()
                            try:
                                pil_img.save(img_bytes, format='PNG')
                            except Exception as save_error:
                                raise ValueError(f"Failed to save PIL Image to bytes: {save_error}")
                            
                            img_bytes.seek(0)
                            img_data = img_bytes.getvalue()
                            
                            # Validate we have image data
                            if len(img_data) == 0:
                                raise ValueError("Image bytes are empty")
                            
                            # Create IPython Image object
                            try:
                                image_obj = Image(img_data)
                            except Exception as image_error:
                                raise ValueError(f"Failed to create IPython Image: {image_error}")
                            
                            return plotter, image_obj
                        else:
                            # Use interactive display
                            if clear_display:
                                from IPython.display import clear_output
                                clear_output(wait=True)
                            # Use show with static backend for consistent display
                            result = plotter.show(return_viewer=False)
                            if result is not None:
                                display(result)
                    except Exception as e:
                        # Fallback: try showing with interactive backend
                        if not return_image:
                            try:
                                # Recreate plotter without off_screen for fallback
                                plotter_fallback = pv.Plotter(notebook=in_jupyter)
                                plotter_fallback.add_mesh(mesh, scalars='theta', cmap=hsv_cmap, show_edges=True, edge_color='black', opacity=1.0)
                                if count is not None:
                                    plotter_fallback.add_text(str(count), font_size=12)
                                result = plotter_fallback.show(return_viewer=False)
                                if result is not None:
                                    from IPython.display import display
                                    display(result)
                            except Exception as fallback_error:
                                if verbose > 2:
                                    print(f"Warning: Plot display failed: {e}, fallback also failed: {fallback_error}")
                        else:
                            return plotter, None
                else:
                    plotter.show(auto_close=False)
            else:
                # Update existing plotter: clear and add new mesh
                plotter.clear()
                
                # Ensure plotter is in off-screen mode if we need screenshots
                if in_jupyter and return_image:
                    # Check if plotter needs to be in off-screen mode
                    # If it's not, we can't take screenshots, so we'll need to handle this
                    pass  # Assume it's already in the right mode from initial creation
                
                # Add new mesh
                plotter.add_mesh(mesh, scalars='theta', cmap=hsv_cmap, show_edges=True, edge_color='black', opacity=1.0)
                
                # Update title
                if count is not None:
                    plotter.add_text(str(count), font_size=12)
                else:
                    plotter.add_text('State of bijective mapping', font_size=12)
                
                # Force render and update display
                plotter.render()
                
                # For Jupyter, update display by capturing screenshot and displaying
                if in_jupyter:
                    try:
                        from IPython.display import Image, display, clear_output
                        import io
                        import PIL.Image
                        
                        import time
                        
                        def _capture_view_update(az_deg, el_deg):
                            plotter.camera.azimuth = az_deg
                            plotter.camera.elevation = el_deg
                            plotter.render()
                            time.sleep(0.2)
                            try:
                                arr = plotter.screenshot(transparent_background=False, window_size=[800, 600])
                            except Exception:
                                arr = plotter.screenshot(transparent_background=False)
                            if arr is None or arr.size == 0:
                                raise ValueError("Screenshot failed")
                            if len(arr.shape) == 3 and arr.shape[2] == 4:
                                arr = arr[:, :, :3]
                            if arr.dtype != np.uint8:
                                arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                            return arr
                        
                        if two_views:
                            img1 = _capture_view_update(45, 10)
                            img2 = _capture_view_update(225, 10)
                            img_array = np.hstack([img1, img2])
                        else:
                            plotter.render()
                            time.sleep(0.2)
                            try:
                                img_array = plotter.screenshot(transparent_background=False, window_size=[800, 600])
                            except Exception as screenshot_error:
                                try:
                                    img_array = plotter.screenshot(transparent_background=False)
                                except:
                                    raise ValueError(f"Screenshot failed: {screenshot_error}")
                        
                        # Validate screenshot was captured
                        if img_array is None:
                            raise ValueError("Screenshot returned None")
                        
                        if img_array.size == 0:
                            raise ValueError("Screenshot returned empty array")
                        
                        # Check array shape
                        if not isinstance(img_array, np.ndarray):
                            raise ValueError(f"Screenshot returned non-array type: {type(img_array)}")
                        
                        # Ensure array is uint8 and has correct shape
                        if img_array.dtype != np.uint8:
                            if img_array.max() <= 1.0:
                                img_array = (img_array * 255).astype(np.uint8)
                            else:
                                img_array = img_array.astype(np.uint8)
                        
                        # Validate shape - should be (height, width, channels)
                        if len(img_array.shape) != 3:
                            raise ValueError(f"Invalid image shape: {img_array.shape}, expected 3D array (height, width, channels)")
                        
                        if img_array.shape[2] not in [3, 4]:
                            raise ValueError(f"Invalid number of channels: {img_array.shape[2]}, expected 3 (RGB) or 4 (RGBA)")
                        
                        # Ensure minimum dimensions
                        if img_array.shape[0] < 10 or img_array.shape[1] < 10:
                            raise ValueError(f"Image dimensions too small: {img_array.shape}")
                        
                        # Convert to RGB if RGBA
                        if img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]
                        
                        # Create PIL Image - this validates the array
                        try:
                            pil_img = PIL.Image.fromarray(img_array, mode='RGB')
                        except Exception as pil_error:
                            raise ValueError(f"Failed to create PIL Image: {pil_error}, array shape: {img_array.shape}, dtype: {img_array.dtype}")
                        
                        # Convert to bytes for IPython display
                        img_bytes = io.BytesIO()
                        try:
                            pil_img.save(img_bytes, format='PNG')
                        except Exception as save_error:
                            raise ValueError(f"Failed to save PIL Image to bytes: {save_error}")
                        
                        img_bytes.seek(0)
                        img_data = img_bytes.getvalue()
                        
                        # Validate we have image data
                        if len(img_data) == 0:
                            raise ValueError("Image bytes are empty")
                        
                        # Create IPython Image object
                        try:
                            image_obj = Image(img_data)
                        except Exception as image_error:
                            raise ValueError(f"Failed to create IPython Image: {image_error}")
                        
                        # Return image if requested, otherwise display
                        if return_image:
                            return plotter, image_obj
                        else:
                            if clear_display:
                                clear_output(wait=True)
                            display(image_obj)
                    except Exception as e:
                        # Fallback: try static backend (would need to recreate without off_screen)
                        if not return_image:
                            pass
                        else:
                            return plotter, None
                else:
                    plotter.render()
            
            # Return plotter (and image if requested and in Jupyter)
            if return_image and in_jupyter:
                # If we got here, we didn't return an image earlier, so return None
                return plotter, None
            return plotter
        except ImportError:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                t, p, r = kk_cart2sph(u, v, w)
                
                # Create or reuse figure
                if fig is None or ax is None:
                    # Create persistent figure with a name
                    fig = plt.figure('Spherical Parameterization State', figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                else:
                    # Clear existing collections for updating
                    while ax.collections:
                        ax.collections[0].remove()
                
                # Create vertex colors based on theta
                t_norm = t / np.max(t) if np.max(t) > 0 else t
                colors = plt.cm.hsv(t_norm)
                
                # Create face colors by averaging vertex colors
                face_colors = np.mean(colors[F], axis=1)
                
                # Create poly collection
                verts = np.column_stack([u, v, w])[F]
                collection = Poly3DCollection(verts, facecolors=face_colors, 
                                             edgecolors='black', alpha=1.0)
                ax.add_collection3d(collection)
                
                # Set limits
                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_zlim([-2, 2])
                ax.set_aspect('equal')
                
                if az is not None and el is not None:
                    ax.view_init(elev=el, azim=az)
                else:
                    ax.view_init(elev=10, azim=45)
                
                if count is not None:
                    ax.set_title(str(count))
                else:
                    ax.set_title('State of bijective mapping')
                
                plt.show(block=False)
                plt.draw()
                plt.pause(0.01)  # Small pause to allow GUI to update
                
                return fig, ax
            except ImportError:
                return None, None
    
    @staticmethod
    def subdivide_XF(vertex, face, nsub):
        """
        Subdivide triangular mesh
        
        Parameters:
        -----------
        vertex : array
            Vertex coordinates (N x 3)
        face : array
            Face connectivity (M x 3)
        nsub : int
            Number of subdivisions
            
        Returns:
        --------
        vertex1 : array
            New vertex coordinates
        face1 : array
            New face connectivity
        """
        vertex = np.asarray(vertex)
        face = np.asarray(face)
        
        for _ in range(nsub):
            nv = len(vertex)
            
            # Create edge-to-vertex mapping
            edge_map = {}
            vertex_new = vertex.tolist()
            vcount = [nv]  # Use list to allow modification in nested function
            
            # Process each face
            new_faces = []
            for f in face:
                v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
                
                # Get or create edge midpoints
                def get_midpoint(v1_idx, v2_idx, vertex, vertex_new, edge_map, vcount):
                    key = tuple(sorted([v1_idx, v2_idx]))
                    if key not in edge_map:
                        mid = (vertex[v1_idx] + vertex[v2_idx]) / 2
                        vertex_new.append(mid)
                        edge_map[key] = vcount[0]
                        vcount[0] += 1
                    return edge_map[key]
                
                v01 = get_midpoint(v0, v1, vertex, vertex_new, edge_map, vcount)
                v12 = get_midpoint(v1, v2, vertex, vertex_new, edge_map, vcount)
                v20 = get_midpoint(v2, v0, vertex, vertex_new, edge_map, vcount)
                
                # Create 4 new triangles
                new_faces.append([v0, v01, v20])
                new_faces.append([v01, v1, v12])
                new_faces.append([v12, v2, v20])
                new_faces.append([v01, v12, v20])
            
            vertex = np.array(vertex_new)
            face = np.array(new_faces)
        
        return vertex, face
    
    def laplacian_smooth(self):
        """
        Apply Laplacian smoothing to mesh
        """
        if self.needs_edge_info:
            self.edge_info()
        
        if self.L is None:
            return self
        
        X_new = self.X.copy()
        
        for _ in range(self.laplacian_smooth_iter):
            for vix in range(len(self.X)):
                if vix in self.L:
                    neighbors = self.L[vix]
                    if len(neighbors) > 0:
                        # Average of neighbors
                        neighbor_pos = self.X[neighbors]
                        avg_pos = np.mean(neighbor_pos, axis=0)
                        # Move towards average
                        X_new[vix] = (1 - self.laplacian_smooth_beta) * self.X[vix] + \
                                     self.laplacian_smooth_beta * avg_pos
        
        self.X = X_new
        self.needs_updating = True
        
        return self
    
    def remesh(self, target_edge_length=None, n_iterations=10, target_faces=None,
               method='isotropic', preserve_boundary=True, smooth_iterations=5):
        """
        Remesh to create high-quality triangles (approximately equilateral and equal area).
        
        This is the main mesh optimization method that addresses two key requirements:
        [1] Re-meshing to make triangles approximately equilateral and equal in area
        [2] Controlled mesh densification with quality preservation
        
        Parameters:
        -----------
        target_edge_length : float, optional
            Target edge length for isotropic remeshing. If None, uses mean edge length.
        n_iterations : int
            Number of remeshing iterations (default: 10)
        target_faces : int, optional
            Target number of faces. If provided, overrides target_edge_length.
        method : str
            Remeshing method: 'isotropic' (default), 'voxel', or 'simplify'
            - 'isotropic': Edge splitting + smoothing (topology-safe, may increase faces)
            - 'voxel': Voxel-based remeshing (similar to MATLAB remesh)
            - 'simplify': Quadric decimation followed by subdivision
        preserve_boundary : bool
            Whether to preserve boundary edges (default: True)
        smooth_iterations : int
            Number of smoothing iterations after remeshing (default: 5)
            
        Returns:
        --------
        self : surface_mesh
            Remeshed mesh
            
        Notes:
        ------
        The 'isotropic' method is topology-safe: it only splits long edges and 
        applies smoothing. This preserves manifold topology but may increase
        face count. Use 'simplify' first if you need to reduce face count.
        """
        if method == 'isotropic':
            return self._remesh_isotropic(target_edge_length, n_iterations, 
                                          preserve_boundary, smooth_iterations)
        elif method == 'voxel':
            return self._remesh_voxel(target_faces, smooth_iterations)
        elif method == 'simplify':
            return self._remesh_simplify(target_faces, smooth_iterations)
        else:
            raise ValueError(f"Unknown remeshing method: {method}")
    
    def remesh_uniform(self, target_faces, n_iterations=10, smooth_iterations=5,
                       preserve_boundary=True, tolerance=0.15):
        """
        High-quality uniform remeshing with approximately a target number of triangles
        and approximately equilateral triangles.
        
        Uses a pipeline: (1) bring face count near target via simplify/subdivide,
        (2) isotropic remeshing with a target edge length derived from target_faces
        and total area to regularize triangles toward equilateral.
        
        Parameters:
        -----------
        target_faces : int
            Target number of triangles (approximate).
        n_iterations : int
            Number of isotropic remeshing iterations (default: 10).
        smooth_iterations : int
            Laplacian smoothing iterations after remeshing (default: 5).
        preserve_boundary : bool
            Whether to preserve boundary vertices (default: True).
        tolerance : float
            Face count is accepted if within target_faces * (1 ± tolerance). Default 0.15.
            
        Returns:
        --------
        self : surface_mesh
        """
        if self.needs_edge_info:
            self.edge_info()
        self.props()
        n_current = len(self.F)
        A = self.A
        if A is None or A <= 0:
            A = np.sum(self.F_areas) if self.F_areas is not None else 1.0
        if A <= 0:
            A = 1.0
        
        # Target edge length for ~equilateral triangles: area_per_face = (sqrt(3)/4)*L^2, n_faces = A / area_per_face
        area_per_face = A / max(target_faces, 1)
        target_edge_length = np.sqrt(2.0 * area_per_face / np.sqrt(3.0))
        
        # Bring face count into range [target*(1-tolerance), target*(1+tolerance)]
        low = target_faces * (1.0 - tolerance)
        high = target_faces * (1.0 + tolerance)
        
        while n_current > high:
            simplify_to = max(int(target_faces * 0.9), 4)
            if n_current <= simplify_to:
                break
            self.simplify_mesh(target_faces=simplify_to)
            if self.needs_edge_info:
                self.edge_info()
            n_current = len(self.F)
        
        while n_current < low:
            self.subdivide(1)
            if self.needs_edge_info:
                self.edge_info()
            n_current = len(self.F)
        
        # If we overshot (e.g. by subdividing), simplify back toward target
        if n_current > high:
            self.simplify_mesh(target_faces=int(target_faces * 1.05))
            if self.needs_edge_info:
                self.edge_info()
            n_current = len(self.F)
        
        # Recompute area and target edge length after simplify/subdivide
        self.props()
        A = self.A
        if A is None or A <= 0:
            A = 1.0
        area_per_face = A / max(target_faces, 1)
        target_edge_length = np.sqrt(2.0 * area_per_face / np.sqrt(3.0))
        
        self._remesh_isotropic(target_edge_length=target_edge_length,
                               n_iterations=n_iterations,
                               preserve_boundary=preserve_boundary,
                               smooth_iterations=smooth_iterations)
        return self
    
    def remesh_curvature_adaptive(self, target_faces, curvature_strength=1.0,
                                  n_iterations=5, smooth_iterations=5,
                                  preserve_boundary=True):
        """
        Curvature-adaptive remeshing: approximately equilateral triangles with
        more triangles in high-curvature regions.
        
        Strategy: (1) compute mean curvature at vertices, (2) iteratively subdivide
        faces in high-curvature regions (and their neighbors for watertightness)
        until face count is near target, (3) isotropic remeshing to regularize.
        
        Parameters:
        -----------
        target_faces : int
            Target number of triangles (approximate).
        curvature_strength : float
            How strongly to favor high-curvature areas (default: 1.0). Larger values
            concentrate more triangles in curved regions.
        n_iterations : int
            Isotropic remeshing iterations after refinement (default: 5).
        smooth_iterations : int
            Laplacian smoothing iterations (default: 5).
        preserve_boundary : bool
            Whether to preserve boundary vertices (default: True).
            
        Returns:
        --------
        self : surface_mesh
        """
        if self.needs_edge_info:
            self.edge_info()
        self.props()
        H = self.H
        if H is None or len(H) != len(self.X):
            self.props()
            H = self.H
        if H is None:
            H = np.zeros(len(self.X))
        abs_H = np.abs(H)
        # Per-face curvature (mean of |H| at vertices)
        face_curv = np.mean(abs_H[self.F], axis=1)
        # Normalize to [0, 1] for selection
        cmin, cmax = face_curv.min(), face_curv.max()
        if cmax > cmin:
            face_curv_n = (face_curv - cmin) / (cmax - cmin)
        else:
            face_curv_n = np.ones(len(self.F)) * 0.5
        
        n_current = len(self.F)
        max_rounds = 50
        round_ = 0
        
        while n_current < target_faces and round_ < max_rounds:
            need = target_faces - n_current
            # Subdivide enough faces to approach target (each subdiv adds 3 faces per face)
            n_to_subdiv = min(len(self.F), max(1, (need + 2) // 3))
            # Select top faces by curvature (highest curvature first)
            sorted_by_curv = np.argsort(-face_curv_n)
            to_subdiv = sorted_by_curv[:n_to_subdiv].tolist()
            # Expand to include edge-neighbors so subdivision is watertight
            to_subdiv = self._expand_face_set_to_watertight(to_subdiv)
            if not to_subdiv:
                break
            self._subdivide_faces_watertight(to_subdiv)
            if self.needs_edge_info:
                self.edge_info()
            self.props()
            face_curv = np.mean(np.abs(self.H[self.F]), axis=1)
            cmin, cmax = face_curv.min(), face_curv.max()
            if cmax > cmin:
                face_curv_n = (face_curv - cmin) / (cmax - cmin)
            else:
                face_curv_n = np.ones(len(self.F)) * 0.5
            n_current = len(self.F)
            round_ += 1
        
        # If over target, simplify slightly
        if n_current > target_faces * 1.2:
            self.simplify_mesh(target_faces=int(target_faces * 1.05))
            if self.needs_edge_info:
                self.edge_info()
        
        self.props()
        A = self.A
        if A is None or A <= 0:
            A = 1.0
        area_per_face = A / max(target_faces, 1)
        target_edge_length = np.sqrt(2.0 * area_per_face / np.sqrt(3.0))
        self._remesh_isotropic(target_edge_length=target_edge_length,
                               n_iterations=n_iterations,
                               preserve_boundary=preserve_boundary,
                               smooth_iterations=smooth_iterations)
        return self

    # ------------------------------------------------------------------
    # Curvature-adaptive remeshing (proper adaptive sizing field)
    # ------------------------------------------------------------------

    def remesh_by_curvature(self, target_faces=None, curvature_strength=2.0,
                            smooth_curvature_field=3, n_iterations=5,
                            preserve_boundary=True, density_field=None,
                            verbose=True):
        """Curvature-adaptive remeshing with spatially varying edge length.

        Produces a mesh that is denser where mean curvature |H| is large
        and coarser where the surface is nearly flat.  This benefits
        downstream spherical parameterization by spreading curvature
        information more uniformly on the sphere and making segmentation
        naturally create more patches in curved regions.

        Algorithm (three-phase: refine, decimate, smooth):
          1. Compute per-vertex |H|, derive a curvature-weighted sizing
             field (target edge length per vertex).
          2. *Refine* high-curvature areas by adaptively splitting edges
             that exceed the local target, for *n_iterations* passes.
          3. *Decimate* back to *target_faces* using quadric error metric
             (QEM), which naturally preserves high-curvature geometry.
          4. Final tangential + Laplacian smooth for triangle quality.

        Parameters
        ----------
        target_faces : int or None
            Approximate target face count.  If None, keeps current count.
        curvature_strength : float
            Controls the density ratio between the highest-curvature and
            flattest regions.  Density ratio = (1 + curvature_strength)^2.
            For example, 2.0 gives a 9:1 face-density ratio.
        smooth_curvature_field : int
            Laplacian smoothing passes on |H| before computing the sizing
            field.  Prevents abrupt density transitions (default 3).
        n_iterations : int
            Number of adaptive-split passes (default 5).
        preserve_boundary : bool
            If True, boundary vertices are never moved or collapsed.
        verbose : bool
            Print progress.

        Returns
        -------
        self : surface_mesh
        """
        if target_faces is None:
            target_faces = len(self.F)

        if self.needs_edge_info:
            self.edge_info()
        self.props()

        # -- Step 1: curvature sizing field --------------------------------
        sizing = self._curvature_sizing_field(
            target_faces, curvature_strength, smooth_curvature_field)

        # Optional extra density: a per-vertex multiplier (>=1) that locally
        # increases face density (e.g. driven by parametric shear). Face density
        # scales as 1/L^2, so to multiply local density by d we divide L by
        # sqrt(d). This packs more (smaller) triangles into marked regions so
        # they can relax / become more equilateral on the sphere.
        if density_field is not None:
            df = np.asarray(density_field, dtype=float)
            if len(df) >= len(self.X):
                df = np.maximum(df[:len(self.X)], 1e-6)
                sizing = sizing / np.sqrt(df)
                if verbose:
                    print(f"  density_field: x{df.min():.2f}..{df.max():.2f} "
                          f"(extra refinement in marked regions)")
            elif verbose:
                print("  WARNING: density_field shorter than #verts; ignored")

        if verbose:
            print(f"remesh_by_curvature: target_faces={target_faces}, "
                  f"curvature_strength={curvature_strength}")
            print(f"  Sizing field: L_min={sizing.min():.4f}, "
                  f"L_max={sizing.max():.4f}, "
                  f"ratio={sizing.max()/max(sizing.min(),1e-15):.1f}")
            print(f"  Starting faces: {len(self.F)}")

        # -- Step 2: adaptive refinement of high-curvature areas -----------
        # Over-refine to ~2x target so that enough vertices are placed
        # in curved areas before QEM decimation trims back to target.
        refine_target = int(target_faces * 2.0)
        for it in range(n_iterations):
            n_before = len(self.F)
            if n_before >= refine_target:
                if verbose:
                    print(f"  Refine: reached {n_before} faces "
                          f"(>= {refine_target} cap), stopping refinement")
                break

            self._split_long_edges_adaptive(sizing, max_split_frac=0.3)
            sizing = self._extend_sizing_to_new_vertices(sizing)

            self.needs_edge_info = True
            self.edge_info()

            if verbose:
                print(f"  Refine iter {it+1}/{n_iterations}: "
                      f"{n_before} -> {len(self.F)} faces")

            if len(self.F) == n_before:
                if verbose:
                    print(f"  Refine: no more edges to split, stopping")
                break

        # -- Step 3: QEM decimation to target_faces ------------------------
        n_after_refine = len(self.F)
        if n_after_refine > target_faces:
            if verbose:
                print(f"  Decimating {n_after_refine} -> ~{target_faces} faces "
                      f"(QEM preserves curvature)")
            self._decimate_to_target(target_faces)
            if self.needs_edge_info:
                self.edge_info()

        # -- Step 4: tangential + Laplacian smooth for quality -------------
        self._tangential_smooth(preserve_boundary=preserve_boundary,
                                iterations=2)
        old_iter = self.laplacian_smooth_iter
        old_beta = self.laplacian_smooth_beta
        self.laplacian_smooth_iter = 3
        self.laplacian_smooth_beta = 0.2
        self.laplacian_smooth()
        self.laplacian_smooth_iter = old_iter
        self.laplacian_smooth_beta = old_beta

        # -- Step 5: guarantee a closed (watertight) mesh ------------------
        # Adaptive refinement, QEM decimation and smoothing can occasionally
        # drop a few faces and leave small holes (boundary edges). Downstream
        # spherical parameterization needs a closed genus-0 mesh, so weld
        # coincident vertices + fill holes + fix winding here.
        self._repair_watertight(verbose=verbose)
        if self.needs_edge_info:
            self.edge_info()

        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True

        if verbose:
            self.props()
            print(f"  Final: {len(self.F)} faces, {len(self.X)} verts")

        return self

    def _decimate_to_target(self, target_faces):
        """Topology-preserving decimation (prefers VTK decimate_pro).

        Uses PyVista/VTK ``decimate_pro`` with ``preserve_topology=True``
        so the result is guaranteed manifold+closed when the input is.
        Falls back to ``fast_simplification`` then ``simplify_mesh``.
        """
        n_current = len(self.F)
        if n_current <= target_faces:
            return

        target_reduction = 1.0 - target_faces / n_current
        target_reduction = min(max(target_reduction, 0.0), 0.99)

        try:
            import pyvista as pv
            faces_pv = np.hstack(
                [np.full((n_current, 1), 3, dtype=int), self.F]
            ).ravel()
            mesh = pv.PolyData(np.asarray(self.X, dtype=float), faces_pv)
            mesh = mesh.decimate_pro(
                target_reduction, preserve_topology=True,
                feature_angle=30.0)
            self.X = np.array(mesh.points)
            self.F = np.array(mesh.faces.reshape(-1, 4)[:, 1:4])
        except Exception:
            try:
                import fast_simplification
                for agg in [5, 7, 10]:
                    pts = np.ascontiguousarray(self.X, dtype=np.float64)
                    tri = np.ascontiguousarray(self.F, dtype=np.int32)
                    pts_s, tri_s = fast_simplification.simplify(
                        pts, tri, target_count=target_faces, agg=agg)
                    self.X = np.array(pts_s)
                    self.F = np.array(tri_s)
                    if len(self.F) <= int(target_faces * 1.1):
                        break
            except Exception:
                self.simplify_mesh(target_faces=target_faces)

        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True

    def _repair_watertight(self, verbose=False):
        """Weld coincident vertices and fill small holes -> closed mesh.

        Curvature refinement + QEM decimation can leave a handful of boundary
        edges (small holes). Downstream spherical parameterization requires a
        closed genus-0 mesh, so this guarantees watertightness. No-op when the
        mesh is already closed or ``trimesh`` is unavailable.
        """
        F = np.asarray(self.F, dtype=int)
        if len(F) == 0:
            return
        # Count boundary edges (used by exactly one face).
        edge_count = {}
        for f in F:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                ek = (a, b) if a < b else (b, a)
                edge_count[ek] = edge_count.get(ek, 0) + 1
        n_bnd = sum(1 for n in edge_count.values() if n == 1)
        if n_bnd == 0:
            return
        try:
            import trimesh
        except Exception:
            if verbose:
                print(f"  WARNING: {n_bnd} boundary edge(s) (holes) but "
                      f"trimesh unavailable -> cannot repair")
            return
        tm = trimesh.Trimesh(vertices=np.asarray(self.X, dtype=float),
                             faces=F, process=False)
        tm.merge_vertices()
        tm.remove_unreferenced_vertices()
        try:
            tm.fill_holes()
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(tm)
        except Exception:
            pass
        self.X = np.asarray(tm.vertices, dtype=float)
        self.F = np.asarray(tm.faces, dtype=int)
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
        if verbose:
            closed = bool(getattr(tm, 'is_watertight', False))
            print(f"  watertight repair: closed {n_bnd} boundary edge(s) -> "
                  f"{'closed' if closed else 'STILL OPEN'} ({len(self.F)} faces)")

    def _curvature_sizing_field(self, target_faces, curvature_strength,
                                smooth_passes):
        """Compute per-vertex target edge length from curvature.

        Returns an array of length len(self.X) with the target edge length
        at each vertex.  High |H| -> small L, low |H| -> large L.
        """
        H = self.H
        if H is None or len(H) != len(self.X):
            H = np.zeros(len(self.X))

        abs_H = np.abs(H).copy()

        # Smooth the curvature field (Laplacian diffusion)
        for _ in range(smooth_passes):
            abs_H_new = abs_H.copy()
            for vi in range(len(self.X)):
                nbrs = self.L.get(vi, [])
                if len(nbrs) > 0:
                    abs_H_new[vi] = (0.5 * abs_H[vi]
                                     + 0.5 * np.mean(abs_H[list(nbrs)]))
            abs_H = abs_H_new

        # Normalize to [0, 1]
        H_min, H_max = abs_H.min(), abs_H.max()
        if H_max > H_min + 1e-15:
            H_norm = (abs_H - H_min) / (H_max - H_min)
        else:
            H_norm = np.zeros(len(abs_H))

        # Per-vertex target edge length:
        #   L(v) = L_max / (1 + curvature_strength * H_norm(v))
        #
        # L_max is chosen so the total face count ≈ target_faces.
        # For equilateral triangles: n_faces ≈ total_area / (√3/4 * L²)
        # With varying L, the effective uniform L² is:
        #   L_eff² = <L(v)²>_area  (area-weighted mean)
        # We approximate this with the vertex-averaged mean.
        A = self.A if (self.A is not None and self.A > 0) else 1.0
        L_uniform = np.sqrt(2.0 * A / (np.sqrt(3.0) * max(target_faces, 1)))

        # Correction factor: the mean of 1/(1+s*h)² gives the density increase.
        # L_max = L_uniform * sqrt(mean_density_factor)
        density = 1.0 / (1.0 + curvature_strength * H_norm) ** 2
        mean_density = np.mean(density)
        if mean_density > 1e-15:
            L_max = L_uniform / np.sqrt(mean_density)
        else:
            L_max = L_uniform

        sizing = L_max / (1.0 + curvature_strength * H_norm)
        return sizing

    def _extend_sizing_to_new_vertices(self, sizing):
        """Extend sizing field to newly created vertices (from edge splits).

        New vertices get the average sizing of their neighbours.
        """
        nV = len(self.X)
        if len(sizing) >= nV:
            return sizing[:nV]

        sizing_ext = np.empty(nV)
        sizing_ext[:len(sizing)] = sizing
        for vi in range(len(sizing), nV):
            nbrs = self.L.get(vi, [])
            valid = [n for n in nbrs if n < len(sizing)]
            if valid:
                sizing_ext[vi] = np.mean(sizing[valid])
            else:
                sizing_ext[vi] = np.mean(sizing)
        return sizing_ext

    def _split_long_edges_adaptive(self, vertex_sizing, max_split_frac=0.25):
        """Split edges longer than 4/3 of the local target edge length.

        Parameters
        ----------
        vertex_sizing : array
            Per-vertex target edge length.
        max_split_frac : float
            At most this fraction of current faces will be split per call,
            preventing explosive face-count growth.
        """
        if self.needs_edge_info:
            self.edge_info()
        if self.E is None or len(self.E) == 0:
            return

        edge_lengths = self._compute_edge_lengths()
        n_sizing = len(vertex_sizing)

        v1s = self.E[:, 0].astype(int)
        v2s = self.E[:, 1].astype(int)
        s1 = np.where(v1s < n_sizing, vertex_sizing[v1s], np.mean(vertex_sizing))
        s2 = np.where(v2s < n_sizing, vertex_sizing[v2s], np.mean(vertex_sizing))
        thresholds = (4.0 / 3.0) * 0.5 * (s1 + s2)

        long_mask = edge_lengths > thresholds
        long_edges = np.where(long_mask)[0]
        if len(long_edges) == 0:
            return

        excess = edge_lengths[long_edges] / thresholds[long_edges]
        long_edges = long_edges[np.argsort(-excess)]

        max_splits = max(1, int(max_split_frac * len(self.F)))
        long_edges = long_edges[:max_splits]

        # Pre-build edge -> face lookup (O(F) once, then O(1) per edge)
        edge_to_faces = {}
        for f_idx in range(len(self.F)):
            face = self.F[f_idx]
            for i in range(3):
                va, vb = int(face[i]), int(face[(i + 1) % 3])
                ek = (min(va, vb), max(va, vb))
                if ek not in edge_to_faces:
                    edge_to_faces[ek] = []
                edge_to_faces[ek].append(f_idx)

        X_list = list(self.X)
        F_list = [list(f) for f in self.F]
        new_vertex_idx = len(self.X)
        deleted = set()

        for edge_idx in long_edges:
            if edge_idx >= len(self.E):
                continue
            v1_idx, v2_idx = int(self.E[edge_idx, 0]), int(self.E[edge_idx, 1])
            ek = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            adj = [fi for fi in edge_to_faces.get(ek, []) if fi not in deleted]
            if not adj:
                continue

            midpoint = (self.X[v1_idx] + self.X[v2_idx]) / 2.0
            X_list.append(midpoint)
            mid_idx = new_vertex_idx
            new_vertex_idx += 1

            for f_idx in adj:
                face = F_list[f_idx]
                fv = [int(v) for v in face]
                others = [v for v in fv if v != v1_idx and v != v2_idx]
                if not others:
                    continue
                v3_idx = others[0]
                deleted.add(f_idx)
                F_list.append([v1_idx, mid_idx, v3_idx])
                F_list.append([mid_idx, v2_idx, v3_idx])

        if deleted:
            F_list = [f for i, f in enumerate(F_list) if i not in deleted]

        self.X = np.array(X_list)
        self.F = np.array(F_list)

        self.X = np.array(X_list)
        self.F = np.array(F_list)

        if self.border_vertex is not None:
            n_new = len(self.X) - len(self.border_vertex)
            if n_new > 0:
                self.border_vertex = np.concatenate(
                    [self.border_vertex, np.zeros(n_new)])

        self.needs_edge_info = True
        self.edge_info()

    def _collapse_short_edges_adaptive(self, vertex_sizing,
                                       preserve_boundary=True):
        """Collapse edges shorter than 4/5 of the local target edge length."""
        if self.needs_edge_info:
            self.edge_info()
        if self.E is None or len(self.E) == 0:
            return

        edge_lengths = self._compute_edge_lengths()
        n_sizing = len(vertex_sizing)

        v1s = self.E[:, 0].astype(int)
        v2s = self.E[:, 1].astype(int)
        s1 = np.where(v1s < n_sizing, vertex_sizing[v1s], np.mean(vertex_sizing))
        s2 = np.where(v2s < n_sizing, vertex_sizing[v2s], np.mean(vertex_sizing))
        thresholds = (4.0 / 5.0) * 0.5 * (s1 + s2)

        short_mask = edge_lengths < thresholds
        short_edges = np.where(short_mask)[0]
        if len(short_edges) == 0:
            return

        # Collapse shortest-relative-to-target first
        deficit = thresholds[short_edges] / np.maximum(edge_lengths[short_edges], 1e-15)
        short_edges = short_edges[np.argsort(-deficit)]

        # Delegate to the existing collapse with topology checks,
        # processing each edge individually against its local threshold.
        edge_to_faces = {}
        for f_idx, face in enumerate(self.F):
            for i in range(3):
                va, vb = int(face[i]), int(face[(i + 1) % 3])
                ek = (min(va, vb), max(va, vb))
                if ek not in edge_to_faces:
                    edge_to_faces[ek] = []
                edge_to_faces[ek].append(f_idx)

        collapsed = set()
        deleted_faces = set()

        for edge_idx in short_edges:
            if edge_idx >= len(self.E):
                continue
            v1_idx = int(self.E[edge_idx, 0])
            v2_idx = int(self.E[edge_idx, 1])
            if v1_idx in collapsed or v2_idx in collapsed:
                continue

            ek = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            adj_faces = [f for f in edge_to_faces.get(ek, [])
                         if f not in deleted_faces]
            if preserve_boundary and len(adj_faces) == 1:
                continue

            # Link condition (topology check)
            nbrs1 = set(self.L.get(v1_idx, []))
            nbrs2 = set(self.L.get(v2_idx, []))
            common = nbrs1 & nbrs2
            expected = set()
            for f_idx in adj_faces:
                for v in self.F[f_idx]:
                    v = int(v)
                    if v != v1_idx and v != v2_idx:
                        expected.add(v)
            if common != expected:
                continue

            # Degeneracy check
            ok = True
            for f_idx, face in enumerate(self.F):
                if f_idx in deleted_faces:
                    continue
                fv = [int(v) for v in face]
                if v2_idx in fv and v1_idx not in fv:
                    nf = [v1_idx if v == v2_idx else v for v in fv]
                    if len(set(nf)) != 3:
                        ok = False
                        break
            if not ok:
                continue

            mid = (self.X[v1_idx] + self.X[v2_idx]) / 2.0
            self.X[v1_idx] = mid
            for f_idx in adj_faces:
                deleted_faces.add(f_idx)
            for f_idx in range(len(self.F)):
                if f_idx in deleted_faces:
                    continue
                for i in range(3):
                    if int(self.F[f_idx, i]) == v2_idx:
                        self.F[f_idx, i] = v1_idx
            collapsed.add(v2_idx)

            for nb in nbrs2:
                if nb == v1_idx:
                    continue
                old_k = (min(v2_idx, nb), max(v2_idx, nb))
                new_k = (min(v1_idx, nb), max(v1_idx, nb))
                if old_k in edge_to_faces:
                    fl = edge_to_faces.pop(old_k)
                    if new_k not in edge_to_faces:
                        edge_to_faces[new_k] = []
                    edge_to_faces[new_k].extend(fl)

        if deleted_faces:
            keep = [f for i, f in enumerate(self.F) if i not in deleted_faces]
            if keep:
                self.F = np.array(keep)
        # Remove degenerate faces
        valid = [f for f in self.F if len(np.unique(f)) == 3]
        if valid:
            self.F = np.array(valid)

        self._remove_unreferenced_vertices()
        self.needs_edge_info = True
        self.edge_info()

    def _expand_face_set_to_watertight(self, face_indices):
        """Expand set of face indices so that any shared edge has both adjacent faces in the set (watertight subdivision)."""
        if not face_indices:
            return []
        face_set = set(face_indices)
        # Build edge -> (face_a, face_b)
        edge_to_faces = {}
        for f_idx in range(len(self.F)):
            face = self.F[f_idx]
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                key = tuple(sorted([v1, v2]))
                if key not in edge_to_faces:
                    edge_to_faces[key] = []
                edge_to_faces[key].append(f_idx)
        changed = True
        while changed:
            changed = False
            add = set()
            for f_idx in list(face_set):
                face = self.F[f_idx]
                for i in range(3):
                    v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                    key = tuple(sorted([v1, v2]))
                    for other in edge_to_faces.get(key, []):
                        if other not in face_set:
                            add.add(other)
                            changed = True
            face_set |= add
        return list(face_set)
    
    def _subdivide_faces_watertight(self, face_indices):
        """Subdivide only the given faces; assumes the set is watertight (every edge has both adjacent faces in set)."""
        if not face_indices:
            return
        face_set = set(face_indices)
        edge_mid = {}
        X_list = list(self.X)
        new_verts = len(self.X)
        # First pass: create all midpoints for edges of faces in face_set
        for f_idx in face_set:
            face = self.F[f_idx]
            va, vb, vc = int(face[0]), int(face[1]), int(face[2])
            for e in [(min(va, vb), max(va, vb)), (min(vb, vc), max(vb, vc)), (min(va, vc), max(va, vc))]:
                if e not in edge_mid:
                    mid = (X_list[e[0]] + X_list[e[1]]) * 0.5
                    edge_mid[e] = new_verts
                    X_list.append(mid)
                    new_verts += 1
        # Second pass: build new face list (each face in face_set -> 4 faces, others unchanged)
        new_F_list = []
        for f_idx in range(len(self.F)):
            face = self.F[f_idx]
            va, vb, vc = int(face[0]), int(face[1]), int(face[2])
            if f_idx not in face_set:
                new_F_list.append([va, vb, vc])
                continue
            e1 = (min(va, vb), max(va, vb))
            e2 = (min(vb, vc), max(vb, vc))
            e3 = (min(va, vc), max(va, vc))
            mab, mbc, mac = edge_mid[e1], edge_mid[e2], edge_mid[e3]
            new_F_list.append([va, mab, mac])
            new_F_list.append([mab, vb, mbc])
            new_F_list.append([mac, mbc, vc])
            new_F_list.append([mab, mbc, mac])
        
        self.X = np.array(X_list)
        self.F = np.array(new_F_list)
        if self.border_vertex is not None:
            n_new = len(self.X) - len(self.border_vertex)
            if n_new > 0:
                self.border_vertex = np.concatenate([self.border_vertex, np.zeros(n_new)])
        self.needs_edge_info = True
        self.needs_updating = True

    
    def _remesh_isotropic(self, target_edge_length=None, n_iterations=10,
                          preserve_boundary=True, smooth_iterations=5):
        """
        Isotropic remeshing using edge operations (split, collapse, flip).
        
        This implements the algorithm from:
        "A Remeshing Approach to Multiresolution Modeling" by Botsch & Kobbelt
        
        The goal is to create a mesh with:
        - Nearly uniform edge lengths
        - Nearly equilateral triangles
        - Preserved geometric features
        """
        # Calculate current edge statistics
        if self.needs_edge_info:
            self.edge_info()
        
        # Compute edge lengths
        edge_lengths = self._compute_edge_lengths()
        
        if target_edge_length is None:
            target_edge_length = np.mean(edge_lengths)
        
        # Bounds for edge operations
        min_edge = 4.0 / 5.0 * target_edge_length
        max_edge = 4.0 / 3.0 * target_edge_length
        
        for iteration in range(n_iterations):
            # 1. Split long edges (safe operation)
            self._split_long_edges(max_edge)
            
            # 2. Collapse short edges (can cause topology issues)
            # Only do this if explicitly enabled and be very conservative
            # self._collapse_short_edges(min_edge, preserve_boundary)
            
            # 3. Flip edges to improve valence (can cause issues)
            # Disabled for now - needs more robust implementation
            # self._flip_edges_for_valence()
            
            # 4. Tangential smoothing (safe operation)
            self._tangential_smooth(preserve_boundary)
            
            # Update edge info
            self.needs_edge_info = True
            self.edge_info()
        
        # Final smoothing
        if smooth_iterations > 0:
            old_iter = self.laplacian_smooth_iter
            old_beta = self.laplacian_smooth_beta
            self.laplacian_smooth_iter = smooth_iterations
            self.laplacian_smooth_beta = 0.3
            self.laplacian_smooth()
            self.laplacian_smooth_iter = old_iter
            self.laplacian_smooth_beta = old_beta
        
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
        
        return self
    
    def _compute_edge_lengths(self):
        """Compute lengths of all edges"""
        if self.E is None or len(self.E) == 0:
            return np.array([])
        
        v1 = self.X[self.E[:, 0]]
        v2 = self.X[self.E[:, 1]]
        return np.linalg.norm(v2 - v1, axis=1)
    
    def _split_long_edges(self, max_edge_length):
        """Split edges longer than max_edge_length"""
        edge_lengths = self._compute_edge_lengths()
        
        # Find edges to split
        long_edges = np.where(edge_lengths > max_edge_length)[0]
        
        if len(long_edges) == 0:
            return
        
        # Sort by length (longest first) for stability
        long_edges = long_edges[np.argsort(-edge_lengths[long_edges])]
        
        X_list = self.X.tolist()
        F_list = self.F.tolist()
        
        # Track which faces have been modified
        new_vertex_idx = len(self.X)
        
        for edge_idx in long_edges:
            if edge_idx >= len(self.E):
                continue
            
            v1_idx, v2_idx = self.E[edge_idx]
            
            # Create midpoint
            midpoint = (self.X[v1_idx] + self.X[v2_idx]) / 2
            X_list.append(midpoint)
            mid_idx = new_vertex_idx
            new_vertex_idx += 1
            
            # Find faces containing this edge and split them
            new_faces = []
            faces_to_remove = []
            
            for f_idx, face in enumerate(F_list):
                if v1_idx in face and v2_idx in face:
                    faces_to_remove.append(f_idx)
                    
                    # Get the third vertex
                    face_arr = np.array(face)
                    v3_idx = face_arr[(face_arr != v1_idx) & (face_arr != v2_idx)][0]
                    
                    # Create two new triangles
                    new_faces.append([v1_idx, mid_idx, v3_idx])
                    new_faces.append([mid_idx, v2_idx, v3_idx])
            
            # Remove old faces (in reverse order to maintain indices)
            for f_idx in sorted(faces_to_remove, reverse=True):
                F_list.pop(f_idx)
            
            # Add new faces
            F_list.extend(new_faces)
        
        self.X = np.array(X_list)
        self.F = np.array(F_list)
        
        # Extend border_vertex for new vertices (new midpoints are interior by default)
        if self.border_vertex is not None:
            n_new = len(self.X) - len(self.border_vertex)
            if n_new > 0:
                self.border_vertex = np.concatenate([self.border_vertex, np.zeros(n_new)])
        
        self.needs_edge_info = True
        self.edge_info()
    
    def _collapse_short_edges(self, min_edge_length, preserve_boundary=True):
        """Collapse edges shorter than min_edge_length with topology preservation"""
        edge_lengths = self._compute_edge_lengths()
        
        # Find edges to collapse
        short_edges = np.where(edge_lengths < min_edge_length)[0]
        
        if len(short_edges) == 0:
            return
        
        # Sort by length (shortest first)
        short_edges = short_edges[np.argsort(edge_lengths[short_edges])]
        
        # Build edge-to-face mapping for topology checks
        edge_to_faces = {}
        for f_idx, face in enumerate(self.F):
            for i in range(3):
                v1, v2 = int(face[i]), int(face[(i + 1) % 3])
                edge_key = tuple(sorted([v1, v2]))
                if edge_key not in edge_to_faces:
                    edge_to_faces[edge_key] = []
                edge_to_faces[edge_key].append(f_idx)
        
        # Track collapsed vertices and modified faces
        collapsed = set()
        deleted_faces = set()
        
        for edge_idx in short_edges:
            if edge_idx >= len(self.E):
                continue
            
            v1_idx, v2_idx = int(self.E[edge_idx, 0]), int(self.E[edge_idx, 1])
            
            # Skip if either vertex already collapsed
            if v1_idx in collapsed or v2_idx in collapsed:
                continue
            
            # Get edge key
            edge_key = tuple(sorted([v1_idx, v2_idx]))
            
            # Check if this is a boundary edge (only 1 adjacent face)
            adj_faces = edge_to_faces.get(edge_key, [])
            is_boundary_edge = len(adj_faces) == 1
            
            if preserve_boundary and is_boundary_edge:
                continue
            
            # TOPOLOGY CHECK: Link condition
            # An edge (v1, v2) can be collapsed only if the link of v1 intersected
            # with the link of v2 equals exactly the link of the edge (v1, v2)
            # In practice: neighbors of v1 that are also neighbors of v2 should be
            # exactly the vertices opposite to the edge in adjacent triangles
            
            neighbors_v1 = set(self.L.get(v1_idx, []))
            neighbors_v2 = set(self.L.get(v2_idx, []))
            common_neighbors = neighbors_v1 & neighbors_v2
            
            # The common neighbors should be exactly the opposite vertices in adjacent faces
            expected_common = set()
            for f_idx in adj_faces:
                if f_idx in deleted_faces:
                    continue
                face = self.F[f_idx]
                for v in face:
                    if v != v1_idx and v != v2_idx:
                        expected_common.add(int(v))
            
            # If common neighbors don't match expected, skip (would create non-manifold)
            if common_neighbors != expected_common:
                continue
            
            # Check that collapse won't create degenerate triangles
            # (triangles where two vertices become the same)
            can_collapse = True
            for f_idx, face in enumerate(self.F):
                if f_idx in deleted_faces:
                    continue
                face_verts = [int(v) for v in face]
                if v2_idx in face_verts and v1_idx not in face_verts:
                    # This face will have v2 replaced by v1
                    # Check if v1 is already in this face (would create degenerate)
                    new_face = [v1_idx if v == v2_idx else v for v in face_verts]
                    if len(set(new_face)) != 3:
                        can_collapse = False
                        break
            
            if not can_collapse:
                continue
            
            # Perform collapse: move v1 to midpoint
            midpoint = (self.X[v1_idx] + self.X[v2_idx]) / 2
            self.X[v1_idx] = midpoint
            
            # Mark faces adjacent to the collapsed edge as deleted
            for f_idx in adj_faces:
                deleted_faces.add(f_idx)
            
            # Replace v2 with v1 in all faces
            for f_idx, face in enumerate(self.F):
                if f_idx in deleted_faces:
                    continue
                for i in range(3):
                    if self.F[f_idx, i] == v2_idx:
                        self.F[f_idx, i] = v1_idx
            
            collapsed.add(v2_idx)
            
            # Update edge_to_faces for subsequent iterations
            # (simplified: just mark v2 edges as involving v1 now)
            for neighbor in neighbors_v2:
                if neighbor == v1_idx:
                    continue
                old_key = tuple(sorted([v2_idx, neighbor]))
                new_key = tuple(sorted([v1_idx, neighbor]))
                if old_key in edge_to_faces:
                    faces = edge_to_faces.pop(old_key)
                    if new_key not in edge_to_faces:
                        edge_to_faces[new_key] = []
                    edge_to_faces[new_key].extend(faces)
        
        # Remove deleted faces
        if deleted_faces:
            valid_faces = [face for f_idx, face in enumerate(self.F) 
                          if f_idx not in deleted_faces]
            if len(valid_faces) > 0:
                self.F = np.array(valid_faces)
        
        # Remove any remaining degenerate faces
        valid_faces = []
        for face in self.F:
            if len(np.unique(face)) == 3:
                valid_faces.append(face)
        
        if len(valid_faces) > 0:
            self.F = np.array(valid_faces)
        
        # Remove unreferenced vertices
        self._remove_unreferenced_vertices()
        
        self.needs_edge_info = True
        self.edge_info()
    
    def _is_boundary_vertex(self, v_idx):
        """Check if vertex is on boundary (connected to boundary edge)"""
        if self.border_vertex is not None and v_idx < len(self.border_vertex):
            return self.border_vertex[v_idx] > 0
        return False
    
    def _remove_unreferenced_vertices(self):
        """Remove vertices not referenced by any face"""
        used_vertices = np.unique(self.F.flatten())
        
        if len(used_vertices) == len(self.X):
            return
        
        # Create mapping from old to new indices
        new_idx = np.zeros(len(self.X), dtype=int)
        new_idx[used_vertices] = np.arange(len(used_vertices))
        
        # Update faces
        self.F = new_idx[self.F]
        
        # Update vertices
        self.X = self.X[used_vertices]
        
        # Update border_vertex if exists and has correct size
        if self.border_vertex is not None:
            if len(self.border_vertex) >= len(used_vertices):
                # Filter to used vertices
                self.border_vertex = self.border_vertex[used_vertices]
            else:
                # border_vertex is smaller than X (vertices were added)
                # Create new border_vertex array with zeros for new vertices
                new_border = np.zeros(len(used_vertices))
                # Copy values for vertices that exist in both
                valid_mask = used_vertices < len(self.border_vertex)
                valid_used = used_vertices[valid_mask]
                new_border[valid_mask] = self.border_vertex[valid_used]
                self.border_vertex = new_border
    
    def _flip_edges_for_valence(self):
        """Flip edges to improve vertex valence (target: 6 for interior vertices)"""
        if self.needs_edge_info:
            self.edge_info()
        
        # Calculate current valence for each vertex
        valence = np.zeros(len(self.X), dtype=int)
        for v_idx in range(len(self.X)):
            if v_idx in self.L:
                valence[v_idx] = len(self.L[v_idx])
        
        target_valence = 6  # Optimal for triangular mesh
        
        # Build face adjacency (which faces share each edge)
        edge_to_faces = {}
        for f_idx, face in enumerate(self.F):
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge_key = tuple(sorted([v1, v2]))
                if edge_key not in edge_to_faces:
                    edge_to_faces[edge_key] = []
                edge_to_faces[edge_key].append(f_idx)
        
        # Try to flip edges
        F_list = self.F.tolist()
        
        for edge_key, face_list in edge_to_faces.items():
            if len(face_list) != 2:
                continue  # Boundary edge or non-manifold
            
            v1, v2 = edge_key
            f1_idx, f2_idx = face_list
            
            # Get the opposite vertices
            face1 = np.array(F_list[f1_idx])
            face2 = np.array(F_list[f2_idx])
            
            v3 = face1[(face1 != v1) & (face1 != v2)][0]
            v4 = face2[(face2 != v1) & (face2 != v2)][0]
            
            # Calculate valence improvement
            current_deviation = (abs(valence[v1] - target_valence) + 
                               abs(valence[v2] - target_valence) +
                               abs(valence[v3] - target_valence) + 
                               abs(valence[v4] - target_valence))
            
            new_valence = valence.copy()
            new_valence[v1] -= 1
            new_valence[v2] -= 1
            new_valence[v3] += 1
            new_valence[v4] += 1
            
            new_deviation = (abs(new_valence[v1] - target_valence) + 
                           abs(new_valence[v2] - target_valence) +
                           abs(new_valence[v3] - target_valence) + 
                           abs(new_valence[v4] - target_valence))
            
            # Flip if it improves valence
            if new_deviation < current_deviation:
                # Check for degenerate flip (would create self-intersection)
                # Simple check: ensure the new edge doesn't already exist
                if not self._would_create_fold(v1, v2, v3, v4):
                    # Perform flip
                    F_list[f1_idx] = [v1, v3, v4]
                    F_list[f2_idx] = [v2, v4, v3]
                    valence = new_valence
        
        self.F = np.array(F_list)
        self.needs_edge_info = True
    
    def _would_create_fold(self, v1, v2, v3, v4):
        """Check if flipping edge (v1,v2) to (v3,v4) would create a fold"""
        # Get positions
        p1, p2, p3, p4 = self.X[v1], self.X[v2], self.X[v3], self.X[v4]
        
        # Check if v3 and v4 are on opposite sides of edge (v1, v2)
        # by computing signed area of triangles
        edge = p2 - p1
        to_v3 = p3 - p1
        to_v4 = p4 - p1
        
        # Cross products
        cross3 = np.cross(edge, to_v3)
        cross4 = np.cross(edge, to_v4)
        
        # If same sign, the new edge would create a fold
        return np.dot(cross3, cross4) > 0
    
    def _tangential_smooth(self, preserve_boundary=True, iterations=1):
        """Move vertices tangent to surface to regularize mesh"""
        if self.needs_edge_info:
            self.edge_info()
        
        for _ in range(iterations):
            X_new = self.X.copy()
            
            for v_idx in range(len(self.X)):
                if preserve_boundary and self._is_boundary_vertex(v_idx):
                    continue
                
                if v_idx not in self.L or len(self.L[v_idx]) == 0:
                    continue
                
                # Compute centroid of neighbors
                neighbors = self.L[v_idx]
                centroid = np.mean(self.X[neighbors], axis=0)
                
                # Compute average normal at vertex
                normal = self._compute_vertex_normal(v_idx)
                
                # Project movement onto tangent plane
                movement = centroid - self.X[v_idx]
                tangent_movement = movement - np.dot(movement, normal) * normal
                
                # Apply with damping
                X_new[v_idx] = self.X[v_idx] + 0.5 * tangent_movement
            
            self.X = X_new
    
    def _compute_vertex_normal(self, v_idx):
        """Compute normal at vertex as area-weighted average of incident face normals"""
        if v_idx not in self.face_memb or len(self.face_memb[v_idx]) == 0:
            return np.array([0, 0, 1])
        
        normals = []
        for f_idx in self.face_memb[v_idx]:
            face = self.F[f_idx]
            v0, v1, v2 = self.X[face[0]], self.X[face[1]], self.X[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normals.append(normal / norm)
        
        if len(normals) == 0:
            return np.array([0, 0, 1])
        
        avg_normal = np.mean(normals, axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-10:
            return avg_normal / norm
        return np.array([0, 0, 1])
    
    def _remesh_voxel(self, dim=None, smooth_iterations=5):
        """
        Voxel-based remeshing similar to MATLAB's remesh function.
        
        This method:
        1. Converts mesh to voxel representation
        2. Extracts isosurface using marching cubes
        3. Applies smoothing
        
        Parameters:
        -----------
        dim : int, optional
            Voxel grid dimension. If None, computed from mesh size.
        smooth_iterations : int
            Number of smoothing iterations
        """
        try:
            # Use trimesh for voxelization
            mesh = trimesh.Trimesh(vertices=self.X, faces=self.F)
            
            # Compute appropriate voxel size
            if dim is None:
                # Target ~10000 faces
                current_faces = len(self.F)
                scale_factor = np.sqrt(10000 / max(current_faces, 100))
                dim = max(50, min(300, int(100 * scale_factor)))
            
            # Voxelize
            voxels = mesh.voxelized(pitch=mesh.bounding_box.extents.max() / dim)
            
            # Convert back to mesh using marching cubes
            new_mesh = voxels.marching_cubes
            
            self.X = np.array(new_mesh.vertices)
            self.F = np.array(new_mesh.faces)
            
            # Apply smoothing
            if smooth_iterations > 0:
                self.needs_edge_info = True
                old_iter = self.laplacian_smooth_iter
                old_beta = self.laplacian_smooth_beta
                self.laplacian_smooth_iter = smooth_iterations
                self.laplacian_smooth_beta = 0.5
                self.laplacian_smooth()
                self.laplacian_smooth_iter = old_iter
                self.laplacian_smooth_beta = old_beta
            
            self.needs_updating = True
            self.needs_edge_info = True
            self.needs_map2sphere = True
            
            return self
            
        except Exception as e:
            print(f"Voxel remeshing failed: {e}. Using isotropic method instead.")
            return self._remesh_isotropic()
    
    def _remesh_simplify(self, target_faces=None, smooth_iterations=5):
        """
        Simplify-then-subdivide remeshing.
        
        This method:
        1. Simplifies mesh to reduce face count
        2. Subdivides to increase density uniformly
        3. Applies smoothing
        """
        if target_faces is None:
            target_faces = len(self.F)
        
        # Simplify to ~25% of target faces
        simplify_target = max(100, target_faces // 4)
        self.simplify_mesh(simplify_target)
        
        # Subdivide to reach target
        current_faces = len(self.F)
        while current_faces < target_faces * 0.8:
            self.subdivide(1)
            current_faces = len(self.F)
        
        # Smooth
        if smooth_iterations > 0:
            self.needs_edge_info = True
            old_iter = self.laplacian_smooth_iter
            old_beta = self.laplacian_smooth_beta
            self.laplacian_smooth_iter = smooth_iterations
            self.laplacian_smooth_beta = 0.3
            self.laplacian_smooth()
            self.laplacian_smooth_iter = old_iter
            self.laplacian_smooth_beta = old_beta
        
        return self
    
    def simplify_mesh(self, target_faces=None, ratio=None):
        """
        Simplify mesh using quadric decimation or edge collapse.
        
        Parameters:
        -----------
        target_faces : int, optional
            Target number of faces
        ratio : float, optional
            Ratio of faces to keep (0-1). If target_faces is given, ratio is ignored.
            
        Returns:
        --------
        self : surface_mesh
        """
        if target_faces is None:
            if ratio is None:
                ratio = self.meshresample_keepratio
            target_faces = int(len(self.F) * ratio)
        
        # Try fast_simplification directly (avoids trimesh API mismatch)
        try:
            import fast_simplification
            pts = np.ascontiguousarray(self.X, dtype=np.float64)
            tri = np.ascontiguousarray(self.F, dtype=np.int32)
            pts_s, tri_s = fast_simplification.simplify(
                pts, tri, target_count=target_faces, agg=7)
            self.X = np.array(pts_s)
            self.F = np.array(tri_s)
            self.needs_updating = True
            self.needs_edge_info = True
            self.needs_map2sphere = True
            return self
        except ImportError:
            pass

        # Fallback: trimesh quadric decimation
        try:
            mesh = trimesh.Trimesh(vertices=self.X, faces=self.F)
            simplified = mesh.simplify_quadric_decimation(target_faces)
            self.X = np.array(simplified.vertices)
            self.F = np.array(simplified.faces)
            self.needs_updating = True
            self.needs_edge_info = True
            self.needs_map2sphere = True
            return self
        except Exception:
            pass

        # Last resort: edge collapse
        try:
            self._simplify_edge_collapse(target_faces)
        except Exception as e:
            print(f"Simplification failed: {e}")

        return self

    def curvature_aware_decimation(self, target_faces=None, target_ratio=None,
                                   curvature_weight=1.0, protected_vertices=None,
                                   verbose=False):
        """
        Decimate mesh using curvature-aware half-edge collapse.

        Retains original vertex positions -- vertices are only removed, never
        moved or interpolated.  High-curvature regions keep more detail.
        Protected vertices (e.g. triple-junction key vertices) are never removed.

        Uses explicit edge-face counting to guarantee manifold output.

        Parameters
        ----------
        target_faces : int, optional
            Target number of faces.  Mutually exclusive with *target_ratio*.
        target_ratio : float, optional
            Fraction of original faces to keep (0 < ratio < 1).
            Used only when *target_faces* is None.  Default 0.1.
        curvature_weight : float
            How strongly curvature biases vertex retention (default 1.0).
            0 = purely geometric cost; higher = more curvature preservation.
        protected_vertices : array-like of int, optional
            Original-mesh vertex indices that must never be collapsed away.
        verbose : bool
            Print progress information.

        Returns
        -------
        result : surface_mesh
            New mesh containing only a subset of the original vertices.
        vert_map : ndarray of int, shape (n_result_verts,)
            ``vert_map[i]`` is the index in *self* that became vertex *i*
            in *result*.
        """
        from collections import defaultdict
        import heapq

        if target_faces is None:
            if target_ratio is None:
                target_ratio = 0.1
            target_faces = max(4, int(len(self.F) * target_ratio))

        if self.H is None:
            self.props()

        X = np.array(self.X, dtype=np.float64)
        F = np.array(self.F, dtype=np.int64)
        nv, nf = len(X), len(F)

        if verbose and self.face_labels is not None:
            fl = np.asarray(self.face_labels)
            print(f"  Input face_labels: {len(fl)} entries, "
                  f"{len(np.unique(fl))} unique values, "
                  f"range [{fl.min()}, {fl.max()}]")

        if target_faces >= nf:
            result = surface_mesh(X.copy(), F.copy())
            if self.face_labels is not None:
                result.face_labels = self.face_labels.copy()
            return result, np.arange(nv, dtype=int)

        protected = set()
        if protected_vertices is not None:
            protected = set(int(v) for v in protected_vertices)

        H_arr = np.asarray(self.H, dtype=np.float64)
        H_arr = np.where(np.isfinite(H_arr), H_arr, 0.0)
        absH = np.abs(H_arr)
        Hmax = absH.max()
        Hnorm = absH / Hmax if Hmax > 1e-15 else np.zeros(nv)

        face_active = np.ones(nf, dtype=bool)
        vert_active = np.ones(nv, dtype=bool)

        v2f = defaultdict(set)
        for fi in range(nf):
            for vi in F[fi]:
                v2f[int(vi)].add(fi)

        def _ekey(a, b):
            return (min(int(a), int(b)), max(int(a), int(b)))

        # Edge -> set of active face indices (the manifold invariant tracker)
        ef = defaultdict(set)
        for fi in range(nf):
            for i in range(3):
                ef[_ekey(F[fi][i], F[fi][(i + 1) % 3])].add(fi)

        # Verify input is manifold
        n_nonmanifold_init = sum(
            1 for e, fset in ef.items() if len(fset) != 2)
        if n_nonmanifold_init > 0 and verbose:
            print(f"  WARNING: input mesh has {n_nonmanifold_init} "
                  f"non-manifold edges -- decimation may be limited")

        def _face_normal(fi):
            v0, v1, v2 = X[F[fi][0]], X[F[fi][1]], X[F[fi][2]]
            n = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(n)
            return n / ln if ln > 1e-15 else np.zeros(3)

        normals = np.array([_face_normal(fi) for fi in range(nf)])

        all_edges = set()
        for fi in range(nf):
            for i in range(3):
                all_edges.add(_ekey(F[fi][i], F[fi][(i + 1) % 3]))

        version = np.zeros(nv, dtype=np.int64)

        def _collapse_cost(keep, remove):
            geo = 0.0
            for fi in v2f[remove]:
                if not face_active[fi] or fi in v2f[keep]:
                    continue
                new_verts = [keep if int(vi) == remove else int(vi)
                             for vi in F[fi]]
                if len(set(new_verts)) < 3:
                    continue
                p0, p1, p2 = X[new_verts[0]], X[new_verts[1]], X[new_verts[2]]
                nn = np.cross(p1 - p0, p2 - p0)
                ln = np.linalg.norm(nn)
                if ln < 1e-15:
                    geo += 100.0
                    continue
                nn /= ln
                dot = np.clip(np.dot(normals[fi], nn), -1.0, 1.0)
                geo += 1.0 - dot
            curv = curvature_weight * Hnorm[remove]
            elen = np.linalg.norm(X[keep] - X[remove])
            return geo + curv + 0.01 * elen

        def _link_ok(u, v):
            nbrs_u, nbrs_v = set(), set()
            for fi in v2f[u]:
                if face_active[fi]:
                    for vi in F[fi]:
                        vi = int(vi)
                        if vi != u and vert_active[vi]:
                            nbrs_u.add(vi)
            for fi in v2f[v]:
                if face_active[fi]:
                    for vi in F[fi]:
                        vi = int(vi)
                        if vi != v and vert_active[vi]:
                            nbrs_v.add(vi)
            common = nbrs_u & nbrs_v
            expected = set()
            n_shared_active = 0
            for fi in (v2f[u] & v2f[v]):
                if face_active[fi]:
                    n_shared_active += 1
                    for vi in F[fi]:
                        vi = int(vi)
                        if vi != u and vi != v:
                            expected.add(vi)
            if n_shared_active != 2:
                return False
            return common == expected and len(expected) == 2

        def _would_create_nonmanifold(keep, remove):
            """Check if any resulting edge would be shared by >2 faces."""
            shared_faces = {fi for fi in (v2f[remove] & v2f[keep])
                            if face_active[fi]}
            new_edge_contrib = defaultdict(int)
            for fi in v2f[remove]:
                if not face_active[fi] or fi in shared_faces:
                    continue
                verts = [int(vi) for vi in F[fi]]
                new_verts = [keep if vi == remove else vi for vi in verts]
                if len(set(new_verts)) < 3:
                    continue
                for i in range(3):
                    e = _ekey(new_verts[i], new_verts[(i + 1) % 3])
                    old_e = _ekey(verts[i], verts[(i + 1) % 3])
                    if e != old_e:
                        new_edge_contrib[e] += 1
            for e, n_new in new_edge_contrib.items():
                existing = sum(
                    1 for fj in ef.get(e, set())
                    if face_active[fj] and fj not in shared_faces)
                if existing + n_new > 2:
                    return True
            return False

        def _would_flip(keep, remove):
            for fi in v2f[remove]:
                if not face_active[fi] or fi in v2f[keep]:
                    continue
                new_verts = [keep if int(vi) == remove else int(vi)
                             for vi in F[fi]]
                if len(set(new_verts)) < 3:
                    continue
                p0, p1, p2 = X[new_verts[0]], X[new_verts[1]], X[new_verts[2]]
                nn = np.cross(p1 - p0, p2 - p0)
                if np.dot(nn, normals[fi]) < 0:
                    return True
            return False

        seq = 0
        heap = []
        for u, v in all_edges:
            if Hnorm[u] >= Hnorm[v]:
                keep, remove = u, v
            else:
                keep, remove = v, u
            c = _collapse_cost(keep, remove)
            heapq.heappush(heap, (c, seq, keep, remove, version[remove]))
            seq += 1

        n_active = nf
        if verbose:
            print(f"curvature_aware_decimation: {nf} faces -> target "
                  f"{target_faces}, curvature_weight={curvature_weight}, "
                  f"{len(protected)} protected vertices")

        while n_active > target_faces and heap:
            cost, _, keep, remove, ver = heapq.heappop(heap)

            if not vert_active[remove] or not vert_active[keep]:
                continue
            if ver != version[remove]:
                continue
            if remove in protected:
                continue
            if not _link_ok(keep, remove):
                continue
            if _would_create_nonmanifold(keep, remove):
                continue
            if _would_flip(keep, remove):
                continue

            # --- perform collapse: merge remove into keep ---
            shared = {fi for fi in (v2f[remove] & v2f[keep])
                      if face_active[fi]}
            for fi in shared:
                face_active[fi] = False
                n_active -= 1
                for i in range(3):
                    ef[_ekey(F[fi][i], F[fi][(i + 1) % 3])].discard(fi)

            for fi in list(v2f[remove]):
                if not face_active[fi]:
                    continue
                old_verts = [int(vi) for vi in F[fi]]
                for i in range(3):
                    ef[_ekey(old_verts[i], old_verts[(i + 1) % 3])].discard(fi)
                for j in range(3):
                    if int(F[fi][j]) == remove:
                        F[fi][j] = keep
                if len(set(int(vi) for vi in F[fi])) < 3:
                    face_active[fi] = False
                    n_active -= 1
                else:
                    normals[fi] = _face_normal(fi)
                    new_verts = [int(vi) for vi in F[fi]]
                    for i in range(3):
                        ef[_ekey(new_verts[i],
                                 new_verts[(i + 1) % 3])].add(fi)
                v2f[keep].add(fi)

            vert_active[remove] = False
            v2f[remove] = set()

            for fi in v2f[keep]:
                if face_active[fi]:
                    for vi in F[fi]:
                        vi = int(vi)
                        if vert_active[vi]:
                            version[vi] += 1

            nbrs = set()
            for fi in v2f[keep]:
                if face_active[fi]:
                    for vi in F[fi]:
                        vi = int(vi)
                        if vi != keep and vert_active[vi]:
                            nbrs.add(vi)
            for nbr in nbrs:
                if Hnorm[keep] >= Hnorm[nbr]:
                    k, r = keep, nbr
                else:
                    k, r = nbr, keep
                c = _collapse_cost(k, r)
                heapq.heappush(heap, (c, seq, k, r, version[r]))
                seq += 1

        active_face_idx = np.where(face_active)[0]
        active_verts_set = set()
        for fi in active_face_idx:
            for vi in F[fi]:
                active_verts_set.add(int(vi))
        active_verts = sorted(active_verts_set)

        old2new = {v: i for i, v in enumerate(active_verts)}
        new_X = X[active_verts]
        new_F = np.array([[old2new[int(vi)] for vi in F[fi]]
                          for fi in active_face_idx], dtype=int)
        vert_map = np.array(active_verts, dtype=int)

        result = surface_mesh(new_X, new_F)

        if self.face_labels is not None:
            fl = np.asarray(self.face_labels)
            if len(fl) == nf:
                result.face_labels = np.array(
                    [int(fl[fi]) for fi in active_face_idx], dtype=int)
                if verbose:
                    ru = np.unique(result.face_labels)
                    print(f"  face_labels transferred: {len(ru)} unique "
                          f"values in decimated mesh")
            elif verbose:
                print(f"  WARNING: face_labels length {len(fl)} != "
                      f"nf {nf} -- labels NOT transferred")

        if verbose:
            print(f"  Result: {len(new_X)} vertices, {len(new_F)} faces "
                  f"(removed {nv - len(new_X)} verts, "
                  f"{nf - len(new_F)} faces)")

        return result, vert_map

    def optimize(self, target_vertices=1000, debug=False, max_iter=20, smooth_after=5,
                 use_pyacvd=True, tolerance=0.2):
        """
        Optimize the mesh to have approximately target_vertices vertices with
        curvature and shape preservation, uniform triangle areas, and approximately
        equilateral triangles.
        
        Uses a two-stage approach:
        1. Resample to target vertex count via CVT-style clustering (pyacvd) if
           available, otherwise via simplify/subdivide + isotropic remeshing.
        2. Iteratively refine with isotropic remeshing and light smoothing to
           improve area uniformity and equilateral-ness while preserving shape.
        
        Parameters
        ----------
        target_vertices : int, optional
            Target number of vertices (default: 1000).
        debug : bool, optional
            If True, show a figure updated each iteration with: current mesh,
            triangle area distribution, edge length distribution, and curvature
            adherence vs. the starting mesh (default: False).
        max_iter : int, optional
            Maximum refinement iterations after resampling (default: 20).
        smooth_after : int, optional
            Laplacian smoothing iterations applied after resampling and after
            each refinement step (default: 5).
        use_pyacvd : bool, optional
            If True, use pyacvd for CVT-based resampling when available
            (default: True).
        tolerance : float, optional
            Accept vertex count in [target_vertices*(1-tolerance),
            target_vertices*(1+tolerance)] (default: 0.2).
            
        Returns
        -------
        self : surface_mesh
        
        Notes
        -----
        After calling optimize(), check whether pyacvd was used with::
            mesh._last_optimize_used_pyacvd   # True if pyacvd was used
        Ensure pyacvd is installed for CVT-based resampling: ``pip install pyacvd``
        """
        if self.X is None or self.F is None or len(self.F) == 0:
            return self
        
        if self.needs_edge_info:
            self.edge_info()
        self.props()
        
        # Backup original geometry and curvature for debug / curvature adherence
        X_orig = np.array(self.X)
        H_orig = np.array(self.H) if self.H is not None else np.zeros(len(self.X))
        
        # Optional: pyacvd for uniform resampling (CVT -> uniform areas, more equilateral)
        if use_pyacvd:
            try:
                import pyvista as pv
                import pyacvd
                
                n_faces = len(self.F)
                cells = np.hstack([np.full((n_faces, 1), 3), self.F]).flatten()
                pv_mesh = pv.PolyData(self.X, cells)
                n_current = pv_mesh.n_points
                
                # PyACVD needs a dense mesh to cluster from; subdivide if needed
                clus = pyacvd.Clustering(pv_mesh)
                subdiv = 0
                while clus.n_points < 2 * target_vertices and subdiv < 5:
                    clus.subdivide(1)
                    subdiv += 1
                
                clus.cluster(target_vertices)
                remesh_pv = clus.create_mesh()
                
                self.X = np.array(remesh_pv.points)
                # PyVista face array: [3, i, j, k, 3, ...] or use regular_faces when all triangles
                if hasattr(remesh_pv, 'regular_faces') and remesh_pv.regular_faces is not None:
                    self.F = np.array(remesh_pv.regular_faces)
                else:
                    farr = np.asarray(remesh_pv.faces)
                    n_cells = remesh_pv.n_cells
                    self.F = np.array(farr.reshape(n_cells, 4)[:, 1:4], dtype=np.int64)
                self.needs_updating = True
                self.needs_edge_info = True
                self.needs_map2sphere = True
                if smooth_after > 0:
                    self.needs_edge_info = True
                    self.edge_info()
                    old_iter = self.laplacian_smooth_iter
                    old_beta = self.laplacian_smooth_beta
                    self.laplacian_smooth_iter = smooth_after
                    self.laplacian_smooth_beta = 0.3
                    self.laplacian_smooth()
                    self.laplacian_smooth_iter = old_iter
                    self.laplacian_smooth_beta = old_beta
                self._last_optimize_used_pyacvd = True
            except Exception:
                use_pyacvd = False
                self._last_optimize_used_pyacvd = False
        
        if not use_pyacvd:
            self._last_optimize_used_pyacvd = False
            # Fallback: uniform remeshing (simplify/subdivide + isotropic) to target vertex count
            # Euler: V - E + F = 2 for closed, 3F ≈ 2E, so F ≈ 2*V - 4
            target_faces = max(100, 2 * target_vertices - 4)
            self.remesh_uniform(
                target_faces=target_faces,
                n_iterations=5,
                smooth_iterations=smooth_after,
                preserve_boundary=True,
                tolerance=tolerance,
            )
        
        # Ensure all triangle normals point outward (fixes dark/light flip in rendering)
        self.mesh_check_repair('normals', verbose=False)
        
        # Refinement loop: isotropic remeshing + smoothing to improve uniformity
        debug_figs = None  # (mesh_figure, stats_figure)
        debug_axes = None
        
        if debug:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            # Separate figure for 3D mesh (large, so details are visible)
            debug_fig_mesh = plt.figure(figsize=(10, 10))
            ax_mesh = debug_fig_mesh.add_subplot(111, projection='3d')
            # Second figure for area/edge/curvature stats (3 panels only)
            debug_fig_stats = plt.figure(figsize=(12, 4))
            ax_area = debug_fig_stats.add_subplot(131)
            ax_edge = debug_fig_stats.add_subplot(132)
            ax_curv = debug_fig_stats.add_subplot(133)
            debug_axes = (ax_mesh, ax_area, ax_edge, ax_curv)
            debug_figs = (debug_fig_mesh, debug_fig_stats)
            plt.ion()
            self._update_optimize_debug_figure(debug_axes, X_orig, H_orig, iteration_label='Initial (after resampling)', debug_figs=debug_figs)
            plt.pause(0.05)
        
        for it in range(max_iter - 1):
            if self.needs_edge_info:
                self.edge_info()
            self.props()
            
            # Check convergence: area and edge uniformity
            areas = self.F_areas
            edge_lens = self._compute_edge_lengths()
            if len(edge_lens) == 0:
                break
            cv_area = np.std(areas) / (np.mean(areas) + 1e-12)
            cv_edge = np.std(edge_lens) / (np.mean(edge_lens) + 1e-12)
            if cv_area < 0.15 and cv_edge < 0.15:
                break
            
            # One step of isotropic remeshing + smoothing
            self._remesh_isotropic(
                target_edge_length=None,
                n_iterations=1,
                preserve_boundary=True,
                smooth_iterations=smooth_after,
            )
            
            if debug and debug_figs is not None:
                mesh_fig, stats_fig = debug_figs
                if plt.fignum_exists(mesh_fig.number) and plt.fignum_exists(stats_fig.number):
                    self._update_optimize_debug_figure(debug_axes, X_orig, H_orig, iteration_label=f'Iteration {it + 1}', debug_figs=debug_figs)
                plt.pause(0.05)
        
        if debug and debug_figs is not None:
            try:
                # Do not call plt.ioff() in Jupyter: it disables interactivity for all
                # subsequent figures in the notebook (e.g. plot(), plot_segmentation).
                try:
                    _in_jupyter = get_ipython() is not None
                except NameError:
                    _in_jupyter = False
                if not _in_jupyter:
                    plt.ioff()
                plt.show(block=False)
            except Exception:
                pass
        
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
        return self
    
    def _update_optimize_debug_figure(self, debug_axes, X_orig, H_orig, iteration_label='', debug_figs=None):
        """Update the debug figures for optimize(debug=True). Mesh in its own figure; stats in second figure."""
        import matplotlib.pyplot as plt
        
        ax_mesh, ax_area, ax_edge, ax_curv = debug_axes
        if debug_figs is not None:
            fig_mesh, fig_stats = debug_figs
        else:
            fig_mesh = ax_mesh.figure
            fig_stats = ax_area.figure

        if self.needs_edge_info:
            self.edge_info()
        self.props()
        
        # Clear and redraw
        ax_mesh.clear()
        ax_area.clear()
        ax_edge.clear()
        ax_curv.clear()
        
        # 1) Current mesh (3D) — in its own figure with iteration in title
        ax_mesh.plot_trisurf(
            self.X[:, 0], self.X[:, 1], self.X[:, 2],
            triangles=self.F, color='lightblue', edgecolor='gray', linewidth=0.3, alpha=0.9
        )
        title = f'Current mesh — {iteration_label}' if iteration_label else 'Current mesh'
        ax_mesh.set_title(title, fontsize=14)
        ax_mesh.set_xlabel('x')
        ax_mesh.set_ylabel('y')
        ax_mesh.set_zlabel('z')
        fig_mesh.suptitle('')  # avoid duplicate; title is on axis
        fig_mesh.tight_layout()
        
        # 2) Triangle area distribution
        areas = self.F_areas if self.F_areas is not None else self._compute_face_areas()
        ax_area.hist(areas, bins=min(50, max(10, len(areas) // 20)), color='steelblue', edgecolor='white')
        ax_area.set_title('Triangle area distribution')
        ax_area.set_xlabel('Area')
        ax_area.set_ylabel('Count')
        ax_area.axvline(np.mean(areas), color='red', linestyle='--', label=f'mean={np.mean(areas):.4f}')
        ax_area.legend()
        
        # 3) Edge length distribution
        edge_lens = self._compute_edge_lengths()
        if len(edge_lens) > 0:
            ax_edge.hist(edge_lens, bins=min(50, max(10, len(edge_lens) // 20)), color='seagreen', edgecolor='white')
            ax_edge.set_title('Edge length distribution')
            ax_edge.set_xlabel('Length')
            ax_edge.set_ylabel('Count')
            ax_edge.axvline(np.mean(edge_lens), color='red', linestyle='--', label=f'mean={np.mean(edge_lens):.4f}')
            ax_edge.legend()
        
        # 4) Curvature adherence
        from scipy.spatial import cKDTree
        tree_orig = cKDTree(X_orig)
        H_current = self.H if self.H is not None else np.zeros(len(self.X))
        _, nn_idx = tree_orig.query(self.X, k=1)
        H_orig_at_current = H_orig[nn_idx]
        valid = np.isfinite(H_orig_at_current) & np.isfinite(H_current)
        if np.sum(valid) > 2:
            ax_curv.scatter(H_orig_at_current[valid], H_current[valid], alpha=0.5, s=5)
            lims = [
                min(H_orig_at_current[valid].min(), H_current[valid].min()),
                max(H_orig_at_current[valid].max(), H_current[valid].max()),
            ]
            ax_curv.plot(lims, lims, 'r--', label='y=x')
            ax_curv.set_xlabel('Original curvature (at nearest vertex)')
            ax_curv.set_ylabel('Current curvature')
            ax_curv.set_title('Curvature adherence')
            ax_curv.legend()
            ax_curv.set_aspect('equal')
        else:
            ax_curv.set_title('Curvature (insufficient data)')
        
        # Iteration label on stats figure (suptitle) so it's always visible
        if iteration_label:
            fig_stats.suptitle(f'Optimize debug — {iteration_label}', fontsize=13)
        fig_stats.tight_layout(rect=[0, 0, 1, 0.92])  # leave room for suptitle
        
        plt.draw()
        plt.pause(0.02)
    
    def _compute_face_areas(self):
        """Compute per-face areas from current X, F (without full props)."""
        X, F = self.X, self.F
        if F.shape[0] == 3 and F.shape[1] != 3:
            F = F.T
        e1 = X[F[:, 1]] - X[F[:, 0]]
        e2 = X[F[:, 2]] - X[F[:, 0]]
        return np.linalg.norm(np.cross(e1, e2, axis=1), axis=1) * 0.5
    
    def _simplify_edge_collapse(self, target_faces):
        """
        Simplify mesh using iterative edge collapse.
        
        This is a fallback when fast_simplification is not available.
        """
        if self.needs_edge_info:
            self.edge_info()
        
        current_faces = len(self.F)
        
        while current_faces > target_faces:
            # Find shortest edge
            edge_lengths = self._compute_edge_lengths()
            if len(edge_lengths) == 0:
                break
            
            # Get shortest edge
            min_idx = np.argmin(edge_lengths)
            v1_idx, v2_idx = self.E[min_idx]
            
            # Collapse edge: move v1 to midpoint, replace v2 with v1
            midpoint = (self.X[v1_idx] + self.X[v2_idx]) / 2
            self.X[v1_idx] = midpoint
            
            # Replace all references to v2 with v1
            self.F[self.F == v2_idx] = v1_idx
            
            # Remove degenerate faces
            valid_faces = []
            for face in self.F:
                if len(np.unique(face)) == 3:
                    valid_faces.append(face)
            
            if len(valid_faces) == 0:
                break
            
            self.F = np.array(valid_faces)
            current_faces = len(self.F)
            
            # Update edge info for next iteration
            self._remove_unreferenced_vertices()
            self.needs_edge_info = True
            self.edge_info()
            
            # Safety check
            if current_faces <= target_faces:
                break
        
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
    
    def subdivide_spherical(self, n_subdivisions=1, project_to_sphere=True):
        """
        Subdivide mesh and optionally project to unit sphere.
        
        This is similar to MATLAB's SubdivideSphericalMesh function.
        Used for creating denser parameterizations.
        
        Parameters:
        -----------
        n_subdivisions : int
            Number of subdivision iterations
        project_to_sphere : bool
            If True, project vertices to unit sphere after each iteration
            
        Returns:
        --------
        self : surface_mesh
        """
        for i in range(n_subdivisions):
            # Perform triangle quadrisection
            self.X, self.F = self._tri_quad(self.X, self.F)
            
            # Project to unit sphere if requested
            if project_to_sphere:
                norms = np.linalg.norm(self.X, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                self.X = self.X / norms
        
        self.needs_updating = True
        self.needs_edge_info = True
        self.needs_map2sphere = True
        
        return self
    
    @staticmethod
    def _tri_quad(vertices, faces):
        r"""
        Triangular quadrisection: subdivide each triangle into 4 triangles.
        
        This is equivalent to MATLAB's TriQuad function.
        
                    v3                        v3
                   /  \      subdivision     /  \
                  /    \         -->       m3__m2
                 /      \                  / \  / \
               v1________v2              v1___m1___v2
        
        Parameters:
        -----------
        vertices : array (N, 3)
            Vertex coordinates
        faces : array (M, 3)
            Face connectivity
            
        Returns:
        --------
        new_vertices : array
            Subdivided vertices
        new_faces : array
            Subdivided faces
        """
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)
        
        n_vertices = len(vertices)
        n_faces = len(faces)
        
        # Edge midpoint map: (v1, v2) -> midpoint index
        edge_map = {}
        new_vertices = vertices.tolist()
        
        # Compute midpoints for each face
        midpoints = np.zeros((n_faces, 3), dtype=int)  # m1, m2, m3 for each face
        
        for f_idx, face in enumerate(faces):
            v1, v2, v3 = face
            
            # Edge 1: v1-v2
            edge_key = tuple(sorted([v1, v2]))
            if edge_key not in edge_map:
                mid = (vertices[v1] + vertices[v2]) / 2
                edge_map[edge_key] = len(new_vertices)
                new_vertices.append(mid)
            midpoints[f_idx, 0] = edge_map[edge_key]
            
            # Edge 2: v2-v3
            edge_key = tuple(sorted([v2, v3]))
            if edge_key not in edge_map:
                mid = (vertices[v2] + vertices[v3]) / 2
                edge_map[edge_key] = len(new_vertices)
                new_vertices.append(mid)
            midpoints[f_idx, 1] = edge_map[edge_key]
            
            # Edge 3: v3-v1
            edge_key = tuple(sorted([v3, v1]))
            if edge_key not in edge_map:
                mid = (vertices[v3] + vertices[v1]) / 2
                edge_map[edge_key] = len(new_vertices)
                new_vertices.append(mid)
            midpoints[f_idx, 2] = edge_map[edge_key]
        
        # Create new faces (4 per original face)
        new_faces = []
        for f_idx, face in enumerate(faces):
            v1, v2, v3 = face
            m1, m2, m3 = midpoints[f_idx]
            
            # 4 new triangles
            new_faces.append([v1, m1, m3])  # Corner at v1
            new_faces.append([m1, v2, m2])  # Corner at v2
            new_faces.append([m3, m2, v3])  # Corner at v3
            new_faces.append([m1, m2, m3])  # Center triangle
        
        return np.array(new_vertices), np.array(new_faces)
    
    def get_mesh_quality_stats(self):
        """
        Compute mesh quality statistics.
        
        Returns:
        --------
        stats : dict
            Dictionary containing:
            - 'n_vertices': number of vertices
            - 'n_faces': number of faces
            - 'n_edges': number of edges
            - 'mean_edge_length': mean edge length
            - 'std_edge_length': standard deviation of edge lengths
            - 'mean_quality': mean triangle quality (0-1, 1 = equilateral)
            - 'min_quality': minimum triangle quality
            - 'mean_area': mean face area
            - 'std_area': standard deviation of face areas
            - 'area_uniformity': coefficient of variation of areas
        """
        if self.needs_edge_info:
            self.edge_info()
        
        # Edge lengths
        edge_lengths = self._compute_edge_lengths()
        
        # Triangle quality: 4 * sqrt(3) * area / (sum of squared edge lengths)
        # This equals 1 for equilateral triangles
        if self.quality is None:
            self.props()
        
        stats = {
            'n_vertices': len(self.X),
            'n_faces': len(self.F),
            'n_edges': len(self.E) if self.E is not None else 0,
            'mean_edge_length': np.mean(edge_lengths) if len(edge_lengths) > 0 else 0,
            'std_edge_length': np.std(edge_lengths) if len(edge_lengths) > 0 else 0,
            'mean_quality': np.mean(self.quality) if self.quality is not None else 0,
            'min_quality': np.min(self.quality) if self.quality is not None else 0,
            'mean_area': np.mean(self.F_areas) if self.F_areas is not None else 0,
            'std_area': np.std(self.F_areas) if self.F_areas is not None else 0,
        }
        
        # Area uniformity (coefficient of variation)
        if stats['mean_area'] > 0:
            stats['area_uniformity'] = stats['std_area'] / stats['mean_area']
        else:
            stats['area_uniformity'] = 0
        
        return stats
    
    def print_mesh_quality(self):
        """Print mesh quality statistics"""
        stats = self.get_mesh_quality_stats()
        print("=" * 50)
        print("Mesh Quality Statistics")
        print("=" * 50)
        print(f"Vertices:           {stats['n_vertices']}")
        print(f"Faces:              {stats['n_faces']}")
        print(f"Edges:              {stats['n_edges']}")
        print(f"Mean edge length:   {stats['mean_edge_length']:.4f}")
        print(f"Std edge length:    {stats['std_edge_length']:.4f}")
        print(f"Mean quality:       {stats['mean_quality']:.4f}")
        print(f"Min quality:        {stats['min_quality']:.4f}")
        print(f"Mean face area:     {stats['mean_area']:.6f}")
        print(f"Area uniformity:    {stats['area_uniformity']:.4f} (lower is better)")
        print("=" * 50)
        return stats
    
    def densify(self, target_faces=None, factor=2.0, quality_threshold=0.3):
        """
        Create a denser mesh while maintaining quality.
        
        This is the main method for controlled mesh densification [requirement 2].
        
        Parameters:
        -----------
        target_faces : int, optional
            Target number of faces. If None, computed from factor.
        factor : float
            Multiplicative factor for face count (default: 2.0 = double the faces)
        quality_threshold : float
            Minimum acceptable triangle quality (0-1). Default: 0.3
            
        Returns:
        --------
        self : surface_mesh
        """
        if target_faces is None:
            target_faces = int(len(self.F) * factor)
        
        current_faces = len(self.F)
        
        # Calculate how many subdivisions needed
        # Each subdivision multiplies faces by 4
        subdivisions_needed = 0
        temp_faces = current_faces
        while temp_faces < target_faces:
            subdivisions_needed += 1
            temp_faces *= 4
        
        if subdivisions_needed > 0:
            # Subdivide
            self.subdivide(subdivisions_needed)
            
            # If we overshot, simplify back
            if len(self.F) > target_faces * 1.5:
                self.simplify_mesh(int(target_faces * 1.2))
            
            # Improve quality with isotropic remeshing
            self._remesh_isotropic(n_iterations=3, smooth_iterations=3)
        
        # Check quality and iterate if needed
        if self.quality is None:
            self.props()
        
        if np.mean(self.quality) < quality_threshold:
            print(f"Quality ({np.mean(self.quality):.3f}) below threshold ({quality_threshold})")
            print("Applying additional optimization...")
            self._remesh_isotropic(n_iterations=5, smooth_iterations=5)
        
        return self
    
    def plot(self, ax=None, show_edges=True, edge_color='black', color='lightblue', 
             opacity=1.0, **kwargs):
        """
        Plot mesh using matplotlib or pyvista
        
        Parameters:
        -----------
        ax : matplotlib axis or pyvista plotter, optional
        show_edges : bool
            Whether to show mesh edges (default: True)
        edge_color : str
            Color of edges (default: 'black')
        color : str
            Face color (default: 'lightblue')
        opacity : float
            Face opacity 0-1 (default: 1.0)
        **kwargs : dict
            Additional plotting options passed to underlying library
        """
        try:
            import pyvista as pv
            
            # Ensure PyVista backend is initialized (fixes "Loading..." spinner issue)
            in_jupyter = _ensure_pyvista_jupyter_backend()
            
            num_faces = self.F.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
            cells = faces_with_n_vertices.flatten()
            
            # Create plotter; in Jupyter use notebook=True so trame embeds interactively
            plotter = pv.Plotter(notebook=in_jupyter)
            mesh_pv = pv.PolyData(self.X, cells)
            
            # Set defaults for pyvista
            pv_kwargs = {
                'show_edges': show_edges,
                'edge_color': edge_color,
                'color': color,
                'opacity': opacity,
            }
            pv_kwargs.update(kwargs)
            
            plotter.add_mesh(mesh_pv, **pv_kwargs)
            plotter.view_isometric()
            plotter.reset_camera()
            if in_jupyter:
                backend = str(getattr(pv.global_theme, 'jupyter_backend', '') or '').lower()
                # Static backend (used in our notebooks) does not embed via return_viewer.
                if backend in ('static', '', 'none'):
                    plotter_off = pv.Plotter(off_screen=True, window_size=[900, 700])
                    plotter_off.add_mesh(mesh_pv, **pv_kwargs)
                    plotter_off.view_isometric()
                    plotter_off.reset_camera()
                    img = plotter_off.screenshot()
                    if img is not None and getattr(img, 'size', 0) > 0:
                        from IPython.display import display, Image
                        import io
                        import PIL.Image
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]
                        buf = io.BytesIO()
                        PIL.Image.fromarray(img).save(buf, format='PNG')
                        display(Image(buf.getvalue()))
                    return plotter_off
                result = plotter.show(return_viewer=False)
                if result is not None:
                    from IPython.display import display
                    display(result)
                return plotter
            plotter.show()
            return plotter
        except ImportError:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            
            # Create poly collection with edge display
            verts = self.X[self.F]
            
            # Set defaults for matplotlib
            mpl_kwargs = {
                'facecolor': color,
                'edgecolor': edge_color if show_edges else 'none',
                'alpha': opacity,
                'linewidth': 0.5 if show_edges else 0,
            }
            mpl_kwargs.update(kwargs)
            
            collection = Poly3DCollection(verts, **mpl_kwargs)
            ax.add_collection3d(collection)
            
            # Set limits
            ax.set_xlim(self.X[:, 0].min(), self.X[:, 0].max())
            ax.set_ylim(self.X[:, 1].min(), self.X[:, 1].max())
            ax.set_zlim(self.X[:, 2].min(), self.X[:, 2].max())
            ax.set_aspect('equal')
            
            plt.show()
        
        return ax
    
    def plot_segmentation_with_seeds(self, slix, verbose=True, title=None):
        """
        Plot mesh segmentation with patch labels and seed faces marked as spheres
        
        Parameters:
        -----------
        slix : array
            Seed face indices
        verbose : bool
            Print diagnostic information (default: True)
        title : str, optional
            Custom title for the plot. If None, uses default title.
        
        Returns:
        --------
        plotter : pyvista.Plotter
            The plotter object (can be used for further customization)
        """
        if self.face_labels is None:
            raise ValueError("face_labels must be set before plotting segmentation")
        
        # Plot with seed faces highlighted using PyVista for interactive plots
        import pyvista as pv
        
        # Prepare mesh for PyVista
        num_faces = len(self.F)
        faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
        cells = faces_with_n_vertices.flatten()
        mesh = pv.PolyData(self.X, cells)
        
        # Add face labels
        face_field = np.asarray(self.face_labels, dtype=float)
        mesh['face_labels'] = face_field
        unique_labels = np.unique(face_field)
        unique_labels = unique_labels[unique_labels > 0]
        n_labels = len(unique_labels)
        
        # Ensure PyVista backend is initialized (fixes "Loading..." spinner issue)
        in_jupyter = _ensure_pyvista_jupyter_backend()
        
        # Choose colormap based on number of labels
        if n_labels <= 10:
            cmap_name = 'tab10'
        elif n_labels <= 20:
            cmap_name = 'tab20'
        else:
            cmap_name = 'jet'
        
        # Calculate average side length of triangles for sphere sizing
        edge_lengths = []
        for face in self.F:
            v0, v1, v2 = self.X[face[0]], self.X[face[1]], self.X[face[2]]
            edge_lengths.append(np.linalg.norm(v1 - v0))
            edge_lengths.append(np.linalg.norm(v2 - v1))
            edge_lengths.append(np.linalg.norm(v0 - v2))
        avg_edge_length = np.mean(edge_lengths)
        radius = avg_edge_length / 2.0
        
        # Two subplots for different perspectives
        plotter = pv.Plotter(shape=(1, 2), notebook=in_jupyter)
        plotter.background_color = 'black'
        
        if title is None:
            title = f'Segmentation (Red spheres = seed faces, {n_labels} patches)'
        
        view_configs = [(45, 25), (225, 25)]  # (azimuth, elevation)
        for col, (az, el) in enumerate(view_configs):
            plotter.subplot(0, col)
            plotter.add_mesh(mesh, scalars='face_labels', cmap=cmap_name,
                           show_edges=True, edge_color='black',
                           line_width=0.3, opacity=1.0)
            
            nseeds = len(slix)
            for ix in range(nseeds):
                seed_face = self.F[slix[ix]]
                centroid = self.X[seed_face].mean(axis=0)
                sphere = pv.Sphere(radius=radius, center=centroid, 
                                 theta_resolution=20, phi_resolution=20)
                plotter.add_mesh(sphere, color='red', opacity=1.0, 
                               show_edges=True, edge_color='yellow', line_width=1)
            
            plotter.camera.azimuth = az
            plotter.camera.elevation = el
            plotter.add_text(f'View {col + 1}', font_size=10)
        
        plotter.subplot(0, 0)
        plotter.add_text(title, font_size=12)
        if verbose:
            nseeds = len(slix)
            print(f"Plotted {nseeds} seed faces as red spheres (radius = {radius:.4f})")
        plotter.show()
        return plotter
    
    def plot_H(self, r=None, flag=False, **kwargs):
        """
        Plot mean curvature H as color field
        
        Parameters:
        -----------
        r : tuple, optional
            (min, max) range for H values
        flag : bool
            If True, use median vertex curvature
        **kwargs : dict
            Plotting options
        """
        self.props()
        H = self.H.copy()
        
        if r is not None and len(r) == 2:
            H[H < r[0]] = r[0]
            H[H > r[1]] = r[1]
        
        if flag:
            # Calculate median vertex curvature
            for vix in range(len(self.L)):
                l = self.L[vix]
                h = self.H[l]
                h_valid = h[(h != 0) & ~np.isnan(h)]
                if len(h_valid) > 0:
                    mval = np.median(h_valid)
                    H[vix] = mval
        
        try:
            import pyvista as pv
            in_jupyter = _ensure_pyvista_jupyter_backend()
            # Use notebook=in_jupyter so interactive figures embed in Jupyter (trame)
            plotter = pv.Plotter(notebook=in_jupyter)
            # Prepare faces with number of vertices
            num_faces = self.F.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
            cells = faces_with_n_vertices.flatten()
            mesh_pv = pv.PolyData(self.X, cells)
            mesh_pv['curvature'] = H
            plotter.add_mesh(mesh_pv, scalars='curvature', cmap='viridis', 
                           show_edges=False, **kwargs)
            plotter.show()
        except ImportError:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            # Create poly collection with color mapping
            verts = self.X[self.F]
            collection = Poly3DCollection(verts, **kwargs)
            # Map H to face colors (simplified - would need proper interpolation)
            face_colors = H[self.F].mean(axis=1)
            collection.set_array(face_colors)
            collection.set_cmap('viridis')
            ax.add_collection3d(collection)
            
            ax.set_xlim(self.X[:, 0].min(), self.X[:, 0].max())
            ax.set_ylim(self.X[:, 1].min(), self.X[:, 1].max())
            ax.set_zlim(self.X[:, 2].min(), self.X[:, 2].max())
            ax.set_aspect('equal')
            plt.colorbar(collection, ax=ax)
            plt.show()
        
        return H
    
    def plot_fast(self, c='r', **kwargs):
        """
        Fast plot with simple color
        
        Parameters:
        -----------
        c : str or array
            Color (default 'r' for red)
        **kwargs : dict
            Plotting options
        """
        try:
            import pyvista as pv
            in_jupyter = False
            try:
                import sys
                in_jupyter = 'ipykernel' in sys.modules or 'IPython' in sys.modules
            except Exception:
                pass
            plotter = pv.Plotter(notebook=in_jupyter)
            # Prepare faces with number of vertices
            num_faces = self.F.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
            cells = faces_with_n_vertices.flatten()
            mesh_pv = pv.PolyData(self.X, cells)
            plotter.add_mesh(mesh_pv, color=c, show_edges=True, **kwargs)
            plotter.show()
        except ImportError:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            verts = self.X[self.F]
            collection = Poly3DCollection(verts, facecolor=c, edgecolor='k', 
                                         alpha=1.0, **kwargs)
            ax.add_collection3d(collection)
            
            ax.set_xlim(self.X[:, 0].min(), self.X[:, 0].max())
            ax.set_ylim(self.X[:, 1].min(), self.X[:, 1].max())
            ax.set_zlim(self.X[:, 2].min(), self.X[:, 2].max())
            ax.set_aspect('equal')
            plt.show()
    
    def plot_quality(self, **kwargs):
        """
        Plot mesh quality as color field
        """
        self.props()
        
        try:
            import pyvista as pv
            in_jupyter = False
            try:
                import sys
                in_jupyter = 'ipykernel' in sys.modules or 'IPython' in sys.modules
            except Exception:
                pass
            plotter = pv.Plotter(notebook=in_jupyter)
            # Prepare faces with number of vertices
            num_faces = self.F.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
            cells = faces_with_n_vertices.flatten()
            mesh_pv = pv.PolyData(self.X, cells)
            mesh_pv['quality'] = self.quality
            plotter.add_mesh(mesh_pv, scalars='quality', cmap='viridis',
                           show_edges=False, **kwargs)
            plotter.add_scalar_bar(title='Quality')
            plotter.show()
        except ImportError:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            verts = self.X[self.F]
            collection = Poly3DCollection(verts, **kwargs)
            face_quality = self.quality[self.F].mean(axis=1)
            collection.set_array(face_quality)
            collection.set_cmap('viridis')
            ax.add_collection3d(collection)
            
            ax.set_xlim(self.X[:, 0].min(), self.X[:, 0].max())
            ax.set_ylim(self.X[:, 1].min(), self.X[:, 1].max())
            ax.set_zlim(self.X[:, 2].min(), self.X[:, 2].max())
            ax.set_aspect('equal')
            plt.colorbar(collection, ax=ax, label='Quality')
            plt.show()
    
    def plot_map_quality(self, **kwargs):
        """
        Plot mapping quality on sphere
        """
        self.props()
        
        # Map to sphere
        if self.t is None or self.p is None:
            self.map2sphere()
        
        from .utils import kk_sph2cart
        x, y, z = kk_sph2cart(self.t, self.p, np.ones_like(self.t))
        X = np.column_stack([x, y, z])
        C = self.F
        
        # Compute quality on sphere (simplified - would need proper computation)
        # For now, use existing quality if available
        if self.quality is None:
            self.quality = np.ones(len(self.F))
        
        try:
            import pyvista as pv
            in_jupyter = False
            try:
                import sys
                in_jupyter = 'ipykernel' in sys.modules or 'IPython' in sys.modules
            except Exception:
                pass
            plotter = pv.Plotter(notebook=in_jupyter)
            # Prepare faces with number of vertices
            num_faces = C.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), C))
            cells = faces_with_n_vertices.flatten()
            mesh_pv = pv.PolyData(X, cells)
            mesh_pv['quality'] = self.quality
            plotter.add_mesh(mesh_pv, scalars='quality', cmap='viridis',
                           show_edges=False, **kwargs)
            plotter.add_scalar_bar(title='Mapping Quality')
            plotter.show()
        except ImportError:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            verts = X[C]
            collection = Poly3DCollection(verts, **kwargs)
            face_quality = self.quality[C].mean(axis=1)
            collection.set_array(face_quality)
            collection.set_cmap('viridis')
            ax.add_collection3d(collection)
            
            ax.set_xlim(X[:, 0].min(), X[:, 0].max())
            ax.set_ylim(X[:, 1].min(), X[:, 1].max())
            ax.set_zlim(X[:, 2].min(), X[:, 2].max())
            ax.set_aspect('equal')
            plt.colorbar(collection, ax=ax, label='Mapping Quality')
            plt.show()
        
        return self

    def compute_shear_spherical_mesh(self):
        """
        Compute per-face shear for this mesh's spherical parameterization (t, p).
        Requires self.t, self.p to be set (e.g. after map2sphere).
        
        Returns:
        --------
        shear_per_face : array (nfaces,)
        summary : dict with 'mean', 'max', 'rms', 'total' (single-number metrics)
        """
        if self.t is None or self.p is None:
            raise ValueError("t and p must be set (e.g. call map2sphere first)")
        return surface_mesh.compute_shear_spherical(self.t, self.p, self.F)

    def compute_shear_3d_mesh(self):
        """
        Compute per-face shear for this mesh's 3D geometry (X, F).
        
        Returns:
        --------
        shear_per_face : array (nfaces,)
        summary : dict with 'mean', 'max', 'rms', 'total'
        """
        return surface_mesh.compute_shear_3d(self.X, self.F)

    def plot_shear_heatmap_spherical_mesh(self, shear_per_face=None, title='Shear on spherical parameterization', **kwargs):
        """
        Plot spherical parameterization with face colors = shear (heat map).
        Uses self.t, self.p, self.F. If shear_per_face is None, computes it.
        """
        if self.t is None or self.p is None:
            raise ValueError("t and p must be set (e.g. call map2sphere first)")
        return surface_mesh.plot_shear_heatmap_spherical(self.t, self.p, self.F, shear_per_face=shear_per_face, title=title, **kwargs)

    def plot_shear_heatmap_3d_mesh(self, shear_per_face=None, title='Shear on 3D mesh', **kwargs):
        """
        Plot 3D mesh with face colors = shear (heat map).
        Uses self.X, self.F. If shear_per_face is None, computes it.
        """
        return surface_mesh.plot_shear_heatmap_3d(self.X, self.F, shear_per_face=shear_per_face, title=title, **kwargs)

    def plot_K(self):
        """Plot Gaussian curvature K"""
        self.props()
        K = self.H.copy()  # Placeholder - would calculate actual Gaussian curvature
        
        try:
            import pyvista as pv
            in_jupyter = False
            try:
                import sys
                in_jupyter = 'ipykernel' in sys.modules or 'IPython' in sys.modules
            except Exception:
                pass
            plotter = pv.Plotter(notebook=in_jupyter)
            mesh = pv.PolyData(self.X, self.F)
            mesh['K'] = K
            plotter.add_mesh(mesh, scalars='K', cmap='viridis')
            plotter.show()
        except ImportError:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_trisurf(self.X[:, 0], self.X[:, 1], self.X[:, 2], 
                               triangles=self.F, facecolors=plt.cm.viridis(K / (np.max(K) + 1e-10)))
                plt.show()
            except ImportError:
                print("No plotting library available")
    
    def plot_field(self, nsf=1):
        """Plot scalar field"""
        if len(self.sf) == 0:
            self.plot()
        else:
            if nsf <= len(self.sf):
                s = self.sf[nsf - 1]
                if isinstance(s, (list, tuple)) and len(s) > 1:
                    s = s[1]
                s_array = np.asarray(s).flatten()
                
                try:
                    import pyvista as pv
                    in_jupyter = _ensure_pyvista_jupyter_backend()
                    plotter = pv.Plotter(notebook=in_jupyter)
                    mesh = pv.PolyData(self.X, self.F)
                    mesh['field'] = s_array
                    plotter.add_mesh(mesh, scalars='field', cmap='viridis', show_edges=False)
                    plotter.show()
                except ImportError:
                    try:
                        import matplotlib.pyplot as plt
                        from mpl_toolkits.mplot3d import Axes3D
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.plot_trisurf(self.X[:, 0], self.X[:, 1], self.X[:, 2],
                                       triangles=self.F, facecolors=plt.cm.viridis(s_array / (np.max(s_array) + 1e-10)))
                        plt.show()
                    except ImportError:
                        print("No plotting library available")
    
    def html_mesh_parameterization_export(self, path, show_reference_sphere=False,
                                          face_field=None, title=None):
        """
        Export the mesh spherical parameterization to an interactive HTML file.

        Use this to inspect the final fine-mesh spherical parameterization in a
        browser before proceeding to spherical harmonics projection. The output
        is a self-contained HTML file with an interactive 3D view (PyVista/trame).

        Parameters
        ----------
        path : str
            Output file path (e.g. 'output/sphere_param.html').
        show_reference_sphere : bool, optional
            If True, draw a semi-transparent reference sphere behind the mesh.
            Default False.
        face_field : array, optional
            Scalar field for face coloring (e.g. face_labels). If None and
            self.face_labels exists, uses face_labels.
        title : str, optional
            Title text in the viewer. Default 'Spherical Parameterization'.

        Returns
        -------
        str
            The path to the exported HTML file.

        Raises
        ------
        ValueError
            If self.t or self.p is None (mesh not parameterized).
        """
        import os
        if self.t is None or self.p is None:
            self.map2sphere()
        u, v, w = kk_sph2cart(self.t, self.p, np.ones(len(self.p)))
        X_sph = np.column_stack([u, v, w])

        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("html_mesh_parameterization_export requires pyvista: pip install pyvista")

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        num_faces = self.F.shape[0]
        faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
        cells = faces_with_n_vertices.flatten()
        mesh = pv.PolyData(X_sph, cells)

        if face_field is not None:
            face_colors_data = face_field
        elif self.face_labels is not None:
            face_colors_data = self.face_labels
        else:
            face_colors_data = None

        plotter = pv.Plotter(off_screen=True)

        if show_reference_sphere:
            sphere = pv.Sphere(radius=0.98, theta_resolution=30, phi_resolution=30)
            plotter.add_mesh(sphere, color='cyan', opacity=0.15, show_edges=False)

        if face_colors_data is not None:
            face_field_arr = np.asarray(face_colors_data, dtype=float)
            if np.max(face_field_arr) > np.min(face_field_arr):
                mesh['face_field'] = face_field_arr
                n_unique = len(np.unique(face_field_arr))
                cmap_name = 'tab10' if n_unique <= 10 else 'jet'
                plotter.add_mesh(mesh, scalars='face_field', cmap=cmap_name,
                               show_edges=True, edge_color='black', line_width=0.3)
            else:
                plotter.add_mesh(mesh, color='lightblue', show_edges=True,
                               edge_color='black', line_width=0.3)
        else:
            plotter.add_mesh(mesh, color='lightblue', show_edges=True,
                           edge_color='black', line_width=0.3)

        plotter.add_text(title or 'Spherical Parameterization', font_size=12)
        plotter.background_color = 'white'

        try:
            plotter.export_html(path)
            plotter.close()
            return path
        except Exception as e:
            plotter.close()
            raise RuntimeError(f"export_html failed: {e}. Install trame if needed: pip install trame") from e

    def plot_spherical_parameterization(self, flag=0, face_field=None, export_html_path=None):
        """Plot spherical parameterization. Optionally export to HTML for browser viewing."""
        if self.t is None or self.p is None:
            self.map2sphere()
        
        u, v, w = kk_sph2cart(self.t, self.p, np.ones(len(self.p)))
        X_sph = np.column_stack([u, v, w])
        
        try:
            import pyvista as pv
            
            # Ensure PyVista backend is initialized (fixes "Loading..." spinner issue)
            in_jupyter = _ensure_pyvista_jupyter_backend()
            
            # Prepare faces with number of vertices
            num_faces = self.F.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
            cells = faces_with_n_vertices.flatten()
            
            # Create mesh
            mesh = pv.PolyData(X_sph, cells)
            
            # Color by face labels if available
            if face_field is not None:
                face_colors_data = face_field
            elif self.face_labels is not None:
                face_colors_data = self.face_labels
            else:
                face_colors_data = None
            
            # Use notebook=in_jupyter so interactive figures embed in Jupyter (trame)
            plotter = pv.Plotter(notebook=in_jupyter)
            
            # Optional: draw reference sphere
            if flag:
                sphere = pv.Sphere(radius=0.98, theta_resolution=30, phi_resolution=30)
                plotter.add_mesh(sphere, color='cyan', opacity=1.0, show_edges=False)
            
            if face_colors_data is not None:
                face_field_arr = np.asarray(face_colors_data, dtype=float)
                if np.max(face_field_arr) > np.min(face_field_arr):
                    face_field_norm = (face_field_arr - np.min(face_field_arr)) / (np.max(face_field_arr) - np.min(face_field_arr))
                else:
                    face_field_norm = np.zeros_like(face_field_arr)
                mesh['face_field'] = face_field_arr
                # Choose colormap based on number of unique values
                n_unique = len(np.unique(face_field_arr))
                cmap_name = 'tab10' if n_unique <= 10 else 'jet'
                plotter.add_mesh(mesh, scalars='face_field', cmap=cmap_name, 
                               show_edges=True, edge_color='black', line_width=0.3)
            else:
                plotter.add_mesh(mesh, color='lightblue', show_edges=True, 
                               edge_color='black', line_width=0.3)
            
            plotter.add_text('Spherical Parameterization', font_size=12)
            
            # Export to HTML for browser viewing when path is given
            if export_html_path:
                try:
                    plotter.export_html(export_html_path)
                    print(f"Exported full parameterization on sphere to: {export_html_path}")
                except Exception as e:
                    print(f"export_html failed: {e}. Install trame if needed: pip install trame")
            
            # Return plotter.show() result so Jupyter displays the figure
            if in_jupyter:
                try:
                    return plotter.show(return_viewer=True)
                except Exception as e:
                    try:
                        result = plotter.show(return_viewer=True)
                        if result is not None:
                            return result
                    except Exception as e2:
                        print(f"Warning: Could not display interactive plot: {e}, {e2}")
                    return plotter
            plotter.show()
            return plotter
        except ImportError:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Optional: draw reference sphere
                if flag:
                    u_sphere = np.linspace(0, 2 * np.pi, 30)
                    v_sphere = np.linspace(0, np.pi, 30)
                    x_sphere = 0.98 * np.outer(np.cos(u_sphere), np.sin(v_sphere))
                    y_sphere = 0.98 * np.outer(np.sin(u_sphere), np.sin(v_sphere))
                    z_sphere = 0.98 * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
                    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=1.0, color='cyan', edgecolor='none')
                
                # Create triangles from spherical coordinates
                triangles = []
                for face in self.F:
                    triangle = [X_sph[face[0]], X_sph[face[1]], X_sph[face[2]]]
                    triangles.append(triangle)
                
                # Color by face labels if available
                if face_field is not None:
                    face_colors_data = face_field
                elif self.face_labels is not None:
                    face_colors_data = self.face_labels
                else:
                    face_colors_data = None
                
                if face_colors_data is not None:
                    face_field_arr = np.asarray(face_colors_data, dtype=float)
                    if np.max(face_field_arr) > np.min(face_field_arr):
                        face_field_norm = (face_field_arr - np.min(face_field_arr)) / (np.max(face_field_arr) - np.min(face_field_arr))
                    else:
                        face_field_norm = np.zeros_like(face_field_arr)
                    cmap = plt.cm.tab10 if np.max(face_field_arr) <= 10 else plt.cm.jet
                    face_colors = cmap(face_field_norm)
                else:
                    face_colors = 'lightblue'
                
                poly = Poly3DCollection(triangles, facecolors=face_colors, 
                                        edgecolors='k', linewidths=0.3, alpha=1.0)
                ax.add_collection3d(poly)
                
                # Set axis limits for unit sphere
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_zlim(-1.2, 1.2)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Spherical Parameterization')
                fig.patch.set_facecolor('white')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("No plotting library available")
    
    def plot_labels(self, flag=1, face_field=None):
        """Plot face labels with different colors per label/patch"""
        if self.face_labels is not None or face_field is not None:
            show_edges = flag != 0
            if face_field is None:
                face_field = self.face_labels.copy()
            
            # Map face labels to distinct colors
            face_field = np.asarray(face_field, dtype=float)
            
            # Get unique labels and create a mapping
            unique_labels = np.unique(face_field)
            unique_labels = unique_labels[unique_labels > 0]  # Remove any zero labels
            n_labels = len(unique_labels)
            
            try:
                import pyvista as pv
                
                # Prepare faces with number of vertices
                num_faces = self.F.shape[0]
                faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
                cells = faces_with_n_vertices.flatten()
                
                # Create mesh
                mesh = pv.PolyData(self.X, cells)
                mesh['face_labels'] = face_field
                
                # Ensure PyVista backend is initialized (fixes "Loading..." spinner issue)
                in_jupyter = _ensure_pyvista_jupyter_backend()
                
                # Choose colormap based on number of labels
                if n_labels <= 10:
                    cmap_name = 'tab10'
                elif n_labels <= 20:
                    cmap_name = 'tab20'
                else:
                    cmap_name = 'jet'
                
                # Two subplots for different perspectives
                plotter = pv.Plotter(shape=(1, 2), notebook=in_jupyter)
                plotter.background_color = 'black'
                
                view_configs = [(45, 25), (225, 25)]  # (azimuth, elevation)
                for col, (az, el) in enumerate(view_configs):
                    plotter.subplot(0, col)
                    plotter.add_mesh(mesh, scalars='face_labels', cmap=cmap_name,
                                   show_edges=show_edges, edge_color='black' if show_edges else None,
                                   line_width=0.5 if show_edges else 0, opacity=1.0)
                    
                    if flag > 1:
                        scale = 1.05
                        for vix in range(len(self.X)):
                            point = scale * self.X[vix]
                            plotter.add_point_labels(point, [str(vix)], font_size=8, text_color='white')
                    
                    plotter.camera.azimuth = az
                    plotter.camera.elevation = el
                    plotter.add_text(f'View {col + 1}', font_size=10)
                
                plotter.subplot(0, 0)
                plotter.add_text(f'Mesh with Patch Labels ({n_labels} patches)', font_size=12)
                plotter.show()
            except ImportError:
                try:
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    
                    fig = plt.figure(figsize=(16, 7))
                    axes = fig.subplots(1, 2, subplot_kw={'projection': '3d'})
                    
                    ec = 'k' if flag else 'none'
                    if n_labels <= 10:
                        cmap = plt.cm.tab10
                        n_colors = 10
                    elif n_labels <= 20:
                        cmap = plt.cm.tab20
                        n_colors = 20
                    else:
                        cmap = plt.cm.jet
                        n_colors = 256
                    
                    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                    face_colors = np.zeros((len(face_field), 4))
                    for i, label in enumerate(face_field):
                        if label in label_to_idx:
                            color_idx = label_to_idx[label]
                            color_val = color_idx / max(1, n_labels - 1) if n_labels > 1 else 0.0
                            face_colors[i] = cmap(color_val)
                        else:
                            face_colors[i] = [0.5, 0.5, 0.5, 0.9]
                    
                    triangles = []
                    for face in self.F:
                        triangles.append([self.X[face[0]], self.X[face[1]], self.X[face[2]]])
                    
                    view_configs = [(25, 45), (25, 225)]
                    for ax, (elev, azim) in zip(axes, view_configs):
                        poly = Poly3DCollection(triangles, facecolors=face_colors, 
                                                edgecolors=ec, linewidths=0.5, alpha=1.0)
                        ax.add_collection3d(poly)
                        
                        max_range = np.array([self.X[:, 0].max() - self.X[:, 0].min(),
                                              self.X[:, 1].max() - self.X[:, 1].min(),
                                              self.X[:, 2].max() - self.X[:, 2].min()]).max() / 2.0
                        mid_x = (self.X[:, 0].max() + self.X[:, 0].min()) * 0.5
                        mid_y = (self.X[:, 1].max() + self.X[:, 1].min()) * 0.5
                        mid_z = (self.X[:, 2].max() + self.X[:, 2].min()) * 0.5
                        ax.set_xlim(mid_x - max_range, mid_x + max_range)
                        ax.set_ylim(mid_y - max_range, mid_y + max_range)
                        ax.set_zlim(mid_z - max_range, mid_z + max_range)
                        
                        if flag > 1:
                            scale = 1.05
                            for vix in range(len(self.X)):
                                ax.text(scale * self.X[vix, 0], scale * self.X[vix, 1], scale * self.X[vix, 2],
                                       str(vix), color='white', fontsize=8)
                        
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.view_init(elev=elev, azim=azim)
                    
                    axes[0].set_title('View 1')
                    axes[1].set_title('View 2')
                    fig.suptitle(f'Mesh with Patch Labels ({n_labels} patches)', fontsize=14)
                    fig.patch.set_facecolor('black')
                    for ax in axes:
                        ax.set_facecolor('black')
                    plt.tight_layout()
                    plt.show()
                except ImportError:
                    print("No plotting library available")
        else:
            self.plot()
    
    def plot_patches(self, PM, pflag=1):
        """Plot patches with key vertices"""
        try:
            import pyvista as pv
            
            # Check if we're in Jupyter for interactive plots
            in_jupyter = False
            try:
                import sys
                in_jupyter = 'ipykernel' in sys.modules or 'IPython' in sys.modules
            except:
                pass
            
            # Use notebook=in_jupyter so interactive figures embed in Jupyter (trame)
            plotter = pv.Plotter(notebook=in_jupyter)
            
            # Plot the mesh with patch colors
            if self.face_labels is not None and pflag < 3:
                show_edges = pflag != 0
                face_field = self.face_labels.copy()
                
                # Map face labels to distinct colors (same logic as plot_labels)
                face_field = np.asarray(face_field, dtype=float)
                unique_labels = np.unique(face_field)
                unique_labels = unique_labels[unique_labels > 0]  # Remove any zero labels
                n_labels = len(unique_labels)
                
                # Prepare faces with number of vertices
                num_faces = self.F.shape[0]
                faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
                cells = faces_with_n_vertices.flatten()
                
                # Create mesh
                mesh = pv.PolyData(self.X, cells)
                mesh['face_labels'] = face_field
                
                # Choose colormap based on number of labels
                if n_labels <= 10:
                    cmap_name = 'tab10'
                elif n_labels <= 20:
                    cmap_name = 'tab20'
                else:
                    cmap_name = 'jet'
                
                plotter.add_mesh(mesh, scalars='face_labels', cmap=cmap_name,
                               show_edges=show_edges, edge_color='black' if show_edges else None,
                               line_width=0.5 if show_edges else 0, opacity=1.0)
            
            # Plot key vertices and center vertices
            if PM is not None:
                # Plot key vertices (from PM['keys'])
                if 'keys' in PM and len(PM['keys']) > 0:
                    key_vertices = np.unique(PM['keys'][:, 1].astype(int))
                    key_vertices = key_vertices[key_vertices < len(self.X)]
                    if len(key_vertices) > 0:
                        key_points = pv.PolyData(self.X[key_vertices])
                        plotter.add_mesh(key_points, color='cyan', point_size=15, 
                                       render_points_as_spheres=True, label='Key vertices')
                
                # Plot center vertices (from PM['CV'])
                if 'CV' in PM:
                    cv_indices = []
                    for pix in range(len(PM['CV'])):
                        if PM['CV'][pix] is not None and PM['CV'][pix] < len(self.X):
                            cv_indices.append(int(PM['CV'][pix]))
                    if len(cv_indices) > 0:
                        cv_indices = np.array(cv_indices)
                        cv_points = pv.PolyData(self.X[cv_indices])
                        plotter.add_mesh(cv_points, color='blue', point_size=15, 
                                       render_points_as_spheres=True, label='Center vertices')
            
            plotter.add_text('Mesh Patches with Key Vertices', font_size=12)
            plotter.background_color = 'black'
            plotter.add_legend()
            plotter.show()
        except ImportError:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot the mesh with patch colors
                if self.face_labels is not None and pflag < 3:
                    ec = 'k' if pflag else 'none'
                    face_field = self.face_labels.copy()
                    
                    # Map face labels to distinct colors (same logic as plot_labels)
                    face_field = np.asarray(face_field, dtype=float)
                    unique_labels = np.unique(face_field)
                    unique_labels = unique_labels[unique_labels > 0]  # Remove any zero labels
                    n_labels = len(unique_labels)
                    
                    # Create triangles as vertex coordinates
                    triangles = []
                    for face in self.F:
                        triangle = [self.X[face[0]], self.X[face[1]], self.X[face[2]]]
                        triangles.append(triangle)
                    
                    # Get colors for each face
                    if n_labels <= 10:
                        cmap = plt.cm.tab10
                        n_colors = 10
                    elif n_labels <= 20:
                        cmap = plt.cm.tab20
                        n_colors = 20
                    else:
                        cmap = plt.cm.jet
                        n_colors = 256
                    
                    # Create a direct mapping from label to color
                    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                    
                    # Create color array directly
                    face_colors = np.zeros((len(face_field), 4))
                    for i, label in enumerate(face_field):
                        if label in label_to_idx:
                            color_idx = label_to_idx[label]
                            # Map to [0, 1] range for colormap
                            color_val = color_idx / max(1, n_labels - 1) if n_labels > 1 else 0.0
                            face_colors[i] = cmap(color_val)
                        else:
                            # Default color for unlabeled faces
                            face_colors[i] = [0.5, 0.5, 0.5, 0.9]
                    
                    # Create Poly3DCollection
                    poly = Poly3DCollection(triangles, facecolors=face_colors, 
                                            edgecolors=ec, linewidths=0.5, alpha=1.0)
                    ax.add_collection3d(poly)
                    
                    # Set axis limits
                    max_range = np.array([self.X[:, 0].max() - self.X[:, 0].min(),
                                          self.X[:, 1].max() - self.X[:, 1].min(),
                                          self.X[:, 2].max() - self.X[:, 2].min()]).max() / 2.0
                    mid_x = (self.X[:, 0].max() + self.X[:, 0].min()) * 0.5
                    mid_y = (self.X[:, 1].max() + self.X[:, 1].min()) * 0.5
                    mid_z = (self.X[:, 2].max() + self.X[:, 2].min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                else:
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                
                # Plot key vertices and center vertices (use smaller markers so they don't cover the mesh)
                if PM is not None:
                    # Use scatter plots for key/center vertices instead of large spheres
                    # Plot key vertices (from PM['keys'])
                    if 'keys' in PM and len(PM['keys']) > 0:
                        key_vertices = np.unique(PM['keys'][:, 1].astype(int))
                        key_vertices = key_vertices[key_vertices < len(self.X)]
                        if len(key_vertices) > 0:
                            ax.scatter(self.X[key_vertices, 0], self.X[key_vertices, 1], self.X[key_vertices, 2],
                                     color='cyan', s=150, marker='o', edgecolors='black', linewidths=1.5, 
                                     label='Key vertices', zorder=10)
                    
                    # Plot center vertices (from PM['CV'])
                    if 'CV' in PM:
                        cv_indices = []
                        for pix in range(len(PM['CV'])):
                            if PM['CV'][pix] is not None and PM['CV'][pix] < len(self.X):
                                cv_indices.append(int(PM['CV'][pix]))
                        if len(cv_indices) > 0:
                            cv_indices = np.array(cv_indices)
                            ax.scatter(self.X[cv_indices, 0], self.X[cv_indices, 1], self.X[cv_indices, 2],
                                     color='blue', s=150, marker='s', edgecolors='black', linewidths=1.5,
                                     label='Center vertices', zorder=10)
                    
                    if ('keys' in PM and len(PM['keys']) > 0) or ('CV' in PM):
                        ax.legend()
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Mesh Patches with Key Vertices')
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("No plotting library available")
        except Exception as e:
            print(f"Error in plot_patches: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_border(self):
        """Plot border vertices"""
        try:
            import pyvista as pv
            
            # Prepare faces with number of vertices
            num_faces = self.F.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), self.F))
            cells = faces_with_n_vertices.flatten()
            
            # Create mesh
            mesh = pv.PolyData(self.X, cells)
            
            # Color by face labels if available (segmentation coloring, same as plot_labels)
            if self.face_labels is not None:
                face_field = np.asarray(self.face_labels, dtype=float)
                unique_labels = np.unique(face_field)
                unique_labels = unique_labels[unique_labels > 0]  # Remove any zero labels
                n_labels = len(unique_labels)
                
                if n_labels > 1:
                    # Assign to cell_data so each face gets its patch color
                    mesh.cell_data['face_labels'] = face_field
                    # Choose colormap based on number of labels
                    if n_labels <= 10:
                        cmap_name = 'tab10'
                    elif n_labels <= 20:
                        cmap_name = 'tab20'
                    else:
                        cmap_name = 'jet'
                    scalars = 'face_labels'
                    cmap = cmap_name
                else:
                    scalars = None
                    cmap = None
            else:
                scalars = None
                cmap = None
            
            # Check if we're in Jupyter for interactive plots
            in_jupyter = False
            try:
                import sys
                in_jupyter = 'ipykernel' in sys.modules or 'IPython' in sys.modules
            except:
                pass
            
            # Use notebook=in_jupyter so interactive figures embed in Jupyter (trame)
            # Two subplots for different perspectives
            plotter = pv.Plotter(shape=(1, 2), notebook=in_jupyter)
            plotter.background_color = 'black'
            
            view_configs = [(45, 25), (225, 25)]  # (azimuth, elevation) - front-right and back-left
            for col, (az, el) in enumerate(view_configs):
                plotter.subplot(0, col)
                if scalars is not None:
                    plotter.add_mesh(mesh, scalars=scalars, cmap=cmap,
                                   show_edges=True, edge_color='black', line_width=0.3, opacity=1.0,
                                   scalar_bar_args={'title': 'Patch'})
                else:
                    plotter.add_mesh(mesh, color='lightblue',
                                   show_edges=True, edge_color='black', line_width=0.3, opacity=1.0)
                
                # Plot border vertices as large yellow spheres - make them clearly visible
                if self.border_vertex is not None:
                    indx = np.where(self.border_vertex)[0]
                    if len(indx) > 0:
                        border_points = pv.PolyData(self.X[indx])
                        label = f'Border vertices ({len(indx)})' if col == 0 else None
                        plotter.add_mesh(border_points, color='yellow', point_size=20,
                                       render_points_as_spheres=True, label=label)
                
                plotter.camera.azimuth = az
                plotter.camera.elevation = el
                plotter.add_text(f'View {col + 1}', font_size=10)
            
            plotter.subplot(0, 0)
            plotter.add_text('Mesh with Border Vertices', font_size=12)
            plotter.add_legend()
            plotter.show()
        except ImportError:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                fig = plt.figure(figsize=(16, 7))
                axes = fig.subplots(1, 2, subplot_kw={'projection': '3d'})
                
                # Create triangles
                triangles = []
                for face in self.F:
                    triangle = [self.X[face[0]], self.X[face[1]], self.X[face[2]]]
                    triangles.append(triangle)
                
                # Color by face labels if available (use same mapping as plot_labels)
                if self.face_labels is not None:
                    face_field = np.asarray(self.face_labels, dtype=float)
                    unique_labels = np.unique(face_field)
                    unique_labels = unique_labels[unique_labels > 0]  # Remove any zero labels
                    n_labels = len(unique_labels)
                    
                    if n_labels > 1:
                        if n_labels <= 10:
                            cmap = plt.cm.tab10
                            n_colors = 10
                        elif n_labels <= 20:
                            cmap = plt.cm.tab20
                            n_colors = 20
                        else:
                            cmap = plt.cm.jet
                            n_colors = 256
                        
                        # Create a direct mapping from label to color
                        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                        
                        # Create color array directly
                        face_colors = np.zeros((len(face_field), 4))
                        for i, label in enumerate(face_field):
                            if label in label_to_idx:
                                color_idx = label_to_idx[label]
                                # Map to [0, 1] range for colormap
                                color_val = color_idx / max(1, n_labels - 1) if n_labels > 1 else 0.0
                                face_colors[i] = cmap(color_val)
                            else:
                                # Default color for unlabeled faces
                                face_colors[i] = [0.5, 0.5, 0.5, 1.0]
                    else:
                        face_colors = 'lightblue'
                else:
                    face_colors = 'lightblue'
                
                # View angles: (elevation, azimuth) for view_init
                view_configs = [(25, 45), (25, 225)]  # front-right and back-left
                for ax, (elev, azim) in zip(axes, view_configs):
                    poly = Poly3DCollection(triangles, facecolors=face_colors, 
                                            edgecolors='k', linewidths=0.3, alpha=1.0)
                    ax.add_collection3d(poly)
                    
                    # Plot border vertices as large yellow stars - make them clearly visible
                    if self.border_vertex is not None:
                        indx = np.where(self.border_vertex)[0]
                        if len(indx) > 0:
                            ax.scatter(self.X[indx, 0], self.X[indx, 1], self.X[indx, 2],
                                      color='yellow', s=200, marker='*', 
                                      edgecolors='red', linewidths=1.5,
                                      label=f'Border vertices ({len(indx)})', zorder=10)
                            ax.legend()
                    
                    max_range = np.array([self.X[:, 0].max() - self.X[:, 0].min(),
                                          self.X[:, 1].max() - self.X[:, 1].min(),
                                          self.X[:, 2].max() - self.X[:, 2].min()]).max() / 2.0
                    mid_x = (self.X[:, 0].max() + self.X[:, 0].min()) * 0.5
                    mid_y = (self.X[:, 1].max() + self.X[:, 1].min()) * 0.5
                    mid_z = (self.X[:, 2].max() + self.X[:, 2].min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.view_init(elev=elev, azim=azim)
                
                axes[0].set_title('View 1')
                axes[1].set_title('View 2')
                fig.suptitle('Mesh with Border Vertices', fontsize=14)
                fig.patch.set_facecolor('black')
                for ax in axes:
                    ax.set_facecolor('black')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("No plotting library available")
    
    @staticmethod
    def plot_simplified_mesh(PM, show_keys=True, show_cv=True, subplot_shape=(1, 2),
                            export_html_path='simplified_mesh_plot.html'):
        """
        Plot the simplified patch-level mesh from PM.

        Uses two subplots by default (different viewing angles) so the mesh is
        visible even with static backend. Exports to HTML by default for
        interactive inspection in a browser (always works when opened in browser).

        Parameters
        ----------
        PM : dict
            Patch mesh structure with 'pm' (simplified mesh)
        show_keys : bool
            Show key vertices (cyan)
        show_cv : bool
            Show center vertices (blue)
        subplot_shape : tuple
            (nrows, ncols) for multi-view. (1,2) = two side-by-side views.
            Use (1,1) for single view.
        export_html_path : str or None
            Path to export interactive HTML. None to skip. Default exports to
            'simplified_mesh_plot.html' in current directory.
        """
        if PM is None or 'pm' not in PM:
            print("PM must contain 'pm' (simplified mesh)")
            return
        
        pm = PM['pm']
        if pm.face_labels is None:
            print("Simplified mesh has no face_labels")
            return
        
        # Check if mesh has faces
        num_faces = pm.F.shape[0] if pm.F is not None and len(pm.F) > 0 else 0
        if num_faces == 0:
            print("WARNING: Simplified mesh has 0 faces - cannot plot mesh structure")
            print(f"  Mesh has {len(pm.X)} vertices but no faces")
            print(f"  This suggests face generation failed in patch_info_gen")
            # Still plot vertices as points
            try:
                import pyvista as pv
                in_jupyter = _ensure_pyvista_jupyter_backend()
                plotter = pv.Plotter(notebook=in_jupyter)
                if len(pm.X) > 0:
                    points = pv.PolyData(pm.X)
                    plotter.add_mesh(points, color='red', point_size=10, 
                                   render_points_as_spheres=True)
                    plotter.add_text('Simplified mesh vertices (no faces)', font_size=12)
                    plotter.show()
                return
            except ImportError:
                print("PyVista not available for plotting")
                return
        
        try:
            import pyvista as pv
            import os

            # Prepare faces with number of vertices
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), pm.F))
            cells = faces_with_n_vertices.flatten()

            # Create mesh
            mesh = pv.PolyData(pm.X, cells)
            face_field = np.asarray(pm.face_labels, dtype=float)
            mesh.cell_data['face_labels'] = face_field

            # Get unique labels
            unique_labels = np.unique(face_field)
            n_labels = len(unique_labels)

            # Choose colormap
            if n_labels <= 10:
                cmap_name = 'tab10'
            elif n_labels <= 20:
                cmap_name = 'tab20'
            else:
                cmap_name = 'jet'

            # Use 'static' for reliable in-notebook display; HTML export gives interactive view
            in_jupyter = _ensure_pyvista_jupyter_backend()
            use_static = True  # Most reliable; HTML export provides interactive alternative
            if use_static:
                try:
                    pv.set_jupyter_backend('static')
                except Exception:
                    pass

            nrows, ncols = subplot_shape if subplot_shape else (1, 1)
            plotter = pv.Plotter(shape=(nrows, ncols), notebook=in_jupyter)
            plotter.background_color = 'black'

            # Camera positions for two views: front and side (or isometric)
            views = [
                [(0, 0, 1.5), (0, 0, 0), (0, 1, 0)],   # front
                [(1.2, 1.2, 1.2), (0, 0, 0), (0, 1, 0)],  # isometric
            ]

            def add_mesh_content(ax_plotter):
                ax_plotter.add_mesh(mesh, scalars='face_labels', cmap=cmap_name,
                                   show_edges=True, edge_color='black', line_width=0.5, opacity=1.0)
                if show_keys and 'keys' in PM and len(PM['keys']) > 0:
                    key_vertices = np.unique(PM['keys'][:, 1].astype(int))
                    if 'Xkeyind' in PM:
                        key_points = []
                        for vix in key_vertices:
                            idx_in_pm = np.where(PM['Xkeyind'] == vix)[0]
                            if len(idx_in_pm) > 0 and idx_in_pm[0] < len(pm.X):
                                key_points.append(pm.X[idx_in_pm[0]])
                        if len(key_points) > 0:
                            key_poly = pv.PolyData(np.array(key_points))
                            ax_plotter.add_mesh(key_poly, color='cyan', point_size=20,
                                             render_points_as_spheres=True)
                if show_cv and 'CV' in PM and 'Xkeyind' in PM:
                    nkeys = len(np.unique(PM['keys'][:, 1])) if len(PM['keys']) > 0 else 0
                    cv_points = []
                    for pix in range(len(PM['CV'])):
                        if PM['CV'][pix] is not None:
                            cv_idx_in_pm = nkeys + pix
                            if cv_idx_in_pm < len(pm.X):
                                cv_points.append(pm.X[cv_idx_in_pm])
                    if len(cv_points) > 0:
                        cv_poly = pv.PolyData(np.array(cv_points))
                        ax_plotter.add_mesh(cv_poly, color='blue', point_size=20,
                                         render_points_as_spheres=True)

            for idx in range(nrows * ncols):
                plotter.subplot(idx // ncols, idx % ncols)
                add_mesh_content(plotter)
                label = 'View 1 (front)' if idx == 0 else ('View 2 (isometric)' if idx == 1 else f'View {idx+1}')
                plotter.add_text(f'{label} — {n_labels} patches', font_size=10)
                if idx < len(views):
                    plotter.camera_position = views[idx]
                plotter.reset_camera()

            # Export to HTML for interactive inspection (open in browser to rotate)
            if export_html_path:
                try:
                    out_path = os.path.abspath(export_html_path)
                    plotter.export_html(out_path)
                    print(f"Exported interactive plot to: {out_path}")
                    print("  (Open in browser to rotate and inspect)")
                except Exception as e:
                    try:
                        plotter.screenshot(export_html_path.replace('.html', '.png'))
                        print(f"HTML export failed ({e}); saved screenshot to {export_html_path.replace('.html', '.png')}")
                    except Exception:
                        print(f"Could not export plot: {e}")

            if in_jupyter:
                return plotter.show(return_viewer=True)
            plotter.show()
        except ImportError:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Map face labels to distinct colors
                face_field = np.asarray(pm.face_labels, dtype=float)
                unique_labels = np.unique(face_field)
                n_labels = len(unique_labels)
                
                if n_labels > 1:
                    label_to_color = {label: idx / (n_labels - 1) for idx, label in enumerate(unique_labels)}
                    face_field_norm = np.array([label_to_color[label] for label in face_field])
                else:
                    face_field_norm = np.zeros_like(face_field)
                
                # Create triangles
                triangles = []
                for face in pm.F:
                    triangle = [pm.X[face[0]], pm.X[face[1]], pm.X[face[2]]]
                    triangles.append(triangle)
                
                # Get colors
                if n_labels <= 10:
                    cmap = plt.cm.tab10
                elif n_labels <= 20:
                    cmap = plt.cm.tab20
                else:
                    cmap = plt.cm.jet
                face_colors = cmap(face_field_norm)
                
                # Create Poly3DCollection
                poly = Poly3DCollection(triangles, facecolors=face_colors, 
                                        edgecolors='k', linewidths=0.5, alpha=1.0)
                ax.add_collection3d(poly)
                
                # Plot key vertices if requested
                if show_keys and 'keys' in PM and len(PM['keys']) > 0:
                    key_vertices = np.unique(PM['keys'][:, 1].astype(int))
                    # Map to simplified mesh indices via Xkeyind
                    if 'Xkeyind' in PM:
                        for vix in key_vertices:
                            idx_in_pm = np.where(PM['Xkeyind'] == vix)[0]
                            if len(idx_in_pm) > 0 and idx_in_pm[0] < len(pm.X):
                                ax.scatter(pm.X[idx_in_pm[0], 0], pm.X[idx_in_pm[0], 1], pm.X[idx_in_pm[0], 2],
                                         color='cyan', s=200, marker='o', edgecolors='black', linewidths=2)
                
                # Plot center vertices if requested
                if show_cv and 'CV' in PM and 'Xkeyind' in PM:
                    nkeys = len(np.unique(PM['keys'][:, 1])) if len(PM['keys']) > 0 else 0
                    for pix in range(len(PM['CV'])):
                        if PM['CV'][pix] is not None:
                            cv_idx_in_pm = nkeys + pix
                            if cv_idx_in_pm < len(pm.X):
                                ax.scatter(pm.X[cv_idx_in_pm, 0], pm.X[cv_idx_in_pm, 1], pm.X[cv_idx_in_pm, 2],
                                         color='blue', s=200, marker='s', edgecolors='black', linewidths=2)
                
                # Set axis limits
                max_range = np.array([pm.X[:, 0].max() - pm.X[:, 0].min(),
                                      pm.X[:, 1].max() - pm.X[:, 1].min(),
                                      pm.X[:, 2].max() - pm.X[:, 2].min()]).max() / 2.0
                mid_x = (pm.X[:, 0].max() + pm.X[:, 0].min()) * 0.5
                mid_y = (pm.X[:, 1].max() + pm.X[:, 1].min()) * 0.5
                mid_z = (pm.X[:, 2].max() + pm.X[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'Simplified Patch Mesh ({n_labels} patches)')
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("No plotting library available")
        except Exception as e:
            print(f"Error plotting simplified mesh: {e}")
            import traceback
            traceback.print_exc()

    def export_blender(self, filename, script_path=None, run_blender=False, blender_executable=None):
        """
        Export surface_mesh to a Blender 5.0 project with showcase scene.

        Creates a Python script that, when run inside Blender, builds a scene with:
        - The mesh object with mildly reflective material
        - Dark background
        - Spot lights illuminating the shape
        - Reflective water plane that the shape hovers above

        Parameters
        ----------
        filename : str
            Output path for the .blend file.
        script_path : str, optional
            Path for the generated Blender Python script. Default: same directory
            as filename, with suffix '_blender_setup.py'.
        run_blender : bool, optional
            If True, attempt to run Blender to generate the .blend file.
            Requires Blender to be installed and in PATH (or blender_executable).
        blender_executable : str, optional
            Path to Blender executable. If None, uses 'blender' from PATH.

        Returns
        -------
        script_path : str
            Path to the generated Python script.

        Notes
        -----
        To generate the .blend file, run:
            blender --background --python <script_path>

        Or set run_blender=True when Blender is installed.
        """
        from .blender_export import export_blender as _export_blender
        return _export_blender(self, filename, script_path=script_path,
                               run_blender=run_blender, blender_executable=blender_executable)


def export_patch_parameterization_html(PM, output_dir, prefix='patch', show_reference_sphere=True):
    """
    Export per-patch spherical parameterization to HTML files for browser inspection.

    Creates one HTML file per patch (e.g. patch_0_sphere.html, patch_1_sphere.html)
    showing each patch's fine mesh on the unit sphere. Use this to inspect the
    final fine-mesh spherical parameterization before spherical harmonics projection.

    Parameters
    ----------
    PM : dict
        Patch mesh structure with PM['P'][pix][0] = surface_mesh for each patch.
        Patches must have t and p attributes (parameterization).
    output_dir : str
        Directory for output HTML files.
    prefix : str, optional
        Filename prefix. Default 'patch' gives patch_0_sphere.html, etc.
    show_reference_sphere : bool, optional
        If True, draw a semi-transparent reference sphere. Default True.

    Returns
    -------
    list of str
        Paths to the exported HTML files.
    """
    import os
    from .utils import kk_sph2cart

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("export_patch_parameterization_html requires pyvista")

    for pix in range(PM['npatches']):
        patm = PM['P'][pix][0]
        html_path = os.path.join(output_dir, f'{prefix}_{pix}_sphere.html')

        plotter = pv.Plotter(off_screen=True)
        if show_reference_sphere:
            sphere = pv.Sphere(radius=0.98, theta_resolution=30, phi_resolution=30)
            plotter.add_mesh(sphere, color='cyan', opacity=0.15, show_edges=False)

        if patm.t is not None and patm.p is not None and np.any(patm.t != 0):
            u, v, w = kk_sph2cart(patm.t, patm.p, np.ones(len(patm.p)))
            X_sph = np.column_stack([u, v, w])
            nf = patm.F.shape[0]
            cells = np.hstack((np.full((nf, 1), 3), patm.F)).flatten()
            mesh_sph = pv.PolyData(X_sph, cells)
            plotter.add_mesh(mesh_sph, color='lightblue', show_edges=True,
                           edge_color='black', line_width=0.5, opacity=0.9)
            if hasattr(patm, 'border_vertex') and patm.border_vertex is not None:
                bv = np.where(patm.border_vertex.astype(bool))[0]
                bv = bv[(patm.t[bv] != 0) | (patm.p[bv] != 0)]
                if len(bv) > 0:
                    plotter.add_mesh(pv.PolyData(X_sph[bv]), color='yellow',
                                   point_size=6, render_points_as_spheres=True)

        plotter.add_text(f'Patch {pix} on sphere ({patm.F.shape[0]} faces)', font_size=12)
        plotter.background_color = 'white'
        plotter.export_html(html_path)
        plotter.close()
        paths.append(html_path)

    return paths
