"""
Spherical harmonics parameterized surface class
Translated from MATLAB @shp_surface class
"""

import numpy as np
from .sh_basis import sh_basis
from .sh_surface import sh_surface
from .utils import indices_gen, kk_cart2sph, kk_sph2cart, kk_cross
from scipy.linalg import svd


class shp_surface:
    """
    Spherical harmonics parameterized surface
    """
    
    def __init__(self, arg1=None, arg2=None, arg3=None):
        """
        Initialize shp_surface
        
        Multiple initialization modes:
        - shp_surface() -> sphere
        - shp_surface(L_max) -> sphere with L_max
        - shp_surface(L_max, basis) -> sphere with basis
        - shp_surface(L_max, basis, mesh) -> from mesh
        - shp_surface(mesh) -> from mesh (L_max=16)
        """
        self.name = 'untitled'
        self.basis = None
        self.L_max = None
        self.trans = []
        self.gdim = 30
        self.residual = 0
        self.xc = None
        self.yc = None
        self.zc = None
        self.x = None
        self.y = None
        self.z = None
        self.A = None
        self.V = None
        self.v = None
        self.H = None
        self.KG = None
        self.T = None
        self.h = None
        self.S = None
        self.Eb = None
        self.da_bc = None
        self.M = None
        self.X_o = None
        self.X_1 = None
        self.X_2 = None
        self.ang = None
        self.res_p = None
        self.res_o = None
        self.needs_updating = True
        self.sf = []
        self.use_camorbit = False
        self.edge_color = 'none'
        self.map_gen = 1
        self.m_sphere = None
        self.Rmx = []
        
        # Handle different initialization modes
        if arg1 is None:
            # Make a sphere
            L_max = 12
            self.basis = sh_basis(L_max, self.gdim)
            xc, yc, zc = self.shp_sphere_gen(L_max, 1)
            X_o = self.tr([xc, yc, zc], L_max)
            self.xc, self.yc, self.zc = self.get_xyz_clks(X_o)
            self.X_o = np.concatenate([self.xc, self.yc, self.zc])
            self.L_max = L_max
            self.update()
        elif isinstance(arg1, str):
            # Load from file name
            if arg1.endswith('.shp3'):
                self.import_shp3(arg1, arg2 if arg2 is not None else 60)
            else:
                raise NotImplementedError("Loading from .mat files not yet implemented")
        elif hasattr(arg1, 'X') and hasattr(arg1, 'F'):
            # Initialize from surface_mesh
            from .surface_mesh import surface_mesh
            if isinstance(arg1, surface_mesh):
                # If arg2 is provided and is an integer, use it as L_max
                # Otherwise use default L_max=16
                if isinstance(arg2, (int, np.integer)):
                    self.L_max = int(arg2)
                else:
                    self.L_max = 16
                self.basis = sh_basis(self.L_max, self.gdim)
                self.mesh2shp(arg1, self.L_max)
        elif isinstance(arg1, (int, np.integer)):
            L_max = arg1
            if arg2 is None:
                # Just L_max
                self.basis = sh_basis(L_max, self.gdim)
                xc, yc, zc = self.shp_sphere_gen(L_max, 1, sh_basis.N_LK_bosh)
                X_o = self.tr([xc, yc, zc], L_max)
                self.xc, self.yc, self.zc = self.get_xyz_clks(X_o)
                self.X_o = np.concatenate([self.xc, self.yc, self.zc])
                self.L_max = L_max
                self.update()
            elif isinstance(arg2, sh_basis):
                # L_max and basis
                self.L_max = L_max
                self.basis = arg2
                self.gdim = arg2.gdim
                xc, yc, zc = self.shp_sphere_gen(L_max, 1, sh_basis.N_LK_bosh)
                X_o = self.tr([xc, yc, zc], L_max)
                self.xc, self.yc, self.zc = self.get_xyz_clks(X_o)
                self.X_o = np.concatenate([self.xc, self.yc, self.zc])
                self.needs_updating = True
                self.update()
            elif hasattr(arg2, 'X'):
                # L_max and mesh
                from .surface_mesh import surface_mesh
                if isinstance(arg2, surface_mesh):
                    self.L_max = L_max
                    self.basis = sh_basis(self.L_max, self.gdim)
                    self.mesh2shp(arg2, self.L_max)
        elif isinstance(arg1, (list, np.ndarray)) and len(arg1) > 1:
            # Initialize with X_o directly
            L_max = self.get_L_max(arg1)
            self.basis = sh_basis(L_max, self.gdim)
            self.X_o = np.asarray(arg1).flatten()
            self.xc, self.yc, self.zc = self.get_xyz_clks(self.X_o)
            self.L_max = L_max
            self.update()
    
    @staticmethod
    def indices_gen(c):
        """Generate l, m indices"""
        return indices_gen(c)
    
    @staticmethod
    def shp_sphere_gen(L_max, R=1, hfn=None):
        """
        Generate sphere coefficients
        
        For a sphere, we need to set the l=1 coefficients:
        - xc[3] (l=1, m=1) = R/hfn(1,1)
        - yc[1] (l=1, m=-1) = R/hfn(1,-1)
        - zc[2] (l=1, m=0) = R/hfn(1,0)
        """
        from .sh_basis import sh_basis
        
        xc = np.zeros((L_max + 1)**2)
        yc = np.zeros((L_max + 1)**2)
        zc = np.zeros((L_max + 1)**2)
        
        # Use N_LK_bosh as normalization function if not provided
        if hfn is None:
            hfn = sh_basis.N_LK_bosh
        
        # Set l=1 coefficients (indices are 0-indexed in Python)
        # Index 1: l=1, m=-1
        # Index 2: l=1, m=0
        # Index 3: l=1, m=1
        xc[3] = R / hfn(1, 1)   # l=1, m=1
        yc[1] = R / hfn(1, -1)  # l=1, m=-1
        zc[2] = R / hfn(1, 0)   # l=1, m=0
        
        return xc, yc, zc
    
    @staticmethod
    def tr(X_o, L_max):
        """Truncate X_o to L_max"""
        X_o = np.asarray(X_o).flatten()
        trunc = (L_max + 1)**2 * 3
        if len(X_o) < trunc:
            X_new = np.zeros(trunc)
            X_new[:len(X_o)] = X_o
            return X_new
        else:
            return X_o[:trunc]
    
    @staticmethod
    def get_xyz_clks(X_o):
        """Extract xc, yc, zc from X_o"""
        X_o = np.asarray(X_o).flatten()
        n = len(X_o) // 3
        xc = X_o[:n]
        yc = X_o[n:2*n]
        zc = X_o[2*n:3*n]
        return xc, yc, zc
    
    @staticmethod
    def get_L_max(X_o):
        """Get L_max from X_o"""
        X_o = np.asarray(X_o).flatten()
        n = len(X_o) // 3
        return int(np.sqrt(n) - 1)
    
    def update(self):
        """Update surface from coefficients"""
        if self.needs_updating:
            lb = (self.L_max + 1)**2
            gdimp = self.basis.p.shape[0]
            gdimt = self.basis.p.shape[1] if self.basis.p.ndim > 1 else 1
            
            # Compute x, y, z coordinates
            c_x = self.xc[:lb].reshape(1, 1, -1)
            c_x = np.tile(c_x, (gdimp, gdimt, 1))
            x = np.sum(c_x * self.basis.Y[:, :, :lb], axis=2)
            
            c_y = self.yc[:lb].reshape(1, 1, -1)
            c_y = np.tile(c_y, (gdimp, gdimt, 1))
            y = np.sum(c_y * self.basis.Y[:, :, :lb], axis=2)
            
            c_z = self.zc[:lb].reshape(1, 1, -1)
            c_z = np.tile(c_z, (gdimp, gdimt, 1))
            z = np.sum(c_z * self.basis.Y[:, :, :lb], axis=2)
            
            self.x = x
            self.y = y
            self.z = z
            self.needs_updating = False
        
        return self
    
    def update_full(self):
        """
        Update surface and compute all geometric properties
        (area, volume, mean curvature H, Gaussian curvature KG, etc.)
        
        Note: This always computes geometric properties, even if coordinates
        are already updated. Use update() for faster coordinate-only updates.
        """
        # Validate that basis exists and has required attributes
        if self.basis is None:
            raise ValueError("Cannot update: basis is None. Initialize shp_surface with a basis.")
        
        if not hasattr(self.basis, 'Y') or self.basis.Y is None:
            raise ValueError("Cannot update: basis.Y is None. Basis may not be properly initialized.")
        
        if not hasattr(self.basis, 'Y_P') or self.basis.Y_P is None:
            raise ValueError("Cannot update: basis.Y_P (phi derivative) is None.")
        
        if not hasattr(self.basis, 'Y_T') or self.basis.Y_T is None:
            raise ValueError("Cannot update: basis.Y_T (theta derivative) is None.")
        
        if not hasattr(self.basis, 'Y_PP') or self.basis.Y_PP is None:
            raise ValueError("Cannot update: basis.Y_PP (phi-phi derivative) is None.")
        
        if not hasattr(self.basis, 'Y_TT') or self.basis.Y_TT is None:
            raise ValueError("Cannot update: basis.Y_TT (theta-theta derivative) is None.")
        
        if not hasattr(self.basis, 'Y_TP') or self.basis.Y_TP is None:
            raise ValueError("Cannot update: basis.Y_TP (theta-phi derivative) is None.")
        
        # First ensure coordinates are updated if needed
        if self.needs_updating:
            self.update()
        
        # Always compute geometric properties (even if coordinates were already updated)
        lb = (self.L_max + 1)**2
        gdimp = self.basis.p.shape[0]
        gdimt = self.basis.p.shape[1] if self.basis.p.ndim > 1 else 1
        
        # Use existing coordinates if available, otherwise compute
        if self.x is not None and self.y is not None and self.z is not None:
            x = self.x
            y = self.y
            z = self.z
        else:
            # Compute x, y, z coordinates
            c_x = self.xc[:lb].reshape(1, 1, -1)
            c_x = np.tile(c_x, (gdimp, gdimt, 1))
            x = np.sum(c_x * self.basis.Y[:, :, :lb], axis=2)
            
            c_y = self.yc[:lb].reshape(1, 1, -1)
            c_y = np.tile(c_y, (gdimp, gdimt, 1))
            y = np.sum(c_y * self.basis.Y[:, :, :lb], axis=2)
            
            c_z = self.zc[:lb].reshape(1, 1, -1)
            c_z = np.tile(c_z, (gdimp, gdimt, 1))
            z = np.sum(c_z * self.basis.Y[:, :, :lb], axis=2)
        
        # Always compute derivatives (needed for geometric properties)
        c_x = self.xc[:lb].reshape(1, 1, -1)
        c_x = np.tile(c_x, (gdimp, gdimt, 1))
        xp = np.sum(c_x * self.basis.Y_P[:, :, :lb], axis=2)
        xt = np.sum(c_x * self.basis.Y_T[:, :, :lb], axis=2)
        xpp = np.sum(c_x * self.basis.Y_PP[:, :, :lb], axis=2)
        xtt = np.sum(c_x * self.basis.Y_TT[:, :, :lb], axis=2)
        xtp = np.sum(c_x * self.basis.Y_TP[:, :, :lb], axis=2)
        
        c_y = self.yc[:lb].reshape(1, 1, -1)
        c_y = np.tile(c_y, (gdimp, gdimt, 1))
        yp = np.sum(c_y * self.basis.Y_P[:, :, :lb], axis=2)
        yt = np.sum(c_y * self.basis.Y_T[:, :, :lb], axis=2)
        ypp = np.sum(c_y * self.basis.Y_PP[:, :, :lb], axis=2)
        ytt = np.sum(c_y * self.basis.Y_TT[:, :, :lb], axis=2)
        ytp = np.sum(c_y * self.basis.Y_TP[:, :, :lb], axis=2)
        
        c_z = self.zc[:lb].reshape(1, 1, -1)
        c_z = np.tile(c_z, (gdimp, gdimt, 1))
        zp = np.sum(c_z * self.basis.Y_P[:, :, :lb], axis=2)
        zt = np.sum(c_z * self.basis.Y_T[:, :, :lb], axis=2)
        zpp = np.sum(c_z * self.basis.Y_PP[:, :, :lb], axis=2)
        ztt = np.sum(c_z * self.basis.Y_TT[:, :, :lb], axis=2)
        ztp = np.sum(c_z * self.basis.Y_TP[:, :, :lb], axis=2)
        
        # Calculate first and second fundamental forms
        X = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        Xt = np.column_stack([xt.flatten(), yt.flatten(), zt.flatten()])
        Xp = np.column_stack([xp.flatten(), yp.flatten(), zp.flatten()])
        Xpp = np.column_stack([xpp.flatten(), ypp.flatten(), zpp.flatten()])
        Xtp = np.column_stack([xtp.flatten(), ytp.flatten(), ztp.flatten()])
        Xtt = np.column_stack([xtt.flatten(), ytt.flatten(), ztt.flatten()])
        
        # First fundamental form coefficients
        E = np.sum(Xt * Xt, axis=1)
        F = np.sum(Xt * Xp, axis=1)
        G = np.sum(Xp * Xp, axis=1)
        
        # Normal vector
        SS = kk_cross(Xt, Xp)
        SSn = np.sqrt(E * G - F * F)
        SSn_safe = SSn + 1e-10  # Avoid division by zero
        n = SS / SSn_safe[:, np.newaxis]
        
        # Second fundamental form coefficients
        L_coeff = np.sum(Xtt * n, axis=1)
        M_coeff = np.sum(Xtp * n, axis=1)
        N_coeff = np.sum(Xpp * n, axis=1)
        
        # Geometric properties
        self.V = abs(1/3. * np.sum(self.basis.w * np.sum(X * n, axis=1) * SSn))
        self.A = np.sum(self.basis.w * SSn)
        
        # Mean curvature H
        denom = 2 * (E * G - F * F)
        denom_safe = denom + 1e-10
        self.H = (E * N_coeff + G * L_coeff - 2 * F * M_coeff) / denom_safe
        self.H = self.H.reshape(x.shape)
        
        # Gaussian curvature KG
        self.KG = (L_coeff * N_coeff - M_coeff * M_coeff) / denom_safe
        self.KG = self.KG.reshape(x.shape)
        
        # Other quantities
        self.M = 1/2 * np.sum(2 * self.H.flatten() * self.basis.w * SSn)
        if self.A > 0:
            self.h = 1./self.A * np.sum(self.H.flatten() * self.basis.w * SSn)
        else:
            self.h = 0
        self.T = np.sum(self.KG.flatten() * self.basis.w * SSn) / 4 / np.pi
        self.S = np.sqrt(2 * self.H**2 - self.KG)  # Curvedness
        self.Eb = 1/2 * np.sum((2 * self.H.flatten())**2 * self.basis.w * SSn) / 8 / np.pi
        
        # Reduced volume
        r = np.sqrt(self.A / 4 / np.pi) if self.A > 0 else 1.0
        V_sphere = 4/3 * np.pi * r**3 if r > 0 else 1.0
        self.v = self.V / V_sphere if V_sphere > 0 else 0
        self.da_bc = self.M / 4 / np.pi / r**3 if r > 0 else 0
        
        # Store coordinates
        self.x = x
        self.y = y
        self.z = z
        
        return self
    
    def mesh2shp(self, m, L_max=None):
        """Convert mesh to spherical harmonics"""
        if L_max is None:
            L_max = self.L_max
        
        # Check if mesh already has valid t and p values (already parameterized)
        has_valid_tp = (hasattr(m, 't') and hasattr(m, 'p') and 
                       m.t is not None and m.p is not None and
                       len(m.t) > 0 and len(m.p) > 0 and
                       np.any(m.t != 0) and np.any(m.p != 0))
        
        # Map mesh to sphere if needed (only if not already parameterized)
        if (self.m_sphere is None or self.map_gen == 1) and not has_valid_tp:
            m = m.map2sphere() if hasattr(m, 'map2sphere') else m
            self.m_sphere = m
            self.map_gen = 0
        elif has_valid_tp:
            # Mesh is already parameterized - skip map2sphere
            # Set needs_map2sphere to False to prevent any future calls
            if hasattr(m, 'needs_map2sphere'):
                m.needs_map2sphere = False
            self.m_sphere = m
            self.map_gen = 0
        
        # Perform SH analysis
        if hasattr(m, 't') and hasattr(m, 'p') and m.t is not None and m.p is not None:
            self.shp_analysis(m.X, m.t, m.p, L_max)
        else:
            # Need to compute t, p from X
            t, p, _ = kk_cart2sph(m.X[:, 0], m.X[:, 1], m.X[:, 2])
            self.shp_analysis(m.X, t, p, L_max)
        
        return self
    
    def shp_analysis(self, X, t, p, L_max):
        """Spherical harmonics analysis"""
        if np.any(np.isnan(p)) or np.any(np.isnan(t)):
            print('NaN found in phi or theta: aborting')
            return self
        
        L, K, _ = self.indices_gen(np.arange(1, (L_max + 1)**2 + 1))
        M = len(L)
        N = len(X)
        
        # Build basis matrix
        A = np.zeros((N, M))
        for S in range(len(L)):
            A[:, S] = self.basis.ylk_bosh(L[S], K[S], p, t).flatten()
        
        # Solve using SVD
        # MATLAB: [U, S, V] = svd(A, 'econ')
        # Note: scipy's svd returns Vh (conjugate transpose of V), so Vh.T = V
        U, S, Vh = svd(A, full_matrices=False)
        
        # MATLAB: invS = 1./(S); invS(invS==inf) = 0
        invS = 1.0 / (S + 1e-10)
        invS[invS == np.inf] = 0
        
        # MATLAB: clks = (V*invS) * (U'*X)
        # V*invS where invS is diagonal: we can use broadcasting
        # U'*X = U.T @ X gives (M x N) @ (N x 3) = (M x 3)
        U_T_X = U.T @ X  # (M x 3)
        
        # V*invS: V is Vh.T, invS is diagonal
        # We can compute this as: Vh.T @ diag(invS) = (M x M) @ (M x M) = (M x M)
        V_invS = Vh.T @ np.diag(invS)  # (M x M)
        
        # Then: (V*invS) * (U'*X) = (M x M) @ (M x 3) = (M x 3)
        clks = V_invS @ U_T_X  # (M x 3)
        
        r = A @ clks - X
        self.residual = r
        
        # Reshape to X_o format
        # MATLAB uses column-major (Fortran) order: reshape(clks,1,M*3)
        # This produces [xc[0], xc[1], ..., xc[M-1], yc[0], yc[1], ..., yc[M-1], zc[0], ...]
        # clks has shape (M, 3) where columns are x, y, z coefficients
        self.X_o = clks.flatten('F')  # Use Fortran (column-major) order to match MATLAB
        self.xc, self.yc, self.zc = self.get_xyz_clks(self.X_o)
        self.L_max = L_max
        
        return self
    
    def get_mesh(self, nico=3, Y_LK_in=None, C=None):
        """
        Get mesh representation of shp_surface
        
        Parameters:
        -----------
        nico : int
            Icosahedron subdivision level
        Y_LK_in : array, optional
            Precomputed basis matrix
        C : array, optional
            Precomputed face connectivity
            
        Returns:
        --------
        m : surface_mesh
            Mesh object
        X : array
            Vertex coordinates (N x 3)
        F : array
            Face connectivity (M x 3)
        Y_LK : array
            Basis matrix (N x M)
        t, p : array
            Spherical coordinates
        """
        self.update()
        
        if Y_LK_in is None:
            # Build the basis
            from .surface_mesh import surface_mesh
            X_sphere, F_sphere = surface_mesh.sphere_mesh_gen(nico)
            t, p, r = kk_cart2sph(X_sphere[:, 0], X_sphere[:, 1], X_sphere[:, 2])
            
            # Generate basis at vertices
            L, K, _ = self.indices_gen(np.arange(1, (self.L_max + 1)**2 + 1))
            M = len(L)
            N = len(t)
            Y_LK = np.zeros((N, M), dtype=np.float32)
            
            for S in range(len(L)):
                Y_LK[:, S] = self.basis.ylk_bosh(L[S], K[S], p, t).flatten()
            
            # Compute 3D coordinates from xc, yc, zc coefficients
            # MATLAB: X = Y_LK(:,1:length(obj.xc))* [obj.xc(:) obj.yc(:) obj.zc(:)];
            lb = len(self.xc)
            
            # Ensure we have enough basis functions
            if Y_LK.shape[1] < lb:
                raise ValueError(f"Not enough basis functions: have {Y_LK.shape[1]}, need {lb}")
            
            # Extract coefficients (ensure they're column vectors like MATLAB)
            # MATLAB: obj.xc(:) makes it a column vector
            xc_vec = np.asarray(self.xc[:lb]).flatten()
            yc_vec = np.asarray(self.yc[:lb]).flatten()
            zc_vec = np.asarray(self.zc[:lb]).flatten()
            
            # MATLAB: [obj.xc(:) obj.yc(:) obj.zc(:)] creates (lb x 3) matrix
            # Column stack creates the same: each vector becomes a column
            coeff_matrix = np.column_stack([xc_vec, yc_vec, zc_vec])
            
            # MATLAB: Y_LK(:,1:length(obj.xc)) * coeff_matrix
            # This gives (N x lb) @ (lb x 3) = (N x 3)
            X = Y_LK[:, :lb] @ coeff_matrix
            
            # Debug: Check if reconstruction makes sense
            # The reconstructed mesh should have reasonable coordinates
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                import warnings
                warnings.warn(f"get_mesh: Reconstructed coordinates contain NaN/Inf. "
                            f"xc range: [{xc_vec.min():.3f}, {xc_vec.max():.3f}], "
                            f"yc range: [{yc_vec.min():.3f}, {yc_vec.max():.3f}], "
                            f"zc range: [{zc_vec.min():.3f}, {zc_vec.max():.3f}]")
            
            # Check for and fix any vertices at origin (numerical precision issue)
            # For a sphere, all vertices should be at distance ~1.0
            distances = np.linalg.norm(X, axis=1)
            zero_mask = distances < 1e-6
            if np.any(zero_mask):
                # For vertices at origin, use the spherical coordinates to reconstruct
                # This happens when basis functions evaluate to zero at special angles
                for idx in np.where(zero_mask)[0]:
                    # Reconstruct from spherical coordinates using the sphere coefficients
                    # For a unit sphere: x = sin(t)*cos(p), y = sin(t)*sin(p), z = cos(t)
                    # But we need to use the SH representation
                    # For now, project to unit sphere in the direction of the original sphere mesh
                    if distances[idx] < 1e-6:
                        # Use the original sphere mesh direction
                        X_orig = X_sphere[idx]
                        X[idx] = X_orig / np.linalg.norm(X_orig) if np.linalg.norm(X_orig) > 0 else X_orig
            
            m = surface_mesh(X, F_sphere)
            
            # Add scalar fields if present
            if len(self.sf) > 0:
                for ix in range(len(self.sf)):
                    sf_name, sf_obj = self.sf[ix]
                    if len(sf_obj.xc) != len(self.xc):
                        # Truncate if needed (would need trunc_sh function)
                        sf_xc = sf_obj.xc[:lb]
                    else:
                        sf_xc = sf_obj.xc[:lb]
                    sf_values = Y_LK[:, :lb] @ sf_xc
                    m.sf.append([sf_name, sf_values])
            
            m.t = t
            m.p = p
            Y_LK_out = Y_LK
            F = F_sphere
        else:
            # Use precomputed basis
            lb = len(self.xc)
            X = Y_LK_in[:, :lb] @ np.column_stack([self.xc[:lb], self.yc[:lb], self.zc[:lb]])
            Y_LK_out = Y_LK_in
            m = surface_mesh(X, C)
            F = C
            # t, p would need to be passed in or computed
            t, p, _ = kk_cart2sph(X[:, 0], X[:, 1], X[:, 2])
        
        # Ensure all face normals point outward (for correct .off export)
        try:
            from .level1.fix_flipped_faces import fix_flipped_faces
            m, _ = fix_flipped_faces(m, verbose=False)
            F = m.F  # F may have been modified
        except Exception:
            pass  # Non-fatal; mesh is still returned
        
        return m, X, F, Y_LK_out, t, p
    
    def import_shp3(self, fn, dim=60):
        """
        Import .shp3 file
        
        Parameters:
        -----------
        fn : str
            Filename
        dim : int
            Grid dimension
        """
        with open(fn, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        n_shapes = int(lines[0].split('=')[1].strip())
        L = int(lines[1].split('=')[1].strip())
        n_components = int(lines[2].split('=')[1].strip())
        
        # Component tags - in this format, tags are typically x, y, z
        # They may be on line 3 or we infer from header
        tags = []
        data_start = 3
        if data_start < len(lines):
            header_line = lines[data_start].strip().lower()
            if header_line in ['x\ty\tz', 'x y z', 'x', 'y', 'z']:
                # Parse tags from header
                if '\t' in header_line:
                    tags = header_line.split('\t')
                elif ' ' in header_line:
                    tags = header_line.split()
                else:
                    tags = ['x', 'y', 'z'][:n_components]
                data_start = 4
            else:
                # No explicit header, use defaults
                tags = ['x', 'y', 'z'][:n_components]
                data_start = 3
        
        # Read coefficients
        nc = (L + 1) * (L + 1)
        X = []
        for i in range(nc):
            if data_start + i < len(lines):
                line = lines[data_start + i].strip()
                if not line:
                    continue
                values = line.split('\t') if '\t' in line else line.split()
                if len(values) >= n_components:
                    try:
                        row = [float(v) for v in values[:n_components]]
                        X.append(row)
                    except ValueError:
                        continue
        
        if len(X) < nc:
            # Pad with zeros if needed
            while len(X) < nc:
                X.append([0.0] * n_components)
        
        X = np.array(X[:nc])  # Shape: (nc, n_components)
        X = X.T  # Shape: (n_components, nc)
        
        # Extract x, y, z coefficients
        # MATLAB: X(1:3,:)'(:) where ' is transpose and (:) is column-major flatten
        # Python: need to use column-major (Fortran) order
        X_xyz = X[:3, :].T  # Shape: (nc, 3) - each row is a coefficient, columns are x, y, z
        X_o = X_xyz.flatten('F')  # Column-major flatten to match MATLAB: [xc_all, yc_all, zc_all]
        
        # Initialize surface
        self.L_max = L
        self.basis = sh_basis(L, dim)
        self.gdim = dim
        self.X_o = X_o
        self.xc, self.yc, self.zc = self.get_xyz_clks(X_o)
        
        # Mark as needing update to compute geometric properties
        self.needs_updating = True
        
        # Load scalar fields if present
        if n_components > 3:
            from .sh_surface import sh_surface
            for ix in range(3, n_components):
                g = sh_surface(L, self.basis)
                g.xc = X[ix, :]
                isf = [tags[ix] if ix < len(tags) else f'field_{ix-3}', g]
                self.sf.append(isf)
        
        # Compute full geometric properties (H, KG, etc.) after import
        self.update_full()
        return self
    
    def plot(self, ax=None, nico=3, **kwargs):
        """Plot surface"""
        self.update()
        
        # Validate coefficients before plotting
        if self.xc is None or self.yc is None or self.zc is None:
            raise ValueError("shp_surface has no coefficients (xc, yc, zc). Call mesh2shp first.")
        
        if len(self.xc) == 0:
            raise ValueError("shp_surface coefficients are empty. Check that shp_analysis completed successfully.")
        
        # Check coefficient ranges
        xc_range = [np.min(self.xc), np.max(self.xc)] if len(self.xc) > 0 else [0, 0]
        yc_range = [np.min(self.yc), np.max(self.yc)] if len(self.yc) > 0 else [0, 0]
        zc_range = [np.min(self.zc), np.max(self.zc)] if len(self.zc) > 0 else [0, 0]
        
        # Warn if coefficients seem unusual
        if (abs(xc_range[0]) < 1e-10 and abs(xc_range[1]) < 1e-10 and
            abs(yc_range[0]) < 1e-10 and abs(yc_range[1]) < 1e-10 and
            abs(zc_range[0]) < 1e-10 and abs(zc_range[1]) < 1e-10):
            import warnings
            warnings.warn(f"shp_surface coefficients are all near zero! "
                        f"This suggests shp_analysis may have failed. "
                        f"L_max={self.L_max}, coefficient ranges: xc={xc_range}, yc={yc_range}, zc={zc_range}")
        
        # Get mesh representation
        m, X, F, Y_LK, t, p = self.get_mesh(nico)
        
        # Validate reconstructed mesh
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            import warnings
            warnings.warn(f"Reconstructed mesh contains NaN/Inf values. "
                        f"This suggests an issue with coefficient reconstruction.")
        
        # Check if mesh is reasonable (not all at origin)
        mesh_center = np.mean(X, axis=0)
        mesh_scale = np.std(X, axis=0)
        if np.all(np.abs(mesh_center) < 1e-6) and np.all(mesh_scale < 1e-6):
            import warnings
            warnings.warn(f"Reconstructed mesh is all at origin! "
                        f"Mesh center: {mesh_center}, scale: {mesh_scale}. "
                        f"This suggests coefficients or basis evaluation is incorrect.")
        
        # Return PyVista viewer in Jupyter (last expression in a cell) or plotter handle
        return m.plot(**kwargs)
    
    def plot_H(self, minH=None, maxH=None, nico=3, **kwargs):
        """
        Plot mean curvature H as color field on surface
        
        Parameters:
        -----------
        minH : float, optional
            Minimum H value for clipping
        maxH : float, optional
            Maximum H value for clipping
        nico : int
            Icosahedron subdivision level for mesh generation
        **kwargs : dict
            Plotting options
        """
        self.needs_updating = 1
        self.update_full()
        
        # Reshape H to match surface grid
        C = self.H.reshape(self.x.shape)
        
        if minH is not None:
            C[C < minH] = minH
        if maxH is not None:
            C[C > maxH] = maxH
        
        # Get mesh representation
        m, X, F, Y_LK, t, p = self.get_mesh(nico)
        
        # Map curvature to mesh vertices
        # Interpolate H from surface grid to mesh vertices
        # For now, use a simple approach: assign mean curvature from nearest surface point
        H_flat = self.H.flatten()
        if len(H_flat) == len(X):
            m.H = H_flat
        else:
            # Use mean value for now (proper interpolation would be better)
            m.H = np.ones(len(X)) * np.mean(H_flat)
        
        m.plot_H()
    
    def plot_fast(self, nico=3, **kwargs):
        """
        Fast plot with edge color
        
        Parameters:
        -----------
        nico : int
            Icosahedron subdivision level
        **kwargs : dict
            Plotting options
        """
        self.update()
        
        # Get mesh representation
        m, X, F, Y_LK, t, p = self.get_mesh(nico)
        
        # Get edge color from object or default
        edge_color = getattr(self, 'edge_color', 'k')
        
        # Handle 'none' edge color for pyvista (convert to show_edges=False)
        show_edges = True
        if edge_color == 'none' or edge_color is None:
            show_edges = False
            edge_color = 'k'  # Default color if edges are shown (though they won't be)
        
        try:
            import pyvista as pv
            
            # Prepare faces with number of vertices
            num_faces = F.shape[0]
            faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), F))
            cells = faces_with_n_vertices.flatten()
            
            plotter = pv.Plotter()
            mesh_pv = pv.PolyData(X, cells)
            # Use Eb (edge brightness) if available
            if hasattr(self, 'Eb') and self.Eb is not None:
                Eb_flat = self.Eb.flatten()
                if len(Eb_flat) == len(X):
                    mesh_pv['edge_brightness'] = Eb_flat
                    plotter.add_mesh(mesh_pv, scalars='edge_brightness', 
                                   show_edges=show_edges, edge_color=edge_color if show_edges else None, **kwargs)
                else:
                    plotter.add_mesh(mesh_pv, color='lightblue', 
                                   show_edges=show_edges, edge_color=edge_color if show_edges else None, **kwargs)
            else:
                plotter.add_mesh(mesh_pv, color='lightblue', 
                               show_edges=show_edges, edge_color=edge_color if show_edges else None, **kwargs)
            plotter.show()
        except ImportError:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            # Use surface coordinates
            # matplotlib accepts 'none' for edgecolor
            ax.plot_surface(self.x, self.y, self.z, edgecolor=edge_color, **kwargs)
            ax.set_aspect('equal')
            plt.show()
    
    def export_shp3(self, filename):
        """
        Export spherical harmonics surface to .shp3 file format.
        
        Translated from MATLAB @shp_surface/export_ascii.m
        
        Parameters:
        -----------
        filename : str
            Output filename (should end with .shp3)
        """
        if self.X_o is None:
            raise ValueError("Cannot export: X_o (spherical harmonics coefficients) not computed. Run mesh2shp() or shp_analysis() first.")
        
        if self.L_max is None:
            raise ValueError("Cannot export: L_max not set.")
        
        # Ensure we have xc, yc, zc
        if self.xc is None or self.yc is None or self.zc is None:
            self.xc, self.yc, self.zc = self.get_xyz_clks(self.X_o)
        
        # Number of coefficients per component
        nc = len(self.xc)
        
        # Number of components: 3 (x, y, z) plus any scalar fields
        n_components = 3 + len(self.sf) if hasattr(self, 'sf') and self.sf else 3
        
        # Open file for writing
        with open(filename, 'w') as fid:
            # Write header
            fid.write(f'n_shapes = {1}\n')
            fid.write(f'L_max = {self.L_max}\n')
            fid.write(f'n_components = {n_components}\n')
            
            # Write component tags
            tags = ['x', 'y', 'z']
            if hasattr(self, 'sf') and self.sf:
                for sf_item in self.sf:
                    if isinstance(sf_item, (list, tuple)) and len(sf_item) >= 1:
                        tags.append(str(sf_item[0]))  # Field name
                    else:
                        tags.append(f'field{len(tags)-2}')  # Default name
            
            # Write tags (tab-separated)
            fid.write('\t'.join(tags) + '\n')
            
            # Write coefficient values
            # Format: each row has coefficients for all components at one (L,K) index
            # MATLAB format: X matrix with shape (n_components, nc)
            for ix in range(nc):
                # Write x, y, z coefficients
                fid.write(f'{self.xc[ix]:.6e}\t{self.yc[ix]:.6e}\t{self.zc[ix]:.6e}')
                
                # Write scalar field coefficients if present
                if hasattr(self, 'sf') and self.sf:
                    for sf_item in self.sf:
                        if isinstance(sf_item, (list, tuple)) and len(sf_item) >= 2:
                            sf_surface = sf_item[1]
                            if hasattr(sf_surface, 'xc'):
                                # Ensure scalar field has same number of coefficients
                                if len(sf_surface.xc) > nc:
                                    # Truncate if too many
                                    sf_xc = sf_surface.xc[:nc]
                                elif len(sf_surface.xc) < nc:
                                    # Pad with zeros if too few
                                    sf_xc = np.pad(sf_surface.xc, (0, nc - len(sf_surface.xc)), 'constant')
                                else:
                                    sf_xc = sf_surface.xc
                                fid.write(f'\t{sf_xc[ix]:.6e}')
                            else:
                                fid.write('\t0.000000e+00')
                        else:
                            fid.write('\t0.000000e+00')
                
                fid.write('\n')
        
        print(f"Exported spherical harmonics surface to: {filename}")
        print(f"  L_max: {self.L_max}")
        print(f"  Number of coefficients: {nc}")
        print(f"  Components: {n_components} ({', '.join(tags)})")
