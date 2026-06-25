"""
Spherical harmonics surface class
Translated from MATLAB @sh_surface class
"""

import numpy as np
from .sh_basis import sh_basis
from .utils import indices_gen, kk_cart2sph, kk_sph2cart
# Import will be done locally to avoid circular dependency


class sh_surface:
    """
    Spherical harmonics surface representation
    """
    
    @staticmethod
    def indices_gen(c):
        """Generate l, m indices for coefficients"""
        return indices_gen(c)
    
    def __init__(self, L_max=None, basis=None):
        """
        Initialize sh_surface
        
        Parameters:
        -----------
        L_max : int or array
            Maximum degree, or coefficient vector
        basis : sh_basis
            Basis object (optional)
        """
        self.basis = None
        self.L_max = None
        self.gdim = 60
        self.xc = None
        self.x = None
        self.y = None
        self.z = None
        self.needs_updating = True
        self.use_camorbit = False
        self.edge_color = 'none'
        
        if L_max is None:
            # Make a sphere
            L_max = 6
            self.basis = sh_basis(10, self.gdim)
            self.xc = np.zeros((L_max + 1)**2)
            self.xc[0] = 1
            self.L_max = L_max
            self.update()
        elif isinstance(L_max, (list, np.ndarray)) and len(L_max) > 1:
            # Initialize with xc vector
            self.xc = np.asarray(L_max).flatten()
            self.L_max = int(np.sqrt(len(self.xc)) - 1)
            self.basis = sh_basis(self.L_max, self.gdim)
            self.update()
        elif isinstance(L_max, (int, np.integer)) and basis is None:
            self.basis = sh_basis(L_max, self.gdim)
            self.xc = np.zeros((L_max + 1)**2)
            self.xc[0] = 1
            self.L_max = L_max
            self.update()
        elif basis is not None:
            self.basis = basis
            self.L_max = L_max
            self.gdim = basis.gdim
            self.xc = np.zeros((L_max + 1)**2)
            self.xc[0] = 1
            self.update()
    
    def update(self):
        """Update surface coordinates from coefficients"""
        if self.needs_updating:
            lb = (self.L_max + 1)**2
            gdimp = self.basis.p.shape[0]
            gdimt = self.basis.p.shape[1] if self.basis.p.ndim > 1 else 1
            
            # Reshape coefficients
            c = self.xc[:lb].reshape(1, 1, -1)
            c = np.tile(c, (gdimp, gdimt, 1))
            
            # Compute coordinates
            x = np.sum(c * self.basis.Y[:, :, :lb], axis=2)
            xp = np.sum(c * self.basis.Y_P[:, :, :lb], axis=2)
            xt = np.sum(c * self.basis.Y_T[:, :, :lb], axis=2)
            xpp = np.sum(c * self.basis.Y_PP[:, :, :lb], axis=2)
            xtt = np.sum(c * self.basis.Y_TT[:, :, :lb], axis=2)
            xtp = np.sum(c * self.basis.Y_TP[:, :, :lb], axis=2)
            
            self.x = x
            self.y = x  # Placeholder - sh_surface only has xc, not yc/zc
            self.z = x  # Placeholder
            self.needs_updating = False
        else:
            x = self.x
            y = self.y
            z = self.z
        
        return self
    
    def flip(self):
        """Flip the surface (top/bottom) by negating all K<0 channels"""
        l, m, flags = indices_gen(self.xc)
        km = np.zeros_like(l)
        kp = np.zeros_like(l)
        
        for ix in range(len(l)):
            if m[ix] < 0:
                self.xc[ix] = -self.xc[ix]
                km[ix] = 1
            if m[ix] > 0:
                kp[ix] = 1
        
        return self, km, kp
    
    def flop(self):
        """Flop the surface (front/back) by negating K<0 channels"""
        l, m, flags = indices_gen(self.xc)
        km = np.zeros_like(l)
        kp = np.zeros_like(l)
        
        for ix in range(len(l)):
            if m[ix] < 0 and (m[ix] % 2 == 0):
                self.xc[ix] = -self.xc[ix]
                km[ix] = 1
            if m[ix] >= 0 and (m[ix] % 2 != 0):
                self.xc[ix] = -self.xc[ix]
                kp[ix] = 1
        
        return self, km, kp
    
    @staticmethod
    def get_mesh(obj, nico=3, Y_LK=None, C=None):
        """
        Get mesh representation of surface
        
        Parameters:
        -----------
        obj : sh_surface
            Surface object
        nico : int
            Icosahedron subdivision level
        Y_LK : array, optional
            Precomputed basis
        C : array, optional
            Precomputed connectivity
            
        Returns:
        --------
        m : surface_mesh
            Mesh object
        X : array
            Vertex coordinates
        C : array
            Face connectivity
        Y_LK : array
            Basis matrix
        t, p : array
            Spherical coordinates
        """
        obj = obj.update()
        
        if Y_LK is None:
            # Build the basis
            from .surface_mesh import surface_mesh
            X, C = surface_mesh.sphere_mesh_gen(nico)
            t, p, r = kk_cart2sph(X[:, 0], X[:, 1], X[:, 2])
            x, y, z = kk_sph2cart(t, p, np.ones_like(t))
            
            # Generate basis at vertices
            L, K, _ = indices_gen(np.arange(1, (obj.L_max + 1)**2 + 1))
            M = len(L)
            N = len(t)
            Y_LK = np.zeros((N, M), dtype=np.float32)
            
            for S in range(len(L)):
                Y_LK[:, S] = obj.basis.ylk_bosh(L[S], K[S], p, t).flatten()
            
            r = Y_LK[:, :len(obj.xc)] @ obj.xc
            x, y, z = kk_sph2cart(t, p, r)
            X = np.column_stack([x, y, z])
            XF = surface_mesh(X, C)
        else:
            r = Y_LK[:, :len(obj.xc)] @ obj.xc
            x, y, z = kk_sph2cart(t, p, r)
            X = np.column_stack([x, y, z])
            m = surface_mesh(X, C)
        
        return m, X, C, Y_LK, t, p
    
    @staticmethod
    def tr_xc(xc, L_max):
        """Truncate coefficient vector to L_max"""
        xc = np.asarray(xc).flatten()
        trunc = (L_max + 1)**2
        lmax_in = int(np.sqrt(len(xc)) - 1)
        
        if lmax_in < L_max:
            xc_new = np.zeros(trunc)
            xc_new[:len(xc)] = xc
            return xc_new
        else:
            return xc[:trunc]
    
    @staticmethod
    def get_L_max_xc(xc):
        """Get L_max from coefficient vector"""
        return int(np.sqrt(len(xc)) - 1)
    
    @staticmethod
    def bosh2nocs_xc(xc):
        """Convert BOSH to NOCS coefficients"""
        # For now, same format
        return xc
    
    @staticmethod
    def nocs2bosh_xc(xc):
        """Convert NOCS to BOSH coefficients"""
        # For now, same format
        return xc
    
    @staticmethod
    def cs2nocs_xc(xc):
        """Convert CS to NOCS coefficients"""
        return xc
    
    @staticmethod
    def nocs2cs_xc(xc):
        """Convert NOCS to CS coefficients"""
        return xc
