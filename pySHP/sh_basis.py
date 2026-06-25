"""
Spherical harmonics basis class
Translated from MATLAB @sh_basis class
"""

import numpy as np
from scipy.special import legendre, factorial, lpmv
from scipy.linalg import eig


class sh_basis:
    """
    Spherical harmonics basis class for computing SH basis functions
    """
    
    def __init__(self, L_max, gdim):
        """
        Initialize spherical harmonics basis
        
        Parameters:
        -----------
        L_max : int
            Maximum degree of spherical harmonics
        gdim : int
            Grid dimension for quadrature
        """
        self.basis = 'bosh'
        self.L_max = L_max
        self.gdim = gdim
        
        # Compute Gaussian quadrature points and weights
        t, wt = self.gaussquad(gdim, 0, np.pi)
        p, wp = self.gaussquad(gdim, 0, 2*np.pi)
        
        # Create meshgrid
        P, T = np.meshgrid(p, t)
        self.p = P
        self.t = T
        
        # Create weight meshgrid
        WP, WT = np.meshgrid(wp, wt)
        self.w = (WP * WT).flatten()
        
        # Compute basis functions
        Y, P_leg = self.ylk_cos_sin_bosh(self.p, self.t, L_max)
        self.Y = Y
        self.Y_P = self.ylk_cos_sin_dphi_bosh(self.p, self.t, L_max, P_leg)
        Y_T, P_T = self.ylk_cos_sin_dtheta_bosh(self.p, self.t, L_max, P_leg)
        self.Y_T = Y_T
        self.Y_PP = self.ylk_cos_sin_dphiphi_bosh(self.p, self.t, L_max, P_leg)
        self.Y_TT = self.ylk_cos_sin_dthetatheta_bosh(self.p, self.t, L_max, P_T)
        self.Y_TP = self.ylk_cos_sin_dthetaphi_bosh(self.p, self.t, L_max, P_T)
    
    @staticmethod
    def N_LK_bosh(L, K):
        """
        Compute normalization factor for spherical harmonics
        
        MATLAB: NLK = sqrt((2-isequal(K,0))*(2*L+1)*factorial(L-K)/factorial(L+K))
        This means:
        - If K == 0: factor = (2 - 1) = 1
        - If K != 0: factor = (2 - 0) = 2
        """
        K = abs(K)
        if abs(K) > L:
            return 0.0
        else:
            # MATLAB: (2-isequal(K,0)) means 1 if K==0, 2 if K!=0
            factor = 1 if K == 0 else 2
            return np.sqrt(factor * (2*L + 1) * factorial(L - K) / factorial(L + K))
    
    @staticmethod
    def N_LK_nocs(L, K):
        """
        NOCS normalization (if different from BOSH)
        For now, same as BOSH
        """
        return sh_basis.N_LK_bosh(L, K)
    
    @staticmethod
    def gaussquad(n, a=-1, b=1):
        """
        Gaussian quadrature integration points and weights
        
        Parameters:
        -----------
        n : int
            Number of quadrature points
        a : float
            Lower bound (default: -1)
        b : float
            Upper bound (default: 1)
            
        Returns:
        --------
        x : array
            Quadrature points
        w : array
            Quadrature weights
        """
        # Handle different input signatures
        if isinstance(a, (list, tuple)) or (isinstance(a, np.ndarray) and a.size > 1):
            # If a is array-like, assume it's [a, b]
            if len(a) == 2:
                a, b = a[0], a[1]
            else:
                b = a
                a = -1
        
        # Build tridiagonal matrix for eigenvalue problem
        u = np.arange(1, n) / np.sqrt(4 * np.arange(1, n)**2 - 1)
        
        # Create tridiagonal matrix
        A = np.zeros((n, n))
        # Fill subdiagonal
        A[1:, :-1] += np.diag(u)
        # Fill superdiagonal
        A[:-1, 1:] += np.diag(u)
        
        # Solve eigenvalue problem
        eigenvals, eigenvecs = eig(A)
        x = np.real(eigenvals)
        v = np.real(eigenvecs)
        
        # Sort eigenvalues
        idx = np.argsort(x)
        x = x[idx]
        v = v[:, idx]
        
        # Compute weights
        w = 2 * v[0, :]**2
        
        # Transform from [-1, 1] to [a, b]
        x = (b - a) / 2 * x + (a + b) / 2
        w = (b - a) / 2 * w
        
        return x, w
    
    @staticmethod
    def ylk_bosh(L, K, phi, theta):
        """
        Compute real spherical harmonic Y_LK using BOSH normalization
        
        Parameters:
        -----------
        L : int
            Degree
        K : int
            Order
        phi : array
            Azimuthal angle (longitude)
        theta : array
            Polar angle (colatitude)
            
        Returns:
        --------
        Y : array
            Spherical harmonic values
        """
        phi = np.asarray(phi)
        theta = np.asarray(theta)
        
        if K == 0 and L == 0:
            Y = sh_basis.N_LK_bosh(0, 0) * np.ones_like(theta)
        else:
            NLK_bosh = sh_basis.N_LK_bosh(L, K)
            
            # Compute associated Legendre polynomial
            # MATLAB's legendre includes Condon-Shortley phase factor (-1)^m
            # scipy's lpmv does NOT include it, so we need to add it to match MATLAB
            cos_theta = np.cos(theta.flatten())
            # Use lpmv which is the correct function for associated Legendre
            # lpmv(m, n, x) where m is order and n is degree
            P_LK = lpmv(abs(K), L, cos_theta)
            
            # MATLAB's legendre includes (-1)^m factor, scipy's lpmv does not
            # Since MATLAB code uses CS=1 (keeping the phase factor from legendre),
            # we need to add the Condon-Shortley phase factor to match MATLAB
            CS_phase = (-1)**abs(K)
            P_LK = CS_phase * P_LK
            
            P_LK = P_LK.reshape(theta.shape)
            
            # BOSH version uses CS=1 (keeping the phase factor)
            CS = 1
            
            if K >= 0:
                Y = CS * NLK_bosh * P_LK * np.cos(K * phi)
            else:  # K < 0
                Y = CS * NLK_bosh * P_LK * np.sin(abs(K) * phi)
        
        return Y
    
    @staticmethod
    def ylk_cos_sin(L, K, phi, theta):
        """
        Compute spherical harmonic (NOCS version)
        For now, same as ylk_bosh
        """
        return sh_basis.ylk_bosh(L, K, phi, theta)
    
    @staticmethod
    def ylk_cos_sin_bosh(phi, theta, L_max):
        """
        Pre-calculate normalized associated Legendre functions up to L_max
        """
        phi = np.asarray(phi)
        theta = np.asarray(theta)
        
        # Handle 1D case
        if phi.ndim == 1 and theta.ndim == 1:
            phi = phi.reshape(-1, 1)
            theta = theta.reshape(-1, 1)
            gdimp = phi.shape[0]
            gdimt = theta.shape[0]
            
            Y = np.zeros((gdimp, 1, (L_max + 1)**2))
            P_dim = int(L_max**2 / 2 + 3 * L_max / 2 + 1)
            P = np.zeros((gdimp, 1, P_dim))
            
            counter = 0
            pcounter = 0
            
            for L in range(L_max + 1):
                # Compute associated Legendre functions for all K values
                theta_flat = theta.flatten()
                cos_theta = np.cos(theta_flat)
                
                for K in range(-L, L + 1):
                    counter += 1
                    # Use lpmv for associated Legendre functions
                    # MATLAB's legendre includes Condon-Shortley phase factor (-1)^m
                    # scipy's lpmv does NOT include it, so we need to add it
                    plk_flat = lpmv(abs(K), L, cos_theta)
                    # Add Condon-Shortley phase factor to match MATLAB
                    CS_phase = (-1)**abs(K)
                    plk_flat = CS_phase * plk_flat
                    plk = plk_flat.reshape(theta.shape)
                    NLK = sh_basis.N_LK_bosh(L, K)
                    CS = 1  # BOSH version uses CS=1
                    
                    if K == 0 and L == 0:
                        pcounter += 1
                        Y[:, 0, counter - 1] = NLK * plk.flatten()
                        P[:, 0, pcounter - 1] = CS * NLK * plk.flatten()
                    elif K >= 0:
                        pcounter += 1
                        P[:, 0, pcounter - 1] = CS * NLK * plk.flatten()
                        Y[:, 0, counter - 1] = CS * NLK * plk.flatten() * np.cos(K * phi.flatten())
                    else:  # K < 0
                        Y[:, 0, counter - 1] = CS * NLK * plk.flatten() * np.sin(abs(K) * phi.flatten())
        else:
            # 2D meshgrid case
            gdimp = phi.shape[0]
            gdimt = phi.shape[1] if phi.ndim > 1 else 1
            
            Y = np.zeros((gdimp, gdimt, (L_max + 1)**2))
            P_dim = int(L_max**2 / 2 + 3 * L_max / 2 + 1)
            P = np.zeros((gdimp, gdimt, P_dim))
            
            counter = 0
            pcounter = 0
            
            for L in range(L_max + 1):
                # Compute associated Legendre functions for all K values
                theta_flat = theta.flatten()
                cos_theta = np.cos(theta_flat)
                
                for K in range(-L, L + 1):
                    counter += 1
                    # Use lpmv for associated Legendre functions
                    # lpmv(m, n, x) where m is the order and n is the degree
                    # MATLAB's legendre includes Condon-Shortley phase factor (-1)^m
                    # scipy's lpmv does NOT include it, so we need to add it
                    plk_flat = lpmv(abs(K), L, cos_theta)
                    # Add Condon-Shortley phase factor to match MATLAB
                    CS_phase = (-1)**abs(K)
                    plk_flat = CS_phase * plk_flat
                    plk = plk_flat.reshape(theta.shape)
                    
                    NLK = sh_basis.N_LK_bosh(L, K)
                    CS = 1  # BOSH version uses CS=1
                    
                    if K == 0 and L == 0:
                        pcounter += 1
                        Y[:, :, counter - 1] = NLK * plk
                        P[:, :, pcounter - 1] = CS * NLK * plk
                    elif K >= 0:
                        pcounter += 1
                        P[:, :, pcounter - 1] = CS * NLK * plk
                        Y[:, :, counter - 1] = CS * NLK * plk * np.cos(K * phi)
                    else:  # K < 0
                        Y[:, :, counter - 1] = CS * NLK * plk * np.sin(abs(K) * phi)
        
        return Y, P
    
    @staticmethod
    def ylk_cos_sin_dphi_bosh(phi, theta, L_max, P):
        """
        Derivative with respect to phi
        """
        phi = np.asarray(phi)
        theta = np.asarray(theta)
        
        Y_P = np.zeros_like(phi)
        if phi.ndim == 1:
            Y_P = np.zeros((phi.size, 1, (L_max + 1)**2))
        else:
            Y_P = np.zeros((phi.shape[0], phi.shape[1], (L_max + 1)**2))
        
        counter = 0
        pcounter = 0
        
        for L in range(L_max + 1):
            for K in range(-L, L + 1):
                counter += 1
                NLK = sh_basis.N_LK_bosh(L, K)
                
                if K == 0:
                    # Derivative of constant is zero
                    if phi.ndim == 1:
                        Y_P[:, 0, counter - 1] = 0
                    else:
                        Y_P[:, :, counter - 1] = 0
                elif K > 0:
                    pcounter += 1
                    plk = P[:, :, pcounter - 1] if P.ndim == 3 else P
                    if phi.ndim == 1:
                        Y_P[:, 0, counter - 1] = -K * NLK * plk.flatten() * np.sin(K * phi.flatten())
                    else:
                        Y_P[:, :, counter - 1] = -K * NLK * plk * np.sin(K * phi)
                else:  # K < 0
                    if phi.ndim == 1:
                        Y_P[:, 0, counter - 1] = abs(K) * NLK * plk.flatten() * np.cos(abs(K) * phi.flatten())
                    else:
                        # Need to get plk from P or compute it
                        # Use lpmv with Condon-Shortley phase factor
                        cos_theta = np.cos(theta.flatten())
                        plk_flat = lpmv(abs(K), L, cos_theta)
                        CS_phase = (-1)**abs(K)
                        plk_flat = CS_phase * plk_flat
                        plk = plk_flat.reshape(theta.shape)
                        Y_P[:, :, counter - 1] = abs(K) * NLK * plk * np.cos(abs(K) * phi)
        
        return Y_P
    
    @staticmethod
    def get_LK_index_P(L, K):
        """
        Get linear index for P array given L and K.
        
        MATLAB: ix = uint8((L-1)^2/2+3*(L-1)/2 + 1) + K + 1;
        This maps (L,K) pairs to indices in the P array which stores
        associated Legendre functions in order: P00, P10, P11, P20, P21, ...
        Note: P array only stores K >= 0 values.
        
        Parameters:
        -----------
        L : int
            Degree
        K : int
            Order (must be >= 0, absolute value is used)
            
        Returns:
        --------
        ix : int
            Index into P array (0-indexed for Python)
        """
        K = abs(K)  # P array only stores K >= 0
        # MATLAB: ix = uint8((L-1)^2/2+3*(L-1)/2 + 1) + K + 1;
        # uint8 truncates, so we use int() which also truncates towards zero
        # Convert to 0-indexed: subtract 1 from final result
        base = int((L-1)**2 / 2 + 3*(L-1) / 2 + 1)  # MATLAB uint8 part
        ix_matlab = base + K + 1  # MATLAB 1-indexed result
        ix = ix_matlab - 1  # Convert to 0-indexed
        return ix
    
    @staticmethod
    def plkt(P, L_max):
        """
        Calculate the first derivative of associated Legendre functions with respect to theta.
        
        Translated from MATLAB plkt function in ylk_cos_sin_dtheta_bosh.m
        Uses recursion relations from Bosh 2000.
        
        The P array stores Legendre functions in order: P00, P10, P11, P20, P21, P22, ...
        where only K >= 0 values are stored.
        
        Parameters:
        -----------
        P : array
            Associated Legendre functions array (gdimp x gdimt x P_dim)
        L_max : int
            Maximum degree
            
        Returns:
        --------
        P_T : array
            First derivative of P with respect to theta
        """
        P_T = np.zeros_like(P)
        
        if P.ndim < 3:
            return P_T
        
        # MATLAB: ia = 1; (1-indexed, starts at 1)
        # ia tracks the starting index for each L in the P array (1-indexed)
        # P array structure (1-indexed): [P00, P10, P11, P20, P21, P22, P30, ...]
        # L=0: index 1, L=1: indices 2-3 (ia=2), L=2: indices 4-6 (ia=4), L=3: indices 7-10 (ia=7)
        # Formula: starting_index(L) = 1 + sum(i=0 to L-1 of (i+1)) = 1 + L*(L+1)/2
        
        # MATLAB: dpnm2 = P(:,:,2); (1-indexed, so 0-indexed: index 1)
        # This is P_10 (L=1, K=0)
        dpnm2 = P[:, :, 1].copy() if P.shape[2] > 1 else None
        
        # MATLAB: ia = 1; (1-indexed)
        # Convert to 0-indexed: ia starts at 0 for L=0, but we skip L=0 in the loop
        # For L=1: ia = 1 (1-indexed) = 0 (0-indexed), but MATLAB uses ia=2, so we need ia=1
        # Actually, MATLAB's ia tracks where each L starts AFTER the initial ia=1
        # Let me trace: MATLAB starts with ia=1, then for L=1: ia = ia + L = 1 + 1 = 2
        ia = 1  # MATLAB 1-indexed equivalent (will convert to 0-indexed when accessing array)
        
        for L in range(1, L_max + 1):
            # MATLAB: ia = ia + L;
            # L=1: ia = 1 + 1 = 2 (1-indexed) = 1 (0-indexed)
            # L=2: ia = 2 + 2 = 4 (1-indexed) = 3 (0-indexed)
            # L=3: ia = 4 + 3 = 7 (1-indexed) = 6 (0-indexed)
            ia = ia + L
            ia_python = ia - 1  # Convert to 0-indexed for array access
            
            if ia_python >= P.shape[2]:
                break
            
            # MATLAB: temp = P(:,:,ia);
            temp = P[:, :, ia_python].copy()
            
            # MATLAB: P_T(:,:,ia) = -sqrt(ia-1).*P(:,:,ia+1);
            # ia-1 in MATLAB (1-indexed) = ia_python in Python (0-indexed)
            if ia_python + 1 < P.shape[2]:
                P_T[:, :, ia_python] = -np.sqrt(ia_python) * P[:, :, ia_python + 1]
            
            # MATLAB: fac1 = sqrt(2*L*(L+1));
            fac1 = np.sqrt(2 * L * (L + 1))
            
            # MATLAB: for K = 1:L-1
            for K in range(1, L):
                # MATLAB: ix = ia+K;
                ix_matlab = ia + K  # MATLAB 1-indexed
                ix_python = ix_matlab - 1  # Convert to 0-indexed
                
                if ix_python >= P.shape[2]:
                    break
                
                # MATLAB: fac2 = sqrt((L-K)*(L+K+1));
                fac2 = np.sqrt((L - K) * (L + K + 1))
                
                # MATLAB: P_T(:,:,ix) = 1/2*(fac1*temp-fac2*P(:,:,ix+1));
                if ix_python + 1 < P.shape[2]:
                    P_T[:, :, ix_python] = 0.5 * (fac1 * temp - fac2 * P[:, :, ix_python + 1])
                else:
                    P_T[:, :, ix_python] = 0.5 * fac1 * temp
                
                # MATLAB: temp = P(:,:,ix);
                temp = P[:, :, ix_python].copy()
                # MATLAB: fac1 = fac2;
                fac1 = fac2
            
            # MATLAB: P_T(:,:,ia+L) = sqrt(L/2)*temp;
            ix_final_matlab = ia + L  # MATLAB 1-indexed
            ix_final_python = ix_final_matlab - 1  # Convert to 0-indexed
            if ix_final_python < P.shape[2]:
                P_T[:, :, ix_final_python] = np.sqrt(L / 2.0) * temp
        
        # MATLAB: P_T(:,:,3) = dpnm2; (index 3 is 1-indexed, so 0-indexed: index 2)
        # This is P_11 (L=1, K=1)
        if dpnm2 is not None and P.shape[2] > 2:
            P_T[:, :, 2] = dpnm2
        
        return P_T
    
    @staticmethod
    def ylk_cos_sin_dtheta_bosh(phi, theta, L_max, P):
        """
        Derivative with respect to theta.
        
        Translated from MATLAB ylk_cos_sin_dtheta_bosh.m
        """
        phi = np.asarray(phi)
        theta = np.asarray(theta)
        
        # MATLAB: gdimp = length(p); gdimt = length(t);
        # For meshgrid: P has shape (len(t), len(p)), so:
        # gdimp = number of columns = phi.shape[1] if 2D, else phi.size
        # gdimt = number of rows = phi.shape[0] if 2D, else theta.size
        if phi.ndim == 1 and theta.ndim == 1:
            gdimp = phi.size
            gdimt = theta.size
            Y_T = np.zeros((gdimp, gdimt, (L_max + 1)**2))
            # Reshape phi if needed
            if len(phi.shape) == 1:
                phi = phi.reshape(-1, 1)
        else:
            # 2D meshgrid case
            gdimp = phi.shape[1] if phi.ndim > 1 else phi.size
            gdimt = phi.shape[0] if phi.ndim > 1 else theta.shape[0] if theta.ndim > 0 else 1
            Y_T = np.zeros((gdimp, gdimt, (L_max + 1)**2))
        
        # Compute P_T using plkt
        P_T = sh_basis.plkt(P, L_max)
        
        counter = 0
        for L in range(L_max + 1):
            for K in range(-L, L + 1):
                counter += 1
                # Get index into P array
                ix = sh_basis.get_LK_index_P(L, K)
                
                if ix < P_T.shape[2]:
                    if K >= 0:
                        Y_T[:, :, counter - 1] = P_T[:, :, ix] * np.cos(K * phi)
                    else:  # K < 0
                        Y_T[:, :, counter - 1] = P_T[:, :, ix] * np.sin(abs(K) * phi)
        
        return Y_T, P_T
    
    @staticmethod
    def ylk_cos_sin_dphiphi_bosh(phi, theta, L_max, P):
        """
        Second derivative with respect to phi
        """
        phi = np.asarray(phi)
        Y_PP = np.zeros_like(phi)
        if phi.ndim == 1:
            Y_PP = np.zeros((phi.size, 1, (L_max + 1)**2))
        else:
            Y_PP = np.zeros((phi.shape[0], phi.shape[1], (L_max + 1)**2))
        
        counter = 0
        pcounter = 0
        
        for L in range(L_max + 1):
            for K in range(-L, L + 1):
                counter += 1
                NLK = sh_basis.N_LK_bosh(L, K)
                
                if K == 0:
                    if phi.ndim == 1:
                        Y_PP[:, 0, counter - 1] = 0
                    else:
                        Y_PP[:, :, counter - 1] = 0
                elif K > 0:
                    pcounter += 1
                    plk = P[:, :, pcounter - 1] if P.ndim == 3 else P
                    if phi.ndim == 1:
                        Y_PP[:, 0, counter - 1] = -K**2 * NLK * plk.flatten() * np.cos(K * phi.flatten())
                    else:
                        Y_PP[:, :, counter - 1] = -K**2 * NLK * plk * np.cos(K * phi)
                else:  # K < 0
                    if phi.ndim == 1:
                        Y_PP[:, 0, counter - 1] = -abs(K)**2 * NLK * plk.flatten() * np.sin(abs(K) * phi.flatten())
                    else:
                        # Use lpmv with Condon-Shortley phase factor
                        cos_theta = np.cos(theta.flatten())
                        plk_flat = lpmv(abs(K), L, cos_theta)
                        CS_phase = (-1)**abs(K)
                        plk_flat = CS_phase * plk_flat
                        plk = plk_flat.reshape(theta.shape)
                        Y_PP[:, :, counter - 1] = -abs(K)**2 * NLK * plk * np.sin(abs(K) * phi)
        
        return Y_PP
    
    @staticmethod
    def ylk_cos_sin_dthetatheta_bosh(phi, theta, L_max, P_T):
        """
        Second derivative with respect to theta.
        
        Translated from MATLAB ylk_cos_sin_dthetatheta_bosh.m
        """
        phi = np.asarray(phi)
        theta = np.asarray(theta)
        
        # MATLAB: gdimp = size(p,2); gdimt = size(t,1);
        # For meshgrid: P has shape (len(t), len(p)), so:
        # gdimp = number of columns = phi.shape[1] if 2D, else phi.size
        # gdimt = number of rows = phi.shape[0] if 2D, else theta.size
        if phi.ndim == 1 and theta.ndim == 1:
            gdimp = phi.size
            gdimt = theta.size
            Y_TT = np.zeros((gdimp, gdimt, (L_max + 1)**2))
            if len(phi.shape) == 1:
                phi = phi.reshape(-1, 1)
        else:
            gdimp = phi.shape[1] if phi.ndim > 1 else phi.size
            gdimt = phi.shape[0] if phi.ndim > 1 else theta.shape[0] if theta.ndim > 0 else 1
            Y_TT = np.zeros((gdimp, gdimt, (L_max + 1)**2))
        
        # MATLAB: P_TT = plkt(P_T, L_max);
        # Compute second derivative by applying plkt to P_T
        P_TT = sh_basis.plkt(P_T, L_max)
        
        counter = 0
        for L in range(L_max + 1):
            for K in range(-L, L + 1):
                counter += 1
                # Get index into P array
                ix = sh_basis.get_LK_index_P(L, K)
                
                if ix < P_TT.shape[2]:
                    if K >= 0:
                        Y_TT[:, :, counter - 1] = P_TT[:, :, ix] * np.cos(K * phi)
                    else:  # K < 0
                        Y_TT[:, :, counter - 1] = P_TT[:, :, ix] * np.sin(abs(K) * phi)
        
        return Y_TT
    
    @staticmethod
    def ylk_cos_sin_dthetaphi_bosh(phi, theta, L_max, P_T):
        """
        Mixed derivative with respect to theta and phi.
        
        Translated from MATLAB ylk_cos_sin_dthetaphi_bosh.m
        """
        phi = np.asarray(phi)
        theta = np.asarray(theta)
        
        # MATLAB: gdimp = size(phi,2); gdimt = size(theta,1);
        # For meshgrid: P has shape (len(t), len(p)), so:
        # gdimp = number of columns = phi.shape[1] if 2D, else phi.size
        # gdimt = number of rows = phi.shape[0] if 2D, else theta.size
        if phi.ndim == 1 and theta.ndim == 1:
            gdimp = phi.size
            gdimt = theta.size
            Y_TP = np.zeros((gdimp, gdimt, (L_max + 1)**2))
            if len(phi.shape) == 1:
                phi = phi.reshape(-1, 1)
        else:
            gdimp = phi.shape[1] if phi.ndim > 1 else phi.size
            gdimt = phi.shape[0] if phi.ndim > 1 else theta.shape[0] if theta.ndim > 0 else 1
            Y_TP = np.zeros((gdimp, gdimt, (L_max + 1)**2))
        
        counter = 0
        for L in range(L_max + 1):
            for K in range(-L, L + 1):
                counter += 1
                # Get index into P array
                ix = sh_basis.get_LK_index_P(L, K)
                
                if ix < P_T.shape[2]:
                    if K == 0:
                        # Y_TP is zero for K=0
                        Y_TP[:, :, counter - 1] = 0
                    elif K > 0:
                        # MATLAB: Y_TP(:,:,counter) = P_T(:,:,get_LK_index_P(L,K)).* (-K).*sin(K * phi);
                        Y_TP[:, :, counter - 1] = P_T[:, :, ix] * (-K) * np.sin(K * phi)
                    else:  # K < 0
                        # MATLAB: Y_TP(:,:,counter) = P_T(:,:,get_LK_index_P(L,K)).* abs(K).* cos(abs(K) * phi);
                        Y_TP[:, :, counter - 1] = P_T[:, :, ix] * abs(K) * np.cos(abs(K) * phi)
        
        return Y_TP
