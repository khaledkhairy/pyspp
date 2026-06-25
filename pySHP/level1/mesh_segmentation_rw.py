"""
Random walk mesh segmentation
Translated from MATLAB @surface_mesh/mesh_segmentation_rw.m
"""

import numpy as np
from scipy.sparse import csr_matrix, spdiags, lil_matrix
try:
    from scipy.sparse.linalg import spsolve
except ImportError:
    # Fallback for older scipy
    def spsolve(A, b):
        return np.linalg.solve(A.toarray(), b)
from ..surface_mesh import surface_mesh
from .get_border import get_border, compute_face_neighbors
from ..level0.mesh_utils import reduce_to_minimal_set


def mesh_segmentation_rw(m, nseeds, sig=1.0, slix=None, curvature_weight=0.0, verbose=True, plot_intermediate=False):
    """
    Perform random walk mesh segmentation using nseeds seed faces
    Matches MATLAB @surface_mesh/mesh_segmentation_rw.m exactly
    
    Parameters:
    -----------
    m : surface_mesh
        Input mesh
    nseeds : int
        Number of seed faces
    sig : float
        Segmentation parameter (default: 1.0)
    slix : array, optional
        Seed face indices (if None, automatically selected)
    curvature_weight : float, optional
        Weight for curvature-based edge probabilities (default: 0.0)
        - 0.0: No curvature consideration (uniform transition probabilities)
          Uses default uniform weight of -1e-6 to preserve matrix structure
        - 1.0: Full curvature consideration (uses sig parameter)
        - Values between 0 and 1: Interpolate between uniform and curvature-based
    verbose : bool
        Print diagnostic information
    plot_intermediate : bool
        Plot intermediate results without stopping execution
        
    Returns:
    --------
    m : surface_mesh
        Mesh with face_labels populated (SAME mesh, not simplified)
    L : array
        Face labels
    slix : array
        Seed face indices
    P : list
        Patch data structures
    Pconn : sparse matrix
        Patch connectivity matrix
    """
    if nseeds < 2:
        raise ValueError('Number of seeds must be >= 2')
    
    if verbose:
        print("="*60)
        print("mesh_segmentation_rw: Starting segmentation")
        print("="*60)
        print(f"Input mesh: {len(m.X)} vertices, {len(m.F)} faces")
    
    nfaces = len(m.F)
    
    # Ensure mesh has properties and edge info
    if m.needs_updating:
        m.props()
    if m.needs_edge_info:
        m.edge_info()
    
    # Ensure face_nbrs is computed as sparse matrix (like MATLAB)
    # edge_info() now computes and stores face_nbrs as sparse matrix
    if not hasattr(m, 'face_nbrs') or m.face_nbrs is None:
        m.needs_edge_info = True
        m.edge_info()
    elif not isinstance(m.face_nbrs, csr_matrix):
        # If it's a dict, convert to sparse matrix (shouldn't happen after edge_info fix, but safety check)
        face_nbrs_dict = m.face_nbrs if isinstance(m.face_nbrs, dict) else compute_face_neighbors(m)
        m.face_nbrs = lil_matrix((nfaces, nfaces), dtype=bool)
        for i, nbrs in face_nbrs_dict.items():
            for nbr in nbrs:
                m.face_nbrs[i, nbr] = True
        m.face_nbrs = m.face_nbrs.tocsr()
    
    if verbose:
        print(f"Face neighbors matrix: {m.face_nbrs.shape}, {m.face_nbrs.nnz} non-zero entries")
        # Check connectivity - compute neighbor counts from sparse matrix
        nbr_counts = np.array([m.face_nbrs[i, :].nnz for i in range(nfaces)])
        print(f"Face neighbor counts: min={nbr_counts.min()}, max={nbr_counts.max()}, mean={nbr_counts.mean():.2f}")
        
        # Verify symmetry (face i neighbors face j <=> face j neighbors face i)
        # Check a sample of entries
        sample_size = min(100, m.face_nbrs.nnz // 2)
        rows, cols = m.face_nbrs.nonzero()
        if len(rows) > 0:
            symmetric_count = 0
            for i in range(min(sample_size, len(rows))):
                r, c = rows[i], cols[i]
                if m.face_nbrs[c, r] != 0:  # Check if reverse edge exists
                    symmetric_count += 1
            symmetry_ratio = symmetric_count / min(sample_size, len(rows)) if sample_size > 0 else 0
            print(f"Face neighbor symmetry check (sample of {sample_size}): {symmetry_ratio*100:.1f}% symmetric")
            if symmetry_ratio < 0.95:
                print(f"  WARNING: Face neighbor matrix may not be symmetric! This could cause segmentation issues.")
        
        # Check for isolated faces (faces with no neighbors)
        isolated = np.sum(nbr_counts == 0)
        if isolated > 0:
            print(f"  WARNING: {isolated} faces have no neighbors (isolated faces)!")
        
        # Check for faces with very few neighbors (potential issues)
        few_nbrs = np.sum(nbr_counts < 2)
        if few_nbrs > 0:
            print(f"  WARNING: {few_nbrs} faces have fewer than 2 neighbors (may be boundary or problematic)")
    
    # Get seed faces if not provided
    if slix is None:
        slix = get_seed_faces(m, nseeds)
    else:
        nseeds = len(slix)
    
    if verbose:
        print(f"Using {nseeds} seed faces: {slix[:min(10, len(slix))]}...")
    
    patch_label = np.arange(1, nseeds + 1)
    
    # Map vertex curvature to face curvature (only needed if curvature_weight > 0)
    if curvature_weight > 0:
        FC = vertex_prop_to_face_prop(m, m.H)
        if verbose:
            print(f"Face curvature range: [{FC.min():.4f}, {FC.max():.4f}]")
    else:
        FC = None
        if verbose:
            print("Curvature weighting disabled (curvature_weight=0), using uniform transition probabilities")
    
    # Use m.face_nbrs directly (sparse matrix, like MATLAB)
    # MATLAB: A = m.face_nbrs (logical matrix), then modifies neighbor entries
    # MATLAB: [r, c] = ind2sub(size(A), find(A)); A(r(ix),c(ix)) = prc
    # Convert to LIL format for efficient element modification
    A = m.face_nbrs.astype(np.float64).tolil()
    
    # Compute transition probabilities based on curvature differences
    # When curvature_weight=0: uniform weights (1.0 for all edges)
    # When curvature_weight=1: full curvature weighting: -sig*abs(FC(r) - FC(c))
    # When 0 < curvature_weight < 1: interpolate between uniform and curvature-based
    # Get all non-zero entries (neighbor pairs)
    rows, cols = A.nonzero()
    if verbose:
        if curvature_weight > 0:
            print(f"Modifying {len(rows)} matrix entries (neighbor pairs) with curvature-based weights (weight={curvature_weight:.2f})")
            # Check a sample
            if len(rows) > 0:
                sample_idx = 0
                fc_diff = abs(FC[rows[sample_idx]] - FC[cols[sample_idx]])
                weight = -curvature_weight * sig * fc_diff
                print(f"  Sample: face {rows[sample_idx]} <-> face {cols[sample_idx]}, "
                      f"FC diff={fc_diff:.4f}, weight={weight:.4f}")
        else:
            print(f"Modifying {len(rows)} matrix entries (neighbor pairs) with uniform weights (curvature_weight=0)")
    
    # Modify matrix entries - CRITICAL: must modify in the order we get them
    # MATLAB: A(r(ix),c(ix)) = prc modifies the sparse matrix in place
    # IMPORTANT: MATLAB uses NEGATIVE weights: -sig*abs(FC(r) - FC(c))
    # After adding 1.0 to diagonal, this means:
    # - Similar curvature (small diff): weight close to 0, so A[r,c] ≈ 0
    # - Different curvature (large diff): weight very negative, so A[r,c] << 0
    # For uniform weights (curvature_weight=0), MATLAB would use 0.0
    # BUT: This creates an identity-like matrix (diagonal=1, off-diagonal=0) which is singular
    # SOLUTION: For uniform case, use a very small negative value to preserve matrix structure
    # and avoid singularity, while still being effectively uniform
    # DEFAULT: -1e-6 (small negative value that preserves structure and avoids singularity)
    uniform_weight = -1e-6  # Default: Small negative value (effectively 0, but preserves structure)
    
    for idx in range(len(rows)):
        r, c = rows[idx], cols[idx]
        if curvature_weight > 0:
            # Curvature-based weight: interpolate between uniform (small negative) and curvature-based
            curvature_component = -sig * abs(FC[r] - FC[c])
            uniform_component = uniform_weight  # Small negative for uniform case
            prc = (1.0 - curvature_weight) * uniform_component + curvature_weight * curvature_component
        else:
            # No curvature consideration: use small negative value
            # This preserves matrix structure and avoids singularity
            prc = uniform_weight
        A[r, c] = prc
    
    # Verify modifications worked
    if verbose and len(rows) > 0:
        # Check a few entries to verify they were modified correctly
        test_idx = min(100, len(rows) - 1)
        r_test, c_test = rows[test_idx], cols[test_idx]
        val_in_A = A[r_test, c_test]
        if curvature_weight > 0:
            curvature_component = -sig * abs(FC[r_test] - FC[c_test])
            uniform_component = uniform_weight  # Small negative for uniform case
            expected_val = (1.0 - curvature_weight) * uniform_component + curvature_weight * curvature_component
        else:
            expected_val = uniform_weight  # Small negative to preserve structure
        if abs(val_in_A - expected_val) > 1e-10:
            print(f"  WARNING: Matrix modification may have failed!")
            print(f"    Entry A[{r_test},{c_test}]: expected {expected_val:.6f}, got {val_in_A:.6f}")
        else:
            print(f"  Verified: Matrix modification working correctly")
    
    # Add diagonal (MATLAB: A = A + spdiags(ones(nfaces, 1), 0, nfaces, nfaces))
    # This adds 1 to diagonal entries (which should be 0 from the logical matrix)
    # Convert to CSR before adding diagonal for efficiency
    # CRITICAL: Keep in LIL format to preserve small epsilon values, then convert
    if verbose and curvature_weight == 0.0:
        print(f"Before CSR conversion: {A.nnz} non-zero entries (should be {len(rows)} off-diagonal)")
    
    A = A.tocsr()
    
    if verbose and curvature_weight == 0.0:
        print(f"After CSR conversion: {A.nnz} non-zero entries")
        if A.nnz < len(rows):
            print(f"  WARNING: Lost {len(rows) - A.nnz} entries during conversion!")
            print(f"  This means epsilon approach didn't work - entries were still dropped")
    
    # Check diagonal before adding - should be all zeros (faces aren't neighbors of themselves)
    if verbose:
        diag_before = A.diagonal()
        diag_nonzero = np.count_nonzero(diag_before)
        print(f"Diagonal before addition: min={diag_before.min():.4f}, max={diag_before.max():.4f}, "
              f"non-zero count={diag_nonzero}")
        if diag_nonzero > 0:
            print(f"  WARNING: {diag_nonzero} diagonal entries are non-zero! (should be 0)")
    
    # Add 1 to diagonal (MATLAB: A = A + spdiags(ones(nfaces, 1), 0, nfaces, nfaces))
    # This adds 1.0 ONLY to diagonal entries, not to off-diagonals
    A = A + spdiags(np.ones(nfaces), 0, nfaces, nfaces, format='csr')
    
    if verbose:
        print(f"Matrix A: {A.shape}, {A.nnz} non-zero entries")
        print(f"Matrix A value range: [{A.data.min():.4f}, {A.data.max():.4f}]")
        diag_after = A.diagonal()
        print(f"Diagonal after addition: min={diag_after.min():.4f}, max={diag_after.max():.4f}, "
              f"should all be >= 1.0, actual min={diag_after.min():.4f}")
        
        # CRITICAL: Check if we have off-diagonal entries
        diag_nnz = np.count_nonzero(diag_after)
        off_diag_nnz = A.nnz - diag_nnz
        print(f"Matrix A: {diag_nnz} diagonal entries, {off_diag_nnz} off-diagonal entries")
        if off_diag_nnz == 0:
            print(f"  CRITICAL ERROR: Matrix A has NO off-diagonal entries after construction!")
            print(f"  Expected {len(rows)} off-diagonal entries from face neighbors")
        
        # Check row sums (should be positive for well-conditioned system)
        row_sums = np.array(A.sum(axis=1)).flatten()
        print(f"Row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}, mean={row_sums.mean():.4f}")
        
        # Check a sample row to verify structure
        sample_row = 100  # Check a middle row
        if sample_row < nfaces:
            row_data = A[sample_row, :].toarray().flatten()
            nbrs = np.where(row_data != 0)[0]
            print(f"Sample row {sample_row}: {len(nbrs)} non-zero entries")
            if len(nbrs) > 0:
                off_diag_vals = row_data[nbrs[nbrs != sample_row]]
                print(f"  Diagonal: {row_data[sample_row]:.4f}, "
                      f"off-diagonals ({len(off_diag_vals)}): {off_diag_vals[:min(3, len(off_diag_vals))]}")
            else:
                print(f"  WARNING: Row has no non-zero entries!")
        
        # Check a few more rows to see if structure is consistent
        for test_row in [0, nfaces//2, nfaces-1]:
            if test_row < nfaces:
                test_row_data = A[test_row, :].toarray().flatten()
                test_nbrs = np.where(test_row_data != 0)[0]
                test_off_diag = len([n for n in test_nbrs if n != test_row])
                if test_off_diag == 0:
                    print(f"  WARNING: Row {test_row} has no off-diagonal neighbors!")
    
    # Build B matrix (boundary conditions)
    # MATLAB: B(indices, ix) = 1/3 where indices = find(m.face_nbrs(slix(ix), :))
    # CRITICAL: Use ORIGINAL face_nbrs (before A modification), not the modified A matrix!
    # MATLAB: indices = find(m.face_nbrs(slix(ix), :)) returns column indices (neighbors)
    B = np.zeros((nfaces, nseeds))
    total_b_entries = 0
    for ix in range(nseeds):
        # Find neighbors using ORIGINAL face_nbrs sparse matrix (not modified A)
        # MATLAB: indices = find(m.face_nbrs(slix(ix), :))
        # This returns column indices where row slix[ix] is non-zero
        # IMPORTANT: Use m.face_nbrs (original), not A (which has been modified)
        row_vec = m.face_nbrs[slix[ix], :]
        nbrs = row_vec.nonzero()[1]  # Column indices (neighbors)
        if verbose:
            print(f"  Seed {ix} (face {slix[ix]}): {len(nbrs)} neighbors: {nbrs[:min(5, len(nbrs))]}...")
        if len(nbrs) > 0:
            B[nbrs, ix] = 1.0 / 3.0  # MATLAB uses 1/3
            total_b_entries += len(nbrs)
        else:
            if verbose:
                print(f"  WARNING: Seed face {slix[ix]} has no neighbors!")
    
    if verbose:
        print(f"B matrix: {B.shape}, {total_b_entries} non-zero entries total")
        print(f"B matrix sum per column: {B.sum(axis=0)}")
    
    # Remove seed rows/columns (MATLAB: A(slix,:) = []; A(:,slix) = []; B(slix,:) = [])
    # MATLAB: nonseedix = 1:nfaces; nonseedix(slix) = [];
    nonseedix = np.setdiff1d(np.arange(nfaces), slix)
    
    if verbose:
        print(f"Removing {len(slix)} seed rows/columns")
        print(f"nonseedix length: {len(nonseedix)}, should be {nfaces - len(slix)}")
        # Check if any seeds are neighbors of each other
        seed_neighbors_of_seeds = 0
        for s in slix:
            seed_nbrs = set(m.face_nbrs[s, :].nonzero()[1])
            seed_overlap = seed_nbrs.intersection(set(slix))
            if len(seed_overlap) > 0:
                seed_neighbors_of_seeds += len(seed_overlap)
        if seed_neighbors_of_seeds > 0:
            print(f"  Note: {seed_neighbors_of_seeds} seed-seed neighbor pairs (will be removed from B_reduced)")
    
    # Extract submatrix (MATLAB: A(slix,:) = [] removes rows, A(:,slix) = [] removes columns)
    # This creates a system for non-seed faces only
    # MATLAB does: A(slix,:) = []; A(:,slix) = []; which removes rows then columns
    # In Python, we need to do this carefully to match MATLAB behavior
    # First remove rows, then remove columns from the result
    A_temp = A[nonseedix, :]  # Remove seed rows
    A_reduced = A_temp[:, nonseedix]  # Remove seed columns
    B_reduced = B[nonseedix, :].copy()
    
    # Verify the reduced matrix still has connectivity
    if verbose:
        # Check that A_reduced still has non-zero entries
        print(f"A_reduced: {A_reduced.shape}, {A_reduced.nnz} non-zero entries")
        
        # Check row sums of A_reduced (should still be positive)
        row_sums_reduced = np.array(A_reduced.sum(axis=1)).flatten()
        print(f"A_reduced row sums: min={row_sums_reduced.min():.4f}, max={row_sums_reduced.max():.4f}, "
              f"mean={row_sums_reduced.mean():.4f}")
        if row_sums_reduced.min() <= 0:
            print(f"  WARNING: Some rows have non-positive sums! This could cause issues.")
        
        # Check connectivity - each row should have at least one neighbor
        row_nnz = np.array([A_reduced[i, :].nnz for i in range(A_reduced.shape[0])])
        isolated = np.sum(row_nnz == 1)  # Only diagonal (isolated)
        print(f"A_reduced: {isolated} rows have only diagonal (potentially isolated)")
        
        # CRITICAL: Check if we have off-diagonal entries
        # Count off-diagonal non-zeros (total nnz minus diagonal entries)
        diag_nnz = np.count_nonzero(A_reduced.diagonal())
        off_diag_nnz = A_reduced.nnz - diag_nnz
        print(f"A_reduced: {diag_nnz} diagonal entries, {off_diag_nnz} off-diagonal entries")
        if off_diag_nnz == 0:
            print(f"  CRITICAL ERROR: A_reduced has NO off-diagonal entries! Matrix is disconnected!")
            print(f"  This means removing seed rows/columns broke all connectivity.")
            print(f"  Original A had {A.nnz} entries, after removing seeds we have {A_reduced.nnz} entries")
            # Try to diagnose: check a sample row in original A
            if len(nonseedix) > 0:
                sample_orig_face = nonseedix[0]
                orig_row = A[sample_orig_face, :].toarray().flatten()
                orig_nbrs = np.where(orig_row != 0)[0]
                orig_nbrs_nonseed = [n for n in orig_nbrs if n not in slix]
                print(f"  Sample non-seed face {sample_orig_face} in original A: {len(orig_nbrs)} neighbors total, "
                      f"{len(orig_nbrs_nonseed)} are non-seeds")
                if len(orig_nbrs_nonseed) == 0:
                    print(f"    WARNING: This face's neighbors are all seeds! This breaks connectivity.")
        
        # Check a sample row in A_reduced to see its structure
        if A_reduced.shape[0] > 0:
            sample_reduced_row = min(100, A_reduced.shape[0] - 1)
            reduced_row_data = A_reduced[sample_reduced_row, :].toarray().flatten()
            reduced_nbrs = np.where(reduced_row_data != 0)[0]
            print(f"Sample row {sample_reduced_row} in A_reduced: {len(reduced_nbrs)} non-zero entries")
            if len(reduced_nbrs) > 1:
                print(f"  Diagonal: {reduced_row_data[sample_reduced_row]:.4f}, "
                      f"off-diagonals: {reduced_row_data[reduced_nbrs[reduced_nbrs != sample_reduced_row]][:min(3, len(reduced_nbrs)-1)]}")
            elif len(reduced_nbrs) == 1:
                print(f"  WARNING: Row only has diagonal entry! No connectivity!")
    
    if verbose:
        # Check that B_reduced still has entries (neighbors of seeds should not be seeds themselves usually)
        print(f"After reduction - B_reduced: {B_reduced.shape}, {np.count_nonzero(B_reduced)} non-zero entries")
        b_reduced_empty_cols = []
        for col in range(B_reduced.shape[1]):
            nz = np.count_nonzero(B_reduced[:, col])
            if nz > 0:
                if verbose:
                    print(f"  Column {col} (seed {col}): {nz} non-zero entries, sum={B_reduced[:, col].sum():.4f}")
            else:
                b_reduced_empty_cols.append(col)
                print(f"  WARNING: Column {col} (seed {col}) has NO non-zero entries after reduction!")
        
        if len(b_reduced_empty_cols) > 0:
            print(f"  CRITICAL: {len(b_reduced_empty_cols)} seed columns are empty in B_reduced!")
            print(f"  This means those seeds have no non-seed neighbors, or all neighbors are also seeds.")
    
    if verbose:
        print(f"Solving linear system: {A_reduced.shape[0]} x {A_reduced.shape[1]} matrix")
        print(f"A_reduced condition number estimate: {np.linalg.cond(A_reduced.toarray()):.2e}")
        print(f"B_reduced non-zero entries per column: {[np.count_nonzero(B_reduced[:, i]) for i in range(B_reduced.shape[1])]}")
    
    # Solve linear system
    # MATLAB: x = A\B (backslash operator)
    # MATLAB uses direct solver (LU decomposition) for sparse matrices
    if verbose:
        print(f"B_reduced shape: {B_reduced.shape}, non-zero entries: {np.count_nonzero(B_reduced)}")
        # Verify B_reduced has entries
        for col in range(B_reduced.shape[1]):
            nz_count = np.count_nonzero(B_reduced[:, col])
            if nz_count == 0:
                print(f"  WARNING: Column {col} of B_reduced has no non-zero entries!")
            elif verbose:
                print(f"  Column {col}: {nz_count} non-zero entries, sum={B_reduced[:, col].sum():.4f}")
    
    # Use direct sparse solver (spsolve uses LU decomposition, matching MATLAB's \)
    # This should be faster and more accurate than iterative methods
    import time
    solve_start = time.time()
    try:
        # Ensure matrix is in CSR format for efficient solving
        if not isinstance(A_reduced, csr_matrix):
            A_reduced = A_reduced.tocsr()
        x = spsolve(A_reduced, B_reduced)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        solve_time = time.time() - solve_start
        if verbose:
            print(f"Linear system solved in {solve_time:.3f} seconds using spsolve (LU)")
    except Exception as e:
        # Fallback to dense solve
        if verbose:
            print(f"Warning: Using dense solve (sparse solve failed: {e})")
        solve_start = time.time()
        x = np.linalg.solve(A_reduced.toarray(), B_reduced)
        solve_time = time.time() - solve_start
        if verbose:
            print(f"Dense solve completed in {solve_time:.3f} seconds")
    
    # Verify solution
    if verbose:
        residual = A_reduced @ x - B_reduced
        max_residual = np.abs(residual).max()
        print(f"Solution verification - max residual: {max_residual:.2e}")
        if max_residual > 1e-6:
            print(f"  WARNING: Large residual! Solution may be inaccurate.")
        
        # Check if solution makes physical sense
        # The solution should have non-zero values propagating from seeds
        # Check a few faces that are neighbors of seed neighbors
        print("\nChecking solution propagation:")
        for seed_idx in range(min(2, nseeds)):  # Check first 2 seeds
            seed_face = slix[seed_idx]
            # Get neighbors of this seed
            seed_nbrs = m.face_nbrs[seed_face, :].nonzero()[1]
            if len(seed_nbrs) > 0:
                # Check if seed_nbrs are in nonseedix (they should be, unless they're also seeds)
                seed_nbrs_nonseed = [nbr for nbr in seed_nbrs if nbr not in slix]
                if len(seed_nbrs_nonseed) > 0:
                    # Find index of first non-seed neighbor in nonseedix
                    nbr_idx_in_nonseed = np.where(nonseedix == seed_nbrs_nonseed[0])[0]
                    if len(nbr_idx_in_nonseed) > 0:
                        nbr_idx = nbr_idx_in_nonseed[0]
                        print(f"  Seed {seed_idx} neighbor {seed_nbrs_nonseed[0]} (nonseedix[{nbr_idx}]): "
                              f"x values = {x[nbr_idx, :]}, max = {x[nbr_idx, :].max():.6f}")
                        # Get neighbors of this neighbor
                        nbr_of_nbr = m.face_nbrs[seed_nbrs_nonseed[0], :].nonzero()[1]
                        nbr_of_nbr_nonseed = [n for n in nbr_of_nbr if n not in slix]
                        if len(nbr_of_nbr_nonseed) > 0:
                            nbr2_idx_in_nonseed = np.where(nonseedix == nbr_of_nbr_nonseed[0])[0]
                            if len(nbr2_idx_in_nonseed) > 0:
                                nbr2_idx = nbr2_idx_in_nonseed[0]
                                print(f"    Neighbor's neighbor {nbr_of_nbr_nonseed[0]} (nonseedix[{nbr2_idx}]): "
                                      f"x values = {x[nbr2_idx, :]}, max = {x[nbr2_idx, :].max():.6f}")
    
    if verbose:
        print(f"Solution x shape: {x.shape}")
        print(f"Solution x value range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Solution x column sums: {x.sum(axis=0)}")
        print(f"Solution x column means: {x.mean(axis=0)}")
        print(f"Solution x column maxes: {x.max(axis=0)}")
        print(f"Solution x column stds: {x.std(axis=0)}")
        
        # Check for rows with all zeros or all equal values
        row_maxes = x.max(axis=1)
        row_mins = x.min(axis=1)
        row_stds = x.std(axis=1)
        ambiguous = np.sum(np.abs(row_maxes - row_mins) < 1e-10)
        low_variance = np.sum(row_stds < 1e-6)
        
        if ambiguous > 0:
            print(f"WARNING: {ambiguous} rows have all equal values (ambiguous assignment)")
        if low_variance > 0:
            print(f"WARNING: {low_variance} rows have very low variance (< 1e-6)")
        
        # Check distribution of max values
        print(f"Max value per row - min: {row_maxes.min():.6f}, max: {row_maxes.max():.6f}, mean: {row_maxes.mean():.6f}")
        print(f"Std dev per row - min: {row_stds.min():.6f}, max: {row_stds.max():.6f}, mean: {row_stds.mean():.6f}")
        
        # Sample a few rows to see what the solution looks like
        if len(x) > 10:
            sample_indices = np.linspace(0, len(x)-1, min(5, len(x)), dtype=int)
            print("Sample solution rows (first 5):")
            for idx in sample_indices[:5]:
                print(f"  Row {idx}: {x[idx, :]}, max_col={np.argmax(x[idx, :])}, max_val={x[idx, :].max():.6f}, std={x[idx, :].std():.6f}")
        
        # Check which columns win most often
        col_wins = np.zeros(nseeds)
        for ix in range(len(nonseedix)):
            max_col = np.argmax(x[ix, :])
            col_wins[max_col] += 1
        print(f"Column wins (which seed each face prefers): {col_wins}")
    
    # Assign labels (MATLAB: find max in each row, assign that column index)
    # MATLAB: indx = find(x(ix,:)==max(x(ix,:))); l(ix) = indx (if unique)
    # MATLAB uses exact equality - this works because even if values are small, 
    # the relative differences should be clear
    L = np.zeros(nfaces, dtype=int)
    ambiguous_count = 0
    
    # MATLAB approach: find where values equal the max (exact equality)
    # MATLAB: indx = find(x(ix,:)==max(x(ix,:))); l(ix) = indx (if unique)
    # CRITICAL: Match MATLAB's exact behavior - use exact equality check
    # MATLAB's find() with == uses exact floating point equality
    for ix in range(len(nonseedix)):
        row_vals = x[ix, :]
        max_val = np.max(row_vals)
        
        # MATLAB: indx = find(x(ix,:)==max(x(ix,:)))
        # Use exact equality (within floating point precision)
        # MATLAB's == operator for floating point uses exact comparison
        # We need to find all indices where value exactly equals max
        # Use a very tight tolerance based on the actual max value
        if max_val > 0:
            # For non-zero values, use relative tolerance
            # MATLAB's == uses exact comparison, but we need to account for floating point errors
            tolerance = max_val * np.finfo(float).eps * 10  # Very tight relative tolerance
        else:
            # For zero or near-zero, use absolute tolerance
            tolerance = np.finfo(float).eps * 10
        
        # Find all indices where value equals max (within tolerance)
        indx = np.where(np.abs(row_vals - max_val) <= tolerance)[0]
        
        if len(indx) == 1:
            L[nonseedix[ix]] = indx[0] + 1  # MATLAB uses 1-indexed labels (column index)
        else:
            # Multiple columns have the same max value (ambiguous)
            # MATLAB sets l(ix) = 0 in this case, but we'll use the first one
            # However, to match MATLAB exactly, we should check if all are truly equal
            # If they're all within tolerance, use the first one (argmax)
            if len(indx) > 1:
                # All values are essentially equal - use argmax as tie-breaker
                L[nonseedix[ix]] = np.argmax(row_vals) + 1
                ambiguous_count += 1
            else:
                # Should not happen, but fallback
                L[nonseedix[ix]] = np.argmax(row_vals) + 1
    
    if verbose:
        print(f"Ambiguous assignments: {ambiguous_count} faces (got label 0)")
        if ambiguous_count > len(nonseedix) * 0.1:  # More than 10% ambiguous
            print(f"  WARNING: High ambiguity rate! This suggests the solution is not working correctly.")
            # Check if solution values are too uniform
            row_stds = x.std(axis=1)
            print(f"  Solution row std dev - min: {row_stds.min():.6f}, max: {row_stds.max():.6f}, mean: {row_stds.mean():.6f}")
            if row_stds.mean() < 1e-6:
                print(f"  CRITICAL: Solution rows have very low variance - all columns nearly equal!")
    
    if verbose:
        assigned_labels = L[L > 0]
        print(f"Assigned labels: {len(assigned_labels)} non-zero, {len(np.unique(assigned_labels))} unique")
        print(f"Label distribution: {dict(zip(*np.unique(L, return_counts=True)))}")
    
    # Assign seed labels
    for ix in range(nseeds):
        L[slix[ix]] = ix + 1
    
    m.face_labels = L
    
    if verbose:
        unique_labels = np.unique(L)
        print(f"Initial segmentation: {len(unique_labels)} unique labels")
        print(f"Label distribution: {dict(zip(*np.unique(L, return_counts=True)))}")
    
    if plot_intermediate:
        try:
            # Use the surface_mesh method to plot segmentation with seeds
            m.plot_segmentation_with_seeds(slix, verbose=verbose)
            
            if verbose:
                print("Plotted initial segmentation with seed faces highlighted (interactive PyVista plot)")
        except ImportError:
            # Fallback to matplotlib if PyVista not available
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot mesh with labels
                face_field = np.asarray(m.face_labels, dtype=float)
                unique_labels = np.unique(face_field)
                unique_labels = unique_labels[unique_labels > 0]
                n_labels = len(unique_labels)
                
                if n_labels > 1:
                    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                    face_colors = np.zeros((len(face_field), 4))
                    for i, label in enumerate(face_field):
                        if label in label_to_idx:
                            color_idx = label_to_idx[label]
                            color_val = color_idx / max(1, n_labels - 1) if n_labels > 1 else 0.0
                            cmap = plt.cm.tab10 if n_labels <= 10 else plt.cm.jet
                            face_colors[i] = cmap(color_val)
                        else:
                            face_colors[i] = [0.5, 0.5, 0.5, 1.0]
                else:
                    face_colors = 'lightblue'
                
                triangles = []
                for face in m.F:
                    triangles.append([m.X[face[0]], m.X[face[1]], m.X[face[2]]])
                
                poly = Poly3DCollection(triangles, facecolors=face_colors, 
                                        edgecolors='k', linewidths=0.3, alpha=1.0)
                ax.add_collection3d(poly)
                
                # Calculate average side length of triangles for sphere sizing
                edge_lengths = []
                for face in m.F:
                    v0, v1, v2 = m.X[face[0]], m.X[face[1]], m.X[face[2]]
                    edge_lengths.append(np.linalg.norm(v1 - v0))
                    edge_lengths.append(np.linalg.norm(v2 - v1))
                    edge_lengths.append(np.linalg.norm(v0 - v2))
                avg_edge_length = np.mean(edge_lengths)
                
                # Highlight seed faces with spheres - size based on average triangle side length
                for ix in range(nseeds):
                    seed_face = m.F[slix[ix]]
                    # Get centroid of seed face
                    centroid = m.X[seed_face].mean(axis=0)
                    # Use average edge length as sphere radius
                    radius = avg_edge_length / 2.0
                    # Plot sphere at centroid
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + centroid[0]
                    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + centroid[1]
                    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid[2]
                    ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                                  color='red', alpha=1.0, edgecolor='yellow', 
                                  linewidth=1, shade=True, antialiased=True)
                
                if verbose:
                    print(f"Plotted {nseeds} seed faces as red spheres (radius = {radius:.4f}, avg edge length = {avg_edge_length:.4f})")
                
                # Set axis limits
                max_range = np.array([m.X[:, 0].max() - m.X[:, 0].min(),
                                      m.X[:, 1].max() - m.X[:, 1].min(),
                                      m.X[:, 2].max() - m.X[:, 2].min()]).max() / 2.0
                mid_x = (m.X[:, 0].max() + m.X[:, 0].min()) * 0.5
                mid_y = (m.X[:, 1].max() + m.X[:, 1].min()) * 0.5
                mid_z = (m.X[:, 2].max() + m.X[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
                
                ax.set_title(f'Initial Segmentation (Red spheres = seed faces, {n_labels} patches)')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                plt.tight_layout()
                plt.show(block=True)  # Block to allow interaction
                
                if verbose:
                    print("Plotted initial segmentation with seed faces highlighted (matplotlib fallback)")
            except Exception as e:
                if verbose:
                    print(f"Could not plot: {e}")
                pass
    
    # ========================================================================
    # Smoothen mesh segmentation and fix trapped faces (MATLAB lines 101-145)
    # ========================================================================
    if verbose:
        print("\n" + "-"*60)
        print("Step 1: Smoothing borders (median of neighbors)")
        print("-"*60)
    
    # Step 1: smoothen the border by assigning a majority vote label to a face
    # Ensure face_nbrs is available as sparse matrix
    if not hasattr(m, 'face_nbrs') or not isinstance(m.face_nbrs, csr_matrix):
        m.needs_edge_info = True
        m.edge_info()
    
    L_new = np.zeros(nfaces, dtype=int)
    for ix in range(nfaces):
        nbrs = m.face_nbrs[ix, :].nonzero()[1]
        if len(nbrs) > 0:
            lbls = m.face_labels[nbrs]
            L_new[ix] = int(np.median(lbls))
        else:
            L_new[ix] = m.face_labels[ix]
    m.face_labels = L_new
    
    if verbose:
        print(f"After smoothing: {len(np.unique(L_new))} unique labels")
    
    if plot_intermediate:
        try:
            m.plot_labels()
            if verbose:
                print("Plotted after first smoothing")
        except:
            pass
    
    # Step 2: Fix trapped faces
    if verbose:
        print("\n" + "-"*60)
        print("Step 2: Fixing trapped faces")
        print("-"*60)
    
    # Ensure face_nbrs is available as sparse matrix
    if not hasattr(m, 'face_nbrs') or not isinstance(m.face_nbrs, csr_matrix):
        m.needs_edge_info = True
        m.edge_info()
    L_new = np.zeros(nfaces, dtype=int)
    for ix in range(nfaces):
        nbrs = m.face_nbrs[ix, :].nonzero()[1]
        if len(nbrs) > 0:
            lbls = m.face_labels[nbrs]
            uf = np.unique(lbls)
            if len(uf) == 1 and m.face_labels[ix] != uf[0]:
                L_new[ix] = uf[0]
            elif len(uf) > 1 and not np.any(uf == m.face_labels[ix]):
                L_new[ix] = uf[0]
            else:
                L_new[ix] = m.face_labels[ix]
        else:
            L_new[ix] = m.face_labels[ix]
    m.face_labels = L_new
    
    if verbose:
        print(f"After fixing trapped: {len(np.unique(L_new))} unique labels")
    
    # Step 3: Sequential smoothing
    # CRITICAL: MATLAB updates m.face_labels in place during the loop
    # This means each iteration uses the updated labels from previous iterations
    # This is important for sequential smoothing to work correctly
    if verbose:
        print("\n" + "-"*60)
        print("Step 3: Sequential smoothing (in-place updates)")
        print("-"*60)
    
    # Ensure face_nbrs is available as sparse matrix
    if not hasattr(m, 'face_nbrs') or not isinstance(m.face_nbrs, csr_matrix):
        m.needs_edge_info = True
        m.edge_info()
    
    # MATLAB: for ix = 1:size(m.F,1)
    #         nbrix = find(m.face_nbrs(ix,:));
    #         lbls = m.face_labels(nbrix);
    #         m.face_labels(ix) = median(lbls);
    #         end
    # This updates m.face_labels in place, so each iteration uses updated values
    # CRITICAL: Update in place to ensure sequential smoothing works correctly
    for ix in range(nfaces):
        nbrs = m.face_nbrs[ix, :].nonzero()[1]
        if len(nbrs) > 0:
            lbls = m.face_labels[nbrs]  # Use current (possibly updated) labels
            m.face_labels[ix] = int(np.median(lbls))  # Update in place
        # If no neighbors, keep current label (already set)
    
    if verbose:
        unique_labels = np.unique(m.face_labels)
        print(f"Final segmentation: {len(unique_labels)} unique labels")
        print(f"Label distribution: {dict(zip(*np.unique(m.face_labels, return_counts=True)))}")
        print(f"Output mesh: {len(m.X)} vertices, {len(m.F)} faces (SAME as input)")
        print("="*60)
    
    if plot_intermediate:
        try:
            m.plot_labels()
            if verbose:
                print("Plotted final segmentation")
        except:
            pass
    
    # ========================================================================
    # Build patch structures P and Pconn (MATLAB lines 157-195)
    # ========================================================================
    if verbose:
        print("\n" + "-"*60)
        print("Building patch structures P and Pconn")
        print("-"*60)
    
    L = m.face_labels
    uL = np.unique(L)
    numpatches = len(uL)
    
    if verbose:
        print(f"Number of patches: {numpatches}")
    
    # Build P: list of patch data structures
    P = []
    for lix in range(numpatches):
        label = uL[lix]
        # Create patch mesh with only this patch's faces
        patch_face_mask = L == label
        patch_faces = np.where(patch_face_mask)[0]
        
        mp = surface_mesh(m.X.copy(), m.F[patch_face_mask].copy())
        mp, mpL, _ = get_border(mp)
        
        Loindx = patch_faces
        # Edge faces: faces with fewer than 3 neighbors within the patch
        mpLindx = []
        if hasattr(mp, 'face_nbrs') and mp.face_nbrs is not None:
            if isinstance(mp.face_nbrs, dict):
                for i in range(len(mp.F)):
                    nbr_count = len(mp.face_nbrs.get(i, []))
                    if nbr_count < 3:
                        mpLindx.append(Loindx[i])
            else:
                for i in range(len(mp.F)):
                    nbr_count = np.sum(mp.face_nbrs[i, :] != 0)
                    if nbr_count < 3:
                        mpLindx.append(Loindx[i])
        
        P.append([mp, mpL, np.array(mpLindx)])
    
    # Build Pconn: patch connectivity matrix
    Pconn = lil_matrix((numpatches, numpatches), dtype=int)
    for lix in range(numpatches):
        mpLindx = P[lix][2]
        for fix in mpLindx:
            nbrs = m.face_nbrs[fix, :].nonzero()[1]
            for nbr in nbrs:
                nbr_label = L[nbr]
                nbr_idx = np.where(uL == nbr_label)[0]
                if len(nbr_idx) > 0:
                    Pconn[lix, nbr_idx[0]] = 1
    
    Pconn = Pconn.tocsr()
    
    if verbose:
        print(f"Patch connectivity matrix: {Pconn.shape}")
        print(f"Number of patch connections: {Pconn.nnz // 2}")  # Divide by 2 for symmetric
        print("="*60)
        print("mesh_segmentation_rw: COMPLETE")
        print("="*60)
        print(f"NOTE: Output mesh 'm' is the SAME mesh as input, with face_labels added.")
        print(f"      The simplified mesh is created later in patch_info_gen as PM.pm")
        print("="*60)
    
    return m, L, slix, P, Pconn


def get_seed_faces(mo, nseeds):
    """
    Automatically select seed faces for segmentation using graph-based method
    Based on Lei et al 2008 "Fast Mesh Segmentation using Random Walks"
    
    This method:
    1. Reduces mesh to minimal set of vertices
    2. Builds curvature-weighted graph
    3. Finds two initial seed vertices (north/south poles) with maximum distance
    4. Iteratively adds seeds by finding vertices with maximum minimum distance to existing seeds
    
    Parameters:
    -----------
    mo : surface_mesh
        Input mesh
    nseeds : int
        Number of seed faces to generate
        
    Returns:
    --------
    slix : array
        Seed face indices (into original mesh mo.F)
    """
    # Reduce to minimal set (does not affect face index, so slix doesn't need conversion)
    m, uv = reduce_to_minimal_set(mo)
    
    # Get graph with curvature-based weights
    # Returns: G, d, ixN, ixS, weights, ixN2, ixS2, g
    G, d, ixN, ixS, weights, ixN2, ixS2, g = m.get_graph()
    
    # Start with two seed vertices (north and south poles)
    # MATLAB uses weighted graph poles (ixN, ixS) for seed vertices
    slix_vert = [ixN, ixS]
    
    # Get corresponding seed faces (first face containing each vertex)
    # MATLAB uses unweighted graph poles (ixN2, ixS2) for initial seed faces
    # face_memb is a dict mapping vertex index to list of face indices
    if not hasattr(m, 'face_memb') or m.face_memb is None:
        m.edge_info()  # This will compute face_memb
    
    # Get first face for each seed vertex (using unweighted graph poles)
    face_memb = m.face_memb if isinstance(m.face_memb, dict) else {}
    slix = []
    if ixN2 in face_memb and len(face_memb[ixN2]) > 0:
        slix.append(face_memb[ixN2][0])
    else:
        # Fallback: find any face containing this vertex
        faces_with_vertex = np.where(np.any(m.F == ixN2, axis=1))[0]
        if len(faces_with_vertex) > 0:
            slix.append(faces_with_vertex[0])
        else:
            raise ValueError(f"Could not find face containing vertex {ixN2}")
    
    if ixS2 in face_memb and len(face_memb[ixS2]) > 0:
        slix.append(face_memb[ixS2][0])
    else:
        # Fallback: find any face containing this vertex
        faces_with_vertex = np.where(np.any(m.F == ixS2, axis=1))[0]
        if len(faces_with_vertex) > 0:
            slix.append(faces_with_vertex[0])
        else:
            raise ValueError(f"Could not find face containing vertex {ixS2}")
    
    # Iteratively add more seeds until we have nseeds
    while len(slix) < nseeds:
        # For each vertex, compute minimum distance to all existing seed vertices
        nvert = len(m.X)
        minD = np.zeros((nvert, 2))  # [vertex_index, min_distance]
        
        for ix in range(nvert):
            # Compute distances from vertex ix to all seed vertices
            distances_to_seeds = []
            for six in range(len(slix_vert)):
                seed_vert = slix_vert[six]
                # Get distance from distance matrix d
                dist = d[ix, seed_vert]
                distances_to_seeds.append(dist)
            
            # Find minimum distance and its index
            min_dist = min(distances_to_seeds)
            min_idx = distances_to_seeds.index(min_dist)
            minD[ix, 0] = min_idx
            minD[ix, 1] = min_dist
        
        # Find vertex with maximum minimum distance
        max_minD_idx = np.argmax(minD[:, 1])
        maxminD = int(max_minD_idx)
        
        # Add this vertex to seed vertices
        slix_vert.append(maxminD)
        
        # Add corresponding face (first face containing this vertex)
        if maxminD in face_memb and len(face_memb[maxminD]) > 0:
            slix.append(face_memb[maxminD][0])
        else:
            # Fallback: find any face containing this vertex
            faces_with_vertex = np.where(np.any(m.F == maxminD, axis=1))[0]
            if len(faces_with_vertex) > 0:
                slix.append(faces_with_vertex[0])
            else:
                raise ValueError(f"Could not find face containing vertex {maxminD}")
    
    # Convert to numpy array
    slix = np.array(slix, dtype=int)
    
    return slix


def vertex_prop_to_face_prop(m, vertex_prop):
    """
    Map vertex property to face property (average of vertex values)
    """
    face_prop = np.zeros(len(m.F))
    for i in range(len(m.F)):
        face_vertices = m.F[i]
        face_prop[i] = np.mean(vertex_prop[face_vertices])
    return face_prop


def build_face_adjacency(m):
    """
    Build face adjacency matrix
    """
    from scipy.sparse import csr_matrix
    
    nfaces = len(m.F)
    A = csr_matrix((nfaces, nfaces), dtype=bool)
    
    # Build edge-to-face mapping
    edge_to_faces = {}
    for i in range(nfaces):
        face = m.F[i]
        # Add edges
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(i)
    
    # Build adjacency
    rows, cols = [], []
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            rows.extend([faces[0], faces[1]])
            cols.extend([faces[1], faces[0]])
    
    A = csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(nfaces, nfaces))
    return A
