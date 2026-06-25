"""
Parameterize patches using Cartesian coordinates
Translated from MATLAB level1/parameterize_patches_cart.m

This module maps each patch to the sphere using:
1. Linear interpolation of boundary values from simplified mesh
2. Solving Laplace equation for interior vertices

Per-patch strategy: For each patch, the boundary vertices (edge chains
connecting key vertices of the simplified mesh) are assigned spherical
positions by interpolation.  The interior vertices are then determined
by solving the Laplace (Dirichlet) boundary value problem.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from ..utils import kk_cart2sph, kk_sph2cart
from ..level0.mesh_utils import reduce_to_minimal_set
import datetime
import uuid


def _make_run_header(run_id=None):
    """Return a one-line header with run-ID and current timestamp."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rid = run_id or 'unknown'
    return f"Run ID: {rid}  |  Timestamp: {ts}"


# ---------------------------------------------------------------------------
# Helpers for multi-boundary patches (annular patches with inner + outer ring)
# ---------------------------------------------------------------------------

def _build_boundary_cycles(patm):
    """
    Build ordered boundary cycles from the patch's border edges.

    For a patch with a single boundary component, returns [cycle].
    For an annular patch (two boundaries), returns [cycle1, cycle2].

    Each cycle is an ordered list of vertex indices (in m.X / patm.X space).
    Only includes cycles with >= 3 vertices.
    """
    if not hasattr(patm, 'border_vertex') or patm.border_vertex is None:
        return []

    border_mask = patm.border_vertex.astype(bool)
    border_verts = np.where(border_mask)[0]

    if len(border_verts) == 0:
        return []

    # Ensure link data is populated
    if not hasattr(patm, 'L') or patm.L is None or len(patm.L) == 0:
        patm.edge_info()

    L = patm.L if isinstance(patm.L, dict) else {}
    border_set = set(border_verts.tolist())

    visited = set()
    cycles = []

    for start_v in border_verts:
        start = int(start_v)
        if start in visited:
            continue

        # Walk boundary edges to build one cycle
        cycle = [start]
        visited.add(start)
        current = start

        while True:
            neighbors = L.get(current, [])
            next_vert = None

            for nbr in neighbors:
                nbr_int = int(nbr)
                if nbr_int in border_set and nbr_int not in visited:
                    next_vert = nbr_int
                    break

            if next_vert is None:
                break

            cycle.append(next_vert)
            visited.add(next_vert)
            current = next_vert

        if len(cycle) >= 3:
            cycles.append(cycle)

    return cycles


def _detect_chain_jumps(patm_X, chain, threshold_factor=5.0):
    """Detect spatial jumps in an edge chain.

    Consecutive vertices in a valid chain should be separated by
    approximately one mesh-edge length.  A "jump" occurs when the
    distance between consecutive vertices is much larger than the
    median, indicating the chain walk failed to find a continuous
    path and appended a distant vertex.

    Parameters
    ----------
    patm_X : ndarray (N, 3)
        Vertex positions (original mesh coordinates).
    chain : array-like of int
        Ordered vertex indices forming the chain.
    threshold_factor : float
        A gap > threshold_factor * median_gap is flagged as a jump.

    Returns
    -------
    jump_indices : list of int
        Indices *i* such that ``chain[i] -> chain[i+1]`` is a jump.
    """
    chain = np.asarray(chain, dtype=int)
    if len(chain) < 3:
        return []

    dists = np.array([
        np.linalg.norm(patm_X[chain[i + 1]] - patm_X[chain[i]])
        if chain[i] < len(patm_X) and chain[i + 1] < len(patm_X)
        else np.inf
        for i in range(len(chain) - 1)
    ])

    median_d = float(np.median(dists[np.isfinite(dists)])) if np.any(np.isfinite(dists)) else 0.0
    threshold = max(median_d * threshold_factor, 1e-8)

    return [i for i, d in enumerate(dists) if d > threshold]


def _fill_boundary_gaps(patm, x, y, z, fixed_mask):
    """
    For border vertices not covered by edge chains, interpolate along
    boundary cycles between the nearest fixed vertices.

    This replaces a naive nearest-neighbour fill and is critical for
    annular patches where one boundary component (e.g. the cap ring)
    is only partially covered by edge chains.

    Parameters
    ----------
    patm : surface_mesh
        Patch mesh (used for border_vertex and L)
    x, y, z : ndarray
        Cartesian coordinates on the unit sphere (modified in-place)
    fixed_mask : ndarray, bool
        Boolean mask – True for vertices with valid boundary values

    Returns
    -------
    x, y, z : ndarray
        Updated Cartesian coordinates
    filled : ndarray, bool
        Boolean mask of vertices that were filled by interpolation
    """
    filled = np.zeros(len(x), dtype=bool)
    cycles = _build_boundary_cycles(patm)

    for cycle in cycles:
        n = len(cycle)

        # Which cycle vertices are already fixed?
        cycle_is_fixed = [bool(fixed_mask[v]) for v in cycle]

        if not any(cycle_is_fixed) or all(cycle_is_fixed):
            continue  # nothing to do

        # Collect interpolated values BEFORE applying (avoids drift)
        new_values = {}  # vertex -> (xi, yi, zi)

        for i in range(n):
            if cycle_is_fixed[i]:
                continue

            # Walk backward to find nearest *originally* fixed vertex
            back_idx = None
            back_dist = 0
            for k in range(1, n):
                idx = (i - k) % n
                if cycle_is_fixed[idx]:
                    back_idx = idx
                    back_dist = k
                    break

            # Walk forward
            fwd_idx = None
            fwd_dist = 0
            for k in range(1, n):
                idx = (i + k) % n
                if cycle_is_fixed[idx]:
                    fwd_idx = idx
                    fwd_dist = k
                    break

            if back_idx is None or fwd_idx is None:
                continue

            # Linear interpolation weight (0 = back vertex, 1 = forward vertex)
            total = back_dist + fwd_dist
            weight = back_dist / total

            v = cycle[i]
            v_back = cycle[back_idx]
            v_fwd = cycle[fwd_idx]

            xi = x[v_back] * (1 - weight) + x[v_fwd] * weight
            yi = y[v_back] * (1 - weight) + y[v_fwd] * weight
            zi = z[v_back] * (1 - weight) + z[v_fwd] * weight

            # Project onto unit sphere
            r = np.sqrt(xi**2 + yi**2 + zi**2)
            if r > 1e-15:
                xi /= r
                yi /= r
                zi /= r

            new_values[v] = (xi, yi, zi)

        # Apply all fills at once
        for v, (xi, yi, zi) in new_values.items():
            x[v] = xi
            y[v] = yi
            z[v] = zi
            filled[v] = True

    return x, y, z, filled


# ======================================================================
#  Annular (cylindrical) patch handler
# ======================================================================

def _is_annular_patch(PM, pix, patm, verbose=True):
    """Detect whether *patm* is annular (has an inner boundary ring).

    A patch is annular if it has two or more disconnected boundary
    cycles and the smaller cycle(s) are collectively shared with
    already-parameterized neighbours.

    Detection strategy (multi-neighbour aware):
    1. If the patch has < 2 boundary cycles → not annular.
    2. (Fast path) If any single parameterized neighbour has > 50%
       of its border overlapping with this patch → annular.
    3. (Multi-neighbour path) For each boundary cycle, check if > 80%
       of its vertices are shared with the union of parameterized
       neighbours.  If the *shorter* cycle passes this test → annular
       (inner ring shared among multiple neighbours, e.g. 3 patches
       around a cap).
    """
    if not hasattr(patm, 'border_vertex') or patm.border_vertex is None:
        if verbose:
            print(f"  _is_annular_patch(Patch {pix}): no border_vertex → not annular")
        return False

    cycles = _build_boundary_cycles(patm)
    if verbose:
        cycle_lens = [len(c) for c in cycles]
        print(f"  _is_annular_patch(Patch {pix}): {len(cycles)} boundary cycle(s), "
              f"lengths={cycle_lens}")
    if len(cycles) < 2:
        return False

    my_border = set(np.where(patm.border_vertex.astype(bool))[0].tolist())
    if len(my_border) == 0:
        return False

    # Collect border sets for all parameterized neighbours (any index)
    PX = PM.get('PX', {})
    nb_borders = {}
    for npix in range(PM.get('npatches', 0)):
        if npix == pix:
            continue
        if npix not in PX:
            continue
        nb_patm = PM['P'][npix][0]
        if nb_patm.t is None:
            continue
        if not hasattr(nb_patm, 'border_vertex') or nb_patm.border_vertex is None:
            continue
        nb_border = set(np.where(nb_patm.border_vertex.astype(bool))[0].tolist())
        if len(nb_border) > 0:
            nb_borders[npix] = nb_border

    # Fast path: single-neighbour enclosure (> 50% of its border on ours)
    for npix, nb_border in nb_borders.items():
        overlap = my_border & nb_border
        frac = len(overlap) / len(nb_border)
        if frac > 0.5:
            if verbose:
                print(f"  _is_annular_patch(Patch {pix}): ANNULAR — "
                      f"enclosed neighbour Patch {npix} "
                      f"(overlap {len(overlap)}/{len(nb_border)} = {frac:.1%})")
            return True

    # Multi-neighbour path: check each boundary cycle for collective coverage
    all_nb_union = set()
    for nb_border in nb_borders.values():
        all_nb_union |= nb_border

    sorted_cycles = sorted(cycles, key=len)
    for cyc in sorted_cycles[:-1]:  # skip the longest (outer) cycle
        cyc_set = set(cyc)
        covered = cyc_set & all_nb_union
        frac = len(covered) / len(cyc_set) if len(cyc_set) > 0 else 0.0
        if frac > 0.8:
            if verbose:
                print(f"  _is_annular_patch(Patch {pix}): ANNULAR — "
                      f"inner cycle ({len(cyc_set)} verts) collectively "
                      f"covered by neighbours ({frac:.1%})")
            return True
        elif verbose:
            print(f"  _is_annular_patch(Patch {pix}): cycle ({len(cyc_set)} verts) "
                  f"neighbour coverage = {frac:.1%}")

    # Spatial-separation fallback: if not enough neighbours are parameterized
    # yet (processing-order problem), check if the two cycles are spatially
    # disjoint — genuinely separate boundary components.
    if len(cycles) >= 2 and hasattr(patm, 'X') and patm.X is not None:
        shortest = sorted_cycles[0]
        longest = sorted_cycles[-1]
        if len(shortest) >= 5 and len(longest) >= 5:
            short_pts = patm.X[np.array(shortest)]
            long_pts = patm.X[np.array(longest)]
            c_short = short_pts.mean(axis=0)
            c_long = long_pts.mean(axis=0)
            sep = np.linalg.norm(c_short - c_long)
            r_short = np.max(np.linalg.norm(short_pts - c_short, axis=1))
            r_long = np.max(np.linalg.norm(long_pts - c_long, axis=1))
            if sep > 0.5 * (r_short + r_long):
                if verbose:
                    print(f"  _is_annular_patch(Patch {pix}): ANNULAR — "
                          f"spatially separated cycles "
                          f"({len(shortest)} + {len(longest)} verts, "
                          f"centroid sep={sep:.4f}, "
                          f"radii={r_short:.4f}+{r_long:.4f})")
                return True
            elif verbose:
                print(f"  _is_annular_patch(Patch {pix}): cycles not well separated "
                      f"(sep={sep:.4f}, radii={r_short:.4f}+{r_long:.4f})")

    if verbose:
        print(f"  _is_annular_patch(Patch {pix}): NOT annular — "
              f"no inner cycle with sufficient neighbour coverage or spatial separation")
    return False


def _build_ring_cycle(patm, ring_verts):
    """Build an ordered boundary cycle from a set of border vertices.

    Walks the mesh link-structure restricted to *ring_verts* to produce
    a single ordered closed loop.  Each border vertex on a simple ring
    has exactly two neighbours that are also in the ring; the walk
    follows these edges.

    Parameters
    ----------
    patm : surface_mesh
        Patch mesh (must have ``L`` or ``edge_info()`` available).
    ring_verts : set of int
        Vertex indices that form the ring.

    Returns
    -------
    cycle : list of int
        Ordered vertex indices forming the ring.  Empty list if the
        ring cannot be traversed.
    """
    ring_set = set(int(v) for v in ring_verts)
    if len(ring_set) < 3:
        return []

    # Ensure link structure is available
    if not hasattr(patm, 'L') or patm.L is None or len(patm.L) == 0:
        patm.edge_info()
    L = patm.L if isinstance(patm.L, dict) else {}

    # Build adjacency restricted to ring vertices
    adj = {}
    for v in ring_set:
        adj[v] = [int(nb) for nb in L.get(v, []) if int(nb) in ring_set]

    start = next(iter(ring_set))
    cycle = [start]
    visited = {start}
    current = start

    for _ in range(len(ring_set) + 1):          # safety bound
        nxt = None
        for nb in adj.get(current, []):
            if nb not in visited:
                nxt = nb
                break
        if nxt is None:
            break
        cycle.append(nxt)
        visited.add(nxt)
        current = nxt

    # Check closure: last vertex should be adjacent to start
    if start not in adj.get(current, []):
        # Couldn't close the cycle — return what we have (best effort)
        pass

    return cycle


def _write_annular_diagnostic(PM, pix, patm, inner_verts, outer_border,
                              result, x_bnd, y_bnd, z_bnd, fixed_patm,
                              inner_npix, plot_flag):
    """Write a comprehensive diagnostic text file for the annular patch.

    Dumps simplified-mesh positions, fine-mesh boundary conditions,
    solver results (reduced mesh), face lists, and interior-vertex
    statistics so the boundary-value problem can be inspected in detail.
    """
    import os
    diag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'tests')
    os.makedirs(diag_dir, exist_ok=True)
    diag_path = os.path.join(diag_dir, 'annular_patch_diagnostic.txt')

    Xkeyind = PM['Xkeyind']
    all_keys = set(Xkeyind.astype(int).tolist())
    spm = PM.get('spm', None)
    my_border = inner_verts | outer_border

    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write(_make_run_header(PM.get('run_id')) + "\n")
        f.write("=" * 72 + "\n")
        f.write(f"  ANNULAR PATCH {pix} — DETAILED DIAGNOSTIC\n")
        f.write("=" * 72 + "\n\n")

        # ---- Section 1: simplified mesh (spm) --------------------------
        f.write("1. SIMPLIFIED MESH (spm) — KEY VERTICES ON PATCH BORDER\n")
        f.write("-" * 60 + "\n")
        if spm is not None:
            f.write(f"   spm total vertices : {len(spm.X)}\n")
            f.write(f"   spm total faces    : {len(spm.F)}\n\n")
            f.write(f"   {'simpl':>5}  {'fine':>5}  {'ring':>6}"
                    f"  {'x':>10} {'y':>10} {'z':>10}"
                    f"  {'theta':>8} {'phi':>8}\n")
            for si in range(len(Xkeyind)):
                mxi = int(Xkeyind[si])
                if mxi not in my_border:
                    continue
                ring = "inner" if mxi in inner_verts else "outer"
                f.write(f"   {si:5d}  {mxi:5d}  {ring:>6}"
                        f"  {spm.X[si, 0]:10.6f} {spm.X[si, 1]:10.6f}"
                        f" {spm.X[si, 2]:10.6f}"
                        f"  {float(spm.t[si]):8.5f} {float(spm.p[si]):8.5f}\n")
            f.write(f"\n   Simplified mesh faces (all):\n")
            for fi in range(len(spm.F)):
                f.write(f"     F[{fi:3d}]: {spm.F[fi].tolist()}\n")
        f.write("\n")

        # ---- Section 2: Enclosed neighbour(s) border values -----
        if inner_npix >= 0 and inner_npix < len(PM['P']):
            nb_patm = PM['P'][inner_npix][0]
        else:
            nb_patm = None
        f.write(f"2. ENCLOSED NEIGHBOUR — Patch {inner_npix} border values\n")
        f.write("-" * 60 + "\n")
        if nb_patm is None:
            f.write(f"   (No single enclosed neighbour — spatial separation used)\n\n")
        else:
            f.write(f"   (These are copied to the inner ring of Patch {pix})\n\n")
        f.write(f"   {'vert':>5}  {'x':>10} {'y':>10} {'z':>10}"
                f"  {'theta':>8} {'phi':>8}  key?\n")
        for v in sorted(inner_verts):
            v = int(v)
            is_key = "KEY" if v in all_keys else ""
            nb_t = float(nb_patm.t[v]) if nb_patm is not None and v < len(nb_patm.t) else 0.0
            nb_p = float(nb_patm.p[v]) if nb_patm is not None and v < len(nb_patm.p) else 0.0
            cx, cy, cz = kk_sph2cart(np.array([nb_t]),
                                      np.array([nb_p]),
                                      np.array([1.0]))
            f.write(f"   {v:5d}  {cx[0]:10.6f} {cy[0]:10.6f} {cz[0]:10.6f}"
                    f"  {nb_t:8.5f} {nb_p:8.5f}  {is_key}\n")
        f.write("\n")

        # ---- Section 3: fine-mesh boundary conditions -------------------
        f.write(f"3. FINE-MESH BOUNDARY CONDITIONS (after annular setup)\n")
        f.write("-" * 60 + "\n")
        n_inner = len(inner_verts)
        n_outer = len(outer_border)
        n_fixed = int(np.sum(fixed_patm))
        f.write(f"   Inner ring vertices : {n_inner}\n")
        f.write(f"   Outer ring vertices : {n_outer}\n")
        f.write(f"   Total border        : {n_inner + n_outer}\n")
        f.write(f"   Total fixed_patm    : {n_fixed}\n\n")

        f.write("   INNER RING boundary values (from x_bnd, y_bnd, z_bnd):\n")
        f.write(f"   {'vert':>5}  {'x':>10} {'y':>10} {'z':>10}"
                f"  {'theta':>8} {'phi':>8}  {'fixed':>5}  key?\n")
        for v in sorted(inner_verts):
            v = int(v)
            is_key = "KEY" if v in all_keys else ""
            t_v, p_v, _ = kk_cart2sph(np.array([x_bnd[v]]),
                                       np.array([y_bnd[v]]),
                                       np.array([z_bnd[v]]))
            f.write(f"   {v:5d}  {x_bnd[v]:10.6f} {y_bnd[v]:10.6f}"
                    f" {z_bnd[v]:10.6f}"
                    f"  {t_v[0]:8.5f} {p_v[0]:8.5f}"
                    f"  {str(bool(fixed_patm[v])):>5}  {is_key}\n")

        f.write("\n   OUTER RING boundary values (from x_bnd, y_bnd, z_bnd):\n")
        f.write(f"   {'vert':>5}  {'x':>10} {'y':>10} {'z':>10}"
                f"  {'theta':>8} {'phi':>8}  {'fixed':>5}  key?\n")
        for v in sorted(outer_border):
            v = int(v)
            is_key = "KEY" if v in all_keys else ""
            t_v, p_v, _ = kk_cart2sph(np.array([x_bnd[v]]),
                                       np.array([y_bnd[v]]),
                                       np.array([z_bnd[v]]))
            f.write(f"   {v:5d}  {x_bnd[v]:10.6f} {y_bnd[v]:10.6f}"
                    f" {z_bnd[v]:10.6f}"
                    f"  {t_v[0]:8.5f} {p_v[0]:8.5f}"
                    f"  {str(bool(fixed_patm[v])):>5}  {is_key}\n")
        # ---- Section 3b: RING GEOMETRY CHECK ON SPHERE --------------------
        f.write("   RING GEOMETRY CHECK (inner should be inside outer on sphere):\n")
        inner_pts = []
        outer_pts = []
        for v in inner_verts:
            v = int(v)
            pt = np.array([x_bnd[v], y_bnd[v], z_bnd[v]])
            rr = np.linalg.norm(pt)
            if rr > 1e-15:
                inner_pts.append(pt / rr)
        for v in outer_border:
            v = int(v)
            pt = np.array([x_bnd[v], y_bnd[v], z_bnd[v]])
            rr = np.linalg.norm(pt)
            if rr > 1e-15:
                outer_pts.append(pt / rr)
        if len(inner_pts) >= 2 and len(outer_pts) >= 2:
            inner_pts_arr = np.array(inner_pts)
            outer_pts_arr = np.array(outer_pts)
            inner_centroid = np.mean(inner_pts_arr, axis=0)
            outer_centroid = np.mean(outer_pts_arr, axis=0)
            n_ic = np.linalg.norm(inner_centroid)
            n_oc = np.linalg.norm(outer_centroid)
            if n_ic > 1e-15:
                inner_centroid /= n_ic
            if n_oc > 1e-15:
                outer_centroid /= n_oc
            centroid_sep = np.arccos(np.clip(
                np.dot(inner_centroid, outer_centroid), -1.0, 1.0))
            inner_max_r = max(np.arccos(np.clip(
                np.dot(inner_centroid, p), -1.0, 1.0))
                for p in inner_pts)
            outer_max_r = max(np.arccos(np.clip(
                np.dot(outer_centroid, p), -1.0, 1.0))
                for p in outer_pts)
            overlap = (centroid_sep + inner_max_r > outer_max_r)
            f.write(f"     Inner centroid->outer centroid sep: "
                    f"{np.degrees(centroid_sep):.1f} deg\n")
            f.write(f"     Inner ring angular radius: "
                    f"{np.degrees(inner_max_r):.1f}°\n")
            f.write(f"     Outer ring angular radius: "
                    f"{np.degrees(outer_max_r):.1f}°\n")
            if overlap:
                f.write("     *** RING OVERLAP DETECTED: inner ring "
                        "extends beyond outer ring on sphere ***\n")
            else:
                f.write("     OK: inner ring contained within outer ring\n")
            cross_count = 0
            for ip in inner_pts:
                d_to_outer_c = np.arccos(np.clip(
                    np.dot(outer_centroid, ip), -1.0, 1.0))
                if d_to_outer_c > outer_max_r:
                    cross_count += 1
            if cross_count > 0:
                f.write(f"     *** {cross_count}/{len(inner_pts)} inner "
                        f"vertices are outside the outer ring ***\n")

        f.write("\n")

        # ---- Section 4: fine-mesh faces (original indices) --------------
        f.write(f"4. FINE-MESH FACES  (patm.F — original vertex indices)\n")
        f.write("-" * 60 + "\n")
        f.write(f"   {len(patm.F)} faces\n\n")
        for fi in range(len(patm.F)):
            tri = patm.F[fi]
            roles = []
            for vi in tri:
                vi = int(vi)
                if vi in inner_verts:
                    roles.append("I")
                elif vi in outer_border:
                    roles.append("O")
                else:
                    roles.append(".")
            f.write(f"     F[{fi:3d}]: [{tri[0]:5d}, {tri[1]:5d}, {tri[2]:5d}]"
                    f"  roles=[{roles[0]},{roles[1]},{roles[2]}]\n")
        f.write("\n")

        # ---- Section 5: solver result (reduced mesh) --------------------
        if result is not None:
            minpatm, uv, x_sol, y_sol, z_sol, t_sol, p_sol, fixed_red = result
            nv_red = len(uv)
            n_fixed_red = int(np.sum(fixed_red))
            n_free_red = nv_red - n_fixed_red

            f.write(f"5. SOLVER RESULT  (reduced mesh)\n")
            f.write("-" * 60 + "\n")
            f.write(f"   Reduced vertices : {nv_red}\n")
            f.write(f"   Reduced faces    : {len(minpatm.F)}\n")
            f.write(f"   Fixed (boundary) : {n_fixed_red}\n")
            f.write(f"   Free  (interior) : {n_free_red}\n\n")

            uv_list = uv.tolist()

            # Classify each reduced vertex
            inner_red = []
            outer_red = []
            free_red = []
            for ri in range(nv_red):
                oi = int(uv[ri])
                if oi in inner_verts:
                    inner_red.append(ri)
                elif oi in outer_border:
                    outer_red.append(ri)
                elif not fixed_red[ri]:
                    free_red.append(ri)

            if inner_red:
                t_ir = t_sol[inner_red]
                f.write(f"   Inner ring (reduced)  theta: "
                        f"[{t_ir.min():.5f}, {t_ir.max():.5f}]\n")
            if outer_red:
                t_or = t_sol[outer_red]
                f.write(f"   Outer ring (reduced)  theta: "
                        f"[{t_or.min():.5f}, {t_or.max():.5f}]\n")
            if free_red:
                fr = np.array(free_red)
                t_fr = t_sol[fr]
                r_xy = np.sqrt(x_sol[fr]**2 + y_sol[fr]**2)
                t_bnd_all = t_sol[np.where(fixed_red)[0]]
                t_band_lo, t_band_hi = float(t_bnd_all.min()), float(t_bnd_all.max())
                n_out = int(np.sum((t_fr < t_band_lo - 1e-9) | (t_fr > t_band_hi + 1e-9)))
                f.write(f"   Theta band (boundary): [{t_band_lo:.5f}, {t_band_hi:.5f}]\n")
                f.write(f"   Interior   (solved)   theta: "
                        f"[{t_fr.min():.5f}, {t_fr.max():.5f}]"
                        f"{f'  *** {n_out} OUTSIDE BAND ***' if n_out > 0 else ''}\n")
                f.write(f"   Interior  x range  : "
                        f"[{x_sol[fr].min():.6f}, {x_sol[fr].max():.6f}]\n")
                f.write(f"   Interior  y range  : "
                        f"[{y_sol[fr].min():.6f}, {y_sol[fr].max():.6f}]\n")
                f.write(f"   Interior  z range  : "
                        f"[{z_sol[fr].min():.6f}, {z_sol[fr].max():.6f}]\n")
                f.write(f"   Interior  r_xy     : "
                        f"[{r_xy.min():.6f}, {r_xy.max():.6f}]\n")
                f.write(f"   Interior verts with r_xy < 0.1 : "
                        f"{int(np.sum(r_xy < 0.1))}\n")
                f.write(f"   Interior verts with r_xy < 0.3 : "
                        f"{int(np.sum(r_xy < 0.3))}\n\n")

            # Per-vertex table
            f.write("   ALL REDUCED VERTICES:\n")
            f.write(f"   {'red':>4} {'orig':>5} {'role':>8}"
                    f"  {'x_sol':>10} {'y_sol':>10} {'z_sol':>10}"
                    f"  {'theta':>8} {'phi':>8}  fixed?\n")
            for ri in range(nv_red):
                oi = int(uv[ri])
                if oi in inner_verts:
                    role = "inner"
                elif oi in outer_border:
                    role = "outer"
                elif fixed_red[ri]:
                    role = "fix?"
                else:
                    role = "FREE"
                f.write(f"   {ri:4d} {oi:5d} {role:>8}"
                        f"  {x_sol[ri]:10.6f} {y_sol[ri]:10.6f}"
                        f" {z_sol[ri]:10.6f}"
                        f"  {t_sol[ri]:8.5f} {p_sol[ri]:8.5f}"
                        f"  {str(bool(fixed_red[ri])):>5}\n")

            f.write(f"\n   REDUCED FACES:\n")
            for fi in range(len(minpatm.F)):
                tri = minpatm.F[fi]
                f.write(f"     F[{fi:3d}]: [{tri[0]:4d}, {tri[1]:4d},"
                        f" {tri[2]:4d}]\n")

        # ---- Section 6: neighbour connectivity --------------------------
        f.write(f"\n6. MESH LINK STRUCTURE (reduced) — "
                f"first 10 free vertices\n")
        f.write("-" * 60 + "\n")
        if result is not None:
            L = minpatm.L
            shown = 0
            for ri in free_red[:10]:
                oi = int(uv[ri])
                nbrs = L.get(ri, [])
                nbr_info = []
                for nb in nbrs:
                    noi = int(uv[nb])
                    if noi in inner_verts:
                        nbr_info.append(f"{nb}(I)")
                    elif noi in outer_border:
                        nbr_info.append(f"{nb}(O)")
                    else:
                        nbr_info.append(f"{nb}")
                f.write(f"   reduced[{ri}] (orig {oi}): "
                        f"deg={len(nbrs)}, nbrs={nbr_info}\n")
                shown += 1

    if plot_flag > 0:
        print(f'  Annular diagnostic written to: {diag_path}')


def _parameterize_annular_patch(PM, pix, patm, plot_flag):
    """Dedicated handler for annular (cylindrical) patches.

    Annular patches have TWO boundary components:

    * **Outer ring** – shared with several neighbours (one edge each).
    * **Inner ring** – a closed loop fully enclosing one neighbour.

    The standard ``border2chain`` / ``PM.Edges`` workflow fails here
    because ``border2chain`` treats both rings as a single circuit
    (zigzagging between them) and ``PM.Edges`` typically carries only
    one incomplete chain for the inner ring.

    Strategy
    --------
    1. **Inner ring** – copy every position from the already-
       parameterized enclosed neighbour.  This is exact by
       construction.
    2. **Outer ring** – interpolate from ``PM.Edges`` chains
       (excluding inner chains) via the normal ``assign_boundary_values``
       path.
    3. **Solve Laplace** with both rings fixed as boundary conditions.

    Returns the same tuple as ``parameterize_single_patch_with_cartesian``
    or ``None`` if the patch cannot be handled as annular.
    """
    # ------------------------------------------------------------------
    # 0. Identify the inner ring and enclosed neighbours
    #
    #    Two strategies:
    #    (a) Single-neighbour: one neighbour has > 50% of its border on ours.
    #    (b) Multi-neighbour:  no single neighbour passes (a) but one of the
    #        boundary cycles is collectively covered (> 80%) by the union of
    #        parameterized neighbours.  This handles caps bordered by 3+
    #        patches where this patch only shares a partial arc with each.
    # ------------------------------------------------------------------
    if not hasattr(patm, 'border_vertex') or patm.border_vertex is None:
        return None

    border_mask = patm.border_vertex.astype(bool)
    my_border = set(np.where(border_mask)[0].tolist())
    if len(my_border) == 0:
        return None

    Xkeyind = PM['Xkeyind']
    all_keys = set(Xkeyind.astype(int).tolist())

    # Collect border sets for all parameterized neighbours (any index)
    PX = PM.get('PX', {})
    nb_borders = {}
    for npix in range(PM.get('npatches', 0)):
        if npix == pix:
            continue
        if npix not in PX:
            continue
        nb_patm = PM['P'][npix][0]
        if nb_patm.t is None:
            continue
        if not hasattr(nb_patm, 'border_vertex') or nb_patm.border_vertex is None:
            continue
        nb_border = set(np.where(nb_patm.border_vertex.astype(bool))[0].tolist())
        if len(nb_border) > 0:
            nb_borders[npix] = nb_border

    # Strategy (a): single-neighbour enclosure
    inner_npix = None
    inner_verts = set()
    for npix, nb_border in nb_borders.items():
        overlap = my_border & nb_border
        if len(nb_border) > 0 and len(overlap) / len(nb_border) > 0.5:
            inner_npix = npix
            inner_verts = overlap
            break

    # Strategy (b): multi-neighbour inner ring via boundary cycles
    cycles = _build_boundary_cycles(patm)
    full_inner_ring = None

    if inner_npix is not None:
        # Single-neighbour path: find the full cycle containing the overlap
        for cyc in cycles:
            cyc_set = set(cyc)
            if inner_verts <= cyc_set:
                full_inner_ring = cyc_set
                break
        if full_inner_ring is None:
            full_inner_ring = inner_verts
    elif len(cycles) >= 2:
        # Multi-neighbour path: identify the inner ring from boundary cycles
        all_nb_union = set()
        for nb_border in nb_borders.values():
            all_nb_union |= nb_border

        sorted_cycles = sorted(cycles, key=len)
        for cyc in sorted_cycles[:-1]:
            cyc_set = set(cyc)
            covered = cyc_set & all_nb_union
            frac = len(covered) / len(cyc_set) if len(cyc_set) > 0 else 0.0
            if frac > 0.8:
                full_inner_ring = cyc_set
                if plot_flag > 0:
                    print(f'  Patch {pix}: inner ring identified via multi-neighbour '
                          f'coverage ({len(cyc_set)} verts, {frac:.1%} covered)')
                break

        # Spatial-separation fallback: use shorter cycle as inner ring
        if full_inner_ring is None and hasattr(patm, 'X') and patm.X is not None:
            shortest = sorted_cycles[0]
            longest = sorted_cycles[-1]
            if len(shortest) >= 5 and len(longest) >= 5:
                short_pts = patm.X[np.array(shortest)]
                long_pts = patm.X[np.array(longest)]
                c_short = short_pts.mean(axis=0)
                c_long = long_pts.mean(axis=0)
                sep = np.linalg.norm(c_short - c_long)
                r_short = np.max(np.linalg.norm(short_pts - c_short, axis=1))
                r_long = np.max(np.linalg.norm(long_pts - c_long, axis=1))
                if sep > 0.5 * (r_short + r_long):
                    full_inner_ring = set(shortest)
                    if plot_flag > 0:
                        print(f'  Patch {pix}: inner ring identified via spatial '
                              f'separation ({len(shortest)} verts, '
                              f'sep={sep:.4f}, radii={r_short:.4f}+{r_long:.4f})')

    if full_inner_ring is None:
        if plot_flag > 0:
            print(f'Patch {pix}: Annular handler – no inner ring found '
                  f'({len(cycles)} cycles, {len(nb_borders)} parameterized neighbours)')
        return None

    # Collect ALL parameterized neighbours that share the inner ring
    inner_neighbours = []
    for npix, nb_border in nb_borders.items():
        overlap = full_inner_ring & nb_border
        if len(overlap) > 0:
            inner_neighbours.append((npix, overlap))

    inner_keys = full_inner_ring & all_keys
    outer_border = my_border - full_inner_ring

    if plot_flag > 0:
        nbr_str = ', '.join(str(n) for n, _ in inner_neighbours)
        print(f'Patch {pix}: ANNULAR – inner ring: {len(full_inner_ring)} verts '
              f'({len(inner_keys)} keys), shared with Patch(es) {nbr_str}; '
              f'outer ring: {len(outer_border)} verts')

    # ------------------------------------------------------------------
    # 1. Build outer-only border_info from PM.Edges (skip inner chains)
    # ------------------------------------------------------------------
    edge_chains = []
    sentinels_list = []
    spm_indices_list = []
    skipped = 0

    for eix in range(len(PM['Edges'])):
        if PM['Edges'][eix, 0] != pix and PM['Edges'][eix, 1] != pix:
            continue
        if eix >= len(PM['edge_dat']) or len(PM['edge_dat'][eix]) == 0:
            continue

        pms1 = int(PM['sentinels'][eix, 0])
        pms2 = int(PM['sentinels'][eix, 1])

        # Skip chains that touch the inner ring (ANY sentinel is inner).
        # Cross-boundary chains (one inner, one outer sentinel) corrupt
        # outer ring interpolation — those outer vertices are handled
        # by the outer ring re-interpolation step instead.
        if pms1 in inner_keys or pms2 in inner_keys:
            skipped += 1
            if plot_flag > 0:
                inner_which = []
                if pms1 in inner_keys:
                    inner_which.append(str(pms1))
                if pms2 in inner_keys:
                    inner_which.append(str(pms2))
                print(f'  Skipping inner/cross chain eix={eix}: '
                      f'sentinels=({pms1},{pms2}), '
                      f'inner key(s)={",".join(inner_which)}')
            continue

        ol = PM['edge_dat'][eix]
        if len(ol) == 0:
            continue

        spms1 = np.where(Xkeyind == pms1)[0]
        spms2 = np.where(Xkeyind == pms2)[0]
        if len(spms1) == 0 or len(spms2) == 0:
            continue

        edge_chains.append(ol)
        sentinels_list.append([pms1, pms2])
        spm_indices_list.append([spms1[0], spms2[0]])

    if plot_flag > 0:
        print(f'  Outer chains: {len(edge_chains)} (skipped {skipped} inner)')

    if len(edge_chains) == 0:
        if plot_flag > 0:
            print(f'Patch {pix}: Annular handler – no outer chains')
        return None

    border_info = {
        'edge_chains': edge_chains,
        'sentinels': sentinels_list,
        'spm_indices': spm_indices_list,
        'use_patch_edges': False,
    }

    # ------------------------------------------------------------------
    # 2. Assign outer boundary values via the standard path
    # ------------------------------------------------------------------
    boundary_result = assign_boundary_values(
        PM, pix, patm, border_info, plot_flag)
    if boundary_result is None:
        return None

    x, y, z, _t, _p, fixed_patm, chain_assigned = boundary_result

    # ------------------------------------------------------------------
    # 3. Copy inner ring from ALL enclosed neighbours
    #
    #    When the inner ring touches multiple neighbours (e.g. two caps
    #    on one side), we must copy from each neighbour for its segment.
    #    Copying only from the first enclosed neighbour leaves the other
    #    segment with wrong/interpolated values, causing scrambled boundaries.
    #    This MUST happen AFTER assign_boundary_values because that
    #    function zeroes patm.t / patm.p and re-initialises x,y,z.
    # ------------------------------------------------------------------
    n_copied = 0
    for npix, overlap_verts in inner_neighbours:
        nb_patm = PM['P'][npix][0]
        for v in overlap_verts:
            v = int(v)
            if v < len(nb_patm.t) \
                    and hasattr(nb_patm, 'border_vertex') \
                    and nb_patm.border_vertex is not None \
                    and v < len(nb_patm.border_vertex) \
                    and int(nb_patm.border_vertex[v]) > 0:
                v_t = float(nb_patm.t[v])
                v_p = float(nb_patm.p[v])
                ux, uy, uz = kk_sph2cart(
                    np.array([v_t]), np.array([v_p]), np.array([1.0]))
                x[v] = ux[0]
                y[v] = uy[0]
                z[v] = uz[0]
                patm.t[v] = v_t
                patm.p[v] = v_p
                fixed_patm[v] = True
                n_copied += 1

    if plot_flag > 0:
        print(f'  Inner ring: copied {n_copied}/{len(full_inner_ring)} vertices '
              f'from {len(inner_neighbours)} neighbour(s)')

    # Fallback: any inner-ring vertex NOT copied (e.g. the neighbour
    # did not flag it as border) gets nearest-neighbour from copied set.
    copied_inner = [int(v) for v in full_inner_ring if fixed_patm[int(v)]]
    not_copied = [int(v) for v in full_inner_ring if not fixed_patm[int(v)]]
    if len(not_copied) > 0 and len(copied_inner) > 0:
        pts_cp = np.array([[x[c], y[c], z[c]] for c in copied_inner])
        for v in not_copied:
            pt = patm.X[v] if v < len(patm.X) else np.zeros(3)
            dists = np.linalg.norm(pts_cp - pt, axis=1)
            nn = copied_inner[np.argmin(dists)]
            x[v], y[v], z[v] = x[nn], y[nn], z[nn]
            patm.t[v], patm.p[v] = patm.t[nn], patm.p[nn]
            fixed_patm[v] = True
        if plot_flag > 0:
            print(f'  Inner ring fallback: nearest-neighbour for '
                  f'{len(not_copied)} vertices')

    # ------------------------------------------------------------------
    # 4. Pin key vertices to simplified-mesh positions
    #
    #    For annular patches: pin ONLY outer-ring keys.  Inner-ring keys
    #    were copied from enclosed neighbours for C0 continuity; overwriting
    #    with spm can create inconsistencies when the inner ring touches
    #    multiple neighbours (e.g. two caps).  Outer-ring keys are pinned
    #    to spm for global consistency.
    # ------------------------------------------------------------------
    spm = PM.get('spm', None)
    if spm is not None:
        n_pinned = 0
        for si in range(len(Xkeyind)):
            mX_idx = int(Xkeyind[si])
            if mX_idx not in my_border:
                continue
            if mX_idx >= len(x):
                continue
            # Skip inner ring keys — keep neighbour-copied values for continuity
            if mX_idx in full_inner_ring:
                continue
            x[mX_idx] = spm.X[si, 0]
            y[mX_idx] = spm.X[si, 1]
            z[mX_idx] = spm.X[si, 2]
            patm.t[mX_idx] = float(spm.t[si])
            patm.p[mX_idx] = float(spm.p[si])
            fixed_patm[mX_idx] = True
            n_pinned += 1
        if plot_flag > 0:
            print(f'  Pinned {n_pinned} outer-ring key vertices (inner kept from neighbours)')

    # Ensure ALL border vertices are marked fixed
    fixed_patm = fixed_patm | border_mask

    # ------------------------------------------------------------------
    # 5. Manual outer ring re-interpolation
    #
    #    Use the boundary cycles from _build_boundary_cycles (which walk
    #    actual boundary edges and always form complete cycles).  Pick
    #    the cycle that contains the most outer border vertices — this
    #    is the outer ring.  Then interpolate non-key outer vertices
    #    between adjacent outer-key anchors along this cycle.
    #
    #    _build_ring_cycle (restricted adjacency) fails when outer
    #    border vertices are only connected via inner ring vertices.
    # ------------------------------------------------------------------
    outer_keys_set = outer_border & all_keys

    # Find the boundary cycle that best represents the outer ring
    all_cycles = _build_boundary_cycles(patm)
    outer_cycle = []
    best_outer_count = -1
    for cyc in all_cycles:
        cyc_set = set(cyc)
        n_outer = len(cyc_set & outer_border)
        if n_outer > best_outer_count:
            best_outer_count = n_outer
            outer_cycle = cyc

    if len(outer_cycle) >= 3 and len(outer_keys_set) >= 2:
        # Find anchor positions: outer-key vertices on this cycle.
        # Some cycle vertices may be inner-ring vertices (at crossing
        # points); they are skipped as anchors.
        anchor_positions = [i for i in range(len(outer_cycle))
                            if outer_cycle[i] in outer_keys_set]
        n_reinterp_outer = 0

        for seg_k in range(len(anchor_positions)):
            start_pos = anchor_positions[seg_k]
            end_pos = anchor_positions[(seg_k + 1) % len(anchor_positions)]
            v_start = outer_cycle[start_pos]
            v_end = outer_cycle[end_pos]

            # Collect segment vertices between the two anchors
            # (only outer non-key vertices; skip inner vertices in the cycle)
            seg_verts = []
            pos = start_pos
            while True:
                pos = (pos + 1) % len(outer_cycle)
                if pos == end_pos:
                    break
                v = outer_cycle[pos]
                if v in outer_border and v not in outer_keys_set:
                    seg_verts.append(v)

            if len(seg_verts) == 0:
                continue

            p_start = np.array([x[v_start], y[v_start], z[v_start]])
            p_end = np.array([x[v_end], y[v_end], z[v_end]])
            n_start = np.linalg.norm(p_start)
            n_end = np.linalg.norm(p_end)
            if n_start < 1e-15 or n_end < 1e-15:
                continue
            p_start = p_start / n_start
            p_end = p_end / n_end

            cos_om = np.clip(np.dot(p_start, p_end), -1.0, 1.0)
            om = np.arccos(cos_om)
            sin_om = np.sin(om)
            do_slerp = (sin_om > 1e-8)

            total = len(seg_verts) + 1
            for idx_s, sv in enumerate(seg_verts):
                alpha = (idx_s + 1) / total
                if do_slerp:
                    c1 = np.sin((1.0 - alpha) * om) / sin_om
                    c2 = np.sin(alpha * om) / sin_om
                    interp = c1 * p_start + c2 * p_end
                else:
                    interp = (1.0 - alpha) * p_start + alpha * p_end
                r_i = np.linalg.norm(interp)
                if r_i > 1e-15:
                    interp /= r_i
                x[sv] = interp[0]
                y[sv] = interp[1]
                z[sv] = interp[2]
                t_tmp, p_tmp, _ = kk_cart2sph(
                    np.array([interp[0]]),
                    np.array([interp[1]]),
                    np.array([interp[2]]))
                patm.t[sv] = t_tmp[0]
                patm.p[sv] = p_tmp[0]
                n_reinterp_outer += 1

        if plot_flag > 0:
            print(f'  Outer ring: re-interpolated {n_reinterp_outer} '
                  f'vertices between {len(outer_keys_set)} key anchors '
                  f'(cycle length {len(outer_cycle)}, '
                  f'outer_in_cycle={best_outer_count})')
    elif plot_flag > 0:
        print(f'  Warning: Could not build outer ring cycle '
              f'(outer_border={len(outer_border)}, '
              f'outer_keys={len(outer_keys_set)}, '
              f'best_cycle={len(outer_cycle)})')

    # ------------------------------------------------------------------
    # 6. Diagnostic: boundary value ranges
    # ------------------------------------------------------------------
    if plot_flag > 0:
        inner_t = [patm.t[int(v)] for v in full_inner_ring]
        outer_t = [patm.t[int(v)] for v in outer_border]
        if inner_t and outer_t:
            print(f'  Boundary theta ranges:  '
                  f'inner=[{min(inner_t):.3f}, {max(inner_t):.3f}]  '
                  f'outer=[{min(outer_t):.3f}, {max(outer_t):.3f}]')

    # ------------------------------------------------------------------
    # 7. Solve using stereographic-projection Laplace.
    #
    #    This solves for BOTH theta and phi jointly in a single 2-D
    #    system (stereographic projection to plane, 2-D Laplace, then
    #    inverse projection back to sphere).
    #
    #    Pole choice: the point on the sphere that maximises the minimum
    #    angular distance to ANY boundary vertex (inner or outer).  For
    #    full-azimuth annular bands (theta_inner ≈ 0.8, theta_outer ≈ 2.1)
    #    this is typically one of the geographic poles.
    #
    #    The previous heuristic (antipode of inner-ring centroid) placed
    #    the pole only ~5° from the outer ring, causing outer-ring
    #    vertices to project to extreme 2-D radii, distorting the Laplace
    #    solve and letting interior vertices leak to the opposite pole.
    #
    #    Alternative solvers tested:
    #    * theta+Tutte-phi (solve_spherical=True): guaranteed theta
    #      within band, but decoupled theta/phi gives ~50% foldovers.
    #    * Cartesian (force_cartesian): pole collapse for full-azimuth
    #      annular bands.
    # ------------------------------------------------------------------
    inner_pts = np.array([[x[int(v)], y[int(v)], z[int(v)]]
                          for v in full_inner_ring])
    outer_pts_arr = np.array([[x[int(v)], y[int(v)], z[int(v)]]
                              for v in outer_border])

    # Normalise boundary points to unit sphere
    all_bnd_pts = np.vstack([inner_pts, outer_pts_arr])
    bnd_norms = np.linalg.norm(all_bnd_pts, axis=1, keepdims=True)
    bnd_norms = np.where(bnd_norms < 1e-15, 1.0, bnd_norms)
    all_bnd_unit = all_bnd_pts / bnd_norms

    # Candidate poles: 6 axis directions + ring centroids and their antipodes
    inner_centroid = inner_pts.mean(axis=0)
    ic_norm = np.linalg.norm(inner_centroid)
    outer_centroid = outer_pts_arr.mean(axis=0)
    oc_norm = np.linalg.norm(outer_centroid)

    pole_candidates = [
        np.array([0.0, 0.0,  1.0]),   # north pole
        np.array([0.0, 0.0, -1.0]),   # south pole
        np.array([1.0, 0.0,  0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0,  1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ]
    if ic_norm > 1e-12:
        ic_hat = inner_centroid / ic_norm
        pole_candidates.extend([ic_hat, -ic_hat])
    if oc_norm > 1e-12:
        oc_hat = outer_centroid / oc_norm
        pole_candidates.extend([oc_hat, -oc_hat])

    best_pole = np.array([0.0, 0.0, -1.0])
    best_min_angle = 0.0
    for cand in pole_candidates:
        cos_angles = all_bnd_unit @ cand
        max_cos = float(cos_angles.max())          # nearest boundary vertex
        min_angle = float(np.arccos(np.clip(max_cos, -1.0, 1.0)))
        if min_angle > best_min_angle:
            best_min_angle = min_angle
            best_pole = cand.copy()

    pole_hint = best_pole
    if plot_flag > 0:
        print(f'  Stereo pole: {best_pole.round(4)}, '
              f'min angular distance to boundary: '
              f'{np.degrees(best_min_angle):.1f} deg')

    result = parameterize_single_patch_with_cartesian(
        patm, x, y, z, fixed_patm, plot_flag,
        pole_hint=pole_hint,
        inner_border_verts=full_inner_ring)

    # ------------------------------------------------------------------
    # 8. Write detailed diagnostic file for inspection
    # ------------------------------------------------------------------
    diag_inner_npix = inner_neighbours[0][0] if inner_neighbours else -1
    _write_annular_diagnostic(PM, pix, patm, full_inner_ring, outer_border,
                              result, x, y, z, fixed_patm,
                              diag_inner_npix, plot_flag)

    return result


def _store_annular_result(PM, pix, patm, annular_result, plot_flag):
    """Store the result of a successful annular parameterization."""
    minpatm, uv, x_sol, y_sol, z_sol, t_sol, p_sol, \
        fixed_patm_reduced = annular_result
    patm.t[uv] = minpatm.t
    patm.p[uv] = minpatm.p
    PM['P'][pix][0] = patm
    PM['PX'][pix] = np.column_stack([
        fixed_patm_reduced,
        x_sol, y_sol, z_sol,
        t_sol, p_sol
    ])
    try:
        X_chk = np.column_stack([x_sol, y_sol, z_sol])
        n_fold_chk = 0
        for fi in range(len(minpatm.F)):
            i0, i1, i2 = minpatm.F[fi]
            va, vb, vc = X_chk[i0], X_chk[i1], X_chk[i2]
            orient_chk = np.dot(
                (va + vb + vc) / 3.0,
                np.cross(vb - va, vc - va))
            if orient_chk < -1e-15:
                n_fold_chk += 1
        if plot_flag > 0 or n_fold_chk > 0:
            status = ('OK' if n_fold_chk == 0
                      else f'** {n_fold_chk} FOLDOVERS **')
            print(f'Patch {pix}: {len(minpatm.F)} faces, {status} (annular)')
    except Exception:
        pass


def parameterize_patches_cart(PM, plot_flag=0):
    """
    Perform spherical parameterization mapping per patch for all patches.
    Based on post-optimization spherical mapping of simplified patch mesh.
    
    This is a faithful translation of the MATLAB function parameterize_patches_cart.m.
    
    PM Structure Documentation:
    ===========================
    The PM (Patch Mesh) structure is a dictionary containing:
    
    Required fields:
    ----------------
    PM['pm'] : surface_mesh
        Simplified patch-level mesh (coarse mesh)
        - PM['pm'].t : theta values (post-optimization spherical coordinates)
        - PM['pm'].p : phi values (post-optimization spherical coordinates)
        - PM['pm'].X : vertex coordinates
        - PM['pm'].F : face indices
    
    PM['P'] : list
        List of patches, each patch is a list:
        - PM['P'][pix][0] : surface_mesh object (full-resolution patch mesh)
          - Vertex indices correspond to original mesh m.X
          - Contains unused vertices (not all vertices are used)
    
    PM['npatches'] : int
        Number of patches
    
    PM['Edges'] : array, shape (n_edges, 2)
        Patch-level edges (pairs of patch indices, not vertex indices)
        - PM['Edges'][eix, 0] and PM['Edges'][eix, 1] are patch indices
    
    PM['edge_dat'] : list
        List of vertex chains for each edge
        - PM['edge_dat'][eix] : array of vertex indices (into m.X) forming the edge chain
        - References indices into original mesh m.X
    
    PM['sentinels'] : array, shape (n_edges, 2)
        Start/end key vertices for each edge
        - PM['sentinels'][eix, 0] : first sentinel vertex index (into m.X)
        - PM['sentinels'][eix, 1] : second sentinel vertex index (into m.X)
        - References indices into original mesh m.X
    
    PM['Xkeyind'] : array
        Indices into original mesh m.X corresponding to:
        - PM['Keyind'] : key vertices in simplified mesh
        - PM['CVind'] : center vertices in simplified mesh
        - Used to map between simplified mesh (PM['pm']) and original mesh (m)
        - PM['Xkeyind'][i] gives the vertex index in m.X for simplified mesh vertex i
    
    PM['Keyind'] : array
        Indices into PM['pm'].X for key vertices
    
    PM['CVind'] : array
        Indices into PM['pm'].X for center vertices
    
    Optional fields (for refined edge chains):
    -----------------------------------------
    PM['patch'] : dict
        More accurate set of edge chains (used in later iterations)
        - PM['patch'][pix]['edge_dat'] : list of edge chains for patch pix
        - PM['patch'][pix]['key_dat'] : array of key vertex pairs for each edge chain
    
    Generated fields (output):
    -------------------------
    PM['spm'] : surface_mesh
        Spherical parameterized mesh (created from PM['pm'] if not present)
        - PM['spm'].X : Cartesian coordinates on unit sphere
        - PM['spm'].t : theta values
        - PM['spm'].p : phi values
    
    PM['PX'] : dict
        Parameterization results for each patch
        - PM['PX'][pix] : array with columns [fixed_mask, x_sol, y_sol, z_sol, t_sol, p_sol]
    
    Parameters:
    -----------
    PM : dict
        Patch mesh structure from patch_info_gen
        Must contain: 'pm', 'P', 'npatches', 'Edges', 'edge_dat', 'sentinels', 'Xkeyind'
    plot_flag : int
        Plotting flag (0 = no plots, 1 = basic plots, 2+ = detailed plots)
        
    Returns:
    --------
    PM : dict
        Updated patch mesh structure with parameterized patches
        Adds/updates: 'spm', 'PX', and updates PM['P'][pix][0] with parameterization
    """
    run_id = PM.get('run_id') or uuid.uuid4().hex[:8]
    PM['run_id'] = run_id
    print(f"parameterize_patches_cart: run_id={run_id}")

    # Create spherical parameterized mesh from pm if not present
    if 'spm' not in PM:
        t = np.asarray(PM['pm'].t, dtype=float)
        p = np.asarray(PM['pm'].p, dtype=float)
        n_pm = len(PM['pm'].X)

        # ------------------------------------------------------------------
        # Guard: PM['pm'].t / .p may have the wrong length.
        # A common notebook pattern is  PM['pm'].t = ms.t.copy()  which
        # copies the *full* mesh array (len = n_full) into the simplified
        # mesh object (should have len = n_pm).  When len(t) > n_pm the
        # indices are misaligned — simplified vertex i maps to full-mesh
        # vertex Xkeyind[i], NOT vertex i.  Fix by remapping.
        # ------------------------------------------------------------------
        if len(t) != n_pm:
            Xkeyind = PM['Xkeyind']
            t_remap = np.zeros(n_pm)
            p_remap = np.zeros(n_pm)
            for i in range(n_pm):
                mX_idx = int(Xkeyind[i]) if i < len(Xkeyind) else 0
                if mX_idx < len(t):
                    t_remap[i] = t[mX_idx]
                    p_remap[i] = p[mX_idx]
            t = t_remap
            p = p_remap
            PM['pm'].t = t
            PM['pm'].p = p
            if plot_flag > 0:
                print(f'parameterize_patches_cart: Remapped PM[pm].t/p via Xkeyind '
                      f'(full-mesh length -> {n_pm} simplified vertices)')

        u, v, w = kk_sph2cart(t, p, np.ones_like(t))
        
        from ..surface_mesh import surface_mesh
        spm = surface_mesh(np.column_stack([u, v, w]), PM['pm'].F.copy())
        spm.face_labels = PM['pm'].face_labels.copy() if PM['pm'].face_labels is not None else None
        spm.t = np.asarray(t).copy()
        spm.p = np.asarray(p).copy()
        # Ensure outward-facing normals on sphere (map2sphere can introduce inward faces)
        try:
            from .fix_flipped_faces import fix_flipped_faces
            spm, _ = fix_flipped_faces(spm, verbose=False)
        except Exception:
            pass
        PM['spm'] = spm
    
    PM['PX'] = {}
    deferred_annular = []

    # ------------------------------------------------------------------
    # Pre-flight: per-patch angular span check on the sphere.
    # Warn about patches whose keys are spread over too large an arc —
    # these will produce boundary self-intersections and foldovers that
    # no fine-mesh solver can fix.
    # ------------------------------------------------------------------
    Xkeyind_arr = np.asarray(PM['Xkeyind']).ravel().astype(int)
    nkeys = int(PM.get('nkeys', len(Xkeyind_arr)))
    spm = PM['spm']
    overspread_patches = []
    for _pix in range(PM['npatches']):
        if len(PM.get('keys', [])) == 0:
            break
        pk_mX = PM['keys'][PM['keys'][:, 0] == _pix, 1].astype(int)
        pk_simpl = [ki for ki in range(min(nkeys, len(Xkeyind_arr)))
                    if int(Xkeyind_arr[ki]) in pk_mX]
        if len(pk_simpl) < 2:
            continue
        max_arc = 0.0
        for i in range(len(pk_simpl)):
            pi = spm.X[pk_simpl[i]]
            ni = np.linalg.norm(pi)
            if ni < 1e-15:
                continue
            pi = pi / ni
            for j in range(i + 1, len(pk_simpl)):
                pj = spm.X[pk_simpl[j]]
                nj = np.linalg.norm(pj)
                if nj < 1e-15:
                    continue
                pj = pj / nj
                arc = np.arccos(np.clip(np.dot(pi, pj), -1.0, 1.0))
                if arc > max_arc:
                    max_arc = arc
        if max_arc > np.pi * 0.5:
            overspread_patches.append((_pix, np.degrees(max_arc)))
    if overspread_patches:
        print(f'\n*** SPHERE MAPPING QUALITY WARNING ***')
        print(f'  {len(overspread_patches)} patch(es) have keys spread '
              f'over >90° on the sphere:')
        for _pix, deg in overspread_patches:
            severity = 'SEVERE' if deg > 135 else 'moderate'
            print(f'    Patch {_pix}: max key arc = {deg:.1f}° ({severity})')
        print(f'  These patches will likely have boundary self-intersections.')
        print(f'  Remedies: (1) re-run with avoid_topological_annuli=True,')
        print(f'            (2) increase optimization iterations for map2sphere,')
        print(f'            (3) try a different segmentation seed.\n')

    # Iterate over all patches to parameterize each one
    for pix in range(PM['npatches']):
        patm = PM['P'][pix][0]  # Non-simplified patch

        # [0] Annular / cylindrical patch detection
        if _is_annular_patch(PM, pix, patm):
            annular_result = _parameterize_annular_patch(
                PM, pix, patm, plot_flag)
            if annular_result is not None:
                _store_annular_result(PM, pix, patm, annular_result, plot_flag)
                continue
            # Defer: inner-ring neighbours may not be parameterized yet
            deferred_annular.append(pix)
            if plot_flag > 0:
                print(f'Patch {pix}: DEFERRED — annular handler needs more '
                      f'parameterized neighbours (will retry in pass 2)')
            continue

        # [1] Validate and prepare border chains for this patch
        border_info = validate_and_prepare_border_chains(PM, pix, plot_flag)
        
        if border_info is None:
            if plot_flag > 0:
                print(f'Patch {pix}: Skipping - no valid border chains found')
            continue
        
        # [2] Assign boundary values from simplified mesh to patch border
        boundary_result = assign_boundary_values(PM, pix, patm, border_info, plot_flag)
        
        if boundary_result is None:
            if plot_flag > 0:
                print(f'Patch {pix}: Skipping - failed to assign boundary values')
            continue
        
        x, y, z, t, p, fixed_patm, chain_assigned = boundary_result

        # [2b] Multi-boundary patches: copy inner boundary values from
        #      already-parameterized neighbor to guarantee perfect
        #      consistency at shared boundaries.
        #
        #      IMPORTANT: Do NOT overwrite vertices that were already
        #      assigned by an edge chain.  When _build_boundary_cycles
        #      splits a single boundary into two "components" (false
        #      positive), the copy would overwrite chain-interpolated
        #      positions with a neighbour's (possibly different) values,
        #      introducing discontinuities and foldovers.
        cycles = _build_boundary_cycles(patm)
        if len(cycles) > 1:
            for cycle in cycles:
                cycle_set = set(cycle)
                # Skip cycles that are mostly covered by chains —
                # they are part of the main boundary, not inner rings.
                n_chain_in_cycle = len(cycle_set & chain_assigned)
                if n_chain_in_cycle > len(cycle) * 0.5:
                    if plot_flag > 0:
                        print(f'Patch {pix}: Skipping cycle ({len(cycle)} verts, '
                              f'{n_chain_in_cycle} chain-assigned) -- not a true inner ring')
                    continue

                for npix in range(pix):  # only already-processed patches
                    nb_patm = PM['P'][npix][0]
                    if nb_patm.t is None or not hasattr(nb_patm, 'border_vertex') \
                            or nb_patm.border_vertex is None:
                        continue
                    nb_border = set(
                        np.where(nb_patm.border_vertex.astype(bool))[0].tolist()
                    )
                    overlap = cycle_set.intersection(nb_border)
                    if len(overlap) < len(cycle) * 0.5:
                        continue
                    # This boundary cycle belongs to the neighbor patch
                    n_copied = 0
                    for v in cycle:
                        # Skip vertices already set by an edge chain
                        if v in chain_assigned:
                            continue
                        if v < len(nb_patm.t) and int(nb_patm.border_vertex[v]) > 0:
                            v_t = nb_patm.t[v]
                            v_p = nb_patm.p[v]
                            uv_, vv_, wv_ = kk_sph2cart(
                                np.array([v_t]), np.array([v_p]),
                                np.array([1.0])
                            )
                            x[v] = uv_[0]
                            y[v] = vv_[0]
                            z[v] = wv_[0]
                            patm.t[v] = v_t
                            patm.p[v] = v_p
                            fixed_patm[v] = True
                            n_copied += 1
                    if n_copied > 0 and plot_flag > 0:
                        print(f'Patch {pix}: Copied {n_copied} inner boundary '
                              f'vertices from already-parameterized Patch {npix}')
                    break  # found the neighbor for this cycle
        
        # [3] Parameterize the patch using the boundary conditions
        result = parameterize_single_patch_with_cartesian(
            patm, x, y, z, fixed_patm, plot_flag
        )
        
        if result is None:
            if plot_flag > 0:
                print(f'Patch {pix}: Skipping - parameterization failed')
            continue
        
        minpatm, uv, x_sol, y_sol, z_sol, t_sol, p_sol, fixed_patm_reduced = result

        # [3b] VERIFICATION: Check key vertex positions after solve
        if plot_flag > 0:
            spm_v = PM.get('spm', None)
            Xkeyind_v = PM.get('Xkeyind', None)
            if spm_v is not None and Xkeyind_v is not None:
                # Check each key vertex on the border of this patch
                border_set_v = set()
                if hasattr(patm, 'border_vertex') and patm.border_vertex is not None:
                    border_set_v = set(
                        np.where(patm.border_vertex.astype(bool))[0].tolist())
                for si_v in range(len(Xkeyind_v)):
                    mX_idx_v = int(Xkeyind_v[si_v])
                    if mX_idx_v not in border_set_v:
                        continue
                    # Find mX_idx_v in uv to get the reduced index
                    red_idx_v = np.where(uv == mX_idx_v)[0]
                    if len(red_idx_v) == 0:
                        continue
                    red_idx_v = red_idx_v[0]
                    actual_pos = np.array([x_sol[red_idx_v],
                                           y_sol[red_idx_v],
                                           z_sol[red_idx_v]])
                    expected_pos = spm_v.X[si_v]
                    err_v = np.linalg.norm(actual_pos - expected_pos)
                    if err_v > 0.01:
                        print(f'  !! Key v{si_v} (mX={mX_idx_v}): '
                              f'expected={expected_pos.round(4)}, '
                              f'actual={actual_pos.round(4)}, '
                              f'err={err_v:.4f}')

                # Compute centroid of ALL solved vertices vs simplified mesh
                face_verts_set = set(patm.F.flatten().tolist())
                pm_chk = PM.get('pm', None)
                if pm_chk is not None and pm_chk.face_labels is not None:
                    pf_idx = np.where(pm_chk.face_labels == pix)[0]
                    if len(pf_idx) > 0:
                        simpl_cents = []
                        for fi_v in pf_idx:
                            verts_v = spm_v.X[spm_v.F[fi_v]]
                            simpl_cents.append(verts_v.mean(axis=0))
                        simpl_c = np.mean(simpl_cents, axis=0)
                        simpl_c_hat = simpl_c / max(np.linalg.norm(simpl_c), 1e-15)

                        fine_c = np.array([x_sol.mean(), y_sol.mean(), z_sol.mean()])
                        fine_c_hat = fine_c / max(np.linalg.norm(fine_c), 1e-15)

                        cos_a = np.clip(np.dot(fine_c_hat, simpl_c_hat), -1, 1)
                        angle_v = np.degrees(np.arccos(cos_a))
                        if angle_v > 15:
                            print(f'  !! Patch {pix} centroid offset: '
                                  f'{angle_v:.1f} deg  '
                                  f'fine={fine_c_hat.round(3)} '
                                  f'simpl={simpl_c_hat.round(3)}')
        
        # [4] Update patch with parameterization
        patm.t[uv] = minpatm.t
        patm.p[uv] = minpatm.p
        
        # Store parameterized patch
        PM['P'][pix][0] = patm
        PM['PX'][pix] = np.column_stack([
            fixed_patm_reduced,
            x_sol, y_sol, z_sol,
            t_sol, p_sol
        ])

        # [5] Per-patch assertion: count residual foldovers
        try:
            X_chk = np.column_stack([x_sol, y_sol, z_sol])
            n_fold_chk = 0
            for fi in range(len(minpatm.F)):
                i0, i1, i2 = minpatm.F[fi]
                va, vb, vc = X_chk[i0], X_chk[i1], X_chk[i2]
                orient_chk = np.dot((va + vb + vc) / 3.0,
                                    np.cross(vb - va, vc - va))
                if orient_chk < -1e-15:
                    n_fold_chk += 1
            if plot_flag > 0 or n_fold_chk > 0:
                status = 'OK' if n_fold_chk == 0 else f'** {n_fold_chk} FOLDOVERS **'
                print(f'Patch {pix}: {len(minpatm.F)} faces, {status}')
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pass 2: retry deferred annular patches (inner-ring neighbours
    #         should now be parameterized)
    # ------------------------------------------------------------------
    if deferred_annular:
        if plot_flag > 0:
            print(f'\n=== Pass 2: retrying {len(deferred_annular)} deferred '
                  f'annular patches: {deferred_annular} ===')
        for pix in deferred_annular:
            patm = PM['P'][pix][0]
            annular_result = _parameterize_annular_patch(
                PM, pix, patm, plot_flag)
            if annular_result is not None:
                _store_annular_result(PM, pix, patm, annular_result, plot_flag)
            else:
                if plot_flag > 0:
                    print(f'Patch {pix}: Pass 2 annular handler also failed — '
                          f'falling through to normal path')
                # Fall through to normal path
                border_info = validate_and_prepare_border_chains(
                    PM, pix, plot_flag)
                if border_info is None:
                    continue
                boundary_result = assign_boundary_values(
                    PM, pix, patm, border_info, plot_flag)
                if boundary_result is None:
                    continue
                x, y, z, t_bv, p_bv, fixed_patm, chain_assigned = boundary_result
                result = parameterize_single_patch_with_cartesian(
                    patm, x, y, z, fixed_patm, plot_flag)
                if result is None:
                    continue
                minpatm, uv, x_sol, y_sol, z_sol, t_sol, p_sol, \
                    fixed_patm_reduced = result
                patm.t[uv] = minpatm.t
                patm.p[uv] = minpatm.p
                PM['P'][pix][0] = patm
                PM['PX'][pix] = np.column_stack([
                    fixed_patm_reduced,
                    x_sol, y_sol, z_sol,
                    t_sol, p_sol
                ])

    # Write compact diagnostic of sphere parameterization (mirrors simplified_mesh_diagnostic)
    try:
        from .diagnose_sphere_parameterization import write_sphere_parameterization_diagnostic
        write_sphere_parameterization_diagnostic(PM)
    except Exception:
        pass

    # Fine-mesh sphere quality check (foldovers, elongated triangles, overlaps)
    try:
        _diagnose_fine_mesh_sphere_quality(PM, plot_flag)
    except Exception as e:
        if plot_flag > 0:
            print(f'Fine mesh sphere quality diagnostic failed: {e}')

    # ------------------------------------------------------------------
    # Post-parameterization quality summary
    # ------------------------------------------------------------------
    if overspread_patches:
        print(f'\n{"="*60}')
        print(f'PARAMETERIZATION QUALITY SUMMARY (run_id={run_id})')
        print(f'{"="*60}')
        print(f'Overspread patches detected — these are the likely root cause')
        print(f'of foldovers and boundary self-intersections in neighboring')
        print(f'patches. The sphere mapping (map2sphere) distributed key')
        print(f'vertices too widely for these patches.')
        print(f'\nRecommended actions:')
        print(f'  1. Set avoid_topological_annuli=True to prevent annular')
        print(f'     patches whose zipper topology distorts the mapping.')
        print(f'  2. Increase newton_niter for the simplified mesh optimization')
        print(f'     (try optimization_method=3 or 4 with more iterations).')
        print(f'  3. Try a different nseeds_range to get a different segmentation.')
        print(f'{"="*60}\n')

    return PM


def _diagnose_fine_mesh_sphere_quality(PM, plot_flag=0):
    """
    Comprehensive quality check for the fine-mesh parameterization on the sphere.

    For every parameterized patch this function:
    - computes signed spherical area per face (orient = v1 . (v2 x v3))
    - detects foldovers (negative orient => face pointing inward)
    - measures triangle aspect ratios (longest / shortest edge)
    - reports the worst offenders

    A text report is written next to the other diagnostics.
    """
    import os

    if 'PX' not in PM or len(PM['PX']) == 0:
        return

    lines = []
    lines.append(_make_run_header(PM.get('run_id')))
    lines.append('=' * 70)
    lines.append('FINE-MESH SPHERE PARAMETERIZATION QUALITY')
    lines.append('=' * 70)

    total_foldovers = 0
    total_faces = 0
    total_elongated = 0
    AR_THRESHOLD = 20.0  # faces with aspect ratio > this are "elongated"

    patch_summaries = []

    for pix in range(PM['npatches']):
        if pix not in PM['PX']:
            continue

        patm = PM['P'][pix][0]
        px = PM['PX'][pix]
        x_s, y_s, z_s = px[:, 1], px[:, 2], px[:, 3]

        # Rebuild the reduced (minimal) mesh so we can iterate over its faces
        minpatm, uv = reduce_to_minimal_set(patm)
        X_sph = np.column_stack([x_s, y_s, z_s])   # already in reduced-mesh space

        n_faces = len(minpatm.F)
        orients = np.zeros(n_faces)
        aspect_ratios = np.zeros(n_faces)
        edge_lengths = np.zeros((n_faces, 3))

        for fi in range(n_faces):
            i0, i1, i2 = minpatm.F[fi]
            v0, v1, v2 = X_sph[i0], X_sph[i1], X_sph[i2]

            # Signed area proxy  (positive = outward on unit sphere)
            cross_vec = np.cross(v1 - v0, v2 - v0)
            centroid = (v0 + v1 + v2) / 3.0
            orients[fi] = np.dot(centroid, cross_vec)

            # Edge lengths (Euclidean chord lengths on the sphere)
            e0 = np.linalg.norm(v1 - v0)
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v0 - v2)
            edge_lengths[fi] = [e0, e1, e2]
            max_e = max(e0, e1, e2)
            min_e = min(e0, e1, e2)
            aspect_ratios[fi] = max_e / max(min_e, 1e-15)

        n_foldovers = int(np.sum(orients < 0))
        n_degenerate = int(np.sum(np.abs(orients) < 1e-14))
        n_elongated = int(np.sum(aspect_ratios > AR_THRESHOLD))
        total_foldovers += n_foldovers
        total_faces += n_faces
        total_elongated += n_elongated

        patch_summaries.append((pix, n_faces, n_foldovers, n_degenerate, n_elongated,
                                aspect_ratios.max(), aspect_ratios.mean(),
                                orients.min(), orients.max()))

        lines.append(f'\n--- Patch {pix}: {n_faces} faces ---')
        lines.append(f'  Foldovers (orient < 0): {n_foldovers}')
        lines.append(f'  Degenerate (|orient| < 1e-14): {n_degenerate}')
        lines.append(f'  Elongated (AR > {AR_THRESHOLD:.0f}): {n_elongated}')
        lines.append(f'  Orient range: [{orients.min():.6f}, {orients.max():.6f}]')
        lines.append(f'  Aspect ratio: mean={aspect_ratios.mean():.2f}, '
                      f'max={aspect_ratios.max():.2f}')

        # Report worst foldovers
        if n_foldovers > 0:
            worst_idx = np.argsort(orients)[:min(5, n_foldovers)]
            lines.append(f'  Worst foldover faces (face_idx, orient, AR):')
            for wi in worst_idx:
                i0, i1, i2 = minpatm.F[wi]
                lines.append(f'    face {wi}: orient={orients[wi]:.6f}, '
                              f'AR={aspect_ratios[wi]:.2f}, '
                              f'verts=[{i0},{i1},{i2}], '
                              f'edges=[{edge_lengths[wi,0]:.4f},{edge_lengths[wi,1]:.4f},{edge_lengths[wi,2]:.4f}]')

        # Report worst elongated faces (only if no foldover, to keep output manageable)
        if n_elongated > 0 and n_foldovers == 0:
            worst_ar = np.argsort(-aspect_ratios)[:min(5, n_elongated)]
            lines.append(f'  Most elongated faces (face_idx, AR, orient):')
            for wi in worst_ar:
                lines.append(f'    face {wi}: AR={aspect_ratios[wi]:.2f}, '
                              f'orient={orients[wi]:.6f}')

        # Boundary self-intersection check (pairwise edge crossings)
        if n_foldovers > 0 and hasattr(minpatm, 'border_vertex') and minpatm.border_vertex is not None:
            bv_mask = minpatm.border_vertex.astype(bool)
            bnd_edges = []
            for fi in range(n_faces):
                tri = minpatm.F[fi]
                for ea, eb in [(0,1),(1,2),(2,0)]:
                    va, vb = int(tri[ea]), int(tri[eb])
                    if bv_mask[va] and bv_mask[vb]:
                        e = (min(va,vb), max(va,vb))
                        bnd_edges.append(e)
            bnd_edges = list(set(bnd_edges))
            # Count pairwise crossings (2D projection via stereographic)
            if len(bnd_edges) > 2:
                bnd_verts_set = set()
                for a,b in bnd_edges:
                    bnd_verts_set.add(a); bnd_verts_set.add(b)
                bnd_pts = X_sph[np.array(list(bnd_verts_set))]
                cent_b = bnd_pts.mean(axis=0)
                cn_b = np.linalg.norm(cent_b)
                if cn_b > 1e-12:
                    pole_b = -cent_b / cn_b
                    try:
                        uv_b, _ = _stereo_project(X_sph, pole_b)
                        # Check pairwise crossings
                        n_cross = 0
                        for i_e in range(len(bnd_edges)):
                            a1, a2 = bnd_edges[i_e]
                            p1, p2 = uv_b[a1], uv_b[a2]
                            for j_e in range(i_e+1, len(bnd_edges)):
                                b1, b2 = bnd_edges[j_e]
                                if b1 == a1 or b1 == a2 or b2 == a1 or b2 == a2:
                                    continue
                                p3, p4 = uv_b[b1], uv_b[b2]
                                # 2D segment intersection test
                                d1 = p2 - p1; d2 = p4 - p3; d3 = p3 - p1
                                cross_d = d1[0]*d2[1] - d1[1]*d2[0]
                                if abs(cross_d) < 1e-15:
                                    continue
                                t_p = (d3[0]*d2[1] - d3[1]*d2[0]) / cross_d
                                u_p = (d3[0]*d1[1] - d3[1]*d1[0]) / cross_d
                                if 0 < t_p < 1 and 0 < u_p < 1:
                                    n_cross += 1
                        if n_cross > 0:
                            lines.append(f'  ** BOUNDARY SELF-INTERSECTIONS: {n_cross} edge crossings **')
                        else:
                            lines.append(f'  Boundary: no self-intersections ({len(bnd_edges)} edges)')
                    except Exception:
                        pass

    # Global summary
    lines.insert(3, f'Total faces: {total_faces}')
    lines.insert(4, f'Total foldovers: {total_foldovers}')
    lines.insert(5, f'Total elongated (AR>{AR_THRESHOLD:.0f}): {total_elongated}')
    if total_foldovers > 0:
        lines.insert(6, 'STATUS: *** FOLDOVERS DETECTED -- sphere mapping has face intersections ***')
    elif total_elongated > 0:
        lines.insert(6, f'STATUS: No foldovers but {total_elongated} elongated faces')
    else:
        lines.insert(6, 'STATUS: OK -- no foldovers, no highly elongated faces')
    lines.append('\n' + '=' * 70)

    # Print summary
    if plot_flag > 0:
        print(f'\nFine-mesh sphere quality: {total_faces} faces, '
              f'{total_foldovers} foldovers, {total_elongated} elongated')
        if total_foldovers > 0:
            for pix, nf, nfo, ndeg, nel, mxar, mnar, mnor, mxor in patch_summaries:
                if nfo > 0:
                    print(f'  Patch {pix}: {nfo}/{nf} foldovers, '
                          f'worst orient={mnor:.6f}, max AR={mxar:.2f}')

    # Write diagnostic file
    try:
        diag_dir = os.path.join(os.path.dirname(__file__), '..', 'tests')
        os.makedirs(diag_dir, exist_ok=True)
        diag_path = os.path.join(diag_dir, 'fine_mesh_sphere_quality.txt')
        with open(diag_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        if plot_flag > 0:
            print(f'  Diagnostic written to: {os.path.abspath(diag_path)}')
    except Exception:
        pass

    return patch_summaries


def analyze_chain_importance(PM, pix, plot_flag=0):
    """
    Analyze the importance and quality of chain vertices for a patch.
    
    This function helps understand:
    - How many border vertices are covered by chains
    - Whether chains are accurate or contain errors
    - The impact of chain quality on parameterization
    
    Parameters:
    -----------
    PM : dict
        Patch mesh structure
    pix : int
        Patch index
    plot_flag : int
        Plotting flag
        
    Returns:
    --------
    analysis : dict
        Analysis results with statistics
    """
    patm = PM['P'][pix][0]
    border_vertices = np.where(patm.border_vertex)[0] if hasattr(patm, 'border_vertex') and patm.border_vertex is not None else np.array([])
    
    # Get chain vertices
    chain_vertices = set()
    if 'patch' in PM and pix in PM.get('patch', {}):
        for edge_chain in PM['patch'][pix]['edge_dat']:
            chain_vertices.update(edge_chain.astype(int))
        chain_source = 'refined (PM.patch)'
    else:
        for eix in range(len(PM['Edges'])):
            if PM['Edges'][eix, 0] == pix or PM['Edges'][eix, 1] == pix:
                if eix < len(PM['edge_dat']) and len(PM['edge_dat'][eix]) > 0:
                    chain_vertices.update(PM['edge_dat'][eix].astype(int))
        chain_source = 'initial (PM.edge_dat)'
    
    chain_vertices = np.array(list(chain_vertices))
    
    # Analyze coverage
    border_set = set(border_vertices)
    chain_set = set(chain_vertices[chain_vertices < len(patm.X)])
    overlap = border_set.intersection(chain_set)
    border_only = border_set - chain_set
    chain_only = chain_set - border_set
    
    coverage_ratio = len(overlap) / len(border_vertices) if len(border_vertices) > 0 else 0.0
    
    # Convert sets to arrays, handling empty sets properly
    # Ensure we always get 1D arrays, not 0D arrays
    if len(border_only) > 0:
        border_only_arr = np.array(list(border_only), dtype=int)
    else:
        border_only_arr = np.array([], dtype=int)
    
    if len(chain_only) > 0:
        chain_only_arr = np.array(list(chain_only), dtype=int)
    else:
        chain_only_arr = np.array([], dtype=int)
    
    analysis = {
        'chain_source': chain_source,
        'border_vertices': len(border_vertices),
        'chain_vertices': len(chain_set),
        'overlap': len(overlap),
        'border_only': len(border_only),
        'chain_only': len(chain_only),
        'coverage_ratio': coverage_ratio,
        'border_only_vertices': border_only_arr,
        'chain_only_vertices': chain_only_arr
    }
    
    if plot_flag > 0:
        print(f"\nPatch {pix} Chain Analysis:")
        print(f"  Chain source: {chain_source}")
        print(f"  Border vertices: {len(border_vertices)}")
        print(f"  Chain vertices: {len(chain_set)}")
        print(f"  Coverage: {len(overlap)} / {len(border_vertices)} ({coverage_ratio*100:.1f}%)")
        print(f"  Border-only (missing from chains): {len(border_only)}")
        print(f"  Chain-only (potentially incorrect): {len(chain_only)}")
        
        if coverage_ratio < 0.8:
            print(f"  WARNING: Low coverage! Only {coverage_ratio*100:.1f}% of border vertices are in chains.")
            print(f"           This will cause incorrect parameterization.")
        if len(chain_only) > len(border_vertices) * 0.2:
            print(f"  WARNING: Many chain-only vertices ({len(chain_only)}) - chains may be incorrect.")
    
    return analysis


def validate_and_prepare_border_chains(PM, pix, plot_flag=0):
    """
    Validate and prepare border chains for a patch.
    
    Parameters:
    -----------
    PM : dict
        Patch mesh structure
    pix : int
        Patch index
    plot_flag : int
        Plotting flag
        
    Returns:
    --------
    border_info : dict or None
        Dictionary containing:
        - 'edge_chains': list of edge chains (ol arrays)
        - 'sentinels': list of [pms1, pms2] pairs for each chain
        - 'spm_indices': list of [spms1, spms2] pairs (indices into PM['spm'])
        - 'use_patch_edges': bool indicating if using refined edge chains
        Returns None if no valid chains found
    """
    # Determine which edge chains to use
    #
    # PM['patch'] chains are built from border2chain (OUT_chain) and are
    # generally more accurate than PM.Edges chains (find_edge_chain can
    # fail and append distant vertices).
    #
    # For single-boundary patches: always prefer PM['patch'] chains.
    # For multi-boundary patches: use PM['patch'] chains IF they cover
    #   >= 90 % of border vertices.  This catches false-positive
    #   multi-boundary detections (e.g. _build_boundary_cycles splitting
    #   a single boundary at a key vertex).  When coverage is low the
    #   patch truly has a disconnected inner ring that PM['patch'] chains
    #   don't reach, so we fall back to PM.Edges chains.
    use_patch_edges = False
    if 'patch' in PM and pix in PM.get('patch', {}):
        patm_check = PM['P'][pix][0]
        n_bnd_components = len(_build_boundary_cycles(patm_check))
        if n_bnd_components <= 1:
            # Single boundary – PM['patch'] chains are reliable
            use_patch_edges = True
        else:
            # Multi-boundary detected.  Check PM.patch chain coverage.
            border_verts = set()
            if hasattr(patm_check, 'border_vertex') and patm_check.border_vertex is not None:
                border_verts = set(
                    np.where(patm_check.border_vertex.astype(bool))[0].tolist())

            if len(border_verts) > 0:
                patch_chain_verts = set()
                for ed in PM['patch'][pix]['edge_dat']:
                    patch_chain_verts.update(np.asarray(ed).astype(int).tolist())
                coverage = len(patch_chain_verts & border_verts) / len(border_verts)
            else:
                coverage = 0.0

            if coverage >= 0.9:
                # PM.patch chains cover (nearly) every border vertex.
                # The multi-boundary detection is likely a false positive
                # — use the more reliable PM.patch chains.
                use_patch_edges = True
                if plot_flag > 0:
                    print(f'Patch {pix}: {n_bnd_components} boundary components '
                          f'but PM.patch chains cover {coverage*100:.0f}% of '
                          f'border -> using PM.patch (false-positive override)')
            else:
                if plot_flag > 0:
                    print(f'Patch {pix}: {n_bnd_components} boundary components, '
                          f'PM.patch chains cover {coverage*100:.0f}% of border '
                          f'-> using PM.Edges chains')

    if use_patch_edges:
        indx = list(range(len(PM['patch'][pix]['edge_dat'])))
    else:
        # Use PM.Edges (one chain per edge, correctly separating boundaries)
        indx = []
        for eix in range(len(PM['Edges'])):
            if PM['Edges'][eix, 0] == pix or PM['Edges'][eix, 1] == pix:
                indx.append(eix)
        use_patch_edges = False
    
    edge_chains = []
    sentinels = []
    spm_indices = []
    
    # Collect all valid edge chains
    for ix in range(len(indx)):
        if use_patch_edges:
            ol = PM['patch'][pix]['edge_dat'][ix]
            pms1 = PM['patch'][pix]['key_dat'][ix, 0]
            pms2 = PM['patch'][pix]['key_dat'][ix, 1]
        else:
            eix = indx[ix]
            if eix >= len(PM['edge_dat']) or len(PM['edge_dat'][eix]) == 0:
                continue
                
            ol = PM['edge_dat'][eix]
            pms1 = int(PM['sentinels'][eix, 0])
            pms2 = int(PM['sentinels'][eix, 1])
        
        if len(ol) == 0:
            continue
        
        # Find sentinel indices in simplified mesh
        spms1 = np.where(PM['Xkeyind'] == pms1)[0]
        spms2 = np.where(PM['Xkeyind'] == pms2)[0]
        
        if len(spms1) == 0 or len(spms2) == 0:
            if plot_flag > 0:
                print(f'Patch {pix}, edge {ix}: Warning - Sentinel vertices not found in Xkeyind '
                      f'(pms1={pms1}, pms2={pms2}). Skipping this edge chain.')
            continue
        
        spms1 = spms1[0]
        spms2 = spms2[0]
        
        edge_chains.append(ol)
        sentinels.append([pms1, pms2])
        spm_indices.append([spms1, spms2])
    
    if len(edge_chains) == 0:
        return None
    
    # Diagnostic: Analyze chain importance and quality
    if plot_flag > 0:
        analysis = analyze_chain_importance(PM, pix, plot_flag=0)  # Get analysis without printing
        # Safely get chain_only count
        chain_only_vertices = analysis['chain_only_vertices']
        if isinstance(chain_only_vertices, np.ndarray) and chain_only_vertices.ndim > 0:
            chain_only_count = len(chain_only_vertices)
        else:
            chain_only_count = analysis['chain_only']  # Use the count from analysis
        
        if analysis['coverage_ratio'] < 0.8 or chain_only_count > analysis['border_vertices'] * 0.2:
            print(f'Patch {pix}: Border chain validation:')
            print(f'  Chain source: {analysis["chain_source"]}')
            print(f'  Border vertices: {analysis["border_vertices"]}')
            print(f'  Chain vertices: {analysis["chain_vertices"]}')
            print(f'  Overlap: {analysis["overlap"]} ({analysis["coverage_ratio"]*100:.1f}% coverage)')
            print(f'  Border-only (not in chains): {analysis["border_only"]}')
            print(f'  Chain-only (potentially incorrect): {analysis["chain_only"]}')
            border_only_arr = analysis['border_only_vertices']
            if isinstance(border_only_arr, np.ndarray) and border_only_arr.ndim > 0 and len(border_only_arr) > 0:
                print(f'    Border-only vertices (first 10): {border_only_arr[:10]}')
            chain_only_arr = analysis['chain_only_vertices']
            if isinstance(chain_only_arr, np.ndarray) and chain_only_arr.ndim > 0 and len(chain_only_arr) > 0:
                print(f'    Chain-only vertices (first 10): {chain_only_arr[:10]}')
            
            # Provide recommendations
            if analysis['coverage_ratio'] < 0.8:
                print(f'  ⚠️  WARNING: Low coverage! Only {analysis["coverage_ratio"]*100:.1f}% of border vertices are in chains.')
                print(f'     Border-only vertices will NOT receive boundary values, causing incorrect parameterization.')
            if chain_only_count > analysis['border_vertices'] * 0.2:
                print(f'  ⚠️  WARNING: Many chain-only vertices ({chain_only_count}) - chains may be incorrect.')
                print(f'     These vertices may be from wrong patches or incorrect paths.')
    
    return {
        'edge_chains': edge_chains,
        'sentinels': sentinels,
        'spm_indices': spm_indices,
        'use_patch_edges': use_patch_edges
    }


def assign_boundary_values(PM, pix, patm, border_info, plot_flag=0):
    """
    Assign boundary values by interpolating along edge chains.
    
    Parameters:
    -----------
    PM : dict
        Patch mesh structure
    pix : int
        Patch index
    patm : surface_mesh
        Patch mesh
    border_info : dict
        Border chain information from validate_and_prepare_border_chains
    plot_flag : int
        Plotting flag
        
    Returns:
    --------
    result : tuple or None
        (x, y, z, t, p, fixed_patm) where:
        - x, y, z: Cartesian coordinates for all vertices
        - t, p: theta and phi from last projection
        - fixed_patm: boolean array marking fixed boundary vertices
        Returns None if assignment fails
    """
    # Initialize
    patm.t = np.zeros(len(patm.X))
    patm.p = np.zeros(len(patm.X))
    
    x = np.zeros(len(patm.X))
    y = np.zeros(len(patm.X))
    z = np.zeros(len(patm.X))
    
    # Track t and p from the last projection inside the loop
    t = None
    p = None
    
    edge_chains = border_info['edge_chains']
    sentinels = border_info['sentinels']
    spm_indices = border_info['spm_indices']
    
    # Track all vertices assigned by edge chains (to protect them from
    # being overwritten by later re-interpolation steps).
    chain_assigned = set()
    
    if plot_flag > 0:
        print(f'Patch {pix}: {len(edge_chains)} edge chains to process')

    # Process each edge chain
    for ix in range(len(edge_chains)):
        ol = edge_chains[ix]
        pms1, pms2 = sentinels[ix]
        spms1, spms2 = spm_indices[ix]
        
        ol_int = ol.astype(int)
        nol = len(ol_int)
        if nol == 0:
            continue

        if plot_flag > 0:
            print(f'  Chain {ix}: sentinels=({pms1},{pms2}) -> '
                  f'simpl=({spms1},{spms2}), len={nol}, '
                  f'ol[0]={ol_int[0]}, ol[-1]={ol_int[-1]}')

        # ------------------------------------------------------------------
        # STEP 1: Ensure chain orientation matches sentinel assignment.
        #
        # This MUST come BEFORE the spatial continuity check so that
        # truncation always keeps the prefix starting at sentinel-1.
        #
        # ol[0] must correspond to pms1 and ol[-1] to pms2.  If the chain
        # is stored in the opposite direction (ol[0]==pms2, ol[-1]==pms1)
        # the interpolation would assign sentinel-1's sphere position to
        # vertices near sentinel-2 and vice-versa, producing a reversed
        # boundary segment and catastrophic foldovers.
        # ------------------------------------------------------------------
        if nol > 1:
            head_is_s1 = (ol_int[0] == pms1)
            head_is_s2 = (ol_int[0] == pms2)
            tail_is_s1 = (ol_int[-1] == pms1)
            tail_is_s2 = (ol_int[-1] == pms2)

            need_reverse = False

            if head_is_s1 and tail_is_s2:
                # Already correct orientation
                pass
            elif head_is_s2 and tail_is_s1:
                # Chain is stored in reverse -- flip it
                need_reverse = True
            elif head_is_s2 and not head_is_s1:
                # Head matches s2 but not s1 -- need to reverse
                need_reverse = True
            elif tail_is_s1 and not tail_is_s2:
                # Tail matches s1 but not s2 -- need to reverse
                need_reverse = True
            elif not head_is_s1 and not head_is_s2:
                # Chain endpoints don't match either sentinel exactly.
                # Use distance to decide orientation.
                patm_X = patm.X
                if pms1 < len(patm_X) and pms2 < len(patm_X):
                    d_head_s1 = np.linalg.norm(
                        patm_X[ol_int[0]] - patm_X[pms1])
                    d_head_s2 = np.linalg.norm(
                        patm_X[ol_int[0]] - patm_X[pms2])
                    if d_head_s2 < d_head_s1:
                        need_reverse = True

            if need_reverse:
                ol_int = ol_int[::-1]
                if plot_flag > 0:
                    print(f'    -> Reversed chain (ol[0] was {pms2}, '
                          f'now starts at {ol_int[0]})')

        # ------------------------------------------------------------------
        # STEP 2: CHAIN SPATIAL CONTINUITY CHECK
        #
        # find_edge_chain() appends s2 when the walk gets stuck, creating a
        # spatial "jump" (two consecutive chain vertices that are NOT mesh-
        # adjacent).  Detect such jumps and truncate the chain to the valid
        # prefix (starting at sentinel-1) so the gap filler can handle the
        # missing segment between the truncation point and sentinel-2.
        # ------------------------------------------------------------------
        jump_indices = _detect_chain_jumps(patm.X, ol_int)
        if len(jump_indices) > 0:
            if plot_flag > 0:
                for ji in jump_indices:
                    v_a, v_b = int(ol_int[ji]), int(ol_int[ji + 1])
                    d_ab = np.linalg.norm(patm.X[v_a] - patm.X[v_b]) if v_a < len(patm.X) and v_b < len(patm.X) else float('inf')
                    print(f'    !! Jump detected at pos {ji}: '
                          f'v{v_a}->v{v_b}, dist={d_ab:.4f}')

            # Strategy: keep only the valid prefix (up to the first jump).
            # The sentinel endpoints are handled by key-vertex pinning,
            # and the gap filler will interpolate the rest of the boundary.
            first_jump = jump_indices[0]
            ol_int = ol_int[:first_jump + 1]  # keep [0..first_jump]
            nol = len(ol_int)
            if plot_flag > 0:
                print(f'    -> Truncated chain to {nol} vertices '
                      f'(kept prefix up to jump)')
            if nol < 2:
                # Only the first sentinel vertex remains; the chain is
                # essentially empty.  Skip it entirely.
                chain_assigned.add(int(ol_int[0]))
                continue

        # Was the chain truncated by the spatial continuity check?
        chain_was_truncated = (len(jump_indices) > 0)
        # Save original chain length for interpolation spacing
        nol_for_interp = len(edge_chains[ix])

        # Get Cartesian coordinates from spherical mesh
        spms1_x_val = PM['spm'].X[spms1, 0]
        spms1_y_val = PM['spm'].X[spms1, 1]
        spms1_z_val = PM['spm'].X[spms1, 2]
        spms2_x_val = PM['spm'].X[spms2, 0]
        spms2_y_val = PM['spm'].X[spms2, 1]
        spms2_z_val = PM['spm'].X[spms2, 2]
        
        # Spherical linear interpolation (slerp) along the great circle
        # between sentinel positions.  This replaces linear Cartesian
        # interpolation + projection, which distorts badly when sentinels
        # are far apart on the sphere (the chord dips toward the origin).
        p1 = np.array([spms1_x_val, spms1_y_val, spms1_z_val])
        p2 = np.array([spms2_x_val, spms2_y_val, spms2_z_val])
        n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
        if n1 > 1e-15:
            p1 = p1 / n1
        if n2 > 1e-15:
            p2 = p2 / n2
        cos_omega = np.clip(np.dot(p1, p2), -1.0, 1.0)
        omega = np.arccos(cos_omega)
        sin_omega = np.sin(omega)
        use_slerp = (sin_omega > 1e-8)

        # Assign sentinel 1 position (first endpoint)
        x[ol_int[0]] = p1[0]
        y[ol_int[0]] = p1[1]
        z[ol_int[0]] = p1[2]

        # Assign sentinel 2 position (last endpoint) — only if chain is
        # complete (not truncated).  For truncated chains the last vertex
        # is NOT sentinel 2 and should be interpolated instead.
        if nol > 1 and not chain_was_truncated:
            x[ol_int[-1]] = p2[0]
            y[ol_int[-1]] = p2[1]
            z[ol_int[-1]] = p2[2]

        # Interpolate interior vertices along the great circle.
        # For truncated chains we also interpolate the last vertex.
        interp_end = (nol - 1) if not chain_was_truncated else nol
        for oix in range(1, interp_end):
            vid = ol_int[oix]
            if vid in chain_assigned:
                continue
            alpha = (oix + 1) / nol_for_interp
            if use_slerp:
                c1 = np.sin((1.0 - alpha) * omega) / sin_omega
                c2 = np.sin(alpha * omega) / sin_omega
                pt = c1 * p1 + c2 * p2
            else:
                pt = (1.0 - alpha) * p1 + alpha * p2
            r_pt = np.linalg.norm(pt)
            if r_pt > 1e-15:
                pt = pt / r_pt
            x[vid] = pt[0]
            y[vid] = pt[1]
            z[vid] = pt[2]

        if plot_flag > 0 and omega > 0.5:
            print(f'    Slerp: arc={np.degrees(omega):.1f}° between '
                  f'sentinels ({pms1},{pms2})')

        # Project ALL positions onto the unit sphere (handles vertices
        # still at origin from earlier chains or initialization).
        t, p, _ = kk_cart2sph(x, y, z)
        x, y, z = kk_sph2cart(t, p, np.ones_like(t))

        # Derive t, p for chain vertices from the slerped Cartesian
        # positions (consistent with x,y,z; avoids phi-wrapping bugs
        # that linear phi interpolation can introduce).
        patm.t[ol_int[0]] = t[ol_int[0]]
        patm.p[ol_int[0]] = p[ol_int[0]]

        if nol > 1 and not chain_was_truncated:
            patm.t[ol_int[-1]] = t[ol_int[-1]]
            patm.p[ol_int[-1]] = p[ol_int[-1]]

        for oix in range(1, interp_end):
            vid = ol_int[oix]
            if vid in chain_assigned:
                continue
            patm.t[vid] = t[vid]
            patm.p[vid] = p[vid]
        
        # Record all vertices assigned by this chain
        chain_assigned.update(ol_int.tolist())
    
    # Determine fixed vertices (boundary) - MATLAB line 84
    # CRITICAL: Use t from the last projection inside the loop
    # MATLAB: fixed_patm = (t~=0 & t~=pi/2);
    if t is None:
        # If no edges were processed, initialize t
        t, _, _ = kk_cart2sph(x, y, z)
    fixed_patm = (np.abs(t) > 1e-10) & (np.abs(t - np.pi / 2) > 1e-10)

    # ------------------------------------------------------------------
    # FIX: Restrict fixed_patm to BORDER vertices only.
    #
    # PM.Edges chains can include vertices that are NOT on this patch's
    # border (they belong to the neighbour).  The theta-based check above
    # would mark these as fixed, incorrectly constraining interior
    # vertices of this patch at wrong positions (wherever the chain
    # interpolation placed them).  Restricting to border vertices
    # prevents this.
    # ------------------------------------------------------------------
    if hasattr(patm, 'border_vertex') and patm.border_vertex is not None:
        fixed_patm = fixed_patm & patm.border_vertex.astype(bool)

    # PM structure fix: All border vertices MUST be fixed for correct boundary conditions.
    # Some border vertices may be at poles (t=0 or pi/2) or not in any edge chain,
    # causing theta-based fixed_patm to miss them. Use border_vertex as authoritative.
    #
    # For multi-boundary patches (annular patches), the edge chain from sentinel
    # to sentinel covers only one direction around a cyclic boundary.  The other
    # half has no chain.  A naive nearest-neighbour fill would assign wrong
    # positions (e.g. from the outer boundary to inner-boundary vertices).
    # Instead we walk boundary cycles and linearly interpolate the gaps between
    # the nearest fixed vertices ON THE SAME CYCLE.
    if hasattr(patm, 'border_vertex') and patm.border_vertex is not None:
        border_mask = patm.border_vertex.astype(bool)
        border_count = np.sum(border_mask)
        fixed_count = np.sum(fixed_patm)

        # Also count how many border vertices were covered by chains
        chain_border = border_mask & np.array(
            [i in chain_assigned for i in range(len(border_mask))], dtype=bool)
        n_chain_border = int(np.sum(chain_border))

        missing = border_mask & ~fixed_patm
        n_missing = int(np.sum(missing))

        if plot_flag > 0:
            print(f'Patch {pix}: border={border_count}, chain-assigned={n_chain_border}, '
                  f'theta-fixed={fixed_count}, missing={n_missing}')

        if n_missing > 0:
            if plot_flag > 0:
                print(f'Patch {pix}: {n_missing} border vertices not in any edge chain '
                      f'(border={border_count}, theta-fixed={fixed_count})')

            # ---- boundary-cycle-aware interpolation ----
            x, y, z, filled = _fill_boundary_gaps(patm, x, y, z, fixed_patm)

            if np.any(filled):
                # Update t, p for filled vertices
                t_filled, p_filled, _ = kk_cart2sph(x, y, z)
                for vi in np.where(filled)[0]:
                    patm.t[vi] = t_filled[vi]
                    patm.p[vi] = p_filled[vi]
                fixed_patm = fixed_patm | filled
                if plot_flag > 0:
                    print(f'  Filled {int(np.sum(filled))} vertices via boundary cycle interpolation')

            # For any border vertices STILL not fixed (e.g. isolated or
            # cycle-build failure), fall back to nearest-chain-vertex.
            still_missing = border_mask & ~fixed_patm
            if np.any(still_missing):
                chain_set = set()
                for ol in edge_chains:
                    chain_set.update(ol.astype(int).tolist())
                chain_arr = np.array([c for c in chain_set if c < len(patm.X)])
                if len(chain_arr) > 0:
                    pts_chain = np.column_stack([x[chain_arr], y[chain_arr], z[chain_arr]])
                    for vi in np.where(still_missing)[0]:
                        pt_vi = np.array([x[vi], y[vi], z[vi]])
                        dists = np.linalg.norm(pts_chain - pt_vi, axis=1)
                        nn = chain_arr[np.argmin(dists)]
                        x[vi], y[vi], z[vi] = x[nn], y[nn], z[nn]
                        patm.t[vi], patm.p[vi] = patm.t[nn], patm.p[nn]
                if plot_flag > 0:
                    print(f'  Fallback nearest-neighbour for {int(np.sum(still_missing))} remaining vertices')

        # All border vertices must be fixed
        fixed_patm = fixed_patm | border_mask

    # ------------------------------------------------------------------
    # KEY-VERTEX PINNING
    #
    # For cap/annular patches the boundary cycle may pass through several
    # key vertices of the simplified mesh but the PM.Edges structure only
    # stores ONE edge chain between a single pair of sentinels.  Key
    # vertices on the "other half" of the cycle receive interpolated
    # positions from _fill_boundary_gaps, which may be wrong (the gap
    # filler doesn't know they are key vertices).
    #
    # Fix: overwrite every key vertex present in this patch with its
    # CORRECT position from PM['spm'] (the parameterized simplified mesh).
    # ------------------------------------------------------------------
    Xkeyind = PM.get('Xkeyind', None)
    spm = PM.get('spm', None)
    if Xkeyind is not None and spm is not None:
        # Only pin key vertices that are on the BOUNDARY of this patch.
        # Center vertices of other patches might be interior — do NOT pin those.
        border_set = set()
        if hasattr(patm, 'border_vertex') and patm.border_vertex is not None:
            border_set = set(np.where(patm.border_vertex.astype(bool))[0].tolist())

        n_pinned = 0
        for si in range(len(Xkeyind)):
            mX_idx = int(Xkeyind[si])
            if mX_idx in border_set and mX_idx < len(x):
                # Overwrite with correct position from simplified mesh on sphere
                x[mX_idx] = spm.X[si, 0]
                y[mX_idx] = spm.X[si, 1]
                z[mX_idx] = spm.X[si, 2]
                # Also update t, p
                patm.t[mX_idx] = spm.t[si] if spm.t is not None else 0.0
                patm.p[mX_idx] = spm.p[si] if spm.p is not None else 0.0
                fixed_patm[mX_idx] = True
                n_pinned += 1

        if n_pinned > 0 and plot_flag > 0:
            print(f'Patch {pix}: Pinned {n_pinned} key vertices to simplified mesh positions')

        # Re-interpolate ONLY boundary vertices that were NOT already
        # assigned by an edge chain.  Chain-assigned vertices already
        # have correct positions; overwriting them via cycle-based
        # re-interpolation can corrupt the boundary when cycle segment
        # order doesn't perfectly match the chain structure.
        if n_pinned > 0 and hasattr(patm, 'border_vertex') and patm.border_vertex is not None:
            border_mask_local = patm.border_vertex.astype(bool)

            # Build the set of key vertex indices that sit on this
            # patch's boundary — these are the "anchors".
            pinned_set = set()
            for si in range(len(Xkeyind)):
                mX_idx = int(Xkeyind[si])
                if mX_idx in border_set:
                    pinned_set.add(mX_idx)

            cycles = _build_boundary_cycles(patm)
            n_reinterp = 0
            for cycle in cycles:
                n_cyc = len(cycle)
                if n_cyc < 3:
                    continue

                # Find indices within this cycle that are anchor (key) vertices
                anchor_positions = [i for i in range(n_cyc) if cycle[i] in pinned_set]
                if len(anchor_positions) < 2:
                    # Not enough anchors on this cycle — fall back to gap fill
                    continue

                # Walk around cycle in segments between consecutive anchors
                for seg_k in range(len(anchor_positions)):
                    start_pos = anchor_positions[seg_k]
                    end_pos = anchor_positions[(seg_k + 1) % len(anchor_positions)]

                    v_start = cycle[start_pos]
                    v_end = cycle[end_pos]

                    # Collect segment vertices (excluding endpoints)
                    # ONLY include vertices NOT already in chain_assigned
                    seg_verts = []
                    all_seg_verts = []
                    pos = start_pos
                    while True:
                        pos = (pos + 1) % n_cyc
                        if pos == end_pos:
                            break
                        all_seg_verts.append(cycle[pos])
                        if cycle[pos] not in chain_assigned:
                            seg_verts.append(cycle[pos])

                    if len(seg_verts) == 0:
                        continue

                    # SLERP-style interpolation between start and end on the sphere
                    # Only for vertices NOT already set by a chain
                    p_start = np.array([x[v_start], y[v_start], z[v_start]])
                    p_end = np.array([x[v_end], y[v_end], z[v_end]])
                    n_start = np.linalg.norm(p_start)
                    n_end = np.linalg.norm(p_end)
                    if n_start < 1e-15 or n_end < 1e-15:
                        continue
                    p_start = p_start / n_start
                    p_end = p_end / n_end

                    # Place non-chain vertices proportionally within
                    # the full segment (including chain-assigned ones
                    # that we skip but count for position)
                    total = len(all_seg_verts) + 1  # +1 for the gap endpoints
                    for idx_s, sv in enumerate(all_seg_verts):
                        if sv in chain_assigned:
                            continue
                        alpha = (idx_s + 1) / total
                        interp = (1.0 - alpha) * p_start + alpha * p_end
                        r_i = np.linalg.norm(interp)
                        if r_i > 1e-15:
                            interp /= r_i
                        x[sv] = interp[0]
                        y[sv] = interp[1]
                        z[sv] = interp[2]
                        t_tmp, p_tmp, _ = kk_cart2sph(
                            np.array([interp[0]]),
                            np.array([interp[1]]),
                            np.array([interp[2]]))
                        patm.t[sv] = t_tmp[0]
                        patm.p[sv] = p_tmp[0]
                        n_reinterp += 1

            if n_reinterp > 0:
                if plot_flag > 0:
                    print(f'Patch {pix}: Re-interpolated {n_reinterp} non-chain boundary vertices between key-vertex anchors')
            elif plot_flag > 0:
                print(f'Patch {pix}: All border vertices already assigned by chains (no re-interpolation needed)')

            # Ensure all border vertices are fixed
            fixed_patm = fixed_patm | border_mask_local

    return (x, y, z, t, p, fixed_patm, chain_assigned)


def _boundary_spread_degrees(x, y, z, fixed_mask):
    """Compute the angular spread (degrees) of boundary vertices on the sphere."""
    bnd_idx = np.where(fixed_mask)[0]
    if len(bnd_idx) < 2:
        return 0.0
    pts = np.column_stack([x[bnd_idx], y[bnd_idx], z[bnd_idx]])
    centroid = pts.mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn < 1e-12:
        return 180.0
    c_hat = centroid / cn
    cos_angles = pts @ c_hat
    min_cos = cos_angles.min()
    return float(np.degrees(np.arccos(np.clip(min_cos, -1, 1))))


def _stereo_project(pts, pole):
    """Stereographic projection from *pole* (unit vector).

    Maps points on the unit sphere to R^2.  The pole itself maps to
    infinity, so it must not be a data point.

    Parameters
    ----------
    pts : (N,3)  Cartesian points on the unit sphere
    pole : (3,)  the projection pole (unit vector)

    Returns
    -------
    uv : (N,2)  projected 2D coordinates
    """
    # Rotate so that *pole* is at the south pole (0,0,-1).
    # Then standard south-pole stereographic: (x,y) / (1+z)
    # Build rotation: align pole -> (0,0,-1)
    target = np.array([0.0, 0.0, -1.0])
    R = _rotation_between(pole, target)
    rpts = (R @ pts.T).T                    # (N,3)
    denom = 1.0 + rpts[:, 2]                # 1 + z
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    u = rpts[:, 0] / denom
    v = rpts[:, 1] / denom
    return np.column_stack([u, v]), R


def _stereo_unproject(uv, R):
    """Inverse stereographic projection (south-pole convention).

    Parameters
    ----------
    uv : (N,2)
    R  : (3,3)  rotation used in the forward projection

    Returns
    -------
    pts : (N,3) Cartesian points on the unit sphere
    """
    u, v = uv[:, 0], uv[:, 1]
    r2 = u**2 + v**2
    denom = r2 + 1.0
    x = 2.0 * u / denom
    y = 2.0 * v / denom
    z = (1.0 - r2) / denom
    rpts = np.column_stack([x, y, z])
    # Undo rotation
    pts = (R.T @ rpts.T).T
    # Normalise to unit sphere (guard against numerical drift)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _rotation_between(a, b):
    """Rotation matrix that maps unit vector *a* to unit vector *b*."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180-degree rotation about any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, a) * a
        perp = perp / np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def parameterize_single_patch_with_cartesian(patm, x, y, z, fixed_patm, plot_flag=0,
                                              pole_hint=None,
                                              solve_spherical=False,
                                              force_cartesian=False,
                                              inner_border_verts=None):
    """
    Parameterize a single patch using Cartesian coordinates with given boundary conditions.
    
    This function:
    1. Reduces patch to minimal set
    2. Builds Laplacian system
    3. Solves for interior vertices
    4. Projects solution onto sphere
    
    For patches whose boundary spans more than ~50 degrees on the sphere,
    a stereographic-projection Laplace solver is used instead of the
    direct 3-D Cartesian solver to avoid foldovers.

    Parameters:
    -----------
    patm : surface_mesh
        Patch mesh (will be modified)
    x, y, z : array
        Cartesian coordinates for all vertices (boundary values already assigned)
    fixed_patm : array, bool
        Boolean array marking fixed boundary vertices
    plot_flag : int
        Plotting flag
    pole_hint : array-like (3,), optional
        Unit vector on the sphere to use as the stereographic projection pole.
        When *None* (default) the antipode of the boundary centroid is used.
    solve_spherical : bool, optional
        When *True*, solve in spherical coordinates (theta, cos phi, sin phi).
    force_cartesian : bool, optional
        When *True*, always use the 3-D Cartesian Laplace solver (L*x=bx,
        L*y=by, L*z=bz) regardless of boundary spread.  Overrides the
        automatic switch to stereographic.
    inner_border_verts : set or iterable of int, optional
        For annular patches: vertex indices (original patm) on the inner ring.
        When provided with solve_spherical=True, used for inner/outer classification
        instead of the theta-gap heuristic. Ensures proper band parameterization.
        
    Returns:
    --------
    result : tuple or None
        (minpatm, uv, x_sol, y_sol, z_sol, t_sol, p_sol, fixed_patm_reduced)
        Returns None if parameterization fails
    """
    STEREO_THRESHOLD_DEG = 50.0   # switch to stereographic above this

    # [1] Reduce to minimal set of vertices (MATLAB line 154)
    minpatm, uv = reduce_to_minimal_set(patm)
    fixed_patm_reduced = fixed_patm[uv]
    
    # Check consistency with border_vertex
    if hasattr(minpatm, 'border_vertex') and minpatm.border_vertex is not None:
        border_count = np.sum(minpatm.border_vertex)
        fixed_count = np.sum(fixed_patm_reduced)
        if abs(border_count - fixed_count) > 0:
            if plot_flag > 0:
                print(f'  Discrepancy between border and edge vertices (reduced): {abs(border_count - fixed_count)}')
    
    x = x[uv]
    y = y[uv]
    z = z[uv]

    # Decide solver strategy based on boundary angular spread
    bnd_spread = _boundary_spread_degrees(x, y, z, fixed_patm_reduced)
    use_stereo = ((bnd_spread > STEREO_THRESHOLD_DEG)
                  and (not solve_spherical)
                  and (not force_cartesian))

    if force_cartesian and plot_flag > 0:
        print(f'  Using FORCED Cartesian Laplace (L*x=bx, L*y=by, L*z=bz)'
              f' — boundary spread {bnd_spread:.1f} deg')
    elif solve_spherical and plot_flag > 0:
        print(f'  Using spherical Laplace solver (theta, cos phi, sin phi)'
              f' — boundary spread {bnd_spread:.1f} deg')
    elif use_stereo and plot_flag > 0:
        print(f'  Using stereographic Laplace (boundary spread {bnd_spread:.1f} deg)')
    
    # [2] Construct the Laplacian matrix (shared by both solvers)
    #     IMPORTANT: the diagonal must equal the number of VALID links
    #     (those with index < nv), not the total link count.  A mismatch
    #     breaks the row-sum-zero property, destroying the maximum
    #     principle and the Tutte bijectivity guarantee.
    L = minpatm.L
    nv = len(L)
    A = lil_matrix((nv, nv))

    n_skipped_links = 0
    for iv in range(nv):
        links = L.get(iv, [])
        valid_count = 0
        for link in links:
            if link < nv:
                A[iv, link] = -1
                valid_count += 1
            else:
                n_skipped_links += 1
        A[iv, iv] = valid_count
    if n_skipped_links > 0 and plot_flag > 0:
        print(f'  WARNING: Laplacian skipped {n_skipped_links} out-of-range links')

    # Apply boundary conditions (shared)
    fixed_ol = np.where(fixed_patm_reduced)[0]
    for idx in fixed_ol:
        A[idx, :] = 0
        A[idx, idx] = 1

    A_csr = csr_matrix(A)

    if solve_spherical:
        # ----------------------------------------------------------
        # Theta + Tutte-phi solver for annular patches
        #
        # Two independent Dirichlet problems:
        #
        # 1. THETA  – solved directly via the graph Laplacian.
        #    Theta varies smoothly from inner ring to outer ring,
        #    has no 2-pi wrapping, and the maximum principle
        #    guarantees all interior values stay in the boundary
        #    range.
        #
        # 2. PHI  – solved via a Tutte-style 2-D embedding.
        #    Both boundary rings are mapped to concentric circles
        #    in 2-D (outer ring → r=1 circle, inner ring → r=0.5
        #    circle), preserving each vertex's phi position.
        #    The 2-D Laplace fill then gives a bijective map of
        #    the annular mesh (guaranteed by the Tutte embedding
        #    theorem for triangulated annuli with convex,
        #    properly-nested boundaries).  Phi is recovered as
        #    atan2(v, u).
        #
        # This avoids the failure modes of all other solvers:
        #   * Cartesian:     x, y cancel around the full ring,
        #                    collapsing interior to the z-axis.
        #   * Stereographic: 2-D projection hole lets Laplace
        #                    fill leak inside the inner ring.
        #   * cos/sin(phi):  same cancellation as Cartesian for
        #                    vertices deep in the interior.
        # ----------------------------------------------------------
        # a) Convert boundary Cartesian values to spherical
        t_bnd, p_bnd, _ = kk_cart2sph(x, y, z)

        # b) THETA solve — direct Laplace
        b_theta = np.zeros(nv)
        b_theta[fixed_ol] = t_bnd[fixed_ol]

        # c) PHI solve — Tutte embedding with concentric circles
        #    Classify boundary verts into inner vs outer ring.
        #    Prefer topological classification (inner_border_verts) when provided;
        #    otherwise fall back to theta-gap heuristic.
        inner_border_set = set(inner_border_verts) if inner_border_verts is not None else None
        if inner_border_set is not None:
            # uv[fi] = original patm index for reduced vertex fi
            inner_mask = np.array([int(uv[fi]) in inner_border_set for fi in fixed_ol])
        else:
            # Fallback: largest gap in sorted theta separates inner from outer
            t_fixed = t_bnd[fixed_ol]
            t_sorted_idx = np.argsort(t_fixed)
            t_sorted = t_fixed[t_sorted_idx]
            gaps = np.diff(t_sorted)
            gap_idx = int(np.argmax(gaps))
            t_threshold = (t_sorted[gap_idx] + t_sorted[gap_idx + 1]) / 2.0
            inner_mask = t_fixed > t_threshold
        outer_mask = ~inner_mask
        R_out = 1.0
        R_in  = 0.5

        b_u = np.zeros(nv)
        b_v = np.zeros(nv)
        for i, fi in enumerate(fixed_ol):
            phi_i = p_bnd[fi]
            if inner_mask[i]:
                b_u[fi] = R_in  * np.cos(phi_i)
                b_v[fi] = R_in  * np.sin(phi_i)
            else:
                b_u[fi] = R_out * np.cos(phi_i)
                b_v[fi] = R_out * np.sin(phi_i)

        if plot_flag > 0:
            n_inner = int(np.sum(inner_mask))
            n_outer = int(np.sum(outer_mask))
            src = 'topology' if inner_border_set is not None else 'theta-gap'
            msg = f'  Tutte rings: {n_inner} inner (R={R_in}), {n_outer} outer (R={R_out})'
            if inner_border_set is not None:
                msg += f', classification={src}'
            else:
                msg += f', theta split={t_threshold:.4f}'
            print(msg)

        # d) Solve the three systems
        try:
            t_sol = spsolve(A_csr, b_theta)
            u_sol = spsolve(A_csr, b_u)
            v_sol = spsolve(A_csr, b_v)
        except Exception as e:
            if plot_flag > 0:
                print(f'  Warning: Tutte solver failed ({e}), '
                      f'falling back to Cartesian')
            solve_spherical = False   # fall through to Cartesian below

        if solve_spherical:
            # d') Clamp interior theta to band bounds (ensures all vertices stay within
            #     the theta range defined by the boundary — avoids leakage outside the band)
            t_bnd_fixed = t_bnd[fixed_ol]
            t_min, t_max = float(t_bnd_fixed.min()), float(t_bnd_fixed.max())
            free_mask = ~fixed_patm_reduced
            if np.any(free_mask):
                t_sol[free_mask] = np.clip(t_sol[free_mask], t_min, t_max)

            # e) Recover phi from 2-D positions, convert to Cartesian
            p_sol = np.arctan2(v_sol, u_sol)
            x_sol, y_sol, z_sol = kk_sph2cart(
                t_sol, p_sol, np.ones_like(t_sol))
            if plot_flag > 0:
                free_mask = ~fixed_patm_reduced
                if np.any(free_mask):
                    t_free = t_sol[free_mask]
                    print(f'  Solved interior theta: '
                          f'[{t_free.min():.4f}, {t_free.max():.4f}]')

    if (not solve_spherical) and use_stereo:
        # ----------------------------------------------------------
        # Stereographic-projection Laplace solver
        # ----------------------------------------------------------
        # a) Choose projection pole
        #    For annular patches a caller-supplied pole_hint (e.g. the
        #    centre of the enclosed patch) gives better-nested
        #    projected boundaries.  Otherwise use the antipode of the
        #    boundary centroid.
        if pole_hint is not None:
            pole = np.asarray(pole_hint, dtype=float)
            pn = np.linalg.norm(pole)
            if pn > 1e-12:
                pole = pole / pn
            else:
                pole = np.array([0.0, 0.0, 1.0])
            if plot_flag > 0:
                print(f'  Stereo pole: caller hint {pole.round(4)}')
        else:
            bnd_pts = np.column_stack([x[fixed_ol], y[fixed_ol], z[fixed_ol]])
            centroid = bnd_pts.mean(axis=0)
            cn = np.linalg.norm(centroid)
            if cn < 1e-12:
                centroid = np.array([0.0, 0.0, 1.0])
            else:
                centroid = centroid / cn
            pole = -centroid          # project FROM the antipode

        # b) Forward-project boundary points
        all_pts = np.column_stack([x, y, z])
        # Normalise boundary points to unit sphere
        norms = np.linalg.norm(all_pts[fixed_ol], axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        all_pts[fixed_ol] = all_pts[fixed_ol] / norms

        bnd_xyz_original = all_pts[fixed_ol].copy()   # save for restoration

        bnd_2d, R = _stereo_project(all_pts[fixed_ol], pole)

        # b') Boundary convexification for annular patches.
        #
        #     Map each ring to a perfect circle centered at the ORIGIN,
        #     preserving vertex angles.  Concentric circles guarantee
        #     convex nested boundaries, satisfying the Tutte bijectivity
        #     theorem for the 2-D Laplace solve.
        #
        #     After solving, boundary vertices are snapped back to their
        #     true sphere positions (see step e).  The iterative foldover
        #     removal that follows absorbs any snap-back artifacts.
        _convexified_annular = False
        if inner_border_verts is not None:
            inner_border_set_s = set(inner_border_verts)
            inner_bnd_mask = np.array([int(uv[fi]) in inner_border_set_s
                                       for fi in fixed_ol])
            outer_bnd_mask = ~inner_bnd_mask
            if np.any(inner_bnd_mask) and np.any(outer_bnd_mask):
                inner_pts_2d = bnd_2d[inner_bnd_mask]
                outer_pts_2d = bnd_2d[outer_bnd_mask]

                inner_radii = np.linalg.norm(inner_pts_2d, axis=1)
                inner_angles = np.arctan2(inner_pts_2d[:, 1],
                                          inner_pts_2d[:, 0])
                inner_mean_r = float(np.mean(inner_radii))

                outer_radii = np.linalg.norm(outer_pts_2d, axis=1)
                outer_angles = np.arctan2(outer_pts_2d[:, 1],
                                          outer_pts_2d[:, 0])
                outer_mean_r = float(np.mean(outer_radii))

                if inner_mean_r < outer_mean_r:
                    bnd_2d[inner_bnd_mask] = inner_mean_r * np.column_stack(
                        [np.cos(inner_angles), np.sin(inner_angles)])
                    bnd_2d[outer_bnd_mask] = outer_mean_r * np.column_stack(
                        [np.cos(outer_angles), np.sin(outer_angles)])
                    _convexified_annular = True

                    if plot_flag > 0:
                        inner_disp = np.linalg.norm(
                            bnd_2d[inner_bnd_mask] - inner_pts_2d, axis=1)
                        outer_disp = np.linalg.norm(
                            bnd_2d[outer_bnd_mask] - outer_pts_2d, axis=1)
                        print(f'  Convexified (concentric at origin): '
                              f'inner r={inner_mean_r:.4f} '
                              f'[{inner_radii.min():.4f},'
                              f'{inner_radii.max():.4f}], '
                              f'outer r={outer_mean_r:.4f} '
                              f'[{outer_radii.min():.4f},'
                              f'{outer_radii.max():.4f}]')
                        print(f'  Max convex. displacement: '
                              f'inner={inner_disp.max():.4f}, '
                              f'outer={outer_disp.max():.4f}')
                elif plot_flag > 0:
                    print(f'  Skipping convexification: inner_r='
                          f'{inner_mean_r:.4f} >= outer_r='
                          f'{outer_mean_r:.4f}')

        # c) Set up 2-D RHS
        bu = np.zeros(nv)
        bv = np.zeros(nv)
        bu[fixed_ol] = bnd_2d[:, 0]
        bv[fixed_ol] = bnd_2d[:, 1]

        # d) Solve
        try:
            u_sol = spsolve(A_csr, bu)
            v_sol = spsolve(A_csr, bv)
        except Exception as e:
            if plot_flag > 0:
                print(f'  Warning: stereo solver failed ({e}), '
                      f'falling back to Cartesian')
            use_stereo = False   # fall through to Cartesian below

        if use_stereo:
            # e') 2-D foldover diagnostic (pre-projection)
            if plot_flag > 0 and inner_border_verts is not None:
                n2d_fold = 0
                for fi in range(len(minpatm.F)):
                    i0, i1, i2 = minpatm.F[fi]
                    cross2d = ((u_sol[i1] - u_sol[i0]) * (v_sol[i2] - v_sol[i0])
                               - (v_sol[i1] - v_sol[i0]) * (u_sol[i2] - u_sol[i0]))
                    if cross2d < 0:
                        n2d_fold += 1
                n2d_pos = len(minpatm.F) - n2d_fold
                print(f'  Stereo 2-D pre-projection: {n2d_fold} neg orient, '
                      f'{n2d_pos} pos orient out of {len(minpatm.F)} faces')
                # Check boundary nesting and convexity
                inner_border_set_chk = set(inner_border_verts) if inner_border_verts is not None else set()
                inner_2d = [(u_sol[fi], v_sol[fi]) for fi in fixed_ol
                            if int(uv[fi]) in inner_border_set_chk]
                outer_2d = [(u_sol[fi], v_sol[fi]) for fi in fixed_ol
                            if int(uv[fi]) not in inner_border_set_chk]
                if inner_2d and outer_2d:
                    inner_r = [np.sqrt(u**2 + v**2) for u, v in inner_2d]
                    outer_r = [np.sqrt(u**2 + v**2) for u, v in outer_2d]
                    print(f'  Inner ring r: [{min(inner_r):.4f}, {max(inner_r):.4f}], '
                          f'Outer ring r: [{min(outer_r):.4f}, {max(outer_r):.4f}]')
                    if max(inner_r) >= min(outer_r):
                        print(f'  *** WARNING: Inner ring OVERLAPS outer ring in 2-D! ***')
            # e) Inverse-project back to sphere
            uv_2d = np.column_stack([u_sol, v_sol])
            pts_3d = _stereo_unproject(uv_2d, R)
            x_sol = pts_3d[:, 0]
            y_sol = pts_3d[:, 1]
            z_sol = pts_3d[:, 2]

            if _convexified_annular:
                # Smooth boundary correction: solve Laplace for the
                # displacement field between convexified and true boundary
                # positions.  This propagates the snap-back smoothly into
                # the interior instead of creating a discontinuity at the
                # boundary-interior interface.
                bdx = np.zeros(nv)
                bdy = np.zeros(nv)
                bdz = np.zeros(nv)
                bdx[fixed_ol] = bnd_xyz_original[:, 0] - x_sol[fixed_ol]
                bdy[fixed_ol] = bnd_xyz_original[:, 1] - y_sol[fixed_ol]
                bdz[fixed_ol] = bnd_xyz_original[:, 2] - z_sol[fixed_ol]

                try:
                    cx = spsolve(A_csr, bdx)
                    cy = spsolve(A_csr, bdy)
                    cz = spsolve(A_csr, bdz)
                    x_sol += cx
                    y_sol += cy
                    z_sol += cz
                except Exception:
                    x_sol[fixed_ol] = bnd_xyz_original[:, 0]
                    y_sol[fixed_ol] = bnd_xyz_original[:, 1]
                    z_sol[fixed_ol] = bnd_xyz_original[:, 2]

            t_sol, p_sol, _ = kk_cart2sph(x_sol, y_sol, z_sol)
            x_sol, y_sol, z_sol = kk_sph2cart(t_sol, p_sol, np.ones_like(t_sol))

    if (not solve_spherical) and (not use_stereo):
        # ----------------------------------------------------------
        # Original 3-D Cartesian Laplace solver
        # ----------------------------------------------------------
        bx = np.zeros(nv)
        by = np.zeros(nv)
        bz = np.zeros(nv)
        bx[fixed_ol] = x[fixed_ol]
        by[fixed_ol] = y[fixed_ol]
        bz[fixed_ol] = z[fixed_ol]

        try:
            x_sol = spsolve(A_csr, bx)
            y_sol = spsolve(A_csr, by)
            z_sol = spsolve(A_csr, bz)
        except Exception as e:
            if plot_flag > 0:
                print(f'  Warning: Solver failed: {e}, using boundary values only')
            x_sol = x.copy()
            y_sol = y.copy()
            z_sol = z.copy()

        # Project on sphere
        t_sol, p_sol, _ = kk_cart2sph(x_sol, y_sol, z_sol)
        x_sol, y_sol, z_sol = kk_sph2cart(t_sol, p_sol, np.ones_like(t_sol))
    
    # ------------------------------------------------------------------
    # Orient check: if the parameterization is mostly inward-facing on
    # the unit sphere, flip the face winding so normals point outward.
    #
    # We use TWO criteria to decide whether to flip:
    #   (a) Classic: n_neg > n_pos  (majority of faces have orient < 0)
    #   (b) Collapsed: max(orient) <= 0  (every face is inward or
    #       degenerate — this catches patches like Patch 7 where many
    #       faces have orient ≈ 0 but none are truly outward)
    # ------------------------------------------------------------------
    X_sph = np.column_stack([x_sol, y_sol, z_sol])
    orients_check = np.zeros(len(minpatm.F))
    for fi in range(len(minpatm.F)):
        i0, i1, i2 = minpatm.F[fi]
        v0, v1, v2 = X_sph[i0], X_sph[i1], X_sph[i2]
        cross_vec = np.cross(v1 - v0, v2 - v0)
        centroid = (v0 + v1 + v2) / 3.0
        orients_check[fi] = np.dot(centroid, cross_vec)

    n_pos = int(np.sum(orients_check > 0))
    n_neg = int(np.sum(orients_check < 0))
    max_orient = float(orients_check.max()) if len(orients_check) > 0 else 0.0

    need_flip = (n_neg > n_pos) or (max_orient <= 0 and n_neg > 0)
    if need_flip:
        # Flip face winding: swap vertex columns 1 and 2
        minpatm.F = minpatm.F[:, [0, 2, 1]]
        patm.F = patm.F[:, [0, 2, 1]]        # keep full mesh in sync
        if plot_flag > 0:
            print(f'  Orient fix: flipped face winding '
                  f'(pos={n_pos}, neg={n_neg}, max_orient={max_orient:.6f})')

    # ------------------------------------------------------------------
    # Iterative foldover removal
    #
    # Strategy: Jacobi-style Laplace smoothing on an expanding ring of
    # free vertices around foldover faces.  Each iteration:
    #   1. Detect foldover faces (orient < 0).
    #   2. Collect free vertices of foldover faces AND their 1-ring
    #      neighbours (expanding the repair zone).
    #   3. Compute ALL new positions as neighbour-averages (Jacobi),
    #      then apply with damping  new = (1-alpha)*old + alpha*avg.
    #   4. Re-project every moved vertex onto the unit sphere.
    # Stops when 0 foldovers or stall or MAX iterations.
    # ------------------------------------------------------------------
    MAX_FOLD_ITERS = 500
    DAMP = 0.6          # blending factor (0 = no change, 1 = full avg)
    X_sph = np.column_stack([x_sol, y_sol, z_sol])
    fixed_set = set(fixed_ol.tolist())
    nv_min = len(X_sph)

    prev_n_fold = None
    stall_count = 0

    for fold_iter in range(MAX_FOLD_ITERS):
        # --- detect foldovers ---
        orients = np.zeros(len(minpatm.F))
        for fi in range(len(minpatm.F)):
            i0, i1, i2 = minpatm.F[fi]
            v0, v1, v2 = X_sph[i0], X_sph[i1], X_sph[i2]
            cross_vec = np.cross(v1 - v0, v2 - v0)
            cent = (v0 + v1 + v2) / 3.0
            orients[fi] = np.dot(cent, cross_vec)

        fold_faces = np.where(orients < -1e-15)[0]
        n_fold = len(fold_faces)
        if n_fold == 0:
            if fold_iter > 0 and plot_flag > 0:
                print(f'  Foldover removal: converged after {fold_iter} iterations')
            break

        # stall detection
        if prev_n_fold is not None and n_fold >= prev_n_fold:
            stall_count += 1
            if stall_count > 30:
                break          # not converging
        else:
            stall_count = 0
        prev_n_fold = n_fold

        # --- collect vertices to smooth (foldover faces + 1-ring) ---
        target_verts = set()
        for fi in fold_faces:
            for vi in minpatm.F[fi]:
                target_verts.add(int(vi))
        # expand by one ring
        expanded = set()
        for vi in target_verts:
            for nb in L.get(vi, []):
                expanded.add(int(nb))
        target_verts |= expanded
        # remove fixed (boundary) vertices
        target_verts -= fixed_set

        if len(target_verts) == 0:
            break

        # --- Jacobi smoothing: compute new positions first ---
        # Use Cartesian averaging for local Jacobi smoothing — it works
        # well because each vertex only averages with its immediate
        # neighbors (no full-ring cancellation).  Spherical averaging is
        # reserved for the global Laplace solve (solve_spherical path).
        new_pos = {}
        t_sph = np.empty(nv_min)
        p_sph = np.empty(nv_min)
        t_sph[:], p_sph[:], _ = kk_cart2sph(
            X_sph[:, 0], X_sph[:, 1], X_sph[:, 2])

        for vi in target_verts:
            nbrs = L.get(vi, [])
            if len(nbrs) == 0:
                continue
            if solve_spherical:
                # Spherical averaging: theta interpolates, phi via cos/sin
                t_nbr = t_sph[nbrs]
                p_nbr = p_sph[nbrs]
                t_avg = float(np.mean(t_nbr))
                cos_avg = float(np.mean(np.cos(p_nbr)))
                sin_avg = float(np.mean(np.sin(p_nbr)))
                p_avg = np.arctan2(sin_avg, cos_avg)
                ux, uy, uz = kk_sph2cart(
                    np.array([t_avg]), np.array([p_avg]), np.array([1.0]))
                avg_on_sphere = np.array([ux[0], uy[0], uz[0]])
            else:
                nbr_pts = X_sph[np.array(nbrs)]
                avg = nbr_pts.mean(axis=0)
                norm_avg = np.linalg.norm(avg)
                if norm_avg > 1e-15:
                    avg_on_sphere = avg / norm_avg
                else:
                    continue
            # damped blend
            blended = (1.0 - DAMP) * X_sph[vi] + DAMP * avg_on_sphere
            bn = np.linalg.norm(blended)
            if bn > 1e-15:
                new_pos[vi] = blended / bn
            else:
                new_pos[vi] = avg_on_sphere

        # --- apply all updates at once ---
        for vi, pos in new_pos.items():
            X_sph[vi] = pos

        # NOTE: theta band clamping inside the foldover loop was removed.
        # It was counterproductive for annular patches — the clamping fought
        # with Jacobi smoothing, creating new foldovers (e.g. 29 → 47).
        # The final theta band clamp after the loop is sufficient.

    # Report if foldovers remain
    if n_fold > 0 and plot_flag > 0:
        reason = 'max iterations' if fold_iter >= MAX_FOLD_ITERS - 1 else f'stalled after {fold_iter} iters'
        print(f'  Foldover removal: {n_fold} foldovers remain ({reason})')

    # Write back from X_sph to solution arrays
    x_sol = X_sph[:, 0]
    y_sol = X_sph[:, 1]
    z_sol = X_sph[:, 2]
    t_sol, p_sol, _ = kk_cart2sph(x_sol, y_sol, z_sol)
    x_sol, y_sol, z_sol = kk_sph2cart(t_sol, p_sol, np.ones_like(t_sol))

    # Final theta band check for annular patches.
    #
    # The Laplace solution (maximum principle) guarantees interior theta
    # values stay within the boundary range.  However, foldover removal
    # uses Cartesian Jacobi smoothing + sphere projection, which can
    # push theta slightly outside the band.
    #
    # Aggressive clamping (clip theta, recompute Cartesian) was found to
    # CREATE new foldovers (e.g. 30 → 48) because it moves vertices
    # without accounting for the neighbouring face geometry.
    #
    # Strategy: only clamp if it does NOT increase the foldover count.
    if inner_border_verts is not None:
        t_bnd = t_sol[fixed_ol]
        t_lo, t_hi = float(t_bnd.min()), float(t_bnd.max())
        free_mask = ~fixed_patm_reduced
        if np.any(free_mask):
            t_free = t_sol[free_mask]
            n_below = int(np.sum(t_free < t_lo))
            n_above = int(np.sum(t_free > t_hi))
            if plot_flag > 0:
                print(f'  Final band check: theta=[{t_lo:.4f}, {t_hi:.4f}], '
                      f'{n_below} below, {n_above} above')
            if n_below + n_above > 0:
                # Try clamping, count foldovers, revert if worse
                t_clamped = t_sol.copy()
                t_clamped[free_mask] = np.clip(t_clamped[free_mask], t_lo, t_hi)
                xc, yc, zc = kk_sph2cart(
                    t_clamped, p_sol, np.ones_like(t_clamped))
                Xc = np.column_stack([xc, yc, zc])
                n_fold_clamped = 0
                for fi in range(len(minpatm.F)):
                    i0, i1, i2 = minpatm.F[fi]
                    cross_v = np.cross(Xc[i1] - Xc[i0], Xc[i2] - Xc[i0])
                    cent = (Xc[i0] + Xc[i1] + Xc[i2]) / 3.0
                    if np.dot(cent, cross_v) < -1e-15:
                        n_fold_clamped += 1
                n_fold_current = 0
                X_cur = np.column_stack([x_sol, y_sol, z_sol])
                for fi in range(len(minpatm.F)):
                    i0, i1, i2 = minpatm.F[fi]
                    cross_v = np.cross(X_cur[i1] - X_cur[i0],
                                       X_cur[i2] - X_cur[i0])
                    cent = (X_cur[i0] + X_cur[i1] + X_cur[i2]) / 3.0
                    if np.dot(cent, cross_v) < -1e-15:
                        n_fold_current += 1
                if n_fold_clamped <= n_fold_current:
                    t_sol = t_clamped
                    x_sol, y_sol, z_sol = xc, yc, zc
                    if plot_flag > 0:
                        print(f'  Applied band clamp (foldovers {n_fold_current}'
                              f' -> {n_fold_clamped})')
                else:
                    if plot_flag > 0:
                        print(f'  Skipped band clamp (would increase foldovers '
                              f'{n_fold_current} -> {n_fold_clamped})')

    # Update minimal patch
    minpatm.t = t_sol
    minpatm.p = p_sol
    
    return (minpatm, uv, x_sol, y_sol, z_sol, t_sol, p_sol, fixed_patm_reduced)


def parameterize_single_patch(patm, boundary_t, boundary_p, boundary_indices):
    """
    Parameterize a single patch with given boundary conditions.
    
    This is a simpler interface that takes spherical coordinates directly.
    
    Parameters:
    -----------
    patm : surface_mesh
        Patch mesh
    boundary_t, boundary_p : array
        Boundary theta and phi values
    boundary_indices : array
        Indices of boundary vertices
        
    Returns:
    --------
    patm : surface_mesh
        Parameterized patch
    """
    # Initialize
    patm.t = np.zeros(len(patm.X))
    patm.p = np.zeros(len(patm.X))
    
    # Set boundary values
    patm.t[boundary_indices] = boundary_t
    patm.p[boundary_indices] = boundary_p
    
    # Reduce to minimal set
    minpatm, uv = reduce_to_minimal_set(patm)
    
    # Build Laplacian (diagonal = valid neighbour count, not total link count)
    L = minpatm.L
    nv = len(L)
    A = lil_matrix((nv, nv))
    
    for iv in range(nv):
        links = L.get(iv, [])
        valid_count = 0
        for link in links:
            if link < nv:
                A[iv, link] = -1
                valid_count += 1
        A[iv, iv] = valid_count
    
    # Map boundary indices to reduced mesh
    fixed_mask = np.zeros(nv, dtype=bool)
    for orig_idx in boundary_indices:
        reduced_idx = np.where(uv == orig_idx)[0]
        if len(reduced_idx) > 0:
            fixed_mask[reduced_idx[0]] = True
    
    fixed_ol = np.where(fixed_mask)[0]
    
    # Apply boundary conditions
    for idx in fixed_ol:
        A[idx, :] = 0
        A[idx, idx] = 1
    
    # Solve for Cartesian coordinates
    u, v, w = kk_sph2cart(minpatm.t, minpatm.p, np.ones(nv))
    
    bx = np.zeros(nv)
    by = np.zeros(nv)
    bz = np.zeros(nv)
    bx[fixed_ol] = u[fixed_ol]
    by[fixed_ol] = v[fixed_ol]
    bz[fixed_ol] = w[fixed_ol]
    
    A_csr = csr_matrix(A)
    x_sol = spsolve(A_csr, bx)
    y_sol = spsolve(A_csr, by)
    z_sol = spsolve(A_csr, bz)
    
    # Convert back to spherical
    t_sol, p_sol, _ = kk_cart2sph(x_sol, y_sol, z_sol)
    
    # Update
    minpatm.t = t_sol
    minpatm.p = p_sol
    
    # Transfer back to full patch
    patm.t[uv] = minpatm.t
    patm.p[uv] = minpatm.p
    
    return patm
