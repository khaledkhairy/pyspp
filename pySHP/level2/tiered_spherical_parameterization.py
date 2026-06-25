"""
Tiered (coarse-to-fine) curvature-aware spherical parameterization on a
*patch* foundation.

Why patches
-----------
A global Brechbuhler ``map2sphere`` is not bijective for shapes with thin
features (e.g. a mushroom stalk): on a curvature-decimated coarse mesh the raw
map already folds over the majority of faces, and area/shear Newton cannot
untangle it. The robust foundation is therefore the patch / simplified-cage
pipeline: the mesh is segmented into well-behaved (disk-like) patches; a coarse
*cage* (keys + centres) is mapped to the sphere -- it is structured enough to be
bijective -- and each fine patch interior is then filled by a Laplace
(Dirichlet) solve whose boundary is read off the cage. Each patch is a harmonic
map into a convex region, so interiors stay foldover-free.

Tier structure (what "tiered" means here)
-----------------------------------------
  Tier 0  : the simplified cage on the sphere (coarsest shape).
  Tier k  : the assembled fine mesh decimated to resolution level k *with every
            patch-boundary vertex protected*, so patch boundaries are identical
            across tiers and the mesh stays watertight. Each patch interior is
            re-filled by a Laplace solve at that resolution (boundary read from
            the authoritative full-resolution parameterization), then the whole
            tier is polished on the sphere (area/shear Newton with flip
            prevention). The finest tier is the full assembled mesh.

Every tier reports global + per-patch diagnostics (bijectivity, foldovers,
curvature/area correlation, parametric triangle quality) and a fitted
spherical-harmonics surface -- so problem regions surface progressively instead
of only after a single full-resolution attempt.

Stages
  0. Preprocess: repair + curvature-adaptive remesh -> fine mesh
  1. Patch structure: validated segmentation + simplified cage (PM)
  2. Tier 0: cage map2sphere (area -> edge/shear)
  3. Full patch fill (parameterize_patches_cart) -> authoritative (t, p)
  4. Interior tiers: boundary-protected decimation + per-patch Laplace re-fill
     + sphere polish
  5. Per-tier diagnostics + SHP projection

Public API
  preprocess_mesh
  build_patch_structure
  parameterize_cage
  fill_full_patches
  build_interior_tiers
  refill_patches_on_mesh / optimize_tier_sphere
  compute_tier_diagnostics / fit_tier_shp
  run_tiered_pipeline
"""

import numpy as np

from ..surface_mesh import surface_mesh
from ..shp_surface import shp_surface
from ..level1.target_areas import compute_curvature_target_areas
from ..level1.bijectivity_gate import (
    check_bijectivity_gate,
    compute_achieved_spherical_areas,
    compute_parametric_quality,
)


# --------------------------------------------------------------------------- #
# Tier container
# --------------------------------------------------------------------------- #
class Tier:
    """One resolution level of the parameterization.

    Attributes
    ----------
    level : int
        0 = coarsest (the cage), increasing toward the full mesh.
    kind : str
        ``'cage'`` for tier 0, ``'interior'`` for the patch-interior tiers.
    mesh : surface_mesh
        Mesh at this resolution carrying ``.t`` and ``.p`` (and ``.face_labels``
        for interior tiers).
    diagnostics : dict
        Filled by :func:`compute_tier_diagnostics`.
    shp : shp_surface or None
    shp_rms : float or None
    """

    def __init__(self, level, mesh, kind='interior'):
        self.level = int(level)
        self.kind = kind
        self.mesh = mesh
        self.diagnostics = {}
        self.shp = None
        self.shp_rms = None

    @property
    def n_verts(self):
        return len(self.mesh.X)

    @property
    def n_faces(self):
        return len(self.mesh.F)

    def __repr__(self):
        return (f"Tier(level={self.level}, kind={self.kind}, "
                f"verts={self.n_verts}, faces={self.n_faces})")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _copy_mesh(m):
    mc = surface_mesh(m.X.copy(), m.F.copy())
    if getattr(m, 'face_labels', None) is not None:
        mc.face_labels = np.asarray(m.face_labels).copy()
    return mc


def _default_weights():
    return dict(lambdaA=1.0, lambda_flip=1e4, lambda1=1e-4, lambda2=1e-2)


def _area_correlation(Ao, achieved_abs):
    Ao = np.asarray(Ao, dtype=float)
    achieved_abs = np.asarray(achieved_abs, dtype=float)
    if len(Ao) < 2 or np.std(Ao) < 1e-15 or np.std(achieved_abs) < 1e-15:
        return 0.0
    return float(np.corrcoef(Ao, achieved_abs)[0, 1])


def _patch_boundary_mask(F, labels):
    """Vertex is on a patch boundary iff its incident faces span >1 label."""
    F = np.asarray(F, dtype=int)
    labels = np.asarray(labels, dtype=int)
    nverts = int(F.max()) + 1 if len(F) else 0
    seen = [None] * nverts
    multi = np.zeros(nverts, dtype=bool)
    for fi in range(len(F)):
        lab = labels[fi]
        for vi in F[fi]:
            vi = int(vi)
            if seen[vi] is None:
                seen[vi] = lab
            elif seen[vi] != lab:
                multi[vi] = True
    return multi


def _extract_submesh(m, face_mask):
    """Return (patm, used) where ``used`` maps local vertex idx -> global idx."""
    F = np.asarray(m.F, dtype=int)[face_mask]
    used = np.unique(F)
    remap = -np.ones(len(m.X), dtype=int)
    remap[used] = np.arange(len(used))
    Floc = remap[F]
    patm = surface_mesh(m.X[used].copy(), Floc.copy())
    return patm, used


# --------------------------------------------------------------------------- #
# Stage 0: preprocessing
# --------------------------------------------------------------------------- #
def preprocess_mesh(mesh, target_verts=3000, curvature_strength=2.0,
                    keep_largest=True, verbose=True):
    """Repair and curvature-adaptively remesh an arbitrary genus-0 mesh.

    Produces the finest working mesh: high quality, dense in high-curvature
    regions and coarse in flat regions. Because triangle density tracks
    curvature, an (approximately) equal-area spherical map automatically gives
    each region a sphere area proportional to its curvature content -- goal (a).
    """
    m = _copy_mesh(mesh)
    m.repair_mesh(verbose=verbose)
    if keep_largest:
        m.keep_largest_surface(verbose=False)
    m.props()

    target_faces = int(round(2 * target_verts))  # closed genus-0: F ~ 2V
    if verbose:
        print(f"  remesh_by_curvature -> ~{target_faces} faces "
              f"(curvature_strength={curvature_strength})")
    m.remesh_by_curvature(target_faces=target_faces,
                          curvature_strength=curvature_strength,
                          verbose=verbose)
    m.props()
    stats = m.get_mesh_quality_stats()
    if verbose:
        print(f"  preprocessed: {len(m.X)} verts, {len(m.F)} faces, "
              f"mean quality {stats['mean_quality']:.3f}, "
              f"min quality {stats['min_quality']:.3f}")
    return m, stats


# --------------------------------------------------------------------------- #
# Stage 1: patch structure (validated segmentation + simplified cage)
# --------------------------------------------------------------------------- #
# ..... patch classification / rescue-by-subdivision .......................... #
def _relabel_contiguous(labels):
    """Map arbitrary integer labels to a contiguous 0..K-1 range."""
    labels = np.asarray(labels, dtype=int)
    uL = np.unique(labels)
    remap = {int(l): i for i, l in enumerate(uL)}
    return np.array([remap[int(l)] for l in labels], dtype=int)


def _patch_neighbor_counts(m, labels):
    """Vertex-based neighbour count per patch (sorted-unique-label order)."""
    from ..level1.find_valid_segmentation import (
        compute_vertex_based_patch_connectivity,
    )
    ms = surface_mesh(m.X, m.F)
    ms.face_labels = np.asarray(labels, dtype=int)
    Pconn = compute_vertex_based_patch_connectivity(ms)
    uL = np.unique(labels)
    counts = np.asarray((Pconn > 0).sum(axis=1)).ravel()
    return uL, counts


def _patch_boundary_components(m, labels, lab):
    """Number of boundary loops of patch ``lab`` (1 = disk, 2 = annulus)."""
    from collections import defaultdict
    F = np.asarray(m.F, dtype=int)
    Fp = F[np.asarray(labels) == lab]
    if len(Fp) == 0:
        return 0
    ecount = defaultdict(int)
    for f in Fp:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            ecount[(min(a, b), max(a, b))] += 1
    border = [e for e, c in ecount.items() if c == 1]
    if not border:
        return 0
    adj = defaultdict(list)
    verts = set()
    for a, b in border:
        adj[a].append(b)
        adj[b].append(a)
        verts.add(a)
        verts.add(b)
    seen = set()
    comps = 0
    for v in verts:
        if v in seen:
            continue
        comps += 1
        stack = [v]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(y for y in adj[x] if y not in seen)
    return comps


def classify_patches(m, labels, min_neighbors=3):
    """Classify patches; flag caps (< min_neighbors) and annuli (>1 boundary loop).

    Returns
    -------
    info : dict[label -> dict]
        ``neighbors``, ``boundary_components``, ``is_cap``, ``is_annular``, ``bad``.
    problematic : list[int]
        Labels that are caps or annular.
    """
    uL, counts = _patch_neighbor_counts(m, labels)
    info = {}
    problematic = []
    for i, lab in enumerate(uL):
        nbr = int(counts[i])
        ncomp = _patch_boundary_components(m, labels, lab)
        is_cap = nbr < min_neighbors
        is_annular = ncomp != 1
        bad = is_cap or is_annular
        info[int(lab)] = dict(neighbors=nbr, boundary_components=ncomp,
                              is_cap=is_cap, is_annular=is_annular, bad=bad)
        if bad:
            problematic.append(int(lab))
    return info, problematic


def _split_patch(m, labels, lab, n_split, sig=1.0, curvature_weight=0.0):
    """Split patch ``lab`` into <= n_split connected sub-patches (random walk)."""
    from ..level1.mesh_segmentation_rw import mesh_segmentation_rw

    labels = np.asarray(labels, dtype=int)
    fidx = np.where(labels == lab)[0]
    if len(fidx) < max(2 * n_split, 6):
        return labels  # too small to split usefully
    F = np.asarray(m.F, dtype=int)[fidx]
    used = np.unique(F)
    remap = -np.ones(int(m.F.max()) + 1, dtype=int)
    remap[used] = np.arange(len(used))
    sub = surface_mesh(m.X[used].copy(), remap[F].copy())
    try:
        _, subL, _slix, _P, _Pc = mesh_segmentation_rw(
            sub, n_split, sig=sig, curvature_weight=curvature_weight,
            verbose=False)
    except Exception:
        return labels
    subL = np.asarray(subL, dtype=int)
    new_labels = labels.copy()
    max_lab = int(labels.max())
    for j, s in enumerate(np.unique(subL)):
        sel = fidx[subL == s]
        if j == 0:
            new_labels[sel] = lab
        else:
            max_lab += 1
            new_labels[sel] = max_lab
    return new_labels


# ..... curvature-aware seeding ............................................... #
def _face_centroids(m):
    F = np.asarray(m.F, dtype=int)
    return m.X[F].mean(axis=1)


def _face_curvature(m):
    """Per-face |mean curvature| (from vertex H, populated by props())."""
    if getattr(m, 'H', None) is None or m.needs_updating:
        m.props()
    H = np.abs(np.asarray(m.H, dtype=float))
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    return H[np.asarray(m.F, dtype=int)].mean(axis=1)


def _curvature_importance(m, curv_alpha):
    """Per-face importance weight ``1 + curv_alpha * normalised_curvature``."""
    fc = _face_curvature(m)
    rng = fc.max() - fc.min()
    cn = (fc - fc.min()) / rng if rng > 1e-15 else np.zeros_like(fc)
    return 1.0 + float(curv_alpha) * cn, fc


def curvature_biased_seed_faces(m, nseeds, curv_alpha=3.0, seed_faces=None,
                                curv_alpha_first=True):
    """Density-weighted farthest-point seed faces, packed in high-curvature areas.

    Greedy blue-noise sampling on face centroids where the effective spacing is
    *smaller* where curvature is high: ``score = min_dist_to_seeds * importance``
    with ``importance = 1 + curv_alpha * normalised_curvature``. This places more
    seeds on thin / detailed regions (e.g. the stalk), which is exactly where
    uniform farthest-point sampling under-seeds and produces caps / annuli.

    Parameters
    ----------
    seed_faces : list[int] or None
        Optional faces to seed the greedy process with (kept in the result).
    curv_alpha_first : bool
        If True (and no ``seed_faces``), the first seed is the highest-curvature
        face; otherwise the geometric extreme.

    Returns
    -------
    slix : ndarray[int]
        ``nseeds`` seed face indices.
    """
    cen = _face_centroids(m)
    w, fc = _curvature_importance(m, curv_alpha)
    nF = len(cen)
    nseeds = int(min(nseeds, nF))

    chosen = list(seed_faces) if seed_faces else []
    if not chosen:
        first = int(np.argmax(fc)) if curv_alpha_first else int(
            np.argmax(np.linalg.norm(cen - cen.mean(0), axis=1)))
        chosen.append(first)

    dmin = np.full(nF, np.inf)
    for c in chosen:
        dmin = np.minimum(dmin, np.linalg.norm(cen - cen[c], axis=1))
    while len(chosen) < nseeds:
        score = dmin * w
        score[chosen] = -1.0
        nxt = int(np.argmax(score))
        if score[nxt] <= 0:
            break
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(cen - cen[nxt], axis=1))
    return np.asarray(chosen, dtype=int)


def _seeds_inside_patch(m, labels, lab, slix, n_add, curv_alpha=3.0):
    """Pick ``n_add`` new seed faces inside patch ``lab`` (look-ahead re-seeding).

    Chosen far from existing seeds and biased to high curvature, so a cap/annular
    patch gets denser seeding exactly where it is under-resolved.
    """
    cen = _face_centroids(m)
    w, _fc = _curvature_importance(m, curv_alpha)
    fidx = np.where(np.asarray(labels) == lab)[0]
    if len(fidx) == 0:
        return []
    dmin = np.full(len(fidx), np.inf)
    for s in slix:
        dmin = np.minimum(dmin, np.linalg.norm(cen[fidx] - cen[s], axis=1))
    added = []
    for _ in range(int(n_add)):
        score = dmin * w[fidx]
        for a in added:
            score[a] = -1.0
        loc = int(np.argmax(score))
        if score[loc] <= 0:
            break
        added.append(loc)
        dmin = np.minimum(dmin, np.linalg.norm(cen[fidx] - cen[fidx[loc]], axis=1))
    return [int(fidx[a]) for a in added]


def generate_tier0_patches(m, nseeds_range=(3, 30), min_neighbors=3,
                           n_split=5, max_rescue_rounds=2, sig=1.0,
                           curvature_weight=0.0, seeding='curvature',
                           curv_alpha=3.0, rescue='reseed', reseed_add=2,
                           reseed_rounds=6, verbose=True):
    """Produce a tier-0 segmentation with **no cap and no annular patches**.

    Strategy (the seeding-first approach):

    * ``seeding='curvature'`` places seeds with a curvature-biased density
      (more seeds on thin / high-curvature regions such as the stalk), which is
      where uniform farthest-point sampling under-seeds and creates caps / annuli.
    * Step [2]: sweep ``nseeds`` over the user range; return the first labelling
      where every patch is a regular disk.
    * Step [3] rescue (only if the whole range fails):
        - ``rescue='reseed'`` (default): **look-ahead re-seeding** -- add
          ``reseed_add`` curvature-biased seeds *inside* each cap/annular patch
          and re-segment, up to ``reseed_rounds`` times. Keeps patches naturally
          shaped (no arbitrary cuts).
        - ``rescue='subdivide'``: split each problematic patch into ``n_split``
          random-walk sub-patches (legacy fallback).

    Returns
    -------
    labels : ndarray
        Contiguous face labels (0..K-1), all patches regular disks.
    meta : dict
        ``nseeds``, ``rescued``, ``n_patches``, ``info``, ``seeding``, ``slix``.

    Raises
    ------
    RuntimeError
        If no well-behaved labelling is found (fail early).
    """
    from ..level1.mesh_segmentation_rw import mesh_segmentation_rw

    if isinstance(nseeds_range, (tuple,)) and len(nseeds_range) == 2:
        seeds = list(range(int(nseeds_range[0]), int(nseeds_range[1]) + 1))
    else:
        seeds = list(nseeds_range)

    def _segment(nseeds, slix=None):
        ms_, _L, slix_out, _P, _Pc = mesh_segmentation_rw(
            surface_mesh(m.X.copy(), m.F.copy()), nseeds, sig=sig, slix=slix,
            curvature_weight=curvature_weight, verbose=False)
        return _relabel_contiguous(ms_.face_labels), np.asarray(slix_out)

    def _seeds_for(nseeds):
        if seeding == 'curvature':
            return curvature_biased_seed_faces(m, nseeds, curv_alpha=curv_alpha)
        return None  # uniform farthest-point (core default)

    # ---- Pass 1 (step [2]): try the full range, no rescue. ---------------- #
    seed_cache = {}
    best = None  # (n_problematic, labels, slix, meta)
    for nseeds in seeds:
        try:
            slix = _seeds_for(nseeds)
            labels, slix = _segment(nseeds, slix=slix)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  nseeds={nseeds}: segmentation failed ({exc})")
            continue
        seed_cache[nseeds] = (labels, slix)
        info, problematic = classify_patches(m, labels, min_neighbors)
        if verbose:
            print(f"  nseeds={nseeds} [{seeding}]: {len(np.unique(labels))} "
                  f"patches, {len(problematic)} problematic "
                  f"(caps/annuli)={problematic if problematic else ''}")
        if not problematic:
            return labels, dict(nseeds=nseeds, rescued=False,
                                n_patches=len(np.unique(labels)), info=info,
                                seeding=seeding, slix=slix)
        if best is None or len(problematic) < best[0]:
            best = (len(problematic), labels, slix,
                    dict(nseeds=nseeds, info=info, problematic=problematic))

    # ---- Pass 2 (step [3]): rescue only after the whole range failed. ----- #
    if rescue == 'reseed' and reseed_rounds:
        if verbose:
            print(f"  no clean raw segmentation in {seeds}; look-ahead "
                  f"re-seeding (add {reseed_add}/patch, {reseed_rounds} rounds)")
        # Start from the best raw attempt and progressively densify its seeding.
        nseeds0 = best[3]['nseeds']
        slix = list(best[2])
        labels = best[1]
        for rnd in range(reseed_rounds):
            _info_r, prob_r = classify_patches(m, labels, min_neighbors)
            if not prob_r:
                break
            new_seeds = []
            for lab in prob_r:
                new_seeds += _seeds_inside_patch(m, labels, lab, slix,
                                                 reseed_add, curv_alpha=curv_alpha)
            new_seeds = [s for s in new_seeds if s not in slix]
            if not new_seeds:
                break
            slix = slix + new_seeds
            labels, slix_arr = _segment(len(slix), slix=np.asarray(slix))
            slix = list(slix_arr)
            info_r, prob_r = classify_patches(m, labels, min_neighbors)
            if verbose:
                print(f"    reseed round {rnd + 1}: +{len(new_seeds)} seeds "
                      f"-> {len(np.unique(labels))} patches, "
                      f"{len(prob_r)} problematic")
            if not prob_r:
                return labels, dict(nseeds=len(slix), rescued=True,
                                    n_patches=len(np.unique(labels)),
                                    info=info_r, seeding=seeding, slix=slix)
            if len(prob_r) < best[0]:
                best = (len(prob_r), labels, np.asarray(slix),
                        dict(nseeds=len(slix), info=info_r, problematic=prob_r))

    elif rescue == 'subdivide' and n_split and max_rescue_rounds:
        if verbose:
            print(f"  no clean raw segmentation in {seeds}; subdivide rescue "
                  f"(n_split={n_split}, rounds={max_rescue_rounds})")
        for nseeds in seeds:
            cached = seed_cache.get(nseeds)
            if cached is None:
                continue
            rl = cached[0].copy()
            for rnd in range(max_rescue_rounds):
                _info_r, prob_r = classify_patches(m, rl, min_neighbors)
                if not prob_r:
                    break
                for lab in prob_r:
                    rl = _split_patch(m, rl, lab, n_split, sig=sig,
                                      curvature_weight=curvature_weight)
                rl = _relabel_contiguous(rl)
            info_r, prob_r = classify_patches(m, rl, min_neighbors)
            if not prob_r:
                return rl, dict(nseeds=nseeds, rescued=True,
                                n_patches=len(np.unique(rl)), info=info_r,
                                seeding=seeding)
            if best is None or len(prob_r) < best[0]:
                best = (len(prob_r), rl, None,
                        dict(nseeds=nseeds, info=info_r, problematic=prob_r))

    msg = ("Could not produce a no-cap/no-annular tier-0 segmentation in "
           f"nseeds range {seeds}.")
    if best is not None:
        msg += (f" Best attempt: nseeds={best[3]['nseeds']}, "
                f"{best[0]} problematic patch(es) remain "
                f"{best[3].get('problematic')}. Try a wider range, higher "
                "curv_alpha, more reseed_rounds, or denser preprocessing.")
    raise RuntimeError(msg)


def build_cage_from_labels(m, labels, target_ratio=0.25, curvature_weight=1.0,
                           verbose=True):
    """Build PM + decimated cage from an explicit face labelling.

    Returns ``(m_seg, PM)``.
    """
    from ..level1.patch_info_gen import patch_info_gen
    from ..level1.build_pm_from_decimated_mesh import build_pm_from_decimated_mesh

    ms = surface_mesh(m.X.copy(), m.F.copy())
    ms.face_labels = np.asarray(labels, dtype=int).copy()
    m_seg, PM, _Pconn = patch_info_gen(ms, validate_segmentation=False)
    build_pm_from_decimated_mesh(m_seg, PM, target_ratio=target_ratio,
                                 curvature_weight=curvature_weight,
                                 verbose=verbose)
    return m_seg, PM


def build_patch_structure(m_fine, nseeds_range=(3, 30), min_neighbors=3,
                          n_split=5, max_rescue_rounds=2, sig=1.0,
                          curvature_weight_seg=0.0, curvature_weight_dec=1.0,
                          target_ratio=0.25, seeding='curvature', curv_alpha=3.0,
                          rescue='reseed', reseed_add=2, reseed_rounds=6,
                          verbose=True):
    """Tier-0 patch structure with **no cap and no annular patches**.

    Generates a well-behaved labelling via :func:`generate_tier0_patches`
    (curvature-biased seeding + look-ahead re-seeding rescue), then builds the
    decimated cage. Fails early (raises) if no well-behaved segmentation exists.

    Returns
    -------
    m_seg : surface_mesh
        Fine mesh carrying ``face_labels``.
    PM : dict
        Patch-mesh structure (``PM['pm']`` is the decimated cage).
    meta : dict
        Segmentation metadata from :func:`generate_tier0_patches`.
    """
    labels, meta = generate_tier0_patches(
        m_fine, nseeds_range=nseeds_range, min_neighbors=min_neighbors,
        n_split=n_split, max_rescue_rounds=max_rescue_rounds, sig=sig,
        curvature_weight=curvature_weight_seg, seeding=seeding,
        curv_alpha=curv_alpha, rescue=rescue, reseed_add=reseed_add,
        reseed_rounds=reseed_rounds, verbose=verbose)
    if verbose:
        print(f"  tier-0 segmentation: nseeds={meta['nseeds']}, "
              f"rescued={meta['rescued']}, {meta['n_patches']} patches "
              f"(all regular disks)")
    m_seg, PM = build_cage_from_labels(
        m_fine, labels, target_ratio=target_ratio,
        curvature_weight=curvature_weight_dec, verbose=verbose)
    if verbose:
        print(f"  cage = {len(PM['pm'].X)} verts, {len(PM['pm'].F)} faces")
    return m_seg, PM, meta


# --------------------------------------------------------------------------- #
# Stage 2: cage parameterization (tier 0)
# --------------------------------------------------------------------------- #
def parameterize_cage(PM, area_niter=40, area_step=0.1,
                      shear_niter=40, shear_step=0.2,
                      target='equal', equal_area_blend=0.4,
                      area_exponent=2.0, shear_weight=1e-2, area_weight=1.0,
                      multi_niter=120, multi_step=0.1, bad_patches=None,
                      patch_area_boost=1.0, verbose=True):
    """Map the simplified cage to the sphere.

    Two targets:

    * ``target='equal'`` (default) -- area Newton (method 1, equalize areas) then
      alternating area/shear Newton (method 2). This is the robust, bijective
      scaffold. It does **not** pursue goal (a); the curvature-area
      correspondence is enforced later, on the *fine* final tier, where there are
      many DOF and good triangles. Pursuing curvature areas on the coarse,
      sliver-y cage tends to introduce foldovers.
    * ``target='curvature'`` -- multi-objective Newton (method 6) that drives
      spherical areas toward a *curvature* target ``Ao`` (goal (a)) while
      regularising shear (goal (c)) and preventing flips. ``equal_area_blend``
      relaxes ``Ao`` toward equal area to reduce shear; ``shear_weight`` /
      ``area_weight`` trade goal (c) against goal (a). ``bad_patches`` (from a
      shear look-ahead) get their target area multiplied by ``patch_area_boost``
      so previously-sheared patches are allotted more room to relax.
    """
    from ..level1.set_edge_n_fine_vertices import set_edge_n_fine_vertices_from_PM
    from ..level1.target_areas import compute_curvature_target_areas

    pm = PM['pm']
    ms = surface_mesh(pm.X.copy(), pm.F.copy())
    if getattr(pm, 'face_labels', None) is not None:
        ms.face_labels = np.asarray(pm.face_labels).copy()
    ms = set_edge_n_fine_vertices_from_PM(ms, PM)
    ms.bijective_plot_flag = 0
    ms.mapping_plot_flag = 0

    # Initial equal-area bijective embedding (good starting point, flip-free).
    ms.needs_map2sphere = True
    ms.optimization_method = 1
    ms.newton_niter = area_niter
    ms.newton_step = area_step
    if verbose:
        print(f"  cage: init area Newton x{area_niter} on {len(ms.X)} verts, "
              f"{len(ms.F)} faces")
    ms.map2sphere()

    if target == 'curvature':
        ms.props()
        Ao, _ = compute_curvature_target_areas(
            ms, area_exponent=area_exponent, equal_area_blend=equal_area_blend)
        if bad_patches and patch_area_boost != 1.0 and \
                getattr(ms, 'face_labels', None) is not None:
            labels = np.asarray(ms.face_labels, dtype=int)
            mask = np.isin(labels, np.asarray(list(bad_patches), dtype=int))
            Ao = Ao.copy()
            Ao[mask] *= float(patch_area_boost)
            Ao = Ao / np.sum(Ao) * (4.0 * np.pi)
        ms.target_areas = Ao
        ms.optimization_method = 6
        ms.prevent_flip = True
        ms.newton_niter = multi_niter
        ms.newton_step = multi_step
        ms.multi_objective_opts = {
            'lambdaA': float(area_weight), 'lambda1': float(shear_weight),
            'lambda2': float(shear_weight * 10.0), 'prevent_flip': True,
        }
        ms.needs_map2sphere = True
        if verbose:
            print(f"  cage: multi-objective Newton x{multi_niter} "
                  f"(area_w={area_weight}, shear_w={shear_weight}, "
                  f"blend={equal_area_blend})")
        ms.map2sphere()
    elif shear_niter > 0:
        ms.optimization_method = 2  # alternating area / shear, flip-prevented
        ms.prevent_flip = True
        ms.newton_niter = shear_niter
        ms.newton_step = shear_step
        ms.needs_map2sphere = True
        if verbose:
            print(f"  cage: area/shear Newton x{shear_niter} (flip-prevented)")
        ms.map2sphere()

    PM['pm'].t = ms.t.copy()
    PM['pm'].p = ms.p.copy()
    return ms


def validate_tier0(tier, max_shear=2.0, min_quality=0.05, verbose=True):
    """Strict tier-0 success gate (step [5]): fail early if anything is wrong.

    Success requires:
      * zero flipped / negatively-oriented faces on the sphere (proxy for no
        foldovers / no self-intersections),
      * total signed area close to 4*pi (no gaps / overlaps),
      * no excessively sheared thin faces: ``max(shear) <= max_shear`` and
        ``min parametric quality >= min_quality``.

    Returns ``(success: bool, report: dict)`` and stores the report on
    ``tier.diagnostics['tier0_gate']``.
    """
    m = tier.mesh
    d = tier.diagnostics
    shear, shear_summary = surface_mesh.compute_shear_spherical(m.t, m.p, m.F)
    max_sh = float(np.max(shear)) if len(shear) else 0.0

    n_fold = int(d.get('n_foldovers', -1))
    area_xs = float(d.get('area_excess_rel', 1.0))
    min_q = float(d.get('min_quality', 0.0))

    checks = {
        'no_foldovers': n_fold == 0,
        'area_closed': area_xs <= 0.05,
        'shear_ok': max_sh <= max_shear,
        'no_thin_faces': min_q >= min_quality,
    }
    success = all(checks.values())
    report = {
        'success': success, 'checks': checks,
        'n_foldovers': n_fold, 'area_excess_rel': area_xs,
        'max_shear': max_sh, 'mean_shear': float(shear_summary.get('mean', 0.0)),
        'min_quality': min_q,
    }
    tier.diagnostics['tier0_gate'] = report
    if verbose:
        status = 'PASSED' if success else 'FAILED'
        print(f"  tier-0 gate: {status}")
        for k, ok in checks.items():
            print(f"    [{'OK' if ok else 'XX'}] {k}")
        print(f"    foldovers={n_fold}, area_excess={100*area_xs:.1f}%, "
              f"max_shear={max_sh:.2f}, min_quality={min_q:.3f}")
    return success, report


def locate_foldovers(mesh, verbose=True):
    """Report *where* the folded faces are on a parameterized mesh.

    Returns a dict with the folded-face indices, their per-patch counts (if the
    mesh carries ``face_labels``), and the 3-D curvature there -- so we can tell
    whether folds cluster on thin / high-curvature regions (the stalk).
    """
    from ..level1.fix_flipped_faces import fix_spherical_parameterization_normals

    m, _ = fix_spherical_parameterization_normals(mesh, verbose=False)
    signed, _abs = compute_achieved_spherical_areas(m)
    fold = np.where(signed < 0)[0]
    fc = _face_curvature(m)
    out = {'n_foldovers': int(len(fold)), 'fold_faces': fold,
           'fold_curv_mean': float(fc[fold].mean()) if len(fold) else 0.0,
           'overall_curv_mean': float(fc.mean()),
           'fold_curv_pctl': (float(100.0 * (fc < fc[fold].mean()).mean())
                              if len(fold) else 0.0)}
    labels = getattr(m, 'face_labels', None)
    if labels is not None and len(fold):
        labels = np.asarray(labels, dtype=int)
        per = {}
        for lab in np.unique(labels[fold]):
            per[int(lab)] = int(np.sum(labels[fold] == lab))
        out['per_patch_foldovers'] = per
    if verbose:
        print(f"  foldovers: {out['n_foldovers']}; "
              f"fold-curv mean={out['fold_curv_mean']:.3f} vs overall "
              f"{out['overall_curv_mean']:.3f} "
              f"(folds sit above ~{out['fold_curv_pctl']:.0f}% of faces)")
        if 'per_patch_foldovers' in out:
            print(f"  folds by patch: {out['per_patch_foldovers']}")
    return out


def select_best_cage(m_fine, labels, ratios=(0.22, 0.30, 0.40),
                     cage_area_niter=50, cage_shear_niter=50,
                     curvature_weight=1.0, cage_opts=None, verbose=True):
    """Pick the cage resolution that maps most bijectively.

    Foldovers on the tier-0 sphere map are driven by *cage triangle quality*
    (sliver faces fold), and the best resolution is shape-dependent (too coarse
    loses the stalk; too fine creates slivers). This builds + maps the cage at a
    few decimation ratios and selects the one with the fewest foldovers, breaking
    ties by lowest max-shear then highest min parametric quality.

    ``cage_opts`` is an optional dict forwarded to :func:`parameterize_cage`
    (e.g. ``target``, ``equal_area_blend``, ``shear_weight``, ``area_weight``,
    ``patch_area_boost``, ``bad_patches``) to control the goal (a)/(c) balance.

    Returns ``(m_seg, PM, tier0)`` for the selected ratio.
    """
    cage_opts = dict(cage_opts or {})
    best = None
    for r in ratios:
        m_seg, PM = build_cage_from_labels(
            m_fine, labels, target_ratio=r, curvature_weight=curvature_weight,
            verbose=False)
        cage = parameterize_cage(PM, area_niter=cage_area_niter,
                                 shear_niter=cage_shear_niter, verbose=False,
                                 **cage_opts)
        tier = Tier(level=0, mesh=cage, kind='cage')
        compute_tier_diagnostics(tier, verbose=False)
        shear, _ = surface_mesh.compute_shear_spherical(cage.t, cage.p, cage.F)
        d = tier.diagnostics
        key = (d['n_foldovers'], float(np.max(shear)), -d['min_quality'])
        if verbose:
            print(f"  cage ratio={r:.2f}: {len(PM['pm'].F)}f, "
                  f"folds={d['n_foldovers']}, max_shear={np.max(shear):.2f}, "
                  f"min_q={d['min_quality']:.3f}")
        if best is None or key < best[0]:
            best = (key, r, m_seg, PM, tier)
    _key, r, m_seg, PM, tier = best
    if verbose:
        print(f"  -> selected cage ratio={r:.2f} "
              f"(foldovers={tier.diagnostics['n_foldovers']})")
    return m_seg, PM, tier


# --------------------------------------------------------------------------- #
# Shear look-ahead -> distortion-driven re-meshing feedback
# --------------------------------------------------------------------------- #
def shear_lookahead(tier, shear_percentile=85.0, thin_quality=0.1,
                    abs_shear=None, verbose=True):
    """Mark high-shear / thin faces on a parameterized tier (a look-ahead).

    A high-shear or low-quality (thin) parametric triangle means that region is
    being stretched on the sphere. We mark those faces, record their 3-D
    locations (face centroids) and the patches they belong to, so the mesh can be
    re-meshed *denser* there and the triangles can relax toward equilateral.

    Returns a dict with ``bad_faces``, ``centroids`` (3-D), ``bad_patches``,
    ``shear`` (per-face), ``max_shear``, ``n_bad``, ``shear_threshold``.
    """
    m = tier.mesh
    shear, _summ = surface_mesh.compute_shear_spherical(m.t, m.p, m.F)
    quality = compute_parametric_quality(m)
    qs = np.asarray(quality.get('qualities', []), dtype=float)
    thr = np.percentile(shear, shear_percentile) if len(shear) else np.inf
    if abs_shear is not None:
        thr = min(thr, float(abs_shear))
    bad = shear >= thr
    if len(qs) == len(shear):
        bad = bad | (qs <= thin_quality)
    cen = m.X[np.asarray(m.F, dtype=int)].mean(axis=1)
    labels = getattr(m, 'face_labels', None)
    bad_patches = (sorted({int(l) for l in np.asarray(labels)[bad]})
                   if labels is not None else [])
    out = {
        'bad_faces': np.where(bad)[0], 'centroids': cen[bad],
        'shear': shear, 'max_shear': float(shear.max()) if len(shear) else 0.0,
        'shear_threshold': float(thr), 'n_bad': int(bad.sum()),
        'bad_patches': bad_patches,
    }
    if verbose:
        print(f"  shear look-ahead: {out['n_bad']} bad faces "
              f"(max_shear={out['max_shear']:.2f}, thr={thr:.2f}); "
              f"patches={bad_patches}")
    return out


def shear_density_field(m_fine, centroids, boost=4.0, radius_frac=0.06,
                        base=1.0):
    """Per-vertex density multiplier on ``m_fine``: ``boost`` near ``centroids``.

    Gaussian falloff with radius = ``radius_frac`` * bounding-box diagonal. Feed
    the result to ``remesh_by_curvature(density_field=...)`` to pack more
    triangles into the marked (high-shear) regions.
    """
    field = np.full(len(m_fine.X), float(base))
    centroids = np.asarray(centroids)
    if len(centroids) == 0:
        return field
    from scipy.spatial import cKDTree
    bbox = m_fine.X.max(axis=0) - m_fine.X.min(axis=0)
    radius = max(radius_frac * float(np.linalg.norm(bbox)), 1e-9)
    d, _ = cKDTree(centroids).query(m_fine.X, k=1)
    return base + (boost - base) * np.exp(-(d / radius) ** 2)


def refine_for_shear(m_fine, tier, grow=1.3, boost=4.0, radius_frac=0.06,
                     shear_percentile=85.0, thin_quality=0.1, abs_shear=None,
                     curvature_strength=2.0, verbose=True):
    """Re-mesh ``m_fine`` denser around a tier's high-shear regions.

    Returns ``(m_refined, lookahead)``. The new mesh keeps curvature adaptivity
    *and* adds extra density where the parameterization was stretched, so a
    subsequent tier-0 attempt can map those triangles with less shear.
    """
    la = shear_lookahead(tier, shear_percentile=shear_percentile,
                         thin_quality=thin_quality, abs_shear=abs_shear,
                         verbose=verbose)
    field = shear_density_field(m_fine, la['centroids'], boost=boost,
                               radius_frac=radius_frac)
    target_faces = int(round(len(m_fine.F) * grow))
    m2 = _copy_mesh(m_fine)
    if verbose:
        print(f"  refine_for_shear: {len(m_fine.F)} -> ~{target_faces} faces, "
              f"boost x{boost} (radius {radius_frac:.2f} bbox) around "
              f"{len(la['centroids'])} markers")
    m2.remesh_by_curvature(target_faces=target_faces,
                           curvature_strength=curvature_strength,
                           density_field=field, verbose=False)
    m2.props()
    return m2, la


def build_tier0_with_shear_feedback(
        m_fine, nseeds_range=(3, 30), min_neighbors=3, seeding='curvature',
        curv_alpha=3.0, rescue='reseed', reseed_add=2, reseed_rounds=6,
        cage_ratios=(0.22, 0.30, 0.40), cage_area_niter=50, cage_shear_niter=50,
        curvature_weight=1.0, max_shear=2.0, min_quality=0.05, max_refine=3,
        grow=1.3, boost=4.0, radius_frac=0.06, curvature_strength=2.0,
        tier0_L_max=3, verbose=True):
    """Tier 0 with shear look-ahead feedback.

    Loop: build no-cap/no-annular patches -> quality-driven cage -> bijective
    area+shear map -> gate. If the gate fails only on shear / thin faces, mark
    those regions and re-mesh the fine mesh denser there, then retry (up to
    ``max_refine`` times). Always fails fast: bijectivity (foldovers) is checked
    first and re-meshing only addresses the shear/quality criteria.

    Returns a dict: ``m_fine`` (possibly re-meshed), ``m_seg``, ``PM``,
    ``tier0``, ``labels``, ``ok``, ``history``.
    """
    history = []
    m_cur = m_fine
    last = None  # best result so far (for graceful return)
    for it in range(max_refine + 1):
        if verbose:
            print(f"\n--- tier-0 attempt {it}: {len(m_cur.F)} faces ---")
        try:
            labels, seg_meta = generate_tier0_patches(
                m_cur, nseeds_range=nseeds_range, min_neighbors=min_neighbors,
                seeding=seeding, curv_alpha=curv_alpha, rescue=rescue,
                reseed_add=reseed_add, reseed_rounds=reseed_rounds, verbose=False)
        except RuntimeError as exc:
            if verbose:
                print(f"  segmentation failed after re-mesh: {exc}")
            if last is not None:
                last['history'] = history
                return last
            raise
        m_seg, PM, tier0 = select_best_cage(
            m_cur, labels, ratios=cage_ratios, cage_area_niter=cage_area_niter,
            cage_shear_niter=cage_shear_niter, curvature_weight=curvature_weight,
            verbose=verbose)
        ok, gate = validate_tier0(tier0, max_shear=max_shear,
                                  min_quality=min_quality, verbose=verbose)
        history.append({'iter': it, 'n_faces': len(m_cur.F),
                        'n_patches': seg_meta['n_patches'],
                        'foldovers': gate['n_foldovers'],
                        'max_shear': gate['max_shear'],
                        'min_quality': gate['min_quality'], 'ok': ok})
        last = {'m_fine': m_cur, 'm_seg': m_seg, 'PM': PM, 'tier0': tier0,
                'labels': labels, 'ok': ok, 'history': history}
        if ok or it == max_refine:
            fit_tier_shp(tier0, L_max=tier0_L_max, verbose=verbose)
            if verbose and not ok:
                print(f"  reached max_refine={max_refine}; returning best tier 0")
            return {'m_fine': m_cur, 'm_seg': m_seg, 'PM': PM, 'tier0': tier0,
                    'labels': labels, 'ok': ok, 'history': history}
        # Gate failed on shear/thin -> distortion-driven re-mesh, then retry.
        m_cur, _la = refine_for_shear(
            m_cur, tier0, grow=grow, boost=boost, radius_frac=radius_frac,
            curvature_strength=curvature_strength, verbose=verbose)
    return {'m_fine': m_cur, 'm_seg': m_seg, 'PM': PM, 'tier0': tier0,
            'labels': labels, 'ok': ok, 'history': history}


# --------------------------------------------------------------------------- #
# Stage 3: full patch fill -> authoritative (t, p)
# --------------------------------------------------------------------------- #
def fill_full_patches(PM, m_seg, plot_flag=0, verbose=True):
    """Fill every fine patch interior and assemble the full parameterized mesh.

    Returns
    -------
    m_full : surface_mesh
        Full fine mesh with authoritative ``(t, p)`` and ``face_labels``.
    """
    from ..level1.parameterize_patches_cart import parameterize_patches_cart
    from ..level1.assemble_parameterized_mesh import assemble_parameterized_mesh
    from ..level1.fix_flipped_faces import (
        fix_spherical_parameterization_normals,
    )

    if verbose:
        print(f"  parameterize_patches_cart on {PM['npatches']} patches")
    parameterize_patches_cart(PM, plot_flag=plot_flag)

    m_full = assemble_parameterized_mesh(m_seg, PM)
    if getattr(m_seg, 'face_labels', None) is not None:
        m_full.face_labels = np.asarray(m_seg.face_labels).copy()
    m_full, _ = fix_spherical_parameterization_normals(m_full, verbose=False)
    if verbose:
        print(f"  assembled full mesh: {len(m_full.X)} verts, "
              f"{len(m_full.F)} faces")
    return m_full


# --------------------------------------------------------------------------- #
# Stage 4: per-patch Laplace re-fill on a (decimated) mesh
# --------------------------------------------------------------------------- #
def refill_patches_on_mesh(mesh, t_auth, p_auth, X_auth, verbose=True):
    """Re-solve every patch interior by a Laplace map at this resolution.

    Boundary vertices take the authoritative full-resolution ``(t, p)`` (matched
    by 3-D position), so neighbouring patches share identical boundaries and the
    assembled result is consistent. Interiors are harmonic -> foldover resistant.

    Sets ``mesh.t`` / ``mesh.p`` in place and returns ``(t, p)``.
    """
    from scipy.spatial import cKDTree
    from ..level1.parameterize_patches_cart import parameterize_single_patch

    labels = np.asarray(mesh.face_labels, dtype=int)
    bmask = _patch_boundary_mask(mesh.F, labels)
    tree = cKDTree(X_auth)

    t = np.zeros(len(mesh.X))
    p = np.zeros(len(mesh.X))

    bidx = np.where(bmask)[0]
    if len(bidx):
        _, nn = tree.query(mesh.X[bidx], k=1)
        t[bidx] = t_auth[nn]
        p[bidx] = p_auth[nn]

    n_fail = 0
    for lab in np.unique(labels):
        fmask = labels == lab
        patm, used = _extract_submesh(mesh, fmask)
        local_b = np.where(bmask[used])[0]
        if len(local_b) < 3:
            # Degenerate boundary: fall back to authoritative by position.
            _, nn = tree.query(mesh.X[used], k=1)
            t[used] = t_auth[nn]
            p[used] = p_auth[nn]
            n_fail += 1
            continue
        bt = t[used[local_b]]
        bp = p[used[local_b]]
        try:
            parameterize_single_patch(patm, bt, bp, local_b)
            interior_local = np.setdiff1d(np.arange(len(used)), local_b)
            g = used[interior_local]
            t[g] = patm.t[interior_local]
            p[g] = patm.p[interior_local]
        except Exception:  # noqa: BLE001 - keep authoritative fallback
            _, nn = tree.query(mesh.X[used], k=1)
            t[used] = t_auth[nn]
            p[used] = p_auth[nn]
            n_fail += 1

    mesh.t = t
    mesh.p = p
    if verbose and n_fail:
        print(f"    refill: {n_fail} patches fell back to nearest authoritative")
    return t, p


def optimize_tier_sphere(mesh, area_niter=60, area_step=0.2,
                         shear_niter=40, shear_step=0.03,
                         target='equal', equal_area_blend=0.4,
                         area_exponent=2.0, shear_weight=1e-2, area_weight=1.0,
                         multi_niter=120, multi_step=0.1, verbose=True):
    """Polish a tier on the sphere, with flip prevention.

    * ``target='equal'`` (default) -- area/shear Newton (method 2) then shear
      Newton (method 5): equalises areas and reduces shear. Robust, but does not
      pursue goal (a) (area proportional to curvature).
    * ``target='curvature'`` -- after a short equal-area / shear warm-up, run
      multi-objective Newton (method 6) driving spherical areas toward a
      *blended* curvature target (goal (a)) while regularising shear (goal (c))
      and preventing flips. This is the right place for goal (a): on the fine
      mesh there are enough DOF and good triangles to move areas without folding,
      unlike on the coarse cage. ``equal_area_blend`` trades goal (a) vs shear.
    """
    from ..level1.fix_flipped_faces import fix_spherical_parameterization_normals
    from ..level1.target_areas import compute_curvature_target_areas

    mesh.bijective_plot_flag = 0
    mesh.mapping_plot_flag = 0
    mesh.prevent_flip = True

    if area_niter > 0:
        mesh.optimization_method = 2
        mesh.newton_niter = area_niter
        mesh.newton_step = area_step
        mesh.needs_map2sphere = False
        mesh.map2sphere()
    if shear_niter > 0:
        mesh.optimization_method = 5
        mesh.newton_niter = shear_niter
        mesh.newton_step = shear_step
        mesh.needs_map2sphere = False
        mesh.map2sphere()

    if target == 'curvature' and multi_niter > 0:
        mesh.props()
        Ao, _ = compute_curvature_target_areas(
            mesh, area_exponent=area_exponent, equal_area_blend=equal_area_blend)
        mesh.target_areas = Ao
        mesh.optimization_method = 6
        mesh.newton_niter = multi_niter
        mesh.newton_step = multi_step
        mesh.multi_objective_opts = {
            'lambdaA': float(area_weight), 'lambda1': float(shear_weight),
            'lambda2': float(shear_weight * 10.0), 'prevent_flip': True,
        }
        mesh.needs_map2sphere = False
        if verbose:
            print(f"    curvature polish: multi-objective x{multi_niter} "
                  f"(blend={equal_area_blend}, shear_w={shear_weight})")
        mesh.map2sphere()

    mesh, _ = fix_spherical_parameterization_normals(mesh, verbose=False)
    return mesh


def build_interior_tiers(m_full, n_interior_tiers=3, coarsest_faces=600,
                         curvature_weight=1.0, area_niter=60, shear_niter=40,
                         target='equal', equal_area_blend=0.4, area_exponent=2.0,
                         shear_weight=1e-2, area_weight=1.0, multi_niter=120,
                         verbose=True):
    """Build the patch-interior resolution tiers from the full assembled mesh.

    Each tier (except the finest, which *is* ``m_full``) is ``m_full`` decimated
    with every patch-boundary vertex protected, then re-filled per patch and
    polished on the sphere. When ``target='curvature'`` the polish drives areas
    toward a blended curvature target (goal (a)); the *finest* tier is then also
    polished (otherwise it just inherits the cage-based fill).

    Returns
    -------
    list[Tier]
        Ordered coarse -> fine; the last entry is the full-resolution tier.
    """
    polish_opts = dict(
        target=target, equal_area_blend=equal_area_blend,
        area_exponent=area_exponent, shear_weight=shear_weight,
        area_weight=area_weight, multi_niter=multi_niter)
    t_auth = m_full.t.copy()
    p_auth = m_full.p.copy()
    X_auth = m_full.X.copy()

    bmask_full = _patch_boundary_mask(m_full.F, m_full.face_labels)
    protected = np.where(bmask_full)[0]

    finest_faces = len(m_full.F)
    n_interior_tiers = max(1, int(n_interior_tiers))
    coarsest_faces = int(min(coarsest_faces, finest_faces))

    # Face schedule for the interior tiers, coarse -> fine; last == finest.
    if n_interior_tiers == 1:
        schedule = [finest_faces]
    else:
        schedule = list(np.geomspace(coarsest_faces, finest_faces,
                                     n_interior_tiers))
        schedule = [int(round(x)) for x in schedule]
        schedule[-1] = finest_faces

    if verbose:
        print(f"  interior-tier face schedule (coarse->fine): {schedule}")
        print(f"  protected boundary vertices: {len(protected)} / "
              f"{len(m_full.X)}")

    tiers = []
    for li, tf in enumerate(schedule):
        is_finest = (tf >= finest_faces)
        if is_finest:
            m_k = _copy_mesh(m_full)
            m_k.t = t_auth.copy()
            m_k.p = p_auth.copy()
            if verbose:
                print(f"  tier interior L{li}: full resolution "
                      f"({len(m_k.X)}v, {len(m_k.F)}f)")
            # The finest tier inherits the cage-based fill; polish it toward the
            # curvature target so goal (a) is enforced at full resolution.
            if target == 'curvature':
                if getattr(m_full, 'face_labels', None) is not None:
                    m_k.face_labels = np.asarray(m_full.face_labels).copy()
                optimize_tier_sphere(m_k, area_niter=area_niter,
                                     shear_niter=shear_niter, verbose=verbose,
                                     **polish_opts)
        else:
            if verbose:
                print(f"  tier interior L{li}: decimate -> ~{tf} faces "
                      f"(boundary protected)")
            m_k, _ = m_full.curvature_aware_decimation(
                target_faces=tf, curvature_weight=curvature_weight,
                protected_vertices=protected, verbose=False)
            m_k.props()
            refill_patches_on_mesh(m_k, t_auth, p_auth, X_auth, verbose=verbose)
            optimize_tier_sphere(m_k, area_niter=area_niter,
                                 shear_niter=shear_niter, verbose=verbose,
                                 **polish_opts)
        tiers.append(Tier(level=1 + li, mesh=m_k, kind='interior'))
    return tiers


# --------------------------------------------------------------------------- #
# Diagnostics ("how bad is each region, at this tier")
# --------------------------------------------------------------------------- #
def compute_tier_diagnostics(tier, area_tol=0.05, verbose=True):
    """Global and per-patch quality of a tier's parameterization.

    The tier mesh is first oriented outward (a globally inside-out but otherwise
    valid map reads as "all faces flipped" to the bijectivity gate); this is
    idempotent for already-outward tiers.
    """
    from ..level1.fix_flipped_faces import fix_spherical_parameterization_normals

    m = tier.mesh
    if m.t is not None and m.p is not None:
        m, _ = fix_spherical_parameterization_normals(m, verbose=False)
        tier.mesh = m
    Ao, _ = compute_curvature_target_areas(m)
    gate = check_bijectivity_gate(m, area_tol=area_tol, verbose=False)
    signed, achieved = compute_achieved_spherical_areas(m)
    achieved_abs = np.abs(achieved)
    quality = compute_parametric_quality(m)
    corr = _area_correlation(Ao, achieved_abs)

    diag = {
        'level': tier.level,
        'kind': tier.kind,
        'n_verts': len(m.X),
        'n_faces': len(m.F),
        'bijective': bool(gate['passed']),
        'n_foldovers': int(gate['n_foldovers']),
        'area_excess_rel': float(gate['area_excess_rel']),
        'area_correlation': corr,
        'mean_quality': float(quality['mean_quality']),
        'min_quality': float(quality['min_quality']),
        'gate': gate,
    }

    labels = getattr(m, 'face_labels', None)
    if labels is not None:
        diag['per_patch'] = _per_patch_diagnostics(
            m, Ao, achieved_abs, signed, quality, labels)

    tier.diagnostics.update(diag)
    if verbose:
        print(f"  tier L{tier.level} ({tier.kind}): bijective={diag['bijective']}, "
              f"foldovers={diag['n_foldovers']}, "
              f"area_excess={100 * diag['area_excess_rel']:.1f}%, "
              f"area_corr={corr:.3f}, mean_q={diag['mean_quality']:.3f}, "
              f"min_q={diag['min_quality']:.3f}")
    return diag


def _per_patch_diagnostics(m, Ao, achieved_abs, signed, quality, labels):
    labels = np.asarray(labels, dtype=int)
    qualities = np.asarray(quality.get('qualities', []), dtype=float)
    out = {}
    for lab in np.unique(labels):
        fmask = labels == lab
        ao_sum = float(Ao[fmask].sum())
        ach_sum = float(achieved_abs[fmask].sum())
        rec = {
            'n_faces': int(fmask.sum()),
            'target_area': ao_sum,
            'achieved_area': ach_sum,
            'area_ratio': (ach_sum / ao_sum) if ao_sum > 1e-15 else np.nan,
            'n_foldovers': int(np.sum(signed[fmask] < 0)),
        }
        if len(qualities) == len(labels):
            rec['mean_quality'] = float(qualities[fmask].mean())
            rec['min_quality'] = float(qualities[fmask].min())
        out[int(lab)] = rec
    return out


# --------------------------------------------------------------------------- #
# Stage 5: SHP per tier
# --------------------------------------------------------------------------- #
def fit_tier_shp(tier, L_max=16, verbose=True):
    """Fit a spherical-harmonics surface to a tier; record reconstruction RMS."""
    s = shp_surface(tier.mesh, L_max)
    res = getattr(s, 'residual', None)
    if res is not None and np.ndim(res) == 2:
        rms = float(np.sqrt(np.mean(np.sum(np.asarray(res) ** 2, axis=1))))
    else:
        rms = 0.0
    tier.shp = s
    tier.shp_rms = rms
    if verbose:
        print(f"  tier L{tier.level}: SHP L_max={L_max}, recon RMS={rms:.6f}")
    return s, rms


# --------------------------------------------------------------------------- #
# End-to-end convenience
# --------------------------------------------------------------------------- #
def run_tiered_pipeline(mesh, target_verts=3000, n_tiers=4,
                        coarsest_faces=600, curvature_strength=2.0,
                        curvature_weight=1.0, nseeds_range=(3, 30),
                        min_neighbors=3, n_split=5, max_rescue_rounds=2,
                        seeding='curvature', curv_alpha=3.0, rescue='reseed',
                        reseed_add=2, reseed_rounds=6, target_ratio=0.25,
                        cage_ratios=(0.22, 0.30, 0.40),
                        cage_area_niter=50, cage_shear_niter=50,
                        tier_area_niter=60, tier_shear_niter=40,
                        tier_target='equal', equal_area_blend=0.4,
                        area_exponent=2.0, shear_weight=1e-2, area_weight=1.0,
                        tier_multi_niter=100,
                        tier0_max_shear=2.0, tier0_min_quality=0.05,
                        stop_on_tier0_failure=True,
                        L_max=16, tier0_L_max=3, area_tol=0.05, verbose=True):
    """Full arbitrary-mesh -> tiered patch-based SHP parameterization.

    ``n_tiers`` counts the cage (tier 0) plus ``n_tiers - 1`` interior tiers.
    Tier 0 is built fail-early: no cap / no annular patches (rescue by
    subdivision), bijective area+shear map, strict success gate, and a low-order
    (``tier0_L_max``) SHP sanity shape. If the tier-0 gate fails and
    ``stop_on_tier0_failure`` is True, the pipeline stops and returns only tier 0.

    Goal (a) -- spherical area proportional to curvature -- is enforced on the
    *fine* interior tiers (not the coarse cage, where it tends to fold). Set
    ``tier_target='curvature'`` to drive each fine tier toward a curvature target
    via multi-objective Newton (method 6). ``equal_area_blend`` in [0, 1] relaxes
    that target toward equal area to reduce shear (goal (a) vs goal (c) trade);
    ``area_exponent`` is the ``|H|`` power; ``shear_weight``/``area_weight``
    weight goal (c) vs goal (a); ``tier_multi_niter`` is the polish iteration
    count. The default ``'equal'`` keeps the robust equal-area behaviour.

    Returns
    -------
    tiers : list[Tier]
        Tier 0 (cage) ... finest. Each carries ``.mesh`` (+ ``.t``/``.p``),
        ``.diagnostics``, ``.shp``, ``.shp_rms``.
    artefacts : dict
        ``m_fine``, ``m_seg``, ``PM``, ``m_full`` (if reached), ``report``.
    """
    report = {'tiers': []}

    if verbose:
        print("=" * 60)
        print("Stage 0: Preprocess (repair + curvature-adaptive remesh)")
        print("=" * 60)
    m_fine, q_stats = preprocess_mesh(
        mesh, target_verts=target_verts,
        curvature_strength=curvature_strength, verbose=verbose)
    report['preprocess'] = q_stats

    if verbose:
        print("\n" + "=" * 60)
        print("Stage 1: Tier-0 patches (no cap / no annular)")
        print("=" * 60)
    labels, seg_meta = generate_tier0_patches(
        m_fine, nseeds_range=nseeds_range, min_neighbors=min_neighbors,
        n_split=n_split, max_rescue_rounds=max_rescue_rounds, seeding=seeding,
        curv_alpha=curv_alpha, rescue=rescue, reseed_add=reseed_add,
        reseed_rounds=reseed_rounds, verbose=verbose)
    report['segmentation'] = seg_meta

    if verbose:
        print("\n" + "=" * 60)
        print("Stage 2: Quality-driven cage + bijective area+shear map + gate")
        print("=" * 60)
    m_seg, PM, tier0 = select_best_cage(
        m_fine, labels, ratios=cage_ratios, cage_area_niter=cage_area_niter,
        cage_shear_niter=cage_shear_niter, curvature_weight=curvature_weight,
        verbose=verbose)
    tier0_ok, tier0_gate = validate_tier0(
        tier0, max_shear=tier0_max_shear, min_quality=tier0_min_quality,
        verbose=verbose)
    fit_tier_shp(tier0, L_max=tier0_L_max, verbose=verbose)
    report['tier0_gate'] = tier0_gate

    if not tier0_ok and stop_on_tier0_failure:
        if verbose:
            print("\n*** Tier 0 gate FAILED -- stopping (fail early). "
                  "Fix tier 0 before proceeding to finer tiers. ***")
        report['tiers'].append({
            'level': 0, 'kind': 'cage', 'n_verts': tier0.n_verts,
            'n_faces': tier0.n_faces,
            'bijective': tier0.diagnostics.get('bijective'),
            'n_foldovers': tier0.diagnostics.get('n_foldovers'),
            'area_correlation': tier0.diagnostics.get('area_correlation'),
            'mean_quality': tier0.diagnostics.get('mean_quality'),
            'shp_rms': tier0.shp_rms})
        return [tier0], {'m_fine': m_fine, 'm_seg': m_seg, 'PM': PM,
                         'report': report}

    if verbose:
        print("\n" + "=" * 60)
        print("Stage 3: Full patch fill (authoritative parameterization)")
        print("=" * 60)
    m_full = fill_full_patches(PM, m_seg, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("Stage 4/5: Interior tiers + diagnostics + SHP")
        print("=" * 60)
    interior = build_interior_tiers(
        m_full, n_interior_tiers=n_tiers - 1, coarsest_faces=coarsest_faces,
        curvature_weight=curvature_weight, area_niter=tier_area_niter,
        shear_niter=tier_shear_niter, target=tier_target,
        equal_area_blend=equal_area_blend, area_exponent=area_exponent,
        shear_weight=shear_weight, area_weight=area_weight,
        multi_niter=tier_multi_niter, verbose=verbose)
    for t in interior:
        compute_tier_diagnostics(t, area_tol=area_tol, verbose=verbose)
        fit_tier_shp(t, L_max=L_max, verbose=verbose)

    tiers = [tier0] + interior

    for t in tiers:
        report['tiers'].append({
            'level': t.level, 'kind': t.kind,
            'n_verts': t.n_verts, 'n_faces': t.n_faces,
            'bijective': t.diagnostics.get('bijective'),
            'n_foldovers': t.diagnostics.get('n_foldovers'),
            'area_correlation': t.diagnostics.get('area_correlation'),
            'mean_quality': t.diagnostics.get('mean_quality'),
            'shp_rms': t.shp_rms,
        })

    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline summary (coarse -> fine)")
        print("=" * 60)
        for ts in report['tiers']:
            print(f"  L{ts['level']} {ts['kind']:8s}: "
                  f"{ts['n_verts']:5d}v {ts['n_faces']:5d}f | "
                  f"bijective={ts['bijective']} folds={ts['n_foldovers']} | "
                  f"area_corr={ts['area_correlation']:.3f} | "
                  f"mean_q={ts['mean_quality']:.3f} | "
                  f"shp_rms={ts['shp_rms']:.5f}")

    artefacts = {'m_fine': m_fine, 'm_seg': m_seg, 'PM': PM,
                 'm_full': m_full, 'report': report}
    return tiers, artefacts
