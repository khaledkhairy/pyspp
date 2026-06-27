# cMCF Spherical Parameterization — Project Handoff

Carry-over doc so this work can continue in a fresh chat thread. In a new
thread: point the agent at this file and at
`pySHP/level2/cmcf_spherical_parameterization.py`. Pipeline entry point:
`cmcf.parameterize_to_sphere(mesh, ...)`.

## Goal
Arbitrary genus-0 triangular mesh -> robust, bijective unit-sphere
parameterization -> spherical-harmonics (SHP) coefficients, for a cloud-scale
shape library. Near-term focus: **proteins** (highly convoluted -> need high
`L_max`, up to 32-72).

## Key files
- **Core**: `pySHP/level2/cmcf_spherical_parameterization.py`
- **Debug notebook** (single mesh, Plotly, per-stage 3-D views):
  `pySHP/tests/test_cmcf_spherical_parameterization_01.ipynb`
- **Batch notebook** (loops `test_set/`, writes 5 artifacts + quality table):
  `pySHP/tests/test_cmcf_batch_testing_01.ipynb`
- **CLI smoke test**: `pySHP/tests/run_parameterization_test.py`
- Test meshes: `code/Matlab/shp_toolbox-main/shp_toolbox-main/test_data/off/test_set/`
- Outputs: `Project_spherical_parameterization/Projection_output/` (outside repo)
- `sh_basis` / `shp_surface` (SH basis + surface): `pySHP/sh_basis.py`, `pySHP/shp_surface.py`

## Commit history (master)
- `0c3a64b` initial (old tiered/patch pipeline — kept for reference; brittle)
- `76802f5` segmentation-free cMCF backbone
- `feecdc3` stretch-aligned anisotropic refinement + shape regularizer
- `a418579` SHP canonicalization + invariant descriptor + batch notebook
- `5cbfc51` equalizer untangle-from-folded + aniso gate + fast untangler + demo fix
- `ccc300f` gated escalation (fold-count+kappa) + Tier-1 local fold surgery + quality classes
- `0c51a73` props() mean-curvature vectorized (~130x faster, exact on consistent winding)
- Git identity is NOT configured; commit with one-off:
  `git -c user.email="khaledkhairy@yahoo.com" -c user.name="Khaled Khairy" commit ...`

## Pipeline stages (`parameterize_to_sphere`)
0. **preprocess** — repair + keep-largest + curvature-adaptive remesh.
1. **cMCF** (`cmcf_sphere_map`) — conformalized mean-curvature flow -> sphere map;
   adaptive tangential untangle. NOTE: cMCF is *conformal*, NOT guaranteed
   bijective -- it folds for high-distortion shapes (hydra).
1b. **guaranteed-bijective fallback** — fires ONLY when cMCF leaves folds (so
   clean shapes keep their nicer conformal map, no degradation/extra compute).
   `tutte_sphere_map` (planar Tutte + inverse stereographic; provably fold-free
   in exact arithmetic). Keeps whichever of cMCF / fallback has fewest folds.
1c. **Mobius centering** (`mobius_center`) — conformal, area-weighted centroid -> 0.
2. **equalize_areas** — interior-point Adam, log-barrier; **untangles from folded
   starts** (accepts steps that don't increase folds); keep-best by (folds, cov);
   `lambda_shape` shape regularizer; `free_mask` for local surgery; `area_blend`
   1.0=uniform (best for SHP) .. 0.0=curvature.
2c. **escalation gate** — trigger = residual foldovers (+ `kappa`):
    0 -> `bijective`; few (<=~2% faces & kappa<=6) -> `local_fold_surgery`;
    many/high-kappa -> `too_complex` (flagged, no wasted compute).
2b. **anisotropic refinement** (`anisotropic_rounds`) — stretch-aligned: split the
    3-D edges the map stretched (long on sphere) + re-equalize (warm start);
    GATED on fold-free (refining a folded map explodes). ~2x thin-feature fidelity.
3. **SHP fit** (`analyze_shp`) — least-squares (`fit_shp`) for low L_max, grid
   quadrature (`shp_analysis_grid`) for high L_max; 'auto' switches above L=24.
Result dict keys: `mesh, shp, quality, n_foldovers, complexity, shp_rms_rel,
stage1_diag, stage1b_diag, tutte_info, stage2_diag, aniso_history, surgery, ...`.

## Status on test_set (now: target_verts~2000, **L_max=60** default in batch)
- 7/9 **bijective**: BDH6230 brain, echinocyte, mushroom(x4), zebrafish.
- `1dpx` (protein) **near_bijective** (3 folds); SHP RMS **0.09%** at L=60
  (was 0.51% at L=16). cMCF path (Tutte fallback was worse here, so gate kept cMCF).
- `hydra_full_smooth` **too_complex** — Tutte fallback cut folds **261 -> 35** (vs
  cMCF), confirming the bijective-start architecture, but it does NOT reach 0: a
  float64 PRECISION WALL (tentacle/body area ratio exceeds machine precision at
  2k verts -> tentacles collapse to sub-epsilon triangles that underflow into
  spurious folds). Mean-value weights were WORSE (117) -> it's precision, not
  weighting. The principled fix is a progressive/multiresolution embedding
  (validity-by-construction, no global ill-conditioned solve) -- see NEXT.

## Conventions / gotchas (IMPORTANT)
- Spherical: `kk_cart2sph`/`kk_sph2cart`; `t`=colatitude∈[0,π], `p`=azimuth.
- **Winding**: enforce consistent-outward (`ensure_outward_winding`,
  trimesh.repair.fix_normals) before ANY foldover count. Foldover = `orient<0`.
- Bijectivity == 0 foldovers (`sphere_foldover_count`). Don't winding-flip to mask.
- `props()` H is signed mean curvature; downstream mostly uses `|H|` / `H^2`.
- **FOE canonicalization** (`canonicalize_shp`): exact for asymmetric shapes,
  AMBIGUOUS up to symmetry for near-symmetric (e.g. mushroom). For shape
  **distance**, use `shp_degree_energy` (per-degree energy, a true rot/scale
  invariant; ~1.3% across pose/scale). (c)/(e) outputs use FOE.
- **SHP analysis (`fit_shp`) is least-squares over the mesh -> does NOT scale**
  past ~L=32 (basis matrix: L=48~8GB, L=72~22GB). High L needs the grid path below.

## DONE: grid-quadrature high-L_max SH analysis + L_max criterion (commit pending)
Kept the `sh_basis` 'bosh' convention (so `.shp3` stays compatible). New functions
in `cmcf_spherical_parameterization.py`:
- `resample_to_grid(mesh, basis)` -- barycentric-interpolate x,y,z onto the basis
  Gauss grid (KDTree on face-centroid directions -> cone/barycentric pick + interp).
- `_quadrature_basis(L_max, gdim)` -- builds an `sh_basis` with ONLY the grid + `Y`
  (its `__init__` eagerly builds 6 huge derivative arrays, ~5.4 GB at L=72 -- skip).
- `shp_analysis_grid(mesh, L_max, gdim=None, method='auto')` -> `(shp, rms_rel)`.
  `method`: 'diag' (`c = sum(w f Y)/sum(w Y^2)`, fast), 'galerkin' (`(Y^T W Y)c=Y^T W f`,
  robust), 'auto' (diag, fall back to galerkin if grid round-trip > 5%).
- `recommend_lmax(mesh, L_max_max=48, rms_target=0.005)` -> `(rec, rms_curve, E)`:
  one grid fit, then incrementally truncate to read RMS(L) for all L (cheap);
  recommend smallest L with grid-RMS <= target. Mesh-adaptive; discriminates
  complexity. `E` = per-degree power spectrum.

**CRITICAL GOTCHA (cost an hour):** `basis.w` is the BARE Gauss product weight for
`int dtheta dphi`; the round-sphere measure is `sin(theta) dtheta dphi` (the
`sin(theta)` lives in `SSn` in `shp_surface.update_full`, NOT in `.w`). The
quadrature MUST use `w = basis.w * sin(basis.t)` or the basis isn't orthogonal and
the diagonal projection aliases (odd-degree energies blow up, RMS(L) increases).
Galerkin hides this (any positive weights still give a valid weighted LS) -- which
is why `shp_analysis_grid` 'auto' looked fine while a diag-only path was garbage.

Validation (params at target_verts~2000; grid `gdim=2*L+2`):
- `1dpx`   grid-RMS  L=16 .51% / 32 .19% / 48 .11% / 72 .06%  (L=48 ~10s, L=72 ~64s)
- `mushroom_repaired_03`            L=16 .63% / 32 .34% / 48 .23% / 72 .14%
- monotone RMS drop; clean decaying spectra. recommend_lmax @0.5%: 1dpx L=15,
  mushroom L=23 (its thin stalk is genuinely higher-frequency -> needs more band).
- L=72 cost is dominated by the `Y` build (5329 `lpmv` calls), not the solve.

## DONE (this session): L_max=60 wiring + guaranteed-bijective Tutte fallback
- `analyze_shp(mesh, L_max, method='auto')` routes low L -> least-squares, high L
  -> grid quadrature. `parameterize_to_sphere` (Stage 3) and `canonicalize_shp`
  both go through it, so the whole pipeline scales to L=60 (LS is infeasible there).
  `canonicalize_shp` fits the degree-1 ellipsoid cheaply at low L, final fit via
  grid on the FULL-res mesh. Batch notebook `L_MAX=60`.
- `tutte_sphere_map` (Stage 1b fallback). Validated end-to-end at L=60 (1dpx
  recon/canon/reconstruct all fine; ~35s/mesh).

## ATTEMPTED + REVERTED: progressive/multiresolution embedding (needs more work)
Built `progressive_sphere_map` (decimate by manifold-safe vertex removals -> Tutte
on a small base -> re-insert each vertex by its link). TWO placement schemes tried:
- spherical centroid: folds scale with #reinsertions (mushroom 521, hydra 1216);
  larger base only reduces proportionally (base=800 still 71 on mushroom).
- in-kernel "max-min-margin" placement: WORSE (mushroom 942) -- a single bad early
  placement tangles later links (cascade); the fold-free invariant isn't actually
  held. Both REVERTED (degrade even easy shapes, violating the no-degrade rule).
A correct version is real Praun-Hoppe-grade work: a *validated* fold-free base
(check base foldovers), provably-in-kernel placement with consistent orientation,
and per-level flip-free relaxation; only then wire as a keep-best Stage-1b fallback.
IMPORTANT empirical finding: higher resolution makes hydra WORSE (Tutte folds 2k=41,
5k=604) -> confirms a float64 precision wall (smaller tentacle triangles underflow).
So even a correct progressive may not fully clear hydra; multi-chart or extended
precision may ultimately be needed for the most extreme (Q~0.007) shapes.

## NEXT
- Progressive embedding, done properly (above), OR accept hydra-class as flagged.
- `recommend_lmax` per-mesh value into the batch summary + `.shp3` sidecar (the
  user has fixed L=60 for now, so lower priority).
- Optional: speed the high-L `Y` build (vectorize/cache `lpmv` across L) for cloud.

## Resolution / test
- Batch now `TARGET_VERTS = 7000` (user: raise everybody to 6-8k for detail).
  (Hydra stays flagged regardless; 7k benefits the 8 good shapes.)
- CLI smoke test (no notebook): `python pySHP/tests/run_parameterization_test.py`
  (optional mesh-name args; `--verts`, `--lmax`, `--aniso`). Prints method /
  bijectivity / folds / RMS / time per mesh.

## Other backlog
- Optional: route `too_complex` shapes to the old patch-based pipeline (user
  prefers NOT to; flagging is the default).
- Optional: edge-flip surgery for kernel-empty single folds (1dpx).
- Profile other O(V*F) loops for 14-15k-vertex meshes if preprocess is slow.
