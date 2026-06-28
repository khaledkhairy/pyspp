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
- `19867e0` this handoff doc
- `2f0cd62` grid-quadrature high-L_max SH analysis (scales to L=72) + recommend_lmax
- `11d00d0` analyze_shp (LS<=24 / grid>24) -> L_max=60 everywhere + Tutte Stage-1b fallback
- `59faf32` batch TARGET_VERTS=7000 + CLI smoke test (run_parameterization_test.py)
- `b46d6e9` robust_foldover_count (exact fallback) + honest Stage-2c classification
- `a24050c` **guaranteed solver: local stereographic-zoom untangle** + Stage-2c
  wiring + guaranteed-bijective resolution cap -> hydra reaches **true robust 0**
  (was too_complex); clean shapes untouched
- (pending) **multiresolution (coarse-to-fine) equal-area map** (fixes protrusion
  truncation: zebrafish recon RMS 6.5% -> 0.18% @L=60) + `export_off` (outward
  winding for clean MeshLab normals) + batch `_f_param_sphere.off` export  <-- HEAD
- Git identity is NOT configured; commit with one-off:
  `git -c user.email="khaledkhairy@yahoo.com" -c user.name="Khaled Khairy" commit ...`
- NOTE: the other files shown modified in `git status` (tiered_spherical_parameterization.py,
  *_diagnostic.txt, *_tier_by_tier_05.ipynb, cmcf_batch_outputs/) are PRE-EXISTING and
  unrelated to this work -- leave them alone.

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
2m. **multiresolution rescue** (`multiresolution_sphere_map`) — fires only when the
   direct equalize leaves the map crushed (`area_cov > 0.35` & `< 2.5`, i.e. a thin
   protrusion the LOCAL equalizer stalled on). Builds an equal-area map coarse-to-
   fine (equalize a coarse vertex-SUBSET cage where the protrusion spreads, transfer
   onto the fine geometry, re-equalize) and **keep-best by (robust folds, cov)** vs
   the direct map. Blobby shapes skip it (no regression); hydra (cov>2.5) is excluded
   (float64-walled on area). Fixes protrusion truncation (zebrafish cov 1.5->0.45,
   recon RMS 6.5%->0.18% @L=60); method tag `+multires`.
2c. **escalation gate + guaranteed solver** — trigger = residual robust folds:
    0 -> `bijective`; few & kappa<=6 -> `local_fold_surgery` (cheap); anything
    remaining -> **`guaranteed_untangle`** = local stereographic-zoom untangle
    from the best-robust snapshot (Tutte/Mobius), which defeats the float64
    precision wall and takes a hydra to **true robust 0** (produce + flag, never
    skip). If a shape hits the float64 *representation* ceiling at high res
    (exactly-collapsed faces, `n_degen>0`), a **resolution cap** re-derives at a
    coarser `target_verts` where a bijective float64 embedding exists.
2b. **anisotropic refinement** (`anisotropic_rounds`) — stretch-aligned: split the
    3-D edges the map stretched (long on sphere) + re-equalize (warm start);
    GATED on fold-free (refining a folded map explodes). ~2x thin-feature fidelity.
3. **SHP fit** (`analyze_shp`) — least-squares (`fit_shp`) for low L_max, grid
   quadrature (`shp_analysis_grid`) for high L_max; 'auto' switches above L=24.
Result dict keys: `mesh, shp, quality, n_foldovers, complexity, shp_rms_rel,
stage1_diag, stage1b_diag, tutte_info, stage2_diag, aniso_history, surgery, ...`.

## Status on test_set (now: target_verts~2000, **L_max=60** default in batch)
- 8/9 **bijective**: BDH6230 brain, echinocyte, mushroom(x4), zebrafish, **+hydra**.
- `1dpx` (protein) **near_bijective** (3 folds); SHP RMS **0.09%** at L=60
  (was 0.51% at L=16). cMCF path (Tutte fallback was worse here, so gate kept cMCF).
  Its 3 folds are **kernel-empty** (inverted but NOT degenerate, `n_degen==0`):
  the zoom untangle is *attempted* but cannot clear them (a fixed-boundary
  vertex move can't undo a non-star-shaped overlap) -- this needs an **edge
  flip**, a different tool (still backlog). Correctly NOT resolution-capped
  (coarsening doesn't fix kernel-empty folds).
- `hydra_full_smooth` is now **bijective (true robust 0)** via the guaranteed
  stereographic-zoom untangle (`tutte+zoom`). SHP RMS **0.13%** at L=60 (the
  high-distortion area distribution is fine for the grid-quadrature analysis,
  which interpolates the tentacle faithfully *because the map is bijective*).
  At 2k it untangles in place; at the batch's 7k it crosses the float64
  *representation* ceiling (~190 tentacle-tip faces collapse to coincident
  float64 points -- unseparable) so the **resolution cap** re-derives at ~3.5k
  (bijective there) and flags `precision_capped`. The earlier diagnosis (a
  float64 precision wall) was right; the key insight is that the wall hits
  FOLDS *locally* (curable by the zoom) and the area allocation only crosses the
  hard *representation* ceiling at high resolution (curable by capping res).

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

## DONE: robust foldover predicate (honest metrics) + KEY hydra diagnosis
- `robust_foldover_count(S, F, return_degenerate=)` -- float64 determinant with an
  a-priori error bound (permanent * eps) + EXACT rational fallback for the
  near-degenerate faces. The float64 test is unreliable BOTH ways: hydra's crushed
  Tutte map reads 41 folds in float64 but **5 exactly** (+2 collapsed faces); an
  equalized map can read 39 vs **45**. Wired into the Stage-2c quality
  classification (`result['n_foldovers']` is now the honest count;
  `result['n_degenerate']` = collapsed faces). Clean shapes unaffected (still 0).
- EXHAUSTIVE float64 untangling study (all REVERTED, none reach true 0 on hydra):
  Tutte (5 robust), progressive centroid/kernel, hierarchical (~46-100),
  local_fold_surgery (107->46 plateau), global equalize (->100), targeted
  fold-only gradient, hinge "push every face to +margin" (1dpx 3->3, hydra 45->45).
  CONCLUSION: residual folds are genuinely stuck (mesh untangling is NP-hard in
  general) AND the worst sit in sub-float64-precision crushed regions. No float64
  trick closes them.

## DONE: the guaranteed solver (local stereographic-zoom untangle) + wiring
Built option #2 from the old plan -- it reaches **true robust 0** on hydra, so
the orbifold-Tutte (#1) and mpmath (#3) fallbacks turned out **not needed** for
bijectivity. New functions in `cmcf_spherical_parameterization.py`:
- `_robust_face_signs(S, F)` -- per-face exact orientation sign (+1/-1/0);
  `robust_foldover_count` now wraps it. The fold *detector* used during untangle.
- `_planar_hinge_untangle(P, T, free, ...)` -- 2-D untangler: push every triangle
  signed area above a small +margin (hinge energy) + light Laplacian smoothing,
  Adam with a fold-count-guarded backtrack; boundary pinned. Patch pre-rescaled
  to O(1) so float64 has full resolution.
- `local_stereographic_untangle(mesh, ring=3, expand_max=2, time_budget=90, ...)`
  -- per connected folded/degenerate cluster: rotate the crushed centroid to the
  +z pole, stereographic-project a ring-padded patch from the *south* pole (so
  the crushed cap lands at the chart ORIGIN), recentre + rescale to O(1),
  `_planar_hinge_untangle`, then map free verts back (inverse stereographic ->
  inverse rotation). The chart is orientation-preserving, so fold-free-in-plane
  => fold-free-on-sphere, and the O(1) separations survive the round-trip to the
  tiny cap. A cleared patch must match the patch (== global) majority sign, so it
  can never be globally flipped into new folds. `time_budget` bounds effort.
- `guaranteed_untangle(mesh, seed_maps=...)` -- picks the **fewest-robust-fold**
  start among the current map + snapshots, then runs the above. CRITICAL: the
  Stage-2 area equalizer optimises an unreliable float64 fold count and can
  *increase* hydra's robust folds (4 -> 43) and spread them, so the rescue must
  start from the Mobius/Tutte snapshot, NOT the equalized map.
Wiring (`parameterize_to_sphere` Stage 2c): residual robust folds -> (optional
`local_fold_surgery`) -> `guaranteed_untangle` (seeded by the `m_centered` Mobius
snapshot + the `m_sphere_raw` Tutte init). On reaching 0, `method += '+zoom'`,
`result['rescued']=True`. Rescued shapes SKIP Stage-2b anisotropic refine (it
re-runs the float64 equalizer and would re-fold the crushed cap + worsen the
precision wall); a defensive post-aniso re-verify zoom-untangles once more if the
warm re-equalize regressed a non-rescued shape.
**Resolution cap** (the high-res representation-ceiling backstop, replaces the
need for mpmath/multi-chart on this test set): if a result is non-bijective AND
has exactly-degenerate faces (`n_degen>0`, the ceiling signature), re-derive at
`target_verts*0.5` (down to `precision_floor_verts=1500`, depth<=4). Gated on
`n_degen>0` (NOT kappa -- kappa>6 for ALL the hard shapes incl. clean mushroom,
so it can't discriminate; only the representation ceiling makes degenerate faces).
Result flagged `precision_capped`. Hydra@7k -> capped to ~3.5k bijective.

## Validation (`run_parameterization_test.py`)
- hydra@2k: `tutte+zoom` **bijective 0 folds**, 0.13% RMS @L=60 (was too_complex).
- hydra@7k: float64 representation ceiling (190 collapsed faces) -> resolution
  cap -> **bijective** @3501 verts, 0.11% RMS @L=60, `precision_capped`.
- 8 clean shapes UNCHANGED (cmcf, bijective); 1dpx near_bijective (3 folds,
  kernel-empty, attempted-but-not-capped); no regressions.

## DONE (this session): equal-area fidelity (multiresolution map) + clean OFF export
Investigated visual recon truncation of lobes/tentacles. KEY DIAGNOSIS (measured):
the LOCAL log-barrier equalizer reaches near-perfect equal-area on blobby shapes
(cov ~0.13) but STALLS on a thin protrusion -- from the conformal cMCF start it
accepts a handful of steps then can't inflate the crushed protrusion (zebrafish:
9 steps then stuck, cov 1.54, 31% of faces ~1e11x too small -> SHP truncates the
tail). It is a LOCAL-MINIMUM, not phantom folds (start has 0 float64 & 0 robust
folds) and not a regression from the zoom work (zebrafish is on the unchanged
bijective-cMCF path; cov is stable across res/aniso). Crucially the SAME equalizer
spreads the protrusion perfectly at COARSE res (zebrafish @600v -> cov 0.06), and a
warm start STAYS equal-area through refinement.
- `multiresolution_sphere_map` -- coarse-to-fine equal-area: equalize a coarse cage
  (a vertex SUBSET via protected-FPS curvature decimation, so fine geometry is kept),
  `_transfer_map_3d` it onto the fine vertices (barycentric, cage pinned exactly),
  re-equalize. NOTE: the Laplacian "follow" is OFF by default -- it pushes toward
  uniform vertex SPACING, which fights equal AREA on a curvature-remeshed mesh.
- Wired as **Stage 2m** keep-best (see pipeline stages). Validated end-to-end:
  zebrafish `cmcf+multires` bijective, recon RMS **6.5% -> 0.18%** @L=60 (0.83% @L=16),
  cov 1.5->0.45; aniso preserves it. mushroom/echinocyte/brain/mushroom_repaired x3
  UNCHANGED (skip multires); hydra/1dpx unchanged. No regressions.
- `export_off(path, X, F)` (in cmcf module) -- writes `.off` with consistent OUTWARD
  winding (`trimesh.repair.fix_normals`), fixing the "flipped triangles" MeshLab
  showed on SH-recon meshes (sampled on a raw icosphere face list -> winding not
  outward). Batch notebook now exports `<name>_f_param_sphere.off` (the parametric
  sphere, same connectivity, for inspecting area/shear evenness in MeshLab) and uses
  `export_off` for every `.off`. (GOTCHA: that notebook is CRLF + lives in Dropbox +
  stays open in the IDE; editing it as raw text races the IDE/Dropbox writer and
  truncated it once -- prefer the notebook tooling or paste the snippet by hand.)

## STILL OPEN: hydra equal-area (orbifold-Tutte)
Multires does NOT help hydra (cov stays ~3.6, excluded by the cov<2.5 trigger): its
tentacle is float64-area-walled even at coarse res. Hydra is bijective (zoom+cap) and
its grid-quadrature SHP is fine (0.11-0.13% @L=60), but its *area* map is still
crushed. The real fix is **orbifold-Tutte** (constructive cone-point area) or
extended precision -- the remaining big-ticket item for the most extreme shapes.

## NEXT (optional, lower priority)
- **1dpx kernel-empty single folds**: need an **edge-flip** surgery (the zoom
  cannot fix a non-star-shaped overlap). Different tool from the precision wall.
- Orbifold-Tutte (#1) / mpmath (#3): only if a future shape's zoom+cap can't
  reach 0 (none on the current test set). Orbifold-Tutte would also give hydra a
  *uniform* area map (constructive cone-point area) if SHP at high L is ever
  insufficient -- but grid-quadrature already reconstructs hydra at 0.13%.
- `recommend_lmax` into batch summary; speed high-L `Y` build.

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
