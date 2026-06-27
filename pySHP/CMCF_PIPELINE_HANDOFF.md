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
1. **cMCF** (`cmcf_sphere_map`) — conformalized mean-curvature flow -> bijective
   sphere map; adaptive tangential untangle (`_untangle_spherical`).
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
3. **SHP fit** (`fit_shp`) — upsample mesh on sphere + least-squares.
Result dict keys: `mesh, shp, quality, n_foldovers, complexity, shp_rms_rel,
stage1_diag, stage2_diag, aniso_history, surgery, ...`.

## Status on test_set (target_verts~2000, L_max=16)
- 7/9 **bijective** (~0.7% RMS): BDH6230 brain, echinocyte, mushroom(x4), zebrafish.
- `1dpx` **near_bijective** (1 fold, ~1% RMS) — localized kernel-empty tangle.
- `hydra_full_smooth` **too_complex** — `Q=0.007` (sphere=1.0), folds at ALL
  resolutions; conformal maps fundamentally cannot embed it (tested: more faces
  made it worse). Flagged, not fought. (1dpx, a real protein, is Q=0.200 and fine.)

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

## NEXT TASK (in progress): grid-quadrature high-L_max SH analysis + L_max criterion
User decisions: do it AFTER props() (done); KEEP the `sh_basis` 'bosh' convention
(so `.shp3` / existing coefficients stay compatible) -> use quadrature, not pyshtools.

Plan:
1. `resample_to_grid(param_mesh, basis)`: barycentric-interpolate `x,y,z` onto the
   basis grid (`basis.p`, `basis.t`; `gdim x gdim` points). Point location: KDTree on
   the param sphere vertices, then test/interp in incident faces (spherical
   barycentric). Returns f_grid (Ngrid x 3).
2. Quadrature projection that INVERTS the synthesis (`shp_surface.update` does
   `f = sum_lk c_lk Y_lk`): if the basis is orthogonal under the quadrature inner
   product `<a,b>=sum_g w_g a_g b_g`, then
      `c_lk = (sum_g w_g f_g Y_lk_g) / (sum_g w_g Y_lk_g^2)`  (per x/y/z channel).
   Use `basis.Y` reshaped `(Ngrid, Ncoeff)` and `basis.w` (Ngrid). Matrix-vector,
   ~1GB at L=72, no SVD. **VALIDATE round-trip** (synthesize known c -> grid ->
   analyze -> recover c within tol). If off-diagonal coupling is non-negligible,
   fall back to grid least-squares (pinv of the (Ngrid x Ncoeff) basis). Use
   `gdim >= ~2*L_max` for accuracy.
3. `complexity -> L_max` criterion: from the SH power spectrum
   `E_l = sqrt(sum_{m,ch} c_{l,m,ch}^2)`, recommend the smallest `L` where
   cumulative energy >= ~99.5% (or where `E_l` drops below the recon-error floor).
4. Validate on `1dpx` at L=16/32/48/72: RMS should keep dropping; report
   recommended `L_max`.
`sh_basis` facts: `sh_basis(L_max, gdim)` builds Gauss-quadrature grid; `.p/.t`
meshgrid `(gdim,gdim)`; `.w` flattened weights `(gdim^2,)`; `.Y` `(gdim,gdim,(L+1)^2)`;
`ylk_bosh(L,K,phi,theta)`. Default `gdim=30`.

## Other backlog
- Optional: route `too_complex` shapes to the old patch-based pipeline (user
  prefers NOT to; flagging is the default).
- Optional: edge-flip surgery for kernel-empty single folds (1dpx).
- Profile other O(V*F) loops for 14-15k-vertex meshes if preprocess is slow.
