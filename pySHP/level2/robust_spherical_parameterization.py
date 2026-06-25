"""
Robust curvature-aware spherical parameterization pipeline.

Stages:
  0. Repair + curvature-adaptive remesh (input quality)
  1. Coarse bijective map (decimate -> map2sphere -> multi-objective Newton)
  2. Propagate to working mesh + re-optimize
  3. Optional densify loop
  4. SHP projection with reconstruction error check
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
from ..level1.interpolate_fine_mesh_from_decimated import (
    interpolate_fine_mesh_parameterization,
)
from ..level1.newton_multi_objective import default_multi_objective_opts


def _copy_mesh(m):
    mc = surface_mesh(m.X.copy(), m.F.copy())
    if m.face_labels is not None:
        mc.face_labels = m.face_labels.copy()
    return mc


def _configure_multi_objective(m, weights, niter, step, prevent_flip=True):
    m.optimization_method = 6
    m.newton_niter = niter
    m.newton_step = step
    m.prevent_flip = prevent_flip
    m.bijective_plot_flag = 0
    m.needs_map2sphere = True
    opts = default_multi_objective_opts(
        lambdaA=weights.get('lambdaA', 1.0),
        lambda_flip=weights.get('lambda_flip', 1e3),
        lambda1=weights.get('lambda1', 1e-4),
        lambda2=weights.get('lambda2', 1e-2),
        prevent_flip=prevent_flip,
        maxiter=niter,
        stepfac=step,
    )
    m.multi_objective_opts = opts
    m.target_areas = None


def _run_map2sphere_with_multi_objective(m, weights, niter, step, verbose):
    _configure_multi_objective(m, weights, niter, step)
    if verbose:
        print(f"  map2sphere + method 6 on {len(m.X)} verts, {len(m.F)} faces")
    m.map2sphere()
    gate = check_bijectivity_gate(m, verbose=verbose)
    Ao, _ = compute_curvature_target_areas(m)
    signed, achieved = compute_achieved_spherical_areas(m)
    quality = compute_parametric_quality(m)
    corr = _area_correlation(Ao, np.abs(achieved))
    return gate, Ao, achieved, quality, corr


def _area_correlation(Ao, achieved_abs):
    if len(Ao) < 2:
        return 1.0
    if np.std(Ao) < 1e-15 or np.std(achieved_abs) < 1e-15:
        return 0.0
    return float(np.corrcoef(Ao, achieved_abs)[0, 1])


def _input_quality_gate(m, min_mean_quality=0.7, min_quality=0.3, verbose=True):
    stats = m.get_mesh_quality_stats()
    issues = []
    if stats['mean_quality'] < min_mean_quality:
        issues.append(f"mean quality {stats['mean_quality']:.3f} < {min_mean_quality}")
    if stats['min_quality'] < min_quality:
        issues.append(f"min quality {stats['min_quality']:.3f} < {min_quality}")
    passed = len(issues) == 0
    if verbose:
        status = 'PASSED' if passed else 'FAILED'
        print(f"Input quality gate: {status}")
        for issue in issues:
            print(f"  - {issue}")
    return passed, stats, issues


def _shp_reconstruction_error(m_param, L_max=16):
    """Fit SHP and measure RMS least-squares fit error at input mesh vertices."""
    s = shp_surface(L_max)
    s.mesh2shp(m_param, L_max)
    if hasattr(s, 'residual') and s.residual is not None:
        err = float(np.sqrt(np.mean(np.sum(s.residual ** 2, axis=1))))
    else:
        err = 0.0
    return err, s


def _match_vertices_by_position(X_src, X_dst, tol=1e-6):
    """Map each src vertex index to matching dst vertex index (-1 if missing)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(X_dst)
    dist, idx = tree.query(X_src, k=1)
    vert_map = np.where(dist <= tol, idx, -1).astype(int)
    return vert_map


def robust_spherical_parameterization(
        mesh,
        target_faces_final=None,
        coarse_faces=1000,
        weights=None,
        densify_steps=None,
        remesh_input=True,
        input_target_faces=None,
        max_coarse_retries=3,
        optimization_niter=200,
        optimization_step=0.1,
        L_max=16,
        area_tol=0.05,
        verbose=True):
    """Full pipeline from arbitrary genus-0 mesh to SHP parameterization.

    Parameters
    ----------
    mesh : surface_mesh
        Input closed genus-0 mesh.
    target_faces_final : int, optional
        Target face count for final working mesh. Defaults to input face count.
    coarse_faces : int
        Face count for coarse parameterization mesh.
    weights : dict, optional
        Multi-objective weights (lambdaA, lambda_flip, lambda1, lambda2).
    densify_steps : list of int, optional
        Face counts for iterative densification levels (ascending).
    remesh_input : bool
        If True, apply curvature-adaptive remesh before parameterization.
    input_target_faces : int, optional
        Face count after input remesh (defaults to max(target_faces_final, coarse_faces*2)).
    max_coarse_retries : int
        Retries with coarser mesh / higher flip penalty if bijectivity gate fails.
    optimization_niter, optimization_step : int/float
        Newton iteration budget and step factor for method 6.
    L_max : int
        Spherical harmonics bandwidth for final SHP fit.
    area_tol : float
        Relative tolerance for bijectivity area gate.
    verbose : bool

    Returns
    -------
    m_param : surface_mesh
        Final parameterized mesh with .t, .p on working resolution.
    shp : shp_surface
        Fitted SHP surface.
    report : dict
        Stage-by-stage diagnostics and acceptance metrics.
    """
    if weights is None:
        weights = dict(lambdaA=1.0, lambda_flip=1e3, lambda1=1e-4, lambda2=1e-2)

    report = {'stages': {}, 'weights': weights.copy()}

    # ---- Stage 0: input canonization ----
    if verbose:
        print("=" * 60)
        print("Stage 0: Input repair and remesh")
        print("=" * 60)

    m_work = _copy_mesh(mesh)
    m_work.repair_mesh(verbose=verbose)
    m_work.props()

    if remesh_input:
        if input_target_faces is None:
            input_target_faces = target_faces_final or len(m_work.F)
            input_target_faces = max(int(input_target_faces),
                                     int(coarse_faces * 2))
        if len(m_work.F) != input_target_faces:
            if verbose:
                print(f"  remesh_by_curvature -> {input_target_faces} faces")
            m_work.remesh_by_curvature(
                target_faces=input_target_faces,
                curvature_strength=2.0,
                verbose=verbose)

    q_pass, q_stats, q_issues = _input_quality_gate(m_work, verbose=verbose)
    report['stages']['input'] = {
        'quality': q_stats,
        'gate_passed': q_pass,
        'issues': q_issues,
        'n_verts': len(m_work.X),
        'n_faces': len(m_work.F),
    }

    if target_faces_final is None:
        target_faces_final = len(m_work.F)

    # ---- Stage 1: coarse bijective map ----
    if verbose:
        print("\n" + "=" * 60)
        print("Stage 1: Coarse parameterization")
        print("=" * 60)

    coarse_target = coarse_faces
    gate = None
    m_coarse = None
    vert_map = None

    for attempt in range(max_coarse_retries):
        attempt_weights = weights.copy()
        if attempt > 0:
            coarse_target = max(200, int(coarse_target * 0.7))
            attempt_weights['lambda_flip'] = weights.get('lambda_flip', 1e3) * (10 ** attempt)
            if verbose:
                print(f"  Retry {attempt}: coarse_faces={coarse_target}, "
                      f"lambda_flip={attempt_weights['lambda_flip']:.0e}")

        m_coarse, vert_map = m_work.curvature_aware_decimation(
            target_faces=coarse_target, verbose=verbose)
        m_coarse = _copy_mesh(m_coarse)
        m_coarse.t = None
        m_coarse.p = None

        gate, Ao, achieved, quality, corr = _run_map2sphere_with_multi_objective(
            m_coarse, attempt_weights, optimization_niter, optimization_step, verbose)

        report['stages'][f'coarse_attempt_{attempt}'] = {
            'gate': gate,
            'area_correlation': corr,
            'parametric_quality': quality,
            'coarse_faces': len(m_coarse.F),
        }

        if gate['passed'] or gate['area_excess_rel'] < area_tol:
            break

    report['stages']['coarse'] = report['stages'].get(
        f'coarse_attempt_{attempt}', {})

    # ---- Stage 2: propagate to working mesh + re-optimize ----
    if verbose:
        print("\n" + "=" * 60)
        print("Stage 2: Propagate to working mesh and re-optimize")
        print("=" * 60)

    if len(m_work.F) > target_faces_final:
        if verbose:
            print(f"  decimate working mesh to {target_faces_final} faces")
        m_work, _ = m_work.curvature_aware_decimation(
            target_faces=target_faces_final, verbose=verbose)

    t_full, p_full, interp_report = interpolate_fine_mesh_parameterization(
        m_work.X, m_work.F,
        m_coarse.X, m_coarse.F,
        m_coarse.t, m_coarse.p,
        vert_map,
        verbose=verbose,
    )

    m_param = surface_mesh(m_work.X.copy(), m_work.F.copy())
    m_param.t = t_full
    m_param.p = p_full
    m_param.ixN = m_coarse.ixN
    m_param.ixS = m_coarse.ixS

    _configure_multi_objective(
        m_param, weights, optimization_niter, optimization_step)
    m_param.needs_map2sphere = False
    m_param.t = t_full
    m_param.p = p_full
    from ..level1.newton_multi_objective import newton_multi_objective
    Ao_full, _ = compute_curvature_target_areas(m_param)
    opts = m_param.multi_objective_opts.copy()
    fine_niter = min(
        optimization_niter,
        max(5, 40000 // max(len(m_param.X) + len(m_param.F), 1)))
    opts['maxiter'] = fine_niter
    opts['stepfac'] = optimization_step
    opts['verbose'] = 1 if verbose else 0
    fixed = []
    if m_param.ixN:
        fixed.append(int(m_param.ixN))
    if m_param.ixS:
        fixed.append(int(m_param.ixS))
    if fixed:
        opts['fixed'] = np.array(fixed, dtype=int)
    t_opt, p_opt, residuals, mo_report = newton_multi_objective(
        m_param.t, m_param.p, m_param.F, Ao_full, opts)
    m_param.t = t_opt
    m_param.p = p_opt
    m_param.target_areas = Ao_full
    m_param.newton_residuals = residuals

    gate_full = check_bijectivity_gate(m_param, area_tol=area_tol, verbose=verbose)
    _, achieved_full = compute_achieved_spherical_areas(m_param)
    quality_full = compute_parametric_quality(m_param)
    corr_full = _area_correlation(Ao_full, np.abs(achieved_full))

    report['stages']['fine'] = {
        'interpolation': interp_report,
        'gate': gate_full,
        'area_correlation': corr_full,
        'parametric_quality': quality_full,
        'multi_objective': mo_report,
    }

    # ---- Stage 3: optional densify loop ----
    if densify_steps:
        if verbose:
            print("\n" + "=" * 60)
            print("Stage 3: Iterative densification")
            print("=" * 60)

        densify_reports = []
        for step_target in densify_steps:
            if len(m_param.F) >= step_target:
                continue
            if verbose:
                print(f"  densify -> {step_target} faces")

            m_prev = _copy_mesh(m_param)
            t_prev = m_param.t.copy()
            p_prev = m_param.p.copy()
            X_prev = m_param.X.copy()
            F_prev = m_param.F.copy()

            m_param.densify(target_faces=step_target)

            vert_map = _match_vertices_by_position(X_prev, m_param.X)
            t_new, p_new, interp_rep = interpolate_fine_mesh_parameterization(
                m_param.X, m_param.F,
                X_prev, F_prev,
                t_prev, p_prev,
                np.maximum(vert_map, 0),
                verbose=False,
            )
            m_param.t = t_new
            m_param.p = p_new

            Ao_step, _ = compute_curvature_target_areas(m_param)
            t_opt, p_opt, res, mo_rep = newton_multi_objective(
                m_param.t, m_param.p, m_param.F, Ao_step, opts)
            m_param.t = t_opt
            m_param.p = p_opt

            gate_step = check_bijectivity_gate(m_param, area_tol=area_tol, verbose=False)
            qual_step = compute_parametric_quality(m_param)
            densify_reports.append({
                'target_faces': step_target,
                'gate': gate_step,
                'quality': qual_step,
            })

        report['stages']['densify'] = densify_reports

    # ---- Stage 4: SHP projection ----
    if verbose:
        print("\n" + "=" * 60)
        print("Stage 4: SHP projection")
        print("=" * 60)

    shp_err, shp = _shp_reconstruction_error(m_param, L_max=L_max)
    if verbose:
        print(f"  SHP reconstruction RMS error: {shp_err:.6f}")

    acceptance = {
        'bijectivity_passed': gate_full['passed'],
        'area_correlation': corr_full,
        'mean_parametric_quality': quality_full['mean_quality'],
        'min_parametric_quality': quality_full['min_quality'],
        'shp_reconstruction_rms': shp_err,
        'area_excess_rel': gate_full['area_excess_rel'],
        'n_foldovers': gate_full['n_foldovers'],
    }
    report['acceptance'] = acceptance

    if verbose:
        print("\n" + "=" * 60)
        print("Acceptance summary")
        print("=" * 60)
        for k, v in acceptance.items():
            print(f"  {k}: {v}")

    return m_param, shp, report


def export_robust_shp(m_param, shp, output_path, verbose=True):
    """Export .shp3 and return reconstruction error."""
    shp.export_shp3(output_path)
    err, _ = _shp_reconstruction_error(m_param, L_max=shp.L_max)
    if verbose:
        print(f"Exported {output_path}, reconstruction RMS={err:.6f}")
    return err
