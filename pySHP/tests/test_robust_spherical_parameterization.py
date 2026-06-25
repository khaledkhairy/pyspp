"""Tests for robust curvature-aware spherical parameterization."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pySHP.surface_mesh import surface_mesh
from pySHP.utils import readoff
from pySHP.level1.target_areas import compute_curvature_target_areas
from pySHP.level1.newton_multi_objective import (
    multi_objective_energy, newton_multi_objective, default_multi_objective_opts)
from pySHP.level1.bijectivity_gate import (
    check_bijectivity_gate, compute_parametric_quality)


CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MUSHROOM = os.path.join(
    CODE_DIR, 'Matlab', 'shp_toolbox-main', 'shp_toolbox-main',
    'test_data', 'off', 'test_set', 'mushroom_repaired_03.off')


class TestTargetAreas(unittest.TestCase):
    def test_sums_to_four_pi(self):
        X, F = readoff(MUSHROOM)
        m = surface_mesh(X, F)
        m.props()
        Ao, weights = compute_curvature_target_areas(m)
        self.assertEqual(len(Ao), len(F))
        self.assertAlmostEqual(np.sum(Ao), 4 * np.pi, places=4)
        self.assertTrue(np.all(Ao > 0))
        self.assertTrue(np.all(weights > 0))


class TestNewtonMultiObjective(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, F = readoff(MUSHROOM)
        m = surface_mesh(X, F)
        m.repair_mesh(verbose=False)
        m_dec, _ = m.curvature_aware_decimation(target_faces=150, verbose=False)
        m_dec.optimization_method = 0
        m_dec.newton_niter = 0
        m_dec.bijective_plot_flag = 0
        m_dec.map2sphere()
        cls.m = m_dec
        cls.Ao, _ = compute_curvature_target_areas(m_dec)

    def test_energy_runs(self):
        E, A, flipped, _ = multi_objective_energy(
            self.m.t, self.m.p, self.m.F, self.Ao)
        self.assertEqual(len(E), len(self.m.F))
        self.assertEqual(len(A), len(self.m.F))

    def test_optimizer_runs(self):
        opts = default_multi_objective_opts(maxiter=2, verbose=0)
        t, p, residuals, report = newton_multi_objective(
            self.m.t, self.m.p, self.m.F, self.Ao, opts)
        self.assertEqual(len(t), len(self.m.t))
        self.assertEqual(len(p), len(self.m.p))
        self.assertGreater(len(residuals), 0)


class TestRobustPipeline(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(MUSHROOM), 'mushroom test mesh missing')
    def test_pipeline_runs(self):
        from pySHP.level2.robust_spherical_parameterization import (
            robust_spherical_parameterization, export_robust_shp)

        X, F = readoff(MUSHROOM)
        m = surface_mesh(X, F)
        m_param, shp, report = robust_spherical_parameterization(
            m,
            target_faces_final=300,
            coarse_faces=100,
            remesh_input=False,
            optimization_niter=2,
            optimization_step=0.08,
            weights=dict(lambdaA=1.0, lambda_flip=1e4, lambda1=1e-4, lambda2=1e-2),
            verbose=False,
        )
        self.assertIsNotNone(m_param.t)
        self.assertIsNotNone(m_param.p)
        gate = report['acceptance']
        self.assertIn('area_correlation', gate)
        qual = compute_parametric_quality(m_param)
        self.assertGreater(qual['mean_quality'], 0.0)

        out = os.path.join(os.path.dirname(__file__), '_test_robust.shp3')
        try:
            err = export_robust_shp(m_param, shp, out, verbose=False)
            self.assertGreaterEqual(err, 0.0)
            self.assertTrue(os.path.exists(out))
        finally:
            if os.path.exists(out):
                os.remove(out)


if __name__ == '__main__':
    unittest.main()
