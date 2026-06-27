"""Quick CLI test of the cMCF spherical-parameterization pipeline.

Runs ``parameterize_to_sphere`` over meshes in the test set and prints a summary
(method, bijectivity, fold count, SHP reconstruction RMS, time). This is the
fast way to sanity-check the pipeline end-to-end at a chosen resolution /
bandwidth without the full batch notebook (which also writes the 5 artifacts).

Usage (from the repo's ``code`` dir):
    python pySHP/tests/run_parameterization_test.py
    python pySHP/tests/run_parameterization_test.py mushroom_repaired_03 hydra_full_smooth
    python pySHP/tests/run_parameterization_test.py --verts 2000 --lmax 16 1dpx

Defaults: target_verts=7000, L_max=60 (matching the batch notebook).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from pySHP.surface_mesh import surface_mesh                       # noqa: E402
from pySHP.utils import readoff                                   # noqa: E402
import pySHP.level2.cmcf_spherical_parameterization as cmcf       # noqa: E402

TEST_DIR = os.path.join(CODE, 'Matlab', 'shp_toolbox-main', 'shp_toolbox-main',
                        'test_data', 'off', 'test_set')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('names', nargs='*', help='mesh names (no .off); default=all')
    ap.add_argument('--verts', type=int, default=7000, help='target_verts')
    ap.add_argument('--lmax', type=int, default=60, help='SHP L_max')
    ap.add_argument('--aniso', type=int, default=3, help='anisotropic rounds')
    args = ap.parse_args()

    if args.names:
        paths = [os.path.join(TEST_DIR, n + '.off') for n in args.names]
    else:
        paths = sorted(glob.glob(os.path.join(TEST_DIR, '*.off')))

    hdr = (f"{'name':<22} {'n_in':>6} {'n_par':>6} {'method':>11} "
           f"{'quality':>14} {'folds':>6} {'rms%':>6} {'t(s)':>6}")
    print(f"\ntarget_verts={args.verts}  L_max={args.lmax}  aniso={args.aniso}\n")
    print(hdr)
    print('-' * len(hdr))

    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            X, F = readoff(p)
            m = surface_mesh(np.asarray(X, float), np.asarray(F, int))
            n_in = len(m.X)
            t0 = time.time()
            res = cmcf.parameterize_to_sphere(
                m, target_verts=args.verts, aniso_rounds=args.aniso,
                fit_shp_L_max=args.lmax, verbose=False)
            dt = time.time() - t0
            pm = res['mesh']
            print(f"{name:<22} {n_in:>6} {len(pm.X):>6} "
                  f"{str(res.get('method')):>11} {str(res.get('quality')):>14} "
                  f"{res.get('n_foldovers', -1):>6} "
                  f"{100 * res.get('shp_rms_rel', float('nan')):>6.2f} {dt:>6.0f}")
        except Exception as exc:                                  # noqa: BLE001
            print(f"{name:<22} FAILED: {exc!r}")

    print("\nDone.")


if __name__ == '__main__':
    main()
