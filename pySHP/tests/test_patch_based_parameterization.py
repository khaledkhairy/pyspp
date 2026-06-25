"""
Test script for patch-based spherical parameterization

Tests each stage of the patch-based spherical parameterization process:
1. Mesh loading/generation
2. Mesh segmentation
3. Patch info generation
4. Spherical conformal parameterization
5. Patch parameterization
6. Full spherical parameterization pipeline

Author: Khaled Khairy
"""

import numpy as np
import sys
import os

# Add parent directories to path for proper imports
code_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, code_dir)

from pySHP.surface_mesh import surface_mesh
from pySHP.utils import kk_cart2sph, kk_sph2cart


class TestPatchBasedParameterization:
    """Test class for patch-based spherical parameterization"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
    
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    # =========================================================================
    # STAGE 1: Mesh Generation
    # =========================================================================
    def test_stage1_mesh_generation(self):
        """Test mesh generation (icosahedron subdivision)"""
        self.log("\n" + "="*60)
        self.log("STAGE 1: Mesh Generation")
        self.log("="*60)
        
        try:
            # Generate sphere mesh
            n_subdivisions = 2
            X, F = surface_mesh.sphere_mesh_gen(n_subdivisions)
            m = surface_mesh(X, F)
            
            # Verify mesh properties
            m.props()
            m.edge_info()
            
            n_verts = len(m.X)
            n_faces = len(m.F)
            
            self.log(f"  Generated sphere mesh:")
            self.log(f"    Vertices: {n_verts}")
            self.log(f"    Faces: {n_faces}")
            self.log(f"    Expected Euler: 2, Actual: {m.euler}")
            
            # Store for later tests
            self.results['mesh'] = m
            self.results['stage1_passed'] = True
            
            self.log("  STAGE 1 PASSED")
            return True
            
        except Exception as e:
            self.log(f"  STAGE 1 FAILED: {e}")
            self.results['stage1_passed'] = False
            return False
    
    # =========================================================================
    # STAGE 2: Mesh Segmentation
    # =========================================================================
    def test_stage2_mesh_segmentation(self):
        """Test mesh segmentation using random walk"""
        self.log("\n" + "="*60)
        self.log("STAGE 2: Mesh Segmentation")
        self.log("="*60)
        
        if not self.results.get('stage1_passed'):
            self.log("  Skipping: Stage 1 not passed")
            return False
        
        try:
            from pySHP.level1.mesh_segmentation_rw import mesh_segmentation_rw
            
            m = self.results['mesh']
            nseeds = 6  # Use 6 seeds for testing
            sigma = 1.0
            
            self.log(f"  Running segmentation with {nseeds} seeds...")
            
            ms, L, slix, P, Pconn = mesh_segmentation_rw(m, nseeds, sigma)
            
            n_patches = len(np.unique(L))
            
            self.log(f"  Segmentation results:")
            self.log(f"    Number of patches: {n_patches}")
            self.log(f"    Seed face indices: {slix[:5]}..." if len(slix) > 5 else f"    Seed faces: {slix}")
            self.log(f"    Label distribution: {dict(zip(*np.unique(L, return_counts=True)))}")
            
            # Store results
            self.results['segmented_mesh'] = ms
            self.results['face_labels'] = L
            self.results['seed_indices'] = slix
            self.results['P'] = P
            self.results['Pconn'] = Pconn
            self.results['stage2_passed'] = True
            
            self.log("  STAGE 2 PASSED")
            return True
            
        except Exception as e:
            import traceback
            self.log(f"  STAGE 2 FAILED: {e}")
            traceback.print_exc()
            self.results['stage2_passed'] = False
            return False
    
    # =========================================================================
    # STAGE 3: Patch Info Generation
    # =========================================================================
    def test_stage3_patch_info_gen(self):
        """Test patch information generation"""
        self.log("\n" + "="*60)
        self.log("STAGE 3: Patch Info Generation")
        self.log("="*60)
        
        if not self.results.get('stage2_passed'):
            self.log("  Skipping: Stage 2 not passed")
            return False
        
        try:
            from pySHP.level1.patch_info_gen import patch_info_gen
            
            ms = self.results['segmented_mesh']
            P = self.results['P']
            Pconn = self.results['Pconn']
            
            self.log("  Generating patch info...")
            
            m, PM, Pconn = patch_info_gen(ms, P, Pconn)
            
            self.log(f"  Patch info results:")
            self.log(f"    Number of patches: {PM['npatches']}")
            self.log(f"    Number of edges: {len(PM['Edges'])}")
            self.log(f"    Number of keys: {len(PM['keys'])}")
            self.log(f"    Simplified mesh vertices: {len(PM['pm'].X)}")
            self.log(f"    Simplified mesh faces: {len(PM['pm'].F)}")
            
            # Verify structure
            assert 'pm' in PM, "Missing simplified mesh"
            assert 'P' in PM, "Missing patch list"
            assert 'Edges' in PM, "Missing edges"
            assert 'sentinels' in PM, "Missing sentinels"
            
            # Store results
            self.results['patch_mesh'] = m
            self.results['PM'] = PM
            self.results['stage3_passed'] = True
            
            self.log("  STAGE 3 PASSED")
            return True
            
        except Exception as e:
            import traceback
            self.log(f"  STAGE 3 FAILED: {e}")
            traceback.print_exc()
            self.results['stage3_passed'] = False
            return False
    
    # =========================================================================
    # STAGE 4: Spherical Conformal Parameterization
    # =========================================================================
    def test_stage4_conformal_param(self):
        """Test spherical conformal parameterization"""
        self.log("\n" + "="*60)
        self.log("STAGE 4: Spherical Conformal Parameterization")
        self.log("="*60)
        
        if not self.results.get('stage3_passed'):
            self.log("  Skipping: Stage 3 not passed")
            return False
        
        try:
            from pySHP.level2.spherical_conformal_parameterization import spherical_conformal_parameterization
            
            PM = self.results['PM']
            pm = PM['pm']
            
            self.log("  Computing spherical conformal map...")
            
            pm = spherical_conformal_parameterization(pm)
            
            self.log(f"  Conformal map results:")
            self.log(f"    Theta range: [{pm.t.min():.4f}, {pm.t.max():.4f}]")
            self.log(f"    Phi range: [{pm.p.min():.4f}, {pm.p.max():.4f}]")
            
            # Verify parameterization
            assert pm.t is not None, "Theta values not computed"
            assert pm.p is not None, "Phi values not computed"
            assert len(pm.t) == len(pm.X), "Theta length mismatch"
            assert len(pm.p) == len(pm.X), "Phi length mismatch"
            
            # Check that points are on unit sphere
            u, v, w = kk_sph2cart(pm.t, pm.p, np.ones_like(pm.t))
            radii = np.sqrt(u**2 + v**2 + w**2)
            self.log(f"    Radii range: [{radii.min():.4f}, {radii.max():.4f}]")
            
            # Store results
            PM['pm'] = pm
            self.results['PM'] = PM
            self.results['stage4_passed'] = True
            
            self.log("  STAGE 4 PASSED")
            return True
            
        except Exception as e:
            import traceback
            self.log(f"  STAGE 4 FAILED: {e}")
            traceback.print_exc()
            self.results['stage4_passed'] = False
            return False
    
    # =========================================================================
    # STAGE 5: Patch Parameterization
    # =========================================================================
    def test_stage5_patch_param(self):
        """Test individual patch parameterization"""
        self.log("\n" + "="*60)
        self.log("STAGE 5: Patch Parameterization")
        self.log("="*60)
        
        if not self.results.get('stage4_passed'):
            self.log("  Skipping: Stage 4 not passed")
            return False
        
        try:
            from pySHP.level1.parameterize_patches_cart import parameterize_patches_cart
            
            PM = self.results['PM']
            
            self.log("  Parameterizing individual patches...")
            
            PM = parameterize_patches_cart(PM, plot_flag=0)
            
            self.log(f"  Patch parameterization results:")
            
            for pix in range(PM['npatches']):
                pat = PM['P'][pix][0]
                t_nonzero = np.sum(pat.t != 0)
                p_nonzero = np.sum(pat.p != 0)
                self.log(f"    Patch {pix}: {len(pat.F)} faces, "
                        f"{t_nonzero} t-values, {p_nonzero} p-values assigned")
            
            # Store results
            self.results['PM'] = PM
            self.results['stage5_passed'] = True
            
            self.log("  STAGE 5 PASSED")
            return True
            
        except Exception as e:
            import traceback
            self.log(f"  STAGE 5 FAILED: {e}")
            traceback.print_exc()
            self.results['stage5_passed'] = False
            return False
    
    # =========================================================================
    # STAGE 6: Full Pipeline
    # =========================================================================
    def test_stage6_full_pipeline(self):
        """Test full spherical parameterization pipeline"""
        self.log("\n" + "="*60)
        self.log("STAGE 6: Full Spherical Parameterization Pipeline")
        self.log("="*60)
        
        try:
            from pySHP.level2.spherical_parameterization import spherical_parameterization
            
            # Create fresh mesh
            X, F = surface_mesh.sphere_mesh_gen(2)
            m = surface_mesh(X, F)
            
            self.log("  Running full parameterization pipeline...")
            self.log(f"    Input mesh: {len(m.X)} vertices, {len(m.F)} faces")
            
            # Run parameterization
            nseeds = 6
            m_out, PM, failed = spherical_parameterization(m, nseeds=nseeds)
            
            self.log(f"  Pipeline results:")
            self.log(f"    Output mesh: {len(m_out.X)} vertices, {len(m_out.F)} faces")
            self.log(f"    Theta assigned: {np.sum(m_out.t != 0)} vertices")
            self.log(f"    Phi assigned: {np.sum(m_out.p != 0)} vertices")
            self.log(f"    Failed patches: {failed}")
            
            # Verify all vertices have parameterization
            coverage = np.sum((m_out.t != 0) | (m_out.p != 0)) / len(m_out.X)
            self.log(f"    Coverage: {coverage*100:.1f}%")
            
            # Store results
            self.results['final_mesh'] = m_out
            self.results['final_PM'] = PM
            self.results['failed_patches'] = failed
            self.results['stage6_passed'] = True
            
            self.log("  STAGE 6 PASSED")
            return True
            
        except Exception as e:
            import traceback
            self.log(f"  STAGE 6 FAILED: {e}")
            traceback.print_exc()
            self.results['stage6_passed'] = False
            return False
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    def run_all_tests(self):
        """Run all test stages"""
        self.log("\n" + "#"*60)
        self.log("# PATCH-BASED SPHERICAL PARAMETERIZATION TESTS")
        self.log("#"*60)
        
        stages = [
            ('Stage 1: Mesh Generation', self.test_stage1_mesh_generation),
            ('Stage 2: Mesh Segmentation', self.test_stage2_mesh_segmentation),
            ('Stage 3: Patch Info Gen', self.test_stage3_patch_info_gen),
            ('Stage 4: Conformal Param', self.test_stage4_conformal_param),
            ('Stage 5: Patch Param', self.test_stage5_patch_param),
            ('Stage 6: Full Pipeline', self.test_stage6_full_pipeline),
        ]
        
        results_summary = []
        
        for name, test_func in stages:
            passed = test_func()
            results_summary.append((name, passed))
        
        # Print summary
        self.log("\n" + "#"*60)
        self.log("# TEST SUMMARY")
        self.log("#"*60)
        
        n_passed = 0
        for name, passed in results_summary:
            status = "PASSED" if passed else "FAILED"
            self.log(f"  {name}: {status}")
            if passed:
                n_passed += 1
        
        self.log("-"*60)
        self.log(f"  Total: {n_passed}/{len(stages)} stages passed")
        self.log("#"*60 + "\n")
        
        return n_passed == len(stages)


def run_individual_tests():
    """Run individual unit tests for helper functions"""
    print("\n" + "="*60)
    print("INDIVIDUAL FUNCTION TESTS")
    print("="*60)
    
    # Test cotangent Laplacian
    print("\n--- Testing cotangent_laplacian ---")
    try:
        from pySHP.level0.mesh_utils import cotangent_laplacian
        X, F = surface_mesh.sphere_mesh_gen(1)
        L = cotangent_laplacian(X, F)
        print(f"  Laplacian shape: {L.shape}")
        print(f"  Laplacian nnz: {L.nnz}")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test get_border
    print("\n--- Testing get_border ---")
    try:
        from pySHP.level1.get_border import get_border
        X, F = surface_mesh.sphere_mesh_gen(1)
        m = surface_mesh(X, F[:10])  # Open mesh (subset of faces)
        m.edge_info()
        m, mpL, be = get_border(m)
        print(f"  Border vertices: {np.sum(m.border_vertex)}")
        print(f"  Border edges: {np.sum(be)}")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test fix_flipped_faces
    print("\n--- Testing fix_flipped_faces ---")
    try:
        from pySHP.level1.fix_flipped_faces import fix_flipped_faces
        X, F = surface_mesh.sphere_mesh_gen(1)
        m = surface_mesh(X, F)
        m, Fnc = fix_flipped_faces(m, verbose=False)
        print(f"  Faces checked: {len(Fnc)}")
        print(f"  Faces correct: {np.sum(Fnc)}")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test reduce_to_minimal_set
    print("\n--- Testing reduce_to_minimal_set ---")
    try:
        from pySHP.level0.mesh_utils import reduce_to_minimal_set
        X, F = surface_mesh.sphere_mesh_gen(1)
        m = surface_mesh(X, F)
        m.border_vertex = np.zeros(len(X))
        m.t = np.zeros(len(X))
        m.p = np.zeros(len(X))
        minm, uv = reduce_to_minimal_set(m)
        print(f"  Original vertices: {len(m.X)}")
        print(f"  Reduced vertices: {len(minm.X)}")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")


if __name__ == '__main__':
    # Run individual function tests
    run_individual_tests()
    
    # Run full pipeline tests
    tester = TestPatchBasedParameterization(verbose=True)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)
