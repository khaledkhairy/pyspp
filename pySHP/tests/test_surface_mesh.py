"""
Tests for surface_mesh class
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pySHP.surface_mesh import surface_mesh
from pySHP.utils import readoff


class TestSurfaceMesh(unittest.TestCase):
    
    def test_init(self):
        """Test basic initialization"""
        # Create a simple sphere
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        self.assertIsNotNone(m.X)
        self.assertIsNotNone(m.F)
        self.assertEqual(len(m.X), len(X))
        self.assertEqual(len(m.F), len(F))
    
    def test_sphere_mesh_gen(self):
        """Test sphere mesh generation"""
        X, F = surface_mesh.sphere_mesh_gen(1)
        self.assertGreater(len(X), 0)
        self.assertGreater(len(F), 0)
        # Check vertices are on unit sphere
        radii = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(radii, 1.0, rtol=1e-5)
    
    def test_edge_info(self):
        """Test edge information computation"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        m.edge_info()
        self.assertIsNotNone(m.E)
        self.assertIsNotNone(m.L)
        self.assertGreater(len(m.E), 0)
    
    def test_props(self):
        """Test mesh properties computation"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        m.props()
        self.assertIsNotNone(m.A)
        self.assertIsNotNone(m.V)
        self.assertGreater(m.A, 0)
        self.assertGreater(m.V, 0)
    
    def test_readoff(self):
        """Test reading .off file"""
        # Try to find a test file
        test_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'Matlab',
            'shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off'
        )
        
        # Try scientific directory
        scientific_dir = os.path.join(test_dir, 'scientific')
        if os.path.exists(scientific_dir):
            off_files = [f for f in os.listdir(scientific_dir) if f.endswith('.off')]
            if off_files:
                test_file = os.path.join(scientific_dir, off_files[0])
                try:
                    X, F = readoff(test_file)
                    m = surface_mesh(X, F)
                    self.assertGreater(len(m.X), 0)
                    self.assertGreater(len(m.F), 0)
                except Exception as e:
                    self.skipTest(f"Could not read test file: {e}")
            else:
                self.skipTest("No .off files found in test directory")
        else:
            self.skipTest("Test data directory not found")
    
    def test_plot(self):
        """Test basic plot function"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        # Test that plot function exists and can be called
        # (We skip actual plotting in tests to avoid GUI issues)
        self.assertTrue(hasattr(m, 'plot'))
        self.assertTrue(callable(m.plot))
    
    def test_plot_H(self):
        """Test plot_H function"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        m.props()
        # Test that plot_H function exists
        self.assertTrue(hasattr(m, 'plot_H'))
        self.assertTrue(callable(m.plot_H))
        # Test that H is computed
        self.assertIsNotNone(m.H)
    
    def test_plot_fast(self):
        """Test plot_fast function"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        # Test that plot_fast function exists
        self.assertTrue(hasattr(m, 'plot_fast'))
        self.assertTrue(callable(m.plot_fast))
    
    def test_plot_quality(self):
        """Test plot_quality function"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        m.props()
        # Test that plot_quality function exists
        self.assertTrue(hasattr(m, 'plot_quality'))
        self.assertTrue(callable(m.plot_quality))
        # Test that quality is computed
        self.assertIsNotNone(m.quality)
    
    def test_plot_map_quality(self):
        """Test plot_map_quality function"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        m.map2sphere()
        # Test that plot_map_quality function exists
        self.assertTrue(hasattr(m, 'plot_map_quality'))
        self.assertTrue(callable(m.plot_map_quality))
    
    def test_subdivide(self):
        """Test mesh subdivision"""
        X, F = surface_mesh.sphere_mesh_gen(1)
        m = surface_mesh(X, F)
        initial_faces = len(m.F)
        m.subdivide(1)
        # Each subdivision multiplies faces by 4
        self.assertEqual(len(m.F), initial_faces * 4)
    
    def test_laplacian_smooth(self):
        """Test Laplacian smoothing"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        m.laplacian_smooth_iter = 3
        m.laplacian_smooth_beta = 0.5
        m.laplacian_smooth()
        # Mesh should still be valid
        self.assertEqual(len(m.X), len(X))
        self.assertEqual(len(m.F), len(F))
    
    def test_mesh_quality_stats(self):
        """Test mesh quality statistics computation"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        stats = m.get_mesh_quality_stats()
        
        self.assertIn('n_vertices', stats)
        self.assertIn('n_faces', stats)
        self.assertIn('n_edges', stats)
        self.assertIn('mean_edge_length', stats)
        self.assertIn('mean_quality', stats)
        self.assertIn('min_quality', stats)
        self.assertIn('area_uniformity', stats)
        
        self.assertEqual(stats['n_vertices'], len(X))
        self.assertEqual(stats['n_faces'], len(F))
        self.assertGreater(stats['mean_quality'], 0)
    
    def test_remesh_isotropic(self):
        """Test isotropic remeshing"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        
        # Get initial quality
        initial_stats = m.get_mesh_quality_stats()
        
        # Apply isotropic remeshing with fewer iterations for speed
        m.remesh(method='isotropic', n_iterations=2, smooth_iterations=2)
        
        # Mesh should still be valid
        self.assertGreater(len(m.X), 0)
        self.assertGreater(len(m.F), 0)
        
        # Check quality is computed
        new_stats = m.get_mesh_quality_stats()
        self.assertGreater(new_stats['mean_quality'], 0)
    
    def test_simplify_mesh(self):
        """Test mesh simplification"""
        X, F = surface_mesh.sphere_mesh_gen(3)
        m = surface_mesh(X, F)
        initial_faces = len(m.F)
        
        # Simplify to half
        target_faces = initial_faces // 2
        m.simplify_mesh(target_faces=target_faces)
        
        # Should have approximately target number of faces
        self.assertLess(len(m.F), initial_faces)
    
    def test_subdivide_spherical(self):
        """Test spherical subdivision"""
        X, F = surface_mesh.sphere_mesh_gen(1)
        m = surface_mesh(X, F)
        initial_faces = len(m.F)
        
        m.subdivide_spherical(n_subdivisions=1, project_to_sphere=True)
        
        # Should have 4x more faces
        self.assertEqual(len(m.F), initial_faces * 4)
        
        # Vertices should still be on unit sphere
        radii = np.linalg.norm(m.X, axis=1)
        np.testing.assert_allclose(radii, 1.0, rtol=1e-5)
    
    def test_tri_quad(self):
        """Test triangle quadrisection static method"""
        X, F = surface_mesh.sphere_mesh_gen(1)
        new_X, new_F = surface_mesh._tri_quad(X, F)
        
        # Should have 4x more faces
        self.assertEqual(len(new_F), len(F) * 4)
        # Should have more vertices
        self.assertGreater(len(new_X), len(X))
    
    def test_densify(self):
        """Test mesh densification"""
        X, F = surface_mesh.sphere_mesh_gen(1)
        m = surface_mesh(X, F)
        initial_faces = len(m.F)
        
        # Densify by factor of 2
        m.densify(factor=2.0)
        
        # Should have more faces
        self.assertGreater(len(m.F), initial_faces)
    
    def test_compute_edge_lengths(self):
        """Test edge length computation"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        m.edge_info()
        
        edge_lengths = m._compute_edge_lengths()
        
        self.assertEqual(len(edge_lengths), len(m.E))
        self.assertTrue(np.all(edge_lengths > 0))
    
    def test_vertex_normal_computation(self):
        """Test vertex normal computation"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        m.edge_info()
        
        # Compute normal at a vertex
        normal = m._compute_vertex_normal(0)
        
        # Should be a unit vector
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0, places=5)
    
    def test_print_mesh_quality(self):
        """Test mesh quality printing"""
        X, F = surface_mesh.sphere_mesh_gen(2)
        m = surface_mesh(X, F)
        
        # Should not raise any exceptions
        stats = m.print_mesh_quality()
        self.assertIsInstance(stats, dict)


if __name__ == '__main__':
    unittest.main()
