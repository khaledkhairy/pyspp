"""
Example script: Load and display .off and .shp3 files
"""

import sys
import os

# Add parent directory to path (code directory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pySHP import surface_mesh, shp_surface
from pySHP.utils import readoff


def example_load_off():
    """Example: Load .off file and display"""
    print("=" * 60)
    print("Example 1: Loading .off file")
    print("=" * 60)
    
    # Find test data
    base_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'Matlab',
        'shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off'
    )
    
    scientific_dir = os.path.join(base_dir, 'basic_shapes')
    if os.path.exists(scientific_dir):
        off_files = [f for f in os.listdir(scientific_dir) if f.endswith('.off')]
        if off_files:
            test_file = os.path.join(scientific_dir, off_files[2])
            print(f"\nLoading: {test_file}")
            
            # Read mesh
            X, F = readoff(test_file)
            print(f"  Vertices: {len(X)}")
            print(f"  Faces: {len(F)}")
            
            # Create surface mesh
            m = surface_mesh(X, F)
            
            # Compute properties
            m.props()
            print(f"  Area: {m.A:.2f}")
            print(f"  Volume: {m.V:.2f}")
            
            # plot
            m.plot()
            m.plot_H()
            
            return m
        else:
            print("No .off files found in test directory")
    else:
        print("Test data directory not found")
    
    return None


def example_load_shp3():
    """Example: Load .shp3 file and display"""
    print("\n" + "=" * 60)
    print("Example 2: Loading .shp3 file")
    print("=" * 60)
    
    # Find test data
    base_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'pySHP',
        'test_shp3'
    )
    
    test_file = os.path.join(base_dir, 'bowling_pin.shp3')
    
    if os.path.exists(test_file):
        print(f"\nLoading: {test_file}")
        
        # Load SHP surface
        s = shp_surface()
        s.import_shp3(test_file)
        
        print(f"  L_max: {s.L_max}")
        print(f"  Grid dimension: {s.gdim}")
        print(f"  Scalar fields: {len(s.sf)}")
        
        # Update surface
        s.update()
        print(f"  Surface shape: {s.x.shape}")
        
        s.plot_fast()
        s.plot()
        s.plot_H()
        
        return s
    else:
        print(f"Test file not found: {test_file}")
    
    return None


def example_mesh_to_shp():
    """Example: Convert mesh to spherical harmonics"""
    print("\n" + "=" * 60)
    print("Example 3: Converting mesh to spherical harmonics")
    print("=" * 60)
    
    # Load a mesh
    base_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'Matlab',
        'shp_toolbox-main', 'shp_toolbox-main', 'test_data', 'off'
    )
    
    scientific_dir = os.path.join(base_dir, 'basic_shapes')
    if os.path.exists(scientific_dir):
        off_files = [f for f in os.listdir(scientific_dir) if f.endswith('.off')]
        if off_files:
            test_file = os.path.join(scientific_dir, off_files[0])
            print(f"\nLoading: {test_file}")
            
            # Read and optimize mesh
            X, F = readoff(test_file)
            m = surface_mesh(X, F)
            m.meshresample_keepratio = 0.8
            m = m.optimize_mesh()
            
            print(f"  Optimized mesh: {len(m.X)} vertices, {len(m.F)} faces")
            
            # Convert to spherical harmonics
            from pySHP import sh_basis, shp_surface
            from pySHP.utils import kk_cart2sph
            
            L_max = 8
            b = sh_basis(L_max, 60)
            s = shp_surface(L_max, b, m)
            
            print(f"  SHP surface: L_max={s.L_max}")
            print(f"  Coefficients: {len(s.xc)} per component")
            
            return s
        else:
            print("No .off files found")
    else:
        print("Test data directory not found")
    
    return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("pySHP Examples: Load and Display")
    print("=" * 60)
    
    # Run examples
    #m = example_load_off()
    #s1 = example_load_shp3()
    s2 = example_mesh_to_shp()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
