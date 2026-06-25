# Final Step: Spherical Harmonics Projection and Export
# Add this as a new cell in test_surface_mesh_segmentation.ipynb after cell 11

print("="*60)
print("Spherical Harmonics Projection and Export")
print("="*60)

# Step 1: Assemble full parameterized mesh from all patches
from pySHP.level1.assemble_parameterized_mesh import assemble_parameterized_mesh

print("\n[Step 1] Assembling full parameterized mesh from all patches...")
m_param = assemble_parameterized_mesh(m_seg, PM)

# Step 2: Convert to spherical harmonics
from pySHP.sh_basis import sh_basis
from pySHP.shp_surface import shp_surface

print("\n[Step 2] Converting to spherical harmonics...")
L_max = 12  # Maximum spherical harmonics degree (adjust as needed)
gdim = 60   # Grid dimension for basis computation

# Create basis
b = sh_basis(L_max, gdim)

# Create spherical harmonics surface from parameterized mesh
s = shp_surface(L_max, b, m_param)

# Perform spherical harmonics analysis
# This uses the t and p values from the parameterized mesh
s.mesh2shp(m_param, L_max=L_max)

print(f"  Spherical harmonics surface created:")
print(f"    L_max: {s.L_max}")
print(f"    Number of coefficients: {len(s.xc)}")
print(f"    Residual (reconstruction error): {np.linalg.norm(s.residual):.6e}")

# Step 3: Export to .shp3 file
print("\n[Step 3] Exporting to .shp3 file...")
output_filename = "parameterized_mesh.shp3"  # Change this to your desired filename
s.export_shp3(output_filename)

print("\n" + "="*60)
print("Spherical harmonics projection complete!")
print(f"  Output file: {output_filename}")
print("="*60)

# Optional: Visualize the spherical harmonics reconstruction
print("\n[Optional] Visualizing spherical harmonics reconstruction...")
try:
    import pyvista as pv
    
    # Get mesh from spherical harmonics
    m_shp, X_shp, F_shp, Y_LK, t_shp, p_shp = s.get_mesh(nico=3)
    
    # Create plotter
    plotter = pv.Plotter(shape=(1, 2))
    
    # Plot 1: Original parameterized mesh on sphere
    plotter.subplot(0, 0)
    # Convert t, p to Cartesian on unit sphere
    from pySHP.utils import kk_sph2cart
    u, v, w = kk_sph2cart(m_param.t, m_param.p, np.ones(len(m_param.p)))
    X_sph_orig = np.column_stack([u, v, w])
    
    # Create mesh
    num_faces_orig = m_param.F.shape[0]
    faces_with_n_vertices_orig = np.hstack((np.full((num_faces_orig, 1), 3), m_param.F))
    cells_orig = faces_with_n_vertices_orig.flatten()
    mesh_orig = pv.PolyData(X_sph_orig, cells_orig)
    
    # Add reference sphere
    sphere = pv.Sphere(radius=0.98, theta_resolution=30, phi_resolution=30)
    plotter.add_mesh(sphere, color='cyan', opacity=0.2, show_edges=False)
    plotter.add_mesh(mesh_orig, color='lightblue', show_edges=True, 
                    edge_color='black', line_width=0.5, opacity=0.9)
    plotter.add_text('Original Parameterized Mesh\n(on unit sphere)', font_size=10)
    plotter.background_color = 'black'
    
    # Plot 2: Spherical harmonics reconstruction
    plotter.subplot(0, 1)
    # Create mesh from SH coefficients
    num_faces_shp = F_shp.shape[0]
    faces_with_n_vertices_shp = np.hstack((np.full((num_faces_shp, 1), 3), F_shp))
    cells_shp = faces_with_n_vertices_shp.flatten()
    mesh_shp = pv.PolyData(X_shp, cells_shp)
    
    # Add reference sphere
    plotter.add_mesh(sphere, color='cyan', opacity=0.2, show_edges=False)
    plotter.add_mesh(mesh_shp, color='lightgreen', show_edges=True, 
                    edge_color='black', line_width=0.5, opacity=0.9)
    plotter.add_text('Spherical Harmonics Reconstruction\n(L_max={})'.format(L_max), font_size=10)
    plotter.background_color = 'black'
    
    plotter.show()
    
except ImportError:
    print("PyVista not available for visualization")
except Exception as e:
    print(f"Visualization error: {e}")
    import traceback
    traceback.print_exc()
