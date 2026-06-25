# Debug visualization for a specific patch
# This shows: fine mesh, border vertices, chain vertices, sentinel vertices
print("="*60)
print("Patch Debug Visualization")
print("="*60)

# Select patch to visualize (change this to debug different patches)
patch_idx = 7  # Change this to visualize different patches (0-indexed)

if patch_idx >= PM['npatches']:
    print(f"Error: Patch index {patch_idx} is out of range (max: {PM['npatches']-1})")
else:
    print(f"Visualizing patch {patch_idx}")
    
    try:
        import pyvista as pv
        
        # Get the patch mesh
        patm = PM['P'][patch_idx][0]
        
        # Get border vertices
        border_vertices = np.where(patm.border_vertex)[0] if hasattr(patm, 'border_vertex') and patm.border_vertex is not None else np.array([])
        
        # Get chain vertices from edge_dat
        chain_vertices = set()
        if 'patch' in PM and patch_idx in PM.get('patch', {}):
            # Use refined edge chains
            for edge_chain in PM['patch'][patch_idx]['edge_dat']:
                chain_vertices.update(edge_chain.astype(int))
        else:
            # Use PM.edge_dat
            for eix in range(len(PM['Edges'])):
                if PM['Edges'][eix, 0] == patch_idx or PM['Edges'][eix, 1] == patch_idx:
                    if eix < len(PM['edge_dat']) and len(PM['edge_dat'][eix]) > 0:
                        chain_vertices.update(PM['edge_dat'][eix].astype(int))
        chain_vertices = np.array(list(chain_vertices))
        
        # Get sentinel vertices for this patch
        sentinel_vertices = set()
        for eix in range(len(PM['Edges'])):
            if PM['Edges'][eix, 0] == patch_idx or PM['Edges'][eix, 1] == patch_idx:
                if eix < len(PM['sentinels']):
                    s1 = int(PM['sentinels'][eix, 0])
                    s2 = int(PM['sentinels'][eix, 1])
                    if s1 >= 0 and s1 < len(m.X):
                        sentinel_vertices.add(s1)
                    if s2 >= 0 and s2 < len(m.X):
                        sentinel_vertices.add(s2)
        sentinel_vertices = np.array(list(sentinel_vertices))
        
        # Create subplot layout: 2x3 grid (5 plots: 2x2 + 1 extra)
        plotter = pv.Plotter(shape=(2, 3))
        
        # ========== Plot 1: Fine mesh of the patch ==========
        plotter.subplot(0, 0)
        # Create mesh from patch
        num_faces = patm.F.shape[0]
        faces_with_n_vertices = np.hstack((np.full((num_faces, 1), 3), patm.F))
        cells = faces_with_n_vertices.flatten()
        mesh_pv = pv.PolyData(patm.X, cells)
        plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True, edge_color='black', line_width=0.5)
        plotter.add_text(f'Patch {patch_idx}: Fine Mesh\n({len(patm.X)} vertices, {len(patm.F)} faces)', font_size=10)
        plotter.background_color = 'white'
        
        # ========== Plot 2: Patch mesh with border vertices marked ==========
        plotter.subplot(0, 1)
        plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True, edge_color='black', line_width=0.5)
        if len(border_vertices) > 0:
            border_points = pv.PolyData(patm.X[border_vertices])
            plotter.add_mesh(border_points, color='yellow', point_size=15,
                           render_points_as_spheres=True, label=f'Border ({len(border_vertices)})')
        plotter.add_text(f'Patch {patch_idx}: Border Vertices\n(Yellow spheres)', font_size=10)
        plotter.background_color = 'white'
        plotter.add_legend()
        
        # ========== Plot 3: Patch mesh with chain vertices marked ==========
        plotter.subplot(1, 0)
        plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True, edge_color='black', line_width=0.5)
        if len(chain_vertices) > 0:
            # Filter chain vertices that are in the patch
            chain_in_patch = chain_vertices[chain_vertices < len(patm.X)]
            if len(chain_in_patch) > 0:
                chain_points = pv.PolyData(patm.X[chain_in_patch])
                plotter.add_mesh(chain_points, color='orange', point_size=12,
                               render_points_as_spheres=True, label=f'Chain ({len(chain_in_patch)})')
        plotter.add_text(f'Patch {patch_idx}: Chain Vertices\n(Orange spheres)', font_size=10)
        plotter.background_color = 'white'
        plotter.add_legend()
        
        # ========== Plot 4: Full object mesh with patch border and sentinels ==========
        plotter.subplot(1, 1)
        # Create full mesh
        num_faces_full = m.F.shape[0]
        faces_with_n_vertices_full = np.hstack((np.full((num_faces_full, 1), 3), m.F))
        cells_full = faces_with_n_vertices_full.flatten()
        mesh_full = pv.PolyData(m.X, cells_full)
        
        # Color only the patch faces
        face_colors = np.ones((len(m.F), 3)) * 0.9  # Light gray for all faces
        if hasattr(m, 'face_labels') and m.face_labels is not None:
            # Get faces belonging to this patch
            unique_labels = np.unique(m.face_labels)
            if patch_idx < len(unique_labels):
                patch_label = unique_labels[patch_idx]
                patch_faces = np.where(m.face_labels == patch_label)[0]
                face_colors[patch_faces] = [0.5, 0.8, 1.0]  # Light blue for patch faces
        
        mesh_full['face_colors'] = face_colors
        plotter.add_mesh(mesh_full, scalars='face_colors', rgb=True, show_edges=True, 
                        edge_color='black', line_width=0.3, opacity=0.7)
        
        # Mark border vertices of this patch on the full mesh
        if len(border_vertices) > 0:
            # Map border vertices from patch to full mesh (they use same indices)
            border_points_full = pv.PolyData(m.X[border_vertices])
            plotter.add_mesh(border_points_full, color='yellow', point_size=12,
                           render_points_as_spheres=True, label=f'Border ({len(border_vertices)})')
        
        # Mark sentinel vertices as red spheres
        if len(sentinel_vertices) > 0:
            sentinel_points = pv.PolyData(m.X[sentinel_vertices])
            plotter.add_mesh(sentinel_points, color='red', point_size=20,
                           render_points_as_spheres=True, label=f'Sentinels ({len(sentinel_vertices)})')
        
        plotter.add_text(f'Full Mesh: Patch {patch_idx} Highlighted\n(Yellow=border, Red=sentinels)', font_size=10)
        plotter.background_color = 'white'
        plotter.add_legend()
        
        # ========== Plot 5: Spherical parameterization of the patch ==========
        plotter.subplot(1, 2)
        
        # Check if patch has t and p values
        if patm.t is not None and patm.p is not None and np.any(patm.t != 0) and np.any(patm.p != 0):
            # Convert spherical coordinates to Cartesian on unit sphere
            # Use the same import as the rest of the codebase
            from pySHP.utils import kk_sph2cart
            u, v, w = kk_sph2cart(patm.t, patm.p, np.ones(len(patm.p)))
            X_sph = np.column_stack([u, v, w])
            
            # Create mesh on sphere
            num_faces_sph = patm.F.shape[0]
            faces_with_n_vertices_sph = np.hstack((np.full((num_faces_sph, 1), 3), patm.F))
            cells_sph = faces_with_n_vertices_sph.flatten()
            mesh_sph = pv.PolyData(X_sph, cells_sph)
            
            # Add reference sphere (semi-transparent)
            sphere = pv.Sphere(radius=0.98, theta_resolution=30, phi_resolution=30)
            plotter.add_mesh(sphere, color='cyan', opacity=0.3, show_edges=False)
            
            # Plot the parameterized patch on sphere
            plotter.add_mesh(mesh_sph, color='lightblue', show_edges=True, 
                           edge_color='black', line_width=0.5, opacity=0.9)
            
            # Mark border vertices on sphere
            if len(border_vertices) > 0:
                border_vertices_valid = border_vertices[(patm.t[border_vertices] != 0) & (patm.p[border_vertices] != 0)]
                if len(border_vertices_valid) > 0:
                    border_points_sph = pv.PolyData(X_sph[border_vertices_valid])
                    plotter.add_mesh(border_points_sph, color='yellow', point_size=12,
                                   render_points_as_spheres=True, label=f'Border ({len(border_vertices_valid)})')
            
            plotter.add_text(f'Patch {patch_idx}: Spherical Parameterization\n(On unit sphere)', font_size=10)
            plotter.background_color = 'black'
            plotter.add_legend()
        else:
            plotter.add_text(f'Patch {patch_idx}: No Parameterization\n(t and p values not available)', font_size=10)
            plotter.background_color = 'white'
        
        # Show the plot
        plotter.show()
        
        # Print diagnostic information
        print(f"\nPatch {patch_idx} Diagnostics:")
        print(f"  Patch vertices: {len(patm.X)}")
        print(f"  Patch faces: {len(patm.F)}")
        print(f"  Border vertices: {len(border_vertices)}")
        print(f"  Chain vertices: {len(chain_vertices)}")
        print(f"  Sentinel vertices: {len(sentinel_vertices)}")
        if len(sentinel_vertices) > 0:
            print(f"    Sentinel indices: {sentinel_vertices}")
        
        # Check overlap
        border_set = set(border_vertices)
        chain_set = set(chain_vertices[chain_vertices < len(patm.X)])
        overlap = border_set.intersection(chain_set)
        border_only = border_set - chain_set
        chain_only = chain_set - border_set
        
        print(f"\n  Border-Chain Overlap:")
        print(f"    Overlap: {len(overlap)} vertices")
        print(f"    Border-only: {len(border_only)} vertices")
        print(f"    Chain-only: {len(chain_only)} vertices")
        
        # Spherical parameterization diagnostics
        if patm.t is not None and patm.p is not None:
            valid_tp = (patm.t != 0) & (patm.p != 0)
            num_valid = np.sum(valid_tp)
            print(f"\n  Spherical Parameterization:")
            print(f"    Vertices with valid t,p: {num_valid} / {len(patm.t)}")
            if num_valid > 0:
                print(f"    Theta range: [{patm.t[valid_tp].min():.4f}, {patm.t[valid_tp].max():.4f}]")
                print(f"    Phi range: [{patm.p[valid_tp].min():.4f}, {patm.p[valid_tp].max():.4f}]")
        else:
            print(f"\n  Spherical Parameterization: Not available (t or p is None)")
        
    except ImportError:
        print("PyVista not available for visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
