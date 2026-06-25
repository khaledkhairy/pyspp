# Simplified Mesh Architecture

This document maps the planning invariants to the codebase. It describes how the fine mesh is reduced to a simplified mesh suitable for bijective spherical parameterization, and how the patch structure (PM) stays compatible with downstream fine-to-sphere mapping.

## Data Flow

```
Fine mesh (1000+ verts) → Segmentation (find_valid_segmentation) → Patches (face labels)
                                                                    ↓
                              PM ← patch_info_gen(m, P, Pconn) ← P, Pconn
                              ↓
                    Simplified mesh (PM['pm'], <100 verts)
                              ↓
                    map2sphere() → Unit sphere positions
                              ↓
                    Fine-mesh vertices mapped via PM (patch-wise, key positions)
```

**Unchanged (per requirements):** Initial segmentation logic and everything after simplified mesh + PM construction (set_edge_n_fine_vertices, parameterize_patches_cart, map2sphere, optimization) stay as-is.

---

## PM as Single Source of Truth

The patch structure `PM` is the canonical representation. The simplified mesh `PM['pm']` is a **derivative** built from PM; it must never contradict PM.

| PM field | Role | Used by |
|----------|------|---------|
| `keys` | Triple+ junctions (patch_index, vertex_index, face_index) | Key detection, boundary cycles |
| `sentinels` | Per-edge [s1, s2] mesh indices | Edge chains, boundary ordering |
| `OUT_chain` | Ordered boundary vertices per patch | Parameterization, edge_dat |
| `edge_dat` | Vertex chains per edge | Face generation, fine-mesh mapping |
| `Xkeyind` | simpl_vertex_idx → fine_mesh_vertex_idx | Fine-to-sphere mapping, set_edge_n_fine_vertices |
| `pm` | Simplified mesh (X, F, face_labels) | map2sphere, visualization |
| `P`, `Edges`, `CV` | Patch geometry and connectivity | Face generation, parameterization |

**Location:** `patch_info_gen()` builds PM; `generate_simplified_mesh()` constructs `pm` from keys, sentinels, centers, and boundary cycles.

---

## Vertex Compatibility (Xkeyind)

Every simplified mesh vertex has a role defined by `PM['Xkeyind']`:

| Xkeyind[i] | Meaning | Used in fine-to-sphere mapping? |
|------------|---------|----------------------------------|
| ≥ 0 | Fine-mesh vertex index (key or center) | Yes – direct correspondence |
| -1 | Fictitious (crown, cylinder quad center) | No – topology only |

**Invariant:** All simplified vertices with `Xkeyind[i] >= 0` must correspond to valid fine-mesh vertices (`0 <= Xkeyind[i] < len(m.X)`). Fictitious vertices are created for manifoldness (caps, cylinders) and are excluded from patch-wise fine-mesh parameterization.

**Code:** `set_edge_n_fine_vertices_from_PM`, `parameterize_patches_cart`, and sphere mapping logic use `Xkeyind` to resolve simplified-mesh indices to fine-mesh indices.

---

## Segmentation Guarantor

A segmentation is **well-behaved** iff every patch has:
- **1 neighbor** (cap): handled with fictitious keys and fan triangulation
- **≥ 3 neighbors** (regular): enough key vertices for a proper boundary cycle

**Invalid:** 0 or 2 neighbors (neck/bridge patches cause genus > 0, gaps, self-intersections).

**Implementation:** `find_valid_segmentation()` iterates over `nseeds` and uses `compute_vertex_based_patch_connectivity()` for accurate neighbor counts (face-based Pconn can miss triple-junction connections).

**Location:** `pySHP/level1/find_valid_segmentation.py`  
**Usage:** Call before `patch_info_gen` (e.g. in notebook); `patch_info_gen` can optionally validate at entry and warn/raise if segmentation is invalid.

---

## Validation Gates

After simplified mesh construction, `validate_simplified_mesh()` checks:

1. Every patch has at least one face  
2. No floating vertices  
3. Manifold edges (each edge shared by exactly 2 faces)  
4. Closed mesh (no boundary edges)  
5. Genus zero (topological sphere)  
6. Single connected component  

**Location:** `pySHP/level1/validate_simplified_mesh.py`  
**Called from:** `patch_info_gen()` after `generate_simplified_mesh`, `fix_flipped_faces`, `add_fictitious_keys_for_single_neighbor_edges`, `ensure_genus_zero_simplified_mesh`

**Strict mode:** When enabled, `patch_info_gen` raises if the simplified mesh fails validation, so callers fail early instead of propagating invalid geometry to map2sphere.

---

## Global Boundary Graph (POC)

`global_boundary_graph.py` implements an alternative boundary-cycle construction:

1. **build_global_boundary_graph(m, PM)**: Traces patch-patch boundaries directly on the mesh (no OUT_chain). Nodes = keys + sentinels; arcs = chains between them. Guarantees each chain appears once; both patches see it in opposite order.

2. **get_patch_boundary_cycles_from_graph_v2(graph, PM, pix, ...)**: Traces patch boundary as a closed walk in the graph. Uses angular ordering at each node for consistent CW/CCW.

3. **Integration**: `generate_simplified_mesh(..., use_global_boundary_graph=True)` builds the graph and uses it for boundary cycles. Falls back to `_patch_boundary_cycle` on failure. Set `use_global_boundary_graph=False` to use the legacy path.

---

## File Map

| File | Responsibility |
|------|----------------|
| `global_boundary_graph.py` | Global boundary graph (POC) for consistent boundary cycles |
| `find_valid_segmentation.py` | Segmentation guarantor (neighbor validation) |
| `mesh_segmentation_rw.py` | Raw segmentation (called by find_valid_segmentation) |
| `patch_info_gen.py` | Build PM, generate simplified mesh, validation gate |
| `validate_simplified_mesh.py` | Topology checks (manifold, genus, closed) |
| `diagnose_simplified_mesh.py` | Detailed diagnostics (boundary edges, patches) |
| `set_edge_n_fine_vertices_from_PM.py` | Edge fine-vertex counts for sphere mapping |
| `parameterize_patches_cart.py` | Patch-wise fine-mesh parameterization (uses PM) |

---

## Compatibility Checklist (for new changes)

When modifying simplified mesh construction:

1. **PM coherence:** Every new vertex/face must have a corresponding entry in `Xkeyind` (or -1 if fictitious).
2. **Edge consistency:** Shared edges between patches must use the same vertex ordering in both patches' boundary cycles.
3. **Fictitious vertices:** Document which vertices are fictitious and ensure they are excluded from fine-to-sphere mapping.
4. **Validation:** Run `validate_simplified_mesh` and `diagnose_simplified_mesh_full` after changes; fix before map2sphere.
