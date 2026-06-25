# Strategy for Well-Behaved Simplified Mesh Topology

## Overview

The simplified mesh is built from:
- **Key vertices**: Triple+ junctions (vertices touching 3+ patches)
- **Sentinel vertices**: Endpoints of patch-patch boundaries (from `PM['sentinels']`)
- **Crown sentinels**: Fictitious vertices inserted along neck/single-neighbor edges (4 for neck, 3 for others)
- **Center vertices**: One per patch (interior point)

## Requirements for Well-Behaved Mesh

1. **Every patch has ≥ 1 face** (no missing patches)
2. **Every vertex is part of ≥ 1 face** (no floating vertices)
3. **Every edge is shared by exactly 2 faces** (manifold, no non-manifold edges)
4. **Mesh is closed** (no boundary edges)
5. **Genus is zero** (topological sphere: V - E + F = 2)
6. **Single connected component** (all patches connected)

## Current Structure Analysis

### Face Generation Process

1. **Build boundary cycle** for each patch:
   - Collect keys + sentinels from incident edges
   - Include crown sentinels (fictitious vertices) if they exist in mesh
   - Order them using: OUT_chain → edge walk → geometric fallback

2. **Generate faces** as fan from center:
   - For cycle `[v0, v1, ..., vn-1]`: faces = `[(center, v_i, v_{i+1})]` for i=0..n-1
   - Each face connects center to two consecutive boundary vertices

### Key Components

- **Xkeyind**: Maps simplified mesh index → original mesh vertex index
  - First `nkeys` entries: keys + sentinels (original mesh vertices)
  - Next `npatches` entries: center vertices
  - After that: fictitious vertices (Xkeyind = -1)

- **PM['sentinels']**: `[n_edges, 2]` array, `[eix] = [s1, s2]` (original mesh indices)

- **PM['fictitious_per_edge']**: `{eix: [simplified_indices]}` - crown sentinels per edge

## Common Failure Modes

### 1. Missing Faces (Patch Skipped)

**Cause**: Boundary cycle has < 2 vertices
- Zero-key patch with < 2 sentinels
- Sentinels not found in simplified mesh (mapping failure)
- Edge walk fails and geometric fallback fails

**Detection**: `validate_simplified_mesh` → `missing_patches`

**Fix Strategy**:
- Ensure every patch has ≥ 2 boundary vertices:
  - Zero-key patches: Add fictitious keys to all incident edges
  - Degenerate sentinels: Fix using edge chains or shared boundary
- Ensure sentinels map correctly to simplified mesh
- Improve geometric fallback (always succeeds if ≥ 2 vertices)

### 2. Floating Vertices

**Cause**: Vertex not part of any face
- Center vertex of patch with no faces
- Key/sentinel vertex not included in any patch's boundary cycle
- Fictitious vertex added but not used in faces

**Detection**: `validate_simplified_mesh` → `floating_vertices`

**Fix Strategy**:
- Ensure every patch gets faces (fix missing faces)
- Ensure all sentinels/keys are included in at least one patch's boundary
- Remove unused fictitious vertices

### 3. Non-Manifold Edges

**Cause**: Edge shared by ≠ 2 faces
- Boundary cycle has duplicate consecutive vertices
- Overlapping patches (same edge in multiple cycles)
- Incorrect face generation

**Detection**: `validate_simplified_mesh` → `non_manifold_edges`

**Fix Strategy**:
- Ensure boundary cycles have no duplicates (except closure)
- Ensure patches don't overlap (each edge belongs to exactly 2 patches)
- Verify face generation produces correct edge counts

### 4. Genus > 0

**Cause**: Topology has handles/holes
- Neck edges create handles
- Missing faces create holes
- Disconnected components

**Detection**: `validate_simplified_mesh` → `genus`

**Fix Strategy**:
- Add crown sentinels to neck edges (cylinder topology)
- Ensure all patches have faces
- Collapse neck edges if safe (don't break patches)

### 5. Disconnected Components

**Cause**: Patches not connected via shared edges
- Missing sentinels between patches
- Incorrect edge connectivity

**Detection**: `validate_simplified_mesh` → `disconnected`

**Fix Strategy**:
- Ensure every patch pair that should be adjacent has an edge with valid sentinels
- Verify patch connectivity matches original mesh

## Comprehensive Strategy

### Phase 1: Sentinel/Key Generation (Robust)

1. **For each patch-patch edge**:
   - Find shared keys (if ≥ 2, use first two as sentinels)
   - Else: Find shared outline vertices (if ≥ 2, use endpoints)
   - Else: Use edge chain endpoints (from `edge_dat` or `_ordered_shared_boundary_chain`)
   - **Never leave sentinels as [0,0] or [s1,s1] unless intentional**

2. **For zero-key patches**:
   - Ensure every incident edge has valid sentinels (≥ 2 distinct vertices)
   - Add fictitious keys to all incident edges (not just single-neighbor/neck)

3. **For neck edges**:
   - Add 4 crown sentinels (cylinder topology)
   - Ensure both incident patches include crown in boundary cycle

### Phase 2: Boundary Cycle Building (Complete)

1. **Collect boundary vertices**:
   - Keys of patch
   - Sentinels from all incident edges
   - Crown sentinels (fictitious vertices) if they exist in mesh

2. **Build ordered cycle**:
   - **Prefer OUT_chain**: Filter to boundary vertices, preserve order
   - **Fallback edge walk**: Use E (edges between sentinels/keys)
   - **Final fallback geometric**: Order by angle around center (always works if ≥ 2 vertices)

3. **Insert crown sentinels**:
   - For each edge with crown, find s1 and s2 in cycle
   - Insert crown between them: s1 → f1 → f2 → f3 → f4 → s2

### Phase 3: Face Generation (Complete)

1. **For each patch**:
   - Build boundary cycle (must have ≥ 2 vertices)
   - Generate fan: `[(center, v_i, v_{i+1})]` for i=0..n-1
   - **Never skip a patch** - if cycle fails, use geometric fallback

2. **Ensure all vertices used**:
   - Every center vertex: part of its patch's faces
   - Every key/sentinel: part of ≥ 1 patch's boundary cycle
   - Every fictitious vertex: inserted into boundary cycle

### Phase 4: Validation & Repair

1. **Validate mesh**:
   - Check missing patches
   - Check floating vertices
   - Check non-manifold edges
   - Check genus
   - Check connectivity

2. **Repair if needed**:
   - Missing patches: Regenerate with geometric fallback
   - Floating vertices: Remove or add to boundary cycles
   - Non-manifold: Fix duplicate vertices in cycles
   - Genus > 0: Add more crown sentinels or collapse edges safely

## Implementation Checklist

- [x] Store fictitious vertex indices per edge (`PM['fictitious_per_edge']`)
- [x] Include fictitious vertices in boundary cycle building
- [x] Insert crown sentinels into boundary cycle in correct order
- [x] Geometric fallback for boundary cycle (always works)
- [x] Safeguard against collapsing edges that would break patches
- [x] Validation function to detect issues
- [ ] Repair function to fix common issues automatically
- [ ] Ensure every patch gets faces (never skip)
- [ ] Verify crown sentinels are always included when they exist

## Key Functions

- `_patch_boundary_cycle()`: Builds boundary cycle with crown support
- `_insert_crown_sentinels_in_cycle()`: Inserts crown into cycle
- `validate_simplified_mesh()`: Validates mesh topology
- `diagnose_patch_boundary_issues()`: Diagnoses why patches fail
- `ensure_genus_zero_simplified_mesh()`: Reduces genus safely

## Special Patches: Cap and Cylinder Only

**Principle**: The simplified mesh is built the same way for all patches except:
- **Cap patches** (one boundary loop, one neighbor): synthetic keys and full boundary chain so key/sentinel counts match the neighbor (e.g. cap–cylinder edge). Handled in [9a'] with `_full_boundary_chain_cap_edge`.
- **Cylinder patches** (two boundary loops): no single center; use two rings from `_patch_boundary_cycle`, resample to same length, best cyclic shift to align rings, then one center per quad and 4 triangles per quad. Handled inside `generate_simplified_mesh` only (cylinder_ring_data, rebuild X, face generation).

**Do not** add global key/sentinel changes (e.g. extra synthetic keys or key propagation for all cylinder neighbors) without comparing to the reference (e.g. MATLAB) and testing: such changes can break connectivity for non–cap/cylinder patches. Any future fix for cylinder–neighbor criss-crossing should:
1. Leave keys/sentinels and edge_dat unchanged for non–cylinder edges.
2. Optionally use a helper like `_cylinder_non_cap_ring_from_edge_chains` only to order the cylinder’s non-cap ring when building cylinder_ring_data, without modifying PM['keys'] or PM['sentinels'].

## Alignment with Reference (e.g. MATLAB)

When changing simplified-mesh construction:
1. **Keys**: Same rule everywhere – triple+ junctions from segmentation; synthetic keys only where the reference adds them (e.g. cap edges with <3 shared keys in [9a']).
2. **Sentinels**: One pair per edge from shared keys or outline; do not overwrite per-edge after [8] except where the reference does.
3. **Edges E**: Consecutive key/sentinel along each edge chain (edge_dat); then center–key edges for non-cylinder patches.
4. **Faces**: Per-patch boundary cycle → fan from center (or cylinder quads). Cap and cylinder are the only patches that get different face-generation paths.

## Next Steps

1. **Ensure every patch gets faces**: Modify `generate_simplified_mesh` to never skip patches - always use geometric fallback if needed
2. **Verify crown inclusion**: Ensure crown sentinels are always included when they exist in the mesh
3. **Add repair function**: Automatically fix missing faces, floating vertices, etc.
4. **Test on problematic cases**: Mushroom with patch 1 (zero-key), patch 8 (complex boundary)
5. **Cylinder–neighbor alignment**: If needed, re-introduce only the minimal change (e.g. non-cap ring order from edge chains in cylinder_ring_data) and validate against full mesh connectivity
