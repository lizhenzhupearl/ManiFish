# Quick Start

This guide will help you get started with ManiFish in 5 minutes.

## Basic Workflow

ManiFish follows a simple workflow:

1. **Get embeddings** from your MLIP model (atom-level)
2. **Aggregate** atom embeddings to material-level
3. **Analyze** generated materials against a reference manifold
4. **Interpret** results and prioritize structures for validation

## Step 1: Prepare Your Data

You need two sets of material embeddings:

- **Reference embeddings**: Known stable structures (your "ground truth")
- **Generated embeddings**: New structures to evaluate

```python
import numpy as np

# Example: Load your embeddings
# In practice, these come from MLIP models
reference_embeddings = np.load("reference_embeddings.npy")  # (n_ref, dim)
generated_embeddings = np.load("generated_embeddings.npy")  # (n_gen, dim)

reference_ids = [f"mp-{i}" for i in range(len(reference_embeddings))]
generated_ids = [f"gen-{i}" for i in range(len(generated_embeddings))]
```

## Step 2: Aggregate Atom Embeddings (if needed)

If you have atom-level embeddings, aggregate them to material-level:

```python
from manifish import aggregate_atom_embeddings

# atom_embeddings: (n_atoms, embed_dim)
# atom_to_material: array mapping each atom to its material index

material_embeddings, material_ids = aggregate_atom_embeddings(
    atom_embeddings,
    atom_to_material,
    material_ids=["mp-1", "mp-2", ...],
    method="mean"  # Options: mean, sum, max, min, std, mean_std
)
```

## Step 3: Create the Analyzer

```python
from manifish import ManifoldFishAnalyzer

analyzer = ManifoldFishAnalyzer(
    reference_embeddings=reference_embeddings,
    reference_ids=reference_ids,
    n_neighbors=5,
    boundary_method="alpha_shape",  # or "convex_hull", "one_class_svm"
)
```

## Step 4: Analyze Generated Structures

```python
results = analyzer.analyze(
    generated_embeddings,
    material_ids=generated_ids
)

# Print summary
analyzer.print_summary(results)
```

Output:
```
================================================================================
                        MANIFOLD FISH ANALYSIS SUMMARY
================================================================================
Total analyzed: 100 structures

Category Distribution:
  redundant_fish:          12 ( 12.0%) ████████████
  fish_in_water:           45 ( 45.0%) █████████████████████████████████████████████
  frontier_fish:           22 ( 22.0%) ██████████████████████
  edge_fish:               10 ( 10.0%) ██████████
  adventurous_fish:         8 (  8.0%) ████████
  structural_hallucination: 3 (  3.0%) ███
```

## Step 5: Find Neighbors

See which reference materials are most similar to your generated ones:

```python
neighbor_indices = analyzer.print_neighbors(
    generated_embeddings,
    material_ids=generated_ids,
    n_neighbors=5
)
```

Output:
```
Neighbor indices for 100 generated materials
(showing 5 nearest neighbors in reference set of 1000 points)
------------------------------------------------------------
gen-0: neighbors = [42, 17, 89, 234, 567]
gen-1: neighbors = [5, 23, 42, 101, 88]
...
```

## Step 6: Export Results

```python
# Export to CSV
analyzer.export_results(results, "analysis_results.csv")

# Or get as DataFrame
df = analyzer.results_to_dataframe(results)
```

## Step 7: Get Results by Category

```python
# Get material IDs by category
frontier_ids = analyzer.get_ids_by_category(results, "frontier_fish")
print(f"Frontier fish: {frontier_ids}")

# Get indices (positions in original array) by category
frontier_indices = analyzer.get_indices_by_category(results, "frontier_fish")
frontier_embeddings = generated_embeddings[frontier_indices]

# Get all categories at once
indices_by_cat = analyzer.get_all_indices_by_category(results)
for cat, indices in indices_by_cat.items():
    if len(indices) > 0:
        print(f"{cat}: {len(indices)} materials")
```

## Step 8: View Detailed Statistics

See the metrics that determined each material's classification:

```python
# Show statistics for specific materials
analyzer.print_statistics(results, material_ids=["gen-0", "gen-5"])

# Show statistics for all frontier fish
frontier_ids = analyzer.get_ids_by_category(results, "frontier_fish")
analyzer.print_statistics(results, material_ids=frontier_ids)
```

Output:
```
DETAILED STATISTICS FOR MATERIALS
────────────────────────────────────────────────────────────────────────────────
Material: gen-0  (index: 0)
────────────────────────────────────────────────────────────────────────────────

  CLASSIFICATION:
    Category:        Frontier Fish
    Confidence:      0.850
    Risk Level:      LOW

  POSITION METRICS:
    Manifold Distance:   0.1234  (normalized distance to nearest references)
    Depth Score:         0.7500  (0=deep inside, 1=shallow/edge)
    Boundary Distance:   +0.0500  (+inside, -outside manifold)

  DENSITY METRICS:
    Local Density:       0.0023
    Density Percentile:  15.2%  (vs reference distribution)

  GEOMETRY CONSISTENCY:
    Local PCA Residual:  0.0150  (reconstruction error)
    Geometry Consistent: Yes

  NEAREST REFERENCES:
    Top 3: mp-123, mp-456, mp-789
```

## Step 9: Prioritize for DFT Validation

```python
# Get high-priority structures for DFT validation
frontier_indices = analyzer.get_indices_by_category(results, "frontier_fish")
adventurous_indices = analyzer.get_indices_by_category(results, "adventurous_fish")

priority_indices = frontier_indices + adventurous_indices
print(f"Priority for DFT: {len(priority_indices)} structures")

# Get embeddings for priority structures
priority_embeddings = generated_embeddings[priority_indices]
```

## Next Steps

- [Tutorials](../tutorials/basic_usage.md) - Detailed walkthroughs
- [API Reference](../api/index.md) - Complete API documentation
- [Concepts](../concepts/index.md) - Understanding the theory
