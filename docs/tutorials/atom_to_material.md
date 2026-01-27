# Atom to Material Aggregation

This tutorial covers different methods for aggregating atom-level MLIP embeddings to material-level representations.

## Why Aggregation?

MLIP models (MACE, CHGNet, ORB, etc.) produce embeddings at the **atom level**. Each atom in a structure gets its own embedding vector. To analyze materials as a whole, we need to aggregate these into a single **material-level** embedding.

## Available Methods

ManiFish provides four aggregation functions for different use cases:

| Function | Use Case |
|----------|----------|
| `aggregate_atom_embeddings` | Index-based mapping (atom → material) |
| `aggregate_from_structure_list` | List of structures (one array per material) |
| `aggregate_by_material_id` | String IDs for each atom |
| `aggregate_with_weights` | Weighted aggregation (e.g., by atomic mass) |

## Method 1: Index-Based Aggregation

Use when you have an index array mapping atoms to materials.

```python
from manifish import aggregate_atom_embeddings
import numpy as np

# Example: 1000 atoms across 50 materials
atom_embeddings = np.random.randn(1000, 128)  # From MLIP

# Each atom knows which material it belongs to (0 to 49)
atom_to_material = np.repeat(np.arange(50), 20)  # 20 atoms per material

# Material IDs
material_ids = [f"mp-{i}" for i in range(50)]

# Aggregate
material_embeddings, ids = aggregate_atom_embeddings(
    atom_embeddings,
    atom_to_material,
    material_ids,
    method="mean"
)

print(f"Shape: {material_embeddings.shape}")  # (50, 128)
```

## Method 2: List-Based Aggregation

Use when you have a list of arrays, one per structure.

```python
from manifish import aggregate_from_structure_list

# Each structure has different number of atoms
structures = [
    np.random.randn(12, 128),   # Material 1: 12 atoms
    np.random.randn(8, 128),    # Material 2: 8 atoms
    np.random.randn(24, 128),   # Material 3: 24 atoms
]
material_ids = ["mp-123", "mp-456", "mp-789"]

material_embeddings, ids = aggregate_from_structure_list(
    structures,
    material_ids,
    method="mean"
)

print(f"Shape: {material_embeddings.shape}")  # (3, 128)
```

## Method 3: ID-Based Aggregation

Use when each atom has a material ID string.

```python
from manifish import aggregate_by_material_id

atom_embeddings = np.random.randn(100, 128)

# Atoms labeled by material ID
atom_material_ids = (
    ["mp-1"] * 30 +   # 30 atoms for mp-1
    ["mp-2"] * 40 +   # 40 atoms for mp-2
    ["mp-3"] * 30     # 30 atoms for mp-3
)

material_embeddings, unique_ids = aggregate_by_material_id(
    atom_embeddings,
    atom_material_ids,
    method="mean"
)

print(unique_ids)  # ["mp-1", "mp-2", "mp-3"]
print(f"Shape: {material_embeddings.shape}")  # (3, 128)
```

## Method 4: Weighted Aggregation

Use for weighted pooling (e.g., by atomic mass for mass-weighted averaging).

```python
from manifish import aggregate_with_weights

atom_embeddings = np.random.randn(100, 128)
atom_to_material = np.repeat(np.arange(5), 20)

# Weights (e.g., atomic masses)
weights = np.array([12.0, 16.0, ...])  # One per atom

material_embeddings, ids = aggregate_with_weights(
    atom_embeddings,
    atom_to_material,
    weights,
    material_ids=["mat-0", "mat-1", "mat-2", "mat-3", "mat-4"]
)
```

## Aggregation Methods

All functions support these pooling methods:

| Method | Description | Output Dimension |
|--------|-------------|------------------|
| `mean` | Average pooling (recommended) | Same as input |
| `sum` | Sum pooling | Same as input |
| `max` | Element-wise maximum | Same as input |
| `min` | Element-wise minimum | Same as input |
| `std` | Standard deviation | Same as input |
| `mean_std` | Concatenate mean and std | 2x input dimension |

### Example: Using Different Methods

```python
# Standard mean pooling
mat_mean, _ = aggregate_atom_embeddings(atoms, mapping, ids, method="mean")

# Capture both mean and variance
mat_mean_std, _ = aggregate_atom_embeddings(atoms, mapping, ids, method="mean_std")
print(f"Mean: {mat_mean.shape}")         # (50, 128)
print(f"Mean+Std: {mat_mean_std.shape}")  # (50, 256)
```

## Best Practices

1. **Use `mean` for most cases** - It's robust and preserves scale
2. **Use `mean_std` for richer representation** - Captures both average and spread
3. **Use weighted aggregation for physics-informed pooling** - E.g., mass-weighted
4. **Normalize after aggregation if comparing across MLIPs**

## Complete Example

```python
from manifish import (
    aggregate_atom_embeddings,
    ManifoldFishAnalyzer,
)
import numpy as np

# 1. Load atom embeddings from MLIP
atom_embeddings = np.load("mace_atom_embeddings.npy")
atom_to_material = np.load("atom_to_material.npy")
material_ids = np.load("material_ids.npy")

# 2. Aggregate to material level
material_embeddings, ids = aggregate_atom_embeddings(
    atom_embeddings,
    atom_to_material,
    material_ids,
    method="mean"
)

# 3. Split into reference and generated
ref_mask = [id.startswith("mp-") for id in ids]
gen_mask = [not m for m in ref_mask]

ref_emb = material_embeddings[ref_mask]
ref_ids = [ids[i] for i, m in enumerate(ref_mask) if m]

gen_emb = material_embeddings[gen_mask]
gen_ids = [ids[i] for i, m in enumerate(gen_mask) if m]

# 4. Analyze
analyzer = ManifoldFishAnalyzer(ref_emb, ref_ids)
results = analyzer.analyze(gen_emb, gen_ids)
analyzer.print_summary(results)
```
