# Platonic Representation

The Platonic representation is ManiFish's approach for comparing embeddings across different MLIP models.

## The Problem

Different MLIP models produce embeddings in different vector spaces:

- **MACE**: 128-dimensional
- **CHGNet**: 64-dimensional
- **ORB**: 256-dimensional
- **SevenNet**: Different architecture entirely

These spaces are **not directly comparable** - you can't simply compute distances between MACE and CHGNet embeddings.

## The Solution: Anchor-Based Projection

The key insight is that while embedding spaces differ, the **relationships** between structures should be similar across models. A stable perovskite should be "similar" to other perovskites regardless of which MLIP you use.

ManiFish uses **anchor structures** to create a common coordinate system:

```
Original Space          Anchor Space
     (128D)                (100D)
    ┌─────┐              ┌─────┐
    │  A  │──similarity──▶│ 0.9 │
    │     │  to anchor 1  │ 0.7 │ similarity to
    │  •  │──similarity──▶│ 0.3 │ each of 100
    │     │  to anchor 2  │ ... │ anchors
    └─────┘              └─────┘
```

## How It Works

### Step 1: Select Diverse Anchors

Select ~100 diverse reference structures to serve as anchors:

```python
from manifish import select_anchors

anchors = select_anchors(
    reference_embeddings,
    n_anchors=100,
    method="direct"  # Uses DIRECT clustering for diversity
)
```

### Step 2: Project to Anchor Space

For each structure, compute similarity to all anchors:

```python
from manifish import calculate_platonic_representation

# Project any MLIP's embeddings
platonic_coords = calculate_platonic_representation(
    embeddings=mace_embeddings,
    anchor_embeddings=anchors
)
# Shape: (n_structures, n_anchors) = (1000, 100)
```

### Step 3: Compare Across Models

Now both MACE and CHGNet embeddings live in the same 100-dimensional anchor space:

```python
mace_platonic = calculate_platonic_representation(mace_emb, anchors)
chgnet_platonic = calculate_platonic_representation(chgnet_emb, anchors)

# Now we can compare directly!
from manifish import compute_mutual_knn
similarity = compute_mutual_knn(mace_platonic, chgnet_platonic)
```

## Mathematical Details

The Platonic representation uses **cosine similarity** to each anchor:

$$
p_i = \left[ \cos(e, a_1), \cos(e, a_2), ..., \cos(e, a_k) \right]
$$

Where:
- $e$ is the original embedding
- $a_j$ is the $j$-th anchor embedding
- $k$ is the number of anchors (typically 100)

## Anchor Selection Methods

### DIRECT (Recommended)

Uses BIRCH clustering to find diverse representatives:

```python
anchors = select_anchors(embeddings, n_anchors=100, method="direct")
```

**Pros**: Good coverage, deterministic
**Cons**: Slightly slower

### Farthest Point Sampling (FPS)

Iteratively selects points maximizing minimum distance:

```python
anchors = select_anchors(embeddings, n_anchors=100, method="fps")
```

**Pros**: Fast, simple
**Cons**: Sensitive to initialization

## Best Practices

1. **Use 100 anchors** for most applications (good balance of coverage vs. dimensionality)

2. **Select anchors from your best MLIP** (usually MACE)

3. **Include diverse structures** in your anchor pool (different crystal systems, compositions)

4. **Keep anchors fixed** when comparing multiple generated sets

## Example: Full Workflow

```python
from manifish import (
    select_anchors,
    calculate_platonic_representation,
    ManifoldFishAnalyzer,
)

# 1. Select anchors from reference
anchors = select_anchors(reference_embeddings, n_anchors=100)

# 2. Project reference to Platonic space
ref_platonic = calculate_platonic_representation(reference_embeddings, anchors)

# 3. Project generated to Platonic space
gen_platonic = calculate_platonic_representation(generated_embeddings, anchors)

# 4. Analyze in Platonic space
analyzer = ManifoldFishAnalyzer(ref_platonic, reference_ids)
results = analyzer.analyze(gen_platonic, generated_ids)
```

## Why "Platonic"?

The name comes from Plato's Theory of Forms - the idea that there exist perfect, abstract "forms" that physical objects approximate.

In ManiFish, the **anchor structures** serve as these idealized forms, and each material is described by how it relates to these fundamental archetypes.
