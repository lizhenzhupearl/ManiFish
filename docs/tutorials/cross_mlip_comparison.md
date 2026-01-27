# Cross-MLIP Comparison

This tutorial shows how to compare structures across different MLIP models using ManiFish's Platonic representation.

## The Challenge

Different MLIP models produce embeddings in different spaces:
- MACE might produce 128-dimensional embeddings
- CHGNet might produce 64-dimensional embeddings
- These spaces are not directly comparable

## The Solution: Platonic Representation

ManiFish projects all embeddings into a common **anchor-based space** where they can be directly compared.

```
MACE embeddings (128D) ──┐
                         ├──> Anchor Space (100D) ──> Compare
CHGNet embeddings (64D) ─┘
```

## Step 1: Select Anchors

Anchors are diverse reference points that define the common space.

```python
from manifish import select_anchors

# Use MACE embeddings to select anchors
anchor_set = select_anchors(
    embeddings=mace_reference_embeddings,
    n_anchors=100,
    method="direct",  # DIRECT clustering for diversity
    model_name="mace"
)

print(f"Selected {len(anchor_set.anchors)} anchors")
print(f"Anchor shape: {anchor_set.embeddings.shape}")  # (100, 128)
```

## Step 2: Project to Platonic Space

```python
from manifish import calculate_platonic_representation

# Project MACE embeddings
mace_platonic = calculate_platonic_representation(
    embeddings=mace_embeddings,
    anchor_embeddings=anchor_set,
    model_name="mace"
)

# Project CHGNet embeddings
chgnet_platonic = calculate_platonic_representation(
    embeddings=chgnet_embeddings,
    anchor_embeddings=anchor_set,  # Same anchors!
    model_name="chgnet"
)

print(f"MACE Platonic: {mace_platonic.shape}")    # (n_samples, 100)
print(f"CHGNet Platonic: {chgnet_platonic.shape}")  # (n_samples, 100)
```

## Step 3: Compare with Metrics

```python
from manifish import (
    compute_mutual_knn,
    compute_cka,
    compute_consensus,
)

# Mutual KNN - neighborhood preservation
mknn = compute_mutual_knn(mace_platonic, chgnet_platonic, k=10)
print(f"Mutual KNN: {mknn:.4f}")  # Higher = more similar

# CKA - structural similarity
cka = compute_cka(mace_platonic, chgnet_platonic)
print(f"CKA: {cka:.4f}")  # Higher = more similar

# Per-structure consensus
consensus = compute_consensus({
    'mace': mace_platonic,
    'chgnet': chgnet_platonic,
})
print(f"Mean consensus: {consensus.mean():.4f}")
print(f"High consensus (>0.8): {(consensus > 0.8).sum()} structures")
```

## Step 4: Identify Disagreements

```python
import numpy as np

# Find structures where models disagree
low_consensus = np.where(consensus < 0.5)[0]
print(f"Models disagree on {len(low_consensus)} structures")

# These might need DFT validation
for idx in low_consensus[:5]:
    print(f"  Structure {idx}: consensus = {consensus[idx]:.3f}")
```

## Using PlatonicProjector Class

For more control, use the `PlatonicProjector` class:

```python
from manifish import PlatonicProjector

# Create projector
projector = PlatonicProjector(
    anchor_set,
    similarity_method="cosine"  # or "euclidean"
)

# Project multiple models
mace_rep = projector.project(mace_embeddings, model_name="mace")
chgnet_rep = projector.project(chgnet_embeddings, model_name="chgnet")

# Access metadata
print(f"Samples: {mace_rep.n_samples}")
print(f"Anchors: {mace_rep.n_anchors}")
```

## Complete Multi-MLIP Workflow

```python
from manifish import (
    select_anchors,
    PlatonicProjector,
    ManifoldFishAnalyzer,
    compute_consensus,
)
import numpy as np

# 1. Load embeddings from multiple MLIPs
mlip_embeddings = {
    'mace': np.load("mace_embeddings.npy"),
    'chgnet': np.load("chgnet_embeddings.npy"),
    'orb': np.load("orb_embeddings.npy"),
}

# 2. Select anchors from one model
anchors = select_anchors(mlip_embeddings['mace'], n_anchors=100)
projector = PlatonicProjector(anchors)

# 3. Project all to common space
platonic_reps = {}
for name, emb in mlip_embeddings.items():
    platonic_reps[name] = projector.project(emb, model_name=name).platonic_coords

# 4. Compute consensus
consensus = compute_consensus(platonic_reps)

# 5. Analyze with high-consensus subset
high_consensus_mask = consensus > 0.7
reliable_embeddings = platonic_reps['mace'][high_consensus_mask]

# 6. Use ManifoldFishAnalyzer on reliable structures
analyzer = ManifoldFishAnalyzer(reliable_embeddings[:500])  # Use some as reference
results = analyzer.analyze(reliable_embeddings[500:])
analyzer.print_summary(results)
```

## Interpreting Results

| Consensus Score | Interpretation |
|-----------------|----------------|
| > 0.8 | High agreement - reliable prediction |
| 0.5 - 0.8 | Moderate agreement - worth validating |
| < 0.5 | Low agreement - needs DFT verification |

## Tips

1. **Use the same anchors** for all MLIPs to ensure comparability
2. **Select anchors from your most trusted MLIP** (often MACE)
3. **100 anchors** is usually sufficient for most applications
4. **High consensus structures** are more likely to be stable
