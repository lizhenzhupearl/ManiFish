# Ensemble Analysis

This tutorial covers multi-MLIP ensemble analysis for robust structure evaluation.

## Why Ensemble Analysis?

Single MLIP models can have blind spots. By analyzing structures across multiple MLIPs, you can:

- Identify structures where all models agree (high confidence)
- Flag structures with model disagreement (needs validation)
- Get more robust stability predictions

## Using EnsembleAnalyzer

```python
from manifish import EnsembleAnalyzer
import numpy as np

# Load embeddings from multiple MLIPs
mace_embeddings = np.load("mace_embeddings.npy")
chgnet_embeddings = np.load("chgnet_embeddings.npy")
orb_embeddings = np.load("orb_embeddings.npy")

# Material IDs (same order for all)
material_ids = [f"gen-{i}" for i in range(len(mace_embeddings))]

# Create ensemble analyzer
analyzer = EnsembleAnalyzer(
    mlip_embeddings={
        'mace': mace_embeddings,
        'chgnet': chgnet_embeddings,
        'orb': orb_embeddings,
    },
    material_ids=material_ids,
)

# Run ensemble analysis
results = analyzer.analyze_ensemble()
```

## Understanding Results

```python
# Overall ensemble metrics
print(f"Mean consensus: {results.mean_consensus:.3f}")
print(f"Model agreement: {results.model_agreement}")

# Per-structure results
for i, r in enumerate(results.structure_results[:5]):
    print(f"Structure {i}:")
    print(f"  Consensus: {r.consensus:.3f}")
    print(f"  Categories: {r.category_votes}")
```

## Filtering by Consensus

```python
# Get high-confidence structures
high_confidence = [
    r for r in results.structure_results
    if r.consensus > 0.8
]
print(f"High confidence: {len(high_confidence)} structures")

# Get structures needing validation
needs_validation = [
    r for r in results.structure_results
    if r.consensus < 0.5
]
print(f"Needs validation: {len(needs_validation)} structures")
```

## Combining with ManifoldFishAnalyzer

```python
from manifish import ManifoldFishAnalyzer, aggregate_atom_embeddings

# Use ensemble consensus to weight analysis
# High consensus → more reliable reference

# 1. Identify reliable reference structures
reliable_mask = np.array([r.consensus > 0.7 for r in results.structure_results])
reliable_embeddings = mace_embeddings[reliable_mask]
reliable_ids = [material_ids[i] for i in np.where(reliable_mask)[0]]

# 2. Create analyzer with reliable reference
analyzer = ManifoldFishAnalyzer(
    reliable_embeddings,
    reliable_ids,
    boundary_method="alpha_shape"
)

# 3. Analyze uncertain structures
uncertain_mask = ~reliable_mask
uncertain_embeddings = mace_embeddings[uncertain_mask]
uncertain_ids = [material_ids[i] for i in np.where(uncertain_mask)[0]]

fish_results = analyzer.analyze(uncertain_embeddings, uncertain_ids)
analyzer.print_summary(fish_results)
```

## Visualization

```python
import matplotlib.pyplot as plt

# Plot consensus distribution
consensus_scores = [r.consensus for r in results.structure_results]

plt.figure(figsize=(10, 6))
plt.hist(consensus_scores, bins=50, edgecolor='black')
plt.xlabel('Consensus Score')
plt.ylabel('Count')
plt.title('Distribution of Model Consensus')
plt.axvline(0.5, color='r', linestyle='--', label='Validation threshold')
plt.axvline(0.8, color='g', linestyle='--', label='High confidence')
plt.legend()
plt.savefig('consensus_distribution.png')
```

## Best Practices

1. **Use at least 3 MLIPs** for meaningful consensus
2. **Include diverse model architectures** (e.g., MACE, CHGNet, ORB)
3. **Weight by model reliability** if you have DFT validation data
4. **Prioritize low-consensus structures** for DFT validation
