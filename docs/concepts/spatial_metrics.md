# Spatial Metrics

ManiFish includes four spatial metrics adapted from image generation research to analyze local manifold structure.

## Overview

These metrics capture different aspects of how generated structures relate to the local geometry of the reference manifold:

| Metric | Full Name | Measures |
|--------|-----------|----------|
| **LDS** | Local vs Distant Similarity | Local clustering strength |
| **CDS** | Correlation Decay Slope | How similarity changes with distance |
| **SRSS** | Semantic-Region Self-Similarity | Coherence within chemical regions |
| **RMSC** | RMS Spatial Contrast | Diversity of local representations |

## LDS: Local vs Distant Similarity

**What it measures**: The ratio of similarity to local neighbors vs. distant points.

**Interpretation**:
- High LDS (> 1.5): Strong local structure, point fits well with neighbors
- Low LDS (< 1.0): Weak local structure, may be in transition zone

```python
from manifish import compute_lds

lds = compute_lds(
    query_coords=generated_embeddings,
    reference_coords=reference_embeddings,
    k_local=10,      # Number of local neighbors
    n_distant=100,   # Number of distant samples
)
```

## CDS: Correlation Decay Slope

**What it measures**: How quickly similarity decays with distance.

**Interpretation**:
- High CDS (> 0.5): Good locality - similar things are nearby
- Low CDS: Poor locality - structure doesn't respect distance

```python
from manifish import compute_cds

cds = compute_cds(
    query_coords=generated_embeddings,
    reference_coords=reference_embeddings,
    distance_bins=[0.1, 0.5, 1.0, 2.0, 5.0],
)
```

## SRSS: Semantic-Region Self-Similarity

**What it measures**: Whether points in the same chemical region are more similar to each other than to other regions.

**Interpretation**:
- High SRSS (> 1.0): Good regional consistency
- Low SRSS: Points don't respect chemical boundaries

```python
from manifish import compute_srss

# Requires region labels (e.g., crystal system, composition family)
region_labels = ["cubic", "cubic", "hexagonal", ...]

srss = compute_srss(
    query_coords=generated_embeddings,
    reference_coords=reference_embeddings,
    region_labels=region_labels,
)
```

### Creating Region Labels

```python
from manifish import (
    create_region_labels_from_crystal_system,
    create_region_labels_from_composition_family,
)

# From crystal systems
labels = create_region_labels_from_crystal_system(structures)

# From composition (binary, ternary, etc.)
labels = create_region_labels_from_composition_family(compositions)
```

## RMSC: RMS Spatial Contrast

**What it measures**: The diversity of local representations.

**Interpretation**:
- Moderate RMSC (0.1-0.5): Good variety without chaos
- Very low RMSC: Too uniform (mode collapse)
- Very high RMSC: Too chaotic

```python
from manifish import compute_rmsc

rmsc = compute_rmsc(
    query_coords=generated_embeddings,
    reference_coords=reference_embeddings,
    k=20,
)
```

## Computing All Metrics

```python
from manifish import compute_all_spatial_metrics

metrics = compute_all_spatial_metrics(
    query_coords=generated_embeddings,
    reference_coords=reference_embeddings,
    region_labels=region_labels,  # Optional
    k_local=10,
    n_distant=100,
)

print(f"LDS: {metrics.lds:.3f}")
print(f"CDS: {metrics.cds:.3f}")
print(f"SRSS: {metrics.srss:.3f}")
print(f"RMSC: {metrics.rmsc:.3f}")
print(f"Overall quality: {metrics.spatial_quality:.3f}")
```

## Spatial Quality Score

ManiFish combines the four metrics into a single **spatial quality score** (-1 to 1):

```python
# Already computed in metrics object
quality = metrics.spatial_quality

if quality > 0.5:
    print("Excellent manifold conformity")
elif quality > 0:
    print("Good manifold conformity")
else:
    print("Poor manifold conformity - investigate")
```

## Using with EnhancedManifoldFishAnalyzer

```python
from manifish import EnhancedManifoldFishAnalyzer

analyzer = EnhancedManifoldFishAnalyzer(
    reference_embeddings,
    reference_ids,
    region_labels=region_labels,
    use_spatial_for_confidence=True,  # Spatial metrics affect confidence
)

results = analyzer.analyze(generated_embeddings, generated_ids)

# Results include spatial metrics
for r in results[:5]:
    print(f"{r.material_id}:")
    print(f"  Category: {r.category}")
    print(f"  Spatial quality: {r.spatial_metrics.spatial_quality:.3f}")
```

## Interpretation Guide

| Scenario | LDS | CDS | SRSS | RMSC | Interpretation |
|----------|-----|-----|------|------|----------------|
| Good fit | High | High | High | Moderate | Excellent candidate |
| Transition zone | Low | High | Varies | Moderate | May be at boundary |
| Wrong region | High | High | Low | Moderate | Check composition |
| Mode collapse | High | High | High | Very low | Lacks diversity |
| Chaotic | Low | Low | Low | High | Likely artifacts |
