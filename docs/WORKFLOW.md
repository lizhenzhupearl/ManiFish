# ManiFish Workflow Guide

This document describes the standard workflow for evaluating AI-generated crystal structures using the ManiFish framework.

## Overview

ManiFish uses a two-stage evaluation process:
1. **Embedding Extraction & Transformation**: Extract MLIP embeddings and project to anchor space (Platonic representation)
2. **Manifold Fish Analysis**: Classify structures into 6 categories based on their position in the manifold

## The Six Fish Categories

| Category | Description | Risk | Action |
|----------|-------------|------|--------|
| Redundant Fish | Deep inside, dense region | Very Low | Skip (redundant) |
| Fish in Water | Inside manifold, normal density | Low | Standard validation |
| Frontier Fish | Sparse region | Low or High* | Priority for DFT |
| Edge Fish | At manifold boundary | Medium | Careful validation |
| Adventurous Fish | Outside manifold | Medium to High* | High priority DFT |
| Structural Hallucination | Bad geometry + LOF outlier | Very High | Reject |

*Risk level varies based on geometry consistency - check `risk_level` in results.

## Standard Workflow

### Step 1: Prepare Reference Data

```python
# Load stable training structures (e.g., MP-20 dataset)
from pymatgen.core import Structure
import numpy as np

# Load your reference structures
reference_structures = [...]  # List of pymatgen/ASE structures
reference_ids = [...]  # List of material IDs
crystal_systems = [...]  # Optional: for SRSS metric
```

### Step 2: Extract Embeddings

```python
from manifish.models import MACEEmbeddingExtractor

# Initialize extractor
extractor = MACEEmbeddingExtractor(model='medium', device='cuda')

# Extract embeddings (per-atom)
reference_embeddings = extractor.get_embeddings(reference_structures)

# Aggregate to per-structure (mean pooling)
ref_emb = np.stack([np.mean(e, axis=0) for e in reference_embeddings])
```

### Step 3: Project to Anchor Space (Platonic Representation)

```python
from manifish import select_anchors, calculate_platonic_representation

# Select anchors (100 diverse structures)
anchor_set = select_anchors(
    embeddings=ref_emb,
    n_anchors=100,
    method='direct',
    model_name='mace'
)

# Project to anchor space
ref_coords = calculate_platonic_representation(
    embeddings=ref_emb,
    anchor_embeddings=anchor_set,
    model_name='mace'
)
```

### Step 4: Initialize Enhanced Fish Analyzer

```python
from manifish.core.enhanced_fish_analyzer import EnhancedManifoldFishAnalyzer

# Create region labels for SRSS (optional but recommended)
from manifish import create_region_labels_from_crystal_system
region_labels = create_region_labels_from_crystal_system(crystal_systems)

# Initialize analyzer with reference data
analyzer = EnhancedManifoldFishAnalyzer(
    reference_embeddings=ref_coords,  # Already transformed!
    reference_ids=reference_ids,
    region_labels=region_labels,  # For SRSS metric
)

# Save for later use
analyzer.save('manifold_analyzer.pkl')
```

### Step 5: Analyze Generated Structures

```python
# Load generated structures
generated_structures = [...]  # From MatterGen, etc.
generated_ids = [...]

# Extract and transform generated embeddings
gen_emb = extractor.get_embeddings(generated_structures)
gen_emb = np.stack([np.mean(e, axis=0) for e in gen_emb])
gen_coords = calculate_platonic_representation(
    embeddings=gen_emb,
    anchor_embeddings=anchor_set,
)

# Analyze with enhanced metrics
results = analyzer.analyze(gen_coords, generated_ids)

# Print summary
analyzer.print_summary(results)

# Export to CSV
analyzer.export_results(results, 'analysis_results.csv')
```

## Quick Workflow (If Already Have Transformed Embeddings)

If you already have Platonic coordinates computed:

```python
from manifish.core.enhanced_fish_analyzer import EnhancedManifoldFishAnalyzer

# Load pre-computed coordinates
ref_coords = np.load('reference_platonic_coords.npy')
gen_coords = np.load('generated_platonic_coords.npy')
ref_ids = [...]
gen_ids = [...]

# Initialize and analyze
analyzer = EnhancedManifoldFishAnalyzer(ref_coords, ref_ids)
results = analyzer.analyze(gen_coords, gen_ids)
analyzer.print_summary(results)
```

## Understanding the Output

Each result contains:

### Original Metrics
- `manifold_distance`: Normalized distance to nearest references
- `depth_score`: Distance to manifold centroid (0=deep, 1=shallow)
- `boundary_distance`: Distance to boundary (+ inside, - outside)
- `local_density`: k-NN density estimate
- `density_percentile`: Percentile vs reference (0-100)
- `local_pca_residual`: Geometry consistency

### NEW: Spatial Metrics
- `lds`: Local vs Distant Similarity (higher = better clustering)
- `cds`: Correlation Decay Slope (higher = sharper locality)
- `srss`: Semantic-Region Self-Similarity (higher = fits chemical family)
- `rmsc`: RMS Spatial Contrast (moderate = good diversity)
- `spatial_quality`: Combined spatial quality score

### Classification
- `category`: One of the 6 fish categories
- `confidence`: Original classification confidence
- `enhanced_confidence`: Confidence incorporating spatial metrics
- `risk_level`: Risk assessment

## Example: Filtering by Category

```python
# Get structures for DFT validation
for r in results:
    if r.category in ['frontier_fish', 'adventurous_fish']:
        if r.enhanced_confidence > 0.6:
            print(f"{r.material_id}: Priority for DFT")
            print(f"  Spatial quality: {r.spatial_metrics.spatial_quality:.3f}")
            print(f"  LDS: {r.spatial_metrics.lds:.3f}")
```

## Example: Summary Statistics

```python
# Category summary
summary = analyzer.get_category_summary(results)

for cat_name, cat_summary in summary.items():
    if cat_summary.count > 0:
        print(f"{cat_name}: {cat_summary.count} structures")
        print(f"  Mean spatial quality: {cat_summary.mean_spatial_quality:.3f}")
        print(f"  Materials: {cat_summary.material_ids[:5]}...")
```

## Saving and Loading

```python
# Save analyzer for reuse
analyzer.save('analyzer.pkl')

# Load later
from manifish.core.enhanced_fish_analyzer import EnhancedManifoldFishAnalyzer
analyzer = EnhancedManifoldFishAnalyzer.load('analyzer.pkl')
```

## Key Files

- `manifish/core/spatial_metrics.py`: LDS, CDS, SRSS, RMSC implementations
- `manifish/core/enhanced_evaluator.py`: Combined distance + spatial evaluator
- `manifish/core/enhanced_fish_analyzer.py`: 6-fish category classifier with spatial metrics
- `manifish/core/structure_evaluator.py`: End-to-end structure evaluation

## Notes

1. **Transformed embeddings are required**: The analyzer works with Platonic coordinates (anchor-projected embeddings), not raw MLIP embeddings.

2. **Region labels improve SRSS**: Providing crystal system or composition-based labels enables the Semantic-Region Self-Similarity metric.

3. **Spatial metrics complement distance metrics**: The enhanced confidence score combines traditional manifold distance with spatial structure quality.

4. **Memory for large datasets**: For very large reference sets (>100k), consider using `local_geometry_mode='skip'` to reduce memory usage.
