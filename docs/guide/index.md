# User Guide

## Complete Workflow

This guide walks through the complete ManiFish workflow from raw embeddings to final analysis.

### Step 1: Prepare Your Data

You need:
1. **Reference embeddings**: Known stable structures (e.g., from Materials Project)
2. **Generated embeddings**: Structures to evaluate
3. **Anchor embeddings**: For Platonic projection (optional, can use reference subset)

```python
import numpy as np

# Load your embeddings (from any MLIP)
ref_emb = np.load("reference_embeddings.npy")  # (n_ref, dim)
gen_emb = np.load("generated_embeddings.npy")  # (n_gen, dim)
ref_ids = load_ids("reference_ids.json")
gen_ids = load_ids("generated_ids.json")
```

### Step 2: Transform to Platonic Space (if needed)

If comparing across different MLIPs, transform to common Platonic space:

```python
from manifish import AnchorTransformer

# Select or load anchors
anchor_emb = ref_emb[np.random.choice(len(ref_emb), 100, replace=False)]

# Or use precomputed anchors
from manifish import MP20_ANCHOR_INDICES_100
anchor_emb = ref_emb[MP20_ANCHOR_INDICES_100]

# Transform
transformer = AnchorTransformer(anchor_emb)
ref_platonic = transformer.transform(ref_emb)
gen_platonic = transformer.transform(gen_emb)
```

### Step 3: Analyze Generated Structures

#### Basic Analysis (without spatial metrics)

```python
from manifish import ManifoldFishAnalyzer

analyzer = ManifoldFishAnalyzer(ref_platonic, ref_ids)
results = analyzer.analyze(gen_platonic, gen_ids)
analyzer.print_summary(results)
```

#### Enhanced Analysis (with spatial metrics)

```python
from manifish import EnhancedManifoldFishAnalyzer

analyzer = EnhancedManifoldFishAnalyzer(ref_platonic, ref_ids)
results = analyzer.analyze(gen_platonic, gen_ids)
analyzer.print_summary(results)
```

Output:
```
======================================================================
ENHANCED MANIFOLD FISH ANALYSIS SUMMARY
======================================================================
Total structures analyzed: 10000
----------------------------------------------------------------------
Category                     Count        % Risk          Spatial Q
----------------------------------------------------------------------
redundant_fish                8500    85.0% very_low          0.234
edge_fish                     1200    12.0% medium           -0.012
frontier_fish                  200     2.0% low_medium        0.156
structural_hallucination       100     1.0% very_high        -0.345
----------------------------------------------------------------------

RECOMMENDED ACTIONS:
  frontier_fish (200): Priority for DFT
  edge_fish (1200): Careful validation
  structural_hallucination (100): Reject
  redundant_fish (8500): Skip (redundant)
```

### Step 4: Visualize Results

```python
# Overview plot
analyzer.plot_overview(results)

# Spatial metrics distribution
analyzer.plot_spatial_metrics(results)

# Pairwise metric scatter
analyzer.plot_pairwise(results)

# Confidence comparison
analyzer.plot_confidence(results)

# Heatmap of metrics by category
analyzer.plot_heatmap(results)

# Generate all plots to a directory
analyzer.generate_report(results, output_dir="./analysis_report")
```

### Step 5: Export Results

```python
# Export to CSV
analyzer.export_results(results, "analysis_results.csv")

# Get specific categories
summary = analyzer.get_category_summary(results)
frontier_ids = summary["frontier_fish"].material_ids
hallucination_ids = summary["structural_hallucination"].material_ids

# Filter for DFT prioritization
priority_for_dft = [r for r in results
                   if r.category in ["frontier_fish", "adventurous_fish"]
                   and r.enhanced_confidence > 0.7]
```

---

## Typical Use Cases

### Use Case 1: Filter Generated Structures

```python
from manifish import EnhancedManifoldFishAnalyzer

# Analyze
analyzer = EnhancedManifoldFishAnalyzer(ref_emb, ref_ids)
results = analyzer.analyze(gen_emb, gen_ids)

# Get structures to reject
rejects = [r.material_id for r in results
           if r.category == "structural_hallucination"]

# Get high-value candidates
candidates = [r.material_id for r in results
              if r.category in ["frontier_fish", "adventurous_fish"]
              and r.spatial_metrics.spatial_quality > 0]

print(f"Rejecting {len(rejects)} hallucinations")
print(f"Found {len(candidates)} high-value candidates")
```

### Use Case 2: Compare Multiple MLIPs

```python
from manifish import AnchorTransformer, EnhancedManifoldFishAnalyzer

# Common anchor set
anchors = ref_mace[np.random.choice(len(ref_mace), 100, replace=False)]
transformer = AnchorTransformer(anchors)

# Transform each MLIP's embeddings
mace_platonic = transformer.transform(gen_mace)
chgnet_platonic = transformer.transform(gen_chgnet)
matgl_platonic = transformer.transform(gen_matgl)

# Analyze each
ref_platonic = transformer.transform(ref_mace)
analyzer = EnhancedManifoldFishAnalyzer(ref_platonic, ref_ids)

for name, emb in [("MACE", mace_platonic), ("CHGNet", chgnet_platonic), ("M3GNet", matgl_platonic)]:
    results = analyzer.analyze(emb, gen_ids)
    print(f"\n{name}:")
    analyzer.print_summary(results)
```

### Use Case 3: Ensemble Analysis

```python
from manifish import EnsembleAnalyzer

# Combine multiple MLIP analyses
ensemble = EnsembleAnalyzer(
    mlip_embeddings={
        'mace': gen_mace,
        'chgnet': gen_chgnet,
        'matgl': gen_matgl,
    },
    anchor_embeddings=anchors,
    reference_ids=gen_ids,
)

# Get consensus
results = ensemble.analyze_ensemble()
high_consensus = [r for r in results if r.consensus_score > 0.8]
```

### Use Case 4: Region-Aware Analysis

```python
from manifish import EnhancedManifoldFishAnalyzer, create_region_labels_from_crystal_system

# Create region labels from crystal systems
region_labels = create_region_labels_from_crystal_system(crystal_systems)

# Analyze with region-aware SRSS
analyzer = EnhancedManifoldFishAnalyzer(
    ref_platonic,
    ref_ids,
    region_labels=region_labels,
)
results = analyzer.analyze(gen_platonic, gen_ids)
```

---

## Best Practices

### Anchor Selection

- Use **100-200 anchors** for good coverage
- Select diverse anchors using **DIRECT** or **FPS** methods
- For reproducibility, use **precomputed anchors** (e.g., `MP20_ANCHOR_INDICES_100`)

### Reference Set

- Use a **high-quality reference set** of known stable structures
- Include diverse compositions and crystal systems
- **10,000+ structures** recommended for robust statistics

### Thresholds

Default thresholds work well for most cases. Adjust if needed:

```python
analyzer = EnhancedManifoldFishAnalyzer(
    ref_emb, ref_ids,
    stability_threshold=0.5,   # Increase for stricter "inside" criterion
    sparse_threshold=20.0,     # Increase to be more strict about "frontier"
    geometry_threshold=0.3,    # Increase for more tolerance of unusual geometry
)
```

### Performance

For large datasets (>100K structures):

```python
# Skip local geometry for speed
analyzer = EnhancedManifoldFishAnalyzer(
    ref_emb, ref_ids,
    local_geometry_mode="skip",  # 10x faster
)
```

### Reproducibility

Set random seeds for reproducible results:

```python
results = analyzer.analyze(gen_emb, gen_ids, random_state=42)
```
