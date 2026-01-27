# API Reference

## Core Modules

### Fish Analyzers

The main analysis classes for evaluating generated structures.

#### ManifoldFishAnalyzer

Base analyzer that classifies structures into 7 categories based on manifold position.

```python
from manifish import ManifoldFishAnalyzer

analyzer = ManifoldFishAnalyzer(
    reference_embeddings,      # np.ndarray (n_ref, dim)
    reference_ids=None,        # Optional[List[str]]
    n_neighbors=10,            # int
    boundary_method="auto",    # "convex_hull" | "alpha_shape" | "one_class_svm" | "auto"
    local_geometry_mode="fast", # "full" | "fast" | "skip"
    stability_threshold=0.5,   # float
    outlier_threshold=-1.5,    # float
    geometry_threshold=0.3,    # float
    sparse_threshold=20.0,     # float (percentile)
    depth_threshold=20.0,      # float (percentile)
    edge_margin=0.1,           # float
)

results = analyzer.analyze(generated_embeddings, material_ids)
analyzer.print_summary(results)
analyzer.export_results(results, "output.csv")
```

#### EnhancedManifoldFishAnalyzer

Extended analyzer with spatial structure metrics (LDS, CDS, SRSS, RMSC).

```python
from manifish import EnhancedManifoldFishAnalyzer

analyzer = EnhancedManifoldFishAnalyzer(
    reference_embeddings,
    reference_ids=None,
    region_labels=None,         # Optional region labels for SRSS
    # ... all ManifoldFishAnalyzer parameters ...
    spatial_k_local=10,         # k for local similarity
    spatial_n_distant=100,      # n for distant sampling
    use_spatial_for_confidence=True,  # Spatial metrics affect confidence
)

results = analyzer.analyze(generated_embeddings, material_ids)
analyzer.print_summary(results)

# Visualization methods
analyzer.plot_overview(results)
analyzer.plot_spatial_metrics(results)
analyzer.plot_pairwise(results)
analyzer.plot_confidence(results)
analyzer.plot_heatmap(results)
analyzer.generate_report(results, output_dir="./reports")
```

### The 7 Fish Categories

| Category | Description | Risk Level | Action |
|----------|-------------|------------|--------|
| `redundant_fish` | Deep inside manifold, dense region | very_low | Skip (redundant) |
| `fish_in_water` | Inside manifold, normal density | low | Standard validation |
| `frontier_fish` | Inside manifold, sparse region | low_medium | Priority for DFT |
| `edge_fish` | At manifold boundary | medium | Careful validation |
| `adventurous_fish` | Slightly outside manifold | medium_high | High priority DFT |
| `geometric_atypical` | High local PCA residual | medium_high | Investigate geometry |
| `structural_hallucination` | Far outside manifold | very_high | Reject |

---

### Embedding Transformation

#### EmbeddingTransformer

Base class for embedding transformations.

```python
from manifish import EmbeddingTransformer, AnchorTransformer, PCATransformer

# Anchor-based transformation (Platonic projection)
transformer = AnchorTransformer(
    anchor_embeddings,     # np.ndarray (n_anchors, dim)
    anchor_ids=None,       # Optional[List[str]]
    normalize=True,        # Normalize output
)
transformed = transformer.transform(embeddings)

# PCA-based transformation
pca_transformer = PCATransformer(n_components=100)
pca_transformer.fit(reference_embeddings)
transformed = pca_transformer.transform(embeddings)
```

---

### Spatial Metrics

Four metrics adapted from image generation research for manifold structure analysis.

```python
from manifish import (
    compute_lds,    # Local vs Distant Similarity
    compute_cds,    # Correlation Decay Slope
    compute_srss,   # Semantic-Region Self-Similarity
    compute_rmsc,   # RMS Spatial Contrast
    compute_all_spatial_metrics,
)

# Individual metrics
lds = compute_lds(query_coords, reference_coords, k_local=10)
cds = compute_cds(query_coords, reference_coords, distance_bins=[0.1, 0.5, 1.0, 2.0])
srss = compute_srss(query_coords, reference_coords, region_labels)
rmsc = compute_rmsc(query_coords, reference_coords, k=20)

# All metrics at once
metrics = compute_all_spatial_metrics(
    query_coords,
    reference_coords,
    region_labels=None,
    k_local=10,
    n_distant=100,
)
```

#### Metric Descriptions

| Metric | Description | Good Value |
|--------|-------------|------------|
| **LDS** | Ratio of local to distant similarity. High = strong local structure | > 1.5 |
| **CDS** | How quickly similarity decays with distance. High = good locality | > 0.5 |
| **SRSS** | Within-region vs cross-region coherence. High = regional consistency | > 1.0 |
| **RMSC** | Diversity in local representations. Moderate = good variety | 0.1-0.5 |

---

### Ensemble Analysis

Multi-MLIP ensemble analysis for cross-model comparison.

```python
from manifish import EnsembleAnalyzer

analyzer = EnsembleAnalyzer(
    mlip_embeddings={
        'mace': mace_embeddings,
        'chgnet': chgnet_embeddings,
        'matgl': matgl_embeddings,
    },
    anchor_embeddings=anchors,
    reference_ids=ref_ids,
)

# Analyze consensus
results = analyzer.analyze_ensemble()
print(results.consensus_score)
print(results.model_agreement)
```

---

### Data Filtering

Utilities for filtering and preprocessing data.

```python
from manifish import DataFilter

filter = DataFilter(
    embeddings,
    ids=material_ids,
)

# Filter by distance
filtered = filter.filter_by_distance(threshold=0.5)

# Filter outliers
filtered = filter.filter_outliers(contamination=0.1)

# Get subset by IDs
subset = filter.get_subset(selected_ids)
```

---

### Anchor Selection

Methods for selecting anchor points for Platonic projection.

```python
from manifish import (
    select_anchors,
    DIRECTAnchorSelector,
    FPSAnchorSelector,
    load_precomputed_anchors,
    MP20_ANCHOR_INDICES_100,
)

# Automatic selection
anchors = select_anchors(
    embeddings,
    n_anchors=100,
    method="direct",  # "direct" | "fps"
)

# Or use precomputed anchors for MP20 dataset
anchor_indices = MP20_ANCHOR_INDICES_100
```

---

### Platonic Projection

Project embeddings into anchor-based space for cross-model comparison.

```python
from manifish import (
    calculate_platonic_representation,
    project_to_anchor_space,
    PlatonicProjector,
)

# Simple projection
platonic_coords = project_to_anchor_space(embeddings, anchor_embeddings)

# Or use projector class
projector = PlatonicProjector(anchor_embeddings)
platonic_coords = projector.transform(embeddings)
```

---

### Distance-Based Metrics

Traditional manifold comparison metrics.

```python
from manifish import (
    compute_optimal_transport,
    compute_wasserstein_distance,
    compute_mutual_knn,
    compute_cka,
    compute_procrustes_distance,
    estimate_intrinsic_dimension,
)

# Compare two manifolds
ot_distance = compute_optimal_transport(manifold_a, manifold_b)
wasserstein = compute_wasserstein_distance(manifold_a, manifold_b)
mknn = compute_mutual_knn(manifold_a, manifold_b, k=10)
cka = compute_cka(manifold_a, manifold_b)
procrustes = compute_procrustes_distance(manifold_a, manifold_b)

# Intrinsic dimension
dim = estimate_intrinsic_dimension(manifold, method="mle")
```

---

## Result Classes

### ManifoldFishResult

```python
@dataclass
class ManifoldFishResult:
    index: int
    material_id: str
    manifold_distance: float      # Distance to nearest references
    depth_score: float            # 0=deep/central, 1=shallow/peripheral
    boundary_distance: float      # + inside, - outside
    local_density: float          # k-NN density estimate
    density_percentile: float     # 0-100
    local_pca_residual: float     # Geometry consistency
    geometry_consistent: bool
    category: str                 # One of 7 categories
    confidence: float             # 0-1
    risk_level: str              # very_low to very_high
    nearest_reference_ids: List[str]
```

### EnhancedManifoldFishResult

```python
@dataclass
class EnhancedManifoldFishResult:
    # All ManifoldFishResult fields plus:
    spatial_metrics: SpatialMetricsResult
    enhanced_confidence: float    # Confidence adjusted by spatial quality
```

### SpatialMetricsResult

```python
@dataclass
class SpatialMetricsResult:
    lds: float           # Local vs Distant Similarity
    cds: float           # Correlation Decay Slope
    srss: float          # Semantic-Region Self-Similarity
    rmsc: float          # RMS Spatial Contrast
    spatial_quality: float  # Combined score (-1 to 1)
```
