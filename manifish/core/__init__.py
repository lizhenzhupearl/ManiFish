"""
Core modules for ManiFish.

- anchors: Anchor selection using DIRECT sampler or FPS
- manifold: Platonic projection for cross-MLIP comparison
- metrics: Comparison metrics (OT, Mutual KNN, CKA, etc.)
- spatial_metrics: Spatial structure metrics (LDS, CDS, SRSS, RMSC)
- enhanced_evaluator: Combined evaluator with distance and spatial metrics
- fish_analyzer: ManifoldFishAnalyzer (7-category classifier)
- enhanced_fish_analyzer: EnhancedManifoldFishAnalyzer (with spatial metrics)
"""

from .anchors import (
    select_anchors,
    DIRECTAnchorSelector,
    FPSAnchorSelector,
    Anchor,
    AnchorSet,
    load_precomputed_anchors,
    MP20_ANCHOR_INDICES_100,
)

from .manifold import (
    calculate_platonic_representation,
    project_to_anchor_space,
    PlatonicProjector,
    MultiModelProjector,
    PlatonicRepresentation,
    MultiModelRepresentation,
)

from .metrics import (
    compute_optimal_transport,
    compute_wasserstein_distance,
    compute_mutual_knn,
    compute_mutual_knn_matrix,
    estimate_intrinsic_dimension,
    compare_intrinsic_dimensions,
    compute_cka,
    compute_procrustes_distance,
    compute_consensus,
    compute_quality_score,
)

from .spatial_metrics import (
    compute_lds,
    compute_cds,
    compute_srss,
    compute_srss_from_composition,
    compute_rmsc,
    compute_all_spatial_metrics,
    compute_spatial_metrics_batch,
    compute_spatial_quality,
    SpatialMetrics,
    SpatialMetricsConfig,
    create_region_labels_from_crystal_system,
    create_region_labels_from_composition_family,
    normalize_spatial_metrics,
)

from .enhanced_evaluator import (
    SpatialManifoldEvaluator,
    evaluate_structure,
    evaluate_structures_batch,
    compare_distance_vs_spatial,
    compute_all_distance_metrics,
    compute_manifold_distance,
    compute_knn_distance,
    compute_local_density,
    compute_hull_distance,
    compute_isolation_score,
    distance_to_stability,
    DistanceMetrics,
    StabilityPrediction,
    EvaluatorConfig,
)

from .structure_evaluator import (
    StructureEvaluator,
    ManifoldReference,
    EvaluationResult,
    quick_evaluate,
    evaluate_from_embeddings,
)

from .enhanced_fish_analyzer import (
    EnhancedManifoldFishAnalyzer,
    EnhancedManifoldFishResult,
    SpatialMetricsResult,
    CategorySummary,
    CATEGORIES,
    RISK_LEVELS,
)

from .fish_analyzer import (
    ManifoldFishAnalyzer,
    ManifoldFishResult,
    aggregate_atom_embeddings,
    aggregate_from_structure_list,
    aggregate_by_material_id,
    aggregate_with_weights,
)

from .transforms import (
    EmbeddingTransformer,
    AnchorTransformer,
    PCATransformer,
    TransformResult,
)

from .ensemble import (
    EnsembleAnalyzer,
    EnsembleResult,
    ManifoldResult,
)

from .data_filter import (
    DataFilter,
)

__all__ = [
    # Anchors
    "select_anchors",
    "DIRECTAnchorSelector",
    "FPSAnchorSelector",
    "Anchor",
    "AnchorSet",
    "load_precomputed_anchors",
    "MP20_ANCHOR_INDICES_100",
    # Manifold
    "calculate_platonic_representation",
    "project_to_anchor_space",
    "PlatonicProjector",
    "MultiModelProjector",
    "PlatonicRepresentation",
    "MultiModelRepresentation",
    # Metrics (distance-based)
    "compute_optimal_transport",
    "compute_wasserstein_distance",
    "compute_mutual_knn",
    "compute_mutual_knn_matrix",
    "estimate_intrinsic_dimension",
    "compare_intrinsic_dimensions",
    "compute_cka",
    "compute_procrustes_distance",
    "compute_consensus",
    "compute_quality_score",
    # Spatial Metrics
    "compute_lds",
    "compute_cds",
    "compute_srss",
    "compute_srss_from_composition",
    "compute_rmsc",
    "compute_all_spatial_metrics",
    "compute_spatial_metrics_batch",
    "compute_spatial_quality",
    "SpatialMetrics",
    "SpatialMetricsConfig",
    "create_region_labels_from_crystal_system",
    "create_region_labels_from_composition_family",
    "normalize_spatial_metrics",
    # Enhanced Evaluator
    "SpatialManifoldEvaluator",
    "evaluate_structure",
    "evaluate_structures_batch",
    "compare_distance_vs_spatial",
    "compute_all_distance_metrics",
    "compute_manifold_distance",
    "compute_knn_distance",
    "compute_local_density",
    "compute_hull_distance",
    "compute_isolation_score",
    "distance_to_stability",
    "DistanceMetrics",
    "StabilityPrediction",
    "EvaluatorConfig",
    # Structure Evaluator (high-level interface)
    "StructureEvaluator",
    "ManifoldReference",
    "EvaluationResult",
    "quick_evaluate",
    "evaluate_from_embeddings",
    # Enhanced Fish Analyzer (7-category classifier with spatial metrics)
    "EnhancedManifoldFishAnalyzer",
    "EnhancedManifoldFishResult",
    "SpatialMetricsResult",
    "CategorySummary",
    "CATEGORIES",
    "RISK_LEVELS",
    # Base Fish Analyzer (without spatial metrics)
    "ManifoldFishAnalyzer",
    "ManifoldFishResult",
    # Aggregation utilities
    "aggregate_atom_embeddings",
    "aggregate_from_structure_list",
    "aggregate_by_material_id",
    "aggregate_with_weights",
    # Transforms
    "EmbeddingTransformer",
    "AnchorTransformer",
    "PCATransformer",
    "TransformResult",
    # Ensemble Analysis
    "EnsembleAnalyzer",
    "EnsembleResult",
    "ManifoldResult",
    # Data Filter
    "DataFilter",
]
