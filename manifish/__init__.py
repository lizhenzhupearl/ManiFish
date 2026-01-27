"""
ManiFish: Unified Manifold Framework for Cross-MLIP Materials Discovery

A framework for evaluating and comparing AI-generated crystal structures
across different Machine Learning Interatomic Potentials using Platonic
(anchor-based) representation.

The key innovation is projecting embeddings from different MLIPs into a
common anchor space using cosine similarity, enabling direct comparison.

New in 0.2.0: Spatial structure metrics (LDS, CDS, SRSS, RMSC) for
improved stability prediction based on local manifold structure.
"""

__version__ = "0.2.0"
__author__ = "Zhen Zhu"

# Core anchor selection
from manifish.core.anchors import (
    select_anchors,
    DIRECTAnchorSelector,
    FPSAnchorSelector,
    Anchor,
    AnchorSet,
    load_precomputed_anchors,
    MP20_ANCHOR_INDICES_100,
)

# Platonic projection
from manifish.core.manifold import (
    calculate_platonic_representation,
    project_to_anchor_space,
    PlatonicProjector,
    MultiModelProjector,
    PlatonicRepresentation,
    MultiModelRepresentation,
)

# Distance-based metrics
from manifish.core.metrics import (
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

# Spatial structure metrics
from manifish.core.spatial_metrics import (
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

# Enhanced evaluator combining distance and spatial metrics
from manifish.core.enhanced_evaluator import (
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

# High-level structure evaluator (main interface)
from manifish.core.structure_evaluator import (
    StructureEvaluator,
    ManifoldReference,
    EvaluationResult,
    quick_evaluate,
    evaluate_from_embeddings,
)

# Enhanced Fish Analyzer (7-category classifier with spatial metrics)
from manifish.core.enhanced_fish_analyzer import (
    EnhancedManifoldFishAnalyzer,
    EnhancedManifoldFishResult,
    SpatialMetricsResult,
    CategorySummary,
    CATEGORIES,
    RISK_LEVELS,
)

# Base Fish Analyzer (without spatial metrics)
from manifish.core.fish_analyzer import (
    ManifoldFishAnalyzer,
    ManifoldFishResult,
    aggregate_atom_embeddings,
    aggregate_from_structure_list,
    aggregate_by_material_id,
    aggregate_with_weights,
)

# Transforms (embedding transformation utilities)
from manifish.core.transforms import (
    EmbeddingTransformer,
    AnchorTransformer,
    PCATransformer,
    TransformResult,
)

# Ensemble Analysis (multi-MLIP ensemble)
from manifish.core.ensemble import (
    EnsembleAnalyzer,
    EnsembleResult,
    ManifoldResult,
)

# Data Filter utilities
from manifish.core.data_filter import (
    DataFilter,
)

__all__ = [
    # Anchor selection
    "select_anchors",
    "DIRECTAnchorSelector",
    "FPSAnchorSelector",
    "Anchor",
    "AnchorSet",
    "load_precomputed_anchors",
    "MP20_ANCHOR_INDICES_100",

    # Platonic projection
    "calculate_platonic_representation",
    "project_to_anchor_space",
    "PlatonicProjector",
    "MultiModelProjector",
    "PlatonicRepresentation",
    "MultiModelRepresentation",

    # Distance-based metrics
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

    # Spatial structure metrics
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

    # Enhanced evaluator
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

    # Structure Evaluator (main interface)
    "StructureEvaluator",
    "ManifoldReference",
    "EvaluationResult",
    "quick_evaluate",
    "evaluate_from_embeddings",

    # Enhanced Fish Analyzer (7-category classifier)
    "EnhancedManifoldFishAnalyzer",
    "EnhancedManifoldFishResult",
    "SpatialMetricsResult",
    "CategorySummary",
    "CATEGORIES",
    "RISK_LEVELS",

    # Base Fish Analyzer
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
