"""
Enhanced Manifold Evaluator with Spatial Structure Metrics.

This module provides an enhanced evaluator that combines traditional
distance-based manifold metrics with spatial structure metrics (LDS, CDS,
SRSS, RMSC) for improved stability prediction.

The key insight is that spatial structure in anchor space is a better
predictor of generation quality than global features alone.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors

from .spatial_metrics import (
    compute_all_spatial_metrics,
    compute_spatial_metrics_batch,
    SpatialMetrics,
    SpatialMetricsConfig,
    create_region_labels_from_crystal_system,
    create_region_labels_from_composition_family,
)
from .manifold import PlatonicRepresentation, calculate_platonic_representation
from .anchors import AnchorSet


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DistanceMetrics:
    """Container for distance-based manifold metrics."""
    manifold_distance: float  # Distance to nearest manifold point
    hull_distance: float  # Distance to convex hull of manifold
    knn_distance: float  # Average distance to k nearest neighbors
    local_density: float  # Local density in manifold space
    isolation_score: float  # How isolated from the manifold

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'manifold_distance': self.manifold_distance,
            'hull_distance': self.hull_distance,
            'knn_distance': self.knn_distance,
            'local_density': self.local_density,
            'isolation_score': self.isolation_score,
        }


@dataclass
class StabilityPrediction:
    """Combined stability prediction from all metrics."""
    stability_basic: float  # Distance-based stability score
    stability_enhanced: float  # Combined with spatial metrics
    stability_confidence: float  # Confidence in prediction

    distance_metrics: DistanceMetrics
    spatial_metrics: SpatialMetrics

    novelty_score: float  # How novel/unusual the structure is
    recommendation: str  # Human-readable recommendation

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'stability_basic': self.stability_basic,
            'stability_enhanced': self.stability_enhanced,
            'stability_confidence': self.stability_confidence,
            'novelty_score': self.novelty_score,
            'recommendation': self.recommendation,
            'distance_metrics': self.distance_metrics.to_dict(),
            'spatial_metrics': self.spatial_metrics.to_dict(),
        }


@dataclass
class EvaluatorConfig:
    """Configuration for the enhanced evaluator."""
    # Distance metric parameters
    k_nearest: int = 10  # Number of neighbors for KNN metrics
    density_radius: float = 0.1  # Radius for density estimation

    # Spatial metric configuration
    spatial_config: SpatialMetricsConfig = field(default_factory=SpatialMetricsConfig)

    # Score combination weights
    distance_weight: float = 0.5  # Weight for distance-based score
    spatial_weight: float = 0.5  # Weight for spatial structure score

    # Thresholds for recommendations
    stability_threshold_high: float = 0.7  # High stability
    stability_threshold_low: float = 0.3  # Low stability
    novelty_threshold: float = 0.8  # High novelty


# =============================================================================
# Distance-Based Metrics
# =============================================================================

def compute_manifold_distance(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
) -> float:
    """
    Compute distance to nearest point in manifold.

    Lower distance = structure is close to known stable materials.

    Args:
        coords: Query coordinates in anchor space
        manifold_coords: Reference manifold coordinates

    Returns:
        Euclidean distance to nearest manifold point
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nbrs.fit(manifold_coords)

    distances, _ = nbrs.kneighbors(coords)
    return np.mean(distances)


def compute_knn_distance(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute average distance to k nearest neighbors.

    Provides a smoother estimate than single nearest neighbor.

    Args:
        coords: Query coordinates
        manifold_coords: Reference manifold coordinates
        k: Number of nearest neighbors

    Returns:
        Average distance to k nearest neighbors
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    k_effective = min(k, len(manifold_coords))
    nbrs = NearestNeighbors(n_neighbors=k_effective, algorithm='auto')
    nbrs.fit(manifold_coords)

    distances, _ = nbrs.kneighbors(coords)
    return np.mean(distances)


def compute_local_density(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    radius: float = 0.1,
) -> float:
    """
    Compute local density around query point.

    Higher density = structure is in a well-populated region of chemical space.

    Args:
        coords: Query coordinates
        manifold_coords: Reference manifold coordinates
        radius: Radius for density estimation

    Returns:
        Number of manifold points within radius (normalized)
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    from scipy.spatial.distance import cdist
    distances = cdist(coords, manifold_coords, metric='euclidean')

    # Count points within radius
    counts = np.sum(distances < radius, axis=1)
    density = np.mean(counts) / len(manifold_coords)

    return density


def compute_hull_distance(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    n_hull_samples: int = 1000,
    random_state: Optional[int] = None,
) -> float:
    """
    Compute approximate distance to convex hull of manifold.

    Uses a sampling approach for efficiency with high-dimensional data.

    Args:
        coords: Query coordinates
        manifold_coords: Reference manifold coordinates
        n_hull_samples: Number of samples for hull approximation
        random_state: Random seed

    Returns:
        Approximate distance to hull (0 if inside)
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    rng = np.random.default_rng(random_state)

    # Sample manifold points for efficiency
    if len(manifold_coords) > n_hull_samples:
        indices = rng.choice(len(manifold_coords), size=n_hull_samples, replace=False)
        hull_coords = manifold_coords[indices]
    else:
        hull_coords = manifold_coords

    # For high-dimensional data, use approximate hull distance
    # Based on projection onto line between query and manifold center
    center = np.mean(hull_coords, axis=0)

    hull_distances = []
    for query in coords:
        # Distance from query to center
        d_to_center = np.linalg.norm(query - center)

        # Max distance from center in manifold
        max_radius = np.max(np.linalg.norm(hull_coords - center, axis=1))

        # If query is within the approximate radius, it's "inside"
        if d_to_center <= max_radius:
            hull_dist = 0.0
        else:
            hull_dist = d_to_center - max_radius

        hull_distances.append(hull_dist)

    return np.mean(hull_distances)


def compute_isolation_score(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute isolation score based on local vs global density.

    Higher isolation = structure is in a sparse region relative to global average.

    Args:
        coords: Query coordinates
        manifold_coords: Reference manifold coordinates
        k: Number of neighbors for local density

    Returns:
        Isolation score (higher = more isolated)
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    k_effective = min(k, len(manifold_coords))
    nbrs = NearestNeighbors(n_neighbors=k_effective, algorithm='auto')
    nbrs.fit(manifold_coords)

    # Local density for query
    query_distances, _ = nbrs.kneighbors(coords)
    local_dist = np.mean(query_distances)

    # Global average distance in manifold
    manifold_distances, _ = nbrs.kneighbors(manifold_coords)
    global_avg_dist = np.mean(manifold_distances)

    # Isolation = ratio of local to global distance
    if global_avg_dist > 1e-10:
        isolation = local_dist / global_avg_dist
    else:
        isolation = 1.0

    return isolation


def compute_all_distance_metrics(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    k_nearest: int = 10,
    density_radius: float = 0.1,
    random_state: Optional[int] = None,
) -> DistanceMetrics:
    """
    Compute all distance-based manifold metrics.

    Args:
        coords: Query coordinates in anchor space
        manifold_coords: Reference manifold coordinates
        k_nearest: Number of neighbors for KNN metrics
        density_radius: Radius for density estimation
        random_state: Random seed

    Returns:
        DistanceMetrics object with all computed metrics
    """
    manifold_distance = compute_manifold_distance(coords, manifold_coords)
    knn_distance = compute_knn_distance(coords, manifold_coords, k=k_nearest)
    local_density = compute_local_density(coords, manifold_coords, radius=density_radius)
    hull_distance = compute_hull_distance(coords, manifold_coords, random_state=random_state)
    isolation = compute_isolation_score(coords, manifold_coords, k=k_nearest)

    return DistanceMetrics(
        manifold_distance=manifold_distance,
        hull_distance=hull_distance,
        knn_distance=knn_distance,
        local_density=local_density,
        isolation_score=isolation,
    )


def distance_to_stability(distance_metrics: DistanceMetrics) -> float:
    """
    Convert distance metrics to stability score.

    Lower distances and higher density = higher stability.

    Args:
        distance_metrics: Computed distance metrics

    Returns:
        Stability score in [0, 1]
    """
    # Normalize each component
    # Distance: use exponential decay
    dist_score = np.exp(-distance_metrics.manifold_distance)

    # KNN distance: use exponential decay
    knn_score = np.exp(-distance_metrics.knn_distance)

    # Density: already in [0, 1] ish, boost it
    density_score = min(1.0, distance_metrics.local_density * 100)

    # Hull: inside hull is good
    hull_score = np.exp(-distance_metrics.hull_distance)

    # Isolation: lower is better
    isolation_score = 1.0 / (1.0 + distance_metrics.isolation_score)

    # Combine with weights
    stability = (
        0.3 * dist_score +
        0.2 * knn_score +
        0.2 * density_score +
        0.15 * hull_score +
        0.15 * isolation_score
    )

    return np.clip(stability, 0.0, 1.0)


# =============================================================================
# Enhanced Evaluator
# =============================================================================

class SpatialManifoldEvaluator:
    """
    Enhanced evaluator combining distance and spatial structure metrics.

    This evaluator uses both traditional distance-based metrics and
    spatial structure metrics to provide improved stability predictions
    for AI-generated crystal structures.

    Example:
        >>> evaluator = SpatialManifoldEvaluator(
        ...     manifold_coords=training_coords,
        ...     anchor_set=anchors,
        ...     region_labels=crystal_systems,
        ... )
        >>> prediction = evaluator.evaluate(new_structure_coords)
        >>> print(f"Stability: {prediction.stability_enhanced:.3f}")
        >>> print(f"Recommendation: {prediction.recommendation}")
    """

    def __init__(
        self,
        manifold_coords: np.ndarray,
        anchor_set: Optional[AnchorSet] = None,
        region_labels: Optional[np.ndarray] = None,
        config: Optional[EvaluatorConfig] = None,
    ):
        """
        Initialize the enhanced evaluator.

        Args:
            manifold_coords: Reference manifold coordinates (n_samples, n_anchors)
            anchor_set: Optional AnchorSet for projection
            region_labels: Optional semantic region labels for SRSS
            config: Evaluator configuration
        """
        self.manifold_coords = np.asarray(manifold_coords)
        self.anchor_set = anchor_set
        self.region_labels = region_labels
        self.config = config or EvaluatorConfig()

        # Pre-compute statistics for normalization
        self._compute_manifold_statistics()

    def _compute_manifold_statistics(self):
        """Pre-compute manifold statistics for normalization."""
        # Mean and std of manifold coordinates
        self.manifold_mean = np.mean(self.manifold_coords, axis=0)
        self.manifold_std = np.std(self.manifold_coords, axis=0)

        # Build KNN model for fast lookup
        self._knn_model = NearestNeighbors(
            n_neighbors=min(self.config.k_nearest + 1, len(self.manifold_coords)),
            algorithm='auto'
        )
        self._knn_model.fit(self.manifold_coords)

        # Compute baseline distances for normalization
        manifold_distances, _ = self._knn_model.kneighbors(self.manifold_coords)
        self.baseline_knn_distance = np.mean(manifold_distances[:, 1:])

    def evaluate(
        self,
        coords: np.ndarray,
        random_state: Optional[int] = None,
    ) -> StabilityPrediction:
        """
        Evaluate a structure using both distance and spatial metrics.

        Args:
            coords: Structure coordinates in anchor space (n_anchors,) or (n_samples, n_anchors)
            random_state: Random seed for reproducibility

        Returns:
            StabilityPrediction with comprehensive assessment
        """
        coords = np.atleast_2d(coords)

        # Compute distance-based metrics
        distance_metrics = compute_all_distance_metrics(
            coords=coords,
            manifold_coords=self.manifold_coords,
            k_nearest=self.config.k_nearest,
            density_radius=self.config.density_radius,
            random_state=random_state,
        )

        # Compute spatial structure metrics
        spatial_metrics = compute_all_spatial_metrics(
            coords=coords,
            manifold_coords=self.manifold_coords,
            region_labels=self.region_labels,
            config=self.config.spatial_config,
            random_state=random_state,
        )

        # Compute basic stability from distance
        stability_basic = distance_to_stability(distance_metrics)

        # Combine with spatial quality for enhanced stability
        spatial_quality_normalized = (spatial_metrics.spatial_quality + 1) / 2  # Map to [0, 1]
        stability_enhanced = (
            self.config.distance_weight * stability_basic +
            self.config.spatial_weight * spatial_quality_normalized
        )
        stability_enhanced = np.clip(stability_enhanced, 0.0, 1.0)

        # Compute confidence based on agreement between metrics
        confidence = self._compute_confidence(
            stability_basic, spatial_quality_normalized, distance_metrics, spatial_metrics
        )

        # Compute novelty score
        novelty_score = self._compute_novelty(distance_metrics, spatial_metrics)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            stability_enhanced, novelty_score, confidence, distance_metrics, spatial_metrics
        )

        return StabilityPrediction(
            stability_basic=stability_basic,
            stability_enhanced=stability_enhanced,
            stability_confidence=confidence,
            distance_metrics=distance_metrics,
            spatial_metrics=spatial_metrics,
            novelty_score=novelty_score,
            recommendation=recommendation,
        )

    def evaluate_batch(
        self,
        coords_batch: np.ndarray,
        random_state: Optional[int] = None,
    ) -> List[StabilityPrediction]:
        """
        Evaluate a batch of structures.

        Args:
            coords_batch: Batch of coordinates (n_samples, n_anchors)
            random_state: Random seed

        Returns:
            List of StabilityPrediction objects
        """
        coords_batch = np.atleast_2d(coords_batch)
        predictions = []

        for i, coords in enumerate(coords_batch):
            seed = random_state + i if random_state else None
            prediction = self.evaluate(coords, random_state=seed)
            predictions.append(prediction)

        return predictions

    def _compute_confidence(
        self,
        stability_basic: float,
        spatial_quality: float,
        distance_metrics: DistanceMetrics,
        spatial_metrics: SpatialMetrics,
    ) -> float:
        """Compute confidence in the stability prediction."""
        # Confidence is higher when:
        # 1. Distance and spatial metrics agree
        agreement = 1.0 - abs(stability_basic - spatial_quality)

        # 2. Local density is high (more data nearby)
        density_conf = min(1.0, distance_metrics.local_density * 50)

        # 3. LDS is reasonable (not extreme)
        lds_conf = np.exp(-abs(spatial_metrics.lds - 1.5) / 2)

        confidence = 0.4 * agreement + 0.3 * density_conf + 0.3 * lds_conf
        return np.clip(confidence, 0.0, 1.0)

    def _compute_novelty(
        self,
        distance_metrics: DistanceMetrics,
        spatial_metrics: SpatialMetrics,
    ) -> float:
        """Compute novelty score for the structure."""
        # Novelty is higher when:
        # 1. Far from manifold
        dist_novelty = 1.0 - np.exp(-distance_metrics.manifold_distance)

        # 2. Low density region
        density_novelty = 1.0 - min(1.0, distance_metrics.local_density * 50)

        # 3. High isolation
        isolation_novelty = distance_metrics.isolation_score / (1.0 + distance_metrics.isolation_score)

        # 4. Low SRSS (doesn't match its chemical family)
        srss_novelty = 1.0 - min(1.0, spatial_metrics.srss / 2)

        novelty = 0.3 * dist_novelty + 0.25 * density_novelty + 0.25 * isolation_novelty + 0.2 * srss_novelty
        return np.clip(novelty, 0.0, 1.0)

    def _generate_recommendation(
        self,
        stability: float,
        novelty: float,
        confidence: float,
        distance_metrics: DistanceMetrics,
        spatial_metrics: SpatialMetrics,
    ) -> str:
        """Generate human-readable recommendation."""
        parts = []

        # Stability assessment
        if stability >= self.config.stability_threshold_high:
            parts.append("HIGH STABILITY: Structure is well within the stable manifold.")
        elif stability >= self.config.stability_threshold_low:
            parts.append("MODERATE STABILITY: Structure shows reasonable stability indicators.")
        else:
            parts.append("LOW STABILITY: Structure may be unstable, recommend DFT verification.")

        # Novelty assessment
        if novelty >= self.config.novelty_threshold:
            parts.append("NOVEL: Structure is in an unexplored region of chemical space.")
            if stability >= self.config.stability_threshold_low:
                parts.append("Consider for experimental synthesis as a discovery candidate.")

        # Confidence note
        if confidence < 0.5:
            parts.append("LOW CONFIDENCE: Limited training data in this region.")

        # Specific warnings based on metrics
        if spatial_metrics.lds < 0.8:
            parts.append("Warning: Poor local chemical coherence (low LDS).")
        if spatial_metrics.srss < 0.7:
            parts.append("Warning: Doesn't match expected patterns for its composition family (low SRSS).")
        if distance_metrics.isolation_score > 2.0:
            parts.append("Warning: Highly isolated in chemical space.")

        return " ".join(parts)

    def get_similar_structures(
        self,
        coords: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar structures in the manifold.

        Args:
            coords: Query coordinates
            k: Number of similar structures to return

        Returns:
            Tuple of (indices, distances) of similar structures
        """
        coords = np.atleast_2d(coords)
        k_effective = min(k, len(self.manifold_coords))

        distances, indices = self._knn_model.kneighbors(coords, n_neighbors=k_effective)
        return indices[0], distances[0]


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_structure(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    region_labels: Optional[np.ndarray] = None,
    config: Optional[EvaluatorConfig] = None,
) -> StabilityPrediction:
    """
    Quick evaluation of a single structure.

    Args:
        coords: Structure coordinates in anchor space
        manifold_coords: Reference manifold coordinates
        region_labels: Optional region labels for SRSS
        config: Evaluator configuration

    Returns:
        StabilityPrediction object
    """
    evaluator = SpatialManifoldEvaluator(
        manifold_coords=manifold_coords,
        region_labels=region_labels,
        config=config,
    )
    return evaluator.evaluate(coords)


def evaluate_structures_batch(
    coords_batch: np.ndarray,
    manifold_coords: np.ndarray,
    region_labels: Optional[np.ndarray] = None,
    config: Optional[EvaluatorConfig] = None,
) -> List[StabilityPrediction]:
    """
    Quick evaluation of a batch of structures.

    Args:
        coords_batch: Batch of structure coordinates
        manifold_coords: Reference manifold coordinates
        region_labels: Optional region labels
        config: Evaluator configuration

    Returns:
        List of StabilityPrediction objects
    """
    evaluator = SpatialManifoldEvaluator(
        manifold_coords=manifold_coords,
        region_labels=region_labels,
        config=config,
    )
    return evaluator.evaluate_batch(coords_batch)


def compare_distance_vs_spatial(
    predictions: List[StabilityPrediction],
) -> Dict[str, float]:
    """
    Compare predictive performance of distance vs spatial metrics.

    Args:
        predictions: List of predictions with known outcomes

    Returns:
        Comparison statistics
    """
    if not predictions:
        return {}

    basic_scores = np.array([p.stability_basic for p in predictions])
    enhanced_scores = np.array([p.stability_enhanced for p in predictions])
    spatial_qualities = np.array([p.spatial_metrics.spatial_quality for p in predictions])

    return {
        'mean_basic': np.mean(basic_scores),
        'mean_enhanced': np.mean(enhanced_scores),
        'std_basic': np.std(basic_scores),
        'std_enhanced': np.std(enhanced_scores),
        'correlation_basic_spatial': np.corrcoef(basic_scores, spatial_qualities)[0, 1],
        'mean_improvement': np.mean(enhanced_scores - basic_scores),
    }
