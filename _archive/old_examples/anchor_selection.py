"""
Anchor Selection for MLIP Embeddings

Select representative anchor embeddings from flat atomic embeddings using
various clustering and sampling strategies.

Supports:
- DIRECT sampling (via maml package)
- K-means clustering
- Farthest point sampling
- Random sampling with stratification

Quality Evaluation (Two Types):
1. Anchor Set Quality: Metrics on anchor embeddings themselves
   - Want diverse anchors: larger pairwise distance, lower silhouette
2. Transformed Quality: Metrics after anchor-based projection
   - Want better clustering: SMALLER pairwise distance (most important), higher silhouette

Key Metrics (interpretation differs by evaluation type):
- Anchor Set: Larger pairwise distance = more diverse anchors (better)
- Anchor Set: Lower silhouette = less clusterable = more diverse (better)
- Transformed: SMALLER pairwise distance = tighter clusters (better)
- Transformed: Higher silhouette = more clusterable (complementary)
- Coverage: Fraction of original points covered by nearby anchors

Usage:
    from anchor_selection import AnchorSelector

    selector = AnchorSelector(method="direct", n_anchors=100)
    result = selector.fit_transform(flat_embeddings)

    anchors = result.anchor_embeddings
    indices = result.anchor_indices

Quality Evaluation:
    from anchor_selection import (
        AnchorQualityEvaluator,
        evaluate_anchor_quality,
        evaluate_transformed_quality,
        compare_anchor_methods,
    )

    evaluator = AnchorQualityEvaluator(metric="cosine")

    # 1. Evaluate anchor set quality (on anchors themselves)
    anchor_metrics = evaluator.evaluate(anchor_embeddings, all_embeddings)
    print(anchor_metrics.summary())

    # 2. Evaluate transformed quality (after anchor projection)
    transformed_metrics = evaluator.evaluate_transformed(
        all_embeddings, anchor_embeddings, transform="cosine"
    )
    print(transformed_metrics.summary())

    # 3. Compare multiple methods (evaluates both)
    comparison = compare_anchor_methods(
        embeddings=flat_embeddings,
        methods=["direct", "kmeans", "fps", "stratified"],
        n_anchors=100,
        transform="cosine",
    )
"""

import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from tqdm import tqdm


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AnchorQualityMetrics:
    """Container for anchor quality evaluation metrics."""
    # Silhouette score: lower = less clusterable = better for diversity
    silhouette_score: float
    # Mean pairwise distance: larger = more diverse anchor set
    mean_pairwise_distance: float
    # Min pairwise distance: larger = better separation
    min_pairwise_distance: float
    # Max pairwise distance
    max_pairwise_distance: float
    # Standard deviation of pairwise distances
    std_pairwise_distance: float
    # Coverage: fraction of original points "covered" by nearby anchors
    coverage_fraction: Optional[float] = None
    # Distance metric used
    metric: str = "cosine"

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Anchor Quality Metrics:",
            f"  Silhouette Score: {self.silhouette_score:.4f} (lower = less clusterable = better diversity)",
            f"  Mean Pairwise Distance: {self.mean_pairwise_distance:.4f} (larger = more diverse)",
            f"  Min Pairwise Distance: {self.min_pairwise_distance:.4f}",
            f"  Max Pairwise Distance: {self.max_pairwise_distance:.4f}",
            f"  Std Pairwise Distance: {self.std_pairwise_distance:.4f}",
        ]
        if self.coverage_fraction is not None:
            lines.append(f"  Coverage Fraction: {self.coverage_fraction:.4f}")
        lines.append(f"  Metric: {self.metric}")
        return "\n".join(lines)


@dataclass
class AnchorResult:
    """Container for anchor selection results."""
    anchor_embeddings: np.ndarray  # (n_anchors, embed_dim)
    anchor_indices: np.ndarray  # (n_anchors,) indices into original flat array
    n_anchors: int
    n_total: int
    embed_dim: int
    method: str

    # Optional: cluster assignments for all points
    cluster_labels: Optional[np.ndarray] = None  # (n_total,)
    cluster_centers: Optional[np.ndarray] = None  # (n_clusters, embed_dim)

    # Optional: PCA features if computed
    pca_features: Optional[np.ndarray] = None

    # Optional: material IDs for anchors (if provided)
    anchor_material_ids: Optional[List[str]] = None

    # Optional: quality metrics
    quality_metrics: Optional[AnchorQualityMetrics] = None

    # Metadata
    params: Dict = field(default_factory=dict)


# =============================================================================
# Base Selector Class
# =============================================================================

class BaseAnchorSelector(ABC):
    """Abstract base class for anchor selection methods."""

    def __init__(self, n_anchors: int = 100):
        self.n_anchors = n_anchors

    @abstractmethod
    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """
        Select anchors from embeddings.

        Args:
            embeddings: Flat embeddings array (n_samples, embed_dim)
            material_ids: Optional material IDs per sample

        Returns:
            AnchorResult with selected anchors
        """
        pass


# =============================================================================
# DIRECT Sampler (via MAML)
# =============================================================================

class DIRECTSelector(BaseAnchorSelector):
    """
    DIRECT sampling using MAML package.

    Uses Birch clustering followed by representative selection from each cluster.

    Args:
        n_anchors: Number of anchors to select (= number of clusters)
        threshold_init: Initial threshold for Birch clustering
        k_per_cluster: Number of samples to select from each cluster
    """

    def __init__(
        self,
        n_anchors: int = 100,
        threshold_init: float = 0.3,
        k_per_cluster: int = 1,
    ):
        super().__init__(n_anchors)
        self.threshold_init = threshold_init
        self.k_per_cluster = k_per_cluster

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Select anchors using DIRECT sampling."""
        try:
            from maml.sampling.direct import (
                BirchClustering,
                DIRECTSampler,
                SelectKFromClusters,
            )
        except ImportError:
            raise ImportError(
                "MAML package required for DIRECT sampling. "
                "Install with: pip install maml"
            )

        embeddings = np.asarray(embeddings)
        n_total, embed_dim = embeddings.shape

        # Create DIRECT sampler
        sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=self.n_anchors, threshold_init=self.threshold_init),
            select_k_from_clusters=SelectKFromClusters(k=self.k_per_cluster),
        )

        # Run selection
        result = sampler.fit_transform(embeddings)

        selected_indices = np.array(result["selected_indexes"])
        anchor_embeddings = embeddings[selected_indices]

        # Get material IDs for anchors if provided
        anchor_mids = None
        if material_ids is not None:
            anchor_mids = [material_ids[i] for i in selected_indices]

        return AnchorResult(
            anchor_embeddings=anchor_embeddings,
            anchor_indices=selected_indices,
            n_anchors=len(selected_indices),
            n_total=n_total,
            embed_dim=embed_dim,
            method="direct",
            pca_features=result.get("PCAfeatures"),
            anchor_material_ids=anchor_mids,
            params={
                "n_anchors": self.n_anchors,
                "threshold_init": self.threshold_init,
                "k_per_cluster": self.k_per_cluster,
            },
        )


# =============================================================================
# K-Means Selector
# =============================================================================

class KMeansSelector(BaseAnchorSelector):
    """
    K-Means clustering based anchor selection.

    Clusters embeddings and selects the point closest to each cluster center.

    Args:
        n_anchors: Number of anchors (= number of clusters)
        random_state: Random seed for reproducibility
        n_init: Number of K-means initializations
    """

    def __init__(
        self,
        n_anchors: int = 100,
        random_state: int = 42,
        n_init: int = 10,
    ):
        super().__init__(n_anchors)
        self.random_state = random_state
        self.n_init = n_init

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Select anchors using K-means clustering."""
        from sklearn.cluster import KMeans

        embeddings = np.asarray(embeddings)
        n_total, embed_dim = embeddings.shape

        # Fit K-means
        kmeans = KMeans(
            n_clusters=self.n_anchors,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_

        # Select point closest to each center
        selected_indices = []
        for i in range(self.n_anchors):
            cluster_mask = labels == i
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_points = embeddings[cluster_indices]
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)

        selected_indices = np.array(selected_indices)
        anchor_embeddings = embeddings[selected_indices]

        anchor_mids = None
        if material_ids is not None:
            anchor_mids = [material_ids[i] for i in selected_indices]

        return AnchorResult(
            anchor_embeddings=anchor_embeddings,
            anchor_indices=selected_indices,
            n_anchors=len(selected_indices),
            n_total=n_total,
            embed_dim=embed_dim,
            method="kmeans",
            cluster_labels=labels,
            cluster_centers=centers,
            anchor_material_ids=anchor_mids,
            params={
                "n_anchors": self.n_anchors,
                "random_state": self.random_state,
                "n_init": self.n_init,
            },
        )


# =============================================================================
# Farthest Point Sampling
# =============================================================================

class FarthestPointSelector(BaseAnchorSelector):
    """
    Farthest Point Sampling (FPS) for anchor selection.

    Iteratively selects points that are farthest from already selected points.
    Provides good coverage of the embedding space.

    Args:
        n_anchors: Number of anchors to select
        random_state: Random seed for initial point selection
        batch_size: Batch size for distance computation (memory efficiency)
    """

    def __init__(
        self,
        n_anchors: int = 100,
        random_state: int = 42,
        batch_size: int = 10000,
    ):
        super().__init__(n_anchors)
        self.random_state = random_state
        self.batch_size = batch_size

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Select anchors using farthest point sampling."""
        embeddings = np.asarray(embeddings)
        n_total, embed_dim = embeddings.shape

        np.random.seed(self.random_state)

        # Initialize with random point
        selected_indices = [np.random.randint(n_total)]
        min_distances = np.full(n_total, np.inf)

        for _ in tqdm(range(self.n_anchors - 1), desc="FPS selection"):
            # Update min distances to selected set
            last_selected = embeddings[selected_indices[-1]]

            # Batch distance computation for memory efficiency
            for start in range(0, n_total, self.batch_size):
                end = min(start + self.batch_size, n_total)
                batch = embeddings[start:end]
                distances = np.linalg.norm(batch - last_selected, axis=1)
                min_distances[start:end] = np.minimum(
                    min_distances[start:end], distances
                )

            # Select point with maximum minimum distance
            # Exclude already selected points
            mask = np.ones(n_total, dtype=bool)
            mask[selected_indices] = False
            masked_distances = np.where(mask, min_distances, -np.inf)
            next_idx = np.argmax(masked_distances)
            selected_indices.append(next_idx)

        selected_indices = np.array(selected_indices)
        anchor_embeddings = embeddings[selected_indices]

        anchor_mids = None
        if material_ids is not None:
            anchor_mids = [material_ids[i] for i in selected_indices]

        return AnchorResult(
            anchor_embeddings=anchor_embeddings,
            anchor_indices=selected_indices,
            n_anchors=len(selected_indices),
            n_total=n_total,
            embed_dim=embed_dim,
            method="fps",
            anchor_material_ids=anchor_mids,
            params={
                "n_anchors": self.n_anchors,
                "random_state": self.random_state,
            },
        )


# =============================================================================
# Random Stratified Selector
# =============================================================================

class StratifiedRandomSelector(BaseAnchorSelector):
    """
    Stratified random sampling based on density estimation.

    Divides the space into density-based strata and samples proportionally
    or uniformly from each stratum.

    Args:
        n_anchors: Number of anchors to select
        n_strata: Number of strata (using K-means for stratification)
        sampling: "uniform" (equal from each stratum) or "proportional"
        random_state: Random seed
    """

    def __init__(
        self,
        n_anchors: int = 100,
        n_strata: int = 20,
        sampling: Literal["uniform", "proportional"] = "uniform",
        random_state: int = 42,
    ):
        super().__init__(n_anchors)
        self.n_strata = n_strata
        self.sampling = sampling
        self.random_state = random_state

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """Select anchors using stratified random sampling."""
        from sklearn.cluster import KMeans

        embeddings = np.asarray(embeddings)
        n_total, embed_dim = embeddings.shape

        np.random.seed(self.random_state)

        # Create strata using K-means
        kmeans = KMeans(
            n_clusters=self.n_strata,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(embeddings)

        # Determine samples per stratum
        if self.sampling == "uniform":
            base_samples = self.n_anchors // self.n_strata
            extra = self.n_anchors % self.n_strata
            samples_per_stratum = [
                base_samples + (1 if i < extra else 0)
                for i in range(self.n_strata)
            ]
        else:  # proportional
            stratum_sizes = np.bincount(labels, minlength=self.n_strata)
            proportions = stratum_sizes / n_total
            samples_per_stratum = np.round(proportions * self.n_anchors).astype(int)
            # Adjust to match exactly n_anchors
            diff = self.n_anchors - samples_per_stratum.sum()
            for i in range(abs(diff)):
                if diff > 0:
                    samples_per_stratum[i % self.n_strata] += 1
                else:
                    idx = np.argmax(samples_per_stratum)
                    samples_per_stratum[idx] -= 1

        # Sample from each stratum
        selected_indices = []
        for stratum_id, n_samples in enumerate(samples_per_stratum):
            stratum_indices = np.where(labels == stratum_id)[0]
            if len(stratum_indices) == 0:
                continue
            n_to_select = min(n_samples, len(stratum_indices))
            selected = np.random.choice(stratum_indices, n_to_select, replace=False)
            selected_indices.extend(selected)

        selected_indices = np.array(selected_indices)
        anchor_embeddings = embeddings[selected_indices]

        anchor_mids = None
        if material_ids is not None:
            anchor_mids = [material_ids[i] for i in selected_indices]

        return AnchorResult(
            anchor_embeddings=anchor_embeddings,
            anchor_indices=selected_indices,
            n_anchors=len(selected_indices),
            n_total=n_total,
            embed_dim=embed_dim,
            method="stratified_random",
            cluster_labels=labels,
            anchor_material_ids=anchor_mids,
            params={
                "n_anchors": self.n_anchors,
                "n_strata": self.n_strata,
                "sampling": self.sampling,
                "random_state": self.random_state,
            },
        )


# =============================================================================
# Unified Interface
# =============================================================================

class AnchorSelector:
    """
    Unified interface for anchor selection methods.

    Args:
        method: Selection method
            - "direct": DIRECT sampling with Birch clustering (requires maml)
            - "kmeans": K-means clustering
            - "fps": Farthest point sampling
            - "stratified": Stratified random sampling
        n_anchors: Number of anchors to select
        **kwargs: Method-specific parameters

    Example:
        >>> selector = AnchorSelector(method="direct", n_anchors=100)
        >>> result = selector.fit_transform(flat_embeddings)
        >>> anchors = result.anchor_embeddings
        >>> indices = result.anchor_indices
    """

    METHODS = {
        "direct": DIRECTSelector,
        "kmeans": KMeansSelector,
        "fps": FarthestPointSelector,
        "farthest_point": FarthestPointSelector,
        "stratified": StratifiedRandomSelector,
        "stratified_random": StratifiedRandomSelector,
    }

    def __init__(
        self,
        method: Literal["direct", "kmeans", "fps", "stratified"] = "direct",
        n_anchors: int = 100,
        **kwargs
    ):
        self.method = method.lower()
        self.n_anchors = n_anchors
        self.kwargs = kwargs

        if self.method not in self.METHODS:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported: {list(self.METHODS.keys())}"
            )

        self.selector = self.METHODS[self.method](n_anchors=n_anchors, **kwargs)

    def fit_transform(
        self,
        embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> AnchorResult:
        """
        Select anchors from embeddings.

        Args:
            embeddings: Flat embeddings (n_samples, embed_dim)
            material_ids: Optional material IDs per sample

        Returns:
            AnchorResult with selected anchors
        """
        return self.selector.fit_transform(embeddings, material_ids)

    def save_anchors(
        self,
        result: AnchorResult,
        output_path: str,
        save_format: str = "pkl",
    ):
        """
        Save anchor selection results.

        Args:
            result: AnchorResult to save
            output_path: Output file path (without extension)
            save_format: "pkl" for pickle, "npy" for numpy, "both"
        """
        output_path = str(output_path)
        if output_path.endswith('.pkl') or output_path.endswith('.npy'):
            output_path = output_path.rsplit('.', 1)[0]

        if save_format in ["pkl", "both"]:
            # Save full result as pickle
            with open(f"{output_path}_anchors.pkl", 'wb') as f:
                pickle.dump(result.anchor_embeddings, f)
            with open(f"{output_path}_indices.pkl", 'wb') as f:
                pickle.dump(result.anchor_indices, f)
            with open(f"{output_path}_result.pkl", 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved anchors to {output_path}_anchors.pkl")

        if save_format in ["npy", "both"]:
            np.save(f"{output_path}_anchors.npy", result.anchor_embeddings)
            np.save(f"{output_path}_indices.npy", result.anchor_indices)
            print(f"Saved anchors to {output_path}_anchors.npy")

    @staticmethod
    def load_anchors(path: str) -> np.ndarray:
        """Load anchor embeddings from file."""
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif path.endswith('.npy'):
            return np.load(path)
        else:
            raise ValueError(f"Unknown file format: {path}")


# =============================================================================
# Anchor Quality Evaluator
# =============================================================================

@dataclass
class TransformedQualityMetrics:
    """Container for transformed embedding quality metrics.

    For transformed embeddings, we want BETTER clustering (opposite of anchor set):
    - Smaller pairwise distance = tighter clusters = better
    - Higher silhouette score = more clusterable = better (complementary)
    """
    # Silhouette score on transformed embeddings (higher = more clusterable = better)
    silhouette_score: float
    # Mean pairwise distance in transformed space (smaller = tighter clusters = better)
    mean_pairwise_distance: float
    # Min pairwise distance in transformed space
    min_pairwise_distance: float
    # Max pairwise distance
    max_pairwise_distance: float
    # Std of pairwise distances
    std_pairwise_distance: float
    # Transformation method used
    transform_method: str = "cosine"
    # Number of samples evaluated
    n_samples: int = 0

    def summary(self) -> str:
        """Return human-readable summary."""
        return "\n".join([
            "Transformed Embedding Quality Metrics:",
            f"  Mean Pairwise Distance: {self.mean_pairwise_distance:.4f} (SMALLER = tighter clusters = better)",
            f"  Silhouette Score: {self.silhouette_score:.4f} (higher = more clusterable, complementary)",
            f"  Min Pairwise Distance: {self.min_pairwise_distance:.4f}",
            f"  Max Pairwise Distance: {self.max_pairwise_distance:.4f}",
            f"  Std Pairwise Distance: {self.std_pairwise_distance:.4f}",
            f"  Transform Method: {self.transform_method}",
            f"  Samples Evaluated: {self.n_samples}",
        ])


class AnchorQualityEvaluator:
    """
    Evaluate anchor quality for maximizing diversity of transformed embeddings.

    Two types of evaluation:
    1. Anchor set quality: Silhouette + pairwise distance on anchor embeddings
    2. Transformed quality: Silhouette + pairwise distance after anchor projection

    Key metrics:
    - Silhouette score: Lower indicates less clusterable (better for diversity)
    - Pairwise distance: Larger indicates more diverse (better)

    Usage:
        >>> evaluator = AnchorQualityEvaluator(metric="cosine")

        # Evaluate anchor set itself
        >>> anchor_metrics = evaluator.evaluate(anchor_embeddings)
        >>> print(anchor_metrics.summary())

        # Evaluate transformed embeddings
        >>> transformed_metrics = evaluator.evaluate_transformed(
        ...     embeddings, anchor_embeddings, transform="cosine"
        ... )
        >>> print(transformed_metrics.summary())

        # Compare multiple methods
        >>> comparison = evaluator.compare_methods(embeddings, methods=["direct", "kmeans", "fps"])
    """

    SUPPORTED_METRICS = ["cosine", "euclidean"]

    def __init__(
        self,
        metric: Literal["cosine", "euclidean"] = "cosine",
        coverage_threshold: float = 0.5,
    ):
        """
        Args:
            metric: Distance metric for evaluation ("cosine" or "euclidean")
            coverage_threshold: Distance threshold for coverage computation
        """
        self.metric = metric
        self.coverage_threshold = coverage_threshold

    def _compute_pairwise_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between embeddings."""
        from scipy.spatial.distance import pdist

        if self.metric == "cosine":
            return pdist(embeddings, metric="cosine")
        else:
            return pdist(embeddings, metric="euclidean")

    def _compute_silhouette_score(
        self,
        anchor_embeddings: np.ndarray,
        n_clusters: int = 5,
    ) -> float:
        """
        Compute silhouette score on anchor embeddings.

        Lower silhouette score indicates anchors are less clusterable,
        which is better for diversity.

        Args:
            anchor_embeddings: Anchor embeddings (n_anchors, embed_dim)
            n_clusters: Number of clusters for silhouette computation

        Returns:
            Silhouette score (-1 to 1)
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_anchors = len(anchor_embeddings)
        if n_anchors <= n_clusters:
            n_clusters = max(2, n_anchors // 2)

        if n_anchors < 3:
            return 0.0

        # Cluster the anchors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(anchor_embeddings)

        # Check if we have at least 2 clusters with samples
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        # Compute silhouette score
        try:
            score = silhouette_score(anchor_embeddings, labels, metric=self.metric)
        except Exception:
            score = 0.0

        return score

    def _compute_coverage(
        self,
        anchor_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
    ) -> float:
        """
        Compute fraction of points covered by nearby anchors.

        Args:
            anchor_embeddings: Anchor embeddings
            all_embeddings: All original embeddings

        Returns:
            Fraction of points within threshold distance of an anchor
        """
        from scipy.spatial.distance import cdist

        if self.metric == "cosine":
            distances = cdist(all_embeddings, anchor_embeddings, metric="cosine")
        else:
            distances = cdist(all_embeddings, anchor_embeddings, metric="euclidean")

        min_distances = distances.min(axis=1)
        covered = (min_distances <= self.coverage_threshold).sum()
        return covered / len(all_embeddings)

    def evaluate(
        self,
        anchor_embeddings: np.ndarray,
        all_embeddings: Optional[np.ndarray] = None,
        n_clusters: int = 5,
    ) -> AnchorQualityMetrics:
        """
        Evaluate anchor set quality (metrics on anchor embeddings themselves).

        Args:
            anchor_embeddings: Selected anchor embeddings (n_anchors, embed_dim)
            all_embeddings: Optional full embeddings for coverage computation
            n_clusters: Number of clusters for silhouette score

        Returns:
            AnchorQualityMetrics with evaluation results
        """
        anchor_embeddings = np.asarray(anchor_embeddings)

        # Compute pairwise distances
        pairwise_dists = self._compute_pairwise_distances(anchor_embeddings)

        # Compute silhouette score
        silhouette = self._compute_silhouette_score(anchor_embeddings, n_clusters)

        # Compute coverage if all embeddings provided
        coverage = None
        if all_embeddings is not None:
            coverage = self._compute_coverage(anchor_embeddings, all_embeddings)

        return AnchorQualityMetrics(
            silhouette_score=silhouette,
            mean_pairwise_distance=float(np.mean(pairwise_dists)),
            min_pairwise_distance=float(np.min(pairwise_dists)),
            max_pairwise_distance=float(np.max(pairwise_dists)),
            std_pairwise_distance=float(np.std(pairwise_dists)),
            coverage_fraction=coverage,
            metric=self.metric,
        )

    def _transform_embeddings(
        self,
        embeddings: np.ndarray,
        anchor_embeddings: np.ndarray,
        transform: str = "cosine",
    ) -> np.ndarray:
        """
        Transform embeddings using anchor-based projection.

        Args:
            embeddings: Original embeddings (n_samples, embed_dim)
            anchor_embeddings: Anchor embeddings (n_anchors, embed_dim)
            transform: Transformation method ("cosine", "euclidean", "rbf")

        Returns:
            Transformed embeddings (n_samples, n_anchors)
        """
        from scipy.spatial.distance import cdist

        if transform == "cosine":
            # Cosine similarity: 1 - cosine_distance
            distances = cdist(embeddings, anchor_embeddings, metric="cosine")
            return 1 - distances  # Convert to similarity
        elif transform == "euclidean":
            return cdist(embeddings, anchor_embeddings, metric="euclidean")
        elif transform == "rbf":
            # RBF kernel: exp(-gamma * d^2)
            distances = cdist(embeddings, anchor_embeddings, metric="euclidean")
            gamma = 1.0 / embeddings.shape[1]  # Default gamma = 1/dim
            return np.exp(-gamma * distances ** 2)
        else:
            raise ValueError(f"Unknown transform: {transform}. Use 'cosine', 'euclidean', or 'rbf'")

    def evaluate_transformed(
        self,
        embeddings: np.ndarray,
        anchor_embeddings: np.ndarray,
        transform: str = "cosine",
        n_clusters: int = 5,
        max_samples: int = 5000,
    ) -> TransformedQualityMetrics:
        """
        Evaluate quality of transformed embeddings after anchor projection.

        This measures how diverse the embeddings become after being projected
        into the anchor-based coordinate space.

        Args:
            embeddings: Original embeddings to transform (n_samples, embed_dim)
            anchor_embeddings: Anchor embeddings (n_anchors, embed_dim)
            transform: Transformation method ("cosine", "euclidean", "rbf")
            n_clusters: Number of clusters for silhouette score
            max_samples: Maximum samples to evaluate (for efficiency)

        Returns:
            TransformedQualityMetrics with evaluation results
        """
        embeddings = np.asarray(embeddings)
        anchor_embeddings = np.asarray(anchor_embeddings)

        # Subsample if too many points
        n_samples = len(embeddings)
        if n_samples > max_samples:
            idx = np.random.choice(n_samples, max_samples, replace=False)
            embeddings_subset = embeddings[idx]
        else:
            embeddings_subset = embeddings

        # Transform embeddings
        transformed = self._transform_embeddings(
            embeddings_subset, anchor_embeddings, transform
        )

        # Compute pairwise distances in transformed space
        from scipy.spatial.distance import pdist
        pairwise_dists = pdist(transformed, metric="euclidean")

        # Compute silhouette score
        silhouette = self._compute_silhouette_score(transformed, n_clusters)

        return TransformedQualityMetrics(
            silhouette_score=silhouette,
            mean_pairwise_distance=float(np.mean(pairwise_dists)),
            min_pairwise_distance=float(np.min(pairwise_dists)),
            max_pairwise_distance=float(np.max(pairwise_dists)),
            std_pairwise_distance=float(np.std(pairwise_dists)),
            transform_method=transform,
            n_samples=len(embeddings_subset),
        )

    def evaluate_result(
        self,
        result: AnchorResult,
        all_embeddings: Optional[np.ndarray] = None,
        n_clusters: int = 5,
    ) -> AnchorResult:
        """
        Evaluate anchor quality and attach metrics to result.

        Args:
            result: AnchorResult from selection
            all_embeddings: Optional full embeddings for coverage
            n_clusters: Number of clusters for silhouette score

        Returns:
            AnchorResult with quality_metrics attached
        """
        metrics = self.evaluate(
            result.anchor_embeddings, all_embeddings, n_clusters
        )
        result.quality_metrics = metrics
        return result

    def compare_methods(
        self,
        embeddings: np.ndarray,
        methods: List[str] = ["direct", "kmeans", "fps", "stratified"],
        n_anchors: int = 100,
        n_clusters: int = 5,
        material_ids: Optional[List[str]] = None,
        transform: str = "cosine",
        verbose: bool = True,
        **method_kwargs,
    ) -> Dict[str, Tuple[AnchorResult, AnchorQualityMetrics, TransformedQualityMetrics]]:
        """
        Compare anchor quality across different selection methods.

        Evaluates both:
        1. Anchor set quality (metrics on anchor embeddings themselves)
        2. Transformed quality (metrics after anchor-based projection)

        Args:
            embeddings: Full embeddings array (n_samples, embed_dim)
            methods: List of methods to compare
            n_anchors: Number of anchors to select
            n_clusters: Number of clusters for silhouette score
            material_ids: Optional material IDs
            transform: Transformation method for transformed quality ("cosine", "euclidean", "rbf")
            verbose: Print comparison summary
            **method_kwargs: Additional method-specific parameters

        Returns:
            Dict mapping method name to (AnchorResult, AnchorQualityMetrics, TransformedQualityMetrics)
        """
        results = {}

        for method in methods:
            try:
                selector = AnchorSelector(
                    method=method,
                    n_anchors=n_anchors,
                    **method_kwargs.get(method, {}),
                )
                result = selector.fit_transform(embeddings, material_ids)

                # Evaluate anchor set quality
                anchor_metrics = self.evaluate(
                    result.anchor_embeddings, embeddings, n_clusters
                )
                result.quality_metrics = anchor_metrics

                # Evaluate transformed quality
                transformed_metrics = self.evaluate_transformed(
                    embeddings, result.anchor_embeddings, transform, n_clusters
                )

                results[method] = (result, anchor_metrics, transformed_metrics)
            except ImportError as e:
                if verbose:
                    print(f"Skipping {method}: {e}")
            except Exception as e:
                if verbose:
                    print(f"Error with {method}: {e}")

        if verbose:
            self._print_comparison(results)

        return results

    def _print_comparison(
        self,
        results: Dict[str, Tuple[AnchorResult, AnchorQualityMetrics, TransformedQualityMetrics]],
    ):
        """Print comparison summary."""
        print("\n" + "=" * 100)
        print("Anchor Selection Method Comparison")
        print("=" * 100)

        # Table 1: Anchor Set Quality
        print("\n[1] Anchor Set Quality (metrics on anchor embeddings themselves)")
        print("-" * 100)
        print(f"{'Method':<15} {'Silhouette':>12} {'Mean Dist':>12} {'Min Dist':>12} {'Coverage':>12}")
        print("-" * 100)

        # Sort by mean pairwise distance (larger is better)
        sorted_methods = sorted(
            results.items(),
            key=lambda x: x[1][1].mean_pairwise_distance,
            reverse=True,
        )

        for method, (result, anchor_metrics, _) in sorted_methods:
            coverage_str = f"{anchor_metrics.coverage_fraction:.4f}" if anchor_metrics.coverage_fraction else "N/A"
            print(
                f"{method:<15} {anchor_metrics.silhouette_score:>12.4f} "
                f"{anchor_metrics.mean_pairwise_distance:>12.4f} "
                f"{anchor_metrics.min_pairwise_distance:>12.4f} "
                f"{coverage_str:>12}"
            )

        # Table 2: Transformed Quality
        print("\n[2] Transformed Embedding Quality (after anchor projection)")
        print("-" * 100)
        print(f"{'Method':<15} {'Mean Dist':>12} {'Silhouette':>12} {'Min Dist':>12} {'Transform':>12}")
        print("-" * 100)

        # Sort by mean pairwise distance in transformed space (SMALLER is better)
        sorted_by_transformed = sorted(
            results.items(),
            key=lambda x: x[1][2].mean_pairwise_distance,
            reverse=False,  # Smaller is better for transformed
        )

        for method, (result, _, transformed_metrics) in sorted_by_transformed:
            print(
                f"{method:<15} {transformed_metrics.mean_pairwise_distance:>12.4f} "
                f"{transformed_metrics.silhouette_score:>12.4f} "
                f"{transformed_metrics.min_pairwise_distance:>12.4f} "
                f"{transformed_metrics.transform_method:>12}"
            )

        print("-" * 100)
        print("\nInterpretation:")
        print("  [Anchor Set]:")
        print("    - Mean Pairwise Distance: LARGER is better (more diverse anchors)")
        print("    - Silhouette Score: LOWER is better (less clusterable = more diverse)")
        print("  [Transformed]:")
        print("    - Mean Pairwise Distance: SMALLER is better (tighter clusters)")
        print("    - Silhouette Score: higher is better (complementary, more clusterable)")

        # Find best methods
        best_anchor_diversity = max(
            sorted_methods,
            key=lambda x: x[1][1].mean_pairwise_distance,
        )
        best_anchor_silhouette = min(
            sorted_methods,
            key=lambda x: x[1][1].silhouette_score,
        )
        # For transformed: smaller distance is better
        best_transformed_clustering = min(
            sorted_by_transformed,
            key=lambda x: x[1][2].mean_pairwise_distance,
        )

        print("\nBest Methods:")
        print(f"  Anchor set - by pairwise distance (larger): {best_anchor_diversity[0]}")
        print(f"  Anchor set - by silhouette (lower): {best_anchor_silhouette[0]}")
        print(f"  Transformed - by pairwise distance (smaller): {best_transformed_clustering[0]}")


def evaluate_anchor_quality(
    anchor_embeddings: np.ndarray,
    all_embeddings: Optional[np.ndarray] = None,
    metric: str = "cosine",
    n_clusters: int = 5,
) -> AnchorQualityMetrics:
    """
    Convenience function to evaluate anchor set quality.

    Args:
        anchor_embeddings: Selected anchor embeddings
        all_embeddings: Optional full embeddings for coverage
        metric: Distance metric ("cosine" or "euclidean")
        n_clusters: Number of clusters for silhouette score

    Returns:
        AnchorQualityMetrics with evaluation results
    """
    evaluator = AnchorQualityEvaluator(metric=metric)
    return evaluator.evaluate(anchor_embeddings, all_embeddings, n_clusters)


def evaluate_transformed_quality(
    embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    transform: str = "cosine",
    n_clusters: int = 5,
    max_samples: int = 5000,
) -> TransformedQualityMetrics:
    """
    Convenience function to evaluate transformed embedding quality.

    Args:
        embeddings: Original embeddings to transform
        anchor_embeddings: Anchor embeddings for projection
        transform: Transformation method ("cosine", "euclidean", "rbf")
        n_clusters: Number of clusters for silhouette score
        max_samples: Maximum samples to evaluate

    Returns:
        TransformedQualityMetrics with evaluation results
    """
    evaluator = AnchorQualityEvaluator()
    return evaluator.evaluate_transformed(
        embeddings, anchor_embeddings, transform, n_clusters, max_samples
    )


def compare_anchor_methods(
    embeddings: np.ndarray,
    methods: List[str] = ["kmeans", "fps", "stratified"],
    n_anchors: int = 100,
    metric: str = "cosine",
    transform: str = "cosine",
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Tuple[AnchorResult, AnchorQualityMetrics, TransformedQualityMetrics]]:
    """
    Convenience function to compare anchor selection methods.

    Evaluates both anchor set quality and transformed embedding quality.

    Args:
        embeddings: Full embeddings array
        methods: List of methods to compare (default excludes "direct" which requires maml)
        n_anchors: Number of anchors
        metric: Distance metric for anchor quality evaluation
        transform: Transformation method for transformed quality ("cosine", "euclidean", "rbf")
        verbose: Print comparison summary
        **kwargs: Additional method-specific parameters

    Returns:
        Dict mapping method name to (AnchorResult, AnchorQualityMetrics, TransformedQualityMetrics)
    """
    evaluator = AnchorQualityEvaluator(metric=metric)
    return evaluator.compare_methods(
        embeddings, methods, n_anchors, transform=transform, verbose=verbose, **kwargs
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def select_anchors_direct(
    embeddings: np.ndarray,
    n_anchors: int = 100,
    threshold_init: float = 0.3,
    k_per_cluster: int = 1,
    material_ids: Optional[List[str]] = None,
) -> AnchorResult:
    """
    Select anchors using DIRECT sampling.

    This is the method you're currently using with MAML.

    Args:
        embeddings: Flat embeddings (n_samples, embed_dim)
        n_anchors: Number of anchors (= number of Birch clusters)
        threshold_init: Initial threshold for Birch clustering
        k_per_cluster: Samples per cluster
        material_ids: Optional material IDs

    Returns:
        AnchorResult with selected anchors
    """
    selector = AnchorSelector(
        method="direct",
        n_anchors=n_anchors,
        threshold_init=threshold_init,
        k_per_cluster=k_per_cluster,
    )
    return selector.fit_transform(embeddings, material_ids)


def select_anchors_kmeans(
    embeddings: np.ndarray,
    n_anchors: int = 100,
    random_state: int = 42,
    material_ids: Optional[List[str]] = None,
) -> AnchorResult:
    """Select anchors using K-means clustering."""
    selector = AnchorSelector(
        method="kmeans",
        n_anchors=n_anchors,
        random_state=random_state,
    )
    return selector.fit_transform(embeddings, material_ids)


def select_anchors_fps(
    embeddings: np.ndarray,
    n_anchors: int = 100,
    random_state: int = 42,
    material_ids: Optional[List[str]] = None,
) -> AnchorResult:
    """Select anchors using farthest point sampling."""
    selector = AnchorSelector(
        method="fps",
        n_anchors=n_anchors,
        random_state=random_state,
    )
    return selector.fit_transform(embeddings, material_ids)


# =============================================================================
# Visualization
# =============================================================================

def plot_anchor_selection(
    embeddings: np.ndarray,
    result: AnchorResult,
    max_points: int = 5000,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Visualize anchor selection results using PCA projection.

    Args:
        embeddings: Original embeddings
        result: AnchorResult from selection
        max_points: Maximum points to plot (for performance)
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # PCA projection
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        anchors_2d = pca.transform(result.anchor_embeddings)
    else:
        embeddings_2d = embeddings
        anchors_2d = result.anchor_embeddings

    # Subsample for visualization
    n_total = len(embeddings_2d)
    if n_total > max_points:
        idx = np.random.choice(n_total, max_points, replace=False)
        embeddings_2d_sub = embeddings_2d[idx]
    else:
        embeddings_2d_sub = embeddings_2d

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: All points with anchors highlighted
    ax1 = axes[0]
    ax1.scatter(embeddings_2d_sub[:, 0], embeddings_2d_sub[:, 1],
               c='lightgray', alpha=0.3, s=5, label='All points')
    ax1.scatter(anchors_2d[:, 0], anchors_2d[:, 1],
               c='red', alpha=0.8, s=50, marker='*', label='Anchors')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title(f'Anchor Selection ({result.method})\n'
                 f'{result.n_anchors} anchors from {result.n_total} points')
    ax1.legend()

    # Plot 2: Cluster distribution if available
    ax2 = axes[1]
    if result.cluster_labels is not None:
        scatter = ax2.scatter(embeddings_2d_sub[:, 0], embeddings_2d_sub[:, 1],
                             c=result.cluster_labels[idx] if n_total > max_points
                             else result.cluster_labels,
                             cmap='tab20', alpha=0.5, s=5)
        ax2.scatter(anchors_2d[:, 0], anchors_2d[:, 1],
                   c='red', alpha=0.8, s=50, marker='*', edgecolors='black')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Cluster Assignments')
        plt.colorbar(scatter, ax=ax2, label='Cluster')
    else:
        # Show anchor density
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(embeddings_2d.T)
            density = kde(embeddings_2d_sub.T)
            scatter = ax2.scatter(embeddings_2d_sub[:, 0], embeddings_2d_sub[:, 1],
                                 c=density, cmap='viridis', alpha=0.5, s=5)
            ax2.scatter(anchors_2d[:, 0], anchors_2d[:, 1],
                       c='red', alpha=0.8, s=50, marker='*', edgecolors='white')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            ax2.set_title('Density with Anchors')
            plt.colorbar(scatter, ax=ax2, label='Density')
        except Exception:
            ax2.text(0.5, 0.5, 'Density estimation failed',
                    transform=ax2.transAxes, ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example demonstrating anchor selection methods."""
    print("=" * 80)
    print("Anchor Selection Examples")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10000
    embed_dim = 128

    # Simulated atom embeddings (mixture of clusters)
    centers = np.random.randn(5, embed_dim) * 2
    embeddings = np.vstack([
        centers[i] + np.random.randn(n_samples // 5, embed_dim) * 0.5
        for i in range(5)
    ])
    np.random.shuffle(embeddings)

    material_ids = [f"mat_{i // 20}" for i in range(n_samples)]

    print(f"\nInput: {embeddings.shape[0]} atoms, {embed_dim}-dim embeddings")

    # =========================================================================
    # Method 1: DIRECT (your current approach)
    # =========================================================================
    print("\n--- Method 1: DIRECT Sampling ---")
    print("""
    # Using MAML's DIRECT sampler (your current approach)
    result = select_anchors_direct(
        embeddings=flat_embeddings,
        n_anchors=100,
        threshold_init=0.3,
        k_per_cluster=1,
    )
    """)

    # =========================================================================
    # Method 2: K-means
    # =========================================================================
    print("\n--- Method 2: K-Means ---")
    result_kmeans = select_anchors_kmeans(embeddings, n_anchors=100)
    print(f"Selected {result_kmeans.n_anchors} anchors using K-means")
    print(f"Anchor shape: {result_kmeans.anchor_embeddings.shape}")

    # =========================================================================
    # Method 3: Farthest Point Sampling
    # =========================================================================
    print("\n--- Method 3: Farthest Point Sampling ---")
    result_fps = select_anchors_fps(embeddings, n_anchors=100)
    print(f"Selected {result_fps.n_anchors} anchors using FPS")

    # =========================================================================
    # Anchor Quality Evaluation (Two Types)
    # =========================================================================
    print("\n--- Anchor Quality Evaluation ---")

    evaluator = AnchorQualityEvaluator(metric="cosine")

    # Type 1: Anchor Set Quality (on anchor embeddings themselves)
    print("\n[Type 1] Anchor Set Quality (metrics on anchors themselves):")
    anchor_metrics_kmeans = evaluator.evaluate(result_kmeans.anchor_embeddings, embeddings)
    print("\nK-means:")
    print(anchor_metrics_kmeans.summary())

    anchor_metrics_fps = evaluator.evaluate(result_fps.anchor_embeddings, embeddings)
    print("\nFPS:")
    print(anchor_metrics_fps.summary())

    # Type 2: Transformed Quality (after anchor projection)
    print("\n[Type 2] Transformed Embedding Quality (after cosine projection):")
    transformed_kmeans = evaluator.evaluate_transformed(
        embeddings, result_kmeans.anchor_embeddings, transform="cosine"
    )
    print("\nK-means transformed:")
    print(transformed_kmeans.summary())

    transformed_fps = evaluator.evaluate_transformed(
        embeddings, result_fps.anchor_embeddings, transform="cosine"
    )
    print("\nFPS transformed:")
    print(transformed_fps.summary())

    # Compare multiple methods (evaluates both types)
    print("\n--- Comparing All Methods ---")
    comparison = compare_anchor_methods(
        embeddings,
        methods=["kmeans", "fps", "stratified"],
        n_anchors=100,
        metric="cosine",
        transform="cosine",
        verbose=True,
    )

    print("""
    # Usage examples for anchor quality evaluation:

    from anchor_selection import (
        AnchorQualityEvaluator,
        evaluate_anchor_quality,
        evaluate_transformed_quality,
        compare_anchor_methods,
    )

    evaluator = AnchorQualityEvaluator(metric="cosine")

    # 1. Evaluate anchor set quality (on anchors themselves)
    anchor_metrics = evaluate_anchor_quality(
        anchor_embeddings=result.anchor_embeddings,
        all_embeddings=full_embeddings,  # optional, for coverage
        metric="cosine",
    )
    print(anchor_metrics.summary())

    # 2. Evaluate transformed quality (after anchor projection)
    transformed_metrics = evaluate_transformed_quality(
        embeddings=full_embeddings,
        anchor_embeddings=result.anchor_embeddings,
        transform="cosine",  # or "euclidean", "rbf"
    )
    print(transformed_metrics.summary())

    # 3. Compare different anchor selection methods (evaluates both)
    comparison = compare_anchor_methods(
        embeddings=flat_embeddings,
        methods=["direct", "kmeans", "fps", "stratified"],
        n_anchors=100,
        transform="cosine",
    )

    # Access results: (AnchorResult, AnchorQualityMetrics, TransformedQualityMetrics)
    for method, (result, anchor_metrics, transformed_metrics) in comparison.items():
        print(f"{method}: anchor_sil={anchor_metrics.silhouette_score:.4f}, "
              f"transformed_sil={transformed_metrics.silhouette_score:.4f}")
    """)

    # =========================================================================
    # Unified Interface
    # =========================================================================
    print("\n--- Unified Interface ---")
    print("""
    from anchor_selection import AnchorSelector

    # DIRECT (requires maml)
    selector = AnchorSelector(method="direct", n_anchors=100)

    # K-means
    selector = AnchorSelector(method="kmeans", n_anchors=100)

    # Farthest point sampling
    selector = AnchorSelector(method="fps", n_anchors=100)

    # Stratified random
    selector = AnchorSelector(method="stratified", n_anchors=100, n_strata=20)

    result = selector.fit_transform(flat_embeddings, material_ids)
    selector.save_anchors(result, "output/anchors")
    """)

    # =========================================================================
    # Full Pipeline Example
    # =========================================================================
    print("\n--- Full Pipeline with Embedding Extraction ---")
    print("""
    from mlip_embedding_extractor import MLIPEmbeddingExtractor
    from anchor_selection import AnchorSelector, AnchorQualityEvaluator

    # 1. Extract embeddings
    extractor = MLIPEmbeddingExtractor("mace", "/path/to/model")
    results = extractor.extract_from_json("structures.json")

    # 2. Get flat embeddings
    flat_emb = results.flat_embeddings
    flat_ids = results.flat_material_ids

    # 3. Select anchors
    selector = AnchorSelector(method="direct", n_anchors=100)
    anchor_result = selector.fit_transform(flat_emb, flat_ids)

    # 4. Evaluate anchor quality
    evaluator = AnchorQualityEvaluator(metric="cosine")
    anchor_result = evaluator.evaluate_result(anchor_result, flat_emb)
    print(anchor_result.quality_metrics.summary())

    # 5. Save anchors
    selector.save_anchors(anchor_result, "mace_anchors_100")

    # Access results
    anchors = anchor_result.anchor_embeddings  # (100, embed_dim)
    indices = anchor_result.anchor_indices     # indices into flat_emb
    quality = anchor_result.quality_metrics    # AnchorQualityMetrics
    """)

    return result_kmeans, result_fps, comparison


if __name__ == "__main__":
    example_usage()
