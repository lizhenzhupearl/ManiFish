"""
Spatial structure metrics for manifold-based stability prediction.

Inspired by spatial structure metrics from image generation research,
these metrics measure the spatial organization of structures in anchor space.
The key insight is that spatial structure (local relationships) is a better
predictor of generation quality than global features alone.

Metrics implemented:
1. Local vs. Distant Similarity (LDS): Contrast between local and distant similarities
2. Correlation Decay Slope (CDS): How quickly similarity decays with distance
3. Semantic-Region Self-Similarity (SRSS): Within-region vs cross-region coherence
4. RMS Spatial Contrast (RMSC): Diversity in local anchor space representations

Reference:
    These metrics are adapted from vision representation alignment research
    for use in materials science manifold analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import warnings


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpatialMetrics:
    """Container for all spatial structure metrics."""
    lds: float  # Local vs Distant Similarity
    cds: float  # Correlation Decay Slope
    srss: float  # Semantic-Region Self-Similarity
    rmsc: float  # RMS Spatial Contrast
    spatial_quality: float  # Combined quality score

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'lds': self.lds,
            'cds': self.cds,
            'srss': self.srss,
            'rmsc': self.rmsc,
            'spatial_quality': self.spatial_quality,
        }


@dataclass
class SpatialMetricsConfig:
    """Configuration for spatial metrics computation."""
    # LDS parameters
    lds_k_local: int = 10  # Number of local neighbors
    lds_n_distant: int = 100  # Number of distant samples
    lds_distance_threshold: float = 0.5  # Threshold for "distant"

    # CDS parameters
    cds_distance_bins: List[float] = None  # Distance bins for decay analysis
    cds_n_samples_per_bin: int = 50  # Samples per distance bin

    # SRSS parameters
    srss_n_same_region: int = 50  # Samples from same region
    srss_n_different_region: int = 50  # Samples from different regions

    # RMSC parameters
    rmsc_k_neighbors: int = 20  # Neighbors for local variance

    # Quality score weights
    quality_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.cds_distance_bins is None:
            self.cds_distance_bins = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        if self.quality_weights is None:
            self.quality_weights = {
                'lds': 0.3,
                'cds': 0.3,
                'srss': 0.2,
                'rmsc': 0.2,
            }


# =============================================================================
# Core Similarity Functions
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def cosine_similarity_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between rows of X and Y."""
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    return X_norm @ Y_norm.T


def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Convert euclidean distance to similarity."""
    dist = np.linalg.norm(a - b)
    return 1.0 / (1.0 + dist)


# =============================================================================
# Local vs. Distant Similarity (LDS)
# =============================================================================

def compute_lds(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    k_local: int = 10,
    n_distant: int = 100,
    distance_threshold: float = 0.5,
    similarity_fn: str = 'cosine',
    random_state: Optional[int] = None,
) -> float:
    """
    Compute Local vs Distant Similarity (LDS) metric.

    LDS measures the contrast between the self-similarity of nearby structures
    and distant structures in anchor space. Higher values indicate stronger
    spatial organization - similar chemistries cluster together.

    Physical meaning for materials:
    - High LDS: Structure has clear "chemical neighborhood" in anchor space
    - Low LDS: Structure is randomly positioned, no clear chemical relationships

    Args:
        coords: Query structure coordinates in anchor space (n_anchors,) or (n_samples, n_anchors)
        manifold_coords: Reference manifold coordinates (n_manifold, n_anchors)
        k_local: Number of local neighbors to consider
        n_distant: Number of distant samples for comparison
        distance_threshold: Minimum distance to be considered "distant" (normalized)
        similarity_fn: 'cosine' or 'euclidean'
        random_state: Random seed for reproducibility

    Returns:
        LDS score (higher = better spatial organization)

    Example:
        >>> coords = project_to_anchor_space(structure, anchors)
        >>> lds = compute_lds(coords, manifold_coords, k_local=10)
        >>> print(f"LDS score: {lds:.4f}")
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    rng = np.random.default_rng(random_state)

    # Build KNN model on manifold
    k_effective = min(k_local + 1, len(manifold_coords) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_effective, algorithm='auto')
    nbrs.fit(manifold_coords)

    lds_scores = []

    for query in coords:
        # Find k nearest neighbors
        distances, indices = nbrs.kneighbors(query.reshape(1, -1))
        local_indices = indices[0, 1:]  # Exclude self if present
        local_coords = manifold_coords[local_indices]

        # Compute local similarity
        if similarity_fn == 'cosine':
            local_sims = [cosine_similarity(query, lc) for lc in local_coords]
        else:
            local_sims = [euclidean_similarity(query, lc) for lc in local_coords]
        local_sim = np.mean(local_sims) if local_sims else 0.0

        # Find distant samples (beyond threshold)
        all_distances = cdist(query.reshape(1, -1), manifold_coords, metric='euclidean')[0]
        max_dist = np.max(all_distances)
        if max_dist > 0:
            normalized_distances = all_distances / max_dist
            distant_mask = normalized_distances > distance_threshold
            distant_indices = np.where(distant_mask)[0]
        else:
            distant_indices = np.array([])

        if len(distant_indices) == 0:
            # Fall back to random sampling from furthest half
            sorted_indices = np.argsort(all_distances)
            distant_indices = sorted_indices[len(sorted_indices)//2:]

        # Sample distant points
        n_sample = min(n_distant, len(distant_indices))
        sampled_distant = rng.choice(distant_indices, size=n_sample, replace=False)
        distant_coords = manifold_coords[sampled_distant]

        # Compute distant similarity
        if similarity_fn == 'cosine':
            distant_sims = [cosine_similarity(query, dc) for dc in distant_coords]
        else:
            distant_sims = [euclidean_similarity(query, dc) for dc in distant_coords]
        distant_sim = np.mean(distant_sims) if distant_sims else 0.0

        # LDS = ratio of local to distant similarity
        if distant_sim > 1e-10:
            lds = local_sim / distant_sim
        else:
            lds = local_sim * 10  # High value when distant is near zero

        lds_scores.append(lds)

    return np.mean(lds_scores)


# =============================================================================
# Correlation Decay Slope (CDS)
# =============================================================================

def compute_cds(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    distance_bins: Optional[List[float]] = None,
    n_samples_per_bin: int = 50,
    similarity_fn: str = 'cosine',
    random_state: Optional[int] = None,
) -> float:
    """
    Compute Correlation Decay Slope (CDS) metric.

    CDS quantifies how quickly similarity between structures decays with
    increasing spatial distance in anchor space. Larger values suggest
    better spatial organization (sharper locality).

    Physical meaning for materials:
    - Large CDS: Structure has well-defined "chemical locality" with sharp transitions
    - Small CDS: Blurred chemical boundaries, weak spatial structure

    Args:
        coords: Query structure coordinates in anchor space
        manifold_coords: Reference manifold coordinates
        distance_bins: Distance bins for decay analysis (default: [0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        n_samples_per_bin: Number of samples per distance bin
        similarity_fn: 'cosine' or 'euclidean'
        random_state: Random seed

    Returns:
        CDS score (slope of exponential decay, larger = better organization)

    Example:
        >>> cds = compute_cds(coords, manifold_coords)
        >>> print(f"Correlation Decay Slope: {cds:.4f}")
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    if distance_bins is None:
        distance_bins = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    rng = np.random.default_rng(random_state)

    all_distances = []
    all_similarities = []

    for query in coords:
        # Compute distances to all manifold points
        dists = cdist(query.reshape(1, -1), manifold_coords, metric='euclidean')[0]

        for i, r in enumerate(distance_bins):
            # Define bin boundaries
            r_low = distance_bins[i-1] if i > 0 else 0
            r_high = r

            # Find points in this distance bin
            mask = (dists > r_low) & (dists <= r_high)
            bin_indices = np.where(mask)[0]

            if len(bin_indices) == 0:
                continue

            # Sample from bin
            n_sample = min(n_samples_per_bin, len(bin_indices))
            sampled = rng.choice(bin_indices, size=n_sample, replace=False)

            # Compute similarities
            for idx in sampled:
                if similarity_fn == 'cosine':
                    sim = cosine_similarity(query, manifold_coords[idx])
                else:
                    sim = euclidean_similarity(query, manifold_coords[idx])

                all_distances.append(dists[idx])
                all_similarities.append(sim)

    if len(all_distances) < 5:
        return 0.0

    all_distances = np.array(all_distances)
    all_similarities = np.array(all_similarities)

    # Filter valid points for log fitting
    valid = (all_similarities > 1e-10) & (all_distances > 1e-10)
    distances_valid = all_distances[valid]
    similarities_valid = all_similarities[valid]

    if len(distances_valid) < 3:
        return 0.0

    # Fit exponential decay: sim = exp(-slope * distance)
    # ln(sim) = -slope * distance
    try:
        log_sims = np.log(similarities_valid + 1e-10)
        reg = LinearRegression(fit_intercept=True)
        reg.fit(distances_valid.reshape(-1, 1), log_sims)
        slope = -reg.coef_[0]  # Negative because we want positive slope for decay
    except Exception:
        slope = 0.0

    return max(slope, 0.0)  # CDS should be non-negative


# =============================================================================
# Semantic-Region Self-Similarity (SRSS)
# =============================================================================

def compute_srss(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    region_labels: np.ndarray,
    query_region: Optional[int] = None,
    n_same_region: int = 50,
    n_different_region: int = 50,
    similarity_fn: str = 'cosine',
    random_state: Optional[int] = None,
) -> float:
    """
    Compute Semantic-Region Self-Similarity (SRSS) metric.

    SRSS compares similarity between structures within the same semantic
    region (e.g., same composition family, crystal system) vs structures
    in different regions. Higher values indicate better chemical coherence.

    Physical meaning for materials:
    - High SRSS: Structure fits well with its "chemical family"
    - Low SRSS: Doesn't match expected chemical patterns (may be novel OR unstable)

    Args:
        coords: Query structure coordinates in anchor space
        manifold_coords: Reference manifold coordinates
        region_labels: Integer labels for semantic regions (e.g., crystal system)
        query_region: Region label for the query (if None, inferred from nearest neighbor)
        n_same_region: Number of samples from same region
        n_different_region: Number of samples from different regions
        similarity_fn: 'cosine' or 'euclidean'
        random_state: Random seed

    Returns:
        SRSS score (higher = better coherence within region)

    Example:
        >>> # Define regions by crystal system
        >>> region_labels = get_crystal_system_labels(manifold_structures)
        >>> srss = compute_srss(coords, manifold_coords, region_labels)
        >>> print(f"SRSS score: {srss:.4f}")
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)
    region_labels = np.asarray(region_labels)

    rng = np.random.default_rng(random_state)

    srss_scores = []

    for query in coords:
        # Infer query region if not provided
        if query_region is None:
            # Use nearest neighbor's region
            dists = cdist(query.reshape(1, -1), manifold_coords, metric='euclidean')[0]
            nearest_idx = np.argmin(dists)
            current_region = region_labels[nearest_idx]
        else:
            current_region = query_region

        # Find same-region and different-region indices
        same_region_mask = region_labels == current_region
        diff_region_mask = ~same_region_mask

        same_indices = np.where(same_region_mask)[0]
        diff_indices = np.where(diff_region_mask)[0]

        # Sample from each group
        n_same = min(n_same_region, len(same_indices))
        n_diff = min(n_different_region, len(diff_indices))

        if n_same == 0 or n_diff == 0:
            continue

        sampled_same = rng.choice(same_indices, size=n_same, replace=False)
        sampled_diff = rng.choice(diff_indices, size=n_diff, replace=False)

        # Compute similarities
        if similarity_fn == 'cosine':
            same_sims = [cosine_similarity(query, manifold_coords[i]) for i in sampled_same]
            diff_sims = [cosine_similarity(query, manifold_coords[i]) for i in sampled_diff]
        else:
            same_sims = [euclidean_similarity(query, manifold_coords[i]) for i in sampled_same]
            diff_sims = [euclidean_similarity(query, manifold_coords[i]) for i in sampled_diff]

        sim_same = np.mean(same_sims)
        sim_diff = np.mean(diff_sims)

        # SRSS = ratio of within-region to cross-region similarity
        if sim_diff > 1e-10:
            srss = sim_same / sim_diff
        else:
            srss = sim_same * 10

        srss_scores.append(srss)

    return np.mean(srss_scores) if srss_scores else 1.0


def compute_srss_from_composition(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    query_elements: set,
    manifold_elements: List[set],
    element_overlap_threshold: float = 0.5,
    n_same_region: int = 50,
    n_different_region: int = 50,
    similarity_fn: str = 'cosine',
    random_state: Optional[int] = None,
) -> float:
    """
    Compute SRSS using composition-based region definition.

    Structures are considered in the same "region" if they share
    a significant fraction of their elements.

    Args:
        coords: Query structure coordinates
        manifold_coords: Reference manifold coordinates
        query_elements: Set of element symbols in query structure
        manifold_elements: List of element sets for manifold structures
        element_overlap_threshold: Jaccard similarity threshold for "same region"
        n_same_region: Number of same-region samples
        n_different_region: Number of different-region samples
        similarity_fn: Similarity function type
        random_state: Random seed

    Returns:
        SRSS score based on composition similarity
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    # Compute Jaccard similarity for region assignment
    def jaccard(s1, s2):
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0

    # Create binary region labels based on composition overlap
    region_labels = np.array([
        1 if jaccard(query_elements, elem_set) >= element_overlap_threshold else 0
        for elem_set in manifold_elements
    ])

    return compute_srss(
        coords=coords,
        manifold_coords=manifold_coords,
        region_labels=region_labels,
        query_region=1,  # Query is always in its own region
        n_same_region=n_same_region,
        n_different_region=n_different_region,
        similarity_fn=similarity_fn,
        random_state=random_state,
    )


# =============================================================================
# RMS Spatial Contrast (RMSC)
# =============================================================================

def compute_rmsc(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    k_neighbors: int = 20,
) -> float:
    """
    Compute RMS Spatial Contrast (RMSC) metric.

    RMSC measures the diversity in local anchor space representations.
    Higher values indicate preserved spatial structure, while lower values
    suggest loss of spatial organization.

    Physical meaning for materials:
    - High RMSC: Structure has rich local diversity, complex but organized pattern
    - Low RMSC: Overly uniform, loss of structural information
    - Very high RMSC: Too much variance, may indicate instability

    Args:
        coords: Query structure coordinates in anchor space
        manifold_coords: Reference manifold coordinates
        k_neighbors: Number of neighbors for local variance estimation

    Returns:
        RMSC score (RMS of local variances across anchor dimensions)

    Example:
        >>> rmsc = compute_rmsc(coords, manifold_coords, k_neighbors=20)
        >>> print(f"RMSC score: {rmsc:.4f}")
    """
    coords = np.atleast_2d(coords)
    manifold_coords = np.asarray(manifold_coords)

    # Build KNN model
    k_effective = min(k_neighbors, len(manifold_coords) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_effective, algorithm='auto')
    nbrs.fit(manifold_coords)

    rmsc_scores = []

    for query in coords:
        # Find k nearest neighbors
        _, indices = nbrs.kneighbors(query.reshape(1, -1))
        neighbor_indices = indices[0]

        # Get neighbor coordinates
        neighbors = manifold_coords[neighbor_indices]

        # Compute variance in each anchor dimension
        variances = np.var(neighbors, axis=0)

        # RMS of variances
        rmsc = np.sqrt(np.mean(variances))
        rmsc_scores.append(rmsc)

    return np.mean(rmsc_scores)


# =============================================================================
# Combined Spatial Quality Score
# =============================================================================

def compute_spatial_quality(
    lds: float,
    cds: float,
    srss: float,
    rmsc: float,
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> float:
    """
    Compute combined spatial quality score from individual metrics.

    Args:
        lds: Local vs Distant Similarity score
        cds: Correlation Decay Slope score
        srss: Semantic-Region Self-Similarity score
        rmsc: RMS Spatial Contrast score
        weights: Custom weights for each metric
        normalize: Whether to normalize scores before combining

    Returns:
        Combined spatial quality score
    """
    if weights is None:
        weights = {'lds': 0.3, 'cds': 0.3, 'srss': 0.2, 'rmsc': 0.2}

    scores = {'lds': lds, 'cds': cds, 'srss': srss, 'rmsc': rmsc}

    if normalize:
        # Normalize each score to [0, 1] range using sigmoid-like transform
        def normalize_score(x, scale=1.0):
            # For ratio-based metrics (LDS, SRSS), values around 1 are neutral
            # For CDS and RMSC, larger is generally better up to a point
            return 2.0 / (1.0 + np.exp(-x / scale)) - 1.0

        # Apply normalization with metric-specific scales
        scales = {'lds': 2.0, 'cds': 1.0, 'srss': 2.0, 'rmsc': 0.5}
        scores = {k: normalize_score(v, scales[k]) for k, v in scores.items()}

    # Weighted sum
    quality = sum(weights[k] * scores[k] for k in weights)

    return quality


# =============================================================================
# High-Level Interface
# =============================================================================

def compute_all_spatial_metrics(
    coords: np.ndarray,
    manifold_coords: np.ndarray,
    region_labels: Optional[np.ndarray] = None,
    config: Optional[SpatialMetricsConfig] = None,
    random_state: Optional[int] = None,
) -> SpatialMetrics:
    """
    Compute all spatial structure metrics for a query structure.

    This is the main entry point for spatial metric computation.

    Args:
        coords: Query structure coordinates in anchor space
        manifold_coords: Reference manifold coordinates
        region_labels: Optional semantic region labels (for SRSS)
        config: Configuration for metric computation
        random_state: Random seed for reproducibility

    Returns:
        SpatialMetrics object with all computed metrics

    Example:
        >>> from manifish.core.spatial_metrics import compute_all_spatial_metrics
        >>> metrics = compute_all_spatial_metrics(coords, manifold_coords)
        >>> print(f"LDS: {metrics.lds:.4f}, CDS: {metrics.cds:.4f}")
        >>> print(f"Spatial Quality: {metrics.spatial_quality:.4f}")
    """
    if config is None:
        config = SpatialMetricsConfig()

    # Compute LDS
    lds = compute_lds(
        coords=coords,
        manifold_coords=manifold_coords,
        k_local=config.lds_k_local,
        n_distant=config.lds_n_distant,
        distance_threshold=config.lds_distance_threshold,
        random_state=random_state,
    )

    # Compute CDS
    cds = compute_cds(
        coords=coords,
        manifold_coords=manifold_coords,
        distance_bins=config.cds_distance_bins,
        n_samples_per_bin=config.cds_n_samples_per_bin,
        random_state=random_state,
    )

    # Compute SRSS (requires region labels)
    if region_labels is not None:
        srss = compute_srss(
            coords=coords,
            manifold_coords=manifold_coords,
            region_labels=region_labels,
            n_same_region=config.srss_n_same_region,
            n_different_region=config.srss_n_different_region,
            random_state=random_state,
        )
    else:
        # Default SRSS when no region labels provided
        srss = 1.0
        warnings.warn("No region_labels provided, SRSS set to default 1.0")

    # Compute RMSC
    rmsc = compute_rmsc(
        coords=coords,
        manifold_coords=manifold_coords,
        k_neighbors=config.rmsc_k_neighbors,
    )

    # Compute combined quality score
    spatial_quality = compute_spatial_quality(
        lds=lds,
        cds=cds,
        srss=srss,
        rmsc=rmsc,
        weights=config.quality_weights,
    )

    return SpatialMetrics(
        lds=lds,
        cds=cds,
        srss=srss,
        rmsc=rmsc,
        spatial_quality=spatial_quality,
    )


def compute_spatial_metrics_batch(
    coords_batch: np.ndarray,
    manifold_coords: np.ndarray,
    region_labels: Optional[np.ndarray] = None,
    config: Optional[SpatialMetricsConfig] = None,
    random_state: Optional[int] = None,
) -> List[SpatialMetrics]:
    """
    Compute spatial metrics for a batch of structures.

    Args:
        coords_batch: Batch of structure coordinates (n_samples, n_anchors)
        manifold_coords: Reference manifold coordinates
        region_labels: Optional semantic region labels
        config: Metric configuration
        random_state: Random seed

    Returns:
        List of SpatialMetrics objects, one per structure
    """
    coords_batch = np.atleast_2d(coords_batch)
    results = []

    for i, coords in enumerate(coords_batch):
        metrics = compute_all_spatial_metrics(
            coords=coords,
            manifold_coords=manifold_coords,
            region_labels=region_labels,
            config=config,
            random_state=random_state + i if random_state else None,
        )
        results.append(metrics)

    return results


# =============================================================================
# Utility Functions
# =============================================================================

def create_region_labels_from_crystal_system(
    crystal_systems: List[str],
) -> np.ndarray:
    """
    Create region labels from crystal system strings.

    Args:
        crystal_systems: List of crystal system names
            (e.g., ['cubic', 'hexagonal', 'tetragonal', ...])

    Returns:
        Integer labels for each structure
    """
    unique_systems = sorted(set(crystal_systems))
    system_to_label = {sys: i for i, sys in enumerate(unique_systems)}
    return np.array([system_to_label[sys] for sys in crystal_systems])


def create_region_labels_from_composition_family(
    compositions: List[str],
    n_elements_match: int = 1,
) -> np.ndarray:
    """
    Create region labels based on composition families.

    Structures with at least n_elements_match common elements
    are grouped together.

    Args:
        compositions: List of composition strings (e.g., ['Li2O', 'LiFePO4', ...])
        n_elements_match: Minimum elements to match for same family

    Returns:
        Integer labels for each structure
    """
    from collections import defaultdict
    import re

    def parse_elements(comp: str) -> set:
        """Extract element symbols from composition string."""
        return set(re.findall(r'[A-Z][a-z]?', comp))

    elements_list = [parse_elements(c) for c in compositions]

    # Use hierarchical clustering approach
    n = len(compositions)
    labels = np.arange(n)  # Start with each structure in its own cluster

    for i in range(n):
        for j in range(i + 1, n):
            common = len(elements_list[i] & elements_list[j])
            if common >= n_elements_match:
                # Merge clusters
                old_label = labels[j]
                labels[labels == old_label] = labels[i]

    # Renumber labels consecutively
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    return np.array([label_map[l] for l in labels])


def normalize_spatial_metrics(
    metrics_list: List[SpatialMetrics],
    method: str = 'minmax',
) -> List[SpatialMetrics]:
    """
    Normalize spatial metrics across a batch for fair comparison.

    Args:
        metrics_list: List of SpatialMetrics objects
        method: 'minmax' or 'zscore'

    Returns:
        List of normalized SpatialMetrics objects
    """
    if not metrics_list:
        return []

    # Extract values
    lds_values = np.array([m.lds for m in metrics_list])
    cds_values = np.array([m.cds for m in metrics_list])
    srss_values = np.array([m.srss for m in metrics_list])
    rmsc_values = np.array([m.rmsc for m in metrics_list])

    def normalize_array(arr, method):
        if method == 'minmax':
            min_val, max_val = arr.min(), arr.max()
            if max_val - min_val > 1e-10:
                return (arr - min_val) / (max_val - min_val)
            return np.ones_like(arr) * 0.5
        elif method == 'zscore':
            mean, std = arr.mean(), arr.std()
            if std > 1e-10:
                return (arr - mean) / std
            return np.zeros_like(arr)
        else:
            raise ValueError(f"Unknown method: {method}")

    lds_norm = normalize_array(lds_values, method)
    cds_norm = normalize_array(cds_values, method)
    srss_norm = normalize_array(srss_values, method)
    rmsc_norm = normalize_array(rmsc_values, method)

    # Recompute quality scores with normalized values
    normalized = []
    for i, m in enumerate(metrics_list):
        quality = compute_spatial_quality(
            lds=lds_norm[i],
            cds=cds_norm[i],
            srss=srss_norm[i],
            rmsc=rmsc_norm[i],
            normalize=False,  # Already normalized
        )
        normalized.append(SpatialMetrics(
            lds=lds_norm[i],
            cds=cds_norm[i],
            srss=srss_norm[i],
            rmsc=rmsc_norm[i],
            spatial_quality=quality,
        ))

    return normalized
