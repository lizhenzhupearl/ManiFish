"""
Metrics for cross-model comparison, similarity, and intrinsic dimension estimation.

This module implements comparison metrics from the Platonic Representation work:
- Optimal Transport (OT) for distribution comparison
- Mutual KNN for representation similarity
- Intrinsic Dimension estimation (Two-NN algorithm)
- CKA and Procrustes for structural similarity
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


# =============================================================================
# Optimal Transport Metrics
# =============================================================================

def compute_optimal_transport(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    metric: str = 'euclidean',
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None,
    use_sinkhorn: bool = False,
    reg: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal transport plan and cost between two embedding distributions.

    Uses the POT (Python Optimal Transport) library.

    Args:
        embeddings1: First set of embeddings (n1, dim)
        embeddings2: Second set of embeddings (n2, dim)
        metric: Distance metric ('euclidean', 'cosine', etc.)
        weights1: Optional weights for first distribution
        weights2: Optional weights for second distribution
        use_sinkhorn: Use entropy-regularized Sinkhorn algorithm
        reg: Regularization parameter for Sinkhorn

    Returns:
        Tuple of (transport_plan, total_cost)

    Example:
        >>> plan, cost = compute_optimal_transport(mace_rep, chgnet_rep)
        >>> print(f"OT cost between MACE and CHGNet: {cost:.4f}")
    """
    try:
        import ot
    except ImportError:
        raise ImportError("POT library required. Install with: pip install POT")

    embeddings1 = np.asarray(embeddings1)
    embeddings2 = np.asarray(embeddings2)

    # Compute cost matrix
    cost_matrix = ot.dist(embeddings1, embeddings2, metric=metric)

    n, m = len(embeddings1), len(embeddings2)
    a = np.asarray(weights1) if weights1 is not None else np.ones(n) / n
    b = np.asarray(weights2) if weights2 is not None else np.ones(m) / m

    if use_sinkhorn:
        ot_plan = ot.sinkhorn(a, b, cost_matrix, reg)
    else:
        ot_plan = ot.emd(a, b, cost_matrix)

    total_cost = np.sum(ot_plan * cost_matrix)

    return ot_plan, total_cost


def compute_wasserstein_distance(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    metric: str = 'euclidean',
) -> float:
    """Convenience function to compute just the Wasserstein distance."""
    _, cost = compute_optimal_transport(embeddings1, embeddings2, metric=metric)
    return cost


# =============================================================================
# Mutual KNN Similarity
# =============================================================================

def compute_mutual_knn(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    k: int = 10,
    n_samples: Optional[int] = None,
    random_state: Optional[int] = None,
) -> float:
    """
    Compute mutual k-nearest neighbor similarity between two representations.

    Measures how well the neighborhood structure is preserved between
    two different representations of the same data.

    Args:
        embeddings1: First representation (n_samples, dim1)
        embeddings2: Second representation (n_samples, dim2)
        k: Number of nearest neighbors to consider
        n_samples: If provided, subsample for efficiency
        random_state: Random seed for subsampling

    Returns:
        Mutual KNN score in [0, 1], higher = more similar neighborhoods

    Example:
        >>> score = compute_mutual_knn(mace_platonic, chgnet_platonic, k=10)
        >>> print(f"Mutual KNN similarity: {score:.4f}")
    """
    embeddings1 = np.asarray(embeddings1)
    embeddings2 = np.asarray(embeddings2)

    assert len(embeddings1) == len(embeddings2), "Must have same number of samples"

    # Subsample if requested
    if n_samples is not None and n_samples < len(embeddings1):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings1), size=n_samples, replace=False)
        embeddings1 = embeddings1[indices]
        embeddings2 = embeddings2[indices]

    n = len(embeddings1)
    k = min(k, n - 1)

    # Find k-nearest neighbors in each space
    nbrs1 = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(embeddings1)
    nbrs2 = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(embeddings2)

    _, indices1 = nbrs1.kneighbors(embeddings1)
    _, indices2 = nbrs2.kneighbors(embeddings2)

    # Remove self from neighbors (first column)
    indices1 = indices1[:, 1:]
    indices2 = indices2[:, 1:]

    # Compute overlap
    total_overlap = 0
    for i in range(n):
        neighbors1 = set(indices1[i])
        neighbors2 = set(indices2[i])
        overlap = len(neighbors1 & neighbors2)
        total_overlap += overlap

    mutual_knn_score = total_overlap / (n * k)

    return mutual_knn_score


def compute_mutual_knn_matrix(
    embeddings_list: List[np.ndarray],
    model_names: Optional[List[str]] = None,
    k: int = 10,
    n_samples: Optional[int] = 1000,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise mutual KNN similarity matrix for multiple models.

    Args:
        embeddings_list: List of embeddings from different models
        model_names: Names for each model
        k: Number of nearest neighbors
        n_samples: Subsample size for efficiency
        random_state: Random seed

    Returns:
        Tuple of (similarity_matrix, model_names)

    Example:
        >>> embeddings = [mace_rep, chgnet_rep, m3gnet_rep]
        >>> matrix, names = compute_mutual_knn_matrix(embeddings, k=10)
    """
    n = len(embeddings_list)
    if model_names is None:
        model_names = [f'model_{i}' for i in range(n)]

    score_matrix = np.zeros((n, n))

    # Subsample indices (same for all)
    n_total = len(embeddings_list[0])
    if n_samples is not None and n_samples < n_total:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_total, size=n_samples, replace=False)
        embeddings_list = [emb[indices] for emb in embeddings_list]

    for i in range(n):
        for j in range(n):
            if i == j:
                score_matrix[i, j] = 1.0
            elif i < j:
                score = compute_mutual_knn(
                    embeddings_list[i],
                    embeddings_list[j],
                    k=k,
                )
                score_matrix[i, j] = score
                score_matrix[j, i] = score

    return score_matrix, model_names


# =============================================================================
# Intrinsic Dimension Estimation
# =============================================================================

def estimate_intrinsic_dimension(
    embeddings: np.ndarray,
    k_omit: int = 1,
) -> Tuple[float, float, np.ndarray]:
    """
    Estimate intrinsic dimension using the Two-NN algorithm.

    Reference: Facco E, et al. "Estimating the intrinsic dimension of datasets
    by a minimal neighborhood information" (2017).

    Args:
        embeddings: Point cloud embeddings (n_samples, dim)
        k_omit: Number of tail points to omit (prevents log(0) error)

    Returns:
        Tuple of (collapse_fraction, intrinsic_dimension, collapsed_indices)
        - collapse_fraction: Fraction of points with zero distance to nearest neighbor
        - intrinsic_dimension: Estimated ID
        - collapsed_indices: Indices of collapsed points

    Example:
        >>> collapse, id_estimate, collapsed = estimate_intrinsic_dimension(embeddings)
        >>> print(f"Intrinsic dimension: {id_estimate:.2f}")
        >>> print(f"Collapse fraction: {collapse:.4f}")
    """
    embeddings = np.asarray(embeddings)
    N = embeddings.shape[0]

    # Compute distances to first 2 nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    # Extract r1 (dist to 1st neighbor) and r2 (dist to 2nd neighbor)
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Track collapsed points
    num_collapsed = np.sum(r1 == 0)
    collapse_fraction = num_collapsed / N
    collapsed_indices = np.where(r1 == 0)[0]

    # Filter out cases where r1 is 0 (duplicates)
    mask = r1 > 0
    r1_filtered = r1[mask]
    r2_filtered = r2[mask]

    if len(r1_filtered) < 2:
        return collapse_fraction, np.nan, collapsed_indices

    # Compute ratios mu = r2 / r1
    mu = r2_filtered / r1_filtered

    # Compute empirical cumulative distribution
    mu_sorted = np.sort(mu)
    F_mu = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)

    # Truncate to avoid log(0)
    if k_omit > 0:
        mu_trunc = mu_sorted[:-k_omit]
        F_mu_trunc = F_mu[:-k_omit]
    else:
        mu_trunc = mu_sorted
        F_mu_trunc = F_mu

    if len(mu_trunc) < 2:
        return collapse_fraction, np.nan, collapsed_indices

    # Linear regression on log coordinates
    x = np.log(mu_trunc).reshape(-1, 1)
    y = -np.log(1 - F_mu_trunc).reshape(-1, 1)

    # Fit line through origin
    reg = LinearRegression(fit_intercept=False).fit(x, y)
    estimated_id = reg.coef_[0][0]

    return collapse_fraction, estimated_id, collapsed_indices


def compare_intrinsic_dimensions(
    original_embeddings: List[np.ndarray],
    platonic_embeddings: List[np.ndarray],
    model_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare intrinsic dimensions between original and Platonic representations.

    Args:
        original_embeddings: List of original embeddings from each model
        platonic_embeddings: List of Platonic representations from each model
        model_names: Names for each model

    Returns:
        Dictionary with ID estimates for each model

    Example:
        >>> results = compare_intrinsic_dimensions(
        ...     [mace_orig, chgnet_orig],
        ...     [mace_platonic, chgnet_platonic],
        ...     ['MACE', 'CHGNet']
        ... )
    """
    if model_names is None:
        model_names = [f'model_{i}' for i in range(len(original_embeddings))]

    results = {}
    for i, name in enumerate(model_names):
        orig_collapse, orig_id, _ = estimate_intrinsic_dimension(original_embeddings[i])
        plat_collapse, plat_id, _ = estimate_intrinsic_dimension(platonic_embeddings[i])

        results[name] = {
            'original_id': orig_id,
            'original_collapse': orig_collapse,
            'platonic_id': plat_id,
            'platonic_collapse': plat_collapse,
        }

    return results


# =============================================================================
# CKA (Centered Kernel Alignment)
# =============================================================================

def compute_cka(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    kernel: str = 'linear',
) -> float:
    """
    Compute Centered Kernel Alignment between two representations.

    CKA measures the similarity between two representations independent
    of orthogonal transformations and isotropic scaling.

    Args:
        embeddings1: First representation (n_samples, dim1)
        embeddings2: Second representation (n_samples, dim2)
        kernel: 'linear' or 'rbf'

    Returns:
        CKA score in [0, 1], higher = more similar

    Example:
        >>> cka_score = compute_cka(mace_rep, chgnet_rep)
        >>> print(f"CKA similarity: {cka_score:.4f}")
    """
    embeddings1 = np.asarray(embeddings1)
    embeddings2 = np.asarray(embeddings2)

    def _center_gram(K):
        """Center a Gram matrix."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def _gram_linear(X):
        return X @ X.T

    def _gram_rbf(X, sigma=None):
        GX = X @ X.T
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            sigma = np.sqrt(np.median(KX[KX > 0]))
        return np.exp(-KX / (2 * sigma ** 2))

    if kernel == 'linear':
        K1 = _gram_linear(embeddings1)
        K2 = _gram_linear(embeddings2)
    elif kernel == 'rbf':
        K1 = _gram_rbf(embeddings1)
        K2 = _gram_rbf(embeddings2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    K1c = _center_gram(K1)
    K2c = _center_gram(K2)

    # HSIC
    hsic = np.sum(K1c * K2c)
    norm1 = np.sqrt(np.sum(K1c * K1c))
    norm2 = np.sqrt(np.sum(K2c * K2c))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return hsic / (norm1 * norm2)


# =============================================================================
# Procrustes Analysis
# =============================================================================

def compute_procrustes_distance(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    angular: bool = True,
) -> float:
    """
    Compute Procrustes distance between two representations.

    Finds the optimal orthogonal transformation aligning the representations.

    Args:
        embeddings1: First representation (n_samples, dim)
        embeddings2: Second representation (n_samples, dim)
        angular: If True, return angular distance; else Frobenius

    Returns:
        Procrustes distance (lower = more similar)

    Example:
        >>> dist = compute_procrustes_distance(mace_rep, chgnet_rep)
        >>> print(f"Procrustes distance: {dist:.4f}")
    """
    from scipy.linalg import orthogonal_procrustes

    embeddings1 = np.asarray(embeddings1)
    embeddings2 = np.asarray(embeddings2)

    # Center
    X = embeddings1 - embeddings1.mean(axis=0)
    Y = embeddings2 - embeddings2.mean(axis=0)

    # Scale to unit Frobenius norm
    X = X / np.linalg.norm(X, 'fro')
    Y = Y / np.linalg.norm(Y, 'fro')

    # Find optimal rotation
    R, scale = orthogonal_procrustes(X, Y)

    # Compute distance
    X_aligned = X @ R
    frobenius_dist = np.linalg.norm(X_aligned - Y, 'fro')

    if angular:
        # Convert to angular distance
        return np.arcsin(frobenius_dist / np.sqrt(2))
    return frobenius_dist


# =============================================================================
# Model Consensus (from original metrics)
# =============================================================================

def compute_consensus(
    platonic_representations: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute multi-model consensus scores for each sample.

    High consensus = all models agree on the Platonic representation.
    Low consensus = models disagree, indicating uncertainty.

    Args:
        platonic_representations: Dict mapping model_name -> (n_samples, n_anchors)

    Returns:
        Consensus scores (n_samples,) in [0, 1]

    Example:
        >>> reps = {'mace': mace_rep, 'chgnet': chgnet_rep}
        >>> consensus = compute_consensus(reps)
        >>> high_confidence = consensus > 0.8
    """
    model_names = list(platonic_representations.keys())
    n_models = len(model_names)

    if n_models < 2:
        # Single model, perfect consensus
        n_samples = len(list(platonic_representations.values())[0])
        return np.ones(n_samples)

    # Stack all representations
    all_reps = np.stack([platonic_representations[name] for name in model_names])
    # Shape: (n_models, n_samples, n_anchors)

    # Compute mean representation
    mean_rep = np.mean(all_reps, axis=0)

    # Compute variance across models for each sample
    variance = np.var(all_reps, axis=0)  # (n_samples, n_anchors)
    mean_variance = np.mean(variance, axis=1)  # (n_samples,)

    # Convert to consensus score (low variance = high consensus)
    consensus = 1.0 / (1.0 + mean_variance)

    return consensus


def compute_quality_score(
    stability: float,
    novelty: float,
    consensus: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute combined quality score.

    Args:
        stability: Stability score [0, 1]
        novelty: Novelty score [0, 1]
        consensus: Consensus score [0, 1]
        weights: Optional custom weights

    Returns:
        Combined quality score [0, 1]
    """
    if weights is None:
        weights = {
            'stability': 0.5,
            'novelty': 0.3,
            'consensus': 0.2,
        }

    score = (
        weights['stability'] * stability +
        weights['novelty'] * novelty +
        weights['consensus'] * consensus
    )

    return score
