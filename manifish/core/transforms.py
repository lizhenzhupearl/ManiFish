"""
Embedding Transformation Module

Transform high-dimensional MLIP embeddings into lower-dimensional representations
suitable for manifold analysis and visualization.

Transformation Methods:
1. Anchor-based projection (relative embeddings)
2. PCA
3. UMAP
4. t-SNE

Usage:
    from embedding_transform import EmbeddingTransformer

    transformer = EmbeddingTransformer(
        method="anchor",
        anchors=anchor_embeddings,
        distance_metric="cosine"
    )
    transformed = transformer.fit_transform(embeddings)
"""

import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np

# PyTorch for anchor-based transformations
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Anchor-based transformations will use numpy.")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TransformResult:
    """Container for transformation results."""
    transformed: np.ndarray  # (n_samples, n_components)
    n_samples: int
    n_components: int
    original_dim: int
    method: str
    params: Dict = field(default_factory=dict)

    # For anchor-based methods
    anchors: Optional[np.ndarray] = None
    distance_metric: Optional[str] = None

    # For fitted models (PCA, UMAP, etc.)
    model: Optional[object] = None


# =============================================================================
# Anchor-Based Transformation
# =============================================================================

class AnchorTransformer:
    """
    Transform embeddings to relative representations based on anchor distances.

    Projects high-dimensional embeddings into a space defined by distances/similarities
    to a set of anchor points. Output dimension = number of anchors.

    Args:
        anchors: Anchor embeddings (n_anchors, embed_dim)
        distance_metric: Distance/similarity metric
            - "cosine": Cosine similarity (default, range [-1, 1])
            - "euclidean": Euclidean distance (L2)
            - "manhattan": Manhattan distance (L1)
            - "geodesic": Geodesic distance on unit sphere (angle in radians)
            - "linf": L-infinity (Chebyshev) distance
            - "rbf": RBF kernel (Gaussian similarity)
        use_torch: Use PyTorch for computation (faster on GPU)
        device: Device for PyTorch ("cuda", "cpu", "auto")
        gamma: Gamma parameter for RBF kernel (default: auto)
    """

    SUPPORTED_METRICS = ["cosine", "euclidean", "manhattan", "geodesic", "linf", "rbf"]

    def __init__(
        self,
        anchors: np.ndarray,
        distance_metric: str = "cosine",
        use_torch: bool = True,
        device: str = "auto",
        gamma: Optional[float] = None,
    ):
        self.anchors = np.asarray(anchors)
        self.n_anchors = len(self.anchors)
        self.embed_dim = self.anchors.shape[1]
        self.distance_metric = distance_metric.lower()
        self.gamma = gamma

        if self.distance_metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {distance_metric}. "
                f"Supported: {self.SUPPORTED_METRICS}"
            )

        self.use_torch = use_torch and HAS_TORCH
        if device == "auto":
            self.device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Precompute anchor tensor if using torch
        if self.use_torch:
            self.anchors_tensor = torch.tensor(
                self.anchors, dtype=torch.float32, device=self.device
            )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to anchor-relative representation.

        Args:
            embeddings: Input embeddings (n_samples, embed_dim)

        Returns:
            Relative embeddings (n_samples, n_anchors)
        """
        embeddings = np.asarray(embeddings)

        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {embeddings.shape[1]}, "
                f"expected {self.embed_dim}"
            )

        if self.use_torch:
            return self._transform_torch(embeddings)
        else:
            return self._transform_numpy(embeddings)

    def _transform_torch(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform using PyTorch."""
        x = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        anchors = self.anchors_tensor

        if self.distance_metric == "cosine":
            x_norm = F.normalize(x, p=2, dim=-1)
            a_norm = F.normalize(anchors, p=2, dim=-1)
            result = torch.einsum("bm, am -> ba", x_norm, a_norm)

        elif self.distance_metric == "euclidean":
            result = torch.cdist(x, anchors, p=2)

        elif self.distance_metric == "manhattan":
            result = torch.cdist(x, anchors, p=1)

        elif self.distance_metric == "geodesic":
            x_norm = F.normalize(x, p=2, dim=-1)
            a_norm = F.normalize(anchors, p=2, dim=-1)
            cos_sim = torch.einsum("bm, am -> ba", x_norm, a_norm).clamp(-1.0, 1.0)
            result = torch.acos(cos_sim)

        elif self.distance_metric == "linf":
            result = torch.cdist(x, anchors, p=float('inf'))

        elif self.distance_metric == "rbf":
            # RBF kernel: exp(-gamma * ||x - a||^2)
            sq_dist = torch.cdist(x, anchors, p=2) ** 2
            gamma = self.gamma if self.gamma else 1.0 / self.embed_dim
            result = torch.exp(-gamma * sq_dist)

        return result.cpu().numpy()

    def _transform_numpy(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform using NumPy (fallback)."""
        from scipy.spatial.distance import cdist

        if self.distance_metric == "cosine":
            x_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            a_norm = self.anchors / (np.linalg.norm(self.anchors, axis=1, keepdims=True) + 1e-8)
            result = x_norm @ a_norm.T

        elif self.distance_metric == "euclidean":
            result = cdist(embeddings, self.anchors, metric='euclidean')

        elif self.distance_metric == "manhattan":
            result = cdist(embeddings, self.anchors, metric='cityblock')

        elif self.distance_metric == "geodesic":
            x_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            a_norm = self.anchors / (np.linalg.norm(self.anchors, axis=1, keepdims=True) + 1e-8)
            cos_sim = np.clip(x_norm @ a_norm.T, -1.0, 1.0)
            result = np.arccos(cos_sim)

        elif self.distance_metric == "linf":
            result = cdist(embeddings, self.anchors, metric='chebyshev')

        elif self.distance_metric == "rbf":
            sq_dist = cdist(embeddings, self.anchors, metric='sqeuclidean')
            gamma = self.gamma if self.gamma else 1.0 / self.embed_dim
            result = np.exp(-gamma * sq_dist)

        return result

    def fit_transform(self, embeddings: np.ndarray) -> TransformResult:
        """Transform and return structured result."""
        transformed = self.transform(embeddings)

        return TransformResult(
            transformed=transformed,
            n_samples=len(embeddings),
            n_components=self.n_anchors,
            original_dim=self.embed_dim,
            method="anchor",
            anchors=self.anchors,
            distance_metric=self.distance_metric,
            params={
                "n_anchors": self.n_anchors,
                "distance_metric": self.distance_metric,
                "gamma": self.gamma,
            },
        )


# =============================================================================
# PCA Transformer
# =============================================================================

class PCATransformer:
    """
    PCA-based dimensionality reduction.

    Args:
        n_components: Number of principal components
        whiten: Whether to whiten the output
    """

    def __init__(self, n_components: int = 3, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None

    def fit(self, embeddings: np.ndarray) -> "PCATransformer":
        """Fit PCA on embeddings."""
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using fitted PCA."""
        if self.pca is None:
            raise ValueError("Must call fit() before transform()")
        return self.pca.transform(embeddings)

    def fit_transform(self, embeddings: np.ndarray) -> TransformResult:
        """Fit and transform embeddings."""
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        transformed = self.pca.fit_transform(embeddings)

        return TransformResult(
            transformed=transformed,
            n_samples=len(embeddings),
            n_components=self.n_components,
            original_dim=embeddings.shape[1],
            method="pca",
            model=self.pca,
            params={
                "n_components": self.n_components,
                "whiten": self.whiten,
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "total_variance_explained": sum(self.pca.explained_variance_ratio_),
            },
        )


# =============================================================================
# UMAP Transformer
# =============================================================================

class UMAPTransformer:
    """
    UMAP-based dimensionality reduction.

    Args:
        n_components: Output dimensions
        n_neighbors: Number of neighbors for manifold approximation
        min_dist: Minimum distance between points in low-dim space
        metric: Distance metric
        random_state: Random seed
    """

    def __init__(
        self,
        n_components: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.umap = None

    def fit(self, embeddings: np.ndarray) -> "UMAPTransformer":
        """Fit UMAP on embeddings."""
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP required. Install with: pip install umap-learn")

        self.umap = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        self.umap.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using fitted UMAP."""
        if self.umap is None:
            raise ValueError("Must call fit() before transform()")
        return self.umap.transform(embeddings)

    def fit_transform(self, embeddings: np.ndarray) -> TransformResult:
        """Fit and transform embeddings."""
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP required. Install with: pip install umap-learn")

        self.umap = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        transformed = self.umap.fit_transform(embeddings)

        return TransformResult(
            transformed=transformed,
            n_samples=len(embeddings),
            n_components=self.n_components,
            original_dim=embeddings.shape[1],
            method="umap",
            model=self.umap,
            params={
                "n_components": self.n_components,
                "n_neighbors": self.n_neighbors,
                "min_dist": self.min_dist,
                "metric": self.metric,
            },
        )


# =============================================================================
# t-SNE Transformer
# =============================================================================

class TSNETransformer:
    """
    t-SNE-based dimensionality reduction.

    Note: t-SNE doesn't support transform() on new data.

    Args:
        n_components: Output dimensions (2 or 3)
        perplexity: Perplexity parameter
        learning_rate: Learning rate
        n_iter: Number of iterations
        random_state: Random seed
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = "auto",
        n_iter: int = 1000,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit_transform(self, embeddings: np.ndarray) -> TransformResult:
        """Fit and transform embeddings."""
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        transformed = tsne.fit_transform(embeddings)

        return TransformResult(
            transformed=transformed,
            n_samples=len(embeddings),
            n_components=self.n_components,
            original_dim=embeddings.shape[1],
            method="tsne",
            params={
                "n_components": self.n_components,
                "perplexity": self.perplexity,
                "learning_rate": self.learning_rate,
                "n_iter": self.n_iter,
            },
        )


# =============================================================================
# Unified Interface
# =============================================================================

class EmbeddingTransformer:
    """
    Unified interface for embedding transformations.

    Args:
        method: Transformation method
            - "anchor": Anchor-based relative embeddings
            - "pca": Principal Component Analysis
            - "umap": UMAP
            - "tsne": t-SNE
        n_components: Output dimensions (for pca, umap, tsne)
        anchors: Anchor embeddings (required for "anchor" method)
        distance_metric: Distance metric for anchor method
        **kwargs: Method-specific parameters

    Example:
        >>> # Anchor-based transformation
        >>> transformer = EmbeddingTransformer(
        ...     method="anchor",
        ...     anchors=anchor_embeddings,
        ...     distance_metric="cosine"
        ... )
        >>> result = transformer.fit_transform(embeddings)

        >>> # PCA transformation
        >>> transformer = EmbeddingTransformer(method="pca", n_components=3)
        >>> result = transformer.fit_transform(embeddings)
    """

    def __init__(
        self,
        method: Literal["anchor", "pca", "umap", "tsne"] = "anchor",
        n_components: Optional[int] = None,
        anchors: Optional[np.ndarray] = None,
        distance_metric: str = "cosine",
        **kwargs
    ):
        self.method = method.lower()

        if self.method == "anchor":
            if anchors is None:
                raise ValueError("anchors required for anchor method")
            self.transformer = AnchorTransformer(
                anchors=anchors,
                distance_metric=distance_metric,
                **kwargs
            )
        elif self.method == "pca":
            n_components = n_components or 3
            self.transformer = PCATransformer(n_components=n_components, **kwargs)
        elif self.method == "umap":
            n_components = n_components or 3
            self.transformer = UMAPTransformer(n_components=n_components, **kwargs)
        elif self.method == "tsne":
            n_components = n_components or 2
            self.transformer = TSNETransformer(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def fit(self, embeddings: np.ndarray) -> "EmbeddingTransformer":
        """Fit transformer on embeddings (not applicable for anchor/tsne)."""
        if hasattr(self.transformer, 'fit'):
            self.transformer.fit(embeddings)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings."""
        return self.transformer.transform(embeddings)

    def fit_transform(self, embeddings: np.ndarray) -> TransformResult:
        """Fit and transform embeddings."""
        return self.transformer.fit_transform(embeddings)

    def save(self, path: str):
        """Save transformer to file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Transformer saved to {path}")

    @staticmethod
    def load(path: str) -> "EmbeddingTransformer":
        """Load transformer from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# Convenience Functions
# =============================================================================

def anchor_transform(
    embeddings: np.ndarray,
    anchors: np.ndarray,
    distance_metric: str = "cosine",
    device: str = "auto",
) -> np.ndarray:
    """
    Quick anchor-based transformation.

    Args:
        embeddings: Input embeddings (n_samples, embed_dim)
        anchors: Anchor embeddings (n_anchors, embed_dim)
        distance_metric: "cosine", "euclidean", "manhattan", "geodesic", "linf", "rbf"
        device: "cuda", "cpu", or "auto"

    Returns:
        Relative embeddings (n_samples, n_anchors)
    """
    transformer = AnchorTransformer(
        anchors=anchors,
        distance_metric=distance_metric,
        device=device,
    )
    return transformer.transform(embeddings)


def pca_transform(
    embeddings: np.ndarray,
    n_components: int = 3,
    whiten: bool = False,
) -> Tuple[np.ndarray, object]:
    """
    Quick PCA transformation.

    Returns:
        (transformed_embeddings, fitted_pca_model)
    """
    transformer = PCATransformer(n_components=n_components, whiten=whiten)
    result = transformer.fit_transform(embeddings)
    return result.transformed, result.model


def transform_and_split(
    ref_embeddings: np.ndarray,
    gen_embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 3,
    anchors: Optional[np.ndarray] = None,
    distance_metric: str = "cosine",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform reference and generated embeddings together.

    Fits on reference, transforms both. Useful for ensuring consistent
    transformation for manifold analysis.

    Args:
        ref_embeddings: Reference embeddings
        gen_embeddings: Generated embeddings
        method: Transformation method
        n_components: Output dimensions
        anchors: Anchors for anchor method
        distance_metric: Metric for anchor method
        **kwargs: Additional method parameters

    Returns:
        (ref_transformed, gen_transformed)
    """
    if method == "anchor":
        if anchors is None:
            raise ValueError("anchors required for anchor method")
        transformer = AnchorTransformer(anchors=anchors, distance_metric=distance_metric, **kwargs)
        ref_transformed = transformer.transform(ref_embeddings)
        gen_transformed = transformer.transform(gen_embeddings)

    elif method == "pca":
        transformer = PCATransformer(n_components=n_components, **kwargs)
        transformer.fit(ref_embeddings)
        ref_transformed = transformer.transform(ref_embeddings)
        gen_transformed = transformer.transform(gen_embeddings)

    elif method == "umap":
        transformer = UMAPTransformer(n_components=n_components, **kwargs)
        transformer.fit(ref_embeddings)
        ref_transformed = transformer.transform(ref_embeddings)
        gen_transformed = transformer.transform(gen_embeddings)

    else:
        raise ValueError(f"Method {method} doesn't support fit/transform split")

    return ref_transformed, gen_transformed


# =============================================================================
# Visualization
# =============================================================================

def plot_transformation_comparison(
    embeddings: np.ndarray,
    anchors: Optional[np.ndarray] = None,
    methods: List[str] = None,
    max_points: int = 5000,
    figsize: Tuple[int, int] = (16, 4),
    save_path: Optional[str] = None,
):
    """
    Compare different transformation methods visually.

    Args:
        embeddings: Original embeddings
        anchors: Anchors for anchor method (optional)
        methods: List of methods to compare
        max_points: Max points to plot
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    if methods is None:
        methods = ["pca"]
        if anchors is not None:
            methods.insert(0, "anchor_cosine")

    # Subsample if needed
    if len(embeddings) > max_points:
        idx = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings_sub = embeddings[idx]
    else:
        embeddings_sub = embeddings

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        if method.startswith("anchor"):
            if anchors is None:
                ax.text(0.5, 0.5, "No anchors provided", ha='center', va='center')
                continue
            metric = method.split("_")[1] if "_" in method else "cosine"
            result = anchor_transform(embeddings_sub, anchors, distance_metric=metric)
            # Use first 2 dimensions for visualization
            ax.scatter(result[:, 0], result[:, 1], c='blue', alpha=0.3, s=5)
            ax.set_xlabel(f"Anchor 1 ({metric})")
            ax.set_ylabel(f"Anchor 2 ({metric})")
            ax.set_title(f"Anchor-based ({metric})")

        elif method == "pca":
            result, _ = pca_transform(embeddings_sub, n_components=2)
            ax.scatter(result[:, 0], result[:, 1], c='green', alpha=0.3, s=5)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA")

        elif method == "umap":
            try:
                transformer = UMAPTransformer(n_components=2)
                result = transformer.fit_transform(embeddings_sub)
                ax.scatter(result.transformed[:, 0], result.transformed[:, 1],
                          c='purple', alpha=0.3, s=5)
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                ax.set_title("UMAP")
            except ImportError:
                ax.text(0.5, 0.5, "UMAP not installed", ha='center', va='center')

        elif method == "tsne":
            transformer = TSNETransformer(n_components=2)
            result = transformer.fit_transform(embeddings_sub)
            ax.scatter(result.transformed[:, 0], result.transformed[:, 1],
                      c='orange', alpha=0.3, s=5)
            ax.set_xlabel("t-SNE1")
            ax.set_ylabel("t-SNE2")
            ax.set_title("t-SNE")

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
    """Example demonstrating embedding transformations."""
    print("=" * 80)
    print("Embedding Transformation Examples")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    embed_dim = 128

    embeddings = np.random.randn(n_samples, embed_dim)
    anchors = np.random.randn(100, embed_dim)  # 100 anchors

    print(f"\nInput: {n_samples} samples, {embed_dim}-dim embeddings")
    print(f"Anchors: {len(anchors)}")

    # =========================================================================
    # Method 1: Anchor-based (your current approach)
    # =========================================================================
    print("\n--- Method 1: Anchor-based Transformation ---")

    transformer = EmbeddingTransformer(
        method="anchor",
        anchors=anchors,
        distance_metric="cosine"
    )
    result = transformer.fit_transform(embeddings)
    print(f"Output shape: {result.transformed.shape}")
    print(f"Distance metric: {result.distance_metric}")

    # Try different metrics
    for metric in ["cosine", "euclidean", "geodesic", "rbf"]:
        trans = anchor_transform(embeddings[:10], anchors, distance_metric=metric)
        print(f"  {metric}: shape {trans.shape}, range [{trans.min():.3f}, {trans.max():.3f}]")

    # =========================================================================
    # Method 2: PCA
    # =========================================================================
    print("\n--- Method 2: PCA ---")

    transformer = EmbeddingTransformer(method="pca", n_components=3)
    result = transformer.fit_transform(embeddings)
    print(f"Output shape: {result.transformed.shape}")
    print(f"Variance explained: {result.params['total_variance_explained']:.2%}")

    # =========================================================================
    # Full Pipeline Example
    # =========================================================================
    print("\n--- Full Pipeline Example ---")
    print("""
    from mlip_embedding_extractor import MLIPEmbeddingExtractor
    from anchor_selection import AnchorSelector
    from embedding_transform import EmbeddingTransformer
    from manifold_fish_analysis import ManifoldFishAnalyzer, aggregate_by_material_id

    # 1. Extract embeddings
    extractor = MLIPEmbeddingExtractor("mace", "/path/to/model")
    ref_results = extractor.extract_from_json("reference.json")
    gen_results = extractor.extract_from_json("generated.json")

    # 2. Select anchors from reference
    selector = AnchorSelector(method="direct", n_anchors=100)
    anchor_result = selector.fit_transform(ref_results.flat_embeddings)
    anchors = anchor_result.anchor_embeddings

    # 3. Transform both to anchor space
    transformer = EmbeddingTransformer(
        method="anchor",
        anchors=anchors,
        distance_metric="cosine"
    )
    ref_transformed = transformer.transform(ref_results.flat_embeddings)
    gen_transformed = transformer.transform(gen_results.flat_embeddings)

    # 4. Aggregate to material level
    ref_mat, ref_ids = aggregate_by_material_id(ref_transformed, ref_results.flat_material_ids)
    gen_mat, gen_ids = aggregate_by_material_id(gen_transformed, gen_results.flat_material_ids)

    # 5. Analyze with ManifoldFishAnalyzer
    analyzer = ManifoldFishAnalyzer(ref_mat, ref_ids, boundary_method="alpha_shape")
    results = analyzer.analyze(gen_mat, gen_ids)
    analyzer.print_summary(results)
    """)

    return result


if __name__ == "__main__":
    example_usage()
