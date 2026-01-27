"""
Anchor selection and management for unified manifold representation.

This module implements anchor selection using the DIRECT sampler approach
with Birch clustering, as described in the Platonic Representation method.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from dataclasses import dataclass, field
import pickle


@dataclass
class Anchor:
    """Represents a single anchor structure with its embedding."""
    embedding: np.ndarray  # The embedding vector from MLIP
    index: int  # Index in the original dataset
    structure_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AnchorSet:
    """A collection of anchors for Platonic projection."""
    anchors: List[Anchor]
    source_model: str  # Which MLIP the anchors were derived from
    n_clusters: int

    @property
    def embeddings(self) -> np.ndarray:
        """Get stacked anchor embeddings as numpy array."""
        return np.stack([a.embedding for a in self.anchors])

    @property
    def indices(self) -> List[int]:
        """Get list of anchor indices."""
        return [a.index for a in self.anchors]

    def save(self, path: str):
        """Save anchor set to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'AnchorSet':
        """Load anchor set from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class DIRECTAnchorSelector:
    """
    Select diverse anchor structures using DIRECT sampler with Birch clustering.

    This is the primary anchor selection method used in Platonic Representation.
    Uses clustering to find diverse representatives that span the embedding space.

    Reference:
        DIRECT (Diversity-based Representative Sampling) from maml package.

    Example:
        >>> selector = DIRECTAnchorSelector(n_clusters=100, k_per_cluster=1)
        >>> anchor_set = selector.select(embeddings, model_name='mace')
        >>> print(f"Selected {len(anchor_set.anchors)} anchors")
    """

    def __init__(
        self,
        n_clusters: int = 100,
        k_per_cluster: int = 1,
        threshold_init: float = 0.3,
        random_state: Optional[int] = None,
    ):
        """
        Initialize DIRECT anchor selector.

        Args:
            n_clusters: Number of Birch clusters to create
            k_per_cluster: Number of samples to select from each cluster
            threshold_init: Initial threshold for Birch clustering
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.k_per_cluster = k_per_cluster
        self.threshold_init = threshold_init
        self.random_state = random_state
        self._sampler = None
        self._selection_result = None

    def select(
        self,
        embeddings: Union[np.ndarray, List[np.ndarray]],
        model_name: str = 'unknown',
        structure_ids: Optional[List[str]] = None,
    ) -> AnchorSet:
        """
        Select anchors from embeddings using DIRECT sampler.

        Args:
            embeddings: Either a 2D array (n_samples, embedding_dim) or
                       list of per-atom embeddings that will be averaged
            model_name: Name of the MLIP model that generated embeddings
            structure_ids: Optional list of structure identifiers

        Returns:
            AnchorSet containing selected anchors
        """
        # Handle list of variable-length embeddings (per-atom)
        if isinstance(embeddings, list):
            embeddings = self._aggregate_atom_embeddings(embeddings)

        embeddings = np.asarray(embeddings)

        try:
            from maml.sampling.direct import (
                BirchClustering,
                DIRECTSampler,
                SelectKFromClusters,
            )

            self._sampler = DIRECTSampler(
                structure_encoder=None,  # Embeddings already computed
                clustering=BirchClustering(
                    n=self.n_clusters,
                    threshold_init=self.threshold_init,
                ),
                select_k_from_clusters=SelectKFromClusters(k=self.k_per_cluster),
            )

            self._selection_result = self._sampler.fit_transform(embeddings)
            selected_indices = self._selection_result['selected_indexes']

        except ImportError:
            print("Warning: maml not installed. Using fallback k-means selection.")
            selected_indices = self._fallback_kmeans_selection(embeddings)

        # Build anchor objects
        anchors = []
        for idx in selected_indices:
            anchor = Anchor(
                embedding=embeddings[idx],
                index=idx,
                structure_id=structure_ids[idx] if structure_ids else None,
            )
            anchors.append(anchor)

        return AnchorSet(
            anchors=anchors,
            source_model=model_name,
            n_clusters=self.n_clusters,
        )

    def _aggregate_atom_embeddings(
        self,
        atom_embeddings: List[np.ndarray],
        method: str = 'mean',
    ) -> np.ndarray:
        """
        Aggregate per-atom embeddings to per-structure embeddings.

        Args:
            atom_embeddings: List of arrays, each (n_atoms, embedding_dim)
            method: 'mean', 'sum', or 'max'

        Returns:
            Array of shape (n_structures, embedding_dim)
        """
        aggregated = []
        for emb in atom_embeddings:
            emb = np.asarray(emb)
            if emb.ndim == 1:
                aggregated.append(emb)
            elif method == 'mean':
                aggregated.append(np.mean(emb, axis=0))
            elif method == 'sum':
                aggregated.append(np.sum(emb, axis=0))
            elif method == 'max':
                aggregated.append(np.max(emb, axis=0))
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

        return np.stack(aggregated)

    def _fallback_kmeans_selection(self, embeddings: np.ndarray) -> List[int]:
        """Fallback selection using k-means when maml is not available."""
        from sklearn.cluster import KMeans

        n_samples = len(embeddings)
        n_clusters = min(self.n_clusters, n_samples)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(embeddings)

        # Select point closest to centroid in each cluster
        selected = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_indices = np.where(mask)[0]
            cluster_points = embeddings[mask]
            centroid = kmeans.cluster_centers_[cluster_id]

            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected.append(closest_idx)

        return selected

    @property
    def selection_info(self) -> Optional[Dict]:
        """Get information about the last selection (if using DIRECT)."""
        if self._selection_result is None:
            return None
        return {
            'n_selected': len(self._selection_result['selected_indexes']),
            'pca_features_shape': self._selection_result.get('PCAfeatures', np.array([])).shape,
        }


class FPSAnchorSelector:
    """
    Farthest Point Sampling anchor selector.

    Alternative to DIRECT for cases where you want guaranteed maximal spread.
    """

    def __init__(
        self,
        n_anchors: int = 100,
        random_state: Optional[int] = None,
    ):
        self.n_anchors = n_anchors
        self.random_state = random_state

    def select(
        self,
        embeddings: Union[np.ndarray, List[np.ndarray]],
        model_name: str = 'unknown',
        structure_ids: Optional[List[str]] = None,
    ) -> AnchorSet:
        """Select anchors using Farthest Point Sampling."""
        if isinstance(embeddings, list):
            embeddings = np.stack([np.mean(e, axis=0) if e.ndim > 1 else e for e in embeddings])

        embeddings = np.asarray(embeddings)
        n_samples = len(embeddings)
        n_select = min(self.n_anchors, n_samples)

        # Initialize with random point
        rng = np.random.default_rng(self.random_state)
        selected = [rng.integers(n_samples)]
        min_distances = np.full(n_samples, np.inf)

        for _ in range(n_select - 1):
            # Update minimum distances
            last_selected = selected[-1]
            distances = np.linalg.norm(embeddings - embeddings[last_selected], axis=1)
            min_distances = np.minimum(min_distances, distances)

            # Select point with maximum minimum distance
            min_distances[selected] = -np.inf  # Exclude already selected
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)

        anchors = [
            Anchor(
                embedding=embeddings[idx],
                index=idx,
                structure_id=structure_ids[idx] if structure_ids else None,
            )
            for idx in selected
        ]

        return AnchorSet(
            anchors=anchors,
            source_model=model_name,
            n_clusters=n_select,
        )


# Convenience functions

def select_anchors(
    embeddings: Union[np.ndarray, List[np.ndarray]],
    n_anchors: int = 100,
    method: str = 'direct',
    model_name: str = 'unknown',
    structure_ids: Optional[List[str]] = None,
    **kwargs,
) -> AnchorSet:
    """
    Convenience function to select anchors.

    Args:
        embeddings: Structure embeddings from an MLIP
        n_anchors: Number of anchors to select
        method: 'direct' (DIRECT sampler) or 'fps' (Farthest Point Sampling)
        model_name: Name of the MLIP model
        structure_ids: Optional structure identifiers
        **kwargs: Additional arguments for the selector

    Returns:
        AnchorSet containing selected anchors

    Example:
        >>> embeddings = extractor.get_embeddings(structures)
        >>> anchors = select_anchors(embeddings, n_anchors=100, method='direct')
    """
    if method == 'direct':
        selector = DIRECTAnchorSelector(n_clusters=n_anchors, **kwargs)
    elif method == 'fps':
        selector = FPSAnchorSelector(n_anchors=n_anchors, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'direct' or 'fps'.")

    return selector.select(embeddings, model_name=model_name, structure_ids=structure_ids)


def load_precomputed_anchors(path: str) -> AnchorSet:
    """Load precomputed anchors from file."""
    return AnchorSet.load(path)


# Example anchor indices from MP-20 train set (for reproducibility)
MP20_ANCHOR_INDICES_100 = [
    222430, 144721, 282484, 38074, 187477, 19816, 38484, 204870, 223370, 180858,
    239896, 271612, 270370, 256655, 164, 108486, 282577, 168218, 155521, 236527,
    139439, 134383, 6039, 30614, 181679, 25264, 185297, 125740, 75455, 195535,
    281120, 158363, 252468, 102981, 190032, 107059, 226948, 44342, 236453, 131321,
    205532, 174744, 169073, 104658, 230, 9999, 17943, 279479, 225757, 34751,
    2456, 1932, 25170, 52622, 222079, 227723, 164773, 135893, 12023, 258104,
    146125, 162523, 188573, 187378, 113021, 63466, 260179, 214, 93285, 277786,
    216748, 248850, 237493, 114495, 7366, 139062, 218251, 143350, 169531, 88557,
    211490, 169214, 100378, 270997, 84444, 241854, 154502, 246833, 112772, 160036,
    117303, 182341, 119695, 37694, 193103, 261515, 97870, 264644, 107630, 196220,
]
