"""
Manifold construction and Platonic projection for cross-MLIP comparison.

The key idea is to project embeddings from different MLIPs into a common
anchor-based space using cosine similarity. This creates a "Platonic
Representation" that allows direct comparison across models.
"""

import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from .anchors import AnchorSet


@dataclass
class PlatonicRepresentation:
    """
    Platonic representation of structure(s) in anchor space.

    This is the core output of the projection - embeddings transformed
    into a common space defined by anchor similarities.
    """
    coordinates: np.ndarray  # Shape: (n_samples, n_anchors) or (n_anchors,)
    model_name: str
    anchor_set: AnchorSet
    original_embeddings: Optional[np.ndarray] = None

    @property
    def n_samples(self) -> int:
        if self.coordinates.ndim == 1:
            return 1
        return self.coordinates.shape[0]

    @property
    def n_anchors(self) -> int:
        if self.coordinates.ndim == 1:
            return len(self.coordinates)
        return self.coordinates.shape[1]


@dataclass
class MultiModelRepresentation:
    """
    Collection of Platonic representations from multiple MLIPs.

    Used for consensus analysis and cross-model comparison.
    """
    representations: Dict[str, PlatonicRepresentation]
    structure_ids: Optional[List[str]] = None

    @property
    def model_names(self) -> List[str]:
        return list(self.representations.keys())

    def get_coordinates(self, model_name: str) -> np.ndarray:
        return self.representations[model_name].coordinates

    def get_all_coordinates(self) -> Dict[str, np.ndarray]:
        return {name: rep.coordinates for name, rep in self.representations.items()}


class PlatonicProjector:
    """
    Project embeddings to Platonic (anchor-based) representation.

    Uses cosine similarity between embeddings and anchor embeddings
    to create a unified representation space for cross-MLIP comparison.

    Example:
        >>> projector = PlatonicProjector(anchor_set)
        >>> platonic_rep = projector.project(embeddings, model_name='mace')
        >>> print(platonic_rep.coordinates.shape)  # (n_samples, n_anchors)
    """

    def __init__(
        self,
        anchor_set: AnchorSet,
        similarity_method: str = 'cosine',
        normalize: bool = True,
    ):
        """
        Initialize Platonic projector.

        Args:
            anchor_set: Set of anchor embeddings to project against
            similarity_method: 'cosine' or 'euclidean'
            normalize: Whether to normalize embeddings before projection
        """
        self.anchor_set = anchor_set
        self.similarity_method = similarity_method
        self.normalize = normalize
        self._anchor_tensor = None

    def project(
        self,
        embeddings: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        model_name: str = 'unknown',
        aggregate_atoms: bool = True,
        aggregation_method: str = 'mean',
    ) -> PlatonicRepresentation:
        """
        Project embeddings to Platonic representation.

        Args:
            embeddings: Input embeddings. Can be:
                - (n_samples, embedding_dim) array
                - List of per-atom embeddings [(n_atoms_i, dim), ...]
                - torch.Tensor
            model_name: Name of the MLIP model
            aggregate_atoms: If True, aggregate per-atom to per-structure
            aggregation_method: 'mean', 'sum', or 'max'

        Returns:
            PlatonicRepresentation with projected coordinates
        """
        # Handle different input types
        if isinstance(embeddings, list):
            if aggregate_atoms:
                embeddings = self._aggregate_atom_embeddings(
                    embeddings, method=aggregation_method
                )
            else:
                # Flatten all atoms
                embeddings = np.vstack(embeddings)

        embeddings = self._to_tensor(embeddings)
        anchors = self._get_anchor_tensor()

        # Compute Platonic representation
        if self.similarity_method == 'cosine':
            platonic_coords = self._cosine_projection(embeddings, anchors)
        elif self.similarity_method == 'euclidean':
            platonic_coords = self._euclidean_projection(embeddings, anchors)
        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")

        return PlatonicRepresentation(
            coordinates=platonic_coords.numpy(),
            model_name=model_name,
            anchor_set=self.anchor_set,
            original_embeddings=embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings,
        )

    def _cosine_projection(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Platonic representation using cosine similarity.

        This is the core projection method from the Platonic Rep paper.
        """
        if self.normalize:
            x_normalized = F.normalize(embeddings, p=2, dim=-1)
            anchors_normalized = F.normalize(anchors, p=2, dim=-1)
        else:
            x_normalized = embeddings
            anchors_normalized = anchors

        # Cosine similarity: (n_samples, n_anchors)
        platonic_rep = torch.einsum("bm, am -> ba", x_normalized, anchors_normalized)

        return platonic_rep

    def _euclidean_projection(
        self,
        embeddings: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Compute representation using negative euclidean distance."""
        # (n_samples, n_anchors)
        distances = torch.cdist(embeddings, anchors, p=2)
        # Convert to similarity (negative distance)
        return -distances

    def _aggregate_atom_embeddings(
        self,
        atom_embeddings: List[np.ndarray],
        method: str = 'mean',
    ) -> np.ndarray:
        """Aggregate per-atom embeddings to per-structure."""
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
                raise ValueError(f"Unknown aggregation: {method}")
        return np.stack(aggregated)

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert to torch tensor."""
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(x, dtype=torch.float32)

    def _get_anchor_tensor(self) -> torch.Tensor:
        """Get anchor embeddings as tensor (cached)."""
        if self._anchor_tensor is None:
            self._anchor_tensor = torch.tensor(
                self.anchor_set.embeddings, dtype=torch.float32
            )
        return self._anchor_tensor


class MultiModelProjector:
    """
    Project embeddings from multiple MLIPs to Platonic space.

    Manages projectors for each model and provides unified interface
    for cross-model analysis.

    Example:
        >>> projector = MultiModelProjector(anchor_set)
        >>> projector.add_model_embeddings('mace', mace_embeddings)
        >>> projector.add_model_embeddings('chgnet', chgnet_embeddings)
        >>> multi_rep = projector.project_all()
    """

    def __init__(
        self,
        anchor_set: AnchorSet,
        similarity_method: str = 'cosine',
    ):
        self.anchor_set = anchor_set
        self.similarity_method = similarity_method
        self._projector = PlatonicProjector(anchor_set, similarity_method)
        self._model_embeddings: Dict[str, np.ndarray] = {}

    def add_model_embeddings(
        self,
        model_name: str,
        embeddings: Union[np.ndarray, List[np.ndarray]],
    ):
        """Add embeddings from an MLIP model."""
        if isinstance(embeddings, list):
            embeddings = self._projector._aggregate_atom_embeddings(embeddings)
        self._model_embeddings[model_name] = np.asarray(embeddings)

    def project_all(
        self,
        structure_ids: Optional[List[str]] = None,
    ) -> MultiModelRepresentation:
        """Project all registered model embeddings."""
        representations = {}
        for model_name, embeddings in self._model_embeddings.items():
            rep = self._projector.project(embeddings, model_name=model_name)
            representations[model_name] = rep

        return MultiModelRepresentation(
            representations=representations,
            structure_ids=structure_ids,
        )

    def project_single_model(
        self,
        model_name: str,
        embeddings: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> PlatonicRepresentation:
        """Project embeddings for a single model."""
        if embeddings is None:
            embeddings = self._model_embeddings.get(model_name)
            if embeddings is None:
                raise ValueError(f"No embeddings registered for {model_name}")

        return self._projector.project(embeddings, model_name=model_name)


# Convenience functions

def calculate_platonic_representation(
    embeddings: Union[np.ndarray, List[np.ndarray]],
    anchor_embeddings: Union[np.ndarray, AnchorSet],
    model_name: str = 'unknown',
    similarity: str = 'cosine',
) -> np.ndarray:
    """
    Calculate Platonic representation using cosine similarity.

    This is a simple functional interface matching the notebook implementation.

    Args:
        embeddings: Structure embeddings (n_samples, dim) or list of per-atom
        anchor_embeddings: Anchor embeddings or AnchorSet
        model_name: Name of the model (for metadata)
        similarity: 'cosine' or 'euclidean'

    Returns:
        Platonic representation coordinates (n_samples, n_anchors)

    Example:
        >>> platonic_rep = calculate_platonic_representation(
        ...     embeddings, anchor_embeddings, model_name='mace'
        ... )
    """
    # Handle AnchorSet input
    if isinstance(anchor_embeddings, AnchorSet):
        anchors = torch.tensor(anchor_embeddings.embeddings, dtype=torch.float32)
    else:
        anchors = torch.tensor(anchor_embeddings, dtype=torch.float32)

    # Handle list of per-atom embeddings
    if isinstance(embeddings, list):
        embeddings = np.stack([
            np.mean(e, axis=0) if np.asarray(e).ndim > 1 else np.asarray(e)
            for e in embeddings
        ])

    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    if similarity == 'cosine':
        x_normalized = F.normalize(embeddings, p=2, dim=-1)
        anchors_normalized = F.normalize(anchors, p=2, dim=-1)
        platonic_rep = torch.einsum("bm, am -> ba", x_normalized, anchors_normalized)
    else:
        platonic_rep = -torch.cdist(embeddings, anchors, p=2)

    return platonic_rep.numpy()


def project_to_anchor_space(
    embeddings: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """
    Simple projection to anchor space using cosine similarity.

    Matches the exact implementation from PlatonicRep.ipynb.

    Args:
        embeddings: (n_samples, embedding_dim)
        anchors: (n_anchors, embedding_dim)

    Returns:
        (n_samples, n_anchors) similarity matrix
    """
    embeddings = torch.tensor(embeddings, dtype=torch.float64)
    anchors = torch.tensor(anchors, dtype=torch.float64)

    x_normalized = F.normalize(embeddings, p=2, dim=-1)
    anchors_normalized = F.normalize(anchors, p=2, dim=-1)
    relative_embeddings = torch.einsum("bm, am -> ba", x_normalized, anchors_normalized)

    return relative_embeddings.numpy()
