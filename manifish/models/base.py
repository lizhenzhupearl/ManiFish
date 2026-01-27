"""
Base class for MLIP embedding extractors.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
from ase import Atoms


class BaseEmbeddingExtractor(ABC):
    """
    Abstract base class for MLIP embedding extractors.

    All MLIP-specific extractors should inherit from this class
    and implement the required methods.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize the embedding extractor.

        Args:
            model_path: Path to model checkpoint or model name
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self._is_loaded = False

    @property
    def name(self) -> str:
        """Return the name of this MLIP."""
        return self.__class__.__name__.replace('EmbeddingExtractor', '')

    @abstractmethod
    def load_model(self):
        """Load the MLIP model."""
        pass

    @abstractmethod
    def get_embeddings(
        self,
        structures: Union[Atoms, List[Atoms]],
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Extract embeddings from structures.

        Args:
            structures: Single ASE Atoms or list of Atoms objects

        Returns:
            List of embedding arrays, one per structure.
            Each array has shape (n_atoms, embedding_dim) for per-atom
            embeddings, or (embedding_dim,) for structure-level.
        """
        pass

    def get_structure_embeddings(
        self,
        structures: Union[Atoms, List[Atoms]],
        aggregation: str = 'mean',
        **kwargs,
    ) -> np.ndarray:
        """
        Extract structure-level embeddings (aggregated from atom embeddings).

        Args:
            structures: Single ASE Atoms or list of Atoms objects
            aggregation: 'mean', 'sum', or 'max'

        Returns:
            Array of shape (n_structures, embedding_dim)
        """
        atom_embeddings = self.get_embeddings(structures, **kwargs)

        aggregated = []
        for emb in atom_embeddings:
            emb = np.asarray(emb)
            if emb.ndim == 1:
                aggregated.append(emb)
            elif aggregation == 'mean':
                aggregated.append(np.mean(emb, axis=0))
            elif aggregation == 'sum':
                aggregated.append(np.sum(emb, axis=0))
            elif aggregation == 'max':
                aggregated.append(np.max(emb, axis=0))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        return np.stack(aggregated)

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"{self.name}EmbeddingExtractor(device='{self.device}', {status})"
