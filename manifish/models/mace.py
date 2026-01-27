"""
MACE embedding extractor.

MACE (Message-passing Atomic Cluster Expansion) is an equivariant
graph neural network for molecular dynamics.

Installation:
    pip install mace-torch
"""

from typing import List, Optional, Union
import numpy as np
from ase import Atoms
from ase.io import read, write
import tempfile
import os

from .base import BaseEmbeddingExtractor


class MACEEmbeddingExtractor(BaseEmbeddingExtractor):
    """
    Extract embeddings from MACE models.

    MACE provides node-level descriptors that capture local atomic
    environments through message passing.

    Example:
        >>> extractor = MACEEmbeddingExtractor(model='medium', device='cpu')
        >>> extractor.load_model()
        >>> embeddings = extractor.get_embeddings(structures)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: str = 'medium',
        device: str = 'cpu',
        num_layers: int = 1,
    ):
        """
        Initialize MACE embedding extractor.

        Args:
            model_path: Path to custom MACE model file
            model: Pretrained model name ('small', 'medium', 'large')
                   Only used if model_path is None
            device: Device to run on ('cpu' or 'cuda')
            num_layers: Number of layers for descriptor extraction
        """
        super().__init__(model_path=model_path, device=device)
        self.model_name = model
        self.num_layers = num_layers
        self._calculator = None

    def load_model(self):
        """Load MACE model and create calculator."""
        try:
            from mace.calculators import MACECalculator, mace_mp
        except ImportError:
            raise ImportError(
                "MACE not installed. Install with: pip install mace-torch"
            )

        if self.model_path is not None:
            self._calculator = MACECalculator(
                model_paths=self.model_path,
                device=self.device,
            )
        else:
            # Use pretrained Materials Project model
            self._calculator = mace_mp(
                model=self.model_name,
                device=self.device,
            )

        self._is_loaded = True

    def get_embeddings(
        self,
        structures: Union[Atoms, List[Atoms]],
        num_layers: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Extract MACE descriptors from structures.

        Args:
            structures: ASE Atoms or list of Atoms
            num_layers: Override default num_layers for descriptor extraction

        Returns:
            List of arrays, each (n_atoms, descriptor_dim)
        """
        if not self._is_loaded:
            self.load_model()

        if isinstance(structures, Atoms):
            structures = [structures]

        num_layers = num_layers or self.num_layers
        embeddings = []

        for atoms in structures:
            # MACE requires proper periodic structure - convert via CIF
            with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as f:
                temp_path = f.name

            try:
                atoms.write(temp_path, format='cif')
                conf = read(temp_path, format='cif')
                descriptors = self._calculator.get_descriptors(
                    conf, num_layers=num_layers
                )
                embeddings.append(np.asarray(descriptors))
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        return embeddings

    def get_atomic_numbers(
        self,
        structures: Union[Atoms, List[Atoms]],
    ) -> List[np.ndarray]:
        """Get atomic numbers for each structure."""
        if isinstance(structures, Atoms):
            structures = [structures]
        return [atoms.get_atomic_numbers() for atoms in structures]
