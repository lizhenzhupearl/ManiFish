"""
Orb model embedding extractor.

Orb is a universal interatomic potential for materials modeling.

Installation:
    pip install orb-models
"""

from typing import List, Optional, Union
import numpy as np
from ase import Atoms
import torch

from .base import BaseEmbeddingExtractor


class OrbEmbeddingExtractor(BaseEmbeddingExtractor):
    """
    Extract embeddings from Orb models.

    Orb provides final node features from the GNN that capture
    rich structural information.

    Example:
        >>> extractor = OrbEmbeddingExtractor(model='orb-v3', device='cuda')
        >>> extractor.load_model()
        >>> embeddings = extractor.get_embeddings(structures)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: str = 'orb_v3_direct_inf_omat',
        device: str = 'cpu',
        precision: str = 'float32-high',
    ):
        """
        Initialize Orb embedding extractor.

        Args:
            model_path: Path to custom model (not typically used for Orb)
            model: Pretrained model name (e.g., 'orb_v3_direct_inf_omat')
            device: Device to run on ('cpu' or 'cuda')
            precision: Precision setting for the model
        """
        super().__init__(model_path=model_path, device=device)
        self.model_name = model
        self.precision = precision
        self._gns_model = None
        self._system_config = None

    def load_model(self):
        """Load Orb model."""
        try:
            from orb_models.forcefield import pretrained
        except ImportError:
            raise ImportError(
                "orb-models not installed. Install with: pip install orb-models"
            )

        # Get the pretrained model loader function
        model_loader = getattr(pretrained, self.model_name, None)
        if model_loader is None:
            raise ValueError(f"Unknown Orb model: {self.model_name}")

        model = model_loader(device=self.device, precision=self.precision)
        self._gns_model = model.model
        self._system_config = model.system_config
        self._gns_model.eval()
        self._is_loaded = True

    def get_embeddings(
        self,
        structures: Union[Atoms, List[Atoms]],
    ) -> List[np.ndarray]:
        """
        Extract final node embeddings from Orb.

        Args:
            structures: ASE Atoms or list of Atoms

        Returns:
            List of arrays, each (n_atoms, embedding_dim)
        """
        if not self._is_loaded:
            self.load_model()

        try:
            from orb_models.forcefield import atomic_system
        except ImportError:
            raise ImportError("orb-models not installed")

        if isinstance(structures, Atoms):
            structures = [structures]

        embeddings = []

        with torch.no_grad():
            for atoms in structures:
                # Convert to Orb graph format
                graph = atomic_system.ase_atoms_to_atom_graphs(
                    atoms, self._system_config, device=self.device
                )

                # Forward pass to get final embeddings
                result = self._gns_model(graph)
                final_node_embeddings = result["node_features"]
                embeddings.append(final_node_embeddings.cpu().numpy())

        return embeddings

    def get_atomic_numbers(
        self,
        structures: Union[Atoms, List[Atoms]],
    ) -> List[np.ndarray]:
        """Get atomic numbers for each structure."""
        if isinstance(structures, Atoms):
            structures = [structures]
        return [atoms.get_atomic_numbers() for atoms in structures]
