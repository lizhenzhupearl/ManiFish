"""
SevenNet embedding extractor.

SevenNet is an equivariant graph neural network potential.

Installation:
    pip install sevenn
"""

from typing import List, Optional, Union
import numpy as np
from ase import Atoms
import ase.io
import torch
from tqdm import tqdm

from .base import BaseEmbeddingExtractor


class SevenNetEmbeddingExtractor(BaseEmbeddingExtractor):
    """
    Extract embeddings from SevenNet models.

    Uses forward hooks to capture final node embeddings before
    the energy prediction layer.

    Example:
        >>> extractor = SevenNetEmbeddingExtractor(model='7net-omat', device='cpu')
        >>> extractor.load_model()
        >>> embeddings = extractor.get_embeddings(structures)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: str = '7net-omat',
        device: str = 'cpu',
        cutoff: float = 5.0,
        modal: Optional[str] = None,
    ):
        """
        Initialize SevenNet embedding extractor.

        Args:
            model_path: Path to SevenNet checkpoint
            model: Model name (e.g., '7net-omat', '7net-mf-ompa')
            device: Device to run on
            cutoff: Cutoff distance for neighbor calculation
            modal: Modal for multi-fidelity models (e.g., 'omat24', 'mpa')
        """
        super().__init__(model_path=model_path or model, device=device)
        self.model_name = model
        self.cutoff = cutoff
        self.modal = modal
        self._hooks = []
        self._embeddings_cache = {}

    def load_model(self):
        """Load SevenNet model and register embedding hooks."""
        try:
            from sevenn.util import model_from_checkpoint
        except ImportError:
            raise ImportError(
                "SevenNet not installed. Install with: pip install sevenn"
            )

        self.model, self.config = model_from_checkpoint(self.model_path)
        self.model.eval()

        # Set modal if specified
        if self.modal and hasattr(self.model, 'modal_map') and self.model.modal_map:
            self._modal = self.modal
        else:
            self._modal = None

        self._register_hooks()
        self._is_loaded = True

    def _register_hooks(self):
        """Register forward hooks to capture final embeddings."""
        layer_names = list(self.model._modules.keys())

        # Find layer before energy readout
        readout_layers = ['reduce_input_to_hidden', 'readout_FCN']
        target_layer = None

        for readout_layer in readout_layers:
            if readout_layer in layer_names:
                idx = layer_names.index(readout_layer)
                if idx > 0:
                    target_layer = layer_names[idx - 1]
                break

        if target_layer is None:
            # Fallback: use last convolution-related layer
            for name in reversed(layer_names):
                if 'equivariant_gate' in name or 'convolution' in name:
                    target_layer = name
                    break

        if target_layer is None:
            raise ValueError("Could not find appropriate layer for embeddings")

        def hook_fn(module, input, output):
            if hasattr(output, 'node_feature'):
                emb = output.node_feature
                self._embeddings_cache['final'] = emb.clone().detach()
            elif hasattr(output, 'x'):
                emb = output.x
                self._embeddings_cache['final'] = emb.clone().detach()

        hook = self.model._modules[target_layer].register_forward_hook(hook_fn)
        self._hooks.append(hook)

    def _atoms_to_graph(self, atoms: Atoms):
        """Convert ASE Atoms to SevenNet graph format."""
        try:
            from sevenn.util import unlabeled_atoms_to_input
        except ImportError:
            raise ImportError("SevenNet not installed")

        return unlabeled_atoms_to_input(atoms, self.cutoff)

    def get_embeddings(
        self,
        structures: Union[Atoms, List[Atoms]],
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        Extract final node embeddings from SevenNet.

        Args:
            structures: ASE Atoms or list of Atoms
            show_progress: Show tqdm progress bar

        Returns:
            List of arrays, each (n_atoms, embedding_dim)
        """
        if not self._is_loaded:
            self.load_model()

        try:
            import sevenn._keys as KEY
        except ImportError:
            raise ImportError("SevenNet not installed")

        if isinstance(structures, Atoms):
            structures = [structures]

        self.model.eval()
        self.model.set_is_batch_data(False)

        embeddings = []
        iterator = tqdm(structures, desc="Extracting embeddings") if show_progress else structures

        for atoms in iterator:
            self._embeddings_cache.clear()

            graph_data = self._atoms_to_graph(atoms)
            if graph_data is None:
                embeddings.append(None)
                continue

            with torch.no_grad():
                data = self.model._preprocess(graph_data)

                # Process through modules until force_output
                for name, module in self.model._modules.items():
                    if name == 'force_output':
                        break
                    data = module(data)

            emb = self._embeddings_cache.get('final', None)
            if emb is not None:
                embeddings.append(emb.cpu().numpy())
            else:
                embeddings.append(None)

        return embeddings

    def get_atomic_numbers(
        self,
        structures: Union[Atoms, List[Atoms]],
    ) -> List[np.ndarray]:
        """Get atomic numbers for each structure."""
        if isinstance(structures, Atoms):
            structures = [structures]
        return [atoms.get_atomic_numbers() for atoms in structures]

    def cleanup(self):
        """Remove forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.cleanup()
