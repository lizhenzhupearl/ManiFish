"""
Unified MLIP Embedding Extractor

Extract atom-level embeddings from various foundation MLIPs:
- MACE (mace-mp, mace-off, etc.)
- ORB (orb-v3, etc.)
- SevenNet (7net-omat, 7net-mf-ompa, etc.)
- NequIP/Allegro

Supports multiple input formats:
- CIF files
- Extended XYZ files
- JSON with structure data (e.g., Materials Project format)
- ASE Atoms objects directly

Usage:
    from mlip_embedding_extractor import MLIPEmbeddingExtractor

    extractor = MLIPEmbeddingExtractor(
        model_type="mace",
        model_path="/path/to/mace-mpa-0-medium.model",
        device="cuda:0"
    )

    results = extractor.extract_from_files(
        file_paths=["struct1.cif", "struct2.cif"],
        material_ids=["mp-1", "mp-2"]
    )

    extractor.save_results(results, output_dir="embeddings/")
"""

import os
import json
import pickle
import warnings
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from tqdm import tqdm

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EmbeddingResult:
    """Container for embedding extraction results."""
    embeddings: List[np.ndarray]  # List of (n_atoms_i, embed_dim) arrays
    atomic_numbers: List[np.ndarray]  # List of atomic number arrays
    material_ids: List[str]  # Material IDs for each structure

    # Flat versions for convenience
    flat_embeddings: Optional[np.ndarray] = None  # (total_atoms, embed_dim)
    flat_atomic_numbers: Optional[np.ndarray] = None  # (total_atoms,)
    flat_material_ids: Optional[List[str]] = None  # (total_atoms,) material ID per atom

    # Metadata
    model_type: str = ""
    model_path: str = ""
    embed_dim: int = 0
    n_structures: int = 0
    n_atoms_total: int = 0

    # Layer information
    layer_name: str = ""  # Which layer embeddings were extracted from
    layer_index: int = -1  # Layer index (-1 = last layer)

    def __post_init__(self):
        """Compute flat versions and metadata."""
        self.n_structures = len(self.embeddings)

        if self.n_structures > 0:
            # Get embedding dimension from first non-empty embedding
            for emb in self.embeddings:
                if emb is not None and len(emb) > 0:
                    self.embed_dim = emb.shape[-1]
                    break

        # Create flat versions
        valid_emb = [e for e in self.embeddings if e is not None]
        valid_an = [a for a in self.atomic_numbers if a is not None]

        if valid_emb:
            self.flat_embeddings = np.vstack(valid_emb)
            self.flat_atomic_numbers = np.concatenate(valid_an)
            self.n_atoms_total = len(self.flat_embeddings)

            # Create flat material IDs (one per atom)
            self.flat_material_ids = []
            for i, (emb, mid) in enumerate(zip(self.embeddings, self.material_ids)):
                if emb is not None:
                    self.flat_material_ids.extend([mid] * len(emb))


@dataclass
class StructureInput:
    """Normalized structure input."""
    atoms: Atoms
    material_id: str
    source_path: Optional[str] = None


# =============================================================================
# Structure Reader
# =============================================================================

class StructureReader:
    """Read structures from various file formats into ASE Atoms objects."""

    @staticmethod
    def read_cif(path: str, material_id: Optional[str] = None) -> StructureInput:
        """Read a CIF file."""
        atoms = ase_read(path, format='cif')
        mid = material_id or Path(path).stem
        return StructureInput(atoms=atoms, material_id=mid, source_path=path)

    @staticmethod
    def read_extxyz(path: str, index: int = 0, material_id: Optional[str] = None) -> StructureInput:
        """Read an extended XYZ file."""
        atoms = ase_read(path, format='extxyz', index=index)
        mid = material_id or f"{Path(path).stem}_{index}"
        return StructureInput(atoms=atoms, material_id=mid, source_path=path)

    @staticmethod
    def read_extxyz_all(path: str) -> List[StructureInput]:
        """Read all structures from an extended XYZ file."""
        all_atoms = ase_read(path, format='extxyz', index=':')
        if not isinstance(all_atoms, list):
            all_atoms = [all_atoms]

        results = []
        stem = Path(path).stem
        for i, atoms in enumerate(all_atoms):
            mid = f"{stem}_{i}"
            results.append(StructureInput(atoms=atoms, material_id=mid, source_path=path))
        return results

    @staticmethod
    def read_cif_string(cif_string: str, material_id: str) -> StructureInput:
        """Read a CIF from a string."""
        atoms = ase_read(StringIO(cif_string), format='cif')
        return StructureInput(atoms=atoms, material_id=material_id)

    @staticmethod
    def read_mp_json(path: str) -> List[StructureInput]:
        """
        Read structures from Materials Project style JSON.

        Expected format:
        [
            {
                "material_id": "mp-123",  # optional
                "lattice": {"matrix": [[...], [...], [...]]},
                "sites": [
                    {"species": [{"element": "Li"}], "xyz": [x, y, z]},
                    ...
                ]
            },
            ...
        ]
        """
        with open(path, 'r') as f:
            data = json.load(f)

        results = []
        for i, entry in enumerate(data):
            lattice = entry['lattice']
            sites = entry['sites']

            symbols = [site['species'][0]['element'] for site in sites]
            positions = [site['xyz'] for site in sites]
            cell = lattice['matrix']

            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

            # Use material_id from entry if available, otherwise use index
            mid = entry.get('material_id', str(i))
            results.append(StructureInput(atoms=atoms, material_id=mid, source_path=path))

        return results

    @staticmethod
    def read_csv_with_cif(path: str, cif_column: str = 'cif',
                          id_column: Optional[str] = None) -> List[StructureInput]:
        """
        Read structures from a CSV file with CIF strings in a column.

        Args:
            path: Path to CSV file
            cif_column: Name of column containing CIF strings
            id_column: Name of column containing material IDs (optional)
        """
        import pandas as pd
        df = pd.read_csv(path)

        results = []
        for i, row in df.iterrows():
            cif_string = row[cif_column]
            mid = row[id_column] if id_column and id_column in df.columns else str(i)

            try:
                atoms = ase_read(StringIO(cif_string), format='cif')
                results.append(StructureInput(atoms=atoms, material_id=str(mid), source_path=path))
            except Exception as e:
                warnings.warn(f"Failed to read structure {i}: {e}")
                results.append(StructureInput(atoms=None, material_id=str(mid), source_path=path))

        return results

    @staticmethod
    def from_atoms(atoms: Atoms, material_id: str) -> StructureInput:
        """Create input from an existing ASE Atoms object."""
        return StructureInput(atoms=atoms, material_id=material_id)

    @staticmethod
    def auto_read(path: str, **kwargs) -> Union[StructureInput, List[StructureInput]]:
        """
        Automatically detect file format and read structures.

        Returns single StructureInput for single-structure files (CIF),
        or List[StructureInput] for multi-structure files (JSON, CSV, XYZ).
        """
        path = str(path)
        ext = Path(path).suffix.lower()

        if ext == '.cif':
            return StructureReader.read_cif(path, **kwargs)
        elif ext in ['.xyz', '.extxyz']:
            return StructureReader.read_extxyz_all(path)
        elif ext == '.json':
            return StructureReader.read_mp_json(path)
        elif ext == '.csv':
            return StructureReader.read_csv_with_cif(path, **kwargs)
        else:
            # Try ASE's generic reader
            try:
                atoms = ase_read(path)
                mid = kwargs.get('material_id', Path(path).stem)
                return StructureInput(atoms=atoms, material_id=mid, source_path=path)
            except Exception as e:
                raise ValueError(f"Unknown file format: {ext}. Error: {e}")


# =============================================================================
# Base Extractor Class
# =============================================================================

class BaseEmbeddingExtractor(ABC):
    """Abstract base class for MLIP embedding extractors."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load the model. Implemented by subclasses."""
        pass

    @abstractmethod
    def extract_single(self, atoms: Atoms) -> Optional[np.ndarray]:
        """
        Extract embeddings for a single structure.

        Args:
            atoms: ASE Atoms object

        Returns:
            Embeddings array of shape (n_atoms, embed_dim), or None if failed
        """
        pass

    def extract_batch(self, structures: List[StructureInput],
                      show_progress: bool = True) -> EmbeddingResult:
        """
        Extract embeddings for a batch of structures.

        Args:
            structures: List of StructureInput objects
            show_progress: Whether to show progress bar

        Returns:
            EmbeddingResult containing all embeddings
        """
        embeddings = []
        atomic_numbers = []
        material_ids = []

        iterator = tqdm(structures, desc="Extracting embeddings") if show_progress else structures

        for struct in iterator:
            if struct.atoms is None:
                embeddings.append(None)
                atomic_numbers.append(None)
                material_ids.append(struct.material_id)
                continue

            try:
                emb = self.extract_single(struct.atoms)
                embeddings.append(emb)
                atomic_numbers.append(np.array(struct.atoms.get_atomic_numbers()))
                material_ids.append(struct.material_id)
            except Exception as e:
                warnings.warn(f"Failed to extract embeddings for {struct.material_id}: {e}")
                embeddings.append(None)
                atomic_numbers.append(None)
                material_ids.append(struct.material_id)

        # Get layer info from subclass
        layer_name, layer_index = self._get_layer_info()

        result = EmbeddingResult(
            embeddings=embeddings,
            atomic_numbers=atomic_numbers,
            material_ids=material_ids,
            model_type=self.__class__.__name__,
            model_path=self.model_path,
            layer_name=layer_name,
            layer_index=layer_index,
        )

        return result

    def _get_layer_info(self) -> Tuple[str, int]:
        """Get layer name and index. Override in subclasses."""
        return "", -1

    def cleanup(self):
        """Clean up resources. Override in subclasses if needed."""
        pass


# =============================================================================
# MACE Extractor
# =============================================================================

class MACEExtractor(BaseEmbeddingExtractor):
    """
    Extract embeddings from MACE models.

    Uses the official MACECalculator's get_descriptors method.

    Layer Selection Guidance:
    -------------------------
    - num_layers=0: After initial embedding (before any message passing)
    - num_layers=1: After first interaction block (RECOMMENDED)
                    Captures local chemical environment before MLP distortion
    - num_layers=2+: After subsequent interaction blocks
    - num_layers=-1: Last layer before readout (may be distorted by MLP)

    For manifold analysis, num_layers=1 is recommended because:
    - MLP layers can distort the geometric structure of the manifold
    - Early layers preserve more of the underlying chemical space geometry
    - Better separation of different chemical environments

    Args:
        model_path: Path to MACE model file (.model)
        device: Device to run on ("cuda", "cuda:0", "cpu")
        num_layers: Which layer's descriptors to extract
                   1 = first interaction layer (recommended for manifold analysis)
                   -1 = last layer (default, but may be distorted)
    """

    def __init__(self, model_path: str, device: str = "cuda", num_layers: int = 1):
        self.num_layers = num_layers
        super().__init__(model_path, device)

    def _load_model(self):
        from mace.calculators import MACECalculator

        # Handle device specification
        if "cuda" in self.device:
            device_id = self.device.split(":")[-1] if ":" in self.device else "0"
            torch.cuda.set_device(int(device_id))
            device = "cuda"
        else:
            device = "cpu"

        self.calculator = MACECalculator(
            model_paths=self.model_path,
            device=device
        )

    def extract_single(self, atoms: Atoms) -> Optional[np.ndarray]:
        """Extract descriptors using MACE's built-in method."""
        descriptors = self.calculator.get_descriptors(atoms, num_layers=self.num_layers)
        return np.array(descriptors)

    def _get_layer_info(self) -> Tuple[str, int]:
        """Return MACE layer info."""
        layer_name = f"mace_layer_{self.num_layers}" if self.num_layers >= 0 else "mace_final_layer"
        return layer_name, self.num_layers


# =============================================================================
# ORB Extractor
# =============================================================================

class ORBExtractor(BaseEmbeddingExtractor):
    """
    Extract embeddings from ORB models (orbital-materials).

    Layer Selection (TODO - needs confirmation):
    ---------------------------------------------
    ORB uses a GNN architecture. Currently extracts final node_features.

    Options to explore:
    - node_features after specific message passing layers
    - edge_features for bond-level analysis
    - Intermediate graph representations

    For manifold analysis, we may want embeddings BEFORE the final
    energy/force prediction heads to avoid task-specific distortion.

    Args:
        model_path: Model name (e.g., "orb_v3_direct_inf_omat") or path
        device: Device to run on
        precision: Precision mode ("float32-high", "float32", etc.)
        extract_layer: Which layer to extract from (TODO: implement options)
                      "final" = final node features (current default)
    """

    def __init__(self, model_path: str = "orb_v3_direct_inf_omat",
                 device: str = "cuda", precision: str = "float32-high",
                 extract_layer: str = "final"):
        self.precision = precision
        self.extract_layer = extract_layer
        super().__init__(model_path, device)

    def _load_model(self):
        from orb_models.forcefield import pretrained, atomic_system

        self.atomic_system = atomic_system

        # Load pretrained model by name
        model_loaders = {
            "orb_v3_direct_inf_omat": pretrained.orb_v3_direct_inf_omat,
            "orb_v3_conservative_inf_omat": pretrained.orb_v3_conservative_inf_omat,
            # Add more as needed
        }

        if self.model_path in model_loaders:
            self.model = model_loaders[self.model_path](
                device=self.device,
                precision=self.precision
            )
        else:
            # Try loading as a path
            try:
                self.model = pretrained.orb_v3_direct_inf_omat(
                    device=self.device,
                    precision=self.precision
                )
                warnings.warn(f"Model path '{self.model_path}' not recognized, "
                            f"using default orb_v3_direct_inf_omat")
            except Exception as e:
                raise ValueError(f"Could not load ORB model: {e}")

        self.gns_model = self.model.model

    def extract_single(self, atoms: Atoms) -> Optional[np.ndarray]:
        """Extract node features from ORB model."""
        with torch.no_grad():
            graph = self.atomic_system.ase_atoms_to_atom_graphs(
                atoms,
                self.model.system_config,
                device=self.device
            )
            result = self.gns_model(graph)
            node_embeddings = result["node_features"]
            return node_embeddings.cpu().numpy()

    def _get_layer_info(self) -> Tuple[str, int]:
        """Return ORB layer info."""
        return f"orb_{self.extract_layer}_node_features", -1


# =============================================================================
# SevenNet Extractor
# =============================================================================

class SevenNetExtractor(BaseEmbeddingExtractor):
    """
    Extract embeddings from SevenNet models.

    Uses forward hooks to capture embeddings before the readout layer.

    Layer Selection (TODO - needs confirmation):
    ---------------------------------------------
    SevenNet uses equivariant convolution layers. Currently hooks capture
    embeddings just before the readout/energy prediction layer.

    The hook targets layers containing 'equivariant_gate' or 'convolution'.
    For manifold analysis, we want embeddings that:
    - Have undergone sufficient message passing for chemical context
    - Are NOT yet processed by MLP heads for energy prediction

    Options to explore:
    - After specific convolution layer (layer_index parameter)
    - Before vs after equivariant gate operations

    Args:
        model_path: Model name (e.g., "7net-omat") or checkpoint path
        device: Device to run on
        modal: Modal for multi-fidelity models (e.g., "omat24", "mpa")
        cutoff: Neighbor list cutoff (default: 5.0)
        target_layer: Which layer to hook (TODO: implement options)
                     "pre_readout" = just before energy prediction (current default)
    """

    def __init__(self, model_path: str = "7net-omat", device: str = "cuda",
                 modal: Optional[str] = None, cutoff: float = 5.0,
                 target_layer: str = "pre_readout"):
        self.modal = modal
        self.cutoff = cutoff
        self.target_layer = target_layer
        self.hooks = []
        self.embeddings_cache = {}
        super().__init__(model_path, device)

    def _load_model(self):
        from sevenn.util import model_from_checkpoint, unlabeled_atoms_to_input
        import sevenn._keys as KEY

        self.KEY = KEY
        self.unlabeled_atoms_to_input = unlabeled_atoms_to_input

        self.model, self.config = model_from_checkpoint(self.model_path)
        self.model.eval()

        if self.modal and hasattr(self.model, 'modal_map') and self.model.modal_map:
            self.model.modal = self.modal

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture embeddings."""
        layer_names = list(self.model._modules.keys())

        # Find the layer before readout
        readout_layers = ['reduce_input_to_hidden', 'readout_FCN']
        target_layer = None

        for readout_layer in readout_layers:
            if readout_layer in layer_names:
                idx = layer_names.index(readout_layer)
                if idx > 0:
                    target_layer = layer_names[idx - 1]
                break

        if target_layer is None:
            # Fallback: use last convolution layer
            for name in reversed(layer_names):
                if 'equivariant_gate' in name or 'convolution' in name:
                    target_layer = name
                    break

        if target_layer:
            def hook_fn(module, input, output):
                if hasattr(output, 'node_feature'):
                    self.embeddings_cache['final'] = output.node_feature.clone().detach()
                elif hasattr(output, 'x'):
                    self.embeddings_cache['final'] = output.x.clone().detach()

            hook = self.model._modules[target_layer].register_forward_hook(hook_fn)
            self.hooks.append(hook)
        else:
            raise ValueError("Could not find appropriate layer for embeddings")

    def extract_single(self, atoms: Atoms) -> Optional[np.ndarray]:
        """Extract embeddings from SevenNet."""
        self.embeddings_cache.clear()
        self.model.set_is_batch_data(False)

        graph_data = self.unlabeled_atoms_to_input(atoms, self.cutoff)

        with torch.no_grad():
            data = self.model._preprocess(graph_data)

            for name, module in self.model._modules.items():
                if name == 'force_output':
                    break
                data = module(data)

        emb = self.embeddings_cache.get('final')
        if emb is not None:
            return emb.cpu().numpy()
        return None

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _get_layer_info(self) -> Tuple[str, int]:
        """Return SevenNet layer info."""
        return f"sevennet_{self.target_layer}", -1


# =============================================================================
# NequIP/Allegro Extractor
# =============================================================================

class NequIPExtractor(BaseEmbeddingExtractor):
    """
    Extract embeddings from NequIP/Allegro models.

    Inserts SaveForOutput modules to capture embeddings at various layers.

    Layer Selection Guidance:
    -------------------------
    NequIP/Allegro models have multiple convolution layers. The user's code
    shows they extract from layers containing "layer" and "convnet".

    For manifold analysis:
    - Early convolution layers: More local, less context
    - Middle layers: Good balance of local + neighborhood context
    - Late layers (before MLP): Full context, but may start to be task-specific
    - After MLP: Distorted for energy/force prediction (avoid)

    The `layers_to_save` parameter controls which layers get SaveForOutput hooks:
    - "layer": All convolution layers (recommended, gives you options)
    - "type_embed": Initial type embedding only
    - "all": Both type_embed and all layers

    The actual returned embedding is from the LAST saved layer by default.
    Use `extract_single_all_layers()` to get embeddings from all saved layers.

    Args:
        model_path: Path to .nequip.zip model package
        device: Device to run on
        r_max: Neighbor list cutoff (default: 7.0)
        layers_to_save: Which layers to save embeddings from
                       "layer" = convolution layers (recommended)
                       "type_embed" = initial embedding only
                       "all" = both type_embed and layers
        return_layer_index: Which saved layer to return (-1 = last, 0 = first)
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 r_max: float = 7.0, layers_to_save: str = "layer",
                 return_layer_index: int = -1):
        self.r_max = r_max
        self.layers_to_save = layers_to_save
        self.return_layer_index = return_layer_index
        super().__init__(model_path, device)

    def _load_model(self):
        from nequip.model import ModelFromPackage
        from nequip.nn.misc import SaveForOutput
        from nequip.nn._graph_mixin import SequentialGraphNetwork
        from nequip.nn.graph_model import GraphModel
        from nequip.data import AtomicDataDict, from_ase
        from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform

        self.AtomicDataDict = AtomicDataDict
        self.from_ase = from_ase
        self.ChemicalSpeciesToAtomTypeMapper = ChemicalSpeciesToAtomTypeMapper
        self.NeighborListTransform = NeighborListTransform

        # Load model
        model = ModelFromPackage(self.model_path)
        graph_model = model['sole_model']

        # Get type names from metadata
        model_type_names_str = graph_model.metadata.get('type_names', '')
        self.model_type_names = model_type_names_str.split(' ') if model_type_names_str else []

        # Get sequential network
        force_stress_wrapper = graph_model.model
        sequential_net = force_stress_wrapper.func

        # Insert SaveForOutput modules
        modified_modules = {}
        self.saved_layer_names = []

        for name, module in sequential_net.named_children():
            modified_modules[name] = module

            # Determine if we should save this layer
            should_save = False
            if self.layers_to_save == "all":
                should_save = "type_embed" in name or "layer" in name
            elif self.layers_to_save == "layer":
                should_save = "layer" in name and "convnet" in name
            elif self.layers_to_save == "type_embed":
                should_save = "type_embed" in name

            if should_save:
                save_module = SaveForOutput(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=f"saved_{name}_embeddings",
                    irreps_in=module.irreps_out
                )
                modified_modules[f"save_{name}"] = save_module
                self.saved_layer_names.append(f"saved_{name}_embeddings")

        # Rebuild model
        self.model = GraphModel(SequentialGraphNetwork(modified_modules))
        self.model = self.model.to(self.device)
        self.model.eval()

    def extract_single(self, atoms: Atoms) -> Optional[np.ndarray]:
        """Extract embeddings from NequIP/Allegro."""
        data = self.from_ase(atoms)

        transforms = [
            self.ChemicalSpeciesToAtomTypeMapper(self.model_type_names),
            self.NeighborListTransform(r_max=self.r_max)
        ]

        for transform in transforms:
            data = transform(data)

        data = self.AtomicDataDict.to_(data, self.device)

        with torch.no_grad():
            output = self.model(data)

        # Get embeddings from specified layer index
        if self.saved_layer_names:
            target_key = self.saved_layer_names[self.return_layer_index]
            if target_key in output:
                return output[target_key].cpu().numpy()

        # Fallback: try any saved layer
        for key in reversed(self.saved_layer_names):
            if key in output:
                return output[key].cpu().numpy()

        return None

    def extract_single_all_layers(self, atoms: Atoms) -> Dict[str, np.ndarray]:
        """Extract embeddings from all saved layers."""
        data = self.from_ase(atoms)

        transforms = [
            self.ChemicalSpeciesToAtomTypeMapper(self.model_type_names),
            self.NeighborListTransform(r_max=self.r_max)
        ]

        for transform in transforms:
            data = transform(data)

        data = self.AtomicDataDict.to_(data, self.device)

        with torch.no_grad():
            output = self.model(data)

        result = {}
        for key in self.saved_layer_names:
            if key in output:
                result[key] = output[key].cpu().numpy()

        return result

    def _get_layer_info(self) -> Tuple[str, int]:
        """Return NequIP layer info."""
        if self.saved_layer_names:
            layer_name = self.saved_layer_names[self.return_layer_index]
            actual_index = self.return_layer_index if self.return_layer_index >= 0 else len(self.saved_layer_names) + self.return_layer_index
            return layer_name, actual_index
        return "nequip_final", -1


# =============================================================================
# Unified Interface
# =============================================================================

class MLIPEmbeddingExtractor:
    """
    Unified interface for extracting embeddings from multiple MLIP architectures.

    Supports:
    - MACE: model_type="mace"
    - ORB: model_type="orb"
    - SevenNet: model_type="sevennet"
    - NequIP/Allegro: model_type="nequip"

    Example:
        >>> extractor = MLIPEmbeddingExtractor(
        ...     model_type="mace",
        ...     model_path="/path/to/mace-mpa-0-medium.model",
        ...     device="cuda:0"
        ... )
        >>> results = extractor.extract_from_files(
        ...     ["struct1.cif", "struct2.cif"],
        ...     material_ids=["mp-1", "mp-2"]
        ... )
        >>> extractor.save_results(results, "embeddings/")
    """

    EXTRACTORS = {
        "mace": MACEExtractor,
        "orb": ORBExtractor,
        "sevennet": SevenNetExtractor,
        "7net": SevenNetExtractor,
        "nequip": NequIPExtractor,
        "allegro": NequIPExtractor,
    }

    def __init__(
        self,
        model_type: Literal["mace", "orb", "sevennet", "7net", "nequip", "allegro"],
        model_path: str,
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_type: Type of MLIP ("mace", "orb", "sevennet", "nequip")
            model_path: Path to model file or model name
            device: Device to run on ("cuda", "cuda:0", "cpu")
            **kwargs: Additional arguments passed to specific extractor
                MACE: num_layers (int)
                ORB: precision (str)
                SevenNet: modal (str), cutoff (float)
                NequIP: r_max (float), layers_to_save (str)
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.device = device

        if self.model_type not in self.EXTRACTORS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Supported: {list(self.EXTRACTORS.keys())}")

        self.extractor = self.EXTRACTORS[self.model_type](
            model_path=model_path,
            device=device,
            **kwargs
        )

    def extract_from_atoms(
        self,
        atoms_list: List[Atoms],
        material_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> EmbeddingResult:
        """
        Extract embeddings from a list of ASE Atoms objects.

        Args:
            atoms_list: List of ASE Atoms objects
            material_ids: Material IDs (optional, will generate if not provided)
            show_progress: Show progress bar

        Returns:
            EmbeddingResult with all embeddings
        """
        if material_ids is None:
            material_ids = [f"struct_{i}" for i in range(len(atoms_list))]

        structures = [
            StructureInput(atoms=atoms, material_id=mid)
            for atoms, mid in zip(atoms_list, material_ids)
        ]

        return self.extractor.extract_batch(structures, show_progress=show_progress)

    def extract_from_files(
        self,
        file_paths: List[str],
        material_ids: Optional[List[str]] = None,
        file_format: Optional[str] = None,
        show_progress: bool = True,
        **reader_kwargs
    ) -> EmbeddingResult:
        """
        Extract embeddings from structure files.

        Args:
            file_paths: List of file paths (CIF, XYZ, JSON, CSV)
            material_ids: Material IDs (optional)
            file_format: Force specific format (auto-detect if None)
            show_progress: Show progress bar
            **reader_kwargs: Additional arguments for StructureReader

        Returns:
            EmbeddingResult with all embeddings
        """
        structures = []

        for i, path in enumerate(file_paths):
            mid = material_ids[i] if material_ids else None
            result = StructureReader.auto_read(path, material_id=mid, **reader_kwargs)

            if isinstance(result, list):
                structures.extend(result)
            else:
                structures.append(result)

        return self.extractor.extract_batch(structures, show_progress=show_progress)

    def extract_from_json(
        self,
        json_path: str,
        show_progress: bool = True,
    ) -> EmbeddingResult:
        """
        Extract embeddings from a Materials Project style JSON file.

        Args:
            json_path: Path to JSON file
            show_progress: Show progress bar

        Returns:
            EmbeddingResult with all embeddings
        """
        structures = StructureReader.read_mp_json(json_path)
        return self.extractor.extract_batch(structures, show_progress=show_progress)

    def extract_from_csv(
        self,
        csv_path: str,
        cif_column: str = "cif",
        id_column: Optional[str] = None,
        show_progress: bool = True,
    ) -> EmbeddingResult:
        """
        Extract embeddings from a CSV file with CIF strings.

        Args:
            csv_path: Path to CSV file
            cif_column: Column name containing CIF strings
            id_column: Column name for material IDs (optional)
            show_progress: Show progress bar

        Returns:
            EmbeddingResult with all embeddings
        """
        structures = StructureReader.read_csv_with_cif(
            csv_path, cif_column=cif_column, id_column=id_column
        )
        return self.extractor.extract_batch(structures, show_progress=show_progress)

    def save_results(
        self,
        results: EmbeddingResult,
        output_dir: str,
        prefix: str = "",
        save_formats: List[str] = ["pkl", "npy"],
    ):
        """
        Save extraction results to files.

        Args:
            results: EmbeddingResult to save
            output_dir: Output directory
            prefix: Prefix for output files
            save_formats: List of formats to save ("pkl", "npy", "npz")
        """
        os.makedirs(output_dir, exist_ok=True)

        prefix = f"{prefix}_" if prefix else ""
        model_name = Path(self.model_path).stem

        if "pkl" in save_formats:
            # Save full results as pickle
            with open(os.path.join(output_dir, f"{prefix}embeddings_{model_name}.pkl"), 'wb') as f:
                pickle.dump(results.embeddings, f)

            with open(os.path.join(output_dir, f"{prefix}atomic_numbers_{model_name}.pkl"), 'wb') as f:
                pickle.dump(results.atomic_numbers, f)

            with open(os.path.join(output_dir, f"{prefix}material_ids_{model_name}.pkl"), 'wb') as f:
                pickle.dump(results.material_ids, f)

            # Save flat material IDs (per atom)
            with open(os.path.join(output_dir, f"{prefix}flat_material_ids_{model_name}.pkl"), 'wb') as f:
                pickle.dump(results.flat_material_ids, f)

        if "npy" in save_formats and results.flat_embeddings is not None:
            # Save flat arrays as numpy
            np.save(
                os.path.join(output_dir, f"{prefix}flat_embeddings_{model_name}.npy"),
                results.flat_embeddings
            )
            np.save(
                os.path.join(output_dir, f"{prefix}flat_atomic_numbers_{model_name}.npy"),
                results.flat_atomic_numbers
            )

        if "npz" in save_formats and results.flat_embeddings is not None:
            # Save everything in a single npz file
            np.savez_compressed(
                os.path.join(output_dir, f"{prefix}all_data_{model_name}.npz"),
                flat_embeddings=results.flat_embeddings,
                flat_atomic_numbers=results.flat_atomic_numbers,
                flat_material_ids=np.array(results.flat_material_ids, dtype=object),
                material_ids=np.array(results.material_ids, dtype=object),
                n_atoms_per_structure=np.array([
                    len(e) if e is not None else 0 for e in results.embeddings
                ]),
            )

        print(f"Results saved to {output_dir}/")
        print(f"  - {results.n_structures} structures")
        print(f"  - {results.n_atoms_total} total atoms")
        print(f"  - Embedding dimension: {results.embed_dim}")

    def cleanup(self):
        """Clean up resources."""
        self.extractor.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_mace_embeddings(
    model_path: str,
    file_paths: Optional[List[str]] = None,
    atoms_list: Optional[List[Atoms]] = None,
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    device: str = "cuda",
    num_layers: int = -1,
    **kwargs
) -> EmbeddingResult:
    """
    Convenience function to extract MACE embeddings.

    Provide ONE of: file_paths, atoms_list, json_path, or csv_path.
    """
    with MLIPEmbeddingExtractor("mace", model_path, device, num_layers=num_layers) as extractor:
        if json_path:
            return extractor.extract_from_json(json_path)
        elif csv_path:
            return extractor.extract_from_csv(csv_path, **kwargs)
        elif atoms_list:
            return extractor.extract_from_atoms(atoms_list, **kwargs)
        elif file_paths:
            return extractor.extract_from_files(file_paths, **kwargs)
        else:
            raise ValueError("Must provide file_paths, atoms_list, json_path, or csv_path")


def extract_orb_embeddings(
    model_name: str = "orb_v3_direct_inf_omat",
    file_paths: Optional[List[str]] = None,
    atoms_list: Optional[List[Atoms]] = None,
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    device: str = "cuda",
    precision: str = "float32-high",
    **kwargs
) -> EmbeddingResult:
    """Convenience function to extract ORB embeddings."""
    with MLIPEmbeddingExtractor("orb", model_name, device, precision=precision) as extractor:
        if json_path:
            return extractor.extract_from_json(json_path)
        elif csv_path:
            return extractor.extract_from_csv(csv_path, **kwargs)
        elif atoms_list:
            return extractor.extract_from_atoms(atoms_list, **kwargs)
        elif file_paths:
            return extractor.extract_from_files(file_paths, **kwargs)
        else:
            raise ValueError("Must provide file_paths, atoms_list, json_path, or csv_path")


def extract_sevennet_embeddings(
    model_name: str = "7net-omat",
    file_paths: Optional[List[str]] = None,
    atoms_list: Optional[List[Atoms]] = None,
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    device: str = "cuda",
    modal: Optional[str] = None,
    cutoff: float = 5.0,
    **kwargs
) -> EmbeddingResult:
    """Convenience function to extract SevenNet embeddings."""
    with MLIPEmbeddingExtractor("sevennet", model_name, device,
                                 modal=modal, cutoff=cutoff) as extractor:
        if json_path:
            return extractor.extract_from_json(json_path)
        elif csv_path:
            return extractor.extract_from_csv(csv_path, **kwargs)
        elif atoms_list:
            return extractor.extract_from_atoms(atoms_list, **kwargs)
        elif file_paths:
            return extractor.extract_from_files(file_paths, **kwargs)
        else:
            raise ValueError("Must provide file_paths, atoms_list, json_path, or csv_path")


def extract_nequip_embeddings(
    model_path: str,
    file_paths: Optional[List[str]] = None,
    atoms_list: Optional[List[Atoms]] = None,
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    device: str = "cuda",
    r_max: float = 7.0,
    layers_to_save: str = "layer",
    **kwargs
) -> EmbeddingResult:
    """Convenience function to extract NequIP/Allegro embeddings."""
    with MLIPEmbeddingExtractor("nequip", model_path, device,
                                 r_max=r_max, layers_to_save=layers_to_save) as extractor:
        if json_path:
            return extractor.extract_from_json(json_path)
        elif csv_path:
            return extractor.extract_from_csv(csv_path, **kwargs)
        elif atoms_list:
            return extractor.extract_from_atoms(atoms_list, **kwargs)
        elif file_paths:
            return extractor.extract_from_files(file_paths, **kwargs)
        else:
            raise ValueError("Must provide file_paths, atoms_list, json_path, or csv_path")


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """
    Example demonstrating the unified embedding extraction interface.
    """
    print("=" * 80)
    print("MLIP Embedding Extraction Examples")
    print("=" * 80)

    # =========================================================================
    # Example 1: MACE with JSON input
    # =========================================================================
    print("\n--- Example 1: MACE from JSON ---")
    print("""
    from mlip_embedding_extractor import MLIPEmbeddingExtractor

    extractor = MLIPEmbeddingExtractor(
        model_type="mace",
        model_path="/path/to/mace-mpa-0-medium.model",
        device="cuda:0",
        num_layers=-1  # Last layer embeddings
    )

    results = extractor.extract_from_json("structures.json")
    extractor.save_results(results, "output/mace/")

    # Access results
    print(f"Extracted {results.n_structures} structures")
    print(f"Total atoms: {results.n_atoms_total}")
    print(f"Embedding dim: {results.embed_dim}")

    # Get flat arrays for analysis
    flat_emb = results.flat_embeddings  # (n_atoms, embed_dim)
    flat_ids = results.flat_material_ids  # List of material IDs per atom
    """)

    # =========================================================================
    # Example 2: ORB with CSV input
    # =========================================================================
    print("\n--- Example 2: ORB from CSV ---")
    print("""
    results = extract_orb_embeddings(
        model_name="orb_v3_direct_inf_omat",
        csv_path="data/train.csv",
        cif_column="cif",
        id_column="material_id",
        device="cuda"
    )

    # Save all formats
    extractor.save_results(results, "output/orb/", save_formats=["pkl", "npy", "npz"])
    """)

    # =========================================================================
    # Example 3: SevenNet with CIF files
    # =========================================================================
    print("\n--- Example 3: SevenNet from CIF files ---")
    print("""
    from pathlib import Path

    cif_files = list(Path("structures/").glob("*.cif"))
    material_ids = [f.stem for f in cif_files]

    results = extract_sevennet_embeddings(
        model_name="7net-omat",
        file_paths=[str(f) for f in cif_files],
        material_ids=material_ids,
        device="cuda",
        cutoff=5.0
    )
    """)

    # =========================================================================
    # Example 4: NequIP with ASE Atoms
    # =========================================================================
    print("\n--- Example 4: NequIP from ASE Atoms ---")
    print("""
    from ase.io import read
    from ase.build import bulk

    # Create some test structures
    atoms_list = [
        bulk("Si", "diamond", a=5.43),
        bulk("Fe", "bcc", a=2.87),
        read("my_structure.cif"),
    ]

    with MLIPEmbeddingExtractor(
        model_type="nequip",
        model_path="/path/to/NequIP-MP-L-0.1.nequip.zip",
        device="cuda",
        r_max=7.0,
        layers_to_save="layer"
    ) as extractor:
        results = extractor.extract_from_atoms(atoms_list, material_ids=["Si", "Fe", "custom"])
        extractor.save_results(results, "output/nequip/")
    """)

    # =========================================================================
    # Example 5: Batch processing multiple models
    # =========================================================================
    print("\n--- Example 5: Compare multiple MLIPs ---")
    print("""
    from mlip_embedding_extractor import (
        extract_mace_embeddings,
        extract_orb_embeddings,
        extract_sevennet_embeddings,
        extract_nequip_embeddings
    )

    # Same input for all models
    json_path = "my_structures.json"

    # Extract from each model
    mace_results = extract_mace_embeddings(
        model_path="/path/to/mace.model",
        json_path=json_path
    )

    orb_results = extract_orb_embeddings(
        model_name="orb_v3_direct_inf_omat",
        json_path=json_path
    )

    sevennet_results = extract_sevennet_embeddings(
        model_name="7net-omat",
        json_path=json_path
    )

    nequip_results = extract_nequip_embeddings(
        model_path="/path/to/nequip.zip",
        json_path=json_path
    )

    # Compare embedding dimensions
    for name, res in [("MACE", mace_results), ("ORB", orb_results),
                      ("SevenNet", sevennet_results), ("NequIP", nequip_results)]:
        print(f"{name}: {res.embed_dim}-dim embeddings, {res.n_atoms_total} atoms")
    """)

    # =========================================================================
    # Example 6: Integration with ManifoldFishAnalyzer
    # =========================================================================
    print("\n--- Example 6: Full pipeline with ManifoldFishAnalyzer ---")
    print("""
    from mlip_embedding_extractor import MLIPEmbeddingExtractor
    from manifold_fish_analysis import ManifoldFishAnalyzer, aggregate_by_material_id
    from sklearn.decomposition import PCA

    # 1. Extract embeddings
    extractor = MLIPEmbeddingExtractor("mace", "/path/to/mace.model")

    ref_results = extractor.extract_from_json("reference_structures.json")
    gen_results = extractor.extract_from_json("generated_structures.json")

    # 2. Aggregate to material level
    ref_mat_emb, ref_ids = aggregate_by_material_id(
        ref_results.flat_embeddings,
        ref_results.flat_material_ids,
        method="mean"
    )
    gen_mat_emb, gen_ids = aggregate_by_material_id(
        gen_results.flat_embeddings,
        gen_results.flat_material_ids,
        method="mean"
    )

    # 3. Optional: PCA reduction for visualization
    pca = PCA(n_components=3)
    ref_pca = pca.fit_transform(ref_mat_emb)
    gen_pca = pca.transform(gen_mat_emb)

    # 4. Analyze with ManifoldFishAnalyzer
    analyzer = ManifoldFishAnalyzer(
        reference_embeddings=ref_pca,
        reference_ids=ref_ids,
        boundary_method="alpha_shape",
        local_geometry_mode="fast"
    )

    results = analyzer.analyze(gen_pca, gen_ids)
    analyzer.print_summary(results)
    analyzer.export_results(results, "analysis_results.csv")

    # 5. Get priority candidates
    frontier_ids = analyzer.get_ids_by_category(results, "frontier_fish")
    print(f"Found {len(frontier_ids)} frontier candidates for DFT validation")
    """)


if __name__ == "__main__":
    example_usage()
