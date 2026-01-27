"""
Structure Evaluator: End-to-end evaluation of generated crystal structures.

This module provides a high-level interface for evaluating AI-generated
crystal structures using the ManiFish framework with spatial structure metrics.

Workflow:
    1. Extract embeddings from structures using MLIP
    2. Project to anchor space (Platonic representation)
    3. Evaluate using combined distance + spatial metrics
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from .manifold import (
    PlatonicProjector,
    calculate_platonic_representation,
    PlatonicRepresentation,
)
from .anchors import AnchorSet, select_anchors
from .enhanced_evaluator import (
    SpatialManifoldEvaluator,
    StabilityPrediction,
    EvaluatorConfig,
    DistanceMetrics,
)
from .spatial_metrics import (
    SpatialMetrics,
    SpatialMetricsConfig,
    create_region_labels_from_crystal_system,
    create_region_labels_from_composition_family,
)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a structure."""
    structure_id: Optional[str]
    stability_score: float  # Primary score (enhanced)
    stability_basic: float  # Distance-only score
    novelty_score: float
    confidence: float

    # Detailed metrics
    distance_metrics: Dict[str, float]
    spatial_metrics: Dict[str, float]

    # Recommendations
    recommendation: str
    is_stable: bool  # Binary classification
    is_novel: bool

    # Raw coordinates (for further analysis)
    anchor_coords: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'structure_id': self.structure_id,
            'stability_score': self.stability_score,
            'stability_basic': self.stability_basic,
            'novelty_score': self.novelty_score,
            'confidence': self.confidence,
            'distance_metrics': self.distance_metrics,
            'spatial_metrics': self.spatial_metrics,
            'recommendation': self.recommendation,
            'is_stable': self.is_stable,
            'is_novel': self.is_novel,
        }


@dataclass
class ManifoldReference:
    """Reference manifold for structure evaluation."""
    anchor_set: AnchorSet
    manifold_coords: np.ndarray  # (n_structures, n_anchors)
    region_labels: Optional[np.ndarray] = None
    structure_ids: Optional[List[str]] = None
    model_name: str = 'unknown'

    @property
    def n_structures(self) -> int:
        return len(self.manifold_coords)

    @property
    def n_anchors(self) -> int:
        return self.manifold_coords.shape[1]

    def save(self, path: Union[str, Path]):
        """Save manifold reference to file."""
        import pickle
        path = Path(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ManifoldReference':
        """Load manifold reference from file."""
        import pickle
        path = Path(path)
        with open(path, 'rb') as f:
            return pickle.load(f)


class StructureEvaluator:
    """
    End-to-end evaluator for AI-generated crystal structures.

    This is the main interface for the ManiFish evaluation pipeline.
    It handles:
    - Embedding extraction from structures
    - Projection to anchor space
    - Combined distance + spatial metric evaluation
    - Stability and novelty scoring

    Example:
        >>> # Setup with pre-built manifold
        >>> evaluator = StructureEvaluator.from_manifold_reference(
        ...     manifold_ref,
        ...     embedding_extractor=mace_extractor,
        ... )
        >>>
        >>> # Evaluate generated structures
        >>> results = evaluator.evaluate_structures(generated_structures)
        >>> for r in results:
        ...     print(f"{r.structure_id}: stability={r.stability_score:.3f}")

        >>> # Or build from training data
        >>> evaluator = StructureEvaluator.from_training_structures(
        ...     training_structures,
        ...     embedding_extractor=mace_extractor,
        ...     n_anchors=100,
        ... )
    """

    def __init__(
        self,
        manifold_reference: ManifoldReference,
        embedding_extractor: Optional[Any] = None,
        config: Optional[EvaluatorConfig] = None,
        stability_threshold: float = 0.5,
        novelty_threshold: float = 0.7,
    ):
        """
        Initialize the structure evaluator.

        Args:
            manifold_reference: Reference manifold with anchors and training coords
            embedding_extractor: MLIP embedding extractor (e.g., MACEEmbeddingExtractor)
            config: Evaluator configuration
            stability_threshold: Threshold for binary stability classification
            novelty_threshold: Threshold for binary novelty classification
        """
        self.manifold_ref = manifold_reference
        self.embedding_extractor = embedding_extractor
        self.config = config or EvaluatorConfig()
        self.stability_threshold = stability_threshold
        self.novelty_threshold = novelty_threshold

        # Initialize projector
        self.projector = PlatonicProjector(
            anchor_set=manifold_reference.anchor_set,
            similarity_method='cosine',
        )

        # Initialize spatial evaluator
        self.spatial_evaluator = SpatialManifoldEvaluator(
            manifold_coords=manifold_reference.manifold_coords,
            anchor_set=manifold_reference.anchor_set,
            region_labels=manifold_reference.region_labels,
            config=self.config,
        )

    @classmethod
    def from_manifold_reference(
        cls,
        manifold_reference: ManifoldReference,
        embedding_extractor: Optional[Any] = None,
        config: Optional[EvaluatorConfig] = None,
        **kwargs,
    ) -> 'StructureEvaluator':
        """Create evaluator from pre-built manifold reference."""
        return cls(
            manifold_reference=manifold_reference,
            embedding_extractor=embedding_extractor,
            config=config,
            **kwargs,
        )

    @classmethod
    def from_training_structures(
        cls,
        training_structures: List[Any],
        embedding_extractor: Any,
        n_anchors: int = 100,
        anchor_method: str = 'direct',
        region_labels: Optional[np.ndarray] = None,
        crystal_systems: Optional[List[str]] = None,
        structure_ids: Optional[List[str]] = None,
        config: Optional[EvaluatorConfig] = None,
        **kwargs,
    ) -> 'StructureEvaluator':
        """
        Create evaluator by building manifold from training structures.

        Args:
            training_structures: List of stable training structures (ASE Atoms or pymatgen)
            embedding_extractor: MLIP embedding extractor
            n_anchors: Number of anchors to select
            anchor_method: 'direct' or 'fps'
            region_labels: Pre-computed region labels
            crystal_systems: List of crystal system names (to create region labels)
            structure_ids: Optional structure identifiers
            config: Evaluator configuration

        Returns:
            Configured StructureEvaluator
        """
        print(f"Building manifold from {len(training_structures)} training structures...")

        # Extract embeddings
        print("  Extracting embeddings...")
        embeddings = embedding_extractor.get_embeddings(training_structures)

        # Aggregate per-atom to per-structure if needed
        if isinstance(embeddings, list):
            embeddings = np.stack([
                np.mean(e, axis=0) if e.ndim > 1 else e
                for e in embeddings
            ])

        # Select anchors
        print(f"  Selecting {n_anchors} anchors using {anchor_method}...")
        anchor_set = select_anchors(
            embeddings=embeddings,
            n_anchors=n_anchors,
            method=anchor_method,
            model_name=getattr(embedding_extractor, 'model_name', 'unknown'),
            structure_ids=structure_ids,
        )

        # Project all training structures to anchor space
        print("  Projecting to anchor space...")
        manifold_coords = calculate_platonic_representation(
            embeddings=embeddings,
            anchor_embeddings=anchor_set,
            model_name=anchor_set.source_model,
        )

        # Create region labels if crystal systems provided
        if region_labels is None and crystal_systems is not None:
            print("  Creating region labels from crystal systems...")
            region_labels = create_region_labels_from_crystal_system(crystal_systems)

        # Build manifold reference
        manifold_ref = ManifoldReference(
            anchor_set=anchor_set,
            manifold_coords=manifold_coords,
            region_labels=region_labels,
            structure_ids=structure_ids,
            model_name=anchor_set.source_model,
        )

        print(f"  Manifold built: {manifold_ref.n_structures} structures, {manifold_ref.n_anchors} anchors")

        return cls(
            manifold_reference=manifold_ref,
            embedding_extractor=embedding_extractor,
            config=config,
            **kwargs,
        )

    def evaluate_structures(
        self,
        structures: List[Any],
        structure_ids: Optional[List[str]] = None,
        return_coords: bool = False,
    ) -> List[EvaluationResult]:
        """
        Evaluate a list of generated structures.

        Args:
            structures: List of structures to evaluate (ASE Atoms or pymatgen)
            structure_ids: Optional identifiers for structures
            return_coords: Whether to include anchor coordinates in results

        Returns:
            List of EvaluationResult objects
        """
        if self.embedding_extractor is None:
            raise ValueError(
                "No embedding extractor provided. Either pass embeddings directly "
                "using evaluate_embeddings() or provide an extractor at initialization."
            )

        # Extract embeddings
        embeddings = self.embedding_extractor.get_embeddings(structures)

        # Aggregate per-atom to per-structure if needed
        if isinstance(embeddings, list):
            embeddings = np.stack([
                np.mean(e, axis=0) if e.ndim > 1 else e
                for e in embeddings
            ])

        return self.evaluate_embeddings(
            embeddings=embeddings,
            structure_ids=structure_ids,
            return_coords=return_coords,
        )

    def evaluate_embeddings(
        self,
        embeddings: np.ndarray,
        structure_ids: Optional[List[str]] = None,
        return_coords: bool = False,
    ) -> List[EvaluationResult]:
        """
        Evaluate structures from their embeddings directly.

        Use this when you already have embeddings computed.

        Args:
            embeddings: Structure embeddings (n_structures, embedding_dim)
            structure_ids: Optional identifiers
            return_coords: Whether to include anchor coordinates

        Returns:
            List of EvaluationResult objects
        """
        embeddings = np.atleast_2d(embeddings)
        n_structures = len(embeddings)

        if structure_ids is None:
            structure_ids = [f"structure_{i}" for i in range(n_structures)]

        # Project to anchor space
        coords = calculate_platonic_representation(
            embeddings=embeddings,
            anchor_embeddings=self.manifold_ref.anchor_set,
            model_name=self.manifold_ref.model_name,
        )

        return self.evaluate_coordinates(
            coords=coords,
            structure_ids=structure_ids,
            return_coords=return_coords,
        )

    def evaluate_coordinates(
        self,
        coords: np.ndarray,
        structure_ids: Optional[List[str]] = None,
        return_coords: bool = False,
    ) -> List[EvaluationResult]:
        """
        Evaluate structures from their anchor-space coordinates directly.

        Use this when you already have projected coordinates.

        Args:
            coords: Anchor-space coordinates (n_structures, n_anchors)
            structure_ids: Optional identifiers
            return_coords: Whether to include coordinates in results

        Returns:
            List of EvaluationResult objects
        """
        coords = np.atleast_2d(coords)
        n_structures = len(coords)

        if structure_ids is None:
            structure_ids = [f"structure_{i}" for i in range(n_structures)]

        # Evaluate each structure
        predictions = self.spatial_evaluator.evaluate_batch(coords)

        # Convert to EvaluationResult objects
        results = []
        for i, (pred, sid) in enumerate(zip(predictions, structure_ids)):
            result = EvaluationResult(
                structure_id=sid,
                stability_score=pred.stability_enhanced,
                stability_basic=pred.stability_basic,
                novelty_score=pred.novelty_score,
                confidence=pred.stability_confidence,
                distance_metrics=pred.distance_metrics.to_dict(),
                spatial_metrics=pred.spatial_metrics.to_dict(),
                recommendation=pred.recommendation,
                is_stable=pred.stability_enhanced >= self.stability_threshold,
                is_novel=pred.novelty_score >= self.novelty_threshold,
                anchor_coords=coords[i] if return_coords else None,
            )
            results.append(result)

        return results

    def evaluate_single(
        self,
        structure: Any,
        structure_id: Optional[str] = None,
    ) -> EvaluationResult:
        """Convenience method to evaluate a single structure."""
        results = self.evaluate_structures(
            structures=[structure],
            structure_ids=[structure_id] if structure_id else None,
        )
        return results[0]

    def get_summary_statistics(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """
        Compute summary statistics for a batch of evaluation results.

        Args:
            results: List of EvaluationResult objects

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}

        stability_scores = [r.stability_score for r in results]
        stability_basic = [r.stability_basic for r in results]
        novelty_scores = [r.novelty_score for r in results]
        confidences = [r.confidence for r in results]

        n_stable = sum(1 for r in results if r.is_stable)
        n_novel = sum(1 for r in results if r.is_novel)
        n_stable_novel = sum(1 for r in results if r.is_stable and r.is_novel)

        # Spatial metric averages
        lds_scores = [r.spatial_metrics['lds'] for r in results]
        cds_scores = [r.spatial_metrics['cds'] for r in results]
        srss_scores = [r.spatial_metrics['srss'] for r in results]
        rmsc_scores = [r.spatial_metrics['rmsc'] for r in results]

        return {
            'n_structures': len(results),
            'n_stable': n_stable,
            'n_novel': n_novel,
            'n_stable_and_novel': n_stable_novel,
            'stability_rate': n_stable / len(results),
            'novelty_rate': n_novel / len(results),
            'mean_stability': np.mean(stability_scores),
            'std_stability': np.std(stability_scores),
            'mean_stability_basic': np.mean(stability_basic),
            'improvement_over_basic': np.mean(stability_scores) - np.mean(stability_basic),
            'mean_novelty': np.mean(novelty_scores),
            'mean_confidence': np.mean(confidences),
            'spatial_metrics': {
                'mean_lds': np.mean(lds_scores),
                'mean_cds': np.mean(cds_scores),
                'mean_srss': np.mean(srss_scores),
                'mean_rmsc': np.mean(rmsc_scores),
            },
        }

    def find_similar_training(
        self,
        structure: Any,
        k: int = 5,
    ) -> Tuple[List[int], List[float]]:
        """
        Find k most similar structures in the training manifold.

        Args:
            structure: Query structure
            k: Number of similar structures to return

        Returns:
            Tuple of (indices, distances) in training set
        """
        if self.embedding_extractor is None:
            raise ValueError("Embedding extractor required for this operation")

        # Get embedding and project
        embedding = self.embedding_extractor.get_embeddings([structure])
        if isinstance(embedding, list):
            embedding = np.mean(embedding[0], axis=0) if embedding[0].ndim > 1 else embedding[0]
        else:
            embedding = embedding[0]

        coords = calculate_platonic_representation(
            embeddings=embedding.reshape(1, -1),
            anchor_embeddings=self.manifold_ref.anchor_set,
        )

        indices, distances = self.spatial_evaluator.get_similar_structures(coords, k=k)
        return list(indices), list(distances)

    def save_manifold(self, path: Union[str, Path]):
        """Save the manifold reference for later use."""
        self.manifold_ref.save(path)

    @classmethod
    def load_with_manifold(
        cls,
        manifold_path: Union[str, Path],
        embedding_extractor: Optional[Any] = None,
        **kwargs,
    ) -> 'StructureEvaluator':
        """Load evaluator from saved manifold reference."""
        manifold_ref = ManifoldReference.load(manifold_path)
        return cls(
            manifold_reference=manifold_ref,
            embedding_extractor=embedding_extractor,
            **kwargs,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_evaluate(
    structures: List[Any],
    training_structures: List[Any],
    embedding_extractor: Any,
    n_anchors: int = 100,
    crystal_systems: Optional[List[str]] = None,
) -> List[EvaluationResult]:
    """
    Quick one-shot evaluation of structures.

    Builds manifold from training structures and evaluates in one call.
    For repeated evaluations, use StructureEvaluator directly.

    Args:
        structures: Structures to evaluate
        training_structures: Stable training structures for manifold
        embedding_extractor: MLIP embedding extractor
        n_anchors: Number of anchors
        crystal_systems: Optional crystal systems for SRSS

    Returns:
        List of EvaluationResult objects
    """
    evaluator = StructureEvaluator.from_training_structures(
        training_structures=training_structures,
        embedding_extractor=embedding_extractor,
        n_anchors=n_anchors,
        crystal_systems=crystal_systems,
    )
    return evaluator.evaluate_structures(structures)


def evaluate_from_embeddings(
    query_embeddings: np.ndarray,
    training_embeddings: np.ndarray,
    n_anchors: int = 100,
    region_labels: Optional[np.ndarray] = None,
) -> List[EvaluationResult]:
    """
    Evaluate structures from pre-computed embeddings.

    Useful when you've already extracted embeddings and want to
    experiment with different evaluation settings.

    Args:
        query_embeddings: Embeddings of structures to evaluate
        training_embeddings: Embeddings of training structures
        n_anchors: Number of anchors
        region_labels: Optional region labels for SRSS

    Returns:
        List of EvaluationResult objects
    """
    # Select anchors from training
    anchor_set = select_anchors(
        embeddings=training_embeddings,
        n_anchors=n_anchors,
        method='direct',
    )

    # Project training to anchor space
    manifold_coords = calculate_platonic_representation(
        embeddings=training_embeddings,
        anchor_embeddings=anchor_set,
    )

    # Build manifold reference
    manifold_ref = ManifoldReference(
        anchor_set=anchor_set,
        manifold_coords=manifold_coords,
        region_labels=region_labels,
    )

    # Create evaluator and evaluate
    evaluator = StructureEvaluator(manifold_reference=manifold_ref)
    return evaluator.evaluate_embeddings(query_embeddings)
