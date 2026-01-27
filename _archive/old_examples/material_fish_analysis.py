"""
Material-Level Fish-Water Analysis

Analyzes materials at the atom level and aggregates to material level.
Distinguishes between:
1. Novel Atoms - Material has unusual atomic environments
2. Novel Arrangement - Material has common atoms in unusual configurations
3. Both - Novel atoms AND novel arrangement

The "fish" is the material, but we analyze it through its atoms.

Usage:
    analyzer = MaterialFishAnalyzer(reference_atom_embeddings, reference_atom_ids)
    results = analyzer.analyze(
        generated_atom_embeddings,  # List of arrays, one per material
        material_ids,               # List of material IDs
        atom_ids=atom_ids           # Optional: List of lists of atom IDs
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle

# Import FishWaterAnalyzer for the mean-pooled approach
try:
    from fish_water_analysis import FishWaterAnalyzer, FishWaterResult
except ImportError:
    # If running from different directory, try relative import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fish_water_analysis import FishWaterAnalyzer, FishWaterResult


@dataclass
class AtomResult:
    """Results for a single atom."""
    atom_index: int
    atom_id: Optional[str]
    material_id: str
    distance_to_manifold: float
    is_novel: bool
    novelty_score: float
    nearest_reference_ids: List[str] = None


@dataclass
class MaterialFishResult:
    """Results for a single material (the fish)."""
    material_id: str
    n_atoms: int

    # Atom-level statistics
    novel_atom_fraction: float      # Fraction of atoms that are novel
    mean_atom_distance: float       # Mean distance of atoms to reference
    max_atom_distance: float        # Max atom distance (most unusual atom)
    atom_distance_std: float        # Spread of atom distances

    # Arrangement metrics
    arrangement_novelty: float      # Novelty of how atoms are arranged
    internal_diversity: float       # How diverse are atoms within this material
    centroid_distance: float        # Distance of material centroid to reference

    # Combined metrics
    overall_novelty: float          # Combined novelty score
    novelty_type: str               # "novel_atoms", "novel_arrangement", "both", "neither"

    # Classification
    category: str                   # Fish-water category
    confidence: float

    # Atom details
    atom_results: List[AtomResult] = None
    novel_atom_indices: List[int] = None


class MaterialFishAnalyzer:
    """
    Analyze materials through their atoms using Fish-Water framework.

    Key insight: A material can be novel because of:
    1. Novel atoms (unusual atomic environments)
    2. Novel arrangement (common atoms, unusual configuration)
    3. Both

    Example:
        >>> # reference_atoms: List of atom embeddings from known materials
        >>> # generated_atoms: List of [n_atoms, dim] arrays, one per generated material
        >>>
        >>> analyzer = MaterialFishAnalyzer(reference_atoms, reference_atom_to_material)
        >>> results = analyzer.analyze(generated_atoms, material_ids)
        >>> analyzer.print_summary(results)
    """

    def __init__(
        self,
        reference_atom_embeddings: Union[np.ndarray, List[np.ndarray]],
        reference_atom_ids: Optional[List[str]] = None,
        reference_material_ids: Optional[List[str]] = None,
        n_neighbors: int = 10,
        atom_novelty_threshold: float = 0.7,      # Percentile for "novel" atom
        arrangement_novelty_threshold: float = 0.5,
        stability_threshold: float = 0.5,
    ):
        """
        Initialize analyzer with reference atom embeddings.

        Args:
            reference_atom_embeddings: All atom embeddings from reference materials
                Can be (n_total_atoms, dim) array or list of per-material arrays
            reference_atom_ids: Optional IDs for each reference atom
            reference_material_ids: Material ID for each reference atom (for grouping)
            n_neighbors: Number of neighbors for distance calculation
            atom_novelty_threshold: Distance percentile above which atom is "novel"
            arrangement_novelty_threshold: Threshold for arrangement novelty
            stability_threshold: Threshold for overall stability
        """
        # Flatten reference atoms if provided as list
        if isinstance(reference_atom_embeddings, list):
            self.reference_atoms = np.vstack(reference_atom_embeddings)
            # Track which material each reference atom belongs to
            if reference_material_ids is None:
                reference_material_ids = []
                for i, arr in enumerate(reference_atom_embeddings):
                    reference_material_ids.extend([f"ref_mat_{i}"] * len(arr))
        else:
            self.reference_atoms = np.asarray(reference_atom_embeddings)

        self.n_reference_atoms = len(self.reference_atoms)

        # Store reference IDs
        if reference_atom_ids is not None:
            self.reference_atom_ids = list(reference_atom_ids)
        else:
            self.reference_atom_ids = [f"ref_atom_{i}" for i in range(self.n_reference_atoms)]

        if reference_material_ids is not None:
            self.reference_material_ids = list(reference_material_ids)
        else:
            self.reference_material_ids = [f"ref_mat_{i}" for i in range(self.n_reference_atoms)]

        self.n_neighbors = min(n_neighbors, self.n_reference_atoms - 1)
        self.atom_novelty_threshold = atom_novelty_threshold
        self.arrangement_novelty_threshold = arrangement_novelty_threshold
        self.stability_threshold = stability_threshold

        self._fit_reference_models()

    def _fit_reference_models(self):
        """Fit models on reference atom data."""
        # KNN for atom-level distances
        self.atom_knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='auto',
            metric='euclidean'
        )
        self.atom_knn.fit(self.reference_atoms)

        # Compute reference atom distance statistics
        ref_distances, _ = self.atom_knn.kneighbors(self.reference_atoms)
        self.ref_atom_mean_dist = np.mean(ref_distances[:, -1])
        self.ref_atom_std_dist = np.std(ref_distances[:, -1])

        # Percentile threshold for "novel" atoms
        all_ref_distances = ref_distances[:, 0]  # Distance to nearest neighbor
        self.atom_novel_distance_threshold = np.percentile(
            all_ref_distances,
            self.atom_novelty_threshold * 100
        )

        # Compute reference material centroids for arrangement analysis
        self._compute_reference_material_stats()

        # LOF for outlier detection
        self.atom_lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,
            contamination=0.1
        )
        self.atom_lof.fit(self.reference_atoms)

    def _compute_reference_material_stats(self):
        """Compute statistics for reference materials (for arrangement comparison)."""
        # Group atoms by material
        material_to_atoms = {}
        for i, mat_id in enumerate(self.reference_material_ids):
            if mat_id not in material_to_atoms:
                material_to_atoms[mat_id] = []
            material_to_atoms[mat_id].append(i)

        # Compute per-material statistics
        self.ref_material_centroids = []
        self.ref_material_internal_vars = []

        for mat_id, atom_indices in material_to_atoms.items():
            atoms = self.reference_atoms[atom_indices]
            centroid = np.mean(atoms, axis=0)
            internal_var = np.mean(np.var(atoms, axis=0))

            self.ref_material_centroids.append(centroid)
            self.ref_material_internal_vars.append(internal_var)

        self.ref_material_centroids = np.array(self.ref_material_centroids)
        self.ref_material_internal_vars = np.array(self.ref_material_internal_vars)

        # KNN for material centroids
        if len(self.ref_material_centroids) > 1:
            self.centroid_knn = NearestNeighbors(
                n_neighbors=min(10, len(self.ref_material_centroids) - 1),
                algorithm='auto'
            )
            self.centroid_knn.fit(self.ref_material_centroids)

            # Reference centroid distance stats
            cent_distances, _ = self.centroid_knn.kneighbors(self.ref_material_centroids)
            self.ref_centroid_mean_dist = np.mean(cent_distances[:, -1])
            self.ref_centroid_std_dist = np.std(cent_distances[:, -1])

    def analyze(
        self,
        generated_atom_embeddings: List[np.ndarray],
        material_ids: List[str],
        atom_ids: Optional[List[List[str]]] = None,
    ) -> List[MaterialFishResult]:
        """
        Analyze generated materials through their atoms.

        Args:
            generated_atom_embeddings: List of arrays, each (n_atoms_i, dim)
                                       One array per material
            material_ids: Material ID for each generated material
            atom_ids: Optional list of lists of atom IDs

        Returns:
            List of MaterialFishResult for each material
        """
        n_materials = len(generated_atom_embeddings)

        if atom_ids is None:
            atom_ids = [
                [f"{material_ids[i]}_atom_{j}" for j in range(len(generated_atom_embeddings[i]))]
                for i in range(n_materials)
            ]

        results = []

        for mat_idx in range(n_materials):
            mat_id = material_ids[mat_idx]
            atoms = np.asarray(generated_atom_embeddings[mat_idx])

            # Ensure 2D array: (n_atoms, embedding_dim)
            if atoms.ndim == 1:
                atoms = atoms.reshape(1, -1)  # Single atom case

            mat_atom_ids = atom_ids[mat_idx] if mat_idx < len(atom_ids) else [f"{mat_id}_atom_0"]
            n_atoms = len(atoms)

            # Skip empty materials
            if n_atoms == 0:
                continue

            # 1. Analyze each atom
            atom_results = self._analyze_atoms(atoms, mat_atom_ids, mat_id)

            # 2. Compute atom-level statistics
            atom_distances = np.array([ar.distance_to_manifold for ar in atom_results])
            atom_novelties = np.array([ar.novelty_score for ar in atom_results])
            is_novel = np.array([ar.is_novel for ar in atom_results])

            novel_atom_fraction = np.mean(is_novel)
            mean_atom_distance = np.mean(atom_distances)
            max_atom_distance = np.max(atom_distances)
            atom_distance_std = np.std(atom_distances)

            novel_atom_indices = [ar.atom_index for ar in atom_results if ar.is_novel]

            # 3. Compute arrangement metrics
            arrangement_novelty, internal_diversity, centroid_distance = \
                self._compute_arrangement_metrics(atoms, atom_distances)

            # 4. Determine novelty type
            has_novel_atoms = novel_atom_fraction > 0.2  # >20% atoms are novel
            has_novel_arrangement = arrangement_novelty > self.arrangement_novelty_threshold

            if has_novel_atoms and has_novel_arrangement:
                novelty_type = "both"
            elif has_novel_atoms:
                novelty_type = "novel_atoms"
            elif has_novel_arrangement:
                novelty_type = "novel_arrangement"
            else:
                novelty_type = "neither"

            # 5. Compute overall novelty
            overall_novelty = self._compute_overall_novelty(
                novel_atom_fraction,
                arrangement_novelty,
                mean_atom_distance,
                centroid_distance
            )

            # 6. Classify material (fish-water)
            category, confidence = self._classify_material(
                mean_atom_distance=mean_atom_distance,
                novel_atom_fraction=novel_atom_fraction,
                arrangement_novelty=arrangement_novelty,
                overall_novelty=overall_novelty,
            )

            result = MaterialFishResult(
                material_id=mat_id,
                n_atoms=n_atoms,
                novel_atom_fraction=novel_atom_fraction,
                mean_atom_distance=mean_atom_distance,
                max_atom_distance=max_atom_distance,
                atom_distance_std=atom_distance_std,
                arrangement_novelty=arrangement_novelty,
                internal_diversity=internal_diversity,
                centroid_distance=centroid_distance,
                overall_novelty=overall_novelty,
                novelty_type=novelty_type,
                category=category,
                confidence=confidence,
                atom_results=atom_results,
                novel_atom_indices=novel_atom_indices,
            )
            results.append(result)

        return results

    def _analyze_atoms(
        self,
        atoms: np.ndarray,
        atom_ids: List[str],
        material_id: str,
    ) -> List[AtomResult]:
        """Analyze individual atoms."""
        n_atoms = len(atoms)

        # Get distances to reference atoms
        distances, indices = self.atom_knn.kneighbors(atoms)
        mean_distances = np.mean(distances, axis=1)

        # Normalize distances
        normalized_distances = (mean_distances - self.ref_atom_mean_dist) / (self.ref_atom_std_dist + 1e-8)

        # Determine which atoms are novel
        nearest_distances = distances[:, 0]
        is_novel = nearest_distances > self.atom_novel_distance_threshold

        # Compute novelty scores
        novelty_scores = normalized_distances / (np.max(np.abs(normalized_distances)) + 1e-8)
        novelty_scores = (novelty_scores + 1) / 2  # Scale to [0, 1]

        atom_results = []
        for i in range(n_atoms):
            nearest_ref_ids = [self.reference_atom_ids[idx] for idx in indices[i]]

            result = AtomResult(
                atom_index=i,
                atom_id=atom_ids[i] if atom_ids else None,
                material_id=material_id,
                distance_to_manifold=normalized_distances[i],
                is_novel=bool(is_novel[i]),
                novelty_score=novelty_scores[i],
                nearest_reference_ids=nearest_ref_ids,
            )
            atom_results.append(result)

        return atom_results

    def _compute_arrangement_metrics(
        self,
        atoms: np.ndarray,
        atom_distances: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute metrics for how atoms are arranged within the material.

        Returns:
            (arrangement_novelty, internal_diversity, centroid_distance)
        """
        # Material centroid
        centroid = np.mean(atoms, axis=0)

        # Distance of centroid to reference material centroids
        if hasattr(self, 'centroid_knn') and len(self.ref_material_centroids) > 0:
            cent_distances, _ = self.centroid_knn.kneighbors([centroid])
            centroid_distance = np.mean(cent_distances[0])
            centroid_distance_normalized = (centroid_distance - self.ref_centroid_mean_dist) / (self.ref_centroid_std_dist + 1e-8)
        else:
            centroid_distance_normalized = 0.0

        # Internal diversity: variance of atoms within material
        internal_var = np.mean(np.var(atoms, axis=0))

        # Compare to reference material internal variances
        if len(self.ref_material_internal_vars) > 0:
            ref_var_mean = np.mean(self.ref_material_internal_vars)
            ref_var_std = np.std(self.ref_material_internal_vars)
            internal_diversity = (internal_var - ref_var_mean) / (ref_var_std + 1e-8)
        else:
            internal_diversity = 0.0

        # Arrangement novelty: combination of centroid position and internal structure
        # High arrangement novelty = centroid is unusual OR internal structure is unusual
        # even if individual atoms are common

        # Check if atoms are common but arrangement is unusual
        mean_atom_dist = np.mean(atom_distances)
        atoms_are_common = mean_atom_dist < 0.5  # Most atoms are near reference

        if atoms_are_common:
            # Atoms are common, so arrangement novelty comes from structure
            arrangement_novelty = 0.7 * abs(centroid_distance_normalized) + 0.3 * abs(internal_diversity)
        else:
            # Atoms are already unusual, arrangement is secondary
            arrangement_novelty = 0.3 * abs(centroid_distance_normalized) + 0.2 * abs(internal_diversity)

        # Normalize to [0, 1]
        arrangement_novelty = np.tanh(arrangement_novelty)
        internal_diversity = np.tanh(internal_diversity)

        return arrangement_novelty, internal_diversity, centroid_distance_normalized

    def _compute_overall_novelty(
        self,
        novel_atom_fraction: float,
        arrangement_novelty: float,
        mean_atom_distance: float,
        centroid_distance: float,
    ) -> float:
        """Compute combined overall novelty score."""
        # Weighted combination
        overall = (
            0.3 * novel_atom_fraction +           # Novel atoms contribution
            0.3 * arrangement_novelty +           # Arrangement contribution
            0.2 * np.tanh(mean_atom_distance) +   # Atom distance contribution
            0.2 * np.tanh(abs(centroid_distance)) # Centroid contribution
        )
        return min(1.0, overall)

    def _classify_material(
        self,
        mean_atom_distance: float,
        novel_atom_fraction: float,
        arrangement_novelty: float,
        overall_novelty: float,
    ) -> Tuple[str, float]:
        """Classify material into fish-water category."""

        # Stability: based on how close atoms are to reference
        is_stable = mean_atom_distance < self.stability_threshold

        # Novelty: based on atoms OR arrangement
        is_novel = (novel_atom_fraction > 0.2) or (arrangement_novelty > self.arrangement_novelty_threshold)

        if is_stable:
            if is_novel:
                confidence = min(1.0, overall_novelty + 0.3)
                return "novel_fish", confidence
            else:
                confidence = min(1.0, (self.stability_threshold - mean_atom_distance) / self.stability_threshold + 0.3)
                return "fish_in_water", confidence
        else:
            if mean_atom_distance > 2 * self.stability_threshold:
                confidence = min(1.0, (mean_atom_distance - self.stability_threshold) / 2)
                return "not_a_fish", confidence
            else:
                confidence = min(1.0, (mean_atom_distance - self.stability_threshold) / self.stability_threshold + 0.3)
                return "fish_jumping_out", confidence

    # =========================================================================
    # Extended Arrangement Novelty Methods (More Rigorous)
    # =========================================================================

    def compute_pairwise_arrangement_novelty(
        self,
        atoms: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute arrangement novelty based on pairwise atom similarities.

        Idea: Two materials with same atoms but different arrangement
        will have different pairwise similarity patterns.

        Args:
            atoms: Atom embeddings for one material (n_atoms, dim)

        Returns:
            Dict with pairwise structure metrics
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.spatial.distance import pdist

        n_atoms = len(atoms)

        if n_atoms < 2:
            return {
                'mean_pairwise_sim': 1.0,
                'std_pairwise_sim': 0.0,
                'mean_pairwise_dist': 0.0,
                'std_pairwise_dist': 0.0,
            }

        # Pairwise cosine similarities
        sim_matrix = cosine_similarity(atoms)
        triu_idx = np.triu_indices(n_atoms, k=1)
        pairwise_sims = sim_matrix[triu_idx]

        # Pairwise euclidean distances
        pairwise_dists = pdist(atoms, metric='euclidean')

        return {
            'mean_pairwise_sim': float(np.mean(pairwise_sims)),
            'std_pairwise_sim': float(np.std(pairwise_sims)),
            'min_pairwise_sim': float(np.min(pairwise_sims)),
            'max_pairwise_sim': float(np.max(pairwise_sims)),
            'mean_pairwise_dist': float(np.mean(pairwise_dists)),
            'std_pairwise_dist': float(np.std(pairwise_dists)),
        }

    def compute_internal_structure_fingerprint(
        self,
        atoms: np.ndarray,
        n_bins: int = 20,
    ) -> np.ndarray:
        """
        Compute a fingerprint of the material's internal structure.

        The fingerprint is a histogram of pairwise distances between atoms.
        Materials with same atoms but different arrangements will have
        different fingerprints.

        Args:
            atoms: Atom embeddings (n_atoms, dim)
            n_bins: Number of histogram bins

        Returns:
            Normalized histogram (fingerprint)
        """
        from scipy.spatial.distance import pdist

        n_atoms = len(atoms)

        if n_atoms < 2:
            return np.zeros(n_bins)

        # Pairwise distances
        pairwise_dists = pdist(atoms, metric='euclidean')

        # Create histogram
        hist, _ = np.histogram(pairwise_dists, bins=n_bins, density=True)

        # Normalize
        hist = hist / (np.sum(hist) + 1e-8)

        return hist

    def compare_arrangement_to_reference(
        self,
        atoms: np.ndarray,
        reference_atom_embeddings_by_material: List[np.ndarray],
        method: str = 'fingerprint',
    ) -> Tuple[float, int]:
        """
        Compare a material's arrangement to reference materials.

        Args:
            atoms: Atom embeddings for generated material (n_atoms, dim)
            reference_atom_embeddings_by_material: List of arrays, one per ref material
            method: 'fingerprint' or 'pairwise'

        Returns:
            (min_distance, closest_ref_index) - lower distance = more similar arrangement
        """
        n_atoms = len(atoms)

        if method == 'fingerprint':
            gen_fingerprint = self.compute_internal_structure_fingerprint(atoms)

            min_dist = float('inf')
            closest_idx = -1

            for i, ref_atoms in enumerate(reference_atom_embeddings_by_material):
                ref_atoms = np.asarray(ref_atoms)
                if ref_atoms.ndim == 1:
                    ref_atoms = ref_atoms.reshape(1, -1)

                ref_fingerprint = self.compute_internal_structure_fingerprint(ref_atoms)

                # Compare fingerprints (histogram intersection or L2)
                dist = np.linalg.norm(gen_fingerprint - ref_fingerprint)

                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            return min_dist, closest_idx

        elif method == 'pairwise':
            gen_metrics = self.compute_pairwise_arrangement_novelty(atoms)

            min_dist = float('inf')
            closest_idx = -1

            for i, ref_atoms in enumerate(reference_atom_embeddings_by_material):
                ref_atoms = np.asarray(ref_atoms)
                if ref_atoms.ndim == 1:
                    ref_atoms = ref_atoms.reshape(1, -1)

                ref_metrics = self.compute_pairwise_arrangement_novelty(ref_atoms)

                # Compare metric vectors
                gen_vec = np.array([gen_metrics['mean_pairwise_sim'],
                                   gen_metrics['std_pairwise_sim'],
                                   gen_metrics['mean_pairwise_dist']])
                ref_vec = np.array([ref_metrics['mean_pairwise_sim'],
                                   ref_metrics['std_pairwise_sim'],
                                   ref_metrics['mean_pairwise_dist']])

                dist = np.linalg.norm(gen_vec - ref_vec)

                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            return min_dist, closest_idx
        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze_with_extended_arrangement(
        self,
        generated_atom_embeddings: List[np.ndarray],
        material_ids: List[str],
        reference_atom_embeddings_by_material: Optional[List[np.ndarray]] = None,
    ) -> List[Dict]:
        """
        Analyze materials with extended arrangement metrics.

        This method adds more rigorous arrangement novelty detection
        on top of the basic analysis.

        Args:
            generated_atom_embeddings: List of arrays, one per material
            material_ids: Material IDs
            reference_atom_embeddings_by_material: Optional list of ref materials
                                                   for arrangement comparison

        Returns:
            List of dicts with extended metrics for each material
        """
        # First run basic analysis
        basic_results = self.analyze(generated_atom_embeddings, material_ids)

        extended_results = []

        for i, (atoms_raw, mat_id) in enumerate(zip(generated_atom_embeddings, material_ids)):
            atoms = np.asarray(atoms_raw)
            if atoms.ndim == 1:
                atoms = atoms.reshape(1, -1)

            # Get basic result
            basic = next((r for r in basic_results if r.material_id == mat_id), None)

            # Compute extended pairwise metrics
            pairwise_metrics = self.compute_pairwise_arrangement_novelty(atoms)

            # Compute fingerprint
            fingerprint = self.compute_internal_structure_fingerprint(atoms)

            # Compare to references if provided
            arrangement_distance = None
            closest_ref_idx = None
            if reference_atom_embeddings_by_material is not None:
                arrangement_distance, closest_ref_idx = self.compare_arrangement_to_reference(
                    atoms, reference_atom_embeddings_by_material, method='fingerprint'
                )

            extended = {
                'material_id': mat_id,
                'n_atoms': len(atoms),
                # Basic results
                'category': basic.category if basic else None,
                'novelty_type': basic.novelty_type if basic else None,
                'novel_atom_fraction': basic.novel_atom_fraction if basic else None,
                'basic_arrangement_novelty': basic.arrangement_novelty if basic else None,
                # Extended pairwise metrics
                'mean_pairwise_similarity': pairwise_metrics['mean_pairwise_sim'],
                'std_pairwise_similarity': pairwise_metrics['std_pairwise_sim'],
                'mean_pairwise_distance': pairwise_metrics['mean_pairwise_dist'],
                'std_pairwise_distance': pairwise_metrics['std_pairwise_dist'],
                # Fingerprint-based comparison
                'arrangement_distance_to_ref': arrangement_distance,
                'closest_reference_idx': closest_ref_idx,
                # Raw fingerprint (for custom analysis)
                'structure_fingerprint': fingerprint,
            }
            extended_results.append(extended)

        return extended_results

    def detect_arrangement_novelty_detailed(
        self,
        atoms: np.ndarray,
        reference_materials: List[np.ndarray],
        threshold_percentile: float = 90,
    ) -> Dict:
        """
        Detailed arrangement novelty detection for a single material.

        Compares the internal structure pattern to all reference materials
        and determines if the arrangement is novel.

        Args:
            atoms: Atom embeddings for one material
            reference_materials: List of reference material atom embeddings
            threshold_percentile: Percentile above which arrangement is "novel"

        Returns:
            Dict with detailed novelty assessment
        """
        atoms = np.asarray(atoms)
        if atoms.ndim == 1:
            atoms = atoms.reshape(1, -1)

        # Compute this material's metrics
        gen_pairwise = self.compute_pairwise_arrangement_novelty(atoms)
        gen_fingerprint = self.compute_internal_structure_fingerprint(atoms)

        # Compute same metrics for all reference materials
        ref_fingerprints = []
        ref_pairwise_means = []
        ref_pairwise_stds = []

        for ref_atoms in reference_materials:
            ref_atoms = np.asarray(ref_atoms)
            if ref_atoms.ndim == 1:
                ref_atoms = ref_atoms.reshape(1, -1)

            ref_fp = self.compute_internal_structure_fingerprint(ref_atoms)
            ref_pw = self.compute_pairwise_arrangement_novelty(ref_atoms)

            ref_fingerprints.append(ref_fp)
            ref_pairwise_means.append(ref_pw['mean_pairwise_sim'])
            ref_pairwise_stds.append(ref_pw['std_pairwise_sim'])

        # Compare to references
        fingerprint_distances = [
            np.linalg.norm(gen_fingerprint - ref_fp)
            for ref_fp in ref_fingerprints
        ]

        min_fp_distance = np.min(fingerprint_distances)
        mean_fp_distance = np.mean(fingerprint_distances)

        # Determine threshold from reference-to-reference distances
        ref_ref_distances = []
        for i, fp1 in enumerate(ref_fingerprints):
            for j, fp2 in enumerate(ref_fingerprints):
                if i < j:
                    ref_ref_distances.append(np.linalg.norm(fp1 - fp2))

        if ref_ref_distances:
            distance_threshold = np.percentile(ref_ref_distances, threshold_percentile)
            is_novel_arrangement = min_fp_distance > distance_threshold
        else:
            distance_threshold = None
            is_novel_arrangement = False

        # Check if pairwise statistics are unusual
        pairwise_mean_percentile = np.mean(
            [gen_pairwise['mean_pairwise_sim'] > ref_m for ref_m in ref_pairwise_means]
        ) * 100

        return {
            'min_fingerprint_distance': min_fp_distance,
            'mean_fingerprint_distance': mean_fp_distance,
            'distance_threshold': distance_threshold,
            'is_novel_arrangement': is_novel_arrangement,
            'pairwise_mean_percentile': pairwise_mean_percentile,
            'gen_mean_pairwise_sim': gen_pairwise['mean_pairwise_sim'],
            'gen_std_pairwise_sim': gen_pairwise['std_pairwise_sim'],
            'closest_reference_distance': min_fp_distance,
            'closest_reference_idx': int(np.argmin(fingerprint_distances)),
        }

    # =========================================================================
    # Output Methods
    # =========================================================================

    def get_ids_by_category(
        self,
        results: List[MaterialFishResult],
        category: str,
    ) -> List[str]:
        """Get material IDs for a specific category."""
        return [r.material_id for r in results if r.category == category]

    def get_ids_by_novelty_type(
        self,
        results: List[MaterialFishResult],
        novelty_type: str,
    ) -> List[str]:
        """Get material IDs for a specific novelty type."""
        return [r.material_id for r in results if r.novelty_type == novelty_type]

    def get_all_ids_by_category(
        self,
        results: List[MaterialFishResult],
    ) -> Dict[str, List[str]]:
        """Get all material IDs grouped by category."""
        categories = ["fish_in_water", "novel_fish", "fish_jumping_out", "not_a_fish"]
        return {cat: self.get_ids_by_category(results, cat) for cat in categories}

    def get_all_ids_by_novelty_type(
        self,
        results: List[MaterialFishResult],
    ) -> Dict[str, List[str]]:
        """Get all material IDs grouped by novelty type."""
        types = ["novel_atoms", "novel_arrangement", "both", "neither"]
        return {t: self.get_ids_by_novelty_type(results, t) for t in types}

    def get_summary(self, results: List[MaterialFishResult]) -> Dict:
        """Get summary statistics."""
        n = len(results)
        if n == 0:
            return {"total": 0}

        categories = [r.category for r in results]
        novelty_types = [r.novelty_type for r in results]

        summary = {
            "total": n,
            # Categories
            "fish_in_water": categories.count("fish_in_water"),
            "novel_fish": categories.count("novel_fish"),
            "fish_jumping_out": categories.count("fish_jumping_out"),
            "not_a_fish": categories.count("not_a_fish"),
            # Novelty types
            "novel_atoms": novelty_types.count("novel_atoms"),
            "novel_arrangement": novelty_types.count("novel_arrangement"),
            "both": novelty_types.count("both"),
            "neither": novelty_types.count("neither"),
            # Statistics
            "mean_novel_atom_fraction": np.mean([r.novel_atom_fraction for r in results]),
            "mean_arrangement_novelty": np.mean([r.arrangement_novelty for r in results]),
            "mean_overall_novelty": np.mean([r.overall_novelty for r in results]),
        }

        # Percentages
        for key in ["fish_in_water", "novel_fish", "fish_jumping_out", "not_a_fish",
                    "novel_atoms", "novel_arrangement", "both", "neither"]:
            summary[f"{key}_pct"] = 100 * summary[key] / n

        return summary

    def print_summary(self, results: List[MaterialFishResult]):
        """Print formatted summary."""
        summary = self.get_summary(results)
        ids_by_cat = self.get_all_ids_by_category(results)
        ids_by_type = self.get_all_ids_by_novelty_type(results)

        print("=" * 70)
        print("Material-Level Fish-Water Analysis")
        print("=" * 70)
        print(f"\nTotal materials analyzed: {summary['total']}\n")

        print("Fish-Water Categories:")
        print("-" * 50)
        cat_info = [
            ("fish_in_water", "Fish in Water    (stable, known)"),
            ("novel_fish", "Novel Fish       (stable, new)"),
            ("fish_jumping_out", "Fish Jumping Out (unstable)"),
            ("not_a_fish", "Not a Fish       (invalid)"),
        ]
        for cat_key, cat_label in cat_info:
            count = summary[cat_key]
            pct = summary[f"{cat_key}_pct"]
            ids = ids_by_cat[cat_key]
            print(f"  {cat_label}: {count:>5} ({pct:>5.1f}%)")
            if ids:
                example_ids = ids[:3]
                ids_str = ", ".join(example_ids)
                if len(ids) > 3:
                    ids_str += f", ... (+{len(ids)-3} more)"
                print(f"      Examples: {ids_str}")

        print("\nNovelty Types (WHY it's novel):")
        print("-" * 50)
        type_info = [
            ("novel_atoms", "Novel Atoms       (unusual atomic environments)"),
            ("novel_arrangement", "Novel Arrangement (common atoms, unusual config)"),
            ("both", "Both              (novel atoms AND arrangement)"),
            ("neither", "Neither           (not novel)"),
        ]
        for type_key, type_label in type_info:
            count = summary[type_key]
            pct = summary[f"{type_key}_pct"]
            ids = ids_by_type[type_key]
            print(f"  {type_label}: {count:>5} ({pct:>5.1f}%)")

        print("\nStatistics:")
        print("-" * 50)
        print(f"  Mean novel atom fraction:    {summary['mean_novel_atom_fraction']:>6.1%}")
        print(f"  Mean arrangement novelty:    {summary['mean_arrangement_novelty']:>6.3f}")
        print(f"  Mean overall novelty:        {summary['mean_overall_novelty']:>6.3f}")

        print("\nRecommendations:")
        print("-" * 50)
        if summary['novel_fish'] > 0:
            print(f"  -> {summary['novel_fish']} novel fish candidates for DFT!")
            if summary['novel_arrangement'] > 0:
                print(f"     - {summary['novel_arrangement']} have novel ARRANGEMENTS (common atoms, new config)")
            if summary['novel_atoms'] > 0:
                print(f"     - {summary['novel_atoms']} have novel ATOMS (unusual environments)")

    def export_results(
        self,
        results: List[MaterialFishResult],
        output_path: str,
    ):
        """Export results to CSV."""
        data = []
        for r in results:
            row = {
                'material_id': r.material_id,
                'n_atoms': r.n_atoms,
                'category': r.category,
                'novelty_type': r.novelty_type,
                'confidence': r.confidence,
                'overall_novelty': r.overall_novelty,
                'novel_atom_fraction': r.novel_atom_fraction,
                'mean_atom_distance': r.mean_atom_distance,
                'max_atom_distance': r.max_atom_distance,
                'arrangement_novelty': r.arrangement_novelty,
                'internal_diversity': r.internal_diversity,
                'centroid_distance': r.centroid_distance,
                'n_novel_atoms': len(r.novel_atom_indices) if r.novel_atom_indices else 0,
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

    def plot_distribution(
        self,
        results: List[MaterialFishResult],
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ):
        """Plot distribution of materials."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Extract data
        novel_atom_fracs = np.array([r.novel_atom_fraction for r in results])
        arrangement_novelties = np.array([r.arrangement_novelty for r in results])
        overall_novelties = np.array([r.overall_novelty for r in results])
        mean_distances = np.array([r.mean_atom_distance for r in results])
        categories = np.array([r.category for r in results])
        novelty_types = np.array([r.novelty_type for r in results])

        cat_colors = {
            "fish_in_water": "#2ecc71",
            "novel_fish": "#3498db",
            "fish_jumping_out": "#e74c3c",
            "not_a_fish": "#95a5a6",
        }

        type_colors = {
            "novel_atoms": "#e74c3c",
            "novel_arrangement": "#9b59b6",
            "both": "#e67e22",
            "neither": "#95a5a6",
        }

        # 1. Category pie chart
        ax1 = axes[0, 0]
        summary = self.get_summary(results)
        sizes = [summary[cat] for cat in cat_colors.keys()]
        ax1.pie(sizes, labels=[c.replace("_", " ").title() for c in cat_colors.keys()],
                colors=cat_colors.values(), autopct=lambda p: f'{p:.1f}%' if p > 0 else '')
        ax1.set_title("Fish-Water Categories", fontweight='bold')

        # 2. Novelty type pie chart
        ax2 = axes[0, 1]
        sizes = [summary[t] for t in type_colors.keys()]
        ax2.pie(sizes, labels=[t.replace("_", " ").title() for t in type_colors.keys()],
                colors=type_colors.values(), autopct=lambda p: f'{p:.1f}%' if p > 0 else '')
        ax2.set_title("Novelty Types", fontweight='bold')

        # 3. Novel atoms vs Arrangement novelty scatter
        ax3 = axes[0, 2]
        for cat in cat_colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax3.scatter(novel_atom_fracs[mask], arrangement_novelties[mask],
                           c=cat_colors[cat], label=cat.replace("_", " ").title(),
                           alpha=0.6, s=30)
        ax3.axvline(x=0.2, color='black', linestyle='--', alpha=0.5, label='Atom threshold')
        ax3.axhline(y=self.arrangement_novelty_threshold, color='black', linestyle=':', alpha=0.5)
        ax3.set_xlabel("Novel Atom Fraction", fontsize=10)
        ax3.set_ylabel("Arrangement Novelty", fontsize=10)
        ax3.set_title("Atom vs Arrangement Novelty", fontweight='bold')
        ax3.legend(fontsize=8)

        # Quadrant labels
        ax3.text(0.05, 0.95, "Novel\nArrangement", transform=ax3.transAxes, fontsize=8, color='#9b59b6')
        ax3.text(0.75, 0.95, "Both", transform=ax3.transAxes, fontsize=8, color='#e67e22')
        ax3.text(0.05, 0.05, "Neither", transform=ax3.transAxes, fontsize=8, color='#95a5a6')
        ax3.text(0.75, 0.05, "Novel\nAtoms", transform=ax3.transAxes, fontsize=8, color='#e74c3c')

        # 4. Histogram of novel atom fraction
        ax4 = axes[1, 0]
        for cat in cat_colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax4.hist(novel_atom_fracs[mask], bins=20, alpha=0.5, color=cat_colors[cat],
                        label=cat.replace("_", " ").title())
        ax4.axvline(x=0.2, color='black', linestyle='--')
        ax4.set_xlabel("Novel Atom Fraction", fontsize=10)
        ax4.set_ylabel("Count", fontsize=10)
        ax4.set_title("Distribution of Novel Atom Fraction", fontweight='bold')
        ax4.legend(fontsize=8)

        # 5. Histogram of arrangement novelty
        ax5 = axes[1, 1]
        for cat in cat_colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax5.hist(arrangement_novelties[mask], bins=20, alpha=0.5, color=cat_colors[cat],
                        label=cat.replace("_", " ").title())
        ax5.axvline(x=self.arrangement_novelty_threshold, color='black', linestyle='--')
        ax5.set_xlabel("Arrangement Novelty", fontsize=10)
        ax5.set_ylabel("Count", fontsize=10)
        ax5.set_title("Distribution of Arrangement Novelty", fontweight='bold')
        ax5.legend(fontsize=8)

        # 6. Mean atom distance vs Overall novelty
        ax6 = axes[1, 2]
        for ntype in type_colors.keys():
            mask = novelty_types == ntype
            if np.sum(mask) > 0:
                ax6.scatter(mean_distances[mask], overall_novelties[mask],
                           c=type_colors[ntype], label=ntype.replace("_", " ").title(),
                           alpha=0.6, s=30)
        ax6.axvline(x=self.stability_threshold, color='black', linestyle='--', alpha=0.5)
        ax6.set_xlabel("Mean Atom Distance", fontsize=10)
        ax6.set_ylabel("Overall Novelty", fontsize=10)
        ax6.set_title("Stability vs Novelty by Type", fontweight='bold')
        ax6.legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig

    def get_top_candidates(
        self,
        results: List[MaterialFishResult],
        category: str = "novel_fish",
        novelty_type: Optional[str] = None,
        top_k: int = 10,
    ) -> List[MaterialFishResult]:
        """Get top candidates, optionally filtered by novelty type."""
        filtered = [r for r in results if r.category == category]

        if novelty_type is not None:
            filtered = [r for r in filtered if r.novelty_type == novelty_type]

        sorted_results = sorted(filtered, key=lambda x: x.overall_novelty, reverse=True)
        return sorted_results[:top_k]


# =============================================================================
# Mean-Pooled Material Analysis (Simpler, More Efficient)
# =============================================================================

def mean_pool_atoms_to_materials(
    atom_embeddings: np.ndarray,
    atom_material_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Mean pool atomic embeddings to get material-level embeddings.

    Atoms with the same material ID are averaged together.

    Args:
        atom_embeddings: All atom embeddings, shape (n_atoms, embedding_dim)
        atom_material_ids: Material ID for each atom (atoms from same material share ID)

    Returns:
        (material_embeddings, unique_material_ids)
        - material_embeddings: shape (n_materials, embedding_dim)
        - unique_material_ids: list of material IDs in same order
    """
    atom_embeddings = np.asarray(atom_embeddings)

    # Group atoms by material ID
    material_to_indices = {}
    for i, mat_id in enumerate(atom_material_ids):
        if mat_id not in material_to_indices:
            material_to_indices[mat_id] = []
        material_to_indices[mat_id].append(i)

    # Mean pool each material
    unique_material_ids = list(material_to_indices.keys())
    material_embeddings = []

    for mat_id in unique_material_ids:
        indices = material_to_indices[mat_id]
        mat_atoms = atom_embeddings[indices]
        mat_embedding = np.mean(mat_atoms, axis=0)
        material_embeddings.append(mat_embedding)

    return np.array(material_embeddings), unique_material_ids


def mean_pool_from_list(
    atom_embeddings_list: List[np.ndarray],
    material_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Mean pool atomic embeddings from a list format.

    Args:
        atom_embeddings_list: List of arrays, one per material
                              Each array has shape (n_atoms_i, embedding_dim)
        material_ids: Material ID for each material

    Returns:
        (material_embeddings, material_ids)
    """
    material_embeddings = []

    for atoms in atom_embeddings_list:
        atoms = np.asarray(atoms)
        if atoms.ndim == 1:
            atoms = atoms.reshape(1, -1)  # Single atom case
        mat_embedding = np.mean(atoms, axis=0)
        material_embeddings.append(mat_embedding)

    return np.array(material_embeddings), list(material_ids)


@dataclass
class MeanPooledMaterialResult:
    """Results for a material using mean-pooled analysis."""
    material_id: str
    n_atoms: int                    # Number of atoms in this material

    # From FishWaterAnalyzer
    manifold_distance: float        # Distance to reference manifold
    novelty_score: float            # How different from neighbors
    lof_score: float                # Local outlier factor
    density_score: float            # Local density estimate
    category: str                   # Fish classification
    confidence: float               # Confidence in classification
    nearest_reference_ids: List[str] = None  # Nearest reference material IDs


class MeanPooledMaterialAnalyzer:
    """
    Material-level Fish-Water analysis using mean-pooled atomic embeddings.

    This is a simpler and more efficient approach than pairwise arrangement comparison:
    1. Mean pool atomic embeddings for each material
    2. Use the same FishWaterAnalyzer approach on material-level embeddings

    Why this works:
    - Materials with similar atoms AND arrangements will have similar mean embeddings
    - Materials with novel arrangements will have different mean embeddings even if
      individual atoms are common
    - Much faster than pairwise comparisons (O(n*m) vs O(n*m^2))

    Example:
        >>> # Reference: atom embeddings with material IDs
        >>> ref_atom_emb = np.array([...])  # (n_ref_atoms, dim)
        >>> ref_atom_mat_ids = ['mp-001', 'mp-001', 'mp-002', ...]  # material ID per atom
        >>>
        >>> # Generated: atom embeddings with material IDs
        >>> gen_atom_emb = np.array([...])  # (n_gen_atoms, dim)
        >>> gen_atom_mat_ids = ['gen-001', 'gen-001', 'gen-001', 'gen-002', ...]
        >>>
        >>> analyzer = MeanPooledMaterialAnalyzer(ref_atom_emb, ref_atom_mat_ids)
        >>> results = analyzer.analyze(gen_atom_emb, gen_atom_mat_ids)
        >>> analyzer.print_summary(results)
    """

    def __init__(
        self,
        reference_atom_embeddings: np.ndarray,
        reference_atom_material_ids: List[str],
        n_neighbors: int = 10,
        stability_threshold: float = 0.5,
        novelty_threshold: float = 0.3,
        outlier_threshold: float = -1.5,
    ):
        """
        Initialize with reference atomic embeddings.

        Args:
            reference_atom_embeddings: All atom embeddings from reference materials
                                       Shape: (n_ref_atoms, embedding_dim)
            reference_atom_material_ids: Material ID for each reference atom
                                         Atoms from same material share the same ID
            n_neighbors: Number of neighbors for analysis
            stability_threshold: Max distance to be considered stable
            novelty_threshold: Min novelty to be considered novel
            outlier_threshold: LOF score below this = invalid
        """
        self.ref_atom_embeddings = np.asarray(reference_atom_embeddings)
        self.ref_atom_material_ids = list(reference_atom_material_ids)

        # Mean pool reference atoms to materials
        self.ref_material_embeddings, self.ref_material_ids = mean_pool_atoms_to_materials(
            self.ref_atom_embeddings,
            self.ref_atom_material_ids,
        )

        self.n_ref_materials = len(self.ref_material_ids)

        # Count atoms per reference material
        self.ref_atoms_per_material = {}
        for mat_id in self.ref_atom_material_ids:
            self.ref_atoms_per_material[mat_id] = self.ref_atoms_per_material.get(mat_id, 0) + 1

        # Create internal FishWaterAnalyzer on material embeddings
        self.fish_analyzer = FishWaterAnalyzer(
            reference_embeddings=self.ref_material_embeddings,
            reference_ids=self.ref_material_ids,
            n_neighbors=min(n_neighbors, self.n_ref_materials - 1),
            stability_threshold=stability_threshold,
            novelty_threshold=novelty_threshold,
            outlier_threshold=outlier_threshold,
        )

        print(f"Mean-pooled {len(self.ref_atom_embeddings)} atoms -> "
              f"{self.n_ref_materials} reference materials")

    def analyze(
        self,
        generated_atom_embeddings: np.ndarray,
        generated_atom_material_ids: List[str],
    ) -> List[MeanPooledMaterialResult]:
        """
        Analyze generated materials using mean-pooled approach.

        Args:
            generated_atom_embeddings: All atom embeddings from generated materials
                                       Shape: (n_gen_atoms, embedding_dim)
            generated_atom_material_ids: Material ID for each generated atom
                                         Atoms from same material share the same ID

        Returns:
            List of MeanPooledMaterialResult for each generated material
        """
        gen_atom_embeddings = np.asarray(generated_atom_embeddings)
        gen_atom_material_ids = list(generated_atom_material_ids)

        # Mean pool generated atoms to materials
        gen_material_embeddings, gen_material_ids = mean_pool_atoms_to_materials(
            gen_atom_embeddings,
            gen_atom_material_ids,
        )

        # Count atoms per generated material
        gen_atoms_per_material = {}
        for mat_id in gen_atom_material_ids:
            gen_atoms_per_material[mat_id] = gen_atoms_per_material.get(mat_id, 0) + 1

        n_gen_materials = len(gen_material_ids)
        print(f"Mean-pooled {len(gen_atom_embeddings)} atoms -> "
              f"{n_gen_materials} generated materials")

        # Use FishWaterAnalyzer on material embeddings
        fish_results = self.fish_analyzer.analyze(
            generated_embeddings=gen_material_embeddings,
            material_ids=gen_material_ids,
        )

        # Convert to MeanPooledMaterialResult
        results = []
        for fr in fish_results:
            result = MeanPooledMaterialResult(
                material_id=fr.material_id,
                n_atoms=gen_atoms_per_material.get(fr.material_id, 0),
                manifold_distance=fr.manifold_distance,
                novelty_score=fr.novelty_score,
                lof_score=fr.lof_score,
                density_score=fr.density_score,
                category=fr.category,
                confidence=fr.confidence,
                nearest_reference_ids=fr.nearest_reference_ids,
            )
            results.append(result)

        self._last_results = results
        self._last_material_embeddings = gen_material_embeddings
        self._last_material_ids = gen_material_ids

        return results

    def analyze_from_list(
        self,
        generated_atom_embeddings_list: List[np.ndarray],
        material_ids: List[str],
    ) -> List[MeanPooledMaterialResult]:
        """
        Analyze generated materials from list format (one array per material).

        This is a convenience method when you have atoms grouped by material already.

        Args:
            generated_atom_embeddings_list: List of arrays, one per material
                                            Each array has shape (n_atoms_i, embedding_dim)
            material_ids: Material ID for each material

        Returns:
            List of MeanPooledMaterialResult for each material
        """
        # Mean pool from list
        gen_material_embeddings, gen_material_ids = mean_pool_from_list(
            generated_atom_embeddings_list,
            material_ids,
        )

        # Get atom counts
        gen_atoms_per_material = {}
        for i, mat_id in enumerate(material_ids):
            atoms = generated_atom_embeddings_list[i]
            if isinstance(atoms, np.ndarray):
                n_atoms = 1 if atoms.ndim == 1 else len(atoms)
            else:
                n_atoms = len(atoms)
            gen_atoms_per_material[mat_id] = n_atoms

        print(f"Mean-pooled {len(material_ids)} materials from list format")

        # Use FishWaterAnalyzer
        fish_results = self.fish_analyzer.analyze(
            generated_embeddings=gen_material_embeddings,
            material_ids=gen_material_ids,
        )

        # Convert to MeanPooledMaterialResult
        results = []
        for fr in fish_results:
            result = MeanPooledMaterialResult(
                material_id=fr.material_id,
                n_atoms=gen_atoms_per_material.get(fr.material_id, 0),
                manifold_distance=fr.manifold_distance,
                novelty_score=fr.novelty_score,
                lof_score=fr.lof_score,
                density_score=fr.density_score,
                category=fr.category,
                confidence=fr.confidence,
                nearest_reference_ids=fr.nearest_reference_ids,
            )
            results.append(result)

        return results

    # =========================================================================
    # Delegation methods (expose FishWaterAnalyzer functionality)
    # =========================================================================

    def get_ids_by_category(
        self,
        results: List[MeanPooledMaterialResult],
        category: str,
    ) -> List[str]:
        """Get material IDs for a specific category."""
        return [r.material_id for r in results if r.category == category]

    def get_all_ids_by_category(
        self,
        results: List[MeanPooledMaterialResult],
    ) -> Dict[str, List[str]]:
        """Get all material IDs grouped by category."""
        categories = ["fish_in_water", "novel_fish", "fish_jumping_out", "not_a_fish"]
        return {cat: self.get_ids_by_category(results, cat) for cat in categories}

    def get_summary(self, results: List[MeanPooledMaterialResult]) -> Dict:
        """Get summary statistics."""
        n = len(results)
        if n == 0:
            return {"total": 0}

        categories = [r.category for r in results]

        summary = {
            "total": n,
            "fish_in_water": categories.count("fish_in_water"),
            "novel_fish": categories.count("novel_fish"),
            "fish_jumping_out": categories.count("fish_jumping_out"),
            "not_a_fish": categories.count("not_a_fish"),
            "mean_manifold_distance": np.mean([r.manifold_distance for r in results]),
            "mean_novelty": np.mean([r.novelty_score for r in results]),
            "mean_lof": np.mean([r.lof_score for r in results]),
            "mean_atoms_per_material": np.mean([r.n_atoms for r in results]),
        }

        for cat in ["fish_in_water", "novel_fish", "fish_jumping_out", "not_a_fish"]:
            summary[f"{cat}_pct"] = 100 * summary[cat] / n if n > 0 else 0

        return summary

    def print_summary(self, results: List[MeanPooledMaterialResult]):
        """Print formatted summary."""
        summary = self.get_summary(results)
        ids_by_cat = self.get_all_ids_by_category(results)

        print("=" * 70)
        print("Mean-Pooled Material Fish-Water Analysis")
        print("=" * 70)
        print(f"\nTotal materials analyzed: {summary['total']}")
        print(f"Mean atoms per material: {summary['mean_atoms_per_material']:.1f}\n")

        print("Category Distribution:")
        print("-" * 50)

        cat_info = [
            ("fish_in_water", "Fish in Water    (stable, known)"),
            ("novel_fish", "Novel Fish       (stable, new)"),
            ("fish_jumping_out", "Fish Jumping Out (unstable)"),
            ("not_a_fish", "Not a Fish       (invalid)"),
        ]

        for cat_key, cat_label in cat_info:
            count = summary[cat_key]
            pct = summary[f"{cat_key}_pct"]
            ids = ids_by_cat[cat_key]

            print(f"  {cat_label}: {count:>5} ({pct:>5.1f}%)")
            if ids:
                example_ids = ids[:3]
                ids_str = ", ".join(example_ids)
                if len(ids) > 3:
                    ids_str += f", ... (+{len(ids)-3} more)"
                print(f"      Examples: {ids_str}")

        print("\nMetric Statistics:")
        print("-" * 50)
        print(f"  Mean manifold distance: {summary['mean_manifold_distance']:>8.3f}")
        print(f"  Mean novelty score:     {summary['mean_novelty']:>8.3f}")
        print(f"  Mean LOF score:         {summary['mean_lof']:>8.3f}")

        print("\nRecommendations:")
        print("-" * 50)
        if summary['novel_fish'] > 0:
            print(f"  -> {summary['novel_fish']} novel materials for DFT validation!")
        if summary['fish_jumping_out_pct'] > 30:
            print(f"  -> High unstable fraction ({summary['fish_jumping_out_pct']:.1f}%)")
        if summary['not_a_fish_pct'] > 10:
            print(f"  -> {summary['not_a_fish_pct']:.1f}% invalid - filter these out")

    def get_top_candidates(
        self,
        results: List[MeanPooledMaterialResult],
        category: str = "novel_fish",
        top_k: int = 10,
    ) -> List[MeanPooledMaterialResult]:
        """Get top candidates from a category, sorted by confidence."""
        filtered = [r for r in results if r.category == category]
        sorted_results = sorted(filtered, key=lambda x: x.confidence, reverse=True)
        return sorted_results[:top_k]

    def export_results(
        self,
        results: List[MeanPooledMaterialResult],
        output_path: str,
    ):
        """Export results to CSV."""
        data = []
        for r in results:
            row = {
                'material_id': r.material_id,
                'n_atoms': r.n_atoms,
                'category': r.category,
                'manifold_distance': r.manifold_distance,
                'novelty_score': r.novelty_score,
                'lof_score': r.lof_score,
                'density_score': r.density_score,
                'confidence': r.confidence,
                'nearest_ref_1': r.nearest_reference_ids[0] if r.nearest_reference_ids else None,
                'nearest_ref_2': r.nearest_reference_ids[1] if r.nearest_reference_ids and len(r.nearest_reference_ids) > 1 else None,
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

    def plot_distribution(
        self,
        results: List[MeanPooledMaterialResult],
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ):
        """Plot distribution of materials by category."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        distances = np.array([r.manifold_distance for r in results])
        novelties = np.array([r.novelty_score for r in results])
        n_atoms = np.array([r.n_atoms for r in results])
        categories = np.array([r.category for r in results])

        colors = {
            "fish_in_water": "#2ecc71",
            "novel_fish": "#3498db",
            "fish_jumping_out": "#e74c3c",
            "not_a_fish": "#95a5a6",
        }

        # 1. Pie chart
        ax1 = axes[0, 0]
        summary = self.get_summary(results)
        sizes = [summary[cat] for cat in colors.keys()]
        ax1.pie(sizes, labels=[c.replace("_", " ").title() for c in colors.keys()],
                colors=colors.values(), autopct=lambda p: f'{p:.1f}%' if p > 0 else '')
        ax1.set_title("Category Distribution", fontweight='bold')

        # 2. Scatter: distance vs novelty
        ax2 = axes[0, 1]
        for cat in colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax2.scatter(distances[mask], novelties[mask],
                           c=colors[cat], label=cat.replace("_", " ").title(),
                           alpha=0.6, s=30)
        ax2.axvline(x=self.fish_analyzer.stability_threshold, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.fish_analyzer.novelty_threshold, color='black', linestyle=':', alpha=0.5)
        ax2.set_xlabel("Manifold Distance", fontsize=10)
        ax2.set_ylabel("Novelty Score", fontsize=10)
        ax2.set_title("Fish-Water Space (Mean-Pooled)", fontweight='bold')
        ax2.legend(fontsize=8)

        # 3. Histogram of distances
        ax3 = axes[1, 0]
        for cat in colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax3.hist(distances[mask], bins=20, alpha=0.5, color=colors[cat],
                        label=cat.replace("_", " ").title())
        ax3.axvline(x=self.fish_analyzer.stability_threshold, color='black', linestyle='--')
        ax3.set_xlabel("Manifold Distance", fontsize=10)
        ax3.set_ylabel("Count", fontsize=10)
        ax3.set_title("Distribution of Manifold Distance", fontweight='bold')
        ax3.legend(fontsize=8)

        # 4. Distance vs n_atoms (to see if material size affects classification)
        ax4 = axes[1, 1]
        for cat in colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax4.scatter(n_atoms[mask], distances[mask],
                           c=colors[cat], label=cat.replace("_", " ").title(),
                           alpha=0.6, s=30)
        ax4.set_xlabel("Number of Atoms", fontsize=10)
        ax4.set_ylabel("Manifold Distance", fontsize=10)
        ax4.set_title("Material Size vs Distance", fontweight='bold')
        ax4.legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig


# =============================================================================
# Example Usage
# =============================================================================

def example_with_mock_data():
    """Example with mock atom-level data."""
    print("=" * 70)
    print("Material-Level Fish-Water Analysis - Example")
    print("=" * 70)

    np.random.seed(42)

    # Create reference atom embeddings (from known materials)
    # Simulate 100 reference materials with varying number of atoms
    n_ref_materials = 100
    embedding_dim = 64

    reference_atom_embeddings = []
    reference_material_ids = []

    for i in range(n_ref_materials):
        n_atoms = np.random.randint(5, 20)
        # Atoms cluster around material-specific centers
        center = np.random.randn(embedding_dim) * 0.3
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.2
        reference_atom_embeddings.append(atoms)
        reference_material_ids.extend([f"mp-{i:05d}"] * n_atoms)

    reference_atoms_flat = np.vstack(reference_atom_embeddings)

    print(f"Reference: {n_ref_materials} materials, {len(reference_atoms_flat)} total atoms")

    # Create generated materials (mix of types)
    n_gen_materials = 50
    generated_atom_embeddings = []
    material_ids = []

    # 20 "fish in water" - similar atoms and arrangement
    for i in range(20):
        n_atoms = np.random.randint(5, 15)
        center = np.random.randn(embedding_dim) * 0.3
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.2
        generated_atom_embeddings.append(atoms)
        material_ids.append(f"gen-water-{i:03d}")

    # 10 "novel atoms" - unusual atomic environments
    for i in range(10):
        n_atoms = np.random.randint(5, 15)
        center = np.random.randn(embedding_dim) * 0.3
        # Some atoms are far from reference
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.2
        atoms[:3] += np.random.randn(3, embedding_dim) * 1.5  # 3 novel atoms
        generated_atom_embeddings.append(atoms)
        material_ids.append(f"gen-novel-atoms-{i:03d}")

    # 10 "novel arrangement" - common atoms, unusual config
    for i in range(10):
        n_atoms = np.random.randint(5, 15)
        # Atoms are common but spread unusually
        center = np.random.randn(embedding_dim) * 1.0  # Unusual center
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.15  # Common atom types
        generated_atom_embeddings.append(atoms)
        material_ids.append(f"gen-novel-arr-{i:03d}")

    # 10 "jumping out" - far from reference
    for i in range(10):
        n_atoms = np.random.randint(5, 15)
        center = np.random.randn(embedding_dim) * 2.0
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.5
        generated_atom_embeddings.append(atoms)
        material_ids.append(f"gen-jumping-{i:03d}")

    print(f"Generated: {len(generated_atom_embeddings)} materials")

    # Shuffle
    shuffle_idx = np.random.permutation(n_gen_materials)
    generated_atom_embeddings = [generated_atom_embeddings[i] for i in shuffle_idx]
    material_ids = [material_ids[i] for i in shuffle_idx]

    # Run analysis
    print("\nRunning Material-Level Fish-Water analysis...")

    analyzer = MaterialFishAnalyzer(
        reference_atom_embeddings=reference_atoms_flat,
        reference_material_ids=reference_material_ids,
        n_neighbors=10,
        atom_novelty_threshold=0.8,
        arrangement_novelty_threshold=0.4,
        stability_threshold=0.5,
    )

    results = analyzer.analyze(
        generated_atom_embeddings=generated_atom_embeddings,
        material_ids=material_ids,
    )

    # Print summary
    analyzer.print_summary(results)

    # Show top novel fish by type
    print("\n" + "=" * 70)
    print("Top Novel Fish - By Novelty Type")
    print("=" * 70)

    for ntype in ["novel_atoms", "novel_arrangement", "both"]:
        top = analyzer.get_top_candidates(results, "novel_fish", novelty_type=ntype, top_k=3)
        if top:
            print(f"\n{ntype.upper()}:")
            for r in top:
                print(f"  {r.material_id}: novelty={r.overall_novelty:.3f}, "
                      f"novel_atoms={r.novel_atom_fraction:.1%}, "
                      f"arr_novelty={r.arrangement_novelty:.3f}")

    # Export
    analyzer.export_results(results, "material_fish_results.csv")

    # Plot
    print("\nGenerating plots...")
    analyzer.plot_distribution(results, save_path="material_fish_analysis.png")

    return analyzer, results


def example_mean_pooled():
    """
    Example using the Mean-Pooled Material Analyzer.

    This is the recommended approach when you have atomic embeddings
    and want to classify materials. It's simpler and more efficient
    than pairwise arrangement comparison.
    """
    print("=" * 70)
    print("Mean-Pooled Material Analysis - Example")
    print("=" * 70)

    np.random.seed(42)
    embedding_dim = 64

    # =========================================================================
    # Create reference atoms with material IDs
    # =========================================================================
    print("\n1. Creating reference atom embeddings...")

    n_ref_materials = 100
    ref_atom_embeddings = []
    ref_atom_material_ids = []

    for i in range(n_ref_materials):
        n_atoms = np.random.randint(5, 20)
        mat_id = f"mp-{i:05d}"

        # Atoms cluster around material-specific center
        center = np.random.randn(embedding_dim) * 0.3
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.2

        for atom in atoms:
            ref_atom_embeddings.append(atom)
            ref_atom_material_ids.append(mat_id)

    ref_atom_embeddings = np.array(ref_atom_embeddings)
    print(f"   Reference: {len(ref_atom_embeddings)} atoms from {n_ref_materials} materials")

    # =========================================================================
    # Create generated atoms with material IDs
    # =========================================================================
    print("\n2. Creating generated atom embeddings...")

    gen_atom_embeddings = []
    gen_atom_material_ids = []

    # 20 "fish in water" materials
    for i in range(20):
        n_atoms = np.random.randint(5, 15)
        mat_id = f"gen-water-{i:03d}"
        center = np.random.randn(embedding_dim) * 0.3
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.2

        for atom in atoms:
            gen_atom_embeddings.append(atom)
            gen_atom_material_ids.append(mat_id)

    # 15 "novel fish" materials (different arrangement/center)
    for i in range(15):
        n_atoms = np.random.randint(5, 15)
        mat_id = f"gen-novel-{i:03d}"
        center = np.random.randn(embedding_dim) * 0.3 + np.array([0.5] * embedding_dim)
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.15

        for atom in atoms:
            gen_atom_embeddings.append(atom)
            gen_atom_material_ids.append(mat_id)

    # 10 "jumping out" materials
    for i in range(10):
        n_atoms = np.random.randint(5, 15)
        mat_id = f"gen-jump-{i:03d}"
        center = np.random.randn(embedding_dim) * 1.5
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 0.4

        for atom in atoms:
            gen_atom_embeddings.append(atom)
            gen_atom_material_ids.append(mat_id)

    # 5 "not a fish" materials
    for i in range(5):
        n_atoms = np.random.randint(3, 10)
        mat_id = f"gen-invalid-{i:03d}"
        center = np.random.randn(embedding_dim) * 3.0
        atoms = center + np.random.randn(n_atoms, embedding_dim) * 1.0

        for atom in atoms:
            gen_atom_embeddings.append(atom)
            gen_atom_material_ids.append(mat_id)

    gen_atom_embeddings = np.array(gen_atom_embeddings)
    n_gen_materials = len(set(gen_atom_material_ids))
    print(f"   Generated: {len(gen_atom_embeddings)} atoms from {n_gen_materials} materials")

    # =========================================================================
    # Run Mean-Pooled Analysis
    # =========================================================================
    print("\n3. Running Mean-Pooled Material Analysis...")

    analyzer = MeanPooledMaterialAnalyzer(
        reference_atom_embeddings=ref_atom_embeddings,
        reference_atom_material_ids=ref_atom_material_ids,
        n_neighbors=10,
        stability_threshold=0.5,
        novelty_threshold=0.3,
    )

    results = analyzer.analyze(
        generated_atom_embeddings=gen_atom_embeddings,
        generated_atom_material_ids=gen_atom_material_ids,
    )

    # =========================================================================
    # Print Results
    # =========================================================================
    analyzer.print_summary(results)

    # Get IDs by category
    print("\n" + "=" * 70)
    print("Material IDs by Category")
    print("=" * 70)

    ids_by_cat = analyzer.get_all_ids_by_category(results)
    for cat, ids in ids_by_cat.items():
        print(f"\n{cat.upper()} ({len(ids)} materials):")
        if ids:
            print(f"  {ids[:5]}")
            if len(ids) > 5:
                print(f"  ... and {len(ids)-5} more")

    # Get top novel fish
    print("\n" + "=" * 70)
    print("Top 5 Novel Fish Candidates for DFT")
    print("=" * 70)

    top_novel = analyzer.get_top_candidates(results, category="novel_fish", top_k=5)
    for i, r in enumerate(top_novel, 1):
        print(f"{i}. {r.material_id} ({r.n_atoms} atoms)")
        print(f"   Distance: {r.manifold_distance:.3f}, Novelty: {r.novelty_score:.3f}")
        print(f"   Nearest refs: {r.nearest_reference_ids[:3]}")

    # Export
    analyzer.export_results(results, "mean_pooled_results.csv")

    # Plot
    print("\nGenerating plots...")
    analyzer.plot_distribution(results, save_path="mean_pooled_analysis.png")

    return analyzer, results


if __name__ == "__main__":
    # Run the original atom-level analysis
    print("\n" + "=" * 80)
    print("PART 1: Original Atom-Level Material Analysis")
    print("=" * 80)
    analyzer1, results1 = example_with_mock_data()

    # Run the mean-pooled analysis
    print("\n" + "=" * 80)
    print("PART 2: Mean-Pooled Material Analysis (Simpler, Recommended)")
    print("=" * 80)
    analyzer2, results2 = example_mean_pooled()
