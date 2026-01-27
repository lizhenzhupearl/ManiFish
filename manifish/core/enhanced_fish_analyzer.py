"""
Enhanced Manifold Fish Analyzer with Spatial Structure Metrics.

This module extends ManifoldFishAnalyzer with spatial structure metrics
(LDS, CDS, SRSS, RMSC) for improved classification and stability assessment
while maintaining the same 7-fish category output format.

The Seven Categories (inherited from ManifoldFishAnalyzer):
1. Redundant Fish       - Deep inside, dense region (very similar to known)
2. Fish in Water        - Inside manifold, normal density (standard candidate)
3. Frontier Fish        - Inside manifold, sparse region (exploring new territory)
4. Edge Fish            - At manifold boundary (on the edge of known physics)
5. Adventurous Fish     - Slightly outside manifold (potentially novel)
6. Geometric Atypical   - High local PCA residual but has neighbors
7. Structural Hallucination - Far outside manifold, no neighbors

NEW: Spatial metrics are computed for each structure and can optionally
influence the confidence scores and classification.

Usage:
    analyzer = EnhancedManifoldFishAnalyzer(ref_emb, ref_ids)
    results = analyzer.analyze(gen_emb, gen_ids)
    analyzer.print_summary(results)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors
import warnings

# Import base class and its components
from .fish_analyzer import (
    ManifoldFishAnalyzer,
    ManifoldFishResult,
    CategorySummary as BaseCategorySummary,
    CATEGORIES,
    RISK_LEVELS,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpatialMetricsResult:
    """Spatial structure metrics for a single structure."""
    lds: float  # Local vs Distant Similarity
    cds: float  # Correlation Decay Slope
    srss: float  # Semantic-Region Self-Similarity
    rmsc: float  # RMS Spatial Contrast
    spatial_quality: float  # Combined quality score

    def to_dict(self) -> Dict[str, float]:
        return {
            'lds': self.lds,
            'cds': self.cds,
            'srss': self.srss,
            'rmsc': self.rmsc,
            'spatial_quality': self.spatial_quality,
        }


@dataclass
class EnhancedManifoldFishResult:
    """Enhanced results with all manifold metrics + spatial metrics."""
    index: int
    material_id: str

    # Position metrics (from base)
    manifold_distance: float
    depth_score: float
    boundary_distance: float

    # Density metrics (from base)
    local_density: float
    density_percentile: float

    # Geometry consistency (from base)
    local_pca_residual: float
    geometry_consistent: bool

    # NEW: Spatial structure metrics
    spatial_metrics: SpatialMetricsResult

    # Uncertainty
    ensemble_variance: float = 0.0

    # Classification (from base)
    category: str = ""
    confidence: float = 0.0
    risk_level: str = ""

    # NEW: Enhanced confidence incorporating spatial metrics
    enhanced_confidence: float = 0.0

    # Traceability
    nearest_reference_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'material_id': self.material_id,
            'manifold_distance': self.manifold_distance,
            'depth_score': self.depth_score,
            'boundary_distance': self.boundary_distance,
            'local_density': self.local_density,
            'density_percentile': self.density_percentile,
            'local_pca_residual': self.local_pca_residual,
            'geometry_consistent': self.geometry_consistent,
            'spatial_metrics': self.spatial_metrics.to_dict(),
            'category': self.category,
            'confidence': self.confidence,
            'enhanced_confidence': self.enhanced_confidence,
            'risk_level': self.risk_level,
            'nearest_reference_ids': self.nearest_reference_ids,
        }

    @classmethod
    def from_base_result(
        cls,
        base_result: ManifoldFishResult,
        spatial_metrics: SpatialMetricsResult,
        enhanced_confidence: float,
    ) -> 'EnhancedManifoldFishResult':
        """Create from a base ManifoldFishResult."""
        return cls(
            index=base_result.index,
            material_id=base_result.material_id,
            manifold_distance=base_result.manifold_distance,
            depth_score=base_result.depth_score,
            boundary_distance=base_result.boundary_distance,
            local_density=base_result.local_density,
            density_percentile=base_result.density_percentile,
            local_pca_residual=base_result.local_pca_residual,
            geometry_consistent=base_result.geometry_consistent,
            spatial_metrics=spatial_metrics,
            ensemble_variance=base_result.ensemble_variance,
            category=base_result.category,
            confidence=base_result.confidence,
            enhanced_confidence=enhanced_confidence,
            risk_level=base_result.risk_level,
            nearest_reference_ids=base_result.nearest_reference_ids,
        )


@dataclass
class CategorySummary:
    """Summary for a category with material IDs and spatial quality."""
    category: str
    count: int
    percentage: float
    risk_level: str
    description: str
    action: str
    material_ids: List[str]
    mean_spatial_quality: float
    results: List[EnhancedManifoldFishResult]


# =============================================================================
# Spatial Metrics Computation
# =============================================================================

class SpatialMetricsComputer:
    """Compute spatial structure metrics for manifold analysis (optimized)."""

    def __init__(
        self,
        reference_coords: np.ndarray,
        region_labels: Optional[np.ndarray] = None,
        k_local: int = 10,
        n_distant: int = 100,
    ):
        self.reference = np.asarray(reference_coords)
        self.region_labels = region_labels
        self.k_local = min(k_local, len(reference_coords) - 1)
        self.n_distant = n_distant
        self.k_rmsc = min(20, len(reference_coords) - 1)

        # Pre-normalize reference for fast cosine similarity
        ref_norms = np.linalg.norm(self.reference, axis=1, keepdims=True)
        ref_norms = np.where(ref_norms == 0, 1e-10, ref_norms)
        self.reference_normalized = self.reference / ref_norms

        # Build KNN model (use more neighbors for RMSC)
        max_k = max(self.k_local + 1, self.k_rmsc)
        self.knn = NearestNeighbors(n_neighbors=max_k, algorithm='auto')
        self.knn.fit(self.reference)

        # Pre-compute region indices if available
        if self.region_labels is not None:
            self.unique_regions = np.unique(self.region_labels)
            self.region_indices = {
                r: np.where(self.region_labels == r)[0]
                for r in self.unique_regions
            }

    def compute_lds(self, coords: np.ndarray, random_state: int = 42) -> np.ndarray:
        """Local vs Distant Similarity (vectorized)."""
        coords = np.atleast_2d(coords)
        n_queries = len(coords)
        rng = np.random.default_rng(random_state)

        # Get k-nearest neighbors for all queries at once
        _, indices = self.knn.kneighbors(coords, n_neighbors=self.k_local + 1)
        local_indices = indices[:, 1:]

        # Normalize queries
        q_norms = np.linalg.norm(coords, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1e-10, q_norms)
        coords_norm = coords / q_norms

        # Compute local similarities
        local_sims = np.zeros(n_queries)
        for i in range(n_queries):
            neighbor_vecs = self.reference_normalized[local_indices[i]]
            local_sims[i] = np.mean(coords_norm[i] @ neighbor_vecs.T)

        # Sample distant points
        n_ref = len(self.reference)
        distant_start = n_ref // 2
        n_sample = min(self.n_distant, n_ref - distant_start)
        distant_sample = rng.choice(np.arange(distant_start, n_ref), size=n_sample, replace=False)
        distant_refs = self.reference_normalized[distant_sample]

        # Compute distant similarities (batch)
        distant_sims = np.mean(coords_norm @ distant_refs.T, axis=1)

        return local_sims / (distant_sims + 1e-10)

    def compute_cds(self, coords: np.ndarray, random_state: int = 42) -> np.ndarray:
        """Correlation Decay Slope (optimized)."""
        coords = np.atleast_2d(coords)
        n_queries = len(coords)

        k_cds = min(50, len(self.reference) - 1)
        distances, indices = self.knn.kneighbors(coords, n_neighbors=k_cds)

        # Normalize queries
        q_norms = np.linalg.norm(coords, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1e-10, q_norms)
        coords_norm = coords / q_norms

        cds_scores = np.zeros(n_queries)

        for i in range(n_queries):
            dists = distances[i]
            neighbor_vecs = self.reference_normalized[indices[i]]
            sims = coords_norm[i] @ neighbor_vecs.T

            valid = (sims > 1e-10) & (dists > 1e-10)
            if np.sum(valid) < 5:
                cds_scores[i] = 0.0
                continue

            log_s = np.log(sims[valid] + 1e-10)
            d_valid = dists[valid]

            # Fast slope computation
            d_mean = np.mean(d_valid)
            s_mean = np.mean(log_s)
            numerator = np.sum((d_valid - d_mean) * (log_s - s_mean))
            denominator = np.sum((d_valid - d_mean) ** 2)

            if denominator > 1e-10:
                slope = numerator / denominator
                cds_scores[i] = max(-slope, 0.0)
            else:
                cds_scores[i] = 0.0

        return cds_scores

    def compute_srss(self, coords: np.ndarray, random_state: int = 42) -> np.ndarray:
        """Semantic-Region Self-Similarity (optimized)."""
        coords = np.atleast_2d(coords)
        n_queries = len(coords)

        if self.region_labels is None:
            return np.ones(n_queries)

        rng = np.random.default_rng(random_state)

        _, nn_indices = self.knn.kneighbors(coords, n_neighbors=1)
        query_regions = self.region_labels[nn_indices[:, 0]]

        q_norms = np.linalg.norm(coords, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1e-10, q_norms)
        coords_norm = coords / q_norms

        srss_scores = np.ones(n_queries)

        n_sample = 50
        region_samples = {}
        for region, indices in self.region_indices.items():
            if len(indices) > 0:
                sample_size = min(n_sample, len(indices))
                region_samples[region] = rng.choice(indices, size=sample_size, replace=False)

        for i in range(n_queries):
            query_region = query_regions[i]
            same_idx = self.region_indices.get(query_region, np.array([]))
            if len(same_idx) == 0:
                continue

            same_sample = region_samples.get(query_region, same_idx[:n_sample])

            diff_samples = []
            for r, samples in region_samples.items():
                if r != query_region:
                    diff_samples.append(samples)

            if len(diff_samples) == 0:
                continue

            diff_idx = np.concatenate(diff_samples)
            if len(diff_idx) > n_sample:
                diff_idx = rng.choice(diff_idx, size=n_sample, replace=False)

            sim_same = np.mean(coords_norm[i] @ self.reference_normalized[same_sample].T)
            sim_diff = np.mean(coords_norm[i] @ self.reference_normalized[diff_idx].T)

            srss_scores[i] = sim_same / (sim_diff + 1e-10)

        return srss_scores

    def compute_rmsc(self, coords: np.ndarray) -> np.ndarray:
        """RMS Spatial Contrast (vectorized)."""
        coords = np.atleast_2d(coords)
        _, indices = self.knn.kneighbors(coords, n_neighbors=self.k_rmsc)

        rmsc_scores = np.zeros(len(coords))
        for i, idx in enumerate(indices):
            neighbors = self.reference[idx]
            variances = np.var(neighbors, axis=0)
            rmsc_scores[i] = np.sqrt(np.mean(variances))

        return rmsc_scores

    def compute_all(
        self,
        coords: np.ndarray,
        random_state: int = 42,
    ) -> List[SpatialMetricsResult]:
        """Compute all spatial metrics."""
        coords = np.atleast_2d(coords)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            lds = self.compute_lds(coords, random_state)
            cds = self.compute_cds(coords, random_state)
            srss = self.compute_srss(coords, random_state)
            rmsc = self.compute_rmsc(coords)

        # Handle NaN/Inf values
        lds = np.nan_to_num(lds, nan=1.0, posinf=10.0, neginf=0.1)
        cds = np.nan_to_num(cds, nan=0.0, posinf=10.0, neginf=0.0)
        srss = np.nan_to_num(srss, nan=1.0, posinf=10.0, neginf=0.1)
        rmsc = np.nan_to_num(rmsc, nan=0.0, posinf=1.0, neginf=0.0)

        quality = self._compute_quality(lds, cds, srss, rmsc)

        results = []
        for i in range(len(coords)):
            results.append(SpatialMetricsResult(
                lds=float(lds[i]),
                cds=float(cds[i]),
                srss=float(srss[i]),
                rmsc=float(rmsc[i]),
                spatial_quality=float(quality[i]),
            ))
        return results

    def _compute_quality(self, lds, cds, srss, rmsc) -> np.ndarray:
        """Combined spatial quality score."""
        def normalize(x, scale=1.0):
            z = np.clip(-np.asarray(x) / scale, -500, 500)
            return 2.0 / (1.0 + np.exp(z)) - 1.0

        lds_n = normalize(lds, 2.0)
        cds_n = normalize(cds, 1.0)
        srss_n = normalize(srss, 2.0)
        rmsc_n = normalize(rmsc, 0.5)

        return 0.3 * lds_n + 0.3 * cds_n + 0.2 * srss_n + 0.2 * rmsc_n


# =============================================================================
# Enhanced Manifold Fish Analyzer (inherits from ManifoldFishAnalyzer)
# =============================================================================

class EnhancedManifoldFishAnalyzer(ManifoldFishAnalyzer):
    """
    Enhanced Manifold Fish Analyzer with Spatial Structure Metrics.

    Inherits from ManifoldFishAnalyzer and adds spatial metrics computation.
    Classification results are identical to the base class.

    Example:
        >>> analyzer = EnhancedManifoldFishAnalyzer(ref_emb, ref_ids)
        >>> results = analyzer.analyze(gen_emb, gen_ids)
        >>> analyzer.print_summary(results)
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        reference_ids: Optional[List[str]] = None,
        region_labels: Optional[np.ndarray] = None,

        # Base class parameters
        n_neighbors: int = 10,
        n_neighbors_local_pca: Optional[int] = None,
        boundary_method: Literal["convex_hull", "alpha_shape", "one_class_svm", "auto"] = "auto",
        alpha: Optional[float] = None,
        local_geometry_mode: Literal["full", "fast", "skip"] = "fast",
        curvature_adjusted: bool = False,
        precompute_local_pca: bool = False,

        # Thresholds (same as base)
        stability_threshold: float = 0.5,
        outlier_threshold: float = -1.5,
        geometry_threshold: float = 0.3,
        sparse_threshold: float = 20.0,
        depth_threshold: float = 20.0,
        edge_margin: float = 0.1,

        # Spatial metrics parameters
        spatial_k_local: int = 10,
        spatial_n_distant: int = 100,
        use_spatial_for_confidence: bool = True,
    ):
        """
        Initialize enhanced analyzer with spatial metrics support.

        Args:
            reference_embeddings: Transformed embeddings (Platonic coordinates)
            reference_ids: Material IDs for reference structures
            region_labels: Optional region labels for SRSS computation
            ... (other parameters same as ManifoldFishAnalyzer)
            spatial_k_local: k for local similarity in spatial metrics
            spatial_n_distant: n for distant sampling in spatial metrics
            use_spatial_for_confidence: If True, spatial metrics influence confidence
        """
        # Initialize base class
        super().__init__(
            reference_embeddings=reference_embeddings,
            reference_ids=reference_ids,
            n_neighbors=n_neighbors,
            n_neighbors_local_pca=n_neighbors_local_pca,
            boundary_method=boundary_method,
            alpha=alpha,
            local_geometry_mode=local_geometry_mode,
            curvature_adjusted=curvature_adjusted,
            precompute_local_pca=precompute_local_pca,
            stability_threshold=stability_threshold,
            outlier_threshold=outlier_threshold,
            geometry_threshold=geometry_threshold,
            sparse_threshold=sparse_threshold,
            depth_threshold=depth_threshold,
            edge_margin=edge_margin,
        )

        self.region_labels = region_labels
        self.use_spatial_for_confidence = use_spatial_for_confidence

        # Initialize spatial metrics computer
        self.spatial_computer = SpatialMetricsComputer(
            reference_coords=self.reference,
            region_labels=region_labels,
            k_local=spatial_k_local,
            n_distant=spatial_n_distant,
        )

    def analyze(
        self,
        generated_embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> List[EnhancedManifoldFishResult]:
        """
        Analyze generated structures with enhanced metrics.

        Args:
            generated_embeddings: Transformed embeddings to analyze
            material_ids: Optional material IDs
            random_state: Random seed for spatial metrics

        Returns:
            List of EnhancedManifoldFishResult objects
        """
        # Get base results using parent class
        base_results = super().analyze(generated_embeddings, material_ids)

        # Compute spatial metrics
        generated = np.atleast_2d(generated_embeddings)
        spatial_results = self.spatial_computer.compute_all(generated, random_state)

        # Combine base results with spatial metrics
        enhanced_results = []
        for base_result, spatial_metric in zip(base_results, spatial_results):
            # Enhanced confidence incorporating spatial quality
            if self.use_spatial_for_confidence:
                sq_norm = (spatial_metric.spatial_quality + 1) / 2
                enhanced_confidence = 0.7 * base_result.confidence + 0.3 * sq_norm
            else:
                enhanced_confidence = base_result.confidence

            enhanced_result = EnhancedManifoldFishResult.from_base_result(
                base_result=base_result,
                spatial_metrics=spatial_metric,
                enhanced_confidence=enhanced_confidence,
            )
            enhanced_results.append(enhanced_result)

        # Store for visualization
        self._last_results = enhanced_results
        self._last_embeddings = generated
        self._last_ids = material_ids

        return enhanced_results

    def get_category_summary(
        self,
        results: List[EnhancedManifoldFishResult],
    ) -> Dict[str, CategorySummary]:
        """Get summary by category with spatial quality."""
        summary = {}
        n_total = len(results)

        for cat_name, cat_info in CATEGORIES.items():
            cat_results = [r for r in results if r.category == cat_name]
            count = len(cat_results)

            if count > 0:
                mean_sq = np.mean([r.spatial_metrics.spatial_quality for r in cat_results])
            else:
                mean_sq = 0.0

            summary[cat_name] = CategorySummary(
                category=cat_name,
                count=count,
                percentage=100.0 * count / n_total if n_total > 0 else 0.0,
                risk_level=cat_info["risk"],
                description=cat_info["description"],
                action=cat_info["action"],
                material_ids=[r.material_id for r in cat_results],
                mean_spatial_quality=mean_sq,
                results=cat_results,
            )

        return summary

    def print_summary(self, results: List[EnhancedManifoldFishResult]):
        """Print enhanced analysis summary."""
        summary = self.get_category_summary(results)
        n_total = len(results)

        print("\n" + "=" * 70)
        print("ENHANCED MANIFOLD FISH ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total structures analyzed: {n_total}")
        print("-" * 70)
        print(f"{'Category':<25} {'Count':>8} {'%':>8} {'Risk':<12} {'Spatial Q':>10}")
        print("-" * 70)

        for cat_name in CATEGORIES.keys():
            s = summary[cat_name]
            if s.count > 0:
                print(f"{cat_name:<25} {s.count:>8} {s.percentage:>7.1f}% {s.risk_level:<12} {s.mean_spatial_quality:>10.3f}")

        print("-" * 70)

        # Recommended actions
        print("\nRECOMMENDED ACTIONS:")
        priority_order = [
            "frontier_fish", "adventurous_fish", "fish_in_water",
            "edge_fish", "geometric_atypical", "structural_hallucination", "redundant_fish",
        ]
        actions_printed = False
        for cat_name in priority_order:
            s = summary[cat_name]
            if s.count > 0:
                print(f"  {cat_name} ({s.count}): {CATEGORIES[cat_name]['action']}")
                actions_printed = True
        if not actions_printed:
            print("  No structures to process.")

    def export_results(
        self,
        results: List[EnhancedManifoldFishResult],
        filepath: str,
    ):
        """Export enhanced results to CSV."""
        import pandas as pd

        rows = []
        for r in results:
            row = {
                'material_id': r.material_id,
                'category': r.category,
                'confidence': r.confidence,
                'enhanced_confidence': r.enhanced_confidence,
                'risk_level': r.risk_level,
                'manifold_distance': r.manifold_distance,
                'depth_score': r.depth_score,
                'boundary_distance': r.boundary_distance,
                'local_density': r.local_density,
                'density_percentile': r.density_percentile,
                'local_pca_residual': r.local_pca_residual,
                'geometry_consistent': r.geometry_consistent,
                'lds': r.spatial_metrics.lds,
                'cds': r.spatial_metrics.cds,
                'srss': r.spatial_metrics.srss,
                'rmsc': r.spatial_metrics.rmsc,
                'spatial_quality': r.spatial_metrics.spatial_quality,
                'nearest_refs': ','.join(r.nearest_reference_ids[:3]),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")

    # =========================================================================
    # Visualization Methods
    # =========================================================================

    def plot_overview(
        self,
        results: Optional[List[EnhancedManifoldFishResult]] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ):
        """Create comprehensive overview plot."""
        from manifish.visualization.fish_spatial_viz import plot_category_spatial_overview

        if results is None:
            results = self._last_results
        if results is None:
            raise ValueError("No results available. Run analyze() first.")

        return plot_category_spatial_overview(results, figsize, save_path)

    def plot_spatial_metrics(
        self,
        results: Optional[List[EnhancedManifoldFishResult]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ):
        """Plot box plots of all spatial metrics by category."""
        from manifish.visualization.fish_spatial_viz import plot_spatial_metrics_grid

        if results is None:
            results = self._last_results
        if results is None:
            raise ValueError("No results available. Run analyze() first.")

        return plot_spatial_metrics_grid(results, figsize, save_path)

    def plot_pairwise(
        self,
        results: Optional[List[EnhancedManifoldFishResult]] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ):
        """Plot pairwise scatter of spatial metrics."""
        from manifish.visualization.fish_spatial_viz import plot_pairwise_scatter

        if results is None:
            results = self._last_results
        if results is None:
            raise ValueError("No results available. Run analyze() first.")

        return plot_pairwise_scatter(results, figsize, save_path)

    def plot_confidence(
        self,
        results: Optional[List[EnhancedManifoldFishResult]] = None,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
    ):
        """Compare original vs enhanced confidence by category."""
        from manifish.visualization.fish_spatial_viz import plot_confidence_comparison

        if results is None:
            results = self._last_results
        if results is None:
            raise ValueError("No results available. Run analyze() first.")

        return plot_confidence_comparison(results, figsize, save_path)

    def plot_heatmap(
        self,
        results: Optional[List[EnhancedManifoldFishResult]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
    ):
        """Create heatmap of mean spatial metrics by category."""
        from manifish.visualization.fish_spatial_viz import plot_category_heatmap

        if results is None:
            results = self._last_results
        if results is None:
            raise ValueError("No results available. Run analyze() first.")

        return plot_category_heatmap(results, figsize, save_path)

    def generate_report(
        self,
        results: Optional[List[EnhancedManifoldFishResult]] = None,
        output_dir: str = ".",
        prefix: str = "fish_spatial",
    ):
        """Generate all visualization plots and save to directory."""
        from manifish.visualization.fish_spatial_viz import create_full_report

        if results is None:
            results = self._last_results
        if results is None:
            raise ValueError("No results available. Run analyze() first.")

        create_full_report(results, output_dir, prefix)
