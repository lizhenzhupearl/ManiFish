"""
Manifold Fish Analyzer - Core Implementation.

This is the base ManifoldFishAnalyzer that classifies structures into 6 categories
based on their position in the manifold. For spatial metrics support, use
EnhancedManifoldFishAnalyzer from enhanced_fish_analyzer.py.

The Six Categories:
1. Redundant Fish       - Deep inside, dense region (very similar to known)
2. Fish in Water        - Inside manifold, normal density (standard candidate)
3. Frontier Fish        - Inside manifold, sparse region (low/high risk based on geometry)
4. Edge Fish            - At manifold boundary (on the edge of known physics)
5. Adventurous Fish     - Outside manifold (risk based on geometry and LOF)
6. Structural Hallucination - Bad geometry + LOF outlier (likely unphysical)

Classification Logic:
- Dense regions: redundant_fish, fish_in_water, edge_fish
- Sparse regions: frontier_fish (geometry determines risk level)
- Outside boundary: adventurous_fish (geometry + LOF determine risk)
- Geometry is the primary validity indicator in frontier/sparse regions

Usage:
    from manifish.core import ManifoldFishAnalyzer

    analyzer = ManifoldFishAnalyzer(reference_embeddings, reference_ids)
    results = analyzer.analyze(generated_embeddings, generated_ids)
    analyzer.print_summary(results)
    analyzer.export_results(results, "results.csv")
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull, Delaunay
import pickle
import warnings

# Optional imports
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Plotting Style Helper
# =============================================================================

def apply_plot_style(ax, xlabel_size: int = 12, ylabel_size: int = 12,
                     tick_labelsize: int = 14, spine_linewidth: float = 1.0):
    """
    Apply consistent plotting style to an axis.

    Style includes:
    - Black spines on all sides with specified linewidth
    - Consistent tick label sizes
    - Arial font for tick labels (if available)

    Args:
        ax: Matplotlib axis object
        xlabel_size: Font size for x-axis label
        ylabel_size: Font size for y-axis label
        tick_labelsize: Font size for tick labels
        spine_linewidth: Line width for axis spines
    """
    # Set spine colors and linewidths
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(spine_linewidth)

    # Customize tick parameters
    ax.tick_params(axis='both', which='both', labelsize=tick_labelsize)
    ax.xaxis.set_tick_params(labelsize=tick_labelsize)
    ax.yaxis.set_tick_params(labelsize=tick_labelsize)

    # Try to use Arial font for tick labels
    try:
        arial_font = fm.FontProperties(family='Arial', size=tick_labelsize)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(arial_font)
    except Exception:
        pass  # Fall back to default font if Arial not available


# =============================================================================
# Category Definitions
# =============================================================================

CATEGORIES = {
    "redundant_fish": {
        "description": "Deep inside manifold, dense region - very similar to known",
        "risk": "very_low",
        "action": "Skip (redundant)",
        "color": "#95a5a6",  # Gray
    },
    "fish_in_water": {
        "description": "Inside manifold, normal density - standard candidate",
        "risk": "low",
        "action": "Standard validation",
        "color": "#2ecc71",  # Green
    },
    "frontier_fish": {
        "description": "Sparse region - low risk if geometry good, high risk if geometry bad",
        "risk": "low",  # Base risk; actual risk varies (low or high) based on geometry
        "action": "Priority for DFT (check geometry for risk level)",
        "color": "#3498db",  # Blue
    },
    "edge_fish": {
        "description": "At manifold boundary - on the edge of known physics",
        "risk": "medium",
        "action": "Careful validation",
        "color": "#f39c12",  # Orange
    },
    "adventurous_fish": {
        "description": "Outside manifold - risk based on geometry and LOF",
        "risk": "medium",  # Base risk; actual risk varies based on geometry
        "action": "High priority DFT (check risk level)",
        "color": "#9b59b6",  # Purple
    },
    "structural_hallucination": {
        "description": "Bad geometry + LOF outlier - likely unphysical",
        "risk": "very_high",
        "action": "Reject, no physical support",
        "color": "#e74c3c",  # Red (was dark gray, now red for emphasis)
    },
}

RISK_LEVELS = ["very_low", "low", "low_medium", "medium", "medium_high", "high", "very_high"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ManifoldFishResult:
    """Enhanced results with all manifold metrics."""
    index: int
    material_id: str

    # Position metrics
    manifold_distance: float      # Distance to nearest references (normalized)
    depth_score: float            # Distance to manifold centroid (normalized, 0=deep, 1=shallow)
    boundary_distance: float      # Distance to manifold boundary (+ inside, - outside)

    # Density metrics
    local_density: float          # k-NN density estimate
    density_percentile: float     # Percentile vs reference distribution (0-100)

    # Geometry consistency
    local_pca_residual: float     # Reconstruction error from local tangent (normalized)
    geometry_consistent: bool     # Below threshold?

    # Uncertainty (future)
    ensemble_variance: float = 0.0  # Placeholder

    # Classification
    category: str = ""
    confidence: float = 0.0
    risk_level: str = ""

    # Traceability
    nearest_reference_ids: List[str] = field(default_factory=list)


@dataclass
class CategorySummary:
    """Summary for a category with material IDs."""
    category: str
    count: int
    percentage: float
    risk_level: str
    description: str
    action: str
    material_ids: List[str]
    results: List[ManifoldFishResult]


# =============================================================================
# Main Analyzer Class
# =============================================================================

class ManifoldFishAnalyzer:
    """
    Manifold-aware analysis of generated structures using the Fish-Water framework.

    This analyzer evaluates generated structures based on:
    - Position relative to the reference manifold (depth, boundary, distance)
    - Local density (sparse vs dense regions)
    - Geometric consistency (local tangent space alignment)

    Reference embeddings define the "water" (known stable manifold).
    Generated embeddings are the "fish" candidates to classify into 7 categories.

    Example:
        >>> analyzer = ManifoldFishAnalyzer(
        ...     reference_embeddings=material_embeddings_ref,
        ...     reference_ids=reference_ids,
        ...     boundary_method="alpha_shape"
        ... )
        >>> results = analyzer.analyze(
        ...     generated_embeddings=material_embeddings_gen,
        ...     material_ids=generated_ids
        ... )
        >>> analyzer.print_summary(results)
        >>> analyzer.export_results(results, "manifold_analysis.csv")
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        reference_ids: Optional[List[str]] = None,

        # Neighbor parameters
        n_neighbors: int = 10,
        n_neighbors_local_pca: Optional[int] = None,  # Default: max(15, 2*n_neighbors)

        # Boundary method
        boundary_method: Literal["convex_hull", "alpha_shape", "one_class_svm", "auto"] = "auto",
        alpha: Optional[float] = None,  # For alpha_shape, None = auto

        # Local geometry computation mode
        local_geometry_mode: Literal["full", "fast", "skip"] = "fast",

        # Curvature adjustment for local PCA residual
        curvature_adjusted: bool = False,

        # OPTIMIZATION: Precompute local PCAs for all reference points during fit()
        # Trades memory for speed (useful when analyzing multiple generated sets)
        precompute_local_pca: bool = False,

        # Thresholds
        stability_threshold: float = 0.5,      # Manifold distance (normalized)
        outlier_threshold: float = -1.5,       # LOF score
        geometry_threshold: float = 0.3,       # Local PCA residual (normalized)
        sparse_threshold: float = 20.0,        # Density percentile (below = sparse)
        depth_threshold: float = 20.0,         # Depth percentile (below = deep/central)
        edge_margin: float = 0.1,              # Boundary distance margin (normalized)
    ):
        """
        Initialize analyzer with reference embeddings (the "water").

        Args:
            reference_embeddings: Known stable structures embeddings
                                  Shape: (n_reference, embedding_dim)
            reference_ids: Material IDs for reference structures
            n_neighbors: Number of neighbors for distance/density calculation
            n_neighbors_local_pca: Number of neighbors for local tangent estimation
                                   Default: max(15, 2*n_neighbors)
            boundary_method: Method for boundary detection
                           "convex_hull" - Fast, simple (may be too loose)
                           "alpha_shape" - Good for non-convex (dim <= 3)
                           "one_class_svm" - Works for any dimension
                           "auto" - alpha_shape for dim<=3, one_class_svm otherwise
            alpha: Alpha parameter for alpha_shape (None = auto-tune)
            local_geometry_mode: Mode for local PCA residual computation
                               "full" - Fit PCA for each generated point (slow, most accurate)
                               "fast" - Reuse precomputed reference PCAs (recommended)
                               "skip" - Skip local geometry computation entirely
            curvature_adjusted: If True, adjust local PCA residual by local curvature.
                               High-curvature regions (like bends in U-shaped manifolds)
                               get more tolerance. Recommended for curved manifolds.
            precompute_local_pca: If True, precompute local PCAs for all reference points
                                 during initialization. Trades memory for speed.
                                 Useful when analyzing multiple generated sets.
            stability_threshold: Max normalized distance to be considered "in water"
            outlier_threshold: LOF score below this = "not a fish"
            geometry_threshold: Max normalized local PCA residual for consistency
            sparse_threshold: Density percentile below this = sparse region
            depth_threshold: Depth percentile below this = deep/central
            edge_margin: Boundary distance margin for edge classification
        """
        self.reference = np.asarray(reference_embeddings)
        self.n_reference = len(self.reference)
        self.dim = self.reference.shape[1]

        # Store reference IDs
        if reference_ids is not None:
            self.reference_ids = list(reference_ids)
        else:
            self.reference_ids = [f"ref_{i}" for i in range(self.n_reference)]

        # Neighbor parameters
        self.n_neighbors = min(n_neighbors, self.n_reference - 1)
        self.n_neighbors_local_pca = n_neighbors_local_pca or max(15, 2 * self.n_neighbors)
        self.n_neighbors_local_pca = min(self.n_neighbors_local_pca, self.n_reference - 1)

        # Boundary method
        if boundary_method == "auto":
            self.boundary_method = "alpha_shape" if self.dim <= 3 else "one_class_svm"
        else:
            self.boundary_method = boundary_method
        self.alpha = alpha

        # Local geometry mode
        self.local_geometry_mode = local_geometry_mode
        self.curvature_adjusted = curvature_adjusted
        self.precompute_local_pca = precompute_local_pca

        # Thresholds
        self.stability_threshold = stability_threshold
        self.outlier_threshold = outlier_threshold
        self.geometry_threshold = geometry_threshold
        self.sparse_threshold = sparse_threshold
        self.depth_threshold = depth_threshold
        self.edge_margin = edge_margin

        # Fit models on reference data
        self._fit_reference_models()

    def _fit_reference_models(self):
        """Fit all models on reference data."""
        # 1. k-NN model for distance and density
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='auto',
            metric='euclidean'
        )
        self.knn.fit(self.reference)

        # k-NN for local PCA (more neighbors)
        self.knn_local_pca = NearestNeighbors(
            n_neighbors=self.n_neighbors_local_pca,
            algorithm='auto',
            metric='euclidean'
        )
        self.knn_local_pca.fit(self.reference)

        # 2. Reference distance statistics
        ref_distances, _ = self.knn.kneighbors(self.reference)
        self.ref_mean_dist = np.mean(ref_distances[:, -1])
        self.ref_std_dist = np.std(ref_distances[:, -1]) + 1e-8

        # 3. LOF model
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,
            contamination=0.1
        )
        self.lof.fit(self.reference)

        # 4. Centroid for depth calculation
        self.centroid = np.mean(self.reference, axis=0)
        ref_centroid_dists = np.linalg.norm(self.reference - self.centroid, axis=1)
        self.max_centroid_dist = np.max(ref_centroid_dists) + 1e-8

        # 5. Reference density distribution (for percentile calculation)
        self.ref_densities = self._compute_density_values(ref_distances)
        # OPTIMIZATION: Pre-sort for fast percentile calculation using searchsorted
        self.ref_densities_sorted = np.sort(self.ref_densities)

        # 6. Local geometry setup based on mode
        if self.local_geometry_mode == "skip":
            self.ref_local_residuals = np.zeros(self.n_reference)
            self.ref_local_pcas = None
            self.ref_local_pca_spreads = None
            self.ref_local_curvatures = None
        elif self.local_geometry_mode == "fast":
            # LAZY CACHING: Don't precompute all, cache on demand
            # This is efficient even when n_ref >> n_gen
            self.ref_local_pcas = {}  # Cache: ref_idx -> PCA model
            self.ref_local_pca_spreads = {}  # Cache: ref_idx -> local spread
            self.ref_local_curvatures = {}  # Cache: ref_idx -> local curvature
            # Skip computing reference residuals (expensive for large n_ref)
            self.ref_local_residuals = None

            # OPTIMIZATION: Optionally precompute all local PCAs during initialization
            # Trades memory for speed (useful when analyzing multiple generated sets)
            if self.precompute_local_pca:
                print(f"Precomputing local PCAs for {self.n_reference} reference points...")
                for ref_idx in range(self.n_reference):
                    self._compute_and_cache_local_pca(ref_idx)
                print(f"✓ Cached {len(self.ref_local_pcas)} local PCAs")
        else:  # "full"
            self.ref_local_pcas = None
            self.ref_local_pca_spreads = None
            self.ref_local_curvatures = None
            self.ref_local_residuals = self._compute_reference_local_residuals()

        # 7. Boundary model
        self._fit_boundary_model()

    def _fit_boundary_model(self):
        """Fit the boundary detection model."""
        if self.boundary_method == "convex_hull":
            self._fit_convex_hull()
        elif self.boundary_method == "alpha_shape":
            self._fit_alpha_shape()
        elif self.boundary_method == "one_class_svm":
            self._fit_one_class_svm()
        else:
            raise ValueError(f"Unknown boundary method: {self.boundary_method}")

    def _fit_convex_hull(self):
        """Fit convex hull boundary."""
        try:
            self.hull = ConvexHull(self.reference)
            self.delaunay = Delaunay(self.reference)
            # Compute max distance to hull for normalization
            self._compute_hull_normalization()
        except Exception as e:
            warnings.warn(f"ConvexHull failed: {e}. Falling back to one_class_svm.")
            self.boundary_method = "one_class_svm"
            self._fit_one_class_svm()

    def _fit_alpha_shape(self):
        """Fit alpha shape boundary using Delaunay triangulation."""
        try:
            self.delaunay = Delaunay(self.reference)

            # Auto-tune alpha if not provided
            if self.alpha is None:
                # Use mean k-NN distance as alpha
                ref_distances, _ = self.knn.kneighbors(self.reference)
                self.alpha = np.mean(ref_distances[:, -1]) * 2

            self._compute_hull_normalization()
        except Exception as e:
            warnings.warn(f"Alpha shape failed: {e}. Falling back to one_class_svm.")
            self.boundary_method = "one_class_svm"
            self._fit_one_class_svm()

    def _fit_one_class_svm(self):
        """Fit one-class SVM boundary."""
        # Standardize for SVM
        self.svm_scaler = StandardScaler()
        ref_scaled = self.svm_scaler.fit_transform(self.reference)

        self.one_class_svm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.1  # Expected fraction of outliers
        )
        self.one_class_svm.fit(ref_scaled)

        # Get reference decision function values for normalization
        ref_decision = self.one_class_svm.decision_function(ref_scaled)
        self.svm_decision_std = np.std(ref_decision) + 1e-8

    def _compute_hull_normalization(self):
        """Compute normalization factor for hull-based boundary distance."""
        # Use the characteristic length scale of the data
        ref_distances, _ = self.knn.kneighbors(self.reference)
        self.boundary_norm = np.mean(ref_distances[:, -1]) + 1e-8

    def _compute_density_values(self, distances: np.ndarray) -> np.ndarray:
        """Compute density values from k-NN distances."""
        mean_dist = np.mean(distances, axis=1)
        density = 1.0 / (mean_dist + 1e-8)
        return density

    def _compute_reference_local_residuals(self) -> np.ndarray:
        """Compute local PCA residuals for reference points."""
        residuals = []
        _, indices = self.knn_local_pca.kneighbors(self.reference)

        for i in range(self.n_reference):
            neighbor_idx = indices[i]
            # Exclude self from neighbors
            neighbor_idx = neighbor_idx[neighbor_idx != i][:self.n_neighbors_local_pca - 1]

            if len(neighbor_idx) < 5:
                residuals.append(0.0)
                continue

            neighbors = self.reference[neighbor_idx]
            residual = self._compute_single_local_residual(self.reference[i], neighbors)
            residuals.append(residual)

        return np.array(residuals)

    def _compute_single_local_residual(
        self,
        point: np.ndarray,
        neighbors: np.ndarray,
    ) -> float:
        """
        Compute local PCA residual for a single point.

        This measures how well the point lies on the local tangent plane
        defined by its neighbors. High residual = geometric inconsistency.
        """
        if len(neighbors) < 5:
            return 0.0

        # Number of components for local tangent (dim - 1)
        n_components = min(self.dim - 1, len(neighbors) - 1)
        n_components = max(1, n_components)

        # Fit local PCA on neighbors
        local_pca = PCA(n_components=n_components)
        try:
            local_pca.fit(neighbors)
        except Exception:
            return 0.0

        # Project point onto local tangent space and back
        point_centered = point - local_pca.mean_
        point_projected = local_pca.inverse_transform(
            local_pca.transform(point_centered.reshape(1, -1))
        )

        # Compute residual (distance to tangent plane)
        residual = np.linalg.norm(point_centered - point_projected)

        # Normalize by local spread with minimum to prevent over-sensitivity
        # When neighbors are tightly clustered, std is small and normalization
        # can make tiny residuals appear large. Use minimum spread threshold.
        local_spread = np.std(neighbors)
        min_spread = 0.1  # Minimum spread to prevent over-normalization
        local_spread = max(local_spread, min_spread)
        normalized_residual = residual / local_spread

        return normalized_residual

    # =========================================================================
    # Main Analysis Method
    # =========================================================================

    def analyze(
        self,
        generated_embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> List[ManifoldFishResult]:
        """
        Analyze generated structures and classify into manifold-aware categories.

        Args:
            generated_embeddings: Generated structures embeddings
                                  Shape: (n_generated, embedding_dim)
            material_ids: Material IDs for generated structures

        Returns:
            List of ManifoldFishResult for each generated structure
        """
        generated = np.asarray(generated_embeddings)
        n_generated = len(generated)

        # Handle material IDs
        if material_ids is not None:
            mat_ids = list(material_ids)
        else:
            mat_ids = [f"gen_{i}" for i in range(n_generated)]

        # 1. Compute all metrics
        metrics = self._compute_all_metrics(generated)

        # 2. Classify each structure
        results = []
        for i in range(n_generated):
            category, confidence, risk_level = self._classify(
                manifold_distance=metrics['manifold_distance'][i],
                depth_score=metrics['depth_score'][i],
                boundary_distance=metrics['boundary_distance'][i],
                local_density=metrics['local_density'][i],
                density_percentile=metrics['density_percentile'][i],
                local_pca_residual=metrics['local_pca_residual'][i],
                lof_score=metrics['lof_score'][i],
            )

            # Get IDs of nearest reference structures
            nearest_ref_ids = [self.reference_ids[idx] for idx in metrics['neighbor_indices'][i]]

            result = ManifoldFishResult(
                index=i,
                material_id=mat_ids[i],
                manifold_distance=metrics['manifold_distance'][i],
                depth_score=metrics['depth_score'][i],
                boundary_distance=metrics['boundary_distance'][i],
                local_density=metrics['local_density'][i],
                density_percentile=metrics['density_percentile'][i],
                local_pca_residual=metrics['local_pca_residual'][i],
                geometry_consistent=metrics['local_pca_residual'][i] < self.geometry_threshold,
                ensemble_variance=0.0,  # Placeholder
                category=category,
                confidence=confidence,
                risk_level=risk_level,
                nearest_reference_ids=nearest_ref_ids,
            )
            results.append(result)

        # Store for later use
        self._last_results = results
        self._last_embeddings = generated
        self._last_ids = mat_ids

        return results

    def _compute_all_metrics(self, generated: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all metrics for generated points."""
        n_generated = len(generated)

        # k-NN distances and indices
        distances, indices = self.knn.kneighbors(generated)

        # 1. Manifold distance (normalized)
        mean_distances = np.mean(distances, axis=1)
        manifold_distance = (mean_distances - self.ref_mean_dist) / self.ref_std_dist

        # 2. Depth score (distance to centroid, normalized)
        centroid_dists = np.linalg.norm(generated - self.centroid, axis=1)
        depth_score = centroid_dists / self.max_centroid_dist

        # 3. Boundary distance
        boundary_distance = self._compute_boundary_distance(generated)

        # 4. Local density
        local_density = self._compute_density_values(distances)

        # 5. Density percentile
        # OPTIMIZED: Use vectorized searchsorted instead of loop (10x faster)
        # searchsorted gives index where value would be inserted to maintain order
        positions = np.searchsorted(self.ref_densities_sorted, local_density, side='right')
        density_percentile = 100.0 * positions / len(self.ref_densities_sorted)

        # 6. Local PCA residual
        local_pca_residual = self._compute_local_pca_residuals(generated)

        # 7. LOF score
        lof_score = self.lof.score_samples(generated)

        return {
            'manifold_distance': manifold_distance,
            'depth_score': depth_score,
            'boundary_distance': boundary_distance,
            'local_density': local_density,
            'density_percentile': density_percentile,
            'local_pca_residual': local_pca_residual,
            'lof_score': lof_score,
            'neighbor_indices': indices,
        }

    def _compute_boundary_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Compute signed distance to boundary.
        Positive = inside, Negative = outside.
        """
        if self.boundary_method in ["convex_hull", "alpha_shape"]:
            return self._compute_hull_boundary_distance(points)
        else:  # one_class_svm
            return self._compute_svm_boundary_distance(points)

    def _compute_hull_boundary_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Compute boundary distance using Delaunay triangulation.

        OPTIMIZED: Uses vectorized operations and kNN approximation for 100x speedup.
        """
        n_points = len(points)

        # Vectorized simplex check
        simplices = self.delaunay.find_simplex(points)
        inside_mask = simplices >= 0
        outside_mask = ~inside_mask

        boundary_distances = np.zeros(n_points)

        # For points inside the hull
        if np.any(inside_mask):
            inside_points = points[inside_mask]

            if hasattr(self, 'hull') and len(self.hull.vertices) < len(self.reference):
                # Use hull vertices for boundary approximation
                hull_points = self.reference[self.hull.vertices]

                # Vectorized distance computation using broadcasting
                # Shape: (n_inside, n_hull_vertices)
                dists = np.linalg.norm(
                    inside_points[:, np.newaxis, :] - hull_points[np.newaxis, :, :],
                    axis=2
                )
                boundary_distances[inside_mask] = np.min(dists, axis=1) / self.boundary_norm
            else:
                # If too many hull vertices, use kNN approximation
                # The nearest neighbor distance is a good proxy for boundary distance
                knn_dists, _ = self.knn.kneighbors(inside_points, n_neighbors=1)
                boundary_distances[inside_mask] = knn_dists[:, 0] / self.boundary_norm

        # For points outside the hull - use kNN for efficiency
        if np.any(outside_mask):
            outside_points = points[outside_mask]
            # Nearest neighbor distance (already computed by kNN)
            knn_dists, _ = self.knn.kneighbors(outside_points, n_neighbors=1)
            boundary_distances[outside_mask] = -knn_dists[:, 0] / self.boundary_norm

        return boundary_distances

    def _compute_svm_boundary_distance(self, points: np.ndarray) -> np.ndarray:
        """Compute boundary distance using one-class SVM decision function."""
        points_scaled = self.svm_scaler.transform(points)
        decision = self.one_class_svm.decision_function(points_scaled)
        # Normalize by reference standard deviation
        return decision / self.svm_decision_std

    def _compute_local_pca_residuals(self, generated: np.ndarray) -> np.ndarray:
        """
        Compute local PCA residuals for all generated points.

        Mode behavior:
        - "skip": Returns zeros (no geometry check)
        - "fast": Lazy caching - compute PCA on demand, cache for reuse
                  Efficient even when n_ref >> n_gen
        - "full": Fits new PCA for each generated point (slow but most accurate)
        """
        n_generated = len(generated)

        # Skip mode: return zeros
        if self.local_geometry_mode == "skip":
            return np.zeros(n_generated)

        residuals = np.zeros(n_generated)

        # Get neighbors from reference set
        distances_local, indices = self.knn_local_pca.kneighbors(generated)

        if self.local_geometry_mode == "fast":
            # FAST MODE with LAZY CACHING
            # Only compute PCA for reference points that are actually used
            # Cache them for reuse if multiple generated points share same nearest ref

            for i in range(n_generated):
                nearest_ref_idx = indices[i, 0]  # Nearest reference point
                point = generated[i]  # Get the current point

                # Check if generated point is identical/very close to nearest reference point
                # Use the precomputed distance from knn search for efficiency
                distance_to_nearest = distances_local[i, 0]

                # Threshold for considering points identical
                # If distance is essentially zero, it's the same point
                identity_threshold = 1e-8
                if distance_to_nearest < identity_threshold:
                    # Point is identical or nearly identical to a reference point
                    # Set residual to 0 since it's part of the reference manifold
                    residuals[i] = 0.0
                    continue

                # Check cache first
                if nearest_ref_idx in self.ref_local_pcas:
                    local_pca = self.ref_local_pcas[nearest_ref_idx]
                    local_spread = self.ref_local_pca_spreads[nearest_ref_idx]
                    local_curvature = self.ref_local_curvatures.get(nearest_ref_idx, 0.0)
                else:
                    # Compute and cache on demand
                    local_pca, local_spread, local_curvature = self._compute_and_cache_local_pca(nearest_ref_idx)

                if local_pca is None:
                    residuals[i] = 0.0
                    continue

                # Compute residual using cached PCA
                point_centered = point - local_pca.mean_
                try:
                    point_projected = local_pca.inverse_transform(
                        local_pca.transform(point_centered.reshape(1, -1))
                    )
                    residual = np.linalg.norm(point_centered - point_projected)

                    # Apply minimum spread to prevent over-normalization
                    min_spread = 0.1
                    effective_spread = max(local_spread, min_spread)

                    # Double-check: if normalized residual is very high but distance to nearest is 0,
                    # this is an artifact from PCA not including the reference point itself
                    # Set residual to 0 to avoid false high-risk classification in sparse regions
                    normalized_residual = residual / effective_spread
                    if normalized_residual > 1.0 and distance_to_nearest < 1e-6:
                        # Point is identical but PCA residual is artificially high
                        residuals[i] = 0.0
                    elif self.curvature_adjusted and local_curvature > 0:
                        # Apply curvature-aware adjustment if enabled
                        # High curvature regions get more tolerance for residuals
                        # Formula: ε_adjusted = ε / (local_spread * sqrt(1 + κ²))
                        # where κ is local curvature ∈ [0, 1]
                        curvature_factor = np.sqrt(1 + local_curvature ** 2)
                        residuals[i] = residual / (effective_spread * curvature_factor)
                    else:
                        residuals[i] = normalized_residual
                except Exception:
                    residuals[i] = 0.0
        else:
            # FULL MODE: Fit new PCA for each generated point
            for i in range(n_generated):
                # Check if generated point is identical/very close to nearest reference point
                # Use the precomputed distance from knn search for efficiency
                distance_to_nearest = distances_local[i, 0]

                # Threshold for considering points identical
                # If distance is essentially zero, it's the same point
                identity_threshold = 1e-8
                if distance_to_nearest < identity_threshold:
                    # Point is identical or nearly identical to a reference point
                    # Set residual to 0 since it's part of the reference manifold
                    residuals[i] = 0.0
                    continue

                neighbors = self.reference[indices[i]]
                residuals[i] = self._compute_single_local_residual(generated[i], neighbors)

        return residuals

    def _compute_and_cache_local_pca(self, ref_idx: int) -> Tuple[Optional[PCA], float, float]:
        """
        Compute local PCA for a reference point and cache it.

        This is called on-demand during analysis (lazy caching).
        Returns (local_pca, local_spread, local_curvature) or (None, 1.0, 0.0) if computation fails.

        Curvature estimation:
        - Uses the ratio of unexplained variance to total variance
        - High curvature = local manifold deviates from flat plane
        - κ ∈ [0, 1], where 0 = flat, 1 = highly curved
        """
        # Get neighbors of this reference point
        _, neighbor_indices = self.knn_local_pca.kneighbors(
            self.reference[ref_idx].reshape(1, -1)
        )
        neighbor_idx = neighbor_indices[0]

        # Exclude self from neighbors
        neighbor_idx = neighbor_idx[neighbor_idx != ref_idx][:self.n_neighbors_local_pca - 1]

        if len(neighbor_idx) < 5:
            self.ref_local_pcas[ref_idx] = None
            self.ref_local_pca_spreads[ref_idx] = 1.0
            self.ref_local_curvatures[ref_idx] = 0.0
            return None, 1.0, 0.0

        neighbors = self.reference[neighbor_idx]
        n_components = min(self.dim - 1, len(neighbor_idx) - 1)
        n_components = max(1, n_components)

        try:
            # Fit PCA with all components first to get full eigenspectrum
            full_pca = PCA()
            full_pca.fit(neighbors)

            # Compute curvature from eigenvalue distribution
            # Curvature = fraction of variance NOT explained by tangent plane (dim-1 components)
            explained_ratios = full_pca.explained_variance_ratio_
            if len(explained_ratios) > n_components:
                # Variance explained by tangent plane (first dim-1 components)
                tangent_variance = np.sum(explained_ratios[:n_components])
                # Curvature = unexplained variance (how much the manifold curves away)
                local_curvature = 1.0 - tangent_variance
            else:
                local_curvature = 0.0

            # Now create the actual PCA with n_components for projection
            local_pca = PCA(n_components=n_components)
            local_pca.fit(neighbors)
            local_spread = np.std(neighbors) + 1e-8

            # Cache for reuse
            self.ref_local_pcas[ref_idx] = local_pca
            self.ref_local_pca_spreads[ref_idx] = local_spread
            self.ref_local_curvatures[ref_idx] = local_curvature

            return local_pca, local_spread, local_curvature
        except Exception:
            self.ref_local_pcas[ref_idx] = None
            self.ref_local_pca_spreads[ref_idx] = 1.0
            self.ref_local_curvatures[ref_idx] = 0.0
            return None, 1.0, 0.0

    def _classify(
        self,
        manifold_distance: float,
        depth_score: float,
        boundary_distance: float,
        local_density: float,
        density_percentile: float,
        local_pca_residual: float,
        lof_score: float,
    ) -> Tuple[str, float, str]:
        """
        Classify structure into one of 6 categories.

        Classification logic:
        0. Near-exact match → redundant_fish (very_low risk)
        1. Outside boundary → adventurous_fish or hallucination (geometry + LOF based)
        2. Inside boundary:
           - Sparse region: frontier_fish (low risk if geometry good, high risk if bad)
           - Dense region: edge_fish, redundant_fish, fish_in_water (position-based)

        Key insight: In sparse/frontier regions, geometry consistency is the primary
        validity indicator. Good geometry = valid frontier even if LOF is outlier-ish.
        """

        near_exact_threshold = 0.05  # Very close to a reference point

        # Helper flags
        geometry_good = (self.local_geometry_mode == "skip" or
                        local_pca_residual <= self.geometry_threshold)
        lof_outlier = lof_score < self.outlier_threshold
        is_sparse = density_percentile < self.sparse_threshold

        # 0. Near-exact match → redundant_fish
        if manifold_distance < near_exact_threshold:
            confidence = min(1.0, (near_exact_threshold - manifold_distance) / near_exact_threshold + 0.6)
            return "redundant_fish", confidence, "very_low"

        # 1. Outside boundary
        if boundary_distance < 0:
            if geometry_good:
                # Good geometry outside = adventurous exploration
                if abs(boundary_distance) < self.edge_margin * 2:
                    confidence = min(1.0, abs(boundary_distance) / self.edge_margin + 0.3)
                    return "adventurous_fish", confidence, "medium"
                else:
                    # Far outside but good geometry - still adventurous but higher risk
                    confidence = min(1.0, abs(boundary_distance) / (self.edge_margin * 4) + 0.5)
                    return "adventurous_fish", confidence, "medium_high"
            else:
                # Bad geometry outside
                if lof_outlier:
                    confidence = min(1.0, abs(lof_score - self.outlier_threshold) / 2 + 0.5)
                    return "structural_hallucination", confidence, "very_high"
                else:
                    # Bad geometry but LOF normal - high risk adventurous
                    confidence = min(1.0, abs(boundary_distance) / self.edge_margin + 0.4)
                    return "adventurous_fish", confidence, "high"

        # 2. Inside boundary

        # 2a. Sparse region (frontier) - geometry is primary indicator
        if is_sparse:
            if geometry_good:
                # Good geometry in sparse = valid frontier (even if LOF outlier)
                confidence = min(1.0, (self.sparse_threshold - density_percentile) / self.sparse_threshold + 0.5)
                return "frontier_fish", confidence, "low"
            else:
                # Bad geometry in sparse
                if lof_outlier:
                    # Both geometry bad AND density anomaly → hallucination
                    confidence = min(1.0, abs(lof_score - self.outlier_threshold) / 2 + 0.5)
                    return "structural_hallucination", confidence, "very_high"
                else:
                    # Bad geometry but fits local density → high risk frontier
                    confidence = min(1.0, (self.sparse_threshold - density_percentile) / self.sparse_threshold + 0.4)
                    return "frontier_fish", confidence, "high"

        # 2b. Dense region - position-based classification
        # In dense regions, geometry doesn't determine category (only sparse/outside uses geometry)

        # LOF outlier in dense region
        if lof_outlier:
            confidence = min(1.0, abs(lof_score - self.outlier_threshold) / 2 + 0.5)
            return "structural_hallucination", confidence, "very_high"

        # Near edge (dense region)
        if boundary_distance < self.edge_margin:
            confidence = min(1.0, (self.edge_margin - boundary_distance) / self.edge_margin + 0.4)
            return "edge_fish", confidence, "medium"

        # Very central/deep → redundant
        if depth_score < self.depth_threshold / 100:
            confidence = min(1.0, (self.depth_threshold / 100 - depth_score) * 5 + 0.5)
            return "redundant_fish", confidence, "very_low"

        # Default: normal fish in water
        confidence = min(1.0, (self.stability_threshold - manifold_distance) / self.stability_threshold + 0.4)
        return "fish_in_water", confidence, "low"

    # =========================================================================
    # ID Tracking Methods
    # =========================================================================

    def get_ids_by_category(
        self,
        results: List[ManifoldFishResult],
        category: str,
    ) -> List[str]:
        """Get material IDs for a specific category."""
        return [r.material_id for r in results if r.category == category]

    def get_all_ids_by_category(
        self,
        results: List[ManifoldFishResult],
    ) -> Dict[str, List[str]]:
        """Get material IDs grouped by all categories."""
        return {cat: self.get_ids_by_category(results, cat) for cat in CATEGORIES.keys()}

    def get_indices_by_category(
        self,
        results: List[ManifoldFishResult],
        category: str,
    ) -> List[int]:
        """
        Get indices (positions in the original generated array) for a specific category.

        Args:
            results: Analysis results from analyze()
            category: Category name (e.g., "frontier_fish", "fish_in_water")

        Returns:
            List of indices into the original generated embeddings array.

        Example:
            >>> results = analyzer.analyze(generated_embeddings, material_ids)
            >>> frontier_indices = analyzer.get_indices_by_category(results, "frontier_fish")
            >>> frontier_embeddings = generated_embeddings[frontier_indices]
        """
        return [r.index for r in results if r.category == category]

    def get_all_indices_by_category(
        self,
        results: List[ManifoldFishResult],
    ) -> Dict[str, List[int]]:
        """
        Get indices grouped by all categories.

        Args:
            results: Analysis results from analyze()

        Returns:
            Dictionary mapping category names to lists of indices.

        Example:
            >>> results = analyzer.analyze(generated_embeddings, material_ids)
            >>> indices_by_cat = analyzer.get_all_indices_by_category(results)
            >>> print(indices_by_cat["frontier_fish"])  # [3, 7, 12, ...]
            >>> print(indices_by_cat["structural_hallucination"])  # [45, 89, ...]
        """
        return {cat: self.get_indices_by_category(results, cat) for cat in CATEGORIES.keys()}

    def get_results_by_category(
        self,
        results: List[ManifoldFishResult],
        category: str,
    ) -> List[ManifoldFishResult]:
        """Get full results for a specific category."""
        return [r for r in results if r.category == category]

    def get_category_summary(
        self,
        results: List[ManifoldFishResult],
    ) -> Dict[str, CategorySummary]:
        """Get detailed summary for each category."""
        total = len(results)
        summaries = {}

        for cat, info in CATEGORIES.items():
            cat_results = self.get_results_by_category(results, cat)
            cat_ids = [r.material_id for r in cat_results]

            summaries[cat] = CategorySummary(
                category=cat,
                count=len(cat_results),
                percentage=100 * len(cat_results) / total if total > 0 else 0,
                risk_level=info["risk"],
                description=info["description"],
                action=info["action"],
                material_ids=cat_ids,
                results=cat_results,
            )

        return summaries

    def get_top_candidates(
        self,
        results: List[ManifoldFishResult],
        category: str = "frontier_fish",
        top_k: int = 10,
        sort_by: str = "confidence",
    ) -> List[ManifoldFishResult]:
        """
        Get top candidates from a specific category.

        Args:
            results: Analysis results
            category: Category to filter
            top_k: Number of top candidates
            sort_by: Metric to sort by ("confidence", "novelty", "density_percentile")
        """
        filtered = [r for r in results if r.category == category]

        if sort_by == "confidence":
            sorted_results = sorted(filtered, key=lambda x: x.confidence, reverse=True)
        elif sort_by == "density_percentile":
            # Lower density = more frontier
            sorted_results = sorted(filtered, key=lambda x: x.density_percentile)
        else:
            sorted_results = sorted(filtered, key=lambda x: x.confidence, reverse=True)

        return sorted_results[:top_k]

    # =========================================================================
    # Neighbor Query Methods
    # =========================================================================

    def print_neighbors(
        self,
        generated: np.ndarray,
        material_ids: Optional[List[str]] = None,
        n_neighbors: Optional[int] = None,
        use_local_pca_neighbors: bool = True,
        diagnose_pca: bool = False,
        pca_components: int = 3,
    ) -> np.ndarray:
        """
        Print and return the neighbor indices (in reference set) for generated materials.

        Args:
            generated: Generated embeddings, shape (n_generated, embedding_dim)
            material_ids: Optional list of material IDs for the generated set
            n_neighbors: Number of neighbors to find. If None, uses the analyzer's
                         default (n_neighbors_local_pca if use_local_pca_neighbors,
                         else n_neighbors)
            use_local_pca_neighbors: If True, use the local PCA k-NN model which
                                     has more neighbors. If False, use the standard k-NN.
            diagnose_pca: If True, print diagnostic comparing full-space vs PCA-space
                          distances to help understand visualization discrepancies.
            pca_components: Number of PCA components to use for diagnosis (default 3).

        Returns:
            neighbor_indices: Array of shape (n_generated, n_neighbors) containing
                              indices into the reference set.

        Example:
            >>> neighbor_indices = analyzer.print_neighbors(
            ...     generated_embeddings,
            ...     material_ids=["gen_0", "gen_1", "gen_2"]
            ... )
            Material gen_0: neighbors in reference = [42, 17, 89, ...]
            Material gen_1: neighbors in reference = [5, 23, 42, ...]
            ...
        """
        generated = np.atleast_2d(generated)
        n_generated = len(generated)

        # Set up material IDs
        if material_ids is None:
            mat_ids = [f"material_{i}" for i in range(n_generated)]
        else:
            mat_ids = list(material_ids)
            if len(mat_ids) != n_generated:
                raise ValueError(f"Length mismatch: {len(mat_ids)} IDs for {n_generated} materials")

        # Choose k-NN model and determine n_neighbors
        if use_local_pca_neighbors:
            knn_model = self.knn_local_pca
            default_k = self.n_neighbors_local_pca
        else:
            knn_model = self.knn
            default_k = self.n_neighbors

        k = n_neighbors if n_neighbors is not None else default_k
        k = min(k, self.n_reference)  # Can't have more neighbors than reference points

        # Query neighbors
        distances, neighbor_indices = knn_model.kneighbors(generated, n_neighbors=k)

        # Print results
        print(f"\nNeighbor indices for {n_generated} generated materials")
        print(f"(showing {k} nearest neighbors in reference set of {self.n_reference} points)")
        print("-" * 60)

        for i, (mat_id, indices, dists) in enumerate(zip(mat_ids, neighbor_indices, distances)):
            print(f"{mat_id}: neighbors = {indices.tolist()}")

        print("-" * 60)

        # PCA diagnostic
        if diagnose_pca:
            self._diagnose_neighbor_distances(
                generated, neighbor_indices, distances, mat_ids, pca_components
            )

        return neighbor_indices

    def _diagnose_neighbor_distances(
        self,
        generated: np.ndarray,
        neighbor_indices: np.ndarray,
        full_space_distances: np.ndarray,
        mat_ids: List[str],
        n_components: int = 3,
    ) -> None:
        """
        Diagnostic to compare neighbor distances in full latent space vs PCA projection.

        This helps explain why neighbors might appear scattered in a 3D PCA visualization
        even though they are genuinely close in the full high-dimensional space.
        """
        from sklearn.decomposition import PCA

        print("\n" + "=" * 70)
        print("PCA DISTANCE DIAGNOSTIC")
        print("=" * 70)

        # Fit PCA on combined data (reference + generated)
        combined = np.vstack([self.reference, generated])
        pca = PCA(n_components=n_components)
        pca.fit(combined)

        # Transform reference and generated to PCA space
        ref_pca = pca.transform(self.reference)
        gen_pca = pca.transform(generated)

        # Report variance explained
        variance_explained = pca.explained_variance_ratio_
        total_variance = sum(variance_explained)
        print(f"\nLatent space dimensionality: {self.reference.shape[1]}")
        print(f"PCA components: {n_components}")
        print(f"Variance explained by each PC: {[f'{v:.1%}' for v in variance_explained]}")
        print(f"Total variance explained: {total_variance:.1%}")
        print(f"Variance NOT captured (hidden dimensions): {1 - total_variance:.1%}")

        print("\n" + "-" * 70)
        print("Per-material neighbor distance analysis:")
        print("-" * 70)

        for i, mat_id in enumerate(mat_ids):
            indices = neighbor_indices[i]
            full_dists = full_space_distances[i]

            # Compute PCA-space distances to the same neighbors
            gen_point_pca = gen_pca[i]
            neighbor_points_pca = ref_pca[indices]
            pca_dists = np.linalg.norm(neighbor_points_pca - gen_point_pca, axis=1)

            # Statistics
            mean_full = np.mean(full_dists)
            mean_pca = np.mean(pca_dists)
            std_full = np.std(full_dists)
            std_pca = np.std(pca_dists)

            # Compute how much distance is "hidden" in non-PCA dimensions
            # full_dist^2 = pca_dist^2 + hidden_dist^2 (Pythagorean in orthogonal subspaces)
            hidden_dists_sq = full_dists**2 - pca_dists**2
            hidden_dists_sq = np.maximum(hidden_dists_sq, 0)  # Numerical safety
            hidden_dists = np.sqrt(hidden_dists_sq)
            mean_hidden = np.mean(hidden_dists)

            # Ratio: how much of the distance is visible in PCA vs hidden
            pca_fraction = mean_pca / mean_full if mean_full > 0 else 0
            hidden_fraction = mean_hidden / mean_full if mean_full > 0 else 0

            print(f"\n{mat_id}:")
            print(f"  Mean distance to neighbors (full {self.reference.shape[1]}D): {mean_full:.4f} ± {std_full:.4f}")
            print(f"  Mean distance to neighbors (PCA {n_components}D):  {mean_pca:.4f} ± {std_pca:.4f}")
            print(f"  Mean 'hidden' distance (dimensions {n_components+1}+): {mean_hidden:.4f}")
            print(f"  Distance visible in PCA: {pca_fraction:.1%}")
            print(f"  Distance hidden (not visible): {hidden_fraction:.1%}")

            if hidden_fraction > 0.5:
                print(f"  ⚠️  WARNING: >50% of neighbor distance is in hidden dimensions!")
                print(f"     Neighbors may appear scattered in {n_components}D visualization.")

        # Summary statistics across all materials
        print("\n" + "-" * 70)
        print("Summary:")
        print("-" * 70)

        all_full_dists = full_space_distances.flatten()
        all_pca_dists = []
        for i in range(len(generated)):
            indices = neighbor_indices[i]
            gen_point_pca = gen_pca[i]
            neighbor_points_pca = ref_pca[indices]
            pca_dists = np.linalg.norm(neighbor_points_pca - gen_point_pca, axis=1)
            all_pca_dists.extend(pca_dists)
        all_pca_dists = np.array(all_pca_dists)

        overall_pca_fraction = np.mean(all_pca_dists) / np.mean(all_full_dists)
        print(f"Overall: {overall_pca_fraction:.1%} of neighbor distance visible in {n_components}D PCA")

        if overall_pca_fraction < 0.7:
            print(f"\n💡 INSIGHT: Only {overall_pca_fraction:.1%} of distance is captured by {n_components} PCs.")
            print("   This explains why neighbors appear scattered in visualization.")
            print("   The neighbors ARE close in the full latent space - just not in the")
            print("   3 dimensions you're visualizing.")
        elif total_variance < 0.7:
            print(f"\n💡 INSIGHT: PCA only captures {total_variance:.1%} of total variance.")
            print("   Consider using more PCA components or accepting that 3D visualization")
            print("   cannot fully represent the high-dimensional neighbor relationships.")

        print("=" * 70 + "\n")

    def print_statistics(
        self,
        results: List[ManifoldFishResult],
        material_ids: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
        show_neighbors: bool = True,
        n_neighbors_to_show: int = 3,
    ) -> None:
        """
        Print detailed statistics for specific materials that led to their classification.

        Args:
            results: Analysis results from analyze()
            material_ids: List of material IDs to show. If None, shows all or uses indices.
            indices: List of indices (positions in results) to show. Ignored if material_ids given.
            show_neighbors: Whether to show nearest reference IDs
            n_neighbors_to_show: Number of nearest neighbors to display

        Example:
            >>> # Show statistics for specific materials
            >>> analyzer.print_statistics(results, material_ids=["gen_0", "gen_5"])

            >>> # Show statistics for first 3 materials
            >>> analyzer.print_statistics(results, indices=[0, 1, 2])

            >>> # Show all frontier fish
            >>> frontier_ids = analyzer.get_ids_by_category(results, "frontier_fish")
            >>> analyzer.print_statistics(results, material_ids=frontier_ids)
        """
        # Build lookup dict for fast access
        results_by_id = {r.material_id: r for r in results}
        results_by_idx = {r.index: r for r in results}

        # Determine which results to show
        if material_ids is not None:
            to_show = []
            for mid in material_ids:
                if mid in results_by_id:
                    to_show.append(results_by_id[mid])
                else:
                    print(f"Warning: material_id '{mid}' not found in results")
        elif indices is not None:
            to_show = []
            for idx in indices:
                if idx in results_by_idx:
                    to_show.append(results_by_idx[idx])
                else:
                    print(f"Warning: index {idx} not found in results")
        else:
            to_show = results

        if not to_show:
            print("No materials to show.")
            return

        print("\n" + "=" * 80)
        print("DETAILED STATISTICS FOR MATERIALS")
        print("=" * 80)

        for r in to_show:
            print(f"\n{'─' * 80}")
            print(f"Material: {r.material_id}  (index: {r.index})")
            print(f"{'─' * 80}")

            # Classification result
            cat_display = r.category.replace("_", " ").title()
            print(f"\n  CLASSIFICATION:")
            print(f"    Category:        {cat_display}")
            print(f"    Confidence:      {r.confidence:.3f}")
            print(f"    Risk Level:      {r.risk_level.upper()}")

            # Position metrics
            print(f"\n  POSITION METRICS:")
            print(f"    Manifold Distance:   {r.manifold_distance:.4f}  (normalized distance to nearest references)")
            print(f"    Depth Score:         {r.depth_score:.4f}  (0=deep inside, 1=shallow/edge)")
            print(f"    Boundary Distance:   {r.boundary_distance:+.4f}  (+inside, -outside manifold)")

            # Density metrics
            print(f"\n  DENSITY METRICS:")
            print(f"    Local Density:       {r.local_density:.4f}")
            print(f"    Density Percentile:  {r.density_percentile:.1f}%  (vs reference distribution)")
            print(f"    LOF Score:           {r.lof_score:.4f}  (≈-1 normal, <<-1 outlier)")

            # Geometry consistency
            print(f"\n  GEOMETRY CONSISTENCY:")
            print(f"    Local PCA Residual:  {r.local_pca_residual:.4f}  (reconstruction error)")
            consistent_str = "Yes" if r.geometry_consistent else "No (ATYPICAL GEOMETRY)"
            print(f"    Geometry Consistent: {consistent_str}")

            # Ensemble variance (if available)
            if r.ensemble_variance > 0:
                print(f"\n  ENSEMBLE:")
                print(f"    Variance:            {r.ensemble_variance:.4f}")

            # Nearest neighbors
            if show_neighbors and r.nearest_reference_ids:
                n_to_show = min(n_neighbors_to_show, len(r.nearest_reference_ids))
                neighbors_str = ", ".join(r.nearest_reference_ids[:n_to_show])
                print(f"\n  NEAREST REFERENCES:")
                print(f"    Top {n_to_show}: {neighbors_str}")

        print("\n" + "=" * 80)
        print(f"Showed {len(to_show)} material(s)")
        print("=" * 80 + "\n")

    def get_result_by_id(
        self,
        results: List[ManifoldFishResult],
        material_id: str,
    ) -> Optional[ManifoldFishResult]:
        """
        Get a single result by material ID.

        Args:
            results: Analysis results from analyze()
            material_id: The material ID to find

        Returns:
            ManifoldFishResult if found, None otherwise
        """
        for r in results:
            if r.material_id == material_id:
                return r
        return None

    def get_result_by_index(
        self,
        results: List[ManifoldFishResult],
        index: int,
    ) -> Optional[ManifoldFishResult]:
        """
        Get a single result by index.

        Args:
            results: Analysis results from analyze()
            index: The index to find

        Returns:
            ManifoldFishResult if found, None otherwise
        """
        for r in results:
            if r.index == index:
                return r
        return None

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_results(
        self,
        results: List[ManifoldFishResult],
        output_path: str,
        format: str = 'auto',
    ):
        """
        Export results to file (CSV or pickle).

        Args:
            results: Analysis results
            output_path: Output file path
            format: 'csv', 'pickle', or 'auto' (detect from extension)
        """
        if format == 'auto':
            if output_path.endswith('.csv'):
                format = 'csv'
            elif output_path.endswith('.pkl') or output_path.endswith('.pickle'):
                format = 'pickle'
            else:
                format = 'csv'

        if format == 'csv':
            self._export_csv(results, output_path)
        elif format == 'pickle':
            self._export_pickle(results, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_csv(self, results: List[ManifoldFishResult], output_path: str):
        """Export results to CSV with all metrics."""
        data = []
        for r in results:
            row = {
                'index': r.index,
                'material_id': r.material_id,
                'category': r.category,
                'risk_level': r.risk_level,
                'confidence': r.confidence,
                'manifold_distance': r.manifold_distance,
                'depth_score': r.depth_score,
                'boundary_distance': r.boundary_distance,
                'local_density': r.local_density,
                'density_percentile': r.density_percentile,
                'local_pca_residual': r.local_pca_residual,
                'geometry_consistent': r.geometry_consistent,
                'ensemble_variance': r.ensemble_variance,
                'nearest_ref_1': r.nearest_reference_ids[0] if r.nearest_reference_ids else None,
                'nearest_ref_2': r.nearest_reference_ids[1] if len(r.nearest_reference_ids) > 1 else None,
                'nearest_ref_3': r.nearest_reference_ids[2] if len(r.nearest_reference_ids) > 2 else None,
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

    def _export_pickle(self, results: List[ManifoldFishResult], output_path: str):
        """Export results to pickle."""
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results exported to {output_path}")

    def export_embeddings_by_category(
        self,
        results: List[ManifoldFishResult],
        generated_embeddings: np.ndarray,
        output_dir: str = ".",
        prefix: str = "embeddings",
    ):
        """Export embeddings and IDs for each category to separate files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        generated = np.asarray(generated_embeddings)

        for cat in CATEGORIES.keys():
            cat_results = self.get_results_by_category(results, cat)
            if not cat_results:
                continue

            cat_indices = [r.index for r in cat_results]
            cat_ids = [r.material_id for r in cat_results]
            cat_embeddings = generated[cat_indices]

            # Save embeddings
            emb_path = os.path.join(output_dir, f"{prefix}_{cat}.npy")
            np.save(emb_path, cat_embeddings)

            # Save IDs
            ids_path = os.path.join(output_dir, f"{prefix}_{cat}_ids.txt")
            with open(ids_path, 'w') as f:
                for mid in cat_ids:
                    f.write(f"{mid}\n")

            print(f"{cat}: saved {len(cat_results)} structures to {emb_path}")

    # =========================================================================
    # Summary and Statistics
    # =========================================================================

    def get_summary(self, results: List[ManifoldFishResult]) -> Dict:
        """Get summary statistics of classification."""
        categories = [r.category for r in results]

        summary = {
            "total": len(results),
            "thresholds": {
                "stability": self.stability_threshold,
                "outlier": self.outlier_threshold,
                "geometry": self.geometry_threshold,
                "sparse": self.sparse_threshold,
                "depth": self.depth_threshold,
                "edge_margin": self.edge_margin,
            },
            "boundary_method": self.boundary_method,
            "local_geometry_mode": self.local_geometry_mode,
            "curvature_adjusted": self.curvature_adjusted,
        }

        # Category counts
        for cat in CATEGORIES.keys():
            summary[cat] = categories.count(cat)
            summary[f"{cat}_pct"] = 100 * summary[cat] / summary["total"] if summary["total"] > 0 else 0

        # Metric statistics
        summary["metrics"] = {
            "mean_manifold_distance": np.mean([r.manifold_distance for r in results]),
            "mean_depth_score": np.mean([r.depth_score for r in results]),
            "mean_boundary_distance": np.mean([r.boundary_distance for r in results]),
            "mean_local_density": np.mean([r.local_density for r in results]),
            "mean_density_percentile": np.mean([r.density_percentile for r in results]),
            "mean_local_pca_residual": np.mean([r.local_pca_residual for r in results]),
            "geometry_consistent_pct": 100 * sum(r.geometry_consistent for r in results) / len(results),
        }

        return summary

    def print_summary(self, results: List[ManifoldFishResult]):
        """Print formatted summary with example IDs."""
        summary = self.get_summary(results)
        ids_by_cat = self.get_all_ids_by_category(results)

        print("=" * 80)
        print("Manifold Fish-Water Analysis Summary")
        print("=" * 80)
        print(f"\nTotal structures analyzed: {summary['total']}")
        print(f"Boundary method: {summary['boundary_method']}")
        print(f"Local geometry mode: {self.local_geometry_mode}")
        print(f"Curvature adjustment: {'Enabled' if self.curvature_adjusted else 'Disabled'}")
        print(f"Embedding dimension: {self.dim}")

        print("\n" + "-" * 80)
        print("Category Distribution:")
        print("-" * 80)

        for cat, info in CATEGORIES.items():
            count = summary[cat]
            pct = summary[f"{cat}_pct"]
            ids = ids_by_cat[cat]

            # Category name and stats
            cat_display = cat.replace("_", " ").title()
            risk_display = f"[{info['risk'].upper()}]"
            print(f"\n  {cat_display:25} {risk_display:15} {count:>5} ({pct:>5.1f}%)")
            print(f"    {info['description']}")
            print(f"    Action: {info['action']}")

            if ids:
                example_ids = ids[:3]
                ids_str = ", ".join(str(eid) for eid in example_ids)
                if len(ids) > 3:
                    ids_str += f", ... (+{len(ids)-3} more)"
                print(f"    Example IDs: {ids_str}")

        print("\n" + "-" * 80)
        print("Metric Statistics:")
        print("-" * 80)
        metrics = summary["metrics"]
        print(f"  Mean manifold distance:    {metrics['mean_manifold_distance']:>8.3f}")
        print(f"  Mean depth score:          {metrics['mean_depth_score']:>8.3f}")
        print(f"  Mean boundary distance:    {metrics['mean_boundary_distance']:>8.3f}")
        print(f"  Mean density percentile:   {metrics['mean_density_percentile']:>8.1f}%")
        print(f"  Mean local PCA residual:   {metrics['mean_local_pca_residual']:>8.3f}")
        print(f"  Geometry consistent:       {metrics['geometry_consistent_pct']:>8.1f}%")

        print("\n" + "-" * 80)
        print("Thresholds Used:")
        print("-" * 80)
        thresholds = summary["thresholds"]
        print(f"  Stability threshold:       {thresholds['stability']:>8.2f}")
        print(f"  Outlier threshold (LOF):   {thresholds['outlier']:>8.2f}")
        print(f"  Geometry threshold:        {thresholds['geometry']:>8.2f}")
        print(f"  Sparse percentile:         {thresholds['sparse']:>8.1f}%")
        print(f"  Depth percentile:          {thresholds['depth']:>8.1f}%")
        print(f"  Edge margin:               {thresholds['edge_margin']:>8.2f}")

        print("\n" + "-" * 80)
        print("Recommendations:")
        print("-" * 80)

        if summary['frontier_fish'] > 0:
            print(f"  -> {summary['frontier_fish']} frontier fish - check risk_level (low=good geometry, high=bad geometry)")
        if summary['adventurous_fish'] > 0:
            print(f"  -> {summary['adventurous_fish']} adventurous fish - outside boundary, check risk_level")
        if summary['edge_fish'] > 0:
            print(f"  -> {summary['edge_fish']} edge fish - validate carefully")
        if summary['structural_hallucination_pct'] > 10:
            print(f"  -> {summary['structural_hallucination_pct']:.1f}% structural hallucinations - reject these")
        if summary['redundant_fish_pct'] > 30:
            print(f"  -> {summary['redundant_fish_pct']:.1f}% redundant - consider diversity sampling")

        print("=" * 80)

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot_distribution(
        self,
        results: List[ManifoldFishResult],
        figsize: Tuple[int, int] = (18, 16),
        save_path: Optional[str] = None,
        show_ids: bool = False,
    ):
        """
        Plot comprehensive distribution of manifold fish categories.

        Args:
            results: Analysis results
            figsize: Figure size
            save_path: Path to save figure
            show_ids: If True, annotate points with material IDs (small datasets only)
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)

        # Extract data
        manifold_dist = np.array([r.manifold_distance for r in results])
        depth_score = np.array([r.depth_score for r in results])
        boundary_dist = np.array([r.boundary_distance for r in results])
        density_pct = np.array([r.density_percentile for r in results])
        local_residual = np.array([r.local_pca_residual for r in results])
        lof_score = np.array([r.lof_score for r in results])
        risk_levels = np.array([r.risk_level for r in results])
        categories = np.array([r.category for r in results])

        colors = {cat: info["color"] for cat, info in CATEGORIES.items()}

        # 1. Pie chart
        ax1 = axes[0, 0]
        summary = self.get_summary(results)
        sizes = [summary[cat] for cat in CATEGORIES.keys()]
        pie_colors = [colors[cat] for cat in CATEGORIES.keys()]
        pie_labels = [cat.replace("_", " ").title() for cat in CATEGORIES.keys()]

        # Filter out zero-size wedges for cleaner pie
        non_zero = [(s, c, l) for s, c, l in zip(sizes, pie_colors, pie_labels) if s > 0]
        if non_zero:
            sizes_nz, colors_nz, labels_nz = zip(*non_zero)
            ax1.pie(sizes_nz, labels=labels_nz, colors=colors_nz,
                   autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                   startangle=90, textprops={'fontsize': 8})
        ax1.set_title("Category Distribution", fontsize=12, fontweight='bold')

        # 2. Manifold distance vs Boundary distance
        ax2 = axes[0, 1]
        for cat in CATEGORIES.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax2.scatter(manifold_dist[mask], boundary_dist[mask],
                           c=colors[cat], label=cat.replace("_", " ").title(),
                           alpha=0.6, s=20)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Boundary')
        ax2.axvline(x=self.stability_threshold, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel("Manifold Distance", fontsize=10)
        ax2.set_ylabel("Boundary Distance", fontsize=10)
        ax2.set_title("Position Analysis", fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=7)

        # 3. Density percentile vs Local PCA residual
        ax3 = axes[0, 2]
        for cat in CATEGORIES.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax3.scatter(density_pct[mask], local_residual[mask],
                           c=colors[cat], label=cat.replace("_", " ").title(),
                           alpha=0.6, s=20)
        ax3.axhline(y=self.geometry_threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'Geometry threshold ({self.geometry_threshold})')
        ax3.axvline(x=self.sparse_threshold, color='blue', linestyle=':', alpha=0.5)
        ax3.set_xlabel("Density Percentile", fontsize=10)
        ax3.set_ylabel("Local PCA Residual", fontsize=10)
        ax3.set_title("Density vs Geometry Consistency", fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=7)

        # 4. Histogram of manifold distances by category
        ax4 = axes[1, 0]
        for cat in CATEGORIES.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax4.hist(manifold_dist[mask], bins=30, alpha=0.5, color=colors[cat],
                        label=cat.replace("_", " ").title())
        ax4.axvline(x=self.stability_threshold, color='black', linestyle='--',
                   label=f'Stability threshold ({self.stability_threshold})')
        ax4.set_xlabel("Manifold Distance", fontsize=10)
        ax4.set_ylabel("Count", fontsize=10)
        ax4.set_title("Manifold Distance Distribution", fontsize=12, fontweight='bold')
        ax4.legend(fontsize=7)

        # 5. Histogram of local PCA residuals
        ax5 = axes[1, 1]
        for cat in CATEGORIES.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax5.hist(local_residual[mask], bins=30, alpha=0.5, color=colors[cat],
                        label=cat.replace("_", " ").title())
        ax5.axvline(x=self.geometry_threshold, color='red', linestyle='--',
                   label=f'Geometry threshold ({self.geometry_threshold})')
        ax5.set_xlabel("Local PCA Residual", fontsize=10)
        ax5.set_ylabel("Count", fontsize=10)
        ax5.set_title("Geometry Consistency Distribution", fontsize=12, fontweight='bold')
        ax5.legend(fontsize=7)

        # 6. Depth vs Boundary distance
        ax6 = axes[1, 2]
        for cat in CATEGORIES.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax6.scatter(depth_score[mask], boundary_dist[mask],
                           c=colors[cat], label=cat.replace("_", " ").title(),
                           alpha=0.6, s=20)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.axhline(y=self.edge_margin, color='orange', linestyle=':', alpha=0.5)
        ax6.set_xlabel("Depth Score (0=central, 1=peripheral)", fontsize=10)
        ax6.set_ylabel("Boundary Distance", fontsize=10)
        ax6.set_title("Depth vs Boundary Position", fontsize=12, fontweight='bold')
        ax6.legend(loc='upper right', fontsize=7)

        # 7. Density percentile vs LOF score (key for sparse region decision)
        ax7 = axes[2, 0]
        for cat in CATEGORIES.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax7.scatter(density_pct[mask], lof_score[mask],
                           c=colors[cat], label=cat.replace("_", " ").title(),
                           alpha=0.6, s=20)
        ax7.axhline(y=self.outlier_threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'LOF threshold ({self.outlier_threshold})')
        ax7.axvline(x=self.sparse_threshold, color='blue', linestyle=':', alpha=0.5,
                   label=f'Sparse threshold ({self.sparse_threshold}%)')
        ax7.set_xlabel("Density Percentile", fontsize=10)
        ax7.set_ylabel("LOF Score (more negative = outlier)", fontsize=10)
        ax7.set_title("Density vs LOF (Sparse Region Decision)", fontsize=12, fontweight='bold')
        ax7.legend(loc='lower right', fontsize=7)

        # 8. Local PCA residual vs LOF score (geometry + LOF interaction)
        ax8 = axes[2, 1]
        for cat in CATEGORIES.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax8.scatter(local_residual[mask], lof_score[mask],
                           c=colors[cat], label=cat.replace("_", " ").title(),
                           alpha=0.6, s=20)
        ax8.axhline(y=self.outlier_threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'LOF threshold ({self.outlier_threshold})')
        ax8.axvline(x=self.geometry_threshold, color='green', linestyle=':', alpha=0.5,
                   label=f'Geometry threshold ({self.geometry_threshold})')
        ax8.set_xlabel("Local PCA Residual", fontsize=10)
        ax8.set_ylabel("LOF Score (more negative = outlier)", fontsize=10)
        ax8.set_title("Geometry vs LOF (Secondary Filter)", fontsize=12, fontweight='bold')
        ax8.legend(loc='lower right', fontsize=7)

        # 9. Risk level distribution
        ax9 = axes[2, 2]
        risk_order = ["very_low", "low", "medium", "medium_high", "high", "very_high"]
        risk_colors = {
            "very_low": "#2ecc71", "low": "#27ae60", "low_medium": "#f1c40f",
            "medium": "#f39c12", "medium_high": "#e67e22", "high": "#e74c3c", "very_high": "#c0392b"
        }
        risk_counts = {r: np.sum(risk_levels == r) for r in risk_order}
        # Filter out zero counts
        non_zero_risks = [(r, c) for r, c in risk_counts.items() if c > 0]
        if non_zero_risks:
            risks, counts = zip(*non_zero_risks)
            bars = ax9.bar(range(len(risks)), counts,
                          color=[risk_colors.get(r, '#888888') for r in risks])
            ax9.set_xticks(range(len(risks)))
            ax9.set_xticklabels([r.replace("_", " ").title() for r in risks], rotation=45, ha='right')
            ax9.set_ylabel("Count", fontsize=10)
            ax9.set_title("Risk Level Distribution", fontsize=12, fontweight='bold')
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=9)

        # Apply consistent style to all axes
        for ax in axes.flatten():
            apply_plot_style(ax, tick_labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig

    def plot_3d_manifold(
        self,
        results: List[ManifoldFishResult],
        generated_embeddings: np.ndarray,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
        show_reference: bool = True,
        max_ref_points: int = 500,
    ):
        """
        Plot 3D visualization of manifold (only for 3D embeddings).

        Args:
            results: Analysis results
            generated_embeddings: Generated embeddings (must be 3D)
            figsize: Figure size
            save_path: Path to save figure
            show_reference: Whether to show reference points
            max_ref_points: Maximum reference points to show (for performance)
        """
        generated = np.asarray(generated_embeddings)

        if generated.shape[1] != 3:
            print(f"3D plot requires 3D embeddings, got {generated.shape[1]}D")
            return None

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        colors = {cat: info["color"] for cat, info in CATEGORIES.items()}
        categories = np.array([r.category for r in results])

        # Plot reference points
        if show_reference:
            ref_subset = self.reference
            if len(ref_subset) > max_ref_points:
                idx = np.random.choice(len(ref_subset), max_ref_points, replace=False)
                ref_subset = ref_subset[idx]
            ax.scatter(ref_subset[:, 0], ref_subset[:, 1], ref_subset[:, 2],
                      c='lightgray', alpha=0.2, s=5, label='Reference')

        # Plot generated points by category
        for cat in CATEGORIES.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax.scatter(generated[mask, 0], generated[mask, 1], generated[mask, 2],
                          c=colors[cat], label=cat.replace("_", " ").title(),
                          alpha=0.7, s=30)

        # Plot centroid
        ax.scatter(*self.centroid, c='black', marker='*', s=200, label='Centroid')

        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_zlabel('PC3', fontsize=12)
        ax.set_title('3D Manifold Visualization', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

        # Apply tick label style (3D axes don't have standard spines)
        ax.tick_params(axis='both', which='both', labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig

    def plot_metric_correlations(
        self,
        results: List[ManifoldFishResult],
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ):
        """Plot correlation matrix of all metrics."""
        import seaborn as sns

        # Build dataframe of metrics
        data = {
            'manifold_distance': [r.manifold_distance for r in results],
            'depth_score': [r.depth_score for r in results],
            'boundary_distance': [r.boundary_distance for r in results],
            'local_density': [r.local_density for r in results],
            'density_percentile': [r.density_percentile for r in results],
            'local_pca_residual': [r.local_pca_residual for r in results],
            'confidence': [r.confidence for r in results],
        }
        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=figsize)
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', ax=ax, square=True)
        ax.set_title('Metric Correlations', fontsize=12, fontweight='bold')

        # Apply consistent style
        apply_plot_style(ax, tick_labelsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """
    Example demonstrating ManifoldFishAnalyzer with synthetic data.

    Replace the synthetic data with your actual material-level embeddings.
    """

    print("=" * 80)
    print("Manifold Fish-Water Analysis - Example")
    print("=" * 80)

    # =========================================================================
    # Generate synthetic data (replace with your actual data)
    # =========================================================================

    np.random.seed(42)
    n_ref, n_gen = 1000, 300
    dim = 3  # 3D embeddings

    # Reference: known stable structures (the "water")
    reference_embeddings = np.random.randn(n_ref, dim) * 0.5
    reference_ids = [f"mp-{i:05d}" for i in range(n_ref)]

    # Generated: mix of categories
    gen_redundant = np.random.randn(30, dim) * 0.2  # Very central
    gen_normal = np.random.randn(80, dim) * 0.5     # Normal fish in water
    gen_frontier = np.random.randn(40, dim) * 0.5   # Will be in sparse regions
    gen_frontier += np.array([0.8, 0.8, 0.8])       # Shift to sparse area
    gen_edge = np.random.randn(50, dim) * 0.3       # Near boundary
    gen_edge += np.array([1.2, 0.0, 0.0])
    gen_adventurous = np.random.randn(40, dim) * 0.4  # Slightly outside
    gen_adventurous += np.array([1.8, 0.5, 0.3])
    gen_hallucination = np.random.randn(30, dim) * 0.3  # Inside but wrong geometry
    gen_hallucination[:, 2] += np.random.randn(30) * 2  # Add noise in one direction
    gen_invalid = np.random.randn(30, dim) * 1.5 + np.array([4.0, 3.0, 2.0])

    generated_embeddings = np.vstack([
        gen_redundant, gen_normal, gen_frontier, gen_edge,
        gen_adventurous, gen_hallucination, gen_invalid
    ])
    generated_ids = [f"gen-{i:04d}" for i in range(len(generated_embeddings))]

    # Shuffle
    shuffle_idx = np.random.permutation(len(generated_embeddings))
    generated_embeddings = generated_embeddings[shuffle_idx]
    generated_ids = [generated_ids[i] for i in shuffle_idx]

    print(f"\nReference embeddings: {reference_embeddings.shape}")
    print(f"Generated embeddings: {generated_embeddings.shape}")

    # =========================================================================
    # Run Analysis
    # =========================================================================

    print("\n" + "-" * 80)
    print("Initializing ManifoldFishAnalyzer...")
    print("-" * 80)

    analyzer = ManifoldFishAnalyzer(
        reference_embeddings=reference_embeddings,
        reference_ids=reference_ids,
        n_neighbors=10,
        boundary_method="auto",  # Will use alpha_shape for 3D
        local_geometry_mode="fast",  # "full" (slow), "fast" (recommended), "skip" (no geometry check)
        curvature_adjusted=True,  # Adjust for U-shaped manifolds (high curvature regions)
        stability_threshold=0.5,
        geometry_threshold=0.3,
    )

    print(f"Boundary method: {analyzer.boundary_method}")
    print(f"Local geometry mode: {analyzer.local_geometry_mode}")
    print(f"Curvature adjustment: {'Enabled' if analyzer.curvature_adjusted else 'Disabled'}")
    print(f"Local PCA neighbors: {analyzer.n_neighbors_local_pca}")

    print("\n" + "-" * 80)
    print("Analyzing generated structures...")
    print("-" * 80)

    results = analyzer.analyze(
        generated_embeddings=generated_embeddings,
        material_ids=generated_ids,
    )

    # =========================================================================
    # Print Summary
    # =========================================================================

    analyzer.print_summary(results)

    # =========================================================================
    # Get Top Candidates
    # =========================================================================

    print("\n" + "=" * 80)
    print("Top 5 Frontier Fish (Priority for DFT)")
    print("=" * 80)

    top_frontier = analyzer.get_top_candidates(results, category="frontier_fish", top_k=5)
    for i, r in enumerate(top_frontier, 1):
        print(f"\n{i}. {r.material_id}")
        print(f"   Manifold distance: {r.manifold_distance:.3f}")
        print(f"   Density percentile: {r.density_percentile:.1f}%")
        print(f"   Local PCA residual: {r.local_pca_residual:.3f}")
        print(f"   Confidence: {r.confidence:.3f}")

    print("\n" + "=" * 80)
    print("Top 5 Adventurous Fish (High-Risk Novel)")
    print("=" * 80)

    top_adventurous = analyzer.get_top_candidates(results, category="adventurous_fish", top_k=5)
    for i, r in enumerate(top_adventurous, 1):
        print(f"\n{i}. {r.material_id}")
        print(f"   Boundary distance: {r.boundary_distance:.3f}")
        print(f"   Local PCA residual: {r.local_pca_residual:.3f}")
        print(f"   Confidence: {r.confidence:.3f}")

    # =========================================================================
    # Export Results
    # =========================================================================

    print("\n" + "=" * 80)
    print("Exporting Results")
    print("=" * 80)

    analyzer.export_results(results, "manifold_fish_results.csv")

    # =========================================================================
    # Visualization
    # =========================================================================

    print("\n" + "=" * 80)
    print("Generating Plots")
    print("=" * 80)

    analyzer.plot_distribution(results, save_path="manifold_fish_distribution.png")
    analyzer.plot_3d_manifold(results, generated_embeddings, save_path="manifold_fish_3d.png")

    return analyzer, results


# =============================================================================
# Aggregation Utilities: Atom-Level → Materials-Level
# =============================================================================

def aggregate_atom_embeddings(
    atom_embeddings: np.ndarray,
    atom_to_material: np.ndarray,
    material_ids: Optional[List[str]] = None,
    method: str = "mean",
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate atom-level embeddings to materials-level embeddings.

    Args:
        atom_embeddings: Atom-level embeddings, shape (n_atoms, embed_dim)
        atom_to_material: Array mapping each atom to its material index,
                          shape (n_atoms,). Values should be 0 to n_materials-1.
        material_ids: List of material IDs, length = n_materials (max(atom_to_material)+1).
                      If None, generates IDs as "mat_0", "mat_1", ...
        method: Aggregation method
                - "mean": Average pooling (recommended)
                - "sum": Sum pooling
                - "max": Element-wise max pooling
                - "min": Element-wise min pooling
                - "std": Standard deviation (captures spread)
                - "mean_std": Concatenate mean and std (doubles dimension)

    Returns:
        materials_embeddings: Shape (n_materials, embed_dim) or (n_materials, 2*embed_dim) for mean_std
        material_ids: List of material IDs (same order as embeddings)

    Example:
        >>> # 1000 atoms across 50 materials
        >>> atom_embeddings = np.random.randn(1000, 128)  # (n_atoms, embed_dim)
        >>> atom_to_material = np.repeat(np.arange(50), 20)  # 20 atoms per material
        >>> material_ids = [f"mp-{i}" for i in range(50)]
        >>> material_embeddings, ids = aggregate_atom_embeddings(
        ...     atom_embeddings, atom_to_material, material_ids, method="mean"
        ... )
        >>> print(material_embeddings.shape)  # (50, 128)
    """
    atom_embeddings = np.asarray(atom_embeddings)
    atom_to_material = np.asarray(atom_to_material)

    n_materials = atom_to_material.max() + 1
    embed_dim = atom_embeddings.shape[1]

    # Handle material IDs
    if material_ids is None:
        material_ids = [f"mat_{i}" for i in range(n_materials)]
    else:
        material_ids = list(material_ids)
        if len(material_ids) != n_materials:
            raise ValueError(f"Length mismatch: {len(material_ids)} IDs for {n_materials} materials")

    if method == "mean":
        material_embeddings = np.zeros((n_materials, embed_dim))
        counts = np.zeros(n_materials)
        for i, mat_idx in enumerate(atom_to_material):
            material_embeddings[mat_idx] += atom_embeddings[i]
            counts[mat_idx] += 1
        material_embeddings /= counts[:, np.newaxis] + 1e-8

    elif method == "sum":
        material_embeddings = np.zeros((n_materials, embed_dim))
        for i, mat_idx in enumerate(atom_to_material):
            material_embeddings[mat_idx] += atom_embeddings[i]

    elif method == "max":
        material_embeddings = np.full((n_materials, embed_dim), -np.inf)
        for i, mat_idx in enumerate(atom_to_material):
            material_embeddings[mat_idx] = np.maximum(
                material_embeddings[mat_idx], atom_embeddings[i]
            )

    elif method == "min":
        material_embeddings = np.full((n_materials, embed_dim), np.inf)
        for i, mat_idx in enumerate(atom_to_material):
            material_embeddings[mat_idx] = np.minimum(
                material_embeddings[mat_idx], atom_embeddings[i]
            )

    elif method == "std":
        # First compute mean
        means = np.zeros((n_materials, embed_dim))
        counts = np.zeros(n_materials)
        for i, mat_idx in enumerate(atom_to_material):
            means[mat_idx] += atom_embeddings[i]
            counts[mat_idx] += 1
        means /= counts[:, np.newaxis] + 1e-8

        # Then compute std
        material_embeddings = np.zeros((n_materials, embed_dim))
        for i, mat_idx in enumerate(atom_to_material):
            material_embeddings[mat_idx] += (atom_embeddings[i] - means[mat_idx]) ** 2
        material_embeddings = np.sqrt(material_embeddings / (counts[:, np.newaxis] + 1e-8))

    elif method == "mean_std":
        # Concatenate mean and std (doubles dimension)
        mean_emb, _ = aggregate_atom_embeddings(atom_embeddings, atom_to_material, material_ids, "mean")
        std_emb, _ = aggregate_atom_embeddings(atom_embeddings, atom_to_material, material_ids, "std")
        material_embeddings = np.concatenate([mean_emb, std_emb], axis=1)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return material_embeddings, material_ids


def aggregate_from_structure_list(
    atom_embeddings_list: List[np.ndarray],
    material_ids: Optional[List[str]] = None,
    method: str = "mean",
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate atom embeddings when you have a list of arrays (one per structure).

    Args:
        atom_embeddings_list: List of atom embeddings, each shape (n_atoms_i, embed_dim)
        material_ids: List of material IDs corresponding to each structure.
                      If None, generates IDs as "mat_0", "mat_1", ...
        method: Aggregation method (see aggregate_atom_embeddings)

    Returns:
        materials_embeddings: Shape (n_materials, embed_dim)
        material_ids: List of material IDs (same order as embeddings)

    Example:
        >>> # List of structures with varying number of atoms
        >>> structures = [
        ...     np.random.randn(10, 128),  # Structure 0: 10 atoms
        ...     np.random.randn(25, 128),  # Structure 1: 25 atoms
        ...     np.random.randn(8, 128),   # Structure 2: 8 atoms
        ... ]
        >>> ids = ["mp-123", "mp-456", "mp-789"]
        >>> material_embeddings, material_ids = aggregate_from_structure_list(structures, ids, method="mean")
        >>> print(material_embeddings.shape)  # (3, 128)
        >>> print(material_ids)  # ["mp-123", "mp-456", "mp-789"]
    """
    n_materials = len(atom_embeddings_list)

    if n_materials == 0:
        raise ValueError("Empty list provided")

    # Handle material IDs
    if material_ids is None:
        material_ids = [f"mat_{i}" for i in range(n_materials)]
    else:
        material_ids = list(material_ids)
        if len(material_ids) != n_materials:
            raise ValueError(f"Length mismatch: {len(material_ids)} IDs for {n_materials} structures")

    # Ensure each embedding array is 2D: (n_atoms, embed_dim)
    # Handle case where single-atom structures are 1D: (embed_dim,)
    processed_list = []
    for emb in atom_embeddings_list:
        emb = np.asarray(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)  # (embed_dim,) -> (1, embed_dim)
        processed_list.append(emb)
    atom_embeddings_list = processed_list

    embed_dim = atom_embeddings_list[0].shape[1]

    # Handle mean_std specially
    if method == "mean_std":
        means = []
        stds = []
        for emb in atom_embeddings_list:
            means.append(np.mean(emb, axis=0))
            stds.append(np.std(emb, axis=0))
        return np.concatenate([np.array(means), np.array(stds)], axis=1), material_ids

    material_embeddings = np.zeros((n_materials, embed_dim))

    for i, emb in enumerate(atom_embeddings_list):
        if method == "mean":
            material_embeddings[i] = np.mean(emb, axis=0)
        elif method == "sum":
            material_embeddings[i] = np.sum(emb, axis=0)
        elif method == "max":
            material_embeddings[i] = np.max(emb, axis=0)
        elif method == "min":
            material_embeddings[i] = np.min(emb, axis=0)
        elif method == "std":
            material_embeddings[i] = np.std(emb, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return material_embeddings, material_ids


def aggregate_by_material_id(
    atom_embeddings: np.ndarray,
    atom_material_ids: List[str],
    method: str = "mean",
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate atom embeddings to material level using atom-level material IDs.

    This function takes a flat array of atom embeddings and a list of material IDs
    (one per atom) indicating which material each atom belongs to. It returns
    aggregated material-level embeddings and unique material IDs.

    Args:
        atom_embeddings: Atom-level embeddings, shape (n_atoms, embed_dim)
                         Can also be (n_atoms,) for 1D embeddings
        atom_material_ids: List of material IDs, one per atom.
                           Length must equal n_atoms.
                           Example: ["mp-1", "mp-1", "mp-1", "mp-2", "mp-2", ...]
        method: Aggregation method
                - "mean": Average pooling (recommended)
                - "sum": Sum pooling
                - "max": Element-wise max pooling
                - "min": Element-wise min pooling
                - "std": Standard deviation
                - "mean_std": Concatenate mean and std (doubles dimension)

    Returns:
        material_embeddings: Shape (n_materials, embed_dim)
        material_ids: List of unique material IDs (same order as embeddings)

    Example:
        >>> # 9 atoms belonging to 3 materials
        >>> atom_embeddings = np.random.randn(9, 128)
        >>> atom_material_ids = ["mp-1", "mp-1", "mp-1",  # 3 atoms in mp-1
        ...                      "mp-2", "mp-2",          # 2 atoms in mp-2
        ...                      "mp-3", "mp-3", "mp-3", "mp-3"]  # 4 atoms in mp-3
        >>> material_emb, material_ids = aggregate_by_material_id(atom_embeddings, atom_material_ids)
        >>> print(material_emb.shape)  # (3, 128)
        >>> print(material_ids)  # ["mp-1", "mp-2", "mp-3"]
    """
    atom_embeddings = np.asarray(atom_embeddings)

    # Handle 1D embeddings
    if atom_embeddings.ndim == 1:
        atom_embeddings = atom_embeddings.reshape(-1, 1)

    n_atoms, embed_dim = atom_embeddings.shape

    if len(atom_material_ids) != n_atoms:
        raise ValueError(f"Length mismatch: {len(atom_material_ids)} IDs for {n_atoms} atoms")

    # Get unique material IDs (preserving order of first occurrence)
    seen = {}
    unique_material_ids = []
    for mid in atom_material_ids:
        if mid not in seen:
            seen[mid] = len(unique_material_ids)
            unique_material_ids.append(mid)

    n_materials = len(unique_material_ids)

    # Create mapping from material ID to index
    material_id_to_idx = {mid: i for i, mid in enumerate(unique_material_ids)}

    # Group atoms by material
    material_atoms = {i: [] for i in range(n_materials)}
    for atom_idx, mid in enumerate(atom_material_ids):
        mat_idx = material_id_to_idx[mid]
        material_atoms[mat_idx].append(atom_idx)

    # Handle mean_std specially
    if method == "mean_std":
        means = []
        stds = []
        for mat_idx in range(n_materials):
            atom_indices = material_atoms[mat_idx]
            mat_atom_emb = atom_embeddings[atom_indices]
            means.append(np.mean(mat_atom_emb, axis=0))
            stds.append(np.std(mat_atom_emb, axis=0))
        material_embeddings = np.concatenate([np.array(means), np.array(stds)], axis=1)
        return material_embeddings, unique_material_ids

    # Aggregate each material
    material_embeddings = np.zeros((n_materials, embed_dim))

    for mat_idx in range(n_materials):
        atom_indices = material_atoms[mat_idx]
        mat_atom_emb = atom_embeddings[atom_indices]

        if method == "mean":
            material_embeddings[mat_idx] = np.mean(mat_atom_emb, axis=0)
        elif method == "sum":
            material_embeddings[mat_idx] = np.sum(mat_atom_emb, axis=0)
        elif method == "max":
            material_embeddings[mat_idx] = np.max(mat_atom_emb, axis=0)
        elif method == "min":
            material_embeddings[mat_idx] = np.min(mat_atom_emb, axis=0)
        elif method == "std":
            material_embeddings[mat_idx] = np.std(mat_atom_emb, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return material_embeddings, unique_material_ids


def aggregate_with_weights(
    atom_embeddings: np.ndarray,
    atom_to_material: np.ndarray,
    weights: np.ndarray,
    material_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Weighted aggregation of atom embeddings (e.g., by atomic mass or importance).

    Args:
        atom_embeddings: Shape (n_atoms, embed_dim)
        atom_to_material: Shape (n_atoms,)
        weights: Shape (n_atoms,) - weight for each atom
        material_ids: List of material IDs, length = n_materials.
                      If None, generates IDs as "mat_0", "mat_1", ...

    Returns:
        materials_embeddings: Shape (n_materials, embed_dim)
        material_ids: List of material IDs (same order as embeddings)

    Example:
        >>> # Weight by atomic mass
        >>> atomic_masses = np.array([12.0, 16.0, 1.0, 1.0, ...])  # C, O, H, H, ...
        >>> material_ids = ["mp-123", "mp-456", ...]
        >>> material_emb, ids = aggregate_with_weights(atom_emb, atom_to_mat, atomic_masses, material_ids)
    """
    atom_embeddings = np.asarray(atom_embeddings)
    atom_to_material = np.asarray(atom_to_material)
    weights = np.asarray(weights)

    n_materials = atom_to_material.max() + 1
    embed_dim = atom_embeddings.shape[1]

    # Handle material IDs
    if material_ids is None:
        material_ids = [f"mat_{i}" for i in range(n_materials)]
    else:
        material_ids = list(material_ids)
        if len(material_ids) != n_materials:
            raise ValueError(f"Length mismatch: {len(material_ids)} IDs for {n_materials} materials")

    material_embeddings = np.zeros((n_materials, embed_dim))
    weight_sums = np.zeros(n_materials)

    for i, mat_idx in enumerate(atom_to_material):
        material_embeddings[mat_idx] += weights[i] * atom_embeddings[i]
        weight_sums[mat_idx] += weights[i]

    material_embeddings /= weight_sums[:, np.newaxis] + 1e-8

    return material_embeddings, material_ids


def example_aggregation():
    """
    Example showing how to aggregate atom-level embeddings to materials-level.
    """
    print("=" * 80)
    print("Atom → Material Embedding Aggregation Example")
    print("=" * 80)

    np.random.seed(42)

    # ----- Scenario 1: Flat array with atom-to-material mapping -----
    print("\n--- Scenario 1: Flat array with mapping ---")

    n_atoms = 1000
    n_materials = 50
    embed_dim = 128

    # Simulate atom embeddings (e.g., from MLIP)
    atom_embeddings = np.random.randn(n_atoms, embed_dim)

    # Mapping: which material each atom belongs to
    # Here: 20 atoms per material
    atom_to_material = np.repeat(np.arange(n_materials), n_atoms // n_materials)

    # Material IDs (one per structure)
    material_ids = [f"mp-{i:05d}" for i in range(n_materials)]

    print(f"Atom embeddings shape: {atom_embeddings.shape}")
    print(f"Number of materials: {n_materials}")
    print(f"Material IDs (first 5): {material_ids[:5]}")

    # Aggregate using different methods
    for method in ["mean", "sum", "max", "std", "mean_std"]:
        mat_emb, ids = aggregate_atom_embeddings(atom_embeddings, atom_to_material, material_ids, method)
        print(f"  {method:10s} -> embeddings: {mat_emb.shape}, IDs: {len(ids)}")

    # ----- Scenario 2: List of structures -----
    print("\n--- Scenario 2: List of structures (varying sizes) ---")

    # Structures with different numbers of atoms
    structures = [
        np.random.randn(10, embed_dim),   # 10 atoms
        np.random.randn(25, embed_dim),   # 25 atoms
        np.random.randn(8, embed_dim),    # 8 atoms
        np.random.randn(42, embed_dim),   # 42 atoms
        np.random.randn(15, embed_dim),   # 15 atoms
    ]

    # Material IDs for each structure
    structure_ids = ["mp-100", "mp-200", "mp-300", "mp-400", "mp-500"]

    print(f"Number of structures: {len(structures)}")
    print(f"Atoms per structure: {[s.shape[0] for s in structures]}")
    print(f"Structure IDs: {structure_ids}")

    mat_emb, ids = aggregate_from_structure_list(structures, structure_ids, method="mean")
    print(f"Material embeddings shape: {mat_emb.shape}")
    print(f"Returned IDs: {ids}")

    # ----- Scenario 3: Weighted aggregation -----
    print("\n--- Scenario 3: Weighted by atomic mass ---")

    # Simulate atomic masses
    atomic_masses = np.random.uniform(1, 200, size=n_atoms)

    mat_emb_weighted, ids = aggregate_with_weights(
        atom_embeddings, atom_to_material, atomic_masses, material_ids
    )
    print(f"Weighted material embeddings shape: {mat_emb_weighted.shape}")
    print(f"IDs (first 5): {ids[:5]}")

    print("\nDone!")

    return mat_emb, ids


def example_with_materials_embeddings():
    """
    Practical example showing how to use ManifoldFishAnalyzer with
    real materials-level embeddings.

    This example assumes you have:
    1. Reference embeddings from known stable materials (e.g., MP-20 dataset)
    2. Generated embeddings from your generative model
    3. Material IDs for both sets

    The embeddings should be at the MATERIALS level (one embedding per structure),
    not atom level. If you have atom-level embeddings, aggregate them first
    (e.g., mean pooling) before using this analyzer.
    """

    # =========================================================================
    # STEP 1: Load your materials-level embeddings
    # =========================================================================

    # Option A: Load from numpy files
    # reference_embeddings = np.load("reference_materials_embeddings.npy")  # Shape: (n_ref, dim)
    # generated_embeddings = np.load("generated_materials_embeddings.npy")  # Shape: (n_gen, dim)

    # Option B: Load from your materials analyzer
    # from your_materials_analyzer import MaterialsAnalyzer
    # analyzer = MaterialsAnalyzer(model_path="your_mlip_model")
    # reference_embeddings = analyzer.get_material_embeddings(reference_structures)
    # generated_embeddings = analyzer.get_material_embeddings(generated_structures)

    # Option C: Use PCA-reduced embeddings
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=3)
    # reference_embeddings = pca.fit_transform(high_dim_ref_embeddings)
    # generated_embeddings = pca.transform(high_dim_gen_embeddings)

    # For this example, we'll create mock data
    np.random.seed(42)

    # Simulate MP-20 style reference data (45K structures)
    n_ref = 45000
    dim = 3  # Using 3D PCA embeddings

    print("=" * 80)
    print("Materials-Level Embedding Analysis Example")
    print("=" * 80)

    print(f"\nSimulating {n_ref} reference materials (like MP-20)...")
    reference_embeddings = np.random.randn(n_ref, dim) * 0.5

    # Reference IDs (e.g., Materials Project IDs)
    reference_ids = [f"mp-{i}" for i in range(n_ref)]

    # Simulate generated structures from a generative model
    n_gen = 5000
    print(f"Simulating {n_gen} generated materials...")

    generated_embeddings = np.random.randn(n_gen, dim) * 0.6 + np.array([0.2, 0.1, 0.05])
    generated_ids = [f"gen-{i:05d}" for i in range(n_gen)]

    print(f"\nReference embeddings shape: {reference_embeddings.shape}")
    print(f"Generated embeddings shape: {generated_embeddings.shape}")

    # =========================================================================
    # STEP 2: Initialize the analyzer
    # =========================================================================

    print("\n" + "-" * 80)
    print("Initializing ManifoldFishAnalyzer...")
    print("-" * 80)

    analyzer = ManifoldFishAnalyzer(
        reference_embeddings=reference_embeddings,
        reference_ids=reference_ids,

        # Neighbor parameters
        n_neighbors=15,              # More neighbors for large datasets
        n_neighbors_local_pca=30,    # More for better tangent estimation

        # Boundary method: "auto" picks best for dimension
        boundary_method="auto",

        # Local geometry mode:
        # - "fast": Lazy caching, recommended for large n_ref (default)
        # - "skip": Disable geometry check for fastest analysis
        # - "full": Most accurate but slow
        local_geometry_mode="fast",

        # Curvature adjustment for curved manifolds (e.g., U-shaped / horseshoe)
        # High-curvature regions get more tolerance for local PCA residual
        curvature_adjusted=True,

        # Thresholds (tune based on your data)
        stability_threshold=0.5,     # Manifold distance threshold
        geometry_threshold=0.3,      # Local PCA residual threshold
        sparse_threshold=20.0,       # Below 20th percentile = sparse
        edge_margin=0.1,             # Boundary margin
    )

    print(f"Boundary method: {analyzer.boundary_method}")
    print(f"Local geometry mode: {analyzer.local_geometry_mode}")
    print(f"Curvature adjustment: {'Enabled' if analyzer.curvature_adjusted else 'Disabled'}")

    # =========================================================================
    # STEP 3: Analyze generated structures
    # =========================================================================

    print("\n" + "-" * 80)
    print("Analyzing generated structures...")
    print("-" * 80)

    results = analyzer.analyze(
        generated_embeddings=generated_embeddings,
        material_ids=generated_ids,
    )

    # =========================================================================
    # STEP 4: View results
    # =========================================================================

    # Print summary
    analyzer.print_summary(results)

    # Get IDs by category
    ids_by_category = analyzer.get_all_ids_by_category(results)

    print("\n" + "=" * 80)
    print("Material IDs by Category")
    print("=" * 80)

    for category, ids in ids_by_category.items():
        print(f"\n{category}: {len(ids)} structures")
        if ids:
            print(f"  First 5: {ids[:5]}")

    # =========================================================================
    # STEP 5: Get priority candidates for DFT validation
    # =========================================================================

    print("\n" + "=" * 80)
    print("Priority Candidates for DFT Validation")
    print("=" * 80)

    # Frontier fish: Inside manifold but in sparse regions (low-risk novel)
    frontier_ids = analyzer.get_ids_by_category(results, "frontier_fish")
    print(f"\n1. FRONTIER FISH ({len(frontier_ids)} structures)")
    print("   These are inside the stable manifold but in under-explored regions.")
    print("   Low risk, good candidates for expanding known materials space.")

    # Adventurous fish: Slightly outside manifold (medium-high risk novel)
    adventurous_ids = analyzer.get_ids_by_category(results, "adventurous_fish")
    print(f"\n2. ADVENTUROUS FISH ({len(adventurous_ids)} structures)")
    print("   These are slightly outside the known manifold.")
    print("   Higher risk but potentially more novel discoveries.")

    # Edge fish: At manifold boundary (medium risk)
    edge_ids = analyzer.get_ids_by_category(results, "edge_fish")
    print(f"\n3. EDGE FISH ({len(edge_ids)} structures)")
    print("   These are at the boundary of known materials space.")
    print("   Medium risk, worth validating.")

    # Get top candidates with detailed metrics
    print("\n" + "-" * 80)
    print("Top 10 Frontier Fish (sorted by confidence)")
    print("-" * 80)

    top_frontier = analyzer.get_top_candidates(results, "frontier_fish", top_k=10)
    for i, r in enumerate(top_frontier, 1):
        print(f"{i:2d}. {r.material_id:12s} | "
              f"dist={r.manifold_distance:+.3f} | "
              f"density={r.density_percentile:5.1f}% | "
              f"residual={r.local_pca_residual:.3f} | "
              f"conf={r.confidence:.2f}")

    # =========================================================================
    # STEP 6: Export results
    # =========================================================================

    print("\n" + "=" * 80)
    print("Exporting Results")
    print("=" * 80)

    # Export full results to CSV
    analyzer.export_results(results, "materials_analysis_results.csv")

    # Export just the IDs for each category (useful for downstream processing)
    import os
    os.makedirs("category_ids", exist_ok=True)

    for category, ids in ids_by_category.items():
        if ids:
            filepath = f"category_ids/{category}_ids.txt"
            with open(filepath, 'w') as f:
                for mid in ids:
                    f.write(f"{mid}\n")
            print(f"Saved {len(ids)} IDs to {filepath}")

    # =========================================================================
    # STEP 7: Visualization (optional)
    # =========================================================================

    # Uncomment to generate plots:
    # analyzer.plot_distribution(results, save_path="materials_distribution.png")
    # analyzer.plot_3d_manifold(results, generated_embeddings, save_path="materials_3d.png")

    return analyzer, results


def quick_analysis_example():
    """
    Minimal example for quick analysis.

    Copy this template and modify for your use case.
    """
    import numpy as np
    from manifold_fish_analysis import ManifoldFishAnalyzer

    # Load your data
    ref_embeddings = np.load("your_reference_embeddings.npy")
    gen_embeddings = np.load("your_generated_embeddings.npy")
    ref_ids = [f"ref-{i}" for i in range(len(ref_embeddings))]
    gen_ids = [f"gen-{i}" for i in range(len(gen_embeddings))]

    # Analyze
    analyzer = ManifoldFishAnalyzer(
        reference_embeddings=ref_embeddings,
        reference_ids=ref_ids,
        local_geometry_mode="fast",  # Use "skip" for very large datasets
    )

    results = analyzer.analyze(gen_embeddings, gen_ids)

    # Get results
    analyzer.print_summary(results)
    analyzer.export_results(results, "results.csv")

    # Get priority candidates
    frontier_ids = analyzer.get_ids_by_category(results, "frontier_fish")
    adventurous_ids = analyzer.get_ids_by_category(results, "adventurous_fish")

    print(f"\nPriority for DFT: {len(frontier_ids)} frontier + {len(adventurous_ids)} adventurous")

    return results


if __name__ == "__main__":
    # Run the synthetic example
    # analyzer, results = example_usage()

    # Or run the materials example
    analyzer, results = example_with_materials_embeddings()
