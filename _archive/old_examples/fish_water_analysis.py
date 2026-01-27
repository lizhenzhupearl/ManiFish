"""
Fish-Water Analysis for Generated Structures

The Four Scenarios:
1. Fish in Water    - Stable & similar to known (low distance, low novelty)
2. Novel Fish       - Stable & different from known (low distance, HIGH novelty)
3. Fish Jumping Out - Unstable, outside manifold (HIGH distance)
4. Not a Fish       - Invalid/noise structures (extreme outliers)

Usage:
    analyzer = FishWaterAnalyzer(reference_embeddings, reference_ids)
    results = analyzer.analyze(generated_embeddings, generated_ids)
    analyzer.plot_distribution(results)

    # Get material IDs by category
    novel_ids = analyzer.get_ids_by_category(results, "novel_fish")

    # Export to CSV
    analyzer.export_results(results, "fish_water_results.csv")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
import matplotlib.pyplot as plt
import pickle


@dataclass
class FishWaterResult:
    """Results for a single structure with material ID tracking."""
    index: int
    material_id: str              # Material ID for tracking
    manifold_distance: float      # Distance to nearest reference points
    novelty_score: float          # How different from neighbors
    lof_score: float              # Local outlier factor
    density_score: float          # Local density estimate
    category: str                 # Fish classification
    confidence: float             # Confidence in classification
    nearest_reference_ids: List[str] = None  # IDs of nearest reference structures


@dataclass
class CategoryResults:
    """Results grouped by category with material IDs."""
    category: str
    count: int
    percentage: float
    material_ids: List[str]
    embeddings: np.ndarray
    results: List[FishWaterResult]


class FishWaterAnalyzer:
    """
    Analyze generated structures using the Fish-Water framework.
    Tracks material IDs for traceability.

    Reference embeddings define the "water" (known stable manifold).
    Generated embeddings are the "fish" candidates to classify.

    Example:
        >>> analyzer = FishWaterAnalyzer(
        ...     reference_embeddings=pca_embeddings_3d,
        ...     reference_ids=reference_material_ids
        ... )
        >>> results = analyzer.analyze(
        ...     generated_embeddings=pca_embeddings_3d_gen,
        ...     material_ids=generated_material_ids
        ... )
        >>> analyzer.print_summary(results)
        >>>
        >>> # Get IDs by category
        >>> novel_ids = analyzer.get_ids_by_category(results, "novel_fish")
        >>> print(f"Novel fish IDs: {novel_ids}")
        >>>
        >>> # Export results
        >>> analyzer.export_results(results, "results.csv")
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        reference_ids: Optional[List[str]] = None,
        n_neighbors: int = 10,
        stability_threshold: float = 0.5,
        novelty_threshold: float = 0.3,
        outlier_threshold: float = -1.5,
    ):
        """
        Initialize analyzer with reference embeddings (the "water").

        Args:
            reference_embeddings: Known stable structures embeddings
                                  Shape: (n_reference, embedding_dim)
            reference_ids: Material IDs for reference structures
            n_neighbors: Number of neighbors for distance/novelty calculation
            stability_threshold: Max distance to be considered "in water"
            novelty_threshold: Min novelty to be considered "novel"
            outlier_threshold: LOF score below this = "not a fish"
        """
        self.reference = np.asarray(reference_embeddings)
        self.n_reference = len(self.reference)

        # Store reference IDs
        if reference_ids is not None:
            self.reference_ids = list(reference_ids)
        else:
            self.reference_ids = [f"ref_{i}" for i in range(self.n_reference)]

        self.n_neighbors = min(n_neighbors, self.n_reference - 1)
        self.stability_threshold = stability_threshold
        self.novelty_threshold = novelty_threshold
        self.outlier_threshold = outlier_threshold

        # Fit models on reference data
        self._fit_reference_models()

    def _fit_reference_models(self):
        """Fit neighbor and outlier models on reference data."""
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='auto',
            metric='euclidean'
        )
        self.knn.fit(self.reference)

        ref_distances, _ = self.knn.kneighbors(self.reference)
        self.ref_mean_dist = np.mean(ref_distances[:, -1])
        self.ref_std_dist = np.std(ref_distances[:, -1])

        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,
            contamination=0.1
        )
        self.lof.fit(self.reference)

    def analyze(
        self,
        generated_embeddings: np.ndarray,
        material_ids: Optional[List[str]] = None,
    ) -> List[FishWaterResult]:
        """
        Analyze generated structures and classify into fish-water categories.

        Args:
            generated_embeddings: Generated structures embeddings
                                  Shape: (n_generated, embedding_dim)
            material_ids: Material IDs for generated structures (REQUIRED for tracking)

        Returns:
            List of FishWaterResult for each generated structure
        """
        generated = np.asarray(generated_embeddings)
        n_generated = len(generated)

        # Handle material IDs
        if material_ids is not None:
            mat_ids = list(material_ids)
        else:
            mat_ids = [f"gen_{i}" for i in range(n_generated)]

        # 1. Compute distances to reference manifold
        distances, neighbor_indices = self.knn.kneighbors(generated)

        # Mean distance to k nearest neighbors (normalized)
        mean_distances = np.mean(distances, axis=1)
        normalized_distances = (mean_distances - self.ref_mean_dist) / (self.ref_std_dist + 1e-8)

        # 2. Compute novelty scores
        novelty_scores = self._compute_novelty(generated, distances, neighbor_indices)

        # 3. Compute LOF scores
        lof_scores = self.lof.score_samples(generated)

        # 4. Compute local density
        density_scores = self._compute_density(distances)

        # 5. Classify each structure
        results = []
        for i in range(n_generated):
            category, confidence = self._classify(
                manifold_distance=normalized_distances[i],
                novelty=novelty_scores[i],
                lof=lof_scores[i],
                density=density_scores[i],
            )

            # Get IDs of nearest reference structures
            nearest_ref_ids = [self.reference_ids[idx] for idx in neighbor_indices[i]]

            result = FishWaterResult(
                index=i,
                material_id=mat_ids[i],
                manifold_distance=normalized_distances[i],
                novelty_score=novelty_scores[i],
                lof_score=lof_scores[i],
                density_score=density_scores[i],
                category=category,
                confidence=confidence,
                nearest_reference_ids=nearest_ref_ids,
            )
            results.append(result)

        # Store for later use
        self._last_results = results
        self._last_embeddings = generated
        self._last_ids = mat_ids

        return results

    def _compute_novelty(
        self,
        generated: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Compute novelty score based on distance ratio."""
        ratio = distances[:, -1] / (distances[:, 0] + 1e-8)

        novelty_scores = []
        for i, idx in enumerate(indices):
            neighbor_mean = np.mean(self.reference[idx], axis=0)
            diff = np.linalg.norm(generated[i] - neighbor_mean)
            neighbor_spread = np.std(self.reference[idx])
            novelty = diff / (neighbor_spread + 1e-8)
            novelty_scores.append(novelty)

        novelty_scores = np.array(novelty_scores)
        combined = 0.5 * (ratio / np.max(ratio + 1e-8)) + 0.5 * (novelty_scores / np.max(novelty_scores + 1e-8))

        return combined

    def _compute_density(self, distances: np.ndarray) -> np.ndarray:
        """Compute local density estimate."""
        mean_dist = np.mean(distances, axis=1)
        density = 1.0 / (mean_dist + 1e-8)
        density = (density - density.min()) / (density.max() - density.min() + 1e-8)
        return density

    def _classify(
        self,
        manifold_distance: float,
        novelty: float,
        lof: float,
        density: float,
    ) -> Tuple[str, float]:
        """Classify structure into fish-water category."""
        if lof < self.outlier_threshold:
            confidence = min(1.0, abs(lof - self.outlier_threshold) / 2)
            return "not_a_fish", confidence

        in_water = manifold_distance < self.stability_threshold

        if in_water:
            if novelty > self.novelty_threshold:
                confidence = min(1.0, (novelty - self.novelty_threshold) / 0.5 + 0.5)
                return "novel_fish", confidence
            else:
                confidence = min(1.0, (self.stability_threshold - manifold_distance) / self.stability_threshold + 0.3)
                return "fish_in_water", confidence
        else:
            confidence = min(1.0, (manifold_distance - self.stability_threshold) / 2 + 0.3)
            return "fish_jumping_out", confidence

    # =========================================================================
    # ID Tracking Methods
    # =========================================================================

    def get_ids_by_category(
        self,
        results: List[FishWaterResult],
        category: str,
    ) -> List[str]:
        """
        Get material IDs for a specific category.

        Args:
            results: Analysis results
            category: One of "fish_in_water", "novel_fish", "fish_jumping_out", "not_a_fish"

        Returns:
            List of material IDs in that category
        """
        return [r.material_id for r in results if r.category == category]

    def get_all_ids_by_category(
        self,
        results: List[FishWaterResult],
    ) -> Dict[str, List[str]]:
        """
        Get material IDs grouped by all categories.

        Returns:
            Dict mapping category name to list of material IDs
        """
        categories = ["fish_in_water", "novel_fish", "fish_jumping_out", "not_a_fish"]
        return {cat: self.get_ids_by_category(results, cat) for cat in categories}

    def get_category_results(
        self,
        results: List[FishWaterResult],
        generated_embeddings: np.ndarray,
    ) -> Dict[str, CategoryResults]:
        """
        Get detailed results grouped by category, including embeddings.

        Args:
            results: Analysis results
            generated_embeddings: Original embeddings array

        Returns:
            Dict mapping category name to CategoryResults
        """
        generated = np.asarray(generated_embeddings)
        categories = ["fish_in_water", "novel_fish", "fish_jumping_out", "not_a_fish"]
        total = len(results)

        category_results = {}
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            cat_indices = [r.index for r in cat_results]
            cat_ids = [r.material_id for r in cat_results]
            cat_embeddings = generated[cat_indices] if cat_indices else np.array([])

            category_results[cat] = CategoryResults(
                category=cat,
                count=len(cat_results),
                percentage=100 * len(cat_results) / total if total > 0 else 0,
                material_ids=cat_ids,
                embeddings=cat_embeddings,
                results=cat_results,
            )

        return category_results

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_results(
        self,
        results: List[FishWaterResult],
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

    def _export_csv(self, results: List[FishWaterResult], output_path: str):
        """Export results to CSV."""
        data = []
        for r in results:
            row = {
                'index': r.index,
                'material_id': r.material_id,
                'category': r.category,
                'manifold_distance': r.manifold_distance,
                'novelty_score': r.novelty_score,
                'lof_score': r.lof_score,
                'density_score': r.density_score,
                'confidence': r.confidence,
                'nearest_ref_1': r.nearest_reference_ids[0] if r.nearest_reference_ids else None,
                'nearest_ref_2': r.nearest_reference_ids[1] if r.nearest_reference_ids and len(r.nearest_reference_ids) > 1 else None,
                'nearest_ref_3': r.nearest_reference_ids[2] if r.nearest_reference_ids and len(r.nearest_reference_ids) > 2 else None,
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

    def _export_pickle(self, results: List[FishWaterResult], output_path: str):
        """Export results to pickle."""
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results exported to {output_path}")

    def export_embeddings_by_category(
        self,
        results: List[FishWaterResult],
        generated_embeddings: np.ndarray,
        output_dir: str = ".",
        prefix: str = "embeddings",
    ):
        """
        Export embeddings and IDs for each category to separate files.

        Args:
            results: Analysis results
            generated_embeddings: Original embeddings
            output_dir: Output directory
            prefix: Filename prefix
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        category_results = self.get_category_results(results, generated_embeddings)

        for cat, cat_result in category_results.items():
            if cat_result.count == 0:
                continue

            # Save embeddings
            emb_path = os.path.join(output_dir, f"{prefix}_{cat}.npy")
            np.save(emb_path, cat_result.embeddings)

            # Save IDs
            ids_path = os.path.join(output_dir, f"{prefix}_{cat}_ids.txt")
            with open(ids_path, 'w') as f:
                for mid in cat_result.material_ids:
                    f.write(f"{mid}\n")

            print(f"{cat}: saved {cat_result.count} structures to {emb_path}")

    # =========================================================================
    # Summary and Visualization
    # =========================================================================

    def get_summary(self, results: List[FishWaterResult]) -> Dict:
        """Get summary statistics of classification."""
        categories = [r.category for r in results]

        summary = {
            "total": len(results),
            "fish_in_water": categories.count("fish_in_water"),
            "novel_fish": categories.count("novel_fish"),
            "fish_jumping_out": categories.count("fish_jumping_out"),
            "not_a_fish": categories.count("not_a_fish"),
            "mean_manifold_distance": np.mean([r.manifold_distance for r in results]),
            "mean_novelty": np.mean([r.novelty_score for r in results]),
            "mean_lof": np.mean([r.lof_score for r in results]),
        }

        for cat in ["fish_in_water", "novel_fish", "fish_jumping_out", "not_a_fish"]:
            summary[f"{cat}_pct"] = 100 * summary[cat] / summary["total"] if summary["total"] > 0 else 0

        return summary

    def print_summary(self, results: List[FishWaterResult]):
        """Print formatted summary with example IDs."""
        summary = self.get_summary(results)
        ids_by_cat = self.get_all_ids_by_category(results)

        print("=" * 70)
        print("Fish-Water Analysis Summary")
        print("=" * 70)
        print(f"\nTotal structures analyzed: {summary['total']}\n")

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
                ids_str = ", ".join(str(eid) for eid in example_ids)
                if len(ids) > 3:
                    ids_str += f", ... (+{len(ids)-3} more)"
                print(f"      Example IDs: {ids_str}")

        print("\nMetric Statistics:")
        print("-" * 50)
        print(f"  Mean manifold distance: {summary['mean_manifold_distance']:>8.3f}")
        print(f"  Mean novelty score:     {summary['mean_novelty']:>8.3f}")
        print(f"  Mean LOF score:         {summary['mean_lof']:>8.3f}")

        print("\nRecommendations:")
        print("-" * 50)
        if summary['novel_fish'] > 0:
            print(f"  -> {summary['novel_fish']} novel fish candidates for DFT validation!")
        if summary['fish_jumping_out_pct'] > 30:
            print(f"  -> High unstable fraction ({summary['fish_jumping_out_pct']:.1f}%) - check generator")
        if summary['not_a_fish_pct'] > 10:
            print(f"  -> {summary['not_a_fish_pct']:.1f}% invalid structures - filter these out")

    def get_top_candidates(
        self,
        results: List[FishWaterResult],
        category: str = "novel_fish",
        top_k: int = 10,
    ) -> List[FishWaterResult]:
        """Get top candidates from a specific category, sorted by confidence."""
        filtered = [r for r in results if r.category == category]
        sorted_results = sorted(filtered, key=lambda x: x.confidence, reverse=True)
        return sorted_results[:top_k]

    def plot_distribution(
        self,
        results: List[FishWaterResult],
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
        show_ids: bool = False,
    ):
        """
        Plot distribution of fish-water categories.

        Args:
            results: Analysis results
            figsize: Figure size
            save_path: Path to save figure
            show_ids: If True, annotate points with material IDs (only for small datasets)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        distances = np.array([r.manifold_distance for r in results])
        novelties = np.array([r.novelty_score for r in results])
        categories = np.array([r.category for r in results])
        mat_ids = np.array([r.material_id for r in results])

        colors = {
            "fish_in_water": "#2ecc71",
            "novel_fish": "#3498db",
            "fish_jumping_out": "#e74c3c",
            "not_a_fish": "#95a5a6",
        }

        labels = {
            "fish_in_water": "Fish in Water\n(stable, known)",
            "novel_fish": "Novel Fish\n(stable, new)",
            "fish_jumping_out": "Fish Jumping Out\n(unstable)",
            "not_a_fish": "Not a Fish\n(invalid)",
        }

        # 1. Pie chart
        ax1 = axes[0, 0]
        summary = self.get_summary(results)
        sizes = [summary[cat] for cat in colors.keys()]
        pie_colors = [colors[cat] for cat in colors.keys()]
        pie_labels = [labels[cat] for cat in colors.keys()]

        wedges, texts, autotexts = ax1.pie(
            sizes, labels=pie_labels, colors=pie_colors,
            autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
            startangle=90
        )
        ax1.set_title("Category Distribution", fontsize=12, fontweight='bold')

        # 2. Scatter plot
        ax2 = axes[0, 1]
        for cat in colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax2.scatter(
                    distances[mask], novelties[mask],
                    c=colors[cat], label=cat.replace("_", " ").title(),
                    alpha=0.6, s=20
                )

                # Optionally show IDs
                if show_ids and np.sum(mask) < 50:
                    for d, n, mid in zip(distances[mask], novelties[mask], mat_ids[mask]):
                        ax2.annotate(mid, (d, n), fontsize=6, alpha=0.7)

        ax2.axvline(x=self.stability_threshold, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=self.novelty_threshold, color='black', linestyle=':', alpha=0.5)
        ax2.set_xlabel("Manifold Distance (normalized)", fontsize=10)
        ax2.set_ylabel("Novelty Score", fontsize=10)
        ax2.set_title("Fish-Water Space", fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)

        # 3. Histogram of manifold distances
        ax3 = axes[1, 0]
        for cat in colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax3.hist(distances[mask], bins=30, alpha=0.5, color=colors[cat],
                        label=cat.replace("_", " ").title())
        ax3.axvline(x=self.stability_threshold, color='black', linestyle='--')
        ax3.set_xlabel("Manifold Distance", fontsize=10)
        ax3.set_ylabel("Count", fontsize=10)
        ax3.set_title("Distribution of Manifold Distance", fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)

        # 4. Histogram of novelty scores
        ax4 = axes[1, 1]
        for cat in colors.keys():
            mask = categories == cat
            if np.sum(mask) > 0:
                ax4.hist(novelties[mask], bins=30, alpha=0.5, color=colors[cat],
                        label=cat.replace("_", " ").title())
        ax4.axvline(x=self.novelty_threshold, color='black', linestyle='--')
        ax4.set_xlabel("Novelty Score", fontsize=10)
        ax4.set_ylabel("Count", fontsize=10)
        ax4.set_title("Distribution of Novelty Score", fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig


# =============================================================================
# Example Usage with Your Data
# =============================================================================

def example_with_your_pca_data():
    """
    Example using your PCA-preprocessed embeddings.

    Your variables:
        - pca_embeddings_3d: Reference embeddings (known stable, the "water")
        - pca_embeddings_3d_gen: Generated embeddings (fish candidates)
        - reference_ids: Material IDs for reference (e.g., from MP-20)
        - generated_ids: Material IDs for generated structures
    """

    # =========================================================================
    # REPLACE THESE WITH YOUR ACTUAL DATA
    # =========================================================================

    # Example: Load your data
    # pca_embeddings_3d = np.load('reference_pca_embeddings.npy')
    # pca_embeddings_3d_gen = np.load('generated_pca_embeddings.npy')
    # reference_ids = ['mp-1234', 'mp-5678', ...]  # Your reference material IDs
    # generated_ids = ['gen-001', 'gen-002', ...]  # Your generated structure IDs

    # For demo, create mock data
    np.random.seed(42)
    n_ref, n_gen = 1000, 300
    dim = 3  # PCA 3D

    pca_embeddings_3d = np.random.randn(n_ref, dim) * 0.5
    reference_ids = [f"mp-{i:05d}" for i in range(n_ref)]

    # Generated: mix of categories
    gen_in_water = np.random.randn(120, dim) * 0.5
    gen_novel = np.random.randn(60, dim) * 0.5 + np.array([1.0, 0.5, 0.3])
    gen_jumping = np.random.randn(90, dim) * 0.8 + np.array([2.5, 1.5, 1.0])
    gen_invalid = np.random.randn(30, dim) * 2.0 + np.array([5.0, 3.0, 2.0])

    pca_embeddings_3d_gen = np.vstack([gen_in_water, gen_novel, gen_jumping, gen_invalid])
    generated_ids = [f"gen-{i:04d}" for i in range(n_gen)]

    # Shuffle
    shuffle_idx = np.random.permutation(n_gen)
    pca_embeddings_3d_gen = pca_embeddings_3d_gen[shuffle_idx]
    generated_ids = [generated_ids[i] for i in shuffle_idx]

    # =========================================================================
    # Run Analysis
    # =========================================================================

    print("=" * 70)
    print("Fish-Water Analysis with Material ID Tracking")
    print("=" * 70)

    print(f"\nReference embeddings: {pca_embeddings_3d.shape}")
    print(f"Generated embeddings: {pca_embeddings_3d_gen.shape}")

    # Initialize analyzer with reference data
    analyzer = FishWaterAnalyzer(
        reference_embeddings=pca_embeddings_3d,
        reference_ids=reference_ids,
        n_neighbors=10,
        stability_threshold=0.5,
        novelty_threshold=0.3,
    )

    # Analyze generated structures
    results = analyzer.analyze(
        generated_embeddings=pca_embeddings_3d_gen,
        material_ids=generated_ids,
    )

    # Print summary
    analyzer.print_summary(results)

    # =========================================================================
    # Get IDs by Category
    # =========================================================================

    print("\n" + "=" * 70)
    print("Material IDs by Category")
    print("=" * 70)

    ids_by_category = analyzer.get_all_ids_by_category(results)

    for cat, ids in ids_by_category.items():
        print(f"\n{cat.upper()} ({len(ids)} structures):")
        if ids:
            print(f"  {ids[:10]}")  # Show first 10
            if len(ids) > 10:
                print(f"  ... and {len(ids) - 10} more")

    # =========================================================================
    # Get Top Novel Fish Candidates
    # =========================================================================

    print("\n" + "=" * 70)
    print("Top 10 Novel Fish Candidates for DFT")
    print("=" * 70)

    top_novel = analyzer.get_top_candidates(results, category="novel_fish", top_k=10)

    for i, r in enumerate(top_novel, 1):
        print(f"{i}. {r.material_id}")
        print(f"   Distance: {r.manifold_distance:.3f}, Novelty: {r.novelty_score:.3f}")
        print(f"   Confidence: {r.confidence:.3f}")
        print(f"   Nearest references: {r.nearest_reference_ids[:3]}")

    # =========================================================================
    # Export Results
    # =========================================================================

    print("\n" + "=" * 70)
    print("Exporting Results")
    print("=" * 70)

    # Export to CSV
    analyzer.export_results(results, "fish_water_results.csv")

    # Export embeddings by category
    analyzer.export_embeddings_by_category(
        results,
        pca_embeddings_3d_gen,
        output_dir="category_embeddings",
        prefix="pca_3d"
    )

    # Plot
    print("\nGenerating plots...")
    analyzer.plot_distribution(results, save_path="fish_water_analysis.png")

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = example_with_your_pca_data()
