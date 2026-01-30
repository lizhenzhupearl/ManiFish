"""
Ensemble Analysis for Multi-MLIP Manifold Voting

Combine fish categorization results from multiple MLIP manifolds to make
robust predictions about material stability and novelty.

Physical/Chemical Interpretation:
---------------------------------
Each MLIP learns a different representation of chemical space based on:
- Training data (MP, OMAT, Alexandria, etc.)
- Architecture (MACE, ORB, SevenNet, NequIP)
- Layer depth (early vs late layers)

By using anchor-based projection, all manifolds are in a unified coordinate
system (relative distances to anchors), making weighted combination meaningful.

Metrics Interpretation:
-----------------------
- manifold_distance: Structural similarity to known stable materials
  → Lower = more similar to training data = higher confidence in predictions
  → Physical: Reflects if local atomic environments are "normal"

- boundary_distance: Distance to the edge of known chemistry
  → Positive = inside known space, Negative = outside
  → Physical: How far the material is from explored composition space

- local_density: How common this type of structure is in reference set
  → High density = well-explored chemistry
  → Low density = under-explored, potentially novel

- density_percentile: Relative density compared to reference
  → Physical: Relates to how "typical" the structure is

- local_pca_residual: Geometric consistency of local environments
  → High = local geometry doesn't match neighborhood patterns
  → Physical: May indicate strained/unstable local coordinations

Physical Quantities to Correlate:
---------------------------------
1. e_Hull (energy above hull): Thermodynamic stability
2. Band gap: Electronic properties
3. Formation energy: Thermodynamic favorability
4. Phonon stability: Dynamical stability (imaginary frequencies)
5. Elastic constants: Mechanical stability
6. Coordination numbers: Local chemistry validation
7. Bond lengths/angles: Geometric validity
8. Oxidation states: Chemical reasonableness
9. Space group: Symmetry and order

Usage:
    from ensemble_analysis import EnsembleAnalyzer

    analyzer = EnsembleAnalyzer()
    analyzer.add_manifold_results("mace", mace_results, mace_csv_path)
    analyzer.add_manifold_results("orb", orb_results, orb_csv_path)
    analyzer.add_manifold_results("sevennet", sevennet_results, sevennet_csv_path)

    ensemble_results = analyzer.compute_ensemble(voting_method="weighted_average")
    analyzer.export_combined_results("ensemble_results.csv")
    analyzer.plot_comparison()
"""

import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal, Any

import numpy as np
import pandas as pd


# =============================================================================
# Category Definitions (from manifold_fish_analysis)
# =============================================================================

CATEGORIES = [
    "redundant_fish",
    "fish_in_water",
    "frontier_fish",
    "edge_fish",
    "adventurous_fish",
    "structural_hallucination",
]

# Risk scores for voting (lower = safer)
# Note: frontier_fish and adventurous_fish have variable risk based on geometry
CATEGORY_RISK_SCORES = {
    "redundant_fish": 0.0,
    "fish_in_water": 0.2,
    "frontier_fish": 0.4,  # Base; actual risk varies (low or high) based on geometry
    "edge_fish": 0.5,
    "adventurous_fish": 0.6,  # Base; actual risk varies based on geometry and LOF
    "structural_hallucination": 1.0,  # Bad geometry + LOF outlier
}

# Stability likelihood (for weighted averaging)
CATEGORY_STABILITY = {
    "redundant_fish": 0.95,
    "fish_in_water": 0.85,
    "frontier_fish": 0.70,  # Varies based on geometry
    "edge_fish": 0.50,
    "adventurous_fish": 0.35,  # Varies based on geometry
    "structural_hallucination": 0.05,  # No support
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ManifoldResult:
    """Results from a single MLIP manifold."""
    name: str  # e.g., "mace_mpa", "orb_v3", "sevennet_omat"
    model_type: str  # e.g., "mace", "orb", "sevennet"
    layer_name: str  # Which layer embeddings came from

    # Per-material results
    material_ids: List[str]
    categories: List[str]
    confidences: np.ndarray

    # Metrics arrays (n_materials,)
    manifold_distance: np.ndarray
    boundary_distance: np.ndarray
    depth_score: np.ndarray
    local_density: np.ndarray
    density_percentile: np.ndarray
    local_pca_residual: np.ndarray

    # Optional: raw CSV path
    csv_path: Optional[str] = None

    # Model quality weight (for ensemble)
    weight: float = 1.0


@dataclass
class EnsembleResult:
    """Combined results from ensemble voting."""
    material_ids: List[str]

    # Ensemble category and confidence
    ensemble_category: List[str]
    ensemble_confidence: np.ndarray
    ensemble_risk_score: np.ndarray
    ensemble_stability_score: np.ndarray

    # Agreement metrics
    category_agreement: np.ndarray  # Fraction of manifolds agreeing
    n_manifolds_agreeing: np.ndarray

    # Aggregated metrics (weighted average across manifolds)
    avg_manifold_distance: np.ndarray
    avg_boundary_distance: np.ndarray
    avg_local_pca_residual: np.ndarray
    avg_density_percentile: np.ndarray

    # Metric spreads (disagreement indicators)
    std_manifold_distance: np.ndarray
    std_boundary_distance: np.ndarray

    # Per-manifold categories for analysis
    manifold_categories: Dict[str, List[str]]  # manifold_name -> categories

    # Physical properties (added later)
    physical_properties: Dict[str, np.ndarray] = field(default_factory=dict)


# =============================================================================
# Ensemble Analyzer
# =============================================================================

class EnsembleAnalyzer:
    """
    Combine and analyze fish categorization from multiple MLIP manifolds.

    Provides various voting/aggregation strategies:
    1. Majority vote: Simple majority across manifolds
    2. Weighted vote: Weight by model quality/reliability
    3. Risk-weighted: Average risk scores, categorize by threshold
    4. Stability-weighted: Average stability likelihood
    5. Conservative: Take highest-risk category
    6. Optimistic: Take lowest-risk category
    """

    def __init__(self, default_weight: float = 1.0):
        """
        Initialize ensemble analyzer.

        Args:
            default_weight: Default weight for manifolds without explicit weight
        """
        self.manifold_results: Dict[str, ManifoldResult] = {}
        self.default_weight = default_weight
        self.ensemble_result: Optional[EnsembleResult] = None

    def add_manifold_results(
        self,
        name: str,
        results_df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
        model_type: str = "",
        layer_name: str = "",
        weight: float = None,
    ):
        """
        Add results from a manifold analysis.

        Args:
            name: Unique name for this manifold (e.g., "mace_mpa")
            results_df: DataFrame with analysis results
            csv_path: Path to CSV file (alternative to results_df)
            model_type: Model type (e.g., "mace")
            layer_name: Layer embeddings came from
            weight: Weight for ensemble voting (default: 1.0)
        """
        if results_df is None and csv_path is None:
            raise ValueError("Must provide either results_df or csv_path")

        if results_df is None:
            results_df = pd.read_csv(csv_path)

        weight = weight if weight is not None else self.default_weight

        self.manifold_results[name] = ManifoldResult(
            name=name,
            model_type=model_type,
            layer_name=layer_name,
            material_ids=results_df['material_id'].tolist(),
            categories=results_df['category'].tolist(),
            confidences=results_df['confidence'].values,
            manifold_distance=results_df['manifold_distance'].values,
            boundary_distance=results_df['boundary_distance'].values,
            depth_score=results_df['depth_score'].values,
            local_density=results_df['local_density'].values,
            density_percentile=results_df['density_percentile'].values,
            local_pca_residual=results_df['local_pca_residual'].values,
            csv_path=csv_path,
            weight=weight,
        )

        print(f"Added manifold '{name}': {len(results_df)} materials, weight={weight}")

    def add_from_analyzer_results(
        self,
        name: str,
        results: List[Any],  # List[ManifoldFishResult]
        model_type: str = "",
        layer_name: str = "",
        weight: float = None,
    ):
        """
        Add results directly from ManifoldFishAnalyzer.analyze() output.

        Args:
            name: Unique name for this manifold
            results: List of ManifoldFishResult from analyzer.analyze()
            model_type: Model type
            layer_name: Layer name
            weight: Ensemble weight
        """
        weight = weight if weight is not None else self.default_weight

        self.manifold_results[name] = ManifoldResult(
            name=name,
            model_type=model_type,
            layer_name=layer_name,
            material_ids=[r.material_id for r in results],
            categories=[r.category for r in results],
            confidences=np.array([r.confidence for r in results]),
            manifold_distance=np.array([r.manifold_distance for r in results]),
            boundary_distance=np.array([r.boundary_distance for r in results]),
            depth_score=np.array([r.depth_score for r in results]),
            local_density=np.array([r.local_density for r in results]),
            density_percentile=np.array([r.density_percentile for r in results]),
            local_pca_residual=np.array([r.local_pca_residual for r in results]),
            weight=weight,
        )

        print(f"Added manifold '{name}': {len(results)} materials, weight={weight}")

    def compute_ensemble(
        self,
        voting_method: Literal[
            "majority",
            "weighted_majority",
            "risk_weighted",
            "stability_weighted",
            "conservative",
            "optimistic"
        ] = "weighted_majority",
        risk_thresholds: Optional[Dict[str, float]] = None,
    ) -> EnsembleResult:
        """
        Compute ensemble predictions from all manifolds.

        Args:
            voting_method: How to combine predictions
                - "majority": Simple majority vote
                - "weighted_majority": Weight votes by manifold weights
                - "risk_weighted": Average risk scores, threshold to category
                - "stability_weighted": Average stability scores
                - "conservative": Take highest-risk category
                - "optimistic": Take lowest-risk category
            risk_thresholds: Custom risk thresholds for risk_weighted method

        Returns:
            EnsembleResult with combined predictions
        """
        if len(self.manifold_results) == 0:
            raise ValueError("No manifold results added. Use add_manifold_results() first.")

        # Get common material IDs (intersection across all manifolds)
        all_ids = [set(r.material_ids) for r in self.manifold_results.values()]
        common_ids = set.intersection(*all_ids)

        if len(common_ids) == 0:
            raise ValueError("No common material IDs across manifolds")

        # Sort for consistency
        material_ids = sorted(common_ids)
        n_materials = len(material_ids)
        n_manifolds = len(self.manifold_results)

        print(f"\nComputing ensemble for {n_materials} materials across {n_manifolds} manifolds")

        # Build aligned data matrices
        id_to_idx = {mid: i for i, mid in enumerate(material_ids)}

        categories_matrix = []  # (n_manifolds, n_materials)
        confidences_matrix = []
        manifold_dist_matrix = []
        boundary_dist_matrix = []
        pca_residual_matrix = []
        density_pct_matrix = []
        weights = []

        manifold_names = []
        for name, result in self.manifold_results.items():
            manifold_names.append(name)
            weights.append(result.weight)

            # Create aligned arrays
            result_id_to_idx = {mid: i for i, mid in enumerate(result.material_ids)}

            cats = []
            confs = []
            m_dist = []
            b_dist = []
            pca_res = []
            dens_pct = []

            for mid in material_ids:
                idx = result_id_to_idx[mid]
                cats.append(result.categories[idx])
                confs.append(result.confidences[idx])
                m_dist.append(result.manifold_distance[idx])
                b_dist.append(result.boundary_distance[idx])
                pca_res.append(result.local_pca_residual[idx])
                dens_pct.append(result.density_percentile[idx])

            categories_matrix.append(cats)
            confidences_matrix.append(np.array(confs))
            manifold_dist_matrix.append(np.array(m_dist))
            boundary_dist_matrix.append(np.array(b_dist))
            pca_residual_matrix.append(np.array(pca_res))
            density_pct_matrix.append(np.array(dens_pct))

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Compute ensemble based on voting method
        if voting_method == "majority":
            ensemble_cats, agreement, n_agreeing = self._majority_vote(
                categories_matrix, weights=None
            )
        elif voting_method == "weighted_majority":
            ensemble_cats, agreement, n_agreeing = self._majority_vote(
                categories_matrix, weights=weights
            )
        elif voting_method == "risk_weighted":
            ensemble_cats, agreement, n_agreeing = self._risk_weighted_vote(
                categories_matrix, weights, risk_thresholds
            )
        elif voting_method == "stability_weighted":
            ensemble_cats, agreement, n_agreeing = self._stability_weighted_vote(
                categories_matrix, weights
            )
        elif voting_method == "conservative":
            ensemble_cats, agreement, n_agreeing = self._conservative_vote(categories_matrix)
        elif voting_method == "optimistic":
            ensemble_cats, agreement, n_agreeing = self._optimistic_vote(categories_matrix)
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")

        # Compute aggregated metrics (weighted average)
        manifold_dist_matrix = np.array(manifold_dist_matrix)
        boundary_dist_matrix = np.array(boundary_dist_matrix)
        pca_residual_matrix = np.array(pca_residual_matrix)
        density_pct_matrix = np.array(density_pct_matrix)

        avg_manifold_dist = np.average(manifold_dist_matrix, axis=0, weights=weights)
        avg_boundary_dist = np.average(boundary_dist_matrix, axis=0, weights=weights)
        avg_pca_residual = np.average(pca_residual_matrix, axis=0, weights=weights)
        avg_density_pct = np.average(density_pct_matrix, axis=0, weights=weights)

        std_manifold_dist = np.std(manifold_dist_matrix, axis=0)
        std_boundary_dist = np.std(boundary_dist_matrix, axis=0)

        # Compute ensemble confidence and scores
        ensemble_confidence = np.array([
            np.average([c[i] for c in confidences_matrix], weights=weights)
            for i in range(n_materials)
        ])

        ensemble_risk = np.array([
            CATEGORY_RISK_SCORES[cat] for cat in ensemble_cats
        ])

        ensemble_stability = np.array([
            CATEGORY_STABILITY[cat] for cat in ensemble_cats
        ])

        # Build per-manifold category dict
        manifold_categories = {
            name: cats for name, cats in zip(manifold_names, categories_matrix)
        }

        self.ensemble_result = EnsembleResult(
            material_ids=material_ids,
            ensemble_category=ensemble_cats,
            ensemble_confidence=ensemble_confidence,
            ensemble_risk_score=ensemble_risk,
            ensemble_stability_score=ensemble_stability,
            category_agreement=agreement,
            n_manifolds_agreeing=n_agreeing,
            avg_manifold_distance=avg_manifold_dist,
            avg_boundary_distance=avg_boundary_dist,
            avg_local_pca_residual=avg_pca_residual,
            avg_density_percentile=avg_density_pct,
            std_manifold_distance=std_manifold_dist,
            std_boundary_distance=std_boundary_dist,
            manifold_categories=manifold_categories,
        )

        return self.ensemble_result

    def _majority_vote(
        self,
        categories_matrix: List[List[str]],
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Majority vote (optionally weighted)."""
        n_manifolds = len(categories_matrix)
        n_materials = len(categories_matrix[0])

        ensemble_cats = []
        agreement = []
        n_agreeing = []

        for i in range(n_materials):
            votes = {}
            for m, cats in enumerate(categories_matrix):
                cat = cats[i]
                w = weights[m] if weights is not None else 1.0 / n_manifolds
                votes[cat] = votes.get(cat, 0) + w

            winner = max(votes.keys(), key=lambda k: votes[k])
            ensemble_cats.append(winner)
            agreement.append(votes[winner])

            # Count manifolds agreeing with winner
            n_agree = sum(1 for cats in categories_matrix if cats[i] == winner)
            n_agreeing.append(n_agree)

        return ensemble_cats, np.array(agreement), np.array(n_agreeing)

    def _risk_weighted_vote(
        self,
        categories_matrix: List[List[str]],
        weights: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Average risk scores, threshold to category."""
        # Default thresholds
        if thresholds is None:
            thresholds = {
                "redundant_fish": 0.1,
                "fish_in_water": 0.3,
                "frontier_fish": 0.45,
                "edge_fish": 0.55,
                "adventurous_fish": 0.75,
                "structural_hallucination": 1.0,
            }

        n_materials = len(categories_matrix[0])
        ensemble_cats = []
        agreement = []
        n_agreeing = []

        for i in range(n_materials):
            # Compute weighted average risk
            risk_scores = [
                CATEGORY_RISK_SCORES[cats[i]] for cats in categories_matrix
            ]
            avg_risk = np.average(risk_scores, weights=weights)

            # Map to category by threshold
            for cat in CATEGORIES:
                if avg_risk <= thresholds[cat]:
                    ensemble_cats.append(cat)
                    break

            # Agreement with final category
            final_cat = ensemble_cats[-1]
            n_agree = sum(1 for cats in categories_matrix if cats[i] == final_cat)
            n_agreeing.append(n_agree)
            agreement.append(n_agree / len(categories_matrix))

        return ensemble_cats, np.array(agreement), np.array(n_agreeing)

    def _stability_weighted_vote(
        self,
        categories_matrix: List[List[str]],
        weights: np.ndarray,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Average stability scores, map to category."""
        n_materials = len(categories_matrix[0])
        ensemble_cats = []
        agreement = []
        n_agreeing = []

        # Stability thresholds (inverted from risk)
        stability_thresholds = [
            ("redundant_fish", 0.90),
            ("fish_in_water", 0.75),
            ("frontier_fish", 0.55),
            ("edge_fish", 0.40),
            ("adventurous_fish", 0.20),
            ("structural_hallucination", 0.0),
        ]

        for i in range(n_materials):
            stability_scores = [
                CATEGORY_STABILITY[cats[i]] for cats in categories_matrix
            ]
            avg_stability = np.average(stability_scores, weights=weights)

            # Map to category
            final_cat = "structural_hallucination"
            for cat, thresh in stability_thresholds:
                if avg_stability >= thresh:
                    final_cat = cat
                    break

            ensemble_cats.append(final_cat)

            n_agree = sum(1 for cats in categories_matrix if cats[i] == final_cat)
            n_agreeing.append(n_agree)
            agreement.append(n_agree / len(categories_matrix))

        return ensemble_cats, np.array(agreement), np.array(n_agreeing)

    def _conservative_vote(
        self,
        categories_matrix: List[List[str]],
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Take highest-risk category across manifolds."""
        n_materials = len(categories_matrix[0])
        ensemble_cats = []
        agreement = []
        n_agreeing = []

        for i in range(n_materials):
            cats = [c[i] for c in categories_matrix]
            risks = [CATEGORY_RISK_SCORES[c] for c in cats]
            max_risk_idx = np.argmax(risks)
            final_cat = cats[max_risk_idx]

            ensemble_cats.append(final_cat)

            n_agree = sum(1 for c in cats if c == final_cat)
            n_agreeing.append(n_agree)
            agreement.append(n_agree / len(categories_matrix))

        return ensemble_cats, np.array(agreement), np.array(n_agreeing)

    def _optimistic_vote(
        self,
        categories_matrix: List[List[str]],
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Take lowest-risk category across manifolds."""
        n_materials = len(categories_matrix[0])
        ensemble_cats = []
        agreement = []
        n_agreeing = []

        for i in range(n_materials):
            cats = [c[i] for c in categories_matrix]
            risks = [CATEGORY_RISK_SCORES[c] for c in cats]
            min_risk_idx = np.argmin(risks)
            final_cat = cats[min_risk_idx]

            ensemble_cats.append(final_cat)

            n_agree = sum(1 for c in cats if c == final_cat)
            n_agreeing.append(n_agree)
            agreement.append(n_agree / len(categories_matrix))

        return ensemble_cats, np.array(agreement), np.array(n_agreeing)

    def add_physical_properties(
        self,
        properties: Dict[str, Dict[str, float]],
    ):
        """
        Add physical properties for correlation analysis.

        Args:
            properties: Dict of property_name -> {material_id: value}
                Example: {"e_hull": {"mp-1": 0.02, "mp-2": 0.15, ...}}
        """
        if self.ensemble_result is None:
            raise ValueError("Must compute ensemble first")

        for prop_name, values in properties.items():
            prop_array = np.array([
                values.get(mid, np.nan)
                for mid in self.ensemble_result.material_ids
            ])
            self.ensemble_result.physical_properties[prop_name] = prop_array
            print(f"Added property '{prop_name}': {np.sum(~np.isnan(prop_array))} non-null values")

    def add_physical_property_from_df(
        self,
        df: pd.DataFrame,
        property_column: str,
        id_column: str = "material_id",
    ):
        """Add physical property from DataFrame."""
        values = dict(zip(df[id_column], df[property_column]))
        self.add_physical_properties({property_column: values})

    def export_combined_results(
        self,
        output_path: str,
        include_per_manifold: bool = True,
    ) -> pd.DataFrame:
        """
        Export combined results to CSV.

        Args:
            output_path: Output CSV path
            include_per_manifold: Include per-manifold categories in output

        Returns:
            DataFrame with all results
        """
        if self.ensemble_result is None:
            raise ValueError("Must compute ensemble first")

        result = self.ensemble_result

        data = {
            'material_id': result.material_ids,
            'ensemble_category': result.ensemble_category,
            'ensemble_confidence': result.ensemble_confidence,
            'ensemble_risk_score': result.ensemble_risk_score,
            'ensemble_stability_score': result.ensemble_stability_score,
            'category_agreement': result.category_agreement,
            'n_manifolds_agreeing': result.n_manifolds_agreeing,
            'avg_manifold_distance': result.avg_manifold_distance,
            'avg_boundary_distance': result.avg_boundary_distance,
            'avg_local_pca_residual': result.avg_local_pca_residual,
            'avg_density_percentile': result.avg_density_percentile,
            'std_manifold_distance': result.std_manifold_distance,
            'std_boundary_distance': result.std_boundary_distance,
        }

        # Add per-manifold categories
        if include_per_manifold:
            for name, cats in result.manifold_categories.items():
                data[f'category_{name}'] = cats

        # Add physical properties
        for prop_name, values in result.physical_properties.items():
            data[prop_name] = values

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Exported combined results to {output_path}")

        return df

    def get_summary(self) -> Dict:
        """Get summary statistics of ensemble results."""
        if self.ensemble_result is None:
            raise ValueError("Must compute ensemble first")

        result = self.ensemble_result
        n_total = len(result.material_ids)

        summary = {
            "n_materials": n_total,
            "n_manifolds": len(result.manifold_categories),
            "manifold_names": list(result.manifold_categories.keys()),
        }

        # Category distribution
        for cat in CATEGORIES:
            count = sum(1 for c in result.ensemble_category if c == cat)
            summary[cat] = count
            summary[f"{cat}_pct"] = 100 * count / n_total

        # Agreement statistics
        summary["mean_agreement"] = np.mean(result.category_agreement)
        summary["min_agreement"] = np.min(result.category_agreement)
        summary["full_agreement_pct"] = 100 * np.mean(
            result.n_manifolds_agreeing == len(result.manifold_categories)
        )

        # Physical property correlations
        if result.physical_properties:
            summary["physical_properties"] = {}
            for prop_name, values in result.physical_properties.items():
                valid = ~np.isnan(values)
                if np.sum(valid) > 10:
                    # Correlation with risk score
                    corr = np.corrcoef(
                        values[valid],
                        result.ensemble_risk_score[valid]
                    )[0, 1]
                    summary["physical_properties"][prop_name] = {
                        "n_valid": int(np.sum(valid)),
                        "correlation_with_risk": float(corr),
                        "mean_by_category": {},
                    }

                    # Mean by category
                    for cat in CATEGORIES:
                        mask = np.array([c == cat for c in result.ensemble_category]) & valid
                        if np.sum(mask) > 0:
                            summary["physical_properties"][prop_name]["mean_by_category"][cat] = \
                                float(np.mean(values[mask]))

        return summary

    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()

        print("=" * 80)
        print("Ensemble Analysis Summary")
        print("=" * 80)
        print(f"\nMaterials analyzed: {summary['n_materials']}")
        print(f"Manifolds combined: {summary['n_manifolds']}")
        print(f"Manifold names: {', '.join(summary['manifold_names'])}")

        print("\n" + "-" * 80)
        print("Category Distribution:")
        print("-" * 80)
        for cat in CATEGORIES:
            count = summary[cat]
            pct = summary[f"{cat}_pct"]
            bar = "█" * int(pct / 2)
            print(f"  {cat:25s} {count:>5d} ({pct:>5.1f}%) {bar}")

        print("\n" + "-" * 80)
        print("Agreement Statistics:")
        print("-" * 80)
        print(f"  Mean agreement: {summary['mean_agreement']:.1%}")
        print(f"  Min agreement:  {summary['min_agreement']:.1%}")
        print(f"  Full agreement: {summary['full_agreement_pct']:.1f}%")

        if "physical_properties" in summary:
            print("\n" + "-" * 80)
            print("Physical Property Correlations:")
            print("-" * 80)
            for prop_name, stats in summary["physical_properties"].items():
                print(f"\n  {prop_name}:")
                print(f"    Valid values: {stats['n_valid']}")
                print(f"    Correlation with risk: {stats['correlation_with_risk']:.3f}")
                print(f"    Mean by category:")
                for cat, mean_val in stats["mean_by_category"].items():
                    print(f"      {cat}: {mean_val:.4f}")

        print("=" * 80)

    def plot_comparison(
        self,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
    ):
        """
        Plot comprehensive comparison of manifolds and ensemble.

        Creates:
        1. Category distribution per manifold
        2. Agreement heatmap
        3. Risk score distribution
        4. Physical property correlations (if available)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.ensemble_result is None:
            raise ValueError("Must compute ensemble first")

        result = self.ensemble_result
        n_manifolds = len(result.manifold_categories)

        # Determine subplot layout
        has_physical = len(result.physical_properties) > 0
        n_rows = 3 if has_physical else 2
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

        # 1. Category distribution per manifold (stacked bar)
        ax1 = axes[0, 0]
        manifold_names = list(result.manifold_categories.keys()) + ["ensemble"]
        cat_counts = {cat: [] for cat in CATEGORIES}

        for name in manifold_names[:-1]:
            cats = result.manifold_categories[name]
            for cat in CATEGORIES:
                cat_counts[cat].append(sum(1 for c in cats if c == cat))

        # Add ensemble
        for cat in CATEGORIES:
            cat_counts[cat].append(sum(1 for c in result.ensemble_category if c == cat))

        # Plot stacked bar
        bottoms = np.zeros(len(manifold_names))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(CATEGORIES)))[::-1]

        for cat, color in zip(CATEGORIES, colors):
            counts = cat_counts[cat]
            ax1.bar(manifold_names, counts, bottom=bottoms, label=cat, color=color)
            bottoms += counts

        ax1.set_xlabel("Manifold")
        ax1.set_ylabel("Count")
        ax1.set_title("Category Distribution by Manifold")
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax1.tick_params(axis='x', rotation=45)

        # 2. Agreement heatmap (manifold-manifold)
        ax2 = axes[0, 1]
        agreement_matrix = np.zeros((n_manifolds, n_manifolds))
        manifold_names_list = list(result.manifold_categories.keys())

        for i, name1 in enumerate(manifold_names_list):
            for j, name2 in enumerate(manifold_names_list):
                cats1 = result.manifold_categories[name1]
                cats2 = result.manifold_categories[name2]
                agreement = sum(1 for c1, c2 in zip(cats1, cats2) if c1 == c2) / len(cats1)
                agreement_matrix[i, j] = agreement

        sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                   xticklabels=manifold_names_list, yticklabels=manifold_names_list,
                   ax=ax2, vmin=0, vmax=1)
        ax2.set_title("Pairwise Agreement Between Manifolds")

        # 3. Risk score distribution by ensemble category
        ax3 = axes[1, 0]
        for cat in CATEGORIES:
            mask = np.array([c == cat for c in result.ensemble_category])
            if np.sum(mask) > 0:
                ax3.hist(result.ensemble_risk_score[mask], bins=20, alpha=0.5,
                        label=f"{cat} ({np.sum(mask)})")

        ax3.set_xlabel("Ensemble Risk Score")
        ax3.set_ylabel("Count")
        ax3.set_title("Risk Score Distribution by Category")
        ax3.legend(fontsize=8)

        # 4. Manifold distance spread (uncertainty indicator)
        ax4 = axes[1, 1]
        for cat in CATEGORIES:
            mask = np.array([c == cat for c in result.ensemble_category])
            if np.sum(mask) > 5:
                ax4.scatter(result.avg_manifold_distance[mask],
                           result.std_manifold_distance[mask],
                           alpha=0.5, label=cat, s=20)

        ax4.set_xlabel("Avg Manifold Distance")
        ax4.set_ylabel("Std Manifold Distance (Disagreement)")
        ax4.set_title("Manifold Distance: Mean vs Spread")
        ax4.legend(fontsize=8)

        # 5 & 6. Physical property correlations (if available)
        if has_physical:
            prop_names = list(result.physical_properties.keys())[:2]  # Max 2

            for idx, prop_name in enumerate(prop_names):
                ax = axes[2, idx]
                values = result.physical_properties[prop_name]
                valid = ~np.isnan(values)

                for cat in CATEGORIES:
                    mask = np.array([c == cat for c in result.ensemble_category]) & valid
                    if np.sum(mask) > 0:
                        ax.scatter(result.ensemble_risk_score[mask], values[mask],
                                  alpha=0.5, label=cat, s=20)

                ax.set_xlabel("Ensemble Risk Score")
                ax.set_ylabel(prop_name)
                ax.set_title(f"{prop_name} vs Risk Score")
                ax.legend(fontsize=7, loc='best')

            # Fill empty subplot if only 1 property
            if len(prop_names) == 1:
                axes[2, 1].text(0.5, 0.5, "No additional properties",
                               ha='center', va='center', transform=axes[2, 1].transAxes)
                axes[2, 1].set_frame_on(False)
                axes[2, 1].tick_params(left=False, bottom=False,
                                       labelleft=False, labelbottom=False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig

    def plot_category_by_property(
        self,
        property_name: str,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
    ):
        """Plot category distribution by physical property."""
        import matplotlib.pyplot as plt

        if self.ensemble_result is None:
            raise ValueError("Must compute ensemble first")

        if property_name not in self.ensemble_result.physical_properties:
            raise ValueError(f"Property '{property_name}' not found. "
                           f"Available: {list(self.ensemble_result.physical_properties.keys())}")

        result = self.ensemble_result
        values = result.physical_properties[property_name]
        valid = ~np.isnan(values)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 1. Box plot by category
        ax1 = axes[0]
        box_data = []
        box_labels = []
        for cat in CATEGORIES:
            mask = np.array([c == cat for c in result.ensemble_category]) & valid
            if np.sum(mask) > 0:
                box_data.append(values[mask])
                box_labels.append(f"{cat}\n(n={np.sum(mask)})")

        ax1.boxplot(box_data, labels=box_labels)
        ax1.set_ylabel(property_name)
        ax1.set_title(f"{property_name} by Category")
        ax1.tick_params(axis='x', rotation=45)

        # 2. Violin plot
        ax2 = axes[1]
        for i, (data, label) in enumerate(zip(box_data, box_labels)):
            parts = ax2.violinplot([data], positions=[i], showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_alpha(0.7)

        ax2.set_xticks(range(len(box_labels)))
        ax2.set_xticklabels(box_labels, rotation=45)
        ax2.set_ylabel(property_name)
        ax2.set_title(f"{property_name} Distribution by Category")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig


# =============================================================================
# Physical Property Calculator
# =============================================================================

class PhysicalPropertyCalculator:
    """
    Helper class for computing physical properties to correlate with fish categories.

    Available properties:
    - e_hull: Energy above convex hull (requires Materials Project data or DFT)
    - formation_energy: Formation energy per atom
    - band_gap: Electronic band gap
    - coordination_stats: Coordination number statistics
    - bond_length_validity: Bond length validation against known compounds
    """

    @staticmethod
    def compute_coordination_stats(
        atoms_list: List,  # List of ASE Atoms
        material_ids: List[str],
        cutoff: float = 3.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute coordination number statistics for structures.

        Returns dict of material_id -> {mean_cn, std_cn, min_cn, max_cn}
        """
        from ase.neighborlist import neighbor_list

        results = {}
        for atoms, mid in zip(atoms_list, material_ids):
            try:
                i_list, j_list = neighbor_list('ij', atoms, cutoff)
                cn = np.bincount(i_list, minlength=len(atoms))

                results[mid] = {
                    "mean_cn": float(np.mean(cn)),
                    "std_cn": float(np.std(cn)),
                    "min_cn": int(np.min(cn)),
                    "max_cn": int(np.max(cn)),
                }
            except Exception as e:
                warnings.warn(f"Failed to compute CN for {mid}: {e}")
                results[mid] = None

        return results

    @staticmethod
    def validate_bond_lengths(
        atoms_list: List,
        material_ids: List[str],
        reference_bonds: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """
        Validate bond lengths against reference values.

        Args:
            atoms_list: List of ASE Atoms
            material_ids: Material IDs
            reference_bonds: Dict of (elem1, elem2) -> (min_length, max_length)
                           If None, uses default covalent radii

        Returns:
            Dict of material_id -> validity_score (0-1, 1 = all bonds valid)
        """
        from ase.neighborlist import neighbor_list
        from ase.data import covalent_radii, atomic_numbers

        results = {}
        for atoms, mid in zip(atoms_list, material_ids):
            try:
                i_list, j_list, d_list = neighbor_list('ijd', atoms, 4.0)
                symbols = atoms.get_chemical_symbols()

                valid_count = 0
                total_count = 0

                for i, j, d in zip(i_list, j_list, d_list):
                    if i >= j:
                        continue

                    elem1, elem2 = sorted([symbols[i], symbols[j]])
                    total_count += 1

                    if reference_bonds and (elem1, elem2) in reference_bonds:
                        min_d, max_d = reference_bonds[(elem1, elem2)]
                    else:
                        # Use covalent radii with tolerance
                        r1 = covalent_radii[atomic_numbers[symbols[i]]]
                        r2 = covalent_radii[atomic_numbers[symbols[j]]]
                        expected = r1 + r2
                        min_d = expected * 0.8
                        max_d = expected * 1.3

                    if min_d <= d <= max_d:
                        valid_count += 1

                results[mid] = valid_count / max(total_count, 1)

            except Exception as e:
                warnings.warn(f"Failed to validate bonds for {mid}: {e}")
                results[mid] = None

        return results

    @staticmethod
    def compute_space_group_order(
        atoms_list: List,
        material_ids: List[str],
        symprec: float = 0.1,
    ) -> Dict[str, int]:
        """
        Compute space group number for structures.

        Higher space group numbers generally indicate higher symmetry.
        """
        try:
            import spglib
        except ImportError:
            raise ImportError("spglib required. Install with: pip install spglib")

        results = {}
        for atoms, mid in zip(atoms_list, material_ids):
            try:
                cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
                spg = spglib.get_spacegroup(cell, symprec=symprec)
                if spg:
                    # Extract space group number
                    sg_num = int(spg.split('(')[-1].rstrip(')'))
                    results[mid] = sg_num
                else:
                    results[mid] = 1  # P1 (lowest symmetry)
            except Exception as e:
                warnings.warn(f"Failed to compute space group for {mid}: {e}")
                results[mid] = None

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def combine_manifold_csvs(
    csv_paths: Dict[str, str],
    output_path: str,
    voting_method: str = "weighted_majority",
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Quick function to combine multiple manifold CSV results.

    Args:
        csv_paths: Dict of manifold_name -> csv_path
        output_path: Output combined CSV path
        voting_method: Voting method for ensemble
        weights: Optional weights per manifold

    Returns:
        Combined DataFrame
    """
    analyzer = EnsembleAnalyzer()

    for name, path in csv_paths.items():
        weight = weights.get(name, 1.0) if weights else 1.0
        analyzer.add_manifold_results(name, csv_path=path, weight=weight)

    analyzer.compute_ensemble(voting_method=voting_method)
    return analyzer.export_combined_results(output_path)


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example demonstrating ensemble analysis."""
    print("=" * 80)
    print("Ensemble Analysis Example")
    print("=" * 80)

    print("""
    # Full Pipeline Example:

    from mlip_embedding_extractor import MLIPEmbeddingExtractor
    from anchor_selection import AnchorSelector
    from embedding_transform import EmbeddingTransformer
    from manifold_fish_analysis import ManifoldFishAnalyzer, aggregate_by_material_id
    from ensemble_analysis import EnsembleAnalyzer

    # =========================================================================
    # Step 1: Extract embeddings from multiple MLIPs
    # =========================================================================

    models = {
        "mace_mpa": {"type": "mace", "path": "/path/to/mace-mpa.model"},
        "orb_v3": {"type": "orb", "path": "orb_v3_direct_inf_omat"},
        "sevennet": {"type": "sevennet", "path": "7net-omat"},
    }

    all_results = {}
    for name, config in models.items():
        extractor = MLIPEmbeddingExtractor(config["type"], config["path"])
        results = extractor.extract_from_json("generated_structures.json")
        all_results[name] = results

    # =========================================================================
    # Step 2: Select anchors (shared across all manifolds for comparability)
    # =========================================================================

    # Use one model's reference as anchor source
    ref_extractor = MLIPEmbeddingExtractor("mace", models["mace_mpa"]["path"])
    ref_results = ref_extractor.extract_from_json("reference_structures.json")

    selector = AnchorSelector(method="direct", n_anchors=100)
    anchor_result = selector.fit_transform(ref_results.flat_embeddings)
    anchors = anchor_result.anchor_embeddings

    # =========================================================================
    # Step 3: Transform all embeddings to unified anchor space
    # =========================================================================

    transformer = EmbeddingTransformer(
        method="anchor",
        anchors=anchors,
        distance_metric="cosine"
    )

    # Transform each model's embeddings
    for name in all_results:
        gen_trans = transformer.transform(all_results[name].flat_embeddings)
        ref_trans = transformer.transform(ref_results.flat_embeddings)

        # Aggregate to material level
        gen_mat, gen_ids = aggregate_by_material_id(
            gen_trans, all_results[name].flat_material_ids
        )
        ref_mat, ref_ids = aggregate_by_material_id(
            ref_trans, ref_results.flat_material_ids
        )

        # Analyze with ManifoldFishAnalyzer
        analyzer = ManifoldFishAnalyzer(ref_mat, ref_ids, boundary_method="alpha_shape")
        fish_results = analyzer.analyze(gen_mat, gen_ids)
        analyzer.export_results(fish_results, f"results_{name}.csv")

    # =========================================================================
    # Step 4: Combine with ensemble voting
    # =========================================================================

    ensemble = EnsembleAnalyzer()
    ensemble.add_manifold_results("mace_mpa", csv_path="results_mace_mpa.csv", weight=1.0)
    ensemble.add_manifold_results("orb_v3", csv_path="results_orb_v3.csv", weight=1.0)
    ensemble.add_manifold_results("sevennet", csv_path="results_sevennet.csv", weight=1.0)

    # Compute ensemble with weighted voting
    ensemble_result = ensemble.compute_ensemble(voting_method="weighted_majority")

    # =========================================================================
    # Step 5: Add physical properties and analyze
    # =========================================================================

    # Add e_hull from your DFT calculations
    e_hull_data = pd.read_csv("e_hull_results.csv")
    ensemble.add_physical_property_from_df(e_hull_data, "e_hull", id_column="material_id")

    # Export and visualize
    ensemble.export_combined_results("ensemble_results.csv")
    ensemble.print_summary()
    ensemble.plot_comparison(save_path="ensemble_comparison.png")
    ensemble.plot_category_by_property("e_hull", save_path="e_hull_by_category.png")

    # =========================================================================
    # Step 6: Interpret results
    # =========================================================================

    # Get priority candidates
    df = pd.read_csv("ensemble_results.csv")

    # High-confidence stable candidates
    stable = df[
        (df['ensemble_category'].isin(['fish_in_water', 'frontier_fish'])) &
        (df['category_agreement'] > 0.8) &
        (df['e_hull'] < 0.1)  # < 100 meV/atom above hull
    ]
    print(f"High-confidence stable candidates: {len(stable)}")

    # Novel but potentially stable
    novel = df[
        (df['ensemble_category'] == 'adventurous_fish') &
        (df['category_agreement'] > 0.6)
    ]
    print(f"Novel candidates for DFT: {len(novel)}")
    """)


if __name__ == "__main__":
    example_usage()
