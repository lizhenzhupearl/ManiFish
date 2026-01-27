"""
Manifold Type Definitions for Materials Analysis

Different manifolds represent different "water" (reference space) concepts:

1. STABILITY MANIFOLD (MP-20 / computational)
   - Source: Materials Project computed structures
   - Criterion: DFT-relaxed, thermodynamically evaluated
   - Interpretation: "Computationally stable" chemistry
   - Use case: General stability screening

2. SYNTHESIZABILITY MANIFOLD (ICSD + e_Hull filter)
   - Source: ICSD (Inorganic Crystal Structure Database)
   - Criterion: Experimentally synthesized AND e_Hull < threshold
   - Interpretation: "Actually makeable" chemistry
   - Use case: Synthesizability prediction

3. APPLICATION-SPECIFIC MANIFOLDS
   - Thermoelectric manifold: Known thermoelectric materials
   - Battery cathode manifold: Known cathode materials
   - Superconductor manifold: Known superconductors
   - etc.

The key insight:
- Same FISH (generated structure)
- Different WATER (reference manifold)
- Different INTERPRETATION (stability vs synthesizability vs application)

Example:
    A structure might be:
    - fish_in_water in STABILITY manifold (computationally stable)
    - adventurous_fish in SYNTHESIZABILITY manifold (hard to synthesize)

    This tells you: "Stable but challenging synthesis route needed"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
from pathlib import Path
import numpy as np
import pandas as pd
import warnings


# =============================================================================
# Manifold Type Definitions
# =============================================================================

@dataclass
class ManifoldType:
    """Definition of a manifold type."""
    name: str
    description: str
    source_description: str
    interpretation: Dict[str, str]  # category -> meaning

    # Filtering criteria
    e_hull_max: Optional[float] = None  # Maximum e_hull (eV/atom)
    requires_icsd: bool = False  # Must be in ICSD
    requires_experimental: bool = False  # Must have experimental data

    # Application-specific filters
    property_filters: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Suggested thresholds (may differ from default)
    suggested_thresholds: Dict[str, float] = field(default_factory=dict)


# Pre-defined manifold types
STABILITY_MANIFOLD = ManifoldType(
    name="stability",
    description="Computational stability manifold based on DFT-relaxed structures",
    source_description="Materials Project / OMAT / Alexandria computed structures",
    interpretation={
        "redundant_fish": "Near-duplicate of known computed structure",
        "fish_in_water": "Computationally stable, similar to known chemistry",
        "frontier_fish": "Computationally stable, unexplored composition space",
        "edge_fish": "At boundary of known computational chemistry",
        "adventurous_fish": "Novel chemistry, computational stability uncertain",
        "geometric_atypical": "Unusual local geometry but has neighbors, potentially novel",
        "structural_hallucination": "Far from known chemistry, no support, likely unphysical",
    },
    e_hull_max=None,  # Include all computed structures
    requires_icsd=False,
)

SYNTHESIZABILITY_MANIFOLD = ManifoldType(
    name="synthesizability",
    description="Synthesizability manifold based on experimentally realized structures",
    source_description="ICSD structures with e_Hull < 0.1 eV/atom",
    interpretation={
        "redundant_fish": "Very similar to known synthesized material",
        "fish_in_water": "Likely synthesizable, follows known synthesis patterns",
        "frontier_fish": "Potentially synthesizable, unexplored but consistent",
        "edge_fish": "At boundary of known synthesis space, may need special conditions",
        "adventurous_fish": "Challenging synthesis, may require novel routes",
        "geometric_atypical": "Unusual geometry but has neighbors, novel synthesis candidate",
        "structural_hallucination": "No synthesis support, unlikely to be realizable",
    },
    e_hull_max=0.1,  # 100 meV/atom above hull
    requires_icsd=True,
    suggested_thresholds={
        "stability_threshold": 0.4,  # Tighter for synthesizability
        "geometry_threshold": 0.25,
    },
)

METASTABLE_MANIFOLD = ManifoldType(
    name="metastable",
    description="Metastable materials manifold (synthesized but above hull)",
    source_description="ICSD structures with 0.05 < e_Hull < 0.2 eV/atom",
    interpretation={
        "redundant_fish": "Similar to known metastable material",
        "fish_in_water": "Consistent with metastable chemistry patterns",
        "frontier_fish": "Novel metastable candidate",
        "edge_fish": "At boundary of metastable space",
        "adventurous_fish": "Highly metastable, kinetic stabilization needed",
        "geometric_atypical": "Unusual geometry, potentially novel metastable",
        "structural_hallucination": "Too unstable even for metastable, no support",
    },
    e_hull_max=0.2,
    requires_icsd=True,
    property_filters={"e_hull": (0.05, 0.2)},
)


# =============================================================================
# Reference Data Filters
# =============================================================================

class ReferenceDataFilter:
    """
    Filter reference structures based on manifold type criteria.

    Example usage:
        filter = ReferenceDataFilter(manifold_type=SYNTHESIZABILITY_MANIFOLD)
        filtered_df = filter.apply(mp_data_df)
    """

    def __init__(self, manifold_type: ManifoldType):
        self.manifold_type = manifold_type

    def apply(
        self,
        df: pd.DataFrame,
        e_hull_column: str = "e_above_hull",
        icsd_column: str = "icsd_ids",  # or "is_icsd", "has_icsd"
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Apply manifold-specific filters to reference data.

        Args:
            df: DataFrame with structure data
            e_hull_column: Column name for energy above hull
            icsd_column: Column name for ICSD IDs/flag
            verbose: Print filtering statistics

        Returns:
            Filtered DataFrame
        """
        original_count = len(df)
        filtered = df.copy()

        # e_Hull filter
        if self.manifold_type.e_hull_max is not None:
            if e_hull_column in filtered.columns:
                filtered = filtered[filtered[e_hull_column] <= self.manifold_type.e_hull_max]
                if verbose:
                    print(f"  After e_Hull <= {self.manifold_type.e_hull_max}: {len(filtered)}")
            else:
                warnings.warn(f"e_hull column '{e_hull_column}' not found")

        # ICSD filter
        if self.manifold_type.requires_icsd:
            if icsd_column in filtered.columns:
                # Handle different ICSD column formats
                if filtered[icsd_column].dtype == bool:
                    filtered = filtered[filtered[icsd_column]]
                elif filtered[icsd_column].dtype == object:
                    # List or string of ICSD IDs
                    filtered = filtered[filtered[icsd_column].notna()]
                    filtered = filtered[filtered[icsd_column].apply(
                        lambda x: len(x) > 0 if isinstance(x, (list, str)) else False
                    )]
                if verbose:
                    print(f"  After ICSD filter: {len(filtered)}")
            else:
                warnings.warn(f"ICSD column '{icsd_column}' not found")

        # Property-specific filters
        for prop, (min_val, max_val) in self.manifold_type.property_filters.items():
            if prop in filtered.columns:
                filtered = filtered[
                    (filtered[prop] >= min_val) & (filtered[prop] <= max_val)
                ]
                if verbose:
                    print(f"  After {prop} in [{min_val}, {max_val}]: {len(filtered)}")

        if verbose:
            print(f"  Final: {len(filtered)} / {original_count} "
                  f"({100*len(filtered)/original_count:.1f}%)")

        return filtered


# =============================================================================
# Multi-Manifold Analyzer
# =============================================================================

class MultiManifoldAnalyzer:
    """
    Analyze structures against multiple manifold types.

    This allows comparing the same generated structures against:
    - Stability manifold (computational)
    - Synthesizability manifold (experimental)
    - Application-specific manifolds

    Example:
        analyzer = MultiManifoldAnalyzer()

        # Add manifolds
        analyzer.add_manifold(
            "stability",
            reference_embeddings=mp20_embeddings,
            reference_ids=mp20_ids,
            manifold_type=STABILITY_MANIFOLD,
        )
        analyzer.add_manifold(
            "synthesizability",
            reference_embeddings=icsd_filtered_embeddings,
            reference_ids=icsd_ids,
            manifold_type=SYNTHESIZABILITY_MANIFOLD,
        )

        # Analyze generated structures
        results = analyzer.analyze_all(generated_embeddings, generated_ids)

        # Compare
        analyzer.compare_manifolds(results)
    """

    def __init__(self):
        self.manifolds: Dict[str, dict] = {}
        self.analyzers: Dict[str, object] = {}  # ManifoldFishAnalyzer instances
        self.results: Dict[str, list] = {}

    def add_manifold(
        self,
        name: str,
        reference_embeddings: np.ndarray,
        reference_ids: List[str],
        manifold_type: ManifoldType,
        **analyzer_kwargs,
    ):
        """
        Add a manifold for analysis.

        Args:
            name: Unique name for this manifold
            reference_embeddings: Reference embeddings (the "water")
            reference_ids: Reference material IDs
            manifold_type: ManifoldType definition
            **analyzer_kwargs: Additional args for ManifoldFishAnalyzer
        """
        # Import here to avoid circular imports
        from manifold_fish_analysis import ManifoldFishAnalyzer

        # Apply suggested thresholds from manifold type
        for key, value in manifold_type.suggested_thresholds.items():
            if key not in analyzer_kwargs:
                analyzer_kwargs[key] = value

        analyzer = ManifoldFishAnalyzer(
            reference_embeddings=reference_embeddings,
            reference_ids=reference_ids,
            **analyzer_kwargs,
        )

        self.manifolds[name] = {
            "type": manifold_type,
            "n_reference": len(reference_embeddings),
            "embed_dim": reference_embeddings.shape[1],
        }
        self.analyzers[name] = analyzer

        print(f"Added manifold '{name}' ({manifold_type.name}): "
              f"{len(reference_embeddings)} reference structures")

    def analyze_all(
        self,
        generated_embeddings: np.ndarray,
        material_ids: List[str],
    ) -> Dict[str, list]:
        """
        Analyze generated structures against all manifolds.

        Returns:
            Dict of manifold_name -> list of ManifoldFishResult
        """
        results = {}

        for name, analyzer in self.analyzers.items():
            print(f"\nAnalyzing against '{name}' manifold...")
            results[name] = analyzer.analyze(generated_embeddings, material_ids)

        self.results = results
        return results

    def get_combined_interpretation(
        self,
        material_id: str,
    ) -> Dict[str, str]:
        """
        Get combined interpretation for a material across all manifolds.

        Returns:
            Dict with category and interpretation per manifold
        """
        interpretation = {}

        for name, results in self.results.items():
            manifold_type = self.manifolds[name]["type"]

            # Find result for this material
            for r in results:
                if r.material_id == material_id:
                    category = r.category
                    meaning = manifold_type.interpretation.get(category, "")
                    interpretation[name] = {
                        "category": category,
                        "interpretation": meaning,
                        "confidence": r.confidence,
                    }
                    break

        return interpretation

    def compare_manifolds(
        self,
        results: Optional[Dict[str, list]] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare categories across manifolds.

        Returns DataFrame with:
        - material_id
        - category_<manifold_name> for each manifold
        - agreement metrics
        """
        results = results or self.results

        if not results:
            raise ValueError("No results. Call analyze_all() first.")

        # Get common material IDs
        all_ids = [set(r.material_id for r in res) for res in results.values()]
        common_ids = sorted(set.intersection(*all_ids))

        # Build comparison DataFrame
        data = {"material_id": common_ids}

        for name, res_list in results.items():
            id_to_result = {r.material_id: r for r in res_list}
            data[f"category_{name}"] = [id_to_result[mid].category for mid in common_ids]
            data[f"confidence_{name}"] = [id_to_result[mid].confidence for mid in common_ids]
            data[f"manifold_dist_{name}"] = [id_to_result[mid].manifold_distance for mid in common_ids]

        df = pd.DataFrame(data)

        # Add agreement metrics
        category_cols = [c for c in df.columns if c.startswith("category_")]

        def count_agreement(row):
            cats = [row[c] for c in category_cols]
            from collections import Counter
            counter = Counter(cats)
            most_common = counter.most_common(1)[0]
            return most_common[1] / len(cats)

        df["category_agreement"] = df.apply(count_agreement, axis=1)

        # Combined interpretation
        def get_synthesis_status(row):
            """Infer synthesizability from manifold comparison."""
            stability_cat = row.get("category_stability", "")
            synth_cat = row.get("category_synthesizability", "")

            if "fish_in_water" in synth_cat or "frontier_fish" in synth_cat:
                return "likely_synthesizable"
            elif "adventurous_fish" in synth_cat and "fish_in_water" in stability_cat:
                return "stable_but_challenging_synthesis"
            elif "structural_hallucination" in synth_cat:
                return "unlikely_synthesizable"
            else:
                return "uncertain"

        if "category_stability" in df.columns and "category_synthesizability" in df.columns:
            df["synthesis_status"] = df.apply(get_synthesis_status, axis=1)

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Comparison saved to {output_path}")

        return df

    def print_comparison_summary(self, df: Optional[pd.DataFrame] = None):
        """Print summary of manifold comparison."""
        if df is None:
            df = self.compare_manifolds()

        print("=" * 80)
        print("Multi-Manifold Comparison Summary")
        print("=" * 80)

        print(f"\nMaterials analyzed: {len(df)}")
        print(f"Manifolds: {list(self.manifolds.keys())}")

        # Per-manifold category distribution
        for name in self.manifolds.keys():
            col = f"category_{name}"
            if col in df.columns:
                print(f"\n{name.upper()} Manifold:")
                for cat in df[col].unique():
                    count = (df[col] == cat).sum()
                    print(f"  {cat}: {count} ({100*count/len(df):.1f}%)")

        # Cross-manifold patterns
        if "synthesis_status" in df.columns:
            print("\n" + "-" * 40)
            print("SYNTHESIS STATUS (cross-manifold):")
            for status in df["synthesis_status"].unique():
                count = (df["synthesis_status"] == status).sum()
                print(f"  {status}: {count} ({100*count/len(df):.1f}%)")

        # Agreement statistics
        print("\n" + "-" * 40)
        print("MANIFOLD AGREEMENT:")
        print(f"  Mean agreement: {df['category_agreement'].mean():.1%}")
        print(f"  Full agreement: {(df['category_agreement'] == 1.0).sum()} materials")
        print(f"  Major disagreement (<50%): {(df['category_agreement'] < 0.5).sum()} materials")

        print("=" * 80)

    def plot_comparison(
        self,
        df: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ):
        """Visualize manifold comparison."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        if df is None:
            df = self.compare_manifolds()

        manifold_names = list(self.manifolds.keys())
        n_manifolds = len(manifold_names)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Category heatmap (confusion matrix style)
        ax1 = axes[0, 0]
        if n_manifolds >= 2:
            cat_col1 = f"category_{manifold_names[0]}"
            cat_col2 = f"category_{manifold_names[1]}"

            cross_tab = pd.crosstab(df[cat_col1], df[cat_col2])
            sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues", ax=ax1)
            ax1.set_xlabel(manifold_names[1])
            ax1.set_ylabel(manifold_names[0])
            ax1.set_title("Category Cross-tabulation")

        # 2. Agreement distribution
        ax2 = axes[0, 1]
        ax2.hist(df["category_agreement"], bins=20, edgecolor="black", alpha=0.7)
        ax2.axvline(df["category_agreement"].mean(), color="red", linestyle="--",
                   label=f"Mean: {df['category_agreement'].mean():.2f}")
        ax2.set_xlabel("Category Agreement")
        ax2.set_ylabel("Count")
        ax2.set_title("Agreement Distribution")
        ax2.legend()

        # 3. Manifold distance comparison
        ax3 = axes[1, 0]
        dist_cols = [f"manifold_dist_{name}" for name in manifold_names if f"manifold_dist_{name}" in df.columns]
        if len(dist_cols) >= 2:
            ax3.scatter(df[dist_cols[0]], df[dist_cols[1]], alpha=0.3, s=10)
            ax3.set_xlabel(f"Manifold Distance ({manifold_names[0]})")
            ax3.set_ylabel(f"Manifold Distance ({manifold_names[1]})")
            ax3.set_title("Manifold Distance Comparison")
            # Add diagonal
            max_val = max(df[dist_cols[0]].max(), df[dist_cols[1]].max())
            ax3.plot([0, max_val], [0, max_val], "r--", alpha=0.5)

        # 4. Synthesis status (if available)
        ax4 = axes[1, 1]
        if "synthesis_status" in df.columns:
            status_counts = df["synthesis_status"].value_counts()
            colors = {
                "likely_synthesizable": "green",
                "stable_but_challenging_synthesis": "orange",
                "uncertain": "gray",
                "unlikely_synthesizable": "red",
            }
            bar_colors = [colors.get(s, "blue") for s in status_counts.index]
            status_counts.plot(kind="bar", ax=ax4, color=bar_colors)
            ax4.set_xlabel("Synthesis Status")
            ax4.set_ylabel("Count")
            ax4.set_title("Combined Synthesis Assessment")
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, "Add both stability and\nsynthesizability manifolds\nfor synthesis assessment",
                    ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig


# =============================================================================
# Helper Functions
# =============================================================================

def filter_for_synthesizability(
    df: pd.DataFrame,
    e_hull_column: str = "e_above_hull",
    icsd_column: str = "icsd_ids",
    e_hull_max: float = 0.1,
) -> pd.DataFrame:
    """
    Quick filter for synthesizability manifold.

    Args:
        df: Materials data with e_hull and ICSD columns
        e_hull_column: Name of e_hull column
        icsd_column: Name of ICSD column
        e_hull_max: Maximum e_hull (eV/atom)

    Returns:
        Filtered DataFrame with ICSD structures and e_Hull < threshold
    """
    filter = ReferenceDataFilter(SYNTHESIZABILITY_MANIFOLD)
    filter.manifold_type.e_hull_max = e_hull_max
    return filter.apply(df, e_hull_column, icsd_column)


def create_application_manifold(
    name: str,
    description: str,
    property_name: str,
    property_range: Tuple[float, float],
    e_hull_max: float = 0.1,
    requires_icsd: bool = False,
) -> ManifoldType:
    """
    Create a custom application-specific manifold.

    Example:
        # Thermoelectric manifold
        te_manifold = create_application_manifold(
            name="thermoelectric",
            description="Known thermoelectric materials",
            property_name="seebeck_coefficient",
            property_range=(100, 1000),  # µV/K
            requires_icsd=True,
        )
    """
    return ManifoldType(
        name=name,
        description=description,
        source_description=f"Materials with {property_name} in {property_range}",
        interpretation={
            "redundant_fish": f"Similar to known {name} material",
            "fish_in_water": f"Consistent with {name} chemistry",
            "frontier_fish": f"Novel {name} candidate",
            "edge_fish": f"At boundary of {name} space",
            "adventurous_fish": f"Unusual for {name}, needs validation",
            "geometric_atypical": f"Atypical geometry for {name}, potentially novel",
            "structural_hallucination": f"No support for {name}, unlikely",
        },
        e_hull_max=e_hull_max,
        requires_icsd=requires_icsd,
        property_filters={property_name: property_range},
    )


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example demonstrating multi-manifold analysis."""
    print("=" * 80)
    print("Multi-Manifold Analysis Example")
    print("=" * 80)

    print("""
    # ==========================================================================
    # Step 1: Prepare reference data for different manifolds
    # ==========================================================================

    import pandas as pd
    from manifold_types import (
        STABILITY_MANIFOLD,
        SYNTHESIZABILITY_MANIFOLD,
        filter_for_synthesizability,
    )

    # Load Materials Project data
    mp_data = pd.read_csv("mp_data.csv")  # Has e_above_hull, icsd_ids columns

    # Full MP data for stability manifold
    stability_structures = mp_data  # All ~150K structures

    # Filtered for synthesizability manifold
    synth_structures = filter_for_synthesizability(
        mp_data,
        e_hull_column="e_above_hull",
        icsd_column="icsd_ids",
        e_hull_max=0.1  # 100 meV/atom
    )
    print(f"Stability: {len(stability_structures)} structures")
    print(f"Synthesizability: {len(synth_structures)} structures")

    # ==========================================================================
    # Step 2: Extract embeddings for each reference set
    # ==========================================================================

    from mlip_embedding_extractor import MLIPEmbeddingExtractor
    from embedding_transform import EmbeddingTransformer
    from anchor_selection import AnchorSelector

    extractor = MLIPEmbeddingExtractor("mace", "/path/to/mace.model", num_layers=1)

    # Extract embeddings
    stability_emb = extractor.extract_from_structures(stability_structures)
    synth_emb = extractor.extract_from_structures(synth_structures)

    # Select shared anchors (use synthesizability set for cleaner space)
    selector = AnchorSelector(method="direct", n_anchors=100)
    anchor_result = selector.fit_transform(synth_emb.flat_embeddings)
    anchors = anchor_result.anchor_embeddings

    # Transform to anchor space
    transformer = EmbeddingTransformer(method="anchor", anchors=anchors)

    stability_trans = transformer.transform(stability_emb.flat_embeddings)
    synth_trans = transformer.transform(synth_emb.flat_embeddings)

    # Aggregate to material level
    from manifold_fish_analysis import aggregate_by_material_id

    stability_mat, stability_ids = aggregate_by_material_id(
        stability_trans, stability_emb.flat_material_ids
    )
    synth_mat, synth_ids = aggregate_by_material_id(
        synth_trans, synth_emb.flat_material_ids
    )

    # ==========================================================================
    # Step 3: Create multi-manifold analyzer
    # ==========================================================================

    from manifold_types import MultiManifoldAnalyzer

    analyzer = MultiManifoldAnalyzer()

    # Add stability manifold
    analyzer.add_manifold(
        name="stability",
        reference_embeddings=stability_mat,
        reference_ids=stability_ids,
        manifold_type=STABILITY_MANIFOLD,
        boundary_method="alpha_shape",
    )

    # Add synthesizability manifold
    analyzer.add_manifold(
        name="synthesizability",
        reference_embeddings=synth_mat,
        reference_ids=synth_ids,
        manifold_type=SYNTHESIZABILITY_MANIFOLD,
        boundary_method="alpha_shape",
    )

    # ==========================================================================
    # Step 4: Analyze generated structures
    # ==========================================================================

    # Load generated structures
    gen_emb = extractor.extract_from_json("generated_structures.json")
    gen_trans = transformer.transform(gen_emb.flat_embeddings)
    gen_mat, gen_ids = aggregate_by_material_id(gen_trans, gen_emb.flat_material_ids)

    # Analyze against both manifolds
    results = analyzer.analyze_all(gen_mat, gen_ids)

    # Compare
    comparison_df = analyzer.compare_manifolds(output_path="manifold_comparison.csv")
    analyzer.print_comparison_summary(comparison_df)
    analyzer.plot_comparison(save_path="manifold_comparison.png")

    # ==========================================================================
    # Step 5: Interpret results
    # ==========================================================================

    # Get detailed interpretation for a specific material
    interp = analyzer.get_combined_interpretation("gen-0001")
    print(f"Material gen-0001:")
    print(f"  Stability: {interp['stability']['category']} - {interp['stability']['interpretation']}")
    print(f"  Synthesizability: {interp['synthesizability']['category']} - {interp['synthesizability']['interpretation']}")

    # Find materials with interesting patterns

    # Stable but hard to synthesize (novel synthesis challenge)
    challenging = comparison_df[
        (comparison_df["category_stability"] == "fish_in_water") &
        (comparison_df["category_synthesizability"] == "adventurous_fish")
    ]
    print(f"\\nStable but challenging synthesis: {len(challenging)} materials")

    # Novel in both (high-risk high-reward)
    novel = comparison_df[
        (comparison_df["category_stability"] == "frontier_fish") &
        (comparison_df["category_synthesizability"] == "frontier_fish")
    ]
    print(f"Novel in both spaces: {len(novel)} materials")

    # High priority: synthesizable and novel
    priority = comparison_df[
        (comparison_df["category_synthesizability"].isin(["fish_in_water", "frontier_fish"])) &
        (comparison_df["category_stability"] == "frontier_fish")
    ]
    print(f"Priority candidates: {len(priority)} materials")
    """)


if __name__ == "__main__":
    example_usage()
