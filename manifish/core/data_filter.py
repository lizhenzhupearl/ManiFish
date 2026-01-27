"""
Data Filter Module for Reference Structure Selection

Filter computed materials data (e.g., MP-20 train.csv) based on various criteria
to create different reference manifolds.

Supported filters:
- e_above_hull: Thermodynamic stability
- formation_energy_per_atom: Formation energy
- band_gap: Electronic properties
- spacegroup.number: Symmetry
- elements: Elemental composition
- n_elements: Number of elements

Input format (train.csv columns):
- material_id
- formation_energy_per_atom
- band_gap
- pretty_formula
- e_above_hull
- elements
- cif
- spacegroup.number
- composition

Usage:
    from data_filter import DataFilter

    # Load data
    df = pd.read_csv("train.csv")

    # Filter for thermodynamically stable materials
    filter = DataFilter()
    stable_df = filter.filter_by_e_hull(df, max_e_hull=0.1)

    # Or use combined filters
    filtered_df = filter.apply_filters(
        df,
        e_hull_max=0.1,
        band_gap_range=(0.5, 3.0),
        exclude_elements=["Tc", "Pm"],
    )
"""

import ast
import warnings
from typing import Dict, List, Optional, Tuple, Union, Set
from pathlib import Path

import numpy as np
import pandas as pd


class DataFilter:
    """
    Filter materials data based on various criteria.

    Works with CSV files containing columns like:
    material_id, e_above_hull, formation_energy_per_atom, band_gap,
    elements, spacegroup.number, etc.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize filter.

        Args:
            verbose: Print filtering statistics
        """
        self.verbose = verbose

    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load CSV file."""
        df = pd.read_csv(path)
        self._log(f"Loaded {len(df)} structures from {path}")
        self._log(f"Columns: {list(df.columns)}")
        return df

    # =========================================================================
    # Individual Filters
    # =========================================================================

    def filter_by_e_hull(
        self,
        df: pd.DataFrame,
        max_e_hull: float = 0.1,
        column: str = "e_above_hull",
    ) -> pd.DataFrame:
        """
        Filter by energy above hull.

        Args:
            df: Input DataFrame
            max_e_hull: Maximum e_above_hull in eV/atom
            column: Column name for e_hull

        Returns:
            Filtered DataFrame
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

        original = len(df)
        filtered = df[df[column] <= max_e_hull].copy()

        self._log(f"e_hull <= {max_e_hull}: {len(filtered)}/{original} "
                  f"({100*len(filtered)/original:.1f}%)")

        return filtered

    def filter_by_formation_energy(
        self,
        df: pd.DataFrame,
        min_fe: Optional[float] = None,
        max_fe: Optional[float] = None,
        column: str = "formation_energy_per_atom",
    ) -> pd.DataFrame:
        """
        Filter by formation energy per atom.

        Args:
            df: Input DataFrame
            min_fe: Minimum formation energy (eV/atom)
            max_fe: Maximum formation energy (eV/atom)
            column: Column name

        Returns:
            Filtered DataFrame
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        original = len(df)
        filtered = df.copy()

        if min_fe is not None:
            filtered = filtered[filtered[column] >= min_fe]
        if max_fe is not None:
            filtered = filtered[filtered[column] <= max_fe]

        self._log(f"formation_energy in [{min_fe}, {max_fe}]: {len(filtered)}/{original}")

        return filtered

    def filter_by_band_gap(
        self,
        df: pd.DataFrame,
        min_gap: Optional[float] = None,
        max_gap: Optional[float] = None,
        column: str = "band_gap",
    ) -> pd.DataFrame:
        """
        Filter by band gap.

        Args:
            df: Input DataFrame
            min_gap: Minimum band gap (eV)
            max_gap: Maximum band gap (eV)
            column: Column name

        Returns:
            Filtered DataFrame
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        original = len(df)
        filtered = df.copy()

        if min_gap is not None:
            filtered = filtered[filtered[column] >= min_gap]
        if max_gap is not None:
            filtered = filtered[filtered[column] <= max_gap]

        self._log(f"band_gap in [{min_gap}, {max_gap}]: {len(filtered)}/{original}")

        return filtered

    def filter_metals(
        self,
        df: pd.DataFrame,
        column: str = "band_gap",
        gap_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Filter to keep only metals (band_gap ~ 0).

        Args:
            df: Input DataFrame
            column: Band gap column name
            gap_threshold: Threshold below which considered metallic

        Returns:
            Filtered DataFrame with metals only
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        original = len(df)
        filtered = df[df[column] <= gap_threshold].copy()

        self._log(f"Metals (gap <= {gap_threshold}): {len(filtered)}/{original}")

        return filtered

    def filter_insulators(
        self,
        df: pd.DataFrame,
        column: str = "band_gap",
        min_gap: float = 0.5,
    ) -> pd.DataFrame:
        """
        Filter to keep only insulators/semiconductors.

        Args:
            df: Input DataFrame
            column: Band gap column name
            min_gap: Minimum gap to be considered insulator

        Returns:
            Filtered DataFrame with insulators only
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        original = len(df)
        filtered = df[df[column] >= min_gap].copy()

        self._log(f"Insulators (gap >= {min_gap}): {len(filtered)}/{original}")

        return filtered

    def filter_by_elements(
        self,
        df: pd.DataFrame,
        include_elements: Optional[List[str]] = None,
        exclude_elements: Optional[List[str]] = None,
        require_all: bool = False,
        column: str = "elements",
    ) -> pd.DataFrame:
        """
        Filter by elemental composition.

        Args:
            df: Input DataFrame
            include_elements: Elements that must be present
                             If require_all=False, at least one must be present
                             If require_all=True, all must be present
            exclude_elements: Elements that must NOT be present
            require_all: Whether all include_elements must be present
            column: Column name containing elements

        Returns:
            Filtered DataFrame
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        original = len(df)
        filtered = df.copy()

        # Parse elements column if it's a string representation of list
        def parse_elements(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    return x.split()
            return x

        elements_parsed = filtered[column].apply(parse_elements)

        # Include filter
        if include_elements:
            include_set = set(include_elements)
            if require_all:
                # All specified elements must be present
                mask = elements_parsed.apply(lambda x: include_set.issubset(set(x)))
            else:
                # At least one must be present
                mask = elements_parsed.apply(lambda x: bool(include_set & set(x)))
            filtered = filtered[mask]
            self._log(f"Include elements {include_elements} (require_all={require_all}): "
                      f"{len(filtered)}/{original}")

        # Exclude filter
        if exclude_elements:
            exclude_set = set(exclude_elements)
            mask = elements_parsed.apply(lambda x: not bool(exclude_set & set(x)))
            filtered = filtered[mask]
            self._log(f"Exclude elements {exclude_elements}: {len(filtered)}/{original}")

        return filtered

    def filter_by_n_elements(
        self,
        df: pd.DataFrame,
        min_elements: Optional[int] = None,
        max_elements: Optional[int] = None,
        column: str = "elements",
    ) -> pd.DataFrame:
        """
        Filter by number of elements in composition.

        Args:
            df: Input DataFrame
            min_elements: Minimum number of elements
            max_elements: Maximum number of elements
            column: Column containing elements

        Returns:
            Filtered DataFrame
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        original = len(df)

        def count_elements(x):
            if isinstance(x, str):
                try:
                    return len(ast.literal_eval(x))
                except:
                    return len(x.split())
            return len(x)

        n_elements = df[column].apply(count_elements)
        filtered = df.copy()

        if min_elements is not None:
            filtered = filtered[n_elements >= min_elements]
        if max_elements is not None:
            filtered = filtered[n_elements <= max_elements]

        self._log(f"n_elements in [{min_elements}, {max_elements}]: {len(filtered)}/{original}")

        return filtered

    def filter_by_spacegroup(
        self,
        df: pd.DataFrame,
        spacegroups: Optional[List[int]] = None,
        exclude_spacegroups: Optional[List[int]] = None,
        min_sg: Optional[int] = None,
        max_sg: Optional[int] = None,
        column: str = "spacegroup.number",
    ) -> pd.DataFrame:
        """
        Filter by space group number.

        Args:
            df: Input DataFrame
            spacegroups: List of space groups to include
            exclude_spacegroups: List of space groups to exclude
            min_sg: Minimum space group number
            max_sg: Maximum space group number
            column: Column name

        Returns:
            Filtered DataFrame
        """
        # Handle column name with dot
        if column not in df.columns:
            # Try alternative names
            alternatives = ["spacegroup_number", "sg_number", "spacegroup"]
            for alt in alternatives:
                if alt in df.columns:
                    column = alt
                    break
            else:
                raise ValueError(f"Column '{column}' not found")

        original = len(df)
        filtered = df.copy()

        if spacegroups is not None:
            filtered = filtered[filtered[column].isin(spacegroups)]
            self._log(f"Spacegroups {spacegroups}: {len(filtered)}/{original}")

        if exclude_spacegroups is not None:
            filtered = filtered[~filtered[column].isin(exclude_spacegroups)]
            self._log(f"Exclude spacegroups {exclude_spacegroups}: {len(filtered)}/{original}")

        if min_sg is not None:
            filtered = filtered[filtered[column] >= min_sg]
        if max_sg is not None:
            filtered = filtered[filtered[column] <= max_sg]

        if min_sg is not None or max_sg is not None:
            self._log(f"Spacegroup in [{min_sg}, {max_sg}]: {len(filtered)}/{original}")

        return filtered

    # =========================================================================
    # Combined Filters
    # =========================================================================

    def apply_filters(
        self,
        df: pd.DataFrame,
        e_hull_max: Optional[float] = None,
        formation_energy_range: Optional[Tuple[float, float]] = None,
        band_gap_range: Optional[Tuple[float, float]] = None,
        include_elements: Optional[List[str]] = None,
        exclude_elements: Optional[List[str]] = None,
        n_elements_range: Optional[Tuple[int, int]] = None,
        spacegroups: Optional[List[int]] = None,
        metals_only: bool = False,
        insulators_only: bool = False,
    ) -> pd.DataFrame:
        """
        Apply multiple filters at once.

        Args:
            df: Input DataFrame
            e_hull_max: Maximum e_above_hull (eV/atom)
            formation_energy_range: (min, max) formation energy
            band_gap_range: (min, max) band gap
            include_elements: Elements that must be present
            exclude_elements: Elements to exclude
            n_elements_range: (min, max) number of elements
            spacegroups: Allowed space groups
            metals_only: Keep only metals
            insulators_only: Keep only insulators

        Returns:
            Filtered DataFrame
        """
        self._log(f"\n{'='*60}")
        self._log(f"Applying filters to {len(df)} structures")
        self._log(f"{'='*60}")

        filtered = df.copy()

        if e_hull_max is not None:
            filtered = self.filter_by_e_hull(filtered, max_e_hull=e_hull_max)

        if formation_energy_range is not None:
            filtered = self.filter_by_formation_energy(
                filtered,
                min_fe=formation_energy_range[0],
                max_fe=formation_energy_range[1],
            )

        if metals_only:
            filtered = self.filter_metals(filtered)
        elif insulators_only:
            filtered = self.filter_insulators(filtered)
        elif band_gap_range is not None:
            filtered = self.filter_by_band_gap(
                filtered,
                min_gap=band_gap_range[0],
                max_gap=band_gap_range[1],
            )

        if include_elements is not None or exclude_elements is not None:
            filtered = self.filter_by_elements(
                filtered,
                include_elements=include_elements,
                exclude_elements=exclude_elements,
            )

        if n_elements_range is not None:
            filtered = self.filter_by_n_elements(
                filtered,
                min_elements=n_elements_range[0],
                max_elements=n_elements_range[1],
            )

        if spacegroups is not None:
            filtered = self.filter_by_spacegroup(filtered, spacegroups=spacegroups)

        self._log(f"{'='*60}")
        self._log(f"Final: {len(filtered)}/{len(df)} structures "
                  f"({100*len(filtered)/len(df):.1f}%)")
        self._log(f"{'='*60}\n")

        return filtered

    # =========================================================================
    # Preset Filters for Common Use Cases
    # =========================================================================

    def filter_stable(
        self,
        df: pd.DataFrame,
        e_hull_max: float = 0.0,
    ) -> pd.DataFrame:
        """
        Filter for thermodynamically stable materials (on hull).

        Args:
            df: Input DataFrame
            e_hull_max: Maximum e_hull (0.0 = on hull only)

        Returns:
            Filtered DataFrame
        """
        self._log("Filtering for STABLE materials (on hull)")
        return self.filter_by_e_hull(df, max_e_hull=e_hull_max)

    def filter_metastable(
        self,
        df: pd.DataFrame,
        e_hull_min: float = 0.0,
        e_hull_max: float = 0.1,
    ) -> pd.DataFrame:
        """
        Filter for metastable materials (above hull but accessible).

        Args:
            df: Input DataFrame
            e_hull_min: Minimum e_hull
            e_hull_max: Maximum e_hull

        Returns:
            Filtered DataFrame
        """
        self._log(f"Filtering for METASTABLE materials ({e_hull_min} < e_hull <= {e_hull_max})")

        column = "e_above_hull"
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        filtered = df[(df[column] > e_hull_min) & (df[column] <= e_hull_max)].copy()
        self._log(f"Metastable: {len(filtered)}/{len(df)}")

        return filtered

    def filter_practical(
        self,
        df: pd.DataFrame,
        e_hull_max: float = 0.1,
        exclude_radioactive: bool = True,
        exclude_noble_gases: bool = True,
    ) -> pd.DataFrame:
        """
        Filter for practical/accessible materials (stable + common elements).

        NOTE: This does NOT guarantee synthesizability. True synthesizability
        requires experimental data (ICSD). This just filters for:
        - Thermodynamic stability (e_hull)
        - Non-radioactive elements
        - Non-noble-gas elements

        Args:
            df: Input DataFrame
            e_hull_max: Maximum e_hull (eV/atom)
            exclude_radioactive: Exclude radioactive elements
            exclude_noble_gases: Exclude noble gases

        Returns:
            Filtered DataFrame
        """
        self._log("Filtering for PRACTICAL candidates (stable + common elements)")

        radioactive = ["Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
                       "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
                       "Fm", "Md", "No", "Lr"]

        noble_gases = ["He", "Ne", "Ar", "Kr", "Xe"]

        exclude = []
        if exclude_radioactive:
            exclude.extend(radioactive)
        if exclude_noble_gases:
            exclude.extend(noble_gases)

        filtered = self.filter_by_e_hull(df, max_e_hull=e_hull_max)

        if exclude:
            filtered = self.filter_by_elements(filtered, exclude_elements=exclude)

        return filtered

    # =========================================================================
    # Statistics
    # =========================================================================

    def print_statistics(self, df: pd.DataFrame):
        """Print statistics about the dataset."""
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)

        print(f"\nTotal structures: {len(df)}")

        # e_hull distribution
        if "e_above_hull" in df.columns:
            e_hull = df["e_above_hull"]
            print(f"\ne_above_hull (eV/atom):")
            print(f"  Min: {e_hull.min():.4f}")
            print(f"  Max: {e_hull.max():.4f}")
            print(f"  Mean: {e_hull.mean():.4f}")
            print(f"  Median: {e_hull.median():.4f}")
            print(f"  On hull (=0): {(e_hull == 0).sum()}")
            print(f"  <= 0.05: {(e_hull <= 0.05).sum()}")
            print(f"  <= 0.1: {(e_hull <= 0.1).sum()}")

        # Band gap distribution
        if "band_gap" in df.columns:
            gap = df["band_gap"]
            print(f"\nband_gap (eV):")
            print(f"  Metals (gap=0): {(gap == 0).sum()}")
            print(f"  Semiconductors (0<gap<3): {((gap > 0) & (gap < 3)).sum()}")
            print(f"  Insulators (gap>=3): {(gap >= 3).sum()}")

        # Elements
        if "elements" in df.columns:
            def count_elements(x):
                if isinstance(x, str):
                    try:
                        return len(ast.literal_eval(x))
                    except:
                        return len(x.split())
                return len(x)

            n_elem = df["elements"].apply(count_elements)
            print(f"\nNumber of elements:")
            for n in sorted(n_elem.unique()):
                print(f"  {n} elements: {(n_elem == n).sum()}")

        print("=" * 60 + "\n")


# =============================================================================
# Convenience Functions
# =============================================================================

def filter_mp20_stable(
    csv_path: str,
    e_hull_max: float = 0.1,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Quick filter of MP-20 train.csv by e_hull.

    Args:
        csv_path: Path to train.csv
        e_hull_max: Maximum e_hull (eV/atom)
        output_path: Optional path to save filtered CSV

    Returns:
        Filtered DataFrame
    """
    data_filter = DataFilter()
    df = data_filter.load_csv(csv_path)
    filtered = data_filter.filter_by_e_hull(df, max_e_hull=e_hull_max)

    if output_path:
        filtered.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return filtered


def get_material_ids_from_filtered(df: pd.DataFrame) -> List[str]:
    """Extract material IDs from filtered DataFrame."""
    return df["material_id"].tolist()


def get_cif_strings_from_filtered(df: pd.DataFrame) -> List[str]:
    """Extract CIF strings from filtered DataFrame."""
    return df["cif"].tolist()


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example demonstrating the filter module."""
    print("=" * 80)
    print("Data Filter Example")
    print("=" * 80)

    print("""
    # ==========================================================================
    # Basic Usage with train.csv
    # ==========================================================================

    from data_filter import DataFilter

    # Initialize filter
    f = DataFilter(verbose=True)

    # Load data
    df = f.load_csv("mp20/train.csv")
    # Columns: material_id, formation_energy_per_atom, band_gap, pretty_formula,
    #          e_above_hull, elements, cif, spacegroup.number, composition

    # See statistics
    f.print_statistics(df)

    # ==========================================================================
    # Filter by e_hull (main filter for stability)
    # ==========================================================================

    # e_hull <= 0.1 eV/atom (common threshold)
    stable_df = f.filter_by_e_hull(df, max_e_hull=0.1)

    # Stricter: only on-hull materials
    on_hull_df = f.filter_by_e_hull(df, max_e_hull=0.0)

    # ==========================================================================
    # Other Filters
    # ==========================================================================

    # Metals only
    metals_df = f.filter_metals(df)

    # Insulators only
    insulators_df = f.filter_insulators(df, min_gap=1.0)

    # Specific elements
    li_df = f.filter_by_elements(df, include_elements=["Li"])

    # Combined filters
    filtered_df = f.apply_filters(
        df,
        e_hull_max=0.1,
        band_gap_range=(0.5, 3.0),
        exclude_elements=["Tc", "Pm"],  # No radioactive
        n_elements_range=(2, 4),  # Binary to quaternary
    )

    # ==========================================================================
    # Use Filtered Data for Manifold
    # ==========================================================================

    from data_filter import get_material_ids_from_filtered, get_cif_strings_from_filtered

    material_ids = get_material_ids_from_filtered(stable_df)
    cif_strings = get_cif_strings_from_filtered(stable_df)

    print(f"Filtered to {len(material_ids)} materials for reference manifold")

    # ==========================================================================
    # Summary: Manifold Types from Computed Data
    # ==========================================================================

    # 1. FULL MANIFOLD: All MP-20 structures (no filter)
    #    → Defines "computational chemistry space"
    full_df = df

    # 2. STABLE MANIFOLD: e_hull <= threshold
    #    → Defines "thermodynamically accessible space"
    stable_df = f.filter_by_e_hull(df, max_e_hull=0.1)

    # 3. APPLICATION-SPECIFIC: e.g., battery cathodes
    #    → Defines "application-relevant space"
    battery_df = f.apply_filters(df, e_hull_max=0.1, include_elements=["Li", "O"])

    # For TRUE SYNTHESIZABILITY: Use ICSD data (separate dataset)
    # ICSD structures are experimentally verified synthesized materials
    """)


if __name__ == "__main__":
    example_usage()
