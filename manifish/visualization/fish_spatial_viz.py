"""
Visualization module for linking Fish Categories with Spatial Metrics.

Provides comprehensive visualizations to understand the relationship between
the 7 fish categories and spatial structure metrics (LDS, CDS, SRSS, RMSC).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Import category definitions
CATEGORIES = {
    "redundant_fish": {"color": "#95a5a6", "short": "Redundant"},
    "fish_in_water": {"color": "#2ecc71", "short": "In Water"},
    "frontier_fish": {"color": "#3498db", "short": "Frontier"},
    "edge_fish": {"color": "#f39c12", "short": "Edge"},
    "adventurous_fish": {"color": "#9b59b6", "short": "Adventurous"},
    "geometric_atypical": {"color": "#e74c3c", "short": "Atypical"},
    "structural_hallucination": {"color": "#2c3e50", "short": "Hallucination"},
}

CATEGORY_ORDER = [
    "redundant_fish", "fish_in_water", "frontier_fish", "edge_fish",
    "adventurous_fish", "geometric_atypical", "structural_hallucination"
]

SPATIAL_METRICS = ["lds", "cds", "srss", "rmsc", "spatial_quality"]
METRIC_LABELS = {
    "lds": "LDS\n(Local vs Distant)",
    "cds": "CDS\n(Correlation Decay)",
    "srss": "SRSS\n(Region Similarity)",
    "rmsc": "RMSC\n(Spatial Contrast)",
    "spatial_quality": "Spatial\nQuality",
}


def _extract_metrics(results) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract spatial metrics grouped by category (vectorized)."""
    # Pre-allocate arrays for speed
    n = len(results)
    categories = np.empty(n, dtype=object)
    lds = np.empty(n)
    cds = np.empty(n)
    srss = np.empty(n)
    rmsc = np.empty(n)
    spatial_quality = np.empty(n)
    confidence = np.empty(n)
    enhanced_confidence = np.empty(n)
    manifold_distance = np.empty(n)

    for i, r in enumerate(results):
        categories[i] = r.category
        lds[i] = r.spatial_metrics.lds
        cds[i] = r.spatial_metrics.cds
        srss[i] = r.spatial_metrics.srss
        rmsc[i] = r.spatial_metrics.rmsc
        spatial_quality[i] = r.spatial_metrics.spatial_quality
        confidence[i] = r.confidence
        enhanced_confidence[i] = r.enhanced_confidence
        manifold_distance[i] = r.manifold_distance

    # Group by category using numpy indexing
    metrics_by_cat = {}
    for cat in CATEGORY_ORDER:
        mask = categories == cat
        if np.any(mask):
            metrics_by_cat[cat] = {
                "lds": lds[mask],
                "cds": cds[mask],
                "srss": srss[mask],
                "rmsc": rmsc[mask],
                "spatial_quality": spatial_quality[mask],
                "confidence": confidence[mask],
                "enhanced_confidence": enhanced_confidence[mask],
                "manifold_distance": manifold_distance[mask],
            }

    return metrics_by_cat


def plot_category_spatial_overview(
    results,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
):
    """
    Create comprehensive overview plot linking categories with spatial metrics.

    Creates a 2x2 layout with:
    - Box plots of spatial metrics by category
    - Scatter plot of LDS vs SRSS colored by category
    - Radar chart of metric profiles
    - Category distribution pie chart

    Args:
        results: List of EnhancedManifoldFishResult objects
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Extract metrics once and reuse
    metrics_by_cat = _extract_metrics(results)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Box plots of spatial metrics by category
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_metric_boxplots(ax1, metrics_by_cat, "spatial_quality")
    ax1.set_title("Spatial Quality by Fish Category", fontsize=12, fontweight='bold')

    # 2. Scatter plot: LDS vs SRSS (pass metrics_by_cat to avoid re-extraction)
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_scatter_metrics(ax2, results, "lds", "srss", metrics_by_cat=metrics_by_cat)
    ax2.set_title("LDS vs SRSS by Category", fontsize=12, fontweight='bold')

    # 3. Radar chart of metric profiles
    ax3 = fig.add_subplot(gs[1, 0], projection='polar')
    _plot_radar_profiles(ax3, metrics_by_cat)
    ax3.set_title("Spatial Metric Profiles by Category", fontsize=12, fontweight='bold', pad=20)

    # 4. Category distribution with mean spatial quality
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_category_distribution(ax4, results, metrics_by_cat)
    ax4.set_title("Category Distribution & Spatial Quality", fontsize=12, fontweight='bold')

    plt.suptitle("Fish Categories vs Spatial Metrics Analysis", fontsize=14, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.tight_layout()
    return fig


def plot_spatial_metrics_grid(
    results,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
):
    """
    Create grid of box plots for all spatial metrics by category.

    Args:
        results: List of EnhancedManifoldFishResult objects
        figsize: Figure size
        save_path: Optional path to save
    """
    metrics_by_cat = _extract_metrics(results)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    metrics = ["lds", "cds", "srss", "rmsc", "spatial_quality"]
    titles = ["Local vs Distant Similarity (LDS)",
              "Correlation Decay Slope (CDS)",
              "Semantic-Region Self-Similarity (SRSS)",
              "RMS Spatial Contrast (RMSC)",
              "Combined Spatial Quality"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        _plot_metric_boxplots(axes[idx], metrics_by_cat, metric)
        axes[idx].set_title(title, fontsize=10, fontweight='bold')

    # Hide the 6th subplot
    axes[5].axis('off')

    # Add legend in the empty subplot
    legend_elements = [
        Patch(facecolor=CATEGORIES[cat]["color"], label=CATEGORIES[cat]["short"])
        for cat in CATEGORY_ORDER
    ]
    axes[5].legend(handles=legend_elements, loc='center', ncol=2, fontsize=9)
    axes[5].set_title("Legend", fontsize=10, fontweight='bold')

    plt.suptitle("Spatial Metrics Distribution by Fish Category", fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_pairwise_scatter(
    results,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
):
    """
    Create pairwise scatter plots of spatial metrics colored by category.

    Args:
        results: List of EnhancedManifoldFishResult objects
        figsize: Figure size
        save_path: Optional path to save
    """
    # Extract metrics once and reuse
    metrics_by_cat = _extract_metrics(results)

    metrics = ["lds", "cds", "srss", "rmsc"]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=figsize)

    # Pre-compute all metric arrays for correlation
    all_metrics = {}
    for m in metrics:
        all_metrics[m] = np.concatenate([metrics_by_cat[cat][m] for cat in metrics_by_cat])

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                _plot_metric_histogram(ax, results, m1, metrics_by_cat)
            elif i > j:
                # Lower triangle: scatter
                _plot_scatter_metrics(ax, results, m2, m1, show_legend=False, metrics_by_cat=metrics_by_cat)
            else:
                # Upper triangle: correlation coefficient
                ax.axis('off')
                corr = np.corrcoef(all_metrics[m1], all_metrics[m2])[0, 1]
                ax.text(0.5, 0.5, f"r = {corr:.2f}",
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, fontweight='bold')

            # Labels
            if i == n_metrics - 1:
                ax.set_xlabel(m2.upper())
            if j == 0 and i != j:
                ax.set_ylabel(m1.upper())

    # Add legend
    legend_elements = [
        Patch(facecolor=CATEGORIES[cat]["color"], label=CATEGORIES[cat]["short"])
        for cat in CATEGORY_ORDER if cat in metrics_by_cat
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle("Pairwise Spatial Metrics by Fish Category", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_confidence_comparison(
    results,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    max_points: int = 5000,
):
    """
    Compare original confidence vs enhanced confidence by category.

    Shows how spatial metrics influence confidence scores.

    Args:
        results: List of EnhancedManifoldFishResult objects
        figsize: Figure size
        save_path: Optional path to save
        max_points: Max points per category in scatter plot
    """
    metrics_by_cat = _extract_metrics(results)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Confidence comparison
    ax1 = axes[0]
    categories = [cat for cat in CATEGORY_ORDER if cat in metrics_by_cat and len(metrics_by_cat[cat]["confidence"]) > 0]
    x = np.arange(len(categories))
    width = 0.35

    orig_means = [np.mean(metrics_by_cat[cat]["confidence"]) for cat in categories]
    enh_means = [np.mean(metrics_by_cat[cat]["enhanced_confidence"]) for cat in categories]

    bars1 = ax1.bar(x - width/2, orig_means, width, label='Original', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, enh_means, width, label='Enhanced', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Mean Confidence')
    ax1.set_xticks(x)
    ax1.set_xticklabels([CATEGORIES[cat]["short"] for cat in categories], rotation=45, ha='right')
    ax1.legend()
    ax1.set_title("Confidence: Original vs Enhanced", fontweight='bold')
    ax1.set_ylim(0, 1.1)

    # Right: Scatter of original vs enhanced (vectorized)
    ax2 = axes[1]
    for cat in CATEGORY_ORDER:
        if cat not in metrics_by_cat:
            continue
        conf = metrics_by_cat[cat]["confidence"]
        enh_conf = metrics_by_cat[cat]["enhanced_confidence"]
        if len(conf) == 0:
            continue

        # Downsample if needed
        if len(conf) > max_points:
            idx = np.random.choice(len(conf), max_points, replace=False)
            conf = conf[idx]
            enh_conf = enh_conf[idx]

        ax2.scatter(conf, enh_conf, c=CATEGORIES[cat]["color"], alpha=0.5, s=20)

    # Diagonal line
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')
    ax2.set_xlabel("Original Confidence")
    ax2.set_ylabel("Enhanced Confidence")
    ax2.set_title("Confidence Enhancement by Spatial Metrics", fontweight='bold')
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    # Legend
    legend_elements = [
        Patch(facecolor=CATEGORIES[cat]["color"], label=CATEGORIES[cat]["short"])
        for cat in CATEGORY_ORDER if cat in metrics_by_cat
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_manifold_vs_spatial(
    results,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    max_points: int = 5000,
):
    """
    Compare manifold distance with spatial quality by category.

    Shows how traditional distance metrics relate to spatial structure.

    Args:
        results: List of EnhancedManifoldFishResult objects
        figsize: Figure size
        save_path: Optional path to save
        max_points: Max points per category in scatter plots
    """
    metrics_by_cat = _extract_metrics(results)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Left: Manifold distance vs spatial quality (vectorized)
    ax1 = axes[0]
    for cat in CATEGORY_ORDER:
        if cat not in metrics_by_cat:
            continue
        m_dist = metrics_by_cat[cat]["manifold_distance"]
        sq = metrics_by_cat[cat]["spatial_quality"]
        if len(m_dist) == 0:
            continue

        # Downsample if needed
        if len(m_dist) > max_points:
            idx = np.random.choice(len(m_dist), max_points, replace=False)
            m_dist = m_dist[idx]
            sq = sq[idx]

        ax1.scatter(m_dist, sq, c=CATEGORIES[cat]["color"], alpha=0.5, s=20)

    ax1.set_xlabel("Manifold Distance (normalized)")
    ax1.set_ylabel("Spatial Quality")
    ax1.set_title("Manifold Distance vs Spatial Quality", fontweight='bold')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Middle: LDS vs Manifold distance (vectorized)
    ax2 = axes[1]
    for cat in CATEGORY_ORDER:
        if cat not in metrics_by_cat:
            continue
        m_dist = metrics_by_cat[cat]["manifold_distance"]
        lds = metrics_by_cat[cat]["lds"]
        if len(m_dist) == 0:
            continue

        # Downsample if needed
        if len(m_dist) > max_points:
            idx = np.random.choice(len(m_dist), max_points, replace=False)
            m_dist = m_dist[idx]
            lds = lds[idx]

        ax2.scatter(m_dist, lds, c=CATEGORIES[cat]["color"], alpha=0.5, s=20)

    ax2.set_xlabel("Manifold Distance")
    ax2.set_ylabel("LDS (Local vs Distant)")
    ax2.set_title("Manifold Distance vs LDS", fontweight='bold')

    # Right: Category comparison
    ax3 = axes[2]
    categories = [cat for cat in CATEGORY_ORDER if cat in metrics_by_cat and len(metrics_by_cat[cat]["manifold_distance"]) > 0]

    x = np.arange(len(categories))
    width = 0.35

    dist_means = [np.mean(metrics_by_cat[cat]["manifold_distance"]) for cat in categories]
    sq_means = [np.mean(metrics_by_cat[cat]["spatial_quality"]) for cat in categories]

    # Normalize for comparison
    dist_norm = (np.array(dist_means) - min(dist_means)) / (max(dist_means) - min(dist_means) + 1e-10)
    sq_norm = (np.array(sq_means) - min(sq_means)) / (max(sq_means) - min(sq_means) + 1e-10)

    ax3.bar(x - width/2, dist_norm, width, label='Manifold Dist', color='#3498db', alpha=0.8)
    ax3.bar(x + width/2, sq_norm, width, label='Spatial Quality', color='#e74c3c', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([CATEGORIES[cat]["short"] for cat in categories], rotation=45, ha='right')
    ax3.set_ylabel("Normalized Score")
    ax3.set_title("Category Comparison", fontweight='bold')
    ax3.legend()

    # Add legend
    legend_elements = [
        Patch(facecolor=CATEGORIES[cat]["color"], label=CATEGORIES[cat]["short"])
        for cat in CATEGORY_ORDER if cat in metrics_by_cat
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=7, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_category_heatmap(
    results,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Create heatmap showing mean spatial metrics for each category.

    Args:
        results: List of EnhancedManifoldFishResult objects
        figsize: Figure size
        save_path: Optional path to save
    """
    metrics_by_cat = _extract_metrics(results)

    # Build data matrix
    categories = [cat for cat in CATEGORY_ORDER if cat in metrics_by_cat and len(metrics_by_cat[cat]["lds"]) > 0]
    metrics = ["lds", "cds", "srss", "rmsc", "spatial_quality"]

    data = np.zeros((len(categories), len(metrics)))
    for i, cat in enumerate(categories):
        for j, metric in enumerate(metrics):
            data[i, j] = np.mean(metrics_by_cat[cat][metric])

    # Normalize each column for better visualization
    data_norm = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=9)
    ax.set_yticklabels([CATEGORIES[cat]["short"] for cat in categories])

    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=9)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Value (z-score)", rotation=-90, va="bottom")

    ax.set_title("Mean Spatial Metrics by Fish Category", fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


# =============================================================================
# Helper Functions
# =============================================================================

def _plot_metric_boxplots(ax, metrics_by_cat: Dict, metric: str):
    """Plot box plots for a single metric across categories."""
    categories = [cat for cat in CATEGORY_ORDER if cat in metrics_by_cat and len(metrics_by_cat[cat][metric]) > 0]

    data = [metrics_by_cat[cat][metric] for cat in categories]
    colors = [CATEGORIES[cat]["color"] for cat in categories]

    bp = ax.boxplot(data, patch_artist=True, labels=[CATEGORIES[cat]["short"] for cat in categories])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric.upper())
    ax.tick_params(axis='x', rotation=45)


def _plot_scatter_metrics(ax, results, x_metric: str, y_metric: str, show_legend: bool = True,
                          metrics_by_cat: Optional[Dict] = None, max_points: int = 5000):
    """Plot scatter of two metrics colored by category (vectorized)."""
    if metrics_by_cat is None:
        metrics_by_cat = _extract_metrics(results)

    present_cats = []
    for cat in CATEGORY_ORDER:
        if cat not in metrics_by_cat:
            continue
        x_vals = metrics_by_cat[cat][x_metric]
        y_vals = metrics_by_cat[cat][y_metric]
        if len(x_vals) == 0:
            continue

        present_cats.append(cat)

        # Downsample if too many points for performance
        if len(x_vals) > max_points:
            idx = np.random.choice(len(x_vals), max_points, replace=False)
            x_vals = x_vals[idx]
            y_vals = y_vals[idx]

        ax.scatter(x_vals, y_vals, c=CATEGORIES[cat]["color"], alpha=0.5, s=20, label=cat)

    ax.set_xlabel(x_metric.upper())
    ax.set_ylabel(y_metric.upper())

    if show_legend:
        legend_elements = [
            Patch(facecolor=CATEGORIES[cat]["color"], label=CATEGORIES[cat]["short"])
            for cat in present_cats
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=8)


def _plot_metric_histogram(ax, results, metric: str, metrics_by_cat: Optional[Dict] = None):
    """Plot histogram for a single metric with category coloring (vectorized)."""
    if metrics_by_cat is None:
        metrics_by_cat = _extract_metrics(results)

    for cat in CATEGORY_ORDER:
        if cat not in metrics_by_cat:
            continue
        values = metrics_by_cat[cat][metric]
        if len(values) > 0:
            ax.hist(values, bins=20, alpha=0.5, color=CATEGORIES[cat]["color"], label=CATEGORIES[cat]["short"])

    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Count")


def _plot_radar_profiles(ax, metrics_by_cat: Dict):
    """Plot radar/spider chart of metric profiles."""
    metrics = ["lds", "cds", "srss", "rmsc"]
    n_metrics = len(metrics)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    categories = [cat for cat in CATEGORY_ORDER if cat in metrics_by_cat and len(metrics_by_cat[cat]["lds"]) > 0]

    for cat in categories:
        values = [np.mean(metrics_by_cat[cat][m]) for m in metrics]
        # Normalize to [0, 1] for radar
        values = [(v - min(values)) / (max(values) - min(values) + 1e-10) for v in values]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=CATEGORIES[cat]["short"],
               color=CATEGORIES[cat]["color"])
        ax.fill(angles, values, alpha=0.15, color=CATEGORIES[cat]["color"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)


def _plot_category_distribution(ax, results, metrics_by_cat: Dict):
    """Plot category distribution as bar chart with spatial quality overlay."""
    categories = [cat for cat in CATEGORY_ORDER if cat in metrics_by_cat and len(metrics_by_cat[cat]["lds"]) > 0]

    counts = [len(metrics_by_cat[cat]["lds"]) for cat in categories]
    colors = [CATEGORIES[cat]["color"] for cat in categories]
    sq_means = [np.mean(metrics_by_cat[cat]["spatial_quality"]) for cat in categories]

    x = np.arange(len(categories))

    # Bar chart for counts
    bars = ax.bar(x, counts, color=colors, alpha=0.7)
    ax.set_ylabel("Count", color='black')
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORIES[cat]["short"] for cat in categories], rotation=45, ha='right')

    # Overlay line for spatial quality
    ax2 = ax.twinx()
    ax2.plot(x, sq_means, 'ro-', linewidth=2, markersize=8, label='Mean Spatial Quality')
    ax2.set_ylabel("Spatial Quality", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha='center', va='bottom', fontsize=9)


def create_full_report(
    results,
    output_dir: str = ".",
    prefix: str = "fish_spatial",
):
    """
    Generate all visualization plots and save to directory.

    Args:
        results: List of EnhancedManifoldFishResult objects
        output_dir: Directory to save plots
        prefix: Filename prefix
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating visualizations in {output_dir}/...")

    # 1. Overview
    plot_category_spatial_overview(results, save_path=f"{output_dir}/{prefix}_overview.png")

    # 2. Metrics grid
    plot_spatial_metrics_grid(results, save_path=f"{output_dir}/{prefix}_metrics_grid.png")

    # 3. Pairwise scatter
    plot_pairwise_scatter(results, save_path=f"{output_dir}/{prefix}_pairwise.png")

    # 4. Confidence comparison
    plot_confidence_comparison(results, save_path=f"{output_dir}/{prefix}_confidence.png")

    # 5. Manifold vs spatial
    plot_manifold_vs_spatial(results, save_path=f"{output_dir}/{prefix}_manifold_vs_spatial.png")

    # 6. Heatmap
    plot_category_heatmap(results, save_path=f"{output_dir}/{prefix}_heatmap.png")

    print(f"Generated 6 visualization files with prefix '{prefix}_'")
