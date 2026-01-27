"""
Visualization module for ManiFish.

Provides plotting functions for:
- Fish category analysis with spatial metrics
- Manifold structure visualization
- Cross-model comparison
"""

from .fish_spatial_viz import (
    plot_category_spatial_overview,
    plot_spatial_metrics_grid,
    plot_pairwise_scatter,
    plot_confidence_comparison,
    plot_manifold_vs_spatial,
    plot_category_heatmap,
    create_full_report,
)

__all__ = [
    "plot_category_spatial_overview",
    "plot_spatial_metrics_grid",
    "plot_pairwise_scatter",
    "plot_confidence_comparison",
    "plot_manifold_vs_spatial",
    "plot_category_heatmap",
    "create_full_report",
]
