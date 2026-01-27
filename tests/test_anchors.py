"""
Tests for anchor selection functionality.
"""

import pytest
import numpy as np
from manifish.core.anchors import AnchorSelector, select_anchors


def test_anchor_selector_initialization():
    """Test AnchorSelector initialization."""
    selector = AnchorSelector(
        database='materials_project',
        n_anchors=100,
        stability_threshold=0.1
    )
    assert selector.n_anchors == 100
    assert selector.stability_threshold == 0.1


def test_select_anchors_function():
    """Test convenience function."""
    anchors = select_anchors(n_anchors=50, method='fps')
    # Currently returns empty list (placeholder)
    assert isinstance(anchors, list)


@pytest.mark.parametrize("method", ['fps', 'maxmin', 'determinant'])
def test_selection_methods(method):
    """Test different selection methods."""
    selector = AnchorSelector(n_anchors=10)
    anchors = selector.select(method=method)
    assert isinstance(anchors, list)
