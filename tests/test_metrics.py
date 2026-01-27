"""
Tests for metric calculations.
"""

import pytest
import numpy as np
from manifish.core.metrics import (
    compute_manifold_distance,
    compute_stability,
    compute_novelty,
)


def test_manifold_distance():
    """Test manifold distance calculation."""
    coords = np.array([1.0, 2.0, 3.0])
    manifold_points = np.array([
        [1.1, 2.1, 3.1],
        [2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5],
    ])
    
    distance = compute_manifold_distance(coords, manifold_points)
    assert isinstance(distance, float)
    assert distance >= 0


def test_stability_score():
    """Test stability score calculation."""
    coords = np.array([1.0, 2.0, 3.0])
    manifold_points = np.array([
        [1.1, 2.1, 3.1],
        [2.0, 3.0, 4.0],
    ])
    
    stability = compute_stability(coords, manifold_points)
    assert 0 <= stability <= 1


def test_novelty_score():
    """Test novelty score calculation."""
    coords = np.array([1.0, 2.0, 3.0])
    known_points = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
    ])
    
    novelty = compute_novelty(coords, known_points, method='distance')
    assert 0 <= novelty <= 1
