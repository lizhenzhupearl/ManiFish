"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_anchors():
    """Sample anchor structures for testing."""
    # TODO: Create mock Anchor objects
    return []


@pytest.fixture
def sample_structure():
    """Sample ASE Atoms structure for testing."""
    # TODO: Create mock structure
    return None


@pytest.fixture
def sample_mlips():
    """Sample MLIP models for testing."""
    return ['chgnet', 'mace', 'm3gnet']
