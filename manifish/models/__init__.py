"""
MLIP model interfaces for embedding extraction.

This module provides unified interfaces for extracting embeddings from
different Machine Learning Interatomic Potentials (MLIPs).

Supported models:
- MACE (using mace-torch)
- Orb (using orb-models)
- SevenNet (using sevenn)
"""

from .mace import MACEEmbeddingExtractor
from .orb import OrbEmbeddingExtractor
from .sevennet import SevenNetEmbeddingExtractor
from .base import BaseEmbeddingExtractor

__all__ = [
    'BaseEmbeddingExtractor',
    'MACEEmbeddingExtractor',
    'OrbEmbeddingExtractor',
    'SevenNetEmbeddingExtractor',
]
