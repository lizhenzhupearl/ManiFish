# ManiFish Documentation

**ManiFish: Unified Manifold Framework for Cross-MLIP Materials Discovery**

Version: 0.2.0

## Overview

ManiFish is a Python framework for evaluating and comparing AI-generated crystal structures across different Machine Learning Interatomic Potentials (MLIPs) using a unified manifold representation.

### Key Innovation

The framework projects embeddings from different MLIPs into a common **anchor space** using cosine similarity (Platonic representation), enabling direct comparison across models.

### Main Features

- **Platonic Projection**: Model-agnostic embedding transformation
- **7-Category Fish Classification**: Intuitive structure categorization
- **Spatial Structure Metrics**: LDS, CDS, SRSS, RMSC for improved stability prediction
- **Multi-MLIP Ensemble Analysis**: Cross-model comparison and consensus
- **Rich Visualizations**: Category distributions, metric correlations, heatmaps

## Installation

```bash
pip install manifish
```

Or from source:
```bash
git clone https://github.com/your-repo/manifish.git
cd manifish
pip install -e .
```

### Dependencies

- numpy >= 1.20
- scipy >= 1.7
- scikit-learn >= 1.0
- matplotlib >= 3.5
- pandas >= 1.3

## Quick Start

```python
from manifish import EnhancedManifoldFishAnalyzer

# Load your transformed embeddings
# ref_emb: reference structures (known stable)
# gen_emb: generated structures to evaluate

analyzer = EnhancedManifoldFishAnalyzer(ref_emb, ref_ids)
results = analyzer.analyze(gen_emb, gen_ids)
analyzer.print_summary(results)
```

## Project Structure

```
manifish/
├── __init__.py              # Main package exports
├── core/                    # Core functionality
│   ├── anchors.py          # Anchor selection (DIRECT, FPS)
│   ├── manifold.py         # Platonic projection
│   ├── metrics.py          # Distance-based metrics
│   ├── spatial_metrics.py  # Spatial structure metrics
│   ├── fish_analyzer.py    # Base 7-category classifier
│   ├── enhanced_fish_analyzer.py  # Enhanced analyzer with spatial metrics
│   ├── structure_evaluator.py     # High-level evaluation interface
│   ├── enhanced_evaluator.py      # Combined distance + spatial evaluator
│   ├── transforms.py       # Embedding transformations
│   ├── ensemble.py         # Multi-MLIP ensemble analysis
│   └── data_filter.py      # Data filtering utilities
├── visualization/          # Visualization tools
│   ├── __init__.py
│   └── fish_spatial_viz.py # Fish category + spatial metrics plots
└── examples/               # Usage examples
    ├── quick_start.py
    └── batch_evaluation.py
```

## Documentation Sections

- [API Reference](api/index.md) - Detailed API documentation
- [User Guide](guide/index.md) - Step-by-step tutorials
- [Concepts](concepts/index.md) - Theoretical background
- [Examples](examples/index.md) - Code examples

## License

MIT License
