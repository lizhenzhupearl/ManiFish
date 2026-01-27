# Installation

## Requirements

- Python 3.9+
- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib (optional, for visualization)

## Install from PyPI

```bash
pip install manifish
```

## Install from Source

```bash
git clone https://github.com/lizhenzhupearl/ManiFish.git
cd ManiFish
pip install -e .
```

## Install with Development Dependencies

```bash
git clone https://github.com/lizhenzhupearl/ManiFish.git
cd ManiFish
pip install -e ".[dev]"
```

## Optional Dependencies

### For MLIP Embedding Extraction

To extract embeddings from MLIP models, install the corresponding packages:

```bash
# For MACE
pip install mace-torch

# For CHGNet
pip install chgnet

# For ORB
pip install orb-models

# For SevenNet
pip install sevenn
```

### For Visualization

```bash
pip install matplotlib seaborn plotly
```

## Verify Installation

```python
import manifish
print(manifish.__version__)

# Test basic import
from manifish import ManifoldFishAnalyzer, aggregate_atom_embeddings
print("Installation successful!")
```

## Troubleshooting

### Import Errors

If you encounter import errors related to `torch`, make sure you have PyTorch installed:

```bash
pip install torch
```

### Memory Issues

For large datasets, consider using the `local_geometry_mode="fast"` option:

```python
analyzer = ManifoldFishAnalyzer(
    reference_embeddings,
    local_geometry_mode="fast",  # Reduces memory usage
)
```
