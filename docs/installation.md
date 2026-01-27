# Installation Guide

## Requirements

- Python 3.8 or higher
- pip or conda

## Basic Installation

### From PyPI (coming soon)

```bash
pip install manifish
```

### From Source

```bash
git clone https://github.com/yourusername/manifish.git
cd manifish
pip install -e .
```

## Optional Dependencies

### MLIP Support

To use specific MLIPs:

```bash
# CHGNet
pip install manifish[mlip]

# Or install individually
pip install chgnet
pip install mace-torch
pip install matgl  # for M3GNet
```

### Visualization

```bash
pip install manifish[viz]
```

### Development

```bash
pip install manifish[dev]
```

## Verification

Test your installation:

```python
import manifish as mf
print(mf.__version__)
```

Run the examples:

```bash
python examples/quick_start.py
```

## Troubleshooting

### Common Issues

**Issue: MLIP not found**
```bash
# Make sure MLIP packages are installed
pip install chgnet mace-torch matgl
```

**Issue: ASE not found**
```bash
pip install ase
```

For more help, see [GitHub Issues](https://github.com/yourusername/manifish/issues).
