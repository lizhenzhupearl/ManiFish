# 🐟 ManiFish Project Setup Guide

This document explains how to set up and use the ManiFish repository.

## 📁 Project Structure

```
manifish/
├── manifish/              # Main package
│   ├── __init__.py       # Package initialization
│   ├── core/             # Core functionality
│   │   ├── anchors.py    # Anchor selection
│   │   ├── manifold.py   # Manifold construction
│   │   └── metrics.py    # Evaluation metrics
│   ├── models/           # MLIP interfaces
│   ├── utils/            # Utility functions
│   ├── data/             # Data handling
│   └── visualization/    # Plotting functions
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── examples/            # Example scripts
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── pyproject.toml       # Package configuration
├── README.md            # Main documentation
└── LICENSE              # MIT License
```

## 🚀 Quick Start

### 1. Clone/Initialize Repository

```bash
# If using git
git init
git add .
git commit -m "Initial commit: ManiFish project structure"

# Create GitHub repository (on GitHub website)
# Then connect local to remote
git remote add origin https://github.com/yourusername/manifish.git
git branch -M main
git push -u origin main
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 3. Run Tests

```bash
pytest tests/
```

### 4. Try Examples

```bash
python examples/quick_start.py
```

## 📝 Development Workflow

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your feature**
   - Add code to appropriate module
   - Add tests in `tests/`
   - Update documentation

3. **Run tests**
   ```bash
   pytest tests/
   black manifish/  # Format code
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

### Code Style

- **Formatting**: Use Black (`black manifish/`)
- **Linting**: Use Flake8 (`flake8 manifish/`)
- **Type hints**: Required for all functions
- **Docstrings**: NumPy style

Example:
```python
def compute_stability(
    coords: np.ndarray,
    manifold_points: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Compute stability score based on manifold distance.
    
    Parameters
    ----------
    coords : np.ndarray
        Point in anchor space, shape (n_anchors,)
    manifold_points : np.ndarray
        Known stable points, shape (n_points, n_anchors)
    threshold : float, optional
        Distance threshold, by default 0.5
        
    Returns
    -------
    float
        Stability score in [0, 1], higher = more stable
        
    Examples
    --------
    >>> stability = compute_stability(coords, manifold_points)
    >>> print(f"Stability: {stability:.3f}")
    """
    # Implementation
    pass
```

## 🔧 Key Implementation Priorities

### Phase 1: Core Functionality (Current)
- [x] Project structure
- [ ] Anchor selection implementation
- [ ] MLIP interface (CHGNet, MACE, M3GNet)
- [ ] Manifold construction
- [ ] Basic metrics (stability, novelty)

### Phase 2: Advanced Features
- [ ] Multi-MLIP consensus
- [ ] Batch evaluation
- [ ] Visualization tools
- [ ] Database integration (Materials Project)

### Phase 3: Publication Ready
- [ ] Complete documentation
- [ ] Comprehensive tests (>80% coverage)
- [ ] Benchmark results
- [ ] Paper integration

## 📦 Publishing to PyPI

When ready to publish:

```bash
# Build package
python -m build

# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ manifish

# Upload to PyPI
python -m twine upload dist/*
```

## 🤝 Using with Claude Code

You can use Claude Code to help develop this project:

```bash
# In your terminal
claude

# Then in Claude Code
"Help me implement the anchor selection using FPS algorithm"
"Add tests for the manifold distance calculation"
"Create a visualization function for the fish-water analogy"
```

## 📊 Current Status

### Completed ✅
- Project structure
- Package configuration
- Documentation skeleton
- Example scripts
- Test framework
- Logo design

### TODO 🔨
- Implement core algorithms
- Add MLIP interfaces
- Complete test coverage
- Add visualization functions
- Write tutorials
- Benchmark on real data

## 🎯 Next Steps

1. **Implement Anchor Selection**
   - FPS algorithm
   - Materials Project integration
   - Descriptor calculation

2. **Implement MLIP Interfaces**
   - CHGNet wrapper
   - MACE wrapper
   - M3GNet wrapper

3. **Build Manifold**
   - Convex hull construction
   - Distance metrics
   - Stability assessment

4. **Add Visualization**
   - Manifold plots
   - Fish-water diagrams
   - MLIP comparison plots

## 📚 Resources

- [Materials Project API](https://materialsproject.org/)
- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
- [CHGNet](https://github.com/CederGroupHub/chgnet)
- [MACE](https://github.com/ACEsuit/mace)
- [M3GNet](https://github.com/materialsvirtuallab/matgl)

## 💡 Tips

- Start with small test cases
- Use mock data during development
- Document as you code
- Write tests before implementation (TDD)
- Keep commits small and focused

## 🐛 Debugging

If you encounter issues:

1. Check Python version (>= 3.8)
2. Verify all dependencies installed
3. Run tests to identify problems
4. Check GitHub issues
5. Ask Claude Code for help!

---

**Happy coding! 🚀**

Let me know if you need help with any specific part!
