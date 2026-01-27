# Contributing

We welcome contributions to ManiFish! This guide will help you get started.

## Ways to Contribute

- **Report bugs**: Open an issue describing the problem
- **Request features**: Open an issue with your idea
- **Fix bugs**: Submit a pull request
- **Add features**: Discuss first in an issue, then submit PR
- **Improve docs**: Documentation improvements are always welcome

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ManiFish.git
cd ManiFish
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public functions (NumPy style)
- Keep functions focused and well-named

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add your feature description"
```

3. Push and create a PR:
```bash
git push origin feature/your-feature-name
```

4. Fill in the PR template and wait for review

## Testing

- Add tests for new features
- Ensure all existing tests pass
- Aim for good coverage of edge cases

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=manifish

# Run specific test file
pytest tests/test_fish_analyzer.py
```

## Documentation

- Update docstrings when changing function signatures
- Add tutorials for new features
- Keep the API reference up to date

Build docs locally:
```bash
cd docs
make html
# Open _build/html/index.html
```

## Questions?

Open an issue or reach out to the maintainers.
