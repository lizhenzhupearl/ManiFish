# Changelog

All notable changes to ManiFish are documented here.

## [0.2.0] - 2025-01-27

### Added
- `print_neighbors()` method for ManifoldFishAnalyzer to display neighbor indices
- Export of aggregation functions: `aggregate_atom_embeddings`, `aggregate_from_structure_list`, `aggregate_with_weights`
- Comprehensive documentation with tutorials and API reference
- GitHub Pages documentation deployment

### Changed
- Project structure reorganized for clarity
- Documentation moved to Sphinx with RTD theme
- Examples cleaned up, old development code archived

### Fixed
- Export consistency between `manifish/__init__.py` and `manifish/core/__init__.py`

## [0.1.0] - 2024-12-16

### Added
- Initial release
- ManifoldFishAnalyzer with 7-category classification
- EnhancedManifoldFishAnalyzer with spatial metrics
- Platonic representation for cross-MLIP comparison
- Anchor selection (DIRECT, FPS methods)
- Spatial metrics (LDS, CDS, SRSS, RMSC)
- Distance-based metrics (Mutual KNN, CKA, Wasserstein, etc.)
- Ensemble analysis for multi-MLIP evaluation
- Aggregation utilities for atom-to-material conversion
- Visualization tools

## Future Plans

- [ ] Integration with ASE and pymatgen
- [ ] Pre-trained anchor sets for common datasets
- [ ] Interactive visualization dashboard
- [ ] GPU acceleration for large datasets
- [ ] Additional MLIP model support
