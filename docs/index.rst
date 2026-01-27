ManiFish Documentation
======================

**ManiFish** is a unified manifold framework for cross-MLIP materials discovery.
It provides tools for evaluating and comparing AI-generated crystal structures
across different Machine Learning Interatomic Potentials (MLIPs).

Key Features
------------

- **ManifoldFishAnalyzer**: 7-category classifier for generated structures
- **Platonic Representation**: Cross-MLIP comparison using anchor-based projection
- **Spatial Metrics**: LDS, CDS, SRSS, RMSC for stability prediction
- **Aggregation Utilities**: Convert atom-level to material-level embeddings
- **Ensemble Analysis**: Multi-MLIP evaluation framework

Quick Start
-----------

.. code-block:: python

   from manifish import (
       ManifoldFishAnalyzer,
       aggregate_atom_embeddings,
       aggregate_by_material_id,
   )

   # Aggregate atom embeddings to material level
   material_embeddings, material_ids = aggregate_atom_embeddings(
       atom_embeddings, atom_to_material, method="mean"
   )

   # Analyze generated materials against reference manifold
   analyzer = ManifoldFishAnalyzer(
       reference_embeddings,
       reference_ids,
       boundary_method="alpha_shape"
   )
   results = analyzer.analyze(generated_embeddings, generated_ids)
   analyzer.print_summary(results)

   # Print neighbors for materials
   neighbor_indices = analyzer.print_neighbors(generated_embeddings, generated_ids)

Installation
------------

.. code-block:: bash

   pip install manifish

Or from source:

.. code-block:: bash

   git clone https://github.com/lizhenzhupearl/ManiFish.git
   cd ManiFish
   pip install -e .

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   guide/index
   tutorials/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/fish_analyzer
   api/aggregation
   api/spatial_metrics

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
