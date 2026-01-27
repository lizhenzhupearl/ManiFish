ManiFish Documentation
======================

**Unified Manifold Framework for Cross-MLIP Materials Discovery**

ManiFish provides tools for evaluating and comparing AI-generated crystal structures
across different Machine Learning Interatomic Potentials (MLIPs) using manifold-based analysis.

.. image:: images/manifish_clean.png
   :width: 600
   :alt: ManiFish Framework Overview
   :align: center

Key Features
------------

.. grid:: 2

   .. grid-item-card:: 7-Category Fish Classifier
      :link: api/fish_analyzer
      :link-type: doc

      Classify generated structures into 7 categories based on their position
      relative to the reference manifold: from "Fish in Water" (reliable) to
      "Structural Hallucination" (likely unphysical).

   .. grid-item-card:: Cross-MLIP Comparison
      :link: concepts/index
      :link-type: doc

      Project embeddings from different MLIPs (MACE, CHGNet, ORB, SevenNet)
      into a common anchor-based "Platonic" space for direct comparison.

   .. grid-item-card:: Spatial Metrics
      :link: api/spatial_metrics
      :link-type: doc

      Analyze local manifold structure with LDS, CDS, SRSS, and RMSC metrics
      for improved stability prediction.

   .. grid-item-card:: Aggregation Utilities
      :link: api/aggregation
      :link-type: doc

      Convert atom-level MLIP embeddings to material-level representations
      using mean, weighted, or other pooling methods.

Quick Example
-------------

.. code-block:: python

   from manifish import (
       ManifoldFishAnalyzer,
       aggregate_atom_embeddings,
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

   # Find neighbors for each generated material
   neighbor_indices = analyzer.print_neighbors(generated_embeddings)

The Seven Fish Categories
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 45 15 20

   * - Category
     - Description
     - Risk
     - Action
   * - Redundant Fish
     - Deep inside manifold, dense region
     - Very Low
     - Skip (redundant)
   * - Fish in Water
     - Inside manifold, normal density
     - Low
     - Standard validation
   * - Frontier Fish
     - Inside manifold, sparse region
     - Low-Medium
     - Priority for DFT
   * - Edge Fish
     - At manifold boundary
     - Medium
     - Careful validation
   * - Adventurous Fish
     - Slightly outside manifold
     - Medium-High
     - High priority DFT
   * - Geometric Atypical
     - High local PCA residual
     - Medium-High
     - Investigate geometry
   * - Structural Hallucination
     - Far outside manifold
     - Very High
     - Reject

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/basic_usage
   tutorials/atom_to_material
   tutorials/cross_mlip_comparison
   tutorials/ensemble_analysis

.. toctree::
   :maxdepth: 2
   :caption: Concepts
   :hidden:

   concepts/index
   concepts/fish_categories
   concepts/platonic_representation
   concepts/spatial_metrics

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/fish_analyzer
   api/aggregation
   api/spatial_metrics
   api/anchors
   api/metrics

.. toctree::
   :maxdepth: 1
   :caption: About
   :hidden:

   about/contributing
   about/changelog
   about/license

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
