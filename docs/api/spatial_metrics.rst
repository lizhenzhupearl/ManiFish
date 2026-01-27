Spatial Metrics
===============

Spatial structure metrics for improved stability prediction based on local manifold structure.

Overview
--------

Spatial metrics analyze the local geometry and structure of the embedding manifold to
provide additional signals for stability prediction:

- **LDS (Local Density Score)**: Measures local point density
- **CDS (Conformity to Density Structure)**: How well a point conforms to local density patterns
- **SRSS (Structural Region Similarity Score)**: Similarity to structural regions
- **RMSC (Region Manifold Structure Conformity)**: Overall conformity to manifold structure

Basic Usage
-----------

.. code-block:: python

   from manifish import (
       compute_lds,
       compute_cds,
       compute_srss,
       compute_rmsc,
       compute_all_spatial_metrics,
   )

   # Compute individual metrics
   lds = compute_lds(embeddings, n_neighbors=10)
   cds = compute_cds(embeddings, reference_embeddings, n_neighbors=10)

   # Compute all metrics at once
   metrics = compute_all_spatial_metrics(
       query_embeddings=generated_embeddings,
       reference_embeddings=reference_embeddings,
       region_labels=region_labels,  # Optional
       n_neighbors=10,
   )

API Reference
-------------

.. autofunction:: manifish.compute_lds

.. autofunction:: manifish.compute_cds

.. autofunction:: manifish.compute_srss

.. autofunction:: manifish.compute_rmsc

.. autofunction:: manifish.compute_all_spatial_metrics

.. autofunction:: manifish.compute_spatial_metrics_batch

.. autoclass:: manifish.SpatialMetrics
   :members:

.. autoclass:: manifish.SpatialMetricsConfig
   :members:
