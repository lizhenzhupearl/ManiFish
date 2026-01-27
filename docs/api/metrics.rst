Distance-Based Metrics
======================

Functions for comparing manifolds and measuring embedding similarity.

Overview
--------

These metrics help you understand how similar two sets of embeddings are,
useful for cross-MLIP comparison and quality assessment.

Quick Start
-----------

.. code-block:: python

   from manifish import (
       compute_mutual_knn,
       compute_cka,
       compute_wasserstein_distance,
   )

   # Compare two embedding sets
   mknn = compute_mutual_knn(embeddings_a, embeddings_b, k=10)
   cka = compute_cka(embeddings_a, embeddings_b)
   wd = compute_wasserstein_distance(embeddings_a, embeddings_b)

Comparison Metrics
------------------

Mutual k-NN
~~~~~~~~~~~

Measures neighborhood preservation between two embedding sets.

.. code-block:: python

   from manifish import compute_mutual_knn

   score = compute_mutual_knn(
       X=mace_embeddings,
       Y=chgnet_embeddings,
       k=10,
       n_samples=500,  # Subsample for efficiency
   )
   # Higher = more similar neighborhoods

CKA (Centered Kernel Alignment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures structural similarity between representations.

.. code-block:: python

   from manifish import compute_cka

   score = compute_cka(
       X=mace_embeddings,
       Y=chgnet_embeddings,
       kernel="linear",  # or "rbf"
   )
   # Higher = more structurally similar

Wasserstein Distance
~~~~~~~~~~~~~~~~~~~~

Optimal transport distance between distributions.

.. code-block:: python

   from manifish import compute_wasserstein_distance

   distance = compute_wasserstein_distance(
       X=generated_embeddings,
       Y=reference_embeddings,
   )
   # Lower = more similar distributions

Procrustes Distance
~~~~~~~~~~~~~~~~~~~

Measures alignment after optimal rotation/scaling.

.. code-block:: python

   from manifish import compute_procrustes_distance

   distance = compute_procrustes_distance(
       X=embeddings_a,
       Y=embeddings_b,
   )

Intrinsic Dimension
-------------------

Estimate the intrinsic dimensionality of an embedding manifold.

.. code-block:: python

   from manifish import estimate_intrinsic_dimension

   collapse, intrinsic_dim, details = estimate_intrinsic_dimension(
       embeddings,
       method="mle",  # Maximum Likelihood Estimation
   )

   print(f"Intrinsic dimension: {intrinsic_dim:.1f}")

Consensus
---------

Compute per-structure consensus across multiple models.

.. code-block:: python

   from manifish import compute_consensus

   platonic_reps = {
       'mace': mace_platonic,
       'chgnet': chgnet_platonic,
       'orb': orb_platonic,
   }

   consensus = compute_consensus(platonic_reps)
   # Array of consensus scores, one per structure

   high_agreement = consensus > 0.8
   print(f"High agreement: {high_agreement.sum()} structures")

API Reference
-------------

.. autofunction:: manifish.compute_mutual_knn

.. autofunction:: manifish.compute_cka

.. autofunction:: manifish.compute_wasserstein_distance

.. autofunction:: manifish.compute_optimal_transport

.. autofunction:: manifish.compute_procrustes_distance

.. autofunction:: manifish.estimate_intrinsic_dimension

.. autofunction:: manifish.compare_intrinsic_dimensions

.. autofunction:: manifish.compute_consensus

.. autofunction:: manifish.compute_quality_score
