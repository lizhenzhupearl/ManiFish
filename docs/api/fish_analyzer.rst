ManifoldFishAnalyzer
====================

The ``ManifoldFishAnalyzer`` is the core classifier that categorizes generated
structures into 7 categories based on their position relative to the reference manifold.

The Seven Categories
--------------------

1. **Redundant Fish** - Deep inside manifold, dense region (very similar to known)
2. **Fish in Water** - Inside manifold, normal density (standard candidate)
3. **Frontier Fish** - Inside manifold, sparse region (exploring new territory)
4. **Edge Fish** - At manifold boundary (on the edge of known physics)
5. **Adventurous Fish** - Slightly outside manifold (potentially novel)
6. **Geometric Atypical** - High local PCA residual but has neighbors
7. **Structural Hallucination** - Far outside manifold (likely unphysical)

Basic Usage
-----------

.. code-block:: python

   from manifish import ManifoldFishAnalyzer

   # Initialize with reference embeddings
   analyzer = ManifoldFishAnalyzer(
       reference_embeddings=ref_embeddings,  # (n_ref, embed_dim)
       reference_ids=ref_ids,                # List of material IDs
       n_neighbors=5,
       boundary_method="alpha_shape",
   )

   # Analyze generated structures
   results = analyzer.analyze(
       generated_embeddings,
       material_ids=gen_ids,
   )

   # Print summary
   analyzer.print_summary(results)

   # Export results
   analyzer.export_results(results, "analysis.csv")

Finding Neighbors
-----------------

Use ``print_neighbors`` to find which reference materials are closest to your
generated materials:

.. code-block:: python

   # Print neighbor indices for each generated material
   neighbor_indices = analyzer.print_neighbors(
       generated_embeddings,
       material_ids=["gen_0", "gen_1", "gen_2"],
       n_neighbors=10,
   )
   # Output:
   # Neighbor indices for 3 generated materials
   # (showing 10 nearest neighbors in reference set of 1000 points)
   # ------------------------------------------------------------
   # gen_0: neighbors = [42, 17, 89, ...]
   # gen_1: neighbors = [5, 23, 42, ...]
   # gen_2: neighbors = [789, 12, 445, ...]

API Reference
-------------

.. autoclass:: manifish.ManifoldFishAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: manifish.ManifoldFishResult
   :members:
   :undoc-members:
