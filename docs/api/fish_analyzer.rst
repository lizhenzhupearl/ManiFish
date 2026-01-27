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

Getting Results by Category
---------------------------

Retrieve materials by their classification category:

.. code-block:: python

   # Get material IDs for a specific category
   frontier_ids = analyzer.get_ids_by_category(results, "frontier_fish")
   print(f"Frontier fish IDs: {frontier_ids}")

   # Get IDs for all categories at once
   ids_by_cat = analyzer.get_all_ids_by_category(results)
   for category, ids in ids_by_cat.items():
       print(f"{category}: {len(ids)} materials")

   # Get indices (positions in original array) for a category
   frontier_indices = analyzer.get_indices_by_category(results, "frontier_fish")
   frontier_embeddings = generated_embeddings[frontier_indices]

   # Get indices for all categories
   indices_by_cat = analyzer.get_all_indices_by_category(results)
   print(indices_by_cat["structural_hallucination"])  # [45, 89, ...]

   # Get full result objects for detailed analysis
   frontier_results = analyzer.get_results_by_category(results, "frontier_fish")
   for r in frontier_results:
       print(f"{r.material_id}: confidence={r.confidence:.2f}")

**Available methods:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Returns
   * - ``get_ids_by_category(results, cat)``
     - List of material IDs for one category
   * - ``get_all_ids_by_category(results)``
     - Dict of category → material IDs
   * - ``get_indices_by_category(results, cat)``
     - List of indices for one category
   * - ``get_all_indices_by_category(results)``
     - Dict of category → indices
   * - ``get_results_by_category(results, cat)``
     - List of full result objects

API Reference
-------------

.. autoclass:: manifish.ManifoldFishAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: manifish.ManifoldFishResult
   :members:
   :undoc-members:
