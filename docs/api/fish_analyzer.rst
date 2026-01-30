ManifoldFishAnalyzer
====================

The ``ManifoldFishAnalyzer`` is the core classifier that categorizes generated
structures into 6 categories based on their position relative to the reference manifold.

The Six Categories
------------------

1. **Redundant Fish** - Deep inside manifold, dense region (very similar to known)
2. **Fish in Water** - Inside manifold, normal density (standard candidate)
3. **Frontier Fish** - Sparse region (low/high risk based on geometry)
4. **Edge Fish** - At manifold boundary (on the edge of known physics)
5. **Adventurous Fish** - Outside manifold (risk based on geometry and LOF)
6. **Structural Hallucination** - Bad geometry + LOF outlier (likely unphysical)

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

Viewing Detailed Statistics
---------------------------

Use ``print_statistics`` to see the metrics that determined each material's classification:

.. code-block:: python

   # Show statistics for specific materials
   analyzer.print_statistics(results, material_ids=["gen_0", "gen_5"])

   # Show statistics for first 3 materials by index
   analyzer.print_statistics(results, indices=[0, 1, 2])

   # Show statistics for all frontier fish
   frontier_ids = analyzer.get_ids_by_category(results, "frontier_fish")
   analyzer.print_statistics(results, material_ids=frontier_ids)

   # Output example:
   # ════════════════════════════════════════════════════════════════════════════════
   # DETAILED STATISTICS FOR MATERIALS
   # ════════════════════════════════════════════════════════════════════════════════
   #
   # ────────────────────────────────────────────────────────────────────────────────
   # Material: gen_0  (index: 0)
   # ────────────────────────────────────────────────────────────────────────────────
   #
   #   CLASSIFICATION:
   #     Category:        Frontier Fish
   #     Confidence:      0.850
   #     Risk Level:      LOW
   #
   #   POSITION METRICS:
   #     Manifold Distance:   0.1234  (normalized distance to nearest references)
   #     Depth Score:         0.7500  (0=deep inside, 1=shallow/edge)
   #     Boundary Distance:   +0.0500  (+inside, -outside manifold)
   #
   #   DENSITY METRICS:
   #     Local Density:       0.0023
   #     Density Percentile:  15.2%  (vs reference distribution)
   #
   #   GEOMETRY CONSISTENCY:
   #     Local PCA Residual:  0.0150  (reconstruction error)
   #     Geometry Consistent: Yes
   #
   #   NEAREST REFERENCES:
   #     Top 3: mp-123, mp-456, mp-789

You can also get individual results programmatically:

.. code-block:: python

   # Get a single result by material ID
   result = analyzer.get_result_by_id(results, "gen_0")
   if result:
       print(f"Category: {result.category}")
       print(f"Manifold distance: {result.manifold_distance}")
       print(f"Local PCA residual: {result.local_pca_residual}")

   # Get a single result by index
   result = analyzer.get_result_by_index(results, 0)

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
   * - ``print_statistics(results, ...)``
     - Print detailed metrics for specific materials
   * - ``get_result_by_id(results, id)``
     - Get single result by material ID
   * - ``get_result_by_index(results, idx)``
     - Get single result by index

API Reference
-------------

.. autoclass:: manifish.ManifoldFishAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: manifish.ManifoldFishResult
   :members:
   :undoc-members:
