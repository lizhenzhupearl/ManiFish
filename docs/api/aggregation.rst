Aggregation Utilities
=====================

Functions for aggregating atom-level embeddings to material-level embeddings.

Overview
--------

MLIP models typically produce atom-level embeddings. To analyze materials as a whole,
you need to aggregate these atom embeddings into a single material-level representation.

ManiFish provides several aggregation functions:

- ``aggregate_atom_embeddings``: When you have an index array mapping atoms to materials
- ``aggregate_from_structure_list``: When you have a list of arrays (one per structure)
- ``aggregate_by_material_id``: When you have material IDs for each atom
- ``aggregate_with_weights``: For weighted aggregation (e.g., by atomic mass)

aggregate_atom_embeddings
-------------------------

Use when you have atom embeddings with an index array mapping each atom to its material.

.. code-block:: python

   from manifish import aggregate_atom_embeddings

   # 1000 atoms across 50 materials
   atom_embeddings = np.random.randn(1000, 128)  # (n_atoms, embed_dim)
   atom_to_material = np.repeat(np.arange(50), 20)  # 20 atoms per material
   material_ids = [f"mp-{i}" for i in range(50)]

   material_embeddings, ids = aggregate_atom_embeddings(
       atom_embeddings,
       atom_to_material,
       material_ids,
       method="mean"  # or "sum", "max", "min", "std", "mean_std"
   )
   print(material_embeddings.shape)  # (50, 128)

aggregate_from_structure_list
-----------------------------

Use when you have a list of arrays, one per structure.

.. code-block:: python

   from manifish import aggregate_from_structure_list

   # List of structures with different numbers of atoms
   structures = [
       np.random.randn(12, 128),  # 12 atoms
       np.random.randn(8, 128),   # 8 atoms
       np.random.randn(24, 128),  # 24 atoms
   ]
   material_ids = ["mp-123", "mp-456", "mp-789"]

   material_embeddings, ids = aggregate_from_structure_list(
       structures,
       material_ids,
       method="mean"
   )
   print(material_embeddings.shape)  # (3, 128)

aggregate_by_material_id
------------------------

Use when you have material IDs for each atom (atoms grouped by material ID).

.. code-block:: python

   from manifish import aggregate_by_material_id

   atom_embeddings = np.random.randn(100, 128)
   atom_material_ids = ["mp-1"] * 30 + ["mp-2"] * 40 + ["mp-3"] * 30

   material_embeddings, unique_ids = aggregate_by_material_id(
       atom_embeddings,
       atom_material_ids,
       method="mean"
   )
   print(unique_ids)  # ["mp-1", "mp-2", "mp-3"]
   print(material_embeddings.shape)  # (3, 128)

aggregate_with_weights
----------------------

Use for weighted aggregation (e.g., by atomic mass).

.. code-block:: python

   from manifish import aggregate_with_weights

   atom_embeddings = np.random.randn(100, 128)
   atom_to_material = np.repeat(np.arange(5), 20)
   atomic_masses = np.random.rand(100) * 100  # Example weights
   material_ids = [f"mp-{i}" for i in range(5)]

   material_embeddings, ids = aggregate_with_weights(
       atom_embeddings,
       atom_to_material,
       atomic_masses,
       material_ids
   )

Aggregation Methods
-------------------

All aggregation functions support these methods:

- ``mean``: Average pooling (recommended for most cases)
- ``sum``: Sum pooling
- ``max``: Element-wise maximum
- ``min``: Element-wise minimum
- ``std``: Standard deviation (captures spread)
- ``mean_std``: Concatenate mean and std (doubles dimension)

API Reference
-------------

.. autofunction:: manifish.aggregate_atom_embeddings

.. autofunction:: manifish.aggregate_from_structure_list

.. autofunction:: manifish.aggregate_by_material_id

.. autofunction:: manifish.aggregate_with_weights
