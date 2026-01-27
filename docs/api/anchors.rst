Anchor Selection
================

Functions and classes for selecting anchor points for Platonic representation.

Overview
--------

Anchors are diverse reference points used to create a common coordinate system
for comparing embeddings across different MLIP models.

Quick Start
-----------

.. code-block:: python

   from manifish import select_anchors

   # Select 100 diverse anchors
   anchor_set = select_anchors(
       embeddings=reference_embeddings,
       n_anchors=100,
       method="direct",  # or "fps"
   )

   print(anchor_set.embeddings.shape)  # (100, embed_dim)

Methods
-------

**DIRECT** (Recommended): Uses BIRCH clustering to find diverse representatives.

**FPS** (Farthest Point Sampling): Iteratively selects points maximizing minimum distance.

API Reference
-------------

.. autofunction:: manifish.select_anchors

.. autoclass:: manifish.DIRECTAnchorSelector
   :members:

.. autoclass:: manifish.FPSAnchorSelector
   :members:

.. autoclass:: manifish.AnchorSet
   :members:

.. autoclass:: manifish.Anchor
   :members:

Precomputed Anchors
-------------------

For the MP20 dataset, precomputed anchor indices are available:

.. code-block:: python

   from manifish import MP20_ANCHOR_INDICES_100, load_precomputed_anchors

   # Use precomputed indices
   anchor_indices = MP20_ANCHOR_INDICES_100
   anchors = reference_embeddings[anchor_indices]
