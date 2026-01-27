"""
ManiFish Quick Start Example

This example demonstrates the core workflow:
1. Extract embeddings from an MLIP
2. Select diverse anchors using DIRECT clustering
3. Project embeddings to Platonic (anchor-based) representation
4. Compare models using various metrics
"""

import numpy as np
import pickle

# Import ManiFish components
from manifish.core.anchors import select_anchors, AnchorSet
from manifish.core.manifold import (
    calculate_platonic_representation,
    project_to_anchor_space,
    PlatonicProjector,
)
from manifish.core.metrics import (
    compute_mutual_knn,
    estimate_intrinsic_dimension,
    compute_cka,
    compute_consensus,
)


def main():
    """Run the quick start example."""

    print("=" * 60)
    print("ManiFish Quick Start Example")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load or generate embeddings
    # =========================================================================
    print("\n1. Loading embeddings...")

    # In practice, you would extract embeddings from MLIPs:
    #
    # from manifish.models import MACEEmbeddingExtractor
    # extractor = MACEEmbeddingExtractor(model='medium', device='cpu')
    # extractor.load_model()
    # embeddings = extractor.get_structure_embeddings(structures)
    #
    # Or load pre-computed embeddings:
    # with open('mace_embeddings.pkl', 'rb') as f:
    #     embeddings = pickle.load(f)

    # For demo, create synthetic embeddings
    np.random.seed(42)
    n_structures = 1000
    embedding_dim = 128

    # Simulate embeddings from two different MLIPs
    mace_embeddings = np.random.randn(n_structures, embedding_dim)
    chgnet_embeddings = mace_embeddings + 0.3 * np.random.randn(n_structures, embedding_dim)

    print(f"   MACE embeddings shape: {mace_embeddings.shape}")
    print(f"   CHGNet embeddings shape: {chgnet_embeddings.shape}")

    # =========================================================================
    # Step 2: Select anchors using DIRECT clustering
    # =========================================================================
    print("\n2. Selecting anchors with DIRECT sampler...")

    # Select 100 diverse anchors from MACE embeddings
    anchor_set = select_anchors(
        embeddings=mace_embeddings,
        n_anchors=100,
        method='direct',  # Uses Birch clustering
        model_name='mace',
    )

    print(f"   Selected {len(anchor_set.anchors)} anchors")
    print(f"   Anchor embedding shape: {anchor_set.embeddings.shape}")

    # =========================================================================
    # Step 3: Project to Platonic representation
    # =========================================================================
    print("\n3. Computing Platonic representation...")

    # Method 1: Simple functional interface
    mace_platonic = calculate_platonic_representation(
        embeddings=mace_embeddings,
        anchor_embeddings=anchor_set,
        model_name='mace',
    )

    chgnet_platonic = calculate_platonic_representation(
        embeddings=chgnet_embeddings,
        anchor_embeddings=anchor_set,
        model_name='chgnet',
    )

    print(f"   MACE Platonic shape: {mace_platonic.shape}")
    print(f"   CHGNet Platonic shape: {chgnet_platonic.shape}")

    # Method 2: Using PlatonicProjector class
    projector = PlatonicProjector(anchor_set, similarity_method='cosine')
    mace_rep = projector.project(mace_embeddings, model_name='mace')
    print(f"   Using projector: {mace_rep.n_samples} samples, {mace_rep.n_anchors} anchors")

    # =========================================================================
    # Step 4: Compare models using metrics
    # =========================================================================
    print("\n4. Computing comparison metrics...")

    # Mutual KNN - measures neighborhood preservation
    mknn_score = compute_mutual_knn(
        mace_platonic,
        chgnet_platonic,
        k=10,
        n_samples=500,  # Subsample for efficiency
    )
    print(f"   Mutual KNN (k=10): {mknn_score:.4f}")

    # CKA - measures structural similarity
    cka_score = compute_cka(mace_platonic, chgnet_platonic, kernel='linear')
    print(f"   CKA (linear): {cka_score:.4f}")

    # Intrinsic dimension - measures manifold complexity
    collapse_mace, id_mace, _ = estimate_intrinsic_dimension(mace_platonic)
    collapse_chgnet, id_chgnet, _ = estimate_intrinsic_dimension(chgnet_platonic)
    print(f"   MACE intrinsic dimension: {id_mace:.2f}")
    print(f"   CHGNet intrinsic dimension: {id_chgnet:.2f}")

    # Consensus - measures agreement across models
    platonic_reps = {
        'mace': mace_platonic,
        'chgnet': chgnet_platonic,
    }
    consensus_scores = compute_consensus(platonic_reps)
    print(f"   Mean consensus: {np.mean(consensus_scores):.4f}")
    print(f"   High consensus (>0.8): {np.sum(consensus_scores > 0.8)} structures")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"""
The Platonic representation projects embeddings from different MLIPs
into a common anchor-based space, enabling direct comparison.

Key findings from this example:
- Mutual KNN: {mknn_score:.4f} (higher = more similar neighborhoods)
- CKA: {cka_score:.4f} (higher = more structurally similar)
- Both models have similar intrinsic dimension (~{(id_mace + id_chgnet)/2:.1f})
- {np.sum(consensus_scores > 0.8)}/{n_structures} structures have high consensus

High consensus structures are more reliable predictions.
Low consensus structures indicate uncertainty and may need DFT validation.
""")


if __name__ == "__main__":
    main()
