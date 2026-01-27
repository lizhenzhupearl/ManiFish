"""
ManiFish Batch Evaluation Example

This example shows how to:
1. Extract embeddings from multiple MLIPs
2. Compute Platonic representations
3. Compare models using cross-model metrics
4. Analyze specific subgroups (e.g., by space group)
"""

import numpy as np
from typing import List, Dict

from manifish.core.anchors import select_anchors, AnchorSet
from manifish.core.manifold import calculate_platonic_representation
from manifish.core.metrics import (
    compute_mutual_knn_matrix,
    compute_optimal_transport,
    estimate_intrinsic_dimension,
    compare_intrinsic_dimensions,
    compute_consensus,
)


def generate_mock_embeddings(n_structures: int, n_models: int = 3) -> Dict[str, np.ndarray]:
    """Generate mock embeddings for demonstration."""
    np.random.seed(42)
    embedding_dim = 128

    # Base embeddings
    base = np.random.randn(n_structures, embedding_dim)

    model_names = ['mace', 'chgnet', 'm3gnet', 'orb', 'sevennet'][:n_models]
    embeddings = {}

    for i, name in enumerate(model_names):
        # Each model has correlated but different embeddings
        noise = 0.2 + 0.1 * i  # Increasing noise
        embeddings[name] = base + noise * np.random.randn(n_structures, embedding_dim)

    return embeddings


def main():
    print("=" * 70)
    print("ManiFish Batch Evaluation Example")
    print("Cross-MLIP Comparison using Platonic Representation")
    print("=" * 70)

    # =========================================================================
    # Setup: Generate/Load Embeddings
    # =========================================================================
    print("\n1. Generating mock embeddings...")

    n_structures = 2000
    n_models = 4

    # In practice, you would:
    # from manifish.models import MACEEmbeddingExtractor, OrbEmbeddingExtractor
    # mace_ext = MACEEmbeddingExtractor(model='medium')
    # mace_ext.load_model()
    # mace_emb = mace_ext.get_structure_embeddings(structures)

    embeddings = generate_mock_embeddings(n_structures, n_models)

    for name, emb in embeddings.items():
        print(f"   {name}: {emb.shape}")

    # =========================================================================
    # Select Anchors
    # =========================================================================
    print("\n2. Selecting anchors...")

    # Use embeddings from one model for anchor selection
    reference_model = 'mace'
    anchor_set = select_anchors(
        embeddings=embeddings[reference_model],
        n_anchors=100,
        method='direct',
        model_name=reference_model,
    )

    print(f"   Selected {len(anchor_set.anchors)} anchors using {reference_model}")

    # =========================================================================
    # Compute Platonic Representations
    # =========================================================================
    print("\n3. Computing Platonic representations...")

    platonic_reps = {}
    for name, emb in embeddings.items():
        platonic_reps[name] = calculate_platonic_representation(
            embeddings=emb,
            anchor_embeddings=anchor_set,
            model_name=name,
        )
        print(f"   {name}: {platonic_reps[name].shape}")

    # =========================================================================
    # Cross-Model Comparison: Mutual KNN Matrix
    # =========================================================================
    print("\n4. Computing cross-model similarity (Mutual KNN)...")

    model_names = list(platonic_reps.keys())
    platonic_list = [platonic_reps[name] for name in model_names]

    mknn_matrix, names = compute_mutual_knn_matrix(
        platonic_list,
        model_names=model_names,
        k=10,
        n_samples=1000,
    )

    print("\n   Mutual KNN Similarity Matrix:")
    print("   " + "".join(f"{n:>10}" for n in names))
    for i, name in enumerate(names):
        row = "   " + f"{name:>10}" + "".join(f"{mknn_matrix[i,j]:>10.3f}" for j in range(len(names)))
        print(row)

    # =========================================================================
    # Intrinsic Dimension Analysis
    # =========================================================================
    print("\n5. Intrinsic Dimension Analysis...")

    for name in model_names:
        original_emb = embeddings[name]
        platonic_emb = platonic_reps[name]

        _, orig_id, _ = estimate_intrinsic_dimension(original_emb)
        _, plat_id, _ = estimate_intrinsic_dimension(platonic_emb)

        print(f"   {name}: Original ID = {orig_id:.1f}, Platonic ID = {plat_id:.1f}")

    # =========================================================================
    # Consensus Analysis
    # =========================================================================
    print("\n6. Multi-Model Consensus Analysis...")

    consensus_scores = compute_consensus(platonic_reps)

    print(f"   Mean consensus: {np.mean(consensus_scores):.4f}")
    print(f"   Std consensus: {np.std(consensus_scores):.4f}")
    print(f"   High consensus (>0.8): {np.sum(consensus_scores > 0.8)}/{n_structures}")
    print(f"   Medium consensus (0.5-0.8): {np.sum((consensus_scores >= 0.5) & (consensus_scores <= 0.8))}/{n_structures}")
    print(f"   Low consensus (<0.5): {np.sum(consensus_scores < 0.5)}/{n_structures}")

    # =========================================================================
    # Optimal Transport (for comparing distributions)
    # =========================================================================
    print("\n7. Optimal Transport Costs...")

    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            _, ot_cost = compute_optimal_transport(
                platonic_reps[name1][:500],  # Subsample for speed
                platonic_reps[name2][:500],
            )
            print(f"   OT({name1}, {name2}): {ot_cost:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    avg_off_diag = (mknn_matrix.sum() - np.trace(mknn_matrix)) / (len(model_names) ** 2 - len(model_names))
    high_consensus_pct = 100 * np.sum(consensus_scores > 0.8) / n_structures

    print(f"""
Results:
- Average cross-model Mutual KNN: {avg_off_diag:.4f}
- High consensus structures: {high_consensus_pct:.1f}%

Interpretation:
- Mutual KNN > 0.5 indicates strong neighborhood preservation
- High consensus (>0.8) structures are reliable across all models
- Low consensus structures should be prioritized for DFT validation
""")


if __name__ == "__main__":
    main()
