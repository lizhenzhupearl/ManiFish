# Ideas and Discussions

This document captures ongoing ideas and discussions for future exploration.

---

## 2025-01-28: Manifold Geometry and Phase Transitions

### Observation: Cu-Cu Dimer Forms a 1D Line

For a Cu-Cu dimer where only the interatomic distance changes, the embeddings form a **1D line/track** in the embedding space.

**Why this happens:**
- Single degree of freedom (distance `d`)
- MLIPs learn smooth functions → continuous path in embedding space
- The line represents the potential energy surface projected into embedding space

### Initial Hypothesis: Phase Transition Lines

**Question:** Can we find "phase transition lines" in embedding space for bulk materials?

**Reality:** Bulk materials are different:
- More degrees of freedom (lattice params, atomic positions, composition)
- Symmetry plays a bigger role
- Higher dimensional than 1D line
- Phase transitions may be discontinuous (first-order) or along curved surfaces (second-order)

### Key Literature Concepts

From manifold learning theory:

> "High-dimensional data is typically concentrated on or near a low-dimensional manifold, M, rather than being uniformly distributed throughout Euclidean space. On a manifold, the meaningful distance is the geodesic distance along its surface, not the Euclidean distance upon which PCA relies."

> "For an S-shaped manifold, global PCA will erroneously identify the Euclidean 'shortcut' between its endpoints as the principal component u1. This path traverses a void where no data exists, thus severely distorting the true geometry of the data."

> "An ideal, manifold-aware calibration should operate within the local tangent space, TxM, at a given data point x. Yet, the global subspace span(U) found by PCA is almost never aligned with the local tangent space TxM for an arbitrary point x on a curved manifold."

### The Core Problem Illustrated

```
Global PCA on curved manifold:

     ●●●●●
    ●     ●
   ●       ●          PCA finds this "shortcut"
   ●       ●          ←------------------------→
   ●       ●          (through empty space!)
    ●     ●
     ●●●●●

True geodesic distance follows the curve (much longer)
```

### Geodesic vs Euclidean Distance

| Distance Type | Description | Problem |
|---------------|-------------|---------|
| Euclidean | Straight line through embedding space | May cross "unphysical" regions |
| Geodesic | Path along the manifold surface | Stays in "physical" space |

### How ManiFish Currently Addresses This

| Problem | Current Approach |
|---------|------------------|
| Global PCA misleading | Uses **local** k-NN neighborhoods |
| Euclidean shortcuts | Uses **boundary methods** (alpha shape, not just distance) |
| Curved manifold | **Local PCA residual** detects points off the local tangent |
| Tangent space alignment | Points with high residual → "geometric_atypical" |

### Open Questions for Exploration

1. **Should ManiFish use geodesic distances?**
   - Currently uses Euclidean k-NN
   - Could use ISOMAP-style geodesic approximation
   - Trade-off: computational cost vs accuracy

2. **Is local PCA enough?**
   - It captures the tangent space
   - But doesn't compute true geodesics
   - May miss global manifold structure

3. **Phase transition detection:**
   - Phase boundary might be where local tangent space orientation changes dramatically
   - Could we detect this by measuring tangent space alignment between neighbors?

4. **Intrinsic dimension variation:**
   - Simple systems (dimers) have low intrinsic dimension
   - Bulk materials have higher dimension
   - Should we adapt analysis based on local intrinsic dimension?

5. **Embedding space structure for bulk materials:**
   - Do phases form distinct clusters?
   - Are transition states between clusters?
   - Can we correlate embedding directions with physical order parameters?

### Potential Future Directions

- [ ] Implement geodesic distance estimation (ISOMAP-style)
- [ ] Add local intrinsic dimension estimation per point
- [ ] Detect tangent space orientation changes (potential phase boundaries)
- [ ] Visualize embedding trajectories for simple systems (dimers, strain paths)
- [ ] Correlate embedding directions with known order parameters

---

## Template for Future Ideas

### Date: YYYY-MM-DD - Topic

**Observation:**

**Hypothesis:**

**Questions:**

**Potential approaches:**

---
