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

## 2025-01-28: Insights from FedMC Paper (Federated Manifold Calibration)

**Source:** "FedMC: Federated Manifold Calibration" (ICLR 2026 submission)

### Core Argument

The paper argues that traditional calibration methods fail because they assume embeddings lie in a **linear subspace**, but real data lives on **curved manifolds**. This directly supports our earlier discussion about PCA finding "shortcuts through empty space."

### Key Concepts

#### 1. Local Kernel PCA (vs Linear PCA)

FedMC uses **local kernel PCA** to capture the manifold's local curvature at each point:

| Approach | What it captures | Limitation |
|----------|------------------|------------|
| Global linear PCA | Single linear subspace | Misses curvature entirely |
| Local linear PCA (ManiFish current) | Local tangent plane | Assumes locally flat |
| Local kernel PCA (FedMC) | Nonlinear local geometry | More computationally expensive |

**Implication for ManiFish:** Our current local PCA residual assumes the manifold is locally flat. Kernel PCA could capture cases where even locally the geometry is curved.

#### 2. Geometry Dictionary

FedMC builds a **"geometry dictionary"** - a collection of local geometry descriptors that can be aggregated across the dataset.

**How it works:**
- Each point has associated local geometry information
- These are stored in a dictionary structure
- Can be used to characterize "normal" local geometry
- Anomalies detected by geometry mismatch, not just distance

**Potential ManiFish application:**
```
For each reference point, store:
  - Local covariance structure
  - Principal directions (tangent space basis)
  - Local intrinsic dimension estimate
  - Curvature indicators

When evaluating a generated point:
  - Find its neighbors
  - Compare its local geometry to neighbors' stored geometry
  - Flag if geometry is inconsistent (not just if distance is large)
```

#### 3. On-Manifold Operations

The paper performs corrections **within the local tangent space** rather than in ambient Euclidean space.

**Key insight:** Operations should respect the manifold structure:
- Don't measure distances through empty space
- Don't apply corrections that move points off-manifold
- Work in the coordinate system natural to the local geometry

**For ManiFish:** This suggests our distance thresholds should potentially be **adaptive** to local geometry. A point might be "close" in Euclidean terms but "far" in geodesic terms if the manifold curves away.

### Relevance Matrix

| FedMC Concept | Current ManiFish | Potential Upgrade |
|---------------|------------------|-------------------|
| Local kernel PCA | Linear local PCA | Add kernel option for curved regions |
| Geometry dictionary | None (stateless) | Store local geometry per reference point |
| On-manifold calibration | Euclidean thresholds | Adaptive thresholds based on local geometry |
| Tangent space alignment | Implicit in PCA residual | Explicit tangent space comparison |

### Concrete Ideas for Implementation

1. **Geometry-aware anomaly detection:**
   - Store local covariance matrix for each reference point
   - When evaluating generated point, check if its local geometry matches neighbors
   - New category: "geometry_mismatch" (has neighbors but wrong local structure)

2. **Adaptive distance thresholds:**
   - Compute local density/spread for each reference region
   - Scale "outside manifold" threshold by local geometry
   - Dense regions: tighter threshold; sparse regions: looser threshold

3. **Curvature-based classification:**
   - Estimate local curvature at each point
   - High curvature regions may need special handling
   - Could explain why some "edge_fish" are actually on a curved boundary

4. **Tangent space consistency score:**
   - For a generated point and its k neighbors
   - Compute tangent space at generated point (from local PCA)
   - Compare alignment with neighbors' tangent spaces
   - Large misalignment → geometric anomaly

### Questions Raised

1. **Is kernel PCA worth the computational cost?**
   - Linear local PCA is O(k * d^2) per point
   - Kernel PCA adds kernel matrix computation
   - For materials discovery, accuracy may be worth the cost

2. **What kernel to use?**
   - RBF kernel is standard but has bandwidth hyperparameter
   - Polynomial kernel might capture specific physics
   - Could the kernel be learned from data?

3. **How to aggregate geometry across materials?**
   - FedMC aggregates across federated clients
   - ManiFish could aggregate across different material classes
   - Cross-MLIP geometry comparison?

### Connection to Earlier Discussion

This paper provides **theoretical grounding** for our intuition that:
- Euclidean distance through embedding space can be misleading
- Local geometry matters more than global structure
- The manifold's curvature affects what "similar" and "different" mean

The **geometry dictionary** concept could be particularly powerful for ManiFish:
- Build a "dictionary of known material geometries"
- Compare generated materials not just by distance, but by geometric consistency
- Potentially detect when a generated structure has "impossible" local geometry

### Potential Future Directions

- [ ] Implement optional kernel PCA for local geometry estimation
- [ ] Add geometry dictionary storage to ManifoldFishAnalyzer
- [ ] Create "geometry_mismatch" as a new fish category
- [ ] Implement adaptive thresholds based on local density/curvature
- [ ] Compare tangent space alignment between neighbors
- [ ] Explore cross-MLIP geometry consistency

---

## Template for Future Ideas

### Date: YYYY-MM-DD - Topic

**Observation:**

**Hypothesis:**

**Questions:**

**Potential approaches:**

---
