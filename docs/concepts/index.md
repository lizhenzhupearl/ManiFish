# Concepts

## The Fish-Water Framework

ManiFish uses an intuitive **Fish-Water** metaphor for understanding generated crystal structures:

- **Water**: The reference manifold of known stable structures
- **Fish**: Generated structures swimming through or near the water
- **Categories**: Where and how fish relate to the water

### Why Fish?

Traditional metrics like distance or density alone can't capture the nuanced relationship between generated and reference structures. A structure might be:

- Very close to known structures (redundant)
- In an unexplored but valid region (frontier)
- At the boundary of physical plausibility (edge)
- Outside known physics (hallucination)

The 7 fish categories capture these nuances intuitively.

---

## Platonic Representation

### The Cross-MLIP Challenge

Different MLIPs (MACE, CHGNet, M3GNet, etc.) produce embeddings in different spaces:
- Different dimensions
- Different scales
- Different semantic meanings

Direct comparison is impossible without a common representation.

### Anchor-Based Projection

The **Platonic representation** solves this by:

1. Selecting a set of **anchor structures** (representative crystals)
2. Computing **cosine similarity** between any structure and all anchors
3. Using the similarity vector as the new representation

```
Original: embedding ∈ R^d (model-specific)
Platonic: similarity_vector ∈ R^n_anchors (model-agnostic)
```

This creates a **model-agnostic** representation where structures from different MLIPs can be directly compared.

### Why Cosine Similarity?

- **Invariant to scale**: Different models have different embedding magnitudes
- **Captures semantic alignment**: Similar structures have similar anchor relationships
- **Bounded output**: Values in [-1, 1] for easy interpretation

---

## The 7 Fish Categories

### 1. Redundant Fish (very_low risk)

**Location**: Deep inside manifold, dense region

**Interpretation**: Nearly identical to known structures. Adds no new information.

**Action**: Skip - not worth computational resources for DFT validation.

### 2. Fish in Water (low risk)

**Location**: Inside manifold, normal density

**Interpretation**: Standard valid structure, similar to known materials.

**Action**: Standard validation pipeline.

### 3. Frontier Fish (low_medium risk)

**Location**: Inside manifold, sparse region

**Interpretation**: Valid region but unexplored. Potentially interesting new materials.

**Action**: **Priority for DFT** - high value discovery candidates.

### 4. Edge Fish (medium risk)

**Location**: At manifold boundary

**Interpretation**: On the edge of known physics. Could be valid or marginal.

**Action**: Careful validation - check physical constraints.

### 5. Adventurous Fish (medium_high risk)

**Location**: Slightly outside manifold

**Interpretation**: Novel but risky. Might be genuine discovery or artifact.

**Action**: High priority DFT with careful analysis.

### 6. Geometric Atypical (medium_high risk)

**Location**: Inside but unusual local geometry

**Interpretation**: Has neighbors but doesn't fit local tangent space. Possible:
- High curvature region
- Novel polymorph
- Numerical artifact

**Action**: Investigate geometry - potentially interesting if physics checks out.

### 7. Structural Hallucination (very_high risk)

**Location**: Far outside manifold, isolated

**Interpretation**: No physical support. Likely unphysical generated structure.

**Action**: **Reject** - don't waste resources.

---

## Spatial Structure Metrics

Adapted from image generation research, these metrics assess the **local structure** of the manifold around each point.

### LDS: Local vs Distant Similarity

**Measures**: Contrast between local and distant similarities

**Formula**: LDS = mean(sim_local) / mean(sim_distant)

**Interpretation**:
- High LDS (>1.5): Strong local structure, point fits well locally
- Low LDS (<1.0): Point doesn't have clear local neighborhood

**Good for**: Detecting points that are "technically inside" but don't fit.

### CDS: Correlation Decay Slope

**Measures**: How quickly similarity decays with distance

**Formula**: Slope of log(similarity) vs distance

**Interpretation**:
- High CDS: Sharp locality - similarity drops quickly with distance
- Low CDS: Diffuse - point is similar to many distant points

**Good for**: Detecting manifold coherence.

### SRSS: Semantic-Region Self-Similarity

**Measures**: Within-region vs cross-region coherence

**Formula**: SRSS = mean(sim_same_region) / mean(sim_different_region)

**Requires**: Region labels (e.g., crystal system, composition family)

**Interpretation**:
- High SRSS (>1.5): Point fits its semantic region well
- Low SRSS (<1.0): Point is more similar to other regions

**Good for**: Detecting misclassified or boundary structures.

### RMSC: RMS Spatial Contrast

**Measures**: Diversity in local representations

**Formula**: RMSC = sqrt(mean(variance(neighbors)))

**Interpretation**:
- High RMSC: Diverse local neighborhood
- Low RMSC: Homogeneous local neighborhood

**Good for**: Detecting redundancy vs diversity.

### Combined Spatial Quality

All four metrics are normalized and combined:

```
quality = 0.3*LDS_norm + 0.3*CDS_norm + 0.2*SRSS_norm + 0.2*RMSC_norm
```

Range: [-1, 1] where higher is better.

---

## Classification Logic

The classification follows a priority order:

```
1. Near-exact match? → redundant_fish
2. Geometry unusual? → geometric_atypical
3. LOF outlier? → structural_hallucination
4. Outside boundary?
   - Slightly → adventurous_fish
   - Far → structural_hallucination
5. Near boundary? → edge_fish
6. Sparse region? → frontier_fish
7. Deep/central? → redundant_fish
8. Default → fish_in_water
```

This ensures proper handling of edge cases and consistent classification.

---

## Confidence Scores

### Base Confidence

Each classification comes with a confidence score (0-1) based on:
- How clearly the structure fits the category
- Distance from decision boundaries
- Strength of the classification signal

### Enhanced Confidence

The enhanced analyzer adjusts confidence using spatial quality:

```
enhanced_conf = 0.7 * base_conf + 0.3 * (spatial_quality + 1) / 2
```

This penalizes structures that are categorically correct but have poor local structure.
