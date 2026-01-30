# The Six Fish Categories

ManiFish classifies generated structures into six categories based on their position relative to the reference manifold. This page explains each category and how to interpret them.

## The Fish-Water Metaphor

Think of the reference manifold as **water** - the space of known stable structures. Generated structures are **fish** that can be:

- Swimming in the water (inside the manifold)
- At the surface (at the boundary)
- Jumping out (outside the manifold)

## Category Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MANIFOLD INTERIOR (DENSE)                    │
│  ┌─────────────┐                                                │
│  │  REDUNDANT  │  Dense region, very similar to known          │
│  │    FISH     │                                                │
│  └─────────────┘                                                │
│           ┌─────────────┐                                       │
│           │   FISH IN   │  Normal region, standard candidates  │
│           │   WATER     │                                       │
│           └─────────────┘                                       │
├─────────────────────────────────────────────────────────────────┤
│                    MANIFOLD INTERIOR (SPARSE)                   │
│                    ┌─────────────┐                              │
│                    │  FRONTIER   │  Sparse region, exploring   │
│                    │    FISH     │  (novel territory)          │
│                    └─────────────┘                              │
├─────────────────────────────────────────────────────────────────┤
│                    MANIFOLD BOUNDARY                            │
│  ┌─────────────┐                                                │
│  │  EDGE FISH  │  At the boundary of known physics             │
│  └─────────────┘                                                │
├─────────────────────────────────────────────────────────────────┤
│                    OUTSIDE MANIFOLD                             │
│  ┌─────────────┐                                                │
│  │ ADVENTUROUS │  Outside boundary, normal LOF                 │
│  │    FISH     │  (exploration candidate)                      │
│  └─────────────┘                                                │
│                    ┌─────────────┐                              │
│                    │HALLUCINATION│  LOF outlier                │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Category Descriptions

### 1. Redundant Fish 🐟

**Location**: Deep inside manifold, dense region

**Characteristics**:
- Very close to many reference structures
- High local density
- Low novelty

**Interpretation**: These structures are essentially duplicates of known materials. While likely stable, they don't add new information.

**Recommended Action**: Skip - they're redundant with existing materials.

---

### 2. Fish in Water 🐠

**Location**: Inside manifold, normal density

**Characteristics**:
- Moderate distance to references
- Normal local density
- Geometrically consistent

**Interpretation**: Standard candidates that fit well within the known chemical space. Good reliability.

**Recommended Action**: Standard DFT validation. Good candidates for property prediction.

---

### 3. Frontier Fish 🐡

**Location**: Inside manifold, sparse region

**Characteristics**:
- Inside the manifold but in low-density areas
- Exploring undersampled regions

**Interpretation**: These are the most interesting discoveries - structures that are still physically plausible but represent new territory in chemical space.

**Recommended Action**: High priority for DFT validation. Potential novel materials.

---

### 4. Edge Fish 🦈

**Location**: At manifold boundary

**Characteristics**:
- Near the edge of the convex hull / alpha shape
- Boundary between known and unknown

**Interpretation**: Structures at the edge of known physics. Could be novel stable materials or could be approaching instability.

**Recommended Action**: Careful validation. Check for potential instabilities.

---

### 5. Adventurous Fish 🐋

**Location**: Outside manifold boundary

**Characteristics**:
- Negative boundary distance (outside hull)
- Normal LOF score (not a density outlier)
- Exploring beyond known territory

**Interpretation**: Potentially novel materials that extend beyond the known manifold. These are exploration candidates that deserve validation.

**Recommended Action**: High priority for DFT validation. Check `risk_level` in results.

---

### 6. Structural Hallucination 👻

**Location**: LOF outlier (any location)

**Characteristics**:
- LOF outlier (density anomaly)
- No structural support from reference data
- Can occur inside or outside manifold

**Interpretation**: Likely unphysical structures. The density anomaly indicates the structure doesn't fit the local neighborhood pattern.

**Recommended Action**: Reject. Not worth computational resources.

## Decision Matrix

| Category | Priority for DFT | Confidence | Risk |
|----------|------------------|------------|------|
| Redundant Fish | Low | High | Very Low |
| Fish in Water | Medium | High | Low |
| Frontier Fish | **High** | Medium | Low |
| Edge Fish | Medium-High | Medium | Medium |
| Adventurous Fish | **High** | Low | Medium |
| Structural Hallucination | None | Very Low | Very High |

## Using Categories in Practice

```python
from manifish import ManifoldFishAnalyzer

analyzer = ManifoldFishAnalyzer(reference_embeddings, reference_ids)
results = analyzer.analyze(generated_embeddings, generated_ids)

# Get structures by category
for r in results:
    if r.category == "frontier_fish":
        print(f"Priority candidate: {r.material_id}")
    elif r.category == "structural_hallucination":
        print(f"Reject: {r.material_id}")
```
