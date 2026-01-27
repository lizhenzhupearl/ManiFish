# Basic Usage Tutorial

## Overview

This tutorial covers the basic workflow of ManiFish.

## Step 1: Select Anchors

```python
import manifish as mf

# Select 200 diverse anchors from Materials Project
anchors = mf.select_anchors(
    database='materials_project',
    n_anchors=200,
    method='fps',  # Farthest Point Sampling
    stability_threshold=0.1  # E_hull < 0.1 eV/atom
)

print(f"Selected {len(anchors)} anchors")
```

## Step 2: Load MLIPs

```python
from manifish.models import load_mlip

mlips = [
    load_mlip('chgnet'),
    load_mlip('mace'),
    load_mlip('m3gnet'),
]
```

## Step 3: Evaluate Structures

```python
from ase.io import read

# Load your structure
structure = read('structure.cif')

# Create evaluator
evaluator = mf.ManifoldEvaluator(
    anchors=anchors,
    mlips=mlips,
)

# Evaluate
results = evaluator.evaluate(structure)

print(f"Stability:  {results['stability']:.3f}")
print(f"Novelty:    {results['novelty']:.3f}")
print(f"Consensus:  {results['consensus']:.3f}")
```

## Step 4: Interpret Results

```python
if results['stability'] > 0.8:
    print("Structure is likely stable!")
    
if results['novelty'] > 0.6:
    print("Structure is novel!")
    
if results['consensus'] < 0.5:
    print("Warning: Low consensus, needs validation")
```

## Next Steps

- [Batch Evaluation](batch_evaluation.md)
- [Custom Metrics](custom_metrics.md)
- [Visualization](visualization.md)
