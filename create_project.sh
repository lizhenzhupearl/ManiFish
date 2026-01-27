#!/bin/bash

# Create ManiFish project structure
echo "🐟 Creating ManiFish project structure..."

# Main directories
mkdir -p manifish/{core,models,utils,data,visualization}
mkdir -p tests/{unit,integration}
mkdir -p docs/{images,tutorials,api}
mkdir -p examples
mkdir -p scripts

# Create __init__.py files
touch manifish/__init__.py
touch manifish/core/__init__.py
touch manifish/models/__init__.py
touch manifish/utils/__init__.py
touch manifish/data/__init__.py
touch manifish/visualization/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

echo "✅ Directory structure created!"
