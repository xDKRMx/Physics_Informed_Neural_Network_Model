# Strain, Stress, and Displacement Prediction Model

This project utilizes Physics-Informed Neural Networks (PINNs) to predict the strain, stress, and displacement in materials under various conditions. The model is designed to handle 1D and 2D cases, incorporating the physical principles of Hooke's Law for accurate predictions.

## Features
- Predicts strain, stress, and displacement for given materials.
- Considers the displacement accuracy based on the dimensional complexity, such as 1D,2D,3D.
- Provides graphical visualization of predictions.
- Seperable PINN system has been used
- Non-dimensionalization technics have been used
  
## Prerequisites
- Python 3.x
- JAX
- Equinox
- Optax
- Matplotlib
- Numpy

## Installation
```bash
git clone https://github.com/xDKRMx/Strain_Stress_Displacement.git
```
## Usage
- Define Displacement Functions: Modify u(x, y) and v(x, y) to define how displacement varies with coordinates.
- Compute Strain and Stress: Use the given formulas to calculate strain and stress components.
- Generate Data: Create a grid of points for which to compute the stress and strain.
- Plot Results: Visualize the computed stress components using contour plots.
