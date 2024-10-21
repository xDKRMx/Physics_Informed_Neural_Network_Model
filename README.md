# Exploring Physics-Informed Neural Networks (PINNs) for 2D Linear Elasticity

## Overview

This project explores the use of Physics-Informed Neural Networks (PINNs) to solve a 2D linear elasticity problem. The goal is to predict displacement in a steel rod subjected to gravitational forces, using governing partial differential equations (PDEs) embedded in the PINN model. 

The model is trained to learn the solution of the forward problem, and then validated by comparing predicted displacement values with exact, analytically derived values. Results are presented in both numerical and graphical formats.

## Features

- **PINNs Architecture**: The model incorporates physical laws (PDEs) directly into its training process, ensuring compliance with governing equations of linear elasticity.
- **Dimensionalization**: Non-dimensionalization of inputs during training, followed by re-dimensionalization for accurate physical predictions.
- **Boundary Conditions**: Dirichlet boundary conditions are applied to the steel rod. The starting edge is fixed while the opposite edge is free.
- **Error Quantification**: The model's performance is assessed by comparing predicted displacements to exact values, analyzing maximum, minimum, and mean errors.

## Mesh Grid Sensitivity

- The model results are tested across various mesh grid densities, demonstrating that while mesh grids impact visualization resolution, they do not affect the accuracy of predicted values.
  
## Model Generalization

The model generalizes well to different sizes of steel rods with identical material properties, ensuring its applicability to a range of scenarios.

## Structure of the Project

1. **Model Definition**: Physics-informed neural network model using JAX, Optax, and Equinox.
2. **Training**: The model is trained using non-dimensional inputs, and predictions are re-dimensionalized for validation.
3. **Testing and Validation**: Displacement values in x and y directions are tested for different rod sizes and compared against exact solutions.
4. **Error Analysis**: Detailed error analysis is performed to ensure the model's accuracy and reliability.
  
## Running the Project

1. **Clone the repository**:
    ```bash
    git clone https://github.com/xDKRMx/Physics_Informed_Neural_Network_Model.git
    ```
   
2. **Install dependencies**:
    ```bash
    pip install jax equinox Matplotlib Scipy Numpy
    ```

3. **Run the notebook**:
    Load and execute the provided Jupyter notebook or Python scripts to train and test the model.

## Results

The model successfully predicts displacement values with high accuracy. Error analysis shows minimal deviation between predicted and exact results, confirming the reliability of the PINN approach in solving 2D elasticity problems.

## Requirements

- **JAX**: For automatic differentiation.
- **Matplotlib**: For plotting and visualizing the results.
- **Numpy**: For numerical operations.
- **Scipy**: For smoothing and analytical computation.
- **Equinox** : For neural network and automatic differentiation
  
## Conclusion

This project demonstrates the robustness of Physics-Informed Neural Networks (PINNs) in solving complex elasticity problems. With the integration of PDEs into the learning process, the model achieves highly accurate results across various scenarios and boundary conditions.
