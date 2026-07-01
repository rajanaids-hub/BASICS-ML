# Linear Regression

Linear Regression is the simplest supervised learning algorithm. It models the relationship between a dependent variable and one or more independent variables by fitting a straight line (or hyperplane) through the data.

## How It Works

The model learns coefficients **w** and a bias **b** such that:

```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Training minimises the **Mean Squared Error (MSE)**:

```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Coefficients | Weight assigned to each feature |
| Intercept (bias) | Baseline prediction when all features are zero |
| R² Score | Proportion of variance explained by the model (1.0 = perfect) |
| MSE / RMSE | Average squared / root-mean-squared prediction error |

## When to Use

- Target variable is continuous
- Relationship between features and target is roughly linear
- Interpretability matters

## Running the Example

```bash
python linear_regression.py
```
