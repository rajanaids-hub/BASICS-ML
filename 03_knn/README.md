# K-Nearest Neighbors (KNN)

KNN is a simple, non-parametric algorithm that classifies a new data point based on the majority class among its **k** nearest neighbours in the training set.

## How It Works

1. Choose a value for **k** (number of neighbours).
2. For each new point, calculate the distance to all training points.
3. Select the **k** closest training points.
4. Assign the majority class label (classification) or average value (regression).

Common distance metric: **Euclidean distance**
```
d(p, q) = √( Σ (pᵢ - qᵢ)² )
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| k | Number of neighbours to consider |
| Distance metric | How "closeness" is measured (Euclidean, Manhattan, …) |
| Lazy learner | No training phase; computation happens at prediction time |
| Curse of dimensionality | Performance degrades in high-dimensional spaces |

## Choosing k

- **Small k** (e.g., k=1): low bias, high variance → overfitting
- **Large k**: high bias, low variance → underfitting
- Use cross-validation to find the optimal k

## When to Use

- Small-to-medium datasets
- Non-linear decision boundaries
- When you need a simple, interpretable baseline

## Running the Example

```bash
python knn.py
```
