# Decision Trees

A Decision Tree learns a set of hierarchical **if-then-else** rules from training data. Each internal node tests a feature, each branch represents an outcome of that test, and each leaf node holds a prediction.

## How It Works

The tree is built by recursively splitting the data on the feature and threshold that maximally reduces **impurity**:

- **Gini impurity**: `G = 1 - Σ pᵢ²`
- **Entropy**: `H = -Σ pᵢ log₂(pᵢ)`

The split is chosen to maximise **Information Gain** (reduction in impurity).

## Key Concepts

| Concept | Description |
|---------|-------------|
| Root node | First decision point (best overall split) |
| Leaf node | Terminal node that holds the prediction |
| Max depth | Limits tree size to prevent overfitting |
| Pruning | Removing branches that add little predictive power |
| Feature importance | How much each feature reduces impurity on average |

## Advantages

- Highly interpretable – you can visualise and explain every decision
- Handles mixed feature types natively
- Requires little data preprocessing

## Limitations

- Prone to overfitting without depth limits
- Small data changes can produce very different trees (high variance)

## Running the Example

```bash
python decision_trees.py
```
