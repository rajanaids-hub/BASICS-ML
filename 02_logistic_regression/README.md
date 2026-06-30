# Logistic Regression

Logistic Regression is a **classification** algorithm that predicts the probability that an input belongs to a particular class.

## How It Works

Despite its name, Logistic Regression outputs a **probability** using the sigmoid function:

```
σ(z) = 1 / (1 + e^(-z))    where z = w·x + b
```

The model is trained by minimising **binary cross-entropy loss**:

```
Loss = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Sigmoid function | Squashes output to (0, 1) – interpreted as probability |
| Decision boundary | Threshold (usually 0.5) that separates classes |
| Log-loss | Training objective that measures probabilistic predictions |
| Accuracy / F1 | Common evaluation metrics |

## Multi-class Classification

Use `multi_class='multinomial'` or `'ovr'` (one-vs-rest) in scikit-learn to extend logistic regression to more than two classes.

## When to Use

- Binary or multi-class classification
- When you need predicted probabilities
- Interpretable, fast baseline model

## Running the Example

```bash
python logistic_regression.py
```
