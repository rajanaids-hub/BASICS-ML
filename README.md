# BASICS-ML

A beginner-friendly introduction to Machine Learning concepts with Python examples.

## Overview

This repository covers the foundational algorithms and concepts in Machine Learning, with clear, well-commented Python implementations.

## Topics Covered

| # | Topic | Description |
|---|-------|-------------|
| 1 | [Linear Regression](01_linear_regression/) | Predicting continuous values |
| 2 | [Logistic Regression](02_logistic_regression/) | Binary and multi-class classification |
| 3 | [K-Nearest Neighbors](03_knn/) | Instance-based classification |
| 4 | [Decision Trees](04_decision_trees/) | Tree-based classification & regression |
| 5 | [K-Means Clustering](05_kmeans/) | Unsupervised clustering |

## Prerequisites

- Python 3.8+
- Basic knowledge of Python programming
- Familiarity with NumPy arrays is helpful

## Installation

```bash
pip install -r requirements.txt
```

## Key Concepts

### Supervised Learning
Learning from labeled data to predict outcomes:
- **Regression** – predict a continuous value (e.g., house price)
- **Classification** – predict a category (e.g., spam / not spam)

### Unsupervised Learning
Finding patterns in unlabeled data:
- **Clustering** – group similar data points together
- **Dimensionality Reduction** – compress data while keeping structure

## Running the Examples

Each topic has its own folder with a self-contained Python script:

```bash
python 01_linear_regression/linear_regression.py
python 02_logistic_regression/logistic_regression.py
python 03_knn/knn.py
python 04_decision_trees/decision_trees.py
python 05_kmeans/kmeans.py
```

## Repository Structure

```
BASICS-ML/
├── README.md
├── requirements.txt
├── 01_linear_regression/
│   ├── README.md
│   └── linear_regression.py
├── 02_logistic_regression/
│   ├── README.md
│   └── logistic_regression.py
├── 03_knn/
│   ├── README.md
│   └── knn.py
├── 04_decision_trees/
│   ├── README.md
│   └── decision_trees.py
└── 05_kmeans/
    ├── README.md
    └── kmeans.py
```

## License

MIT