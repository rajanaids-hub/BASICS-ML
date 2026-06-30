"""
K-Nearest Neighbors (KNN) – Basics of Machine Learning
========================================================
Classifies a data point by looking at the k closest training examples
and taking a majority vote.

Example: Classify breast cancer tumours as malignant or benign using
         30 numeric features derived from cell nuclei measurements.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def find_best_k(X_train, y_train, k_range=range(1, 21)):
    """Use cross-validation to find the k with the highest mean accuracy."""
    scores = {}
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        scores[k] = cv_scores.mean()

    best_k = max(scores, key=scores.get)
    return best_k, scores


def main():
    print("=" * 58)
    print("  K-Nearest Neighbors – Breast Cancer Classification")
    print("=" * 58)

    # 1. Load dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    class_names = [str(n) for n in cancer.target_names]  # ['malignant', 'benign']

    print(f"\nDataset  : Breast Cancer Wisconsin")
    print(f"Samples  : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")
    print(f"Classes  : {class_names}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scale (critical for distance-based methods)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Find the best k via cross-validation
    print("\nSearching for the best k (1–20) using 5-fold cross-validation …")
    best_k, cv_scores = find_best_k(X_train_scaled, y_train)
    print(f"  Best k : {best_k}  (CV accuracy: {cv_scores[best_k]:.4f})")

    # 5. Train with best k and evaluate on test set
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy : {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 6. Show cross-validation accuracy across k values
    print("Cross-Validation Accuracy for k = 1 … 10:")
    for k in range(1, 11):
        bar = "█" * int(cv_scores[k] * 30)
        print(f"  k={k:>2}  {cv_scores[k]:.4f}  {bar}")

    print("\nKey Takeaway:")
    print("  KNN stores all training data and classifies by majority vote")
    print("  among the k nearest neighbours – no explicit training phase.\n")


if __name__ == "__main__":
    main()
