"""
Logistic Regression – Basics of Machine Learning
==================================================
Classifies inputs into discrete categories by predicting the probability
that each input belongs to a given class.

Example: Classify iris flowers into three species based on petal/sepal
         measurements (multi-class classification).
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("=" * 55)
    print("  Logistic Regression – Iris Flower Classification")
    print("=" * 55)

    # 1. Load the classic Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names

    print(f"\nDataset   : Iris")
    print(f"Samples   : {X.shape[0]}")
    print(f"Features  : {X.shape[1]}  ({', '.join(iris.feature_names)})")
    print(f"Classes   : {len(class_names)}  ({', '.join(class_names)})")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the model
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy : {accuracy:.4f} ({accuracy * 100:.1f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    header = "          " + "  ".join(f"{n:>10}" for n in class_names)
    print(header)
    for label, row in zip(class_names, cm):
        print(f"{label:>10}  " + "  ".join(f"{v:>10}" for v in row))

    # 6. Show predicted probabilities for a few test samples
    print("\nSample Predicted Probabilities (first 5 test rows):")
    proba = model.predict_proba(X_test_scaled[:5])
    header2 = f"{'Actual':>12}  {'Predicted':>12}  " + "  ".join(
        f"P({n})" for n in class_names
    )
    print(f"  {header2}")
    for actual, pred, prob in zip(y_test[:5], y_pred[:5], proba):
        prob_str = "  ".join(f"{p:.3f}" for p in prob)
        print(
            f"  {class_names[actual]:>12}  {class_names[pred]:>12}  {prob_str}"
        )

    print("\nKey Takeaway:")
    print("  Logistic Regression maps features → probability → class label.")
    print("  The sigmoid/softmax function ensures outputs sum to 1.0.\n")


if __name__ == "__main__":
    main()
