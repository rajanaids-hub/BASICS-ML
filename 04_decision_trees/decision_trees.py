"""
Decision Trees – Basics of Machine Learning
============================================
Learns a set of hierarchical if-then-else rules from training data
to classify or predict new examples.

Example: Classify wine quality into categories using chemical properties.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report


def print_tree_rules(model, feature_names, max_depth=3):
    """Print a text representation of the decision tree rules."""
    rules = export_text(model, feature_names=list(feature_names), max_depth=max_depth)
    print(rules)


def print_feature_importance(model, feature_names, top_n=5):
    """Print the top-n most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"  Top {top_n} features by Gini importance:")
    for rank, idx in enumerate(indices[:top_n], start=1):
        bar = "█" * int(importances[idx] * 40)
        print(f"    {rank}. {feature_names[idx]:<30} {importances[idx]:.4f}  {bar}")


def main():
    print("=" * 52)
    print("  Decision Trees – Wine Classification")
    print("=" * 52)

    # 1. Load dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    class_names = [str(n) for n in wine.target_names]

    print(f"\nDataset  : Wine")
    print(f"Samples  : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")
    print(f"Classes  : {class_names}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train an unconstrained tree (likely to overfit)
    deep_tree = DecisionTreeClassifier(random_state=42)
    deep_tree.fit(X_train, y_train)
    y_pred_deep = deep_tree.predict(X_test)
    acc_deep = accuracy_score(y_test, y_pred_deep)

    # 4. Train a depth-limited tree (regularised)
    limited_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    limited_tree.fit(X_train, y_train)
    y_pred_limited = limited_tree.predict(X_test)
    acc_limited = accuracy_score(y_test, y_pred_limited)

    print(f"\nAccuracy comparison:")
    print(f"  Unconstrained tree (depth {deep_tree.get_depth()})  : {acc_deep:.4f}")
    print(f"  Depth-limited tree  (depth 4)   : {acc_limited:.4f}")

    # 5. Full report for the depth-limited tree
    print("\nClassification Report (depth-limited tree):")
    print(classification_report(y_test, y_pred_limited, target_names=class_names))

    # 6. Feature importances
    print("\nFeature Importances (depth-limited tree):")
    print_feature_importance(limited_tree, feature_names)

    # 7. Print first 3 levels of the tree rules
    print("\nDecision Rules (first 3 levels):")
    print_tree_rules(limited_tree, feature_names, max_depth=3)

    print("Key Takeaway:")
    print("  Limiting tree depth prevents overfitting and produces a model")
    print("  that generalises better to unseen data.\n")


if __name__ == "__main__":
    main()
