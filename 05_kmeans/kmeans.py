"""
K-Means Clustering – Basics of Machine Learning
=================================================
An unsupervised algorithm that groups data into k clusters by minimising
the within-cluster sum of squares (inertia).

Example: Cluster the Iris dataset without using the true labels, then
         compare the discovered clusters against the ground truth.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score


def elbow_analysis(X_scaled, k_range=range(1, 11)):
    """Compute inertia for each k to support the elbow method."""
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_
    return inertias


def print_cluster_summary(labels, true_labels, class_names):
    """Show how the discovered clusters map to the true species."""
    n_clusters = len(set(labels))
    print(f"\n  {'Cluster':<10}  " + "  ".join(f"{n:>10}" for n in class_names))
    for c in range(n_clusters):
        mask = labels == c
        counts = [np.sum((labels == c) & (true_labels == t)) for t in range(3)]
        row = "  ".join(f"{v:>10}" for v in counts)
        print(f"  Cluster {c:<3}  {row}")


def main():
    print("=" * 52)
    print("  K-Means Clustering – Iris Dataset")
    print("=" * 52)

    # 1. Load dataset (ignore labels during clustering)
    iris = load_iris()
    X, y_true = iris.data, iris.target
    class_names = [str(n) for n in iris.target_names]

    print(f"\nDataset  : Iris")
    print(f"Samples  : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")
    print(f"True classes (NOT used during clustering): {class_names}")

    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Elbow analysis to choose k
    print("\nElbow Analysis (Inertia vs k):")
    inertias = elbow_analysis(X_scaled)
    for k, inertia in inertias.items():
        bar = "█" * int(inertia / 15)
        print(f"  k={k:>2}  inertia={inertia:>8.2f}  {bar}")

    # 4. Fit with k=3 (matching the true number of species)
    k = 3
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    # 5. Evaluate clustering quality
    sil_score = silhouette_score(X_scaled, labels)
    ari = adjusted_rand_score(y_true, labels)

    print(f"\nClustering with k={k}:")
    print(f"  Inertia          : {model.inertia_:.4f}")
    print(f"  Silhouette Score : {sil_score:.4f}  (closer to 1.0 = better)")
    print(f"  Adjusted Rand Index vs true labels: {ari:.4f}  (1.0 = perfect)")

    # 6. Compare clusters to true labels
    print("\nCluster vs True Label Breakdown:")
    print_cluster_summary(labels, y_true, class_names)

    print("\nKey Takeaway:")
    print("  K-Means discovers structure without any labels.")
    print("  A high ARI means the clusters align well with the ground truth.\n")


if __name__ == "__main__":
    main()
