# K-Means Clustering

K-Means is the most popular **unsupervised learning** algorithm. It partitions data into **k** clusters where each data point belongs to the cluster with the nearest mean (centroid).

## How It Works

1. Randomly initialise **k** centroids.
2. **Assign** each point to the nearest centroid (Euclidean distance).
3. **Update** each centroid to be the mean of all points assigned to it.
4. Repeat steps 2–3 until centroids stop moving (convergence).

This minimises the **Within-Cluster Sum of Squares (WCSS)**:
```
WCSS = Σₖ Σ_{x ∈ Cₖ} ‖x - μₖ‖²
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Centroid | Mean position of all points in a cluster |
| WCSS / Inertia | Measure of cluster compactness (lower is better) |
| Elbow method | Heuristic to choose k by plotting WCSS vs k |
| Silhouette score | How well-separated the clusters are (−1 to +1) |

## Choosing k – The Elbow Method

Plot WCSS against different values of k. The "elbow" in the curve (where additional clusters give diminishing returns) suggests the optimal k.

## Limitations

- Must specify k in advance
- Assumes clusters are roughly spherical and equally sized
- Sensitive to outliers

## When to Use

- Customer segmentation
- Document / image clustering
- Exploratory data analysis

## Running the Example

```bash
python kmeans.py
```
