
# K-Means Clustering

## 1. What is K-Means Clustering?

K-Means is an unsupervised Machine Learning algorithm used to group similar data points into clusters based on their characteristics.

It divides a dataset into K distinct clusters, where each data point belongs to the cluster with the nearest centroid.

The objective is to minimize the distance between data points and their assigned cluster centroids.

Examples:
- Customer segmentation
- Grouping products based on purchasing behavior
- Image compression
- Grouping users based on activity patterns

## 2. How K-Means Works

1. Choose the number of clusters (K).
2. Initialize K cluster centroids.
3. Calculate the distance between each data point and all centroids.
4. Assign each point to its nearest centroid.
5. Recalculate the centroid of each cluster by taking the mean of its assigned points.
6. Repeat the assignment and centroid update steps.
7. Stop when cluster assignments or centroids no longer change significantly.

## 3. Important Terms

| Term | Description |
|---|---|
| Cluster | Group of similar data points |
| Centroid | Mean position of all points in a cluster |
| K | Number of clusters |
| Inertia | Sum of squared distances between points and their assigned centroids |
| Convergence | Point at which the algorithm stops updating clusters |

## 4. Distance Metric

K-Means commonly uses Euclidean distance to determine the nearest centroid.

Euclidean Distance:

d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

The algorithm assigns each point to the centroid with the smallest distance.

## 5. Choosing the Optimal K

Selecting the correct number of clusters is important.

### Elbow Method

The Elbow Method helps identify a suitable K by calculating the inertia for different cluster counts.

Steps:
1. Train K-Means with different K values.
2. Calculate inertia for each K.
3. Plot K against inertia.
4. Identify the point where the reduction in inertia begins to slow down significantly.

This point resembles an elbow and can be a useful choice for K.

### Silhouette Score

Measures how well each data point fits within its assigned cluster compared with other clusters.

Score ranges from -1 to +1.

- Close to +1: Well-separated clusters.
- Around 0: Overlapping clusters.
- Close to -1: Potentially incorrect cluster assignments.

## 6. Feature Scaling

Feature scaling is important because K-Means relies on distance calculations.

Features with larger numerical ranges can dominate the clustering process.

Common techniques:
- StandardScaler
- MinMaxScaler

## 7. Advantages and Disadvantages

### Advantages

- Simple and easy to implement.
- Fast and efficient on many datasets.
- Works well for compact, well-separated clusters.
- Easy to interpret and visualize.

### Disadvantages

- Requires the number of clusters (K) beforehand.
- Sensitive to initial centroid positions.
- Sensitive to outliers.
- Performs poorly with irregularly shaped clusters.
- Requires numerical features and suitable distance calculations.

## 8. Important Hyperparameters

| Parameter | Description |
|---|---|
| n_clusters | Number of clusters (K) |
| init | Centroid initialization method |
| n_init | Number of initializations |
| max_iter | Maximum iterations for each initialization |
| random_state | Controls randomness |
| tol | Convergence tolerance |
| algorithm | Algorithm used to perform clustering |

K-Means++ is a commonly used initialization method that helps select better starting centroids.

## 9. Python Implementation

### K-Means using the Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
model = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

# Train and predict clusters
labels = model.fit_predict(X_scaled)

# Evaluate clustering
print("Inertia:", model.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Visualize clusters using first two features
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis"
)

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200,
    color="red",
    label="Centroids"
)

plt.title("K-Means Clustering")
plt.legend()
plt.show()
```

Note: Iris is a labeled dataset, but K-Means ignores its target labels during clustering. The example uses the first two features for visualization.

## 10. Key Takeaways

- K-Means is an unsupervised learning algorithm.
- It divides data into K clusters.
- It uses centroids to represent clusters.
- It commonly uses Euclidean distance.
- The Elbow Method and Silhouette Score help evaluate cluster choices.
- Feature scaling is important.
- K-Means works best with compact, reasonably separated clusters.

---


