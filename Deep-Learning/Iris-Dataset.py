from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering

# Load Dataset
data = load_iris()
X, y_true = data.data, data.target 

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. List of Clustering Models
clustering_algos = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=3),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Spectral Clustering": SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
}

# Train & Evaluate
for name, algo in clustering_algos.items():
    cluster_labels = algo.fit_predict(X_scaled)
    print(f"{name}")
    print("Silhouette Score:", silhouette_score(X_scaled, cluster_labels))
    print("Adjusted Rand Index (vs true labels):", adjusted_rand_score(y_true, cluster_labels))
