import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
print('running')
# Load the prepared RFM data
rfm = pd.read_csv("rfm.csv")

# Drop Segment/Cluster if re-running evaluations
rfm_eval = rfm[['Recency', 'Frequency', 'Monetary']].copy()

# Scale RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_eval)

# Elbow Curve 
sse = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, sse, marker='o')
plt.title('Elbow Curve for Optimal K (KMeans)')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE (Inertia)')
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_curve.png")
plt.show()

# Function to Evaluate Models
def evaluate_model(name, labels, data):
    if len(set(labels)) <= 1:
        print(f"\n{name}: Not a valid clustering (only 1 cluster)")
        return

    sil = silhouette_score(data, labels)
    ch = calinski_harabasz_score(data, labels)
    db = davies_bouldin_score(data, labels)

    print(f"\n {name} Evaluation:")
    print(f"  Silhouette Score:        {sil:.4f}")
    print(f"  Calinski-Harabasz Index: {ch:.2f}")
    print(f"  Davies-Bouldin Index:    {db:.4f}")

# Evaluate KMeans (k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(rfm_scaled)
evaluate_model("KMeans (k=4)", kmeans_labels, rfm_scaled)

# Evaluate Agglomerative
agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(rfm_scaled)
evaluate_model("Agglomerative (k=4)", agglo_labels, rfm_scaled)

# Evaluate DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(rfm_scaled)
evaluate_model("DBSCAN", dbscan_labels, rfm_scaled)

# Show DBSCAN label count
print("\nDBSCAN Cluster Distribution:", pd.Series(dbscan_labels).value_counts().to_dict())
