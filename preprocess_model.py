print("Running preprocessing pipeline...")
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 1. Load Raw Dataset

df = pd.read_csv("online_retail.csv")  

# 2. Clean the Data

df.dropna(subset=['CustomerID'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 3. Create RFM Features

latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
})


# 4. Scale + Cluster with KMeans

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 5. Label Clusters (Human-Readable Segments)

cluster_profiles = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

label_map = {}
for cluster_id, row in cluster_profiles.iterrows():
    if row['Recency'] < 50 and row['Frequency'] > 10 and row['Monetary'] > 3000:
        label_map[cluster_id] = 'High-Value'
    elif row['Recency'] < 100 and row['Frequency'] > 2:
        label_map[cluster_id] = 'Regular'
    elif row['Recency'] > 200 and row['Frequency'] <= 1:
        label_map[cluster_id] = 'At-Risk'
    else:
        label_map[cluster_id] = 'Occasional'

rfm['Segment'] = rfm['Cluster'].map(label_map)

# 6. Save RFM with Cluster & Segment

rfm.to_csv("rfm.csv", index=False)


# 7. Create Product Similarity Matrix

item_matrix = df.pivot_table(
    index='CustomerID',
    columns='Description',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

item_similarity = cosine_similarity(item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=item_matrix.columns, columns=item_matrix.columns)

with open("product_similarity.pkl", "wb") as f:
    pickle.dump(item_similarity_df, f)


# 8. Show Segment Summary

segment_summary = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
print("\nCustomer Segment Profiles:")
print(segment_summary)
print("Preprocessing complete. Files saved: rfm.csv, product_similarity.pkl")
