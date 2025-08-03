import pickle

# Load similarity matrix
with open("product_similarity.pkl", "rb") as f:
    similarity_df = pickle.load(f)

def get_similar_products(product_name, n=5):
    if product_name not in similarity_df.columns:
        return []
    similar_scores = similarity_df[product_name].sort_values(ascending=False)
    return similar_scores.iloc[1:n+1].index.tolist()
