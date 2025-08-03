import streamlit as st
import pandas as pd
from recommend import get_similar_products

# Load product similarity matrix
import pickle
with open("product_similarity.pkl", "rb") as f:
    similarity_df = pickle.load(f)

product_list = list(similarity_df.columns)

st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("Product Recommendation System")

product_name = st.selectbox("Choose a product:", options=product_list)

if st.button("Get Recommendations"):
    recommendations = get_similar_products(product_name)
    if recommendations:
        st.success("Top 5 Recommended Products:")
        for i, prod in enumerate(recommendations, 1):
            st.write(f"{i}. {prod}")
    else:
        st.warning("Product not found or insufficient data.")
