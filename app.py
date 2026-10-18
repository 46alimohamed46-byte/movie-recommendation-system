import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

@st.cache_resource
def load_data():
    df = pd.read_pickle("movies_df.pkl")
    with open("tfidf_matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    return df, matrix

df, matrix = load_data()

def recommend(title, n=10):
    matches = df[df["title"] == title]
    if matches.empty:
        return []
    idx = matches["vote_count"].astype(float).idxmax()
    scores = linear_kernel(matrix[idx], matrix).flatten()
    top_idx = scores.argsort()[::-1][1:100]
    candidates = df.iloc[top_idx]
    candidates = candidates[candidates["vote_count"] >= 50]
    return candidates["title"].head(n).tolist()

st.title("🎬 Movie Recommendation System")
st.write("اختار فيلم وهنرشحلك أفلام شبهه")

movie_list = df["title"].sort_values().unique()
selected_movie = st.selectbox("اختار فيلم:", movie_list)

if st.button("رشّح لي 🎥"):
    results = recommend(selected_movie)
    if results:
        st.subheader("أفلام مقترحة:")
        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("مفيش نتايج كفاية لهذا الفيلم")