import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import linear_kernel

TMDB_API_KEY = "ضع_المفتاح_هنا"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

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
    return candidates[["id", "title"]].head(n)

@st.cache_data
def get_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url).json()
        poster_path = response.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return "https://via.placeholder.com/300x450?text=No+Image"

st.title("Movie Recommendation System")

movie_list = df["title"].sort_values().unique()
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    results = recommend(selected_movie)
    if len(results) > 0:
        cols = st.columns(5)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 5]:
                st.image(get_poster(row["id"]), use_container_width=True)
                st.caption(row["title"])
    else:
        st.warning("No results found for this movie.")
