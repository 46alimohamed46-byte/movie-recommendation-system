import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import linear_kernel

TMDB_API_KEY = "5edcdbd29ff7f608cca52112cec96703"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0d0d0d;
}
h1 {
    color: #f5f5f5;
    text-align: center;
    font-weight: 700;
}
.stButton button {
    background-color: #d4af37;
    color: #000000;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 30px;
    border: none;
}
.stButton button:hover {
    background-color: #e5c158;
    color: #000000;
}
.movie-title {
    color: #d4af37;
    font-size: 1.3rem;
    font-weight: 700;
}
.stat-label {
    color: #cccccc;
}
hr {
    border-color: #d4af37;
}
</style>
""", unsafe_allow_html=True)

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
def get_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        data = requests.get(url).json()
        poster_path = data.get("poster_path")
        poster = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/300x450?text=No+Image"
        return {
            "poster": poster,
            "vote_average": data.get("vote_average", 0),
            "vote_count": data.get("vote_count", 0),
            "revenue": data.get("revenue", 0),
            "budget": data.get("budget", 0),
        }
    except:
        return {
            "poster": "https://via.placeholder.com/300x450?text=No+Image",
            "vote_average": 0, "vote_count": 0, "revenue": 0, "budget": 0,
        }

def format_money(amount):
    if amount == 0:
        return "N/A"
    if amount >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    return f"${amount:,}"

st.title("Movie Recommendation System")

movie_list = df["title"].sort_values().unique()
selected_movie = st.selectbox("Search for a movie:", movie_list)

if st.button("Recommend"):
    results = recommend(selected_movie)
    if len(results) > 0:
        for _, row in results.iterrows():
            details = get_movie_details(row["id"])
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(details["poster"], use_container_width=True)
            with col2:
                st.markdown(f"<p class='movie-title'>{row['title']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='stat-label'>Rating: {details['vote_average']} / 10</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='stat-label'>Votes: {details['vote_count']:,}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='stat-label'>Budget: {format_money(details['budget'])}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='stat-label'>Revenue: {format_money(details['revenue'])}</p>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.warning("No results found for this movie.")
