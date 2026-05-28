"""
app.py — Movie Recommendation System
Streamlit Web Application (no large .pkl needed)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_and_build():
    """Load movies and build cosine similarity matrix on startup."""
    try:
        movies = pd.read_csv('movies_processed.csv')
    except FileNotFoundError:
        st.error("movies_processed.csv not found. Make sure it is in the same folder as app.py")
        st.stop()

    movies_clean = movies.dropna(subset=['avg_rating']).copy()
    movies_clean = movies_clean[movies_clean['genres'] != '(no genres listed)']
    movies_clean = movies_clean.reset_index(drop=True)

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies_clean['genres'].str.split('|'))

    cosine_sim = cosine_similarity(genre_matrix)
    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=movies_clean['title'].values,
        columns=movies_clean['title'].values
    )
    return cosine_sim_df, movies_clean

def recommend_movies(movie_title, cosine_sim_df, movies, n=5):
    all_titles = cosine_sim_df.columns.tolist()
    if movie_title not in all_titles:
        matches = [t for t in all_titles if movie_title.lower() in t.lower()]
        if not matches:
            return None, None
        movie_title = matches[0]

    sim_scores = cosine_sim_df[movie_title].sort_values(ascending=False).iloc[1:n+1]
    rec_titles = sim_scores.index.tolist()
    result = movies[movies['title'].isin(rec_titles)][
        ['title', 'genres', 'avg_rating', 'rating_count']
    ].copy()
    result['similarity'] = result['title'].map(sim_scores)
    result = result.sort_values('similarity', ascending=False).reset_index(drop=True)
    result.index += 1
    return movie_title, result

# ── UI ───────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.markdown("**Built with MovieLens Dataset & Content-Based Filtering**")
st.markdown("---")

with st.spinner("Loading recommendation engine... (first load takes ~10 seconds)"):
    cosine_sim_df, movies = load_and_build()

st.success("Ready! Enter a movie name below.")

movie_input = st.text_input(
    "Movie Title",
    placeholder="e.g. Toy Story, The Dark Knight, Forrest Gump..."
)
num_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("Get Recommendations", use_container_width=True):
    if not movie_input.strip():
        st.warning("Please enter a movie title first.")
    else:
        matched_title, results = recommend_movies(
            movie_input.strip(), cosine_sim_df, movies, n=num_recs
        )
        if results is None:
            st.error(f"No movie found matching '{movie_input}'. Try a different name.")
        else:
            st.success(f"Showing recommendations similar to: **{matched_title}**")
            st.markdown("---")
            for i, row in results.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {row['title']}**")
                    st.caption(f"🎭 {row['genres']}")
                with col2:
                    if pd.notna(row['avg_rating']):
                        st.metric("Rating", f"⭐ {row['avg_rating']:.2f}")
                st.markdown("---")

with st.sidebar:
    st.markdown("### Popular movies to try")
    for movie in ["Toy Story (1995)", "The Dark Knight (2008)",
                  "Forrest Gump (1994)", "Pulp Fiction (1994)",
                  "Schindler's List (1993)"]:
        st.markdown(f"- {movie}")
    st.markdown("---")
    st.markdown("**Dataset:** MovieLens Latest Small")
    st.markdown("**Model:** Content-Based Filtering")

st.caption("Final Project — Movie Recommendation System")
