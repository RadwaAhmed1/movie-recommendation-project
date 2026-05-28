"""
app.py — Movie Recommendation System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import pickle
import os

# ─── Page Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ─── Load saved model files ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load the cosine similarity matrix and movie data."""
    with open('cosine_sim_df.pkl', 'rb') as f:
        cosine_sim_df = pickle.load(f)
    movies = pd.read_csv('movies_processed.csv')
    return cosine_sim_df, movies

try:
    cosine_sim_df, movies = load_models()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─── Recommendation Function ────────────────────────────────────────────
def recommend_movies(movie_title, n=5):
    """Return top N similar movies based on genre cosine similarity."""
    all_titles = cosine_sim_df.columns.tolist()

    # Partial match
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

# ─── App UI ─────────────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.markdown("**Built with MovieLens Dataset & Content-Based Filtering**")
st.markdown("---")

if not model_loaded:
    st.error(
        "⚠️ Model files not found! Please run the Jupyter notebook first to generate: "
        "`cosine_sim_df.pkl` and `movies_processed.csv`"
    )
else:
    st.markdown("### 🔍 Find Movies You'll Love")
    st.markdown("Enter any movie title below (partial names work too!)")

    # ── Input box ─────────────────────────────────────────────────────
    movie_input = st.text_input(
        "Movie Title",
        placeholder="e.g. Toy Story, The Dark Knight, Forrest Gump..."
    )

    num_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    # ── Search button ──────────────────────────────────────────────────
    if st.button("🎯 Get Recommendations", use_container_width=True):
        if not movie_input.strip():
            st.warning("Please enter a movie title first.")
        else:
            matched_title, results = recommend_movies(movie_input.strip(), n=num_recs)

            if results is None:
                st.error(
                    f"❌ No movie found matching **'{movie_input}'**.\n\n"
                    "Try a different name or check your spelling."
                )
            else:
                st.success(f"✅ Showing recommendations similar to: **{matched_title}**")
                st.markdown("---")

                for i, row in results.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{i}. {row['title']}**")
                            st.caption(f"🎭 {row['genres']}")
                        with col2:
                            if pd.notna(row['avg_rating']):
                                st.metric("Rating", f"⭐ {row['avg_rating']:.2f}")
                        st.markdown("---")

    # ── Popular movies sidebar ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🍿 Popular Movies to Try")
        popular = [
            "Toy Story (1995)",
            "The Dark Knight (2008)",
            "Forrest Gump (1994)",
            "The Silence of the Lambs (1991)",
            "Schindler's List (1993)",
            "Pulp Fiction (1994)",
        ]
        for movie in popular:
            if st.button(movie, key=movie):
                st.session_state['selected'] = movie

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This app uses **Content-Based Filtering** with cosine similarity "
            "on movie genres to find similar movies."
        )
        st.markdown("Dataset: **MovieLens Latest Small**")

st.markdown("---")
st.caption("Final Project — Movie Recommendation System | Data Science Course")
