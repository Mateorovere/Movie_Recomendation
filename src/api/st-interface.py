"""
Movie Recommender Streamlit Interface
Professional web application for movie recommendations
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ml.training import MovieRecommenderSystem

# Page configuration
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "recommender" not in st.session_state:
    st.session_state.recommender = None
    st.session_state.loaded = False


@st.cache_resource
def load_recommender() -> MovieRecommenderSystem | None:
    """Load the trained recommender model"""
    try:
        recommender = MovieRecommenderSystem()
        recommender.load_model("src/ml/movie_recommender.pkl")
        return recommender
    except FileNotFoundError:
        return None


def display_movie_card(movie_row, show_score=True, score_col="hybrid_score") -> None:
    """Display a movie card with details"""
    col1, col2 = st.columns([1, 3])

    with col1:
        poster_url = f"https://image.tmdb.org/t/p/w500{movie_row['poster_path']}"
        if movie_row["poster_path"] and not pd.isna(movie_row["poster_path"]):
            st.image(poster_url, use_container_width=True)
        else:
            st.image(
                "https://via.placeholder.com/300x450?text=No+Poster",
                use_container_width=True,
            )

    with col2:
        st.markdown(f"### {movie_row['title']}")

        # Metrics
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Rating", f"⭐ {movie_row['vote_average']:.1f}/10")
        with metric_cols[1]:
            st.metric("Votes", f"{int(movie_row['vote_count']):,}")
        with metric_cols[2]:
            if show_score and score_col in movie_row:
                st.metric("Match Score", f"{movie_row[score_col]*100:.1f}%")

        # Genres
        genres = (
            movie_row["genres"].replace("|", ", ") if movie_row["genres"] else "N/A"
        )
        st.markdown(f"**Genres:** {genres}")

        # Overview
        overview = (
            movie_row["overview"] if movie_row["overview"] else "No overview available."
        )
        st.markdown(f"**Overview:** {overview[:300]}...")

        # Additional info
        if movie_row["director"]:
            st.markdown(f"**Director:** {movie_row['director']}")
        if movie_row["cast"]:
            cast = movie_row["cast"].replace("|", ", ")
            st.markdown(f"**Cast:** {cast}")


def main() -> None:
    # Header
    st.markdown(
        '<h1 class="main-header">🎬 AI Movie Recommender</h1>', unsafe_allow_html=True
    )
    st.markdown("---")

    # Load model
    if not st.session_state.loaded:
        with st.spinner("Loading AI model..."):
            st.session_state.recommender = load_recommender()
            st.session_state.loaded = True

    recommender = st.session_state.recommender

    if recommender is None:
        st.error(
            "❌ Model not found! Please train the model first using `movie_recommender_model.py`"
        )
        st.info("👉 Run the data collection and training scripts to create the model.")
        return

    # Sidebar
    with st.sidebar:
        st.header("🎯 Recommendation Settings")

        rec_type = st.selectbox(
            "Recommendation Type",
            ["Based on Movie", "By Genre", "Top Rated", "Dataset Explorer"],
        )

        num_recommendations = st.slider(
            "Number of Recommendations", min_value=5, max_value=20, value=10
        )

        st.markdown("---")
        st.header("📊 Dataset Stats")

        total_movies = len(recommender.movies_df)
        avg_rating = recommender.movies_df["vote_average"].mean()
        total_genres = len(
            set("|".join(recommender.movies_df["genres"].dropna()).split("|"))
        )

        st.metric("Total Movies", f"{total_movies:,}")
        st.metric("Average Rating", f"{avg_rating:.2f}/10")
        st.metric("Unique Genres", total_genres)

    # Main content
    if rec_type == "Based on Movie":
        st.header("🎥 Get Recommendations Based on a Movie")

        # Movie selection
        movie_titles = sorted(recommender.movies_df["title"].tolist())
        selected_movie = st.selectbox(
            "Select a movie you like:", options=movie_titles, index=0
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            get_recommendations = st.button(
                "🔍 Get Recommendations", type="primary", use_container_width=True
            )

        if get_recommendations and selected_movie:
            with st.spinner("Analyzing your preferences..."):
                recommendations = recommender.get_hybrid_recommendations(
                    selected_movie, n=num_recommendations
                )

            if not recommendations.empty:
                st.success(
                    f"✨ Found {len(recommendations)} movies similar to **{selected_movie}**"
                )

                # Display selected movie
                st.markdown("### 📌 Your Selected Movie")
                selected_movie_data = recommender.movies_df[
                    recommender.movies_df["title"] == selected_movie
                ].iloc[0]
                display_movie_card(selected_movie_data, show_score=False)

                st.markdown("---")
                st.markdown("### 🎯 Recommended for You")

                # Display recommendations
                for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                    with st.expander(
                        f"#{idx} - {movie['title']} ({movie['hybrid_score']*100:.1f}% match)",
                        expanded=(idx == 1),
                    ):
                        display_movie_card(
                            movie, show_score=True, score_col="hybrid_score"
                        )
            else:
                st.warning("No recommendations found. Try another movie!")

    elif rec_type == "By Genre":
        st.header("🎭 Explore Movies by Genre")

        # Extract all genres
        all_genres = set()
        for genres in recommender.movies_df["genres"].dropna():
            all_genres.update(genres.split("|"))

        selected_genre = st.selectbox("Select a genre:", options=sorted(all_genres))

        if st.button("🔍 Show Top Movies", type="primary"):
            with st.spinner(f"Finding best {selected_genre} movies..."):
                genre_movies = recommender.get_genre_recommendations(
                    selected_genre, n=num_recommendations
                )

            if not genre_movies.empty:
                st.success(f"✨ Top {len(genre_movies)} {selected_genre} movies")

                for idx, (_, movie) in enumerate(genre_movies.iterrows(), 1):
                    with st.expander(
                        f"#{idx} - {movie['title']} (Score: {movie['score']*100:.1f})",
                        expanded=(idx == 1),
                    ):
                        display_movie_card(movie, show_score=True, score_col="score")

    elif rec_type == "Top Rated":
        st.header("⭐ Highest Rated Movies")

        min_votes = st.slider("Minimum number of votes", 100, 1000, 500)

        top_movies = recommender.movies_df[
            recommender.movies_df["vote_count"] >= min_votes
        ].nlargest(num_recommendations, "vote_average")

        st.success(f"✨ Top {len(top_movies)} movies with at least {min_votes} votes")

        for idx, (_, movie) in enumerate(top_movies.iterrows(), 1):
            with st.expander(
                f"#{idx} - {movie['title']} (⭐ {movie['vote_average']:.1f}/10)",
                expanded=(idx == 1),
            ):
                display_movie_card(movie, show_score=False)

    else:  # Dataset Explorer
        st.header("📊 Dataset Explorer")

        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(
            ["📈 Rating Distribution", "🎭 Genre Analysis", "📅 Release Trends"]
        )

        with tab1:
            fig = px.histogram(
                recommender.movies_df,
                x="vote_average",
                nbins=20,
                title="Movie Rating Distribution",
                labels={"vote_average": "Rating", "count": "Number of Movies"},
                color_discrete_sequence=["#667eea"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Genre frequency
            genres_list = []
            for genres in recommender.movies_df["genres"].dropna():
                genres_list.extend(genres.split("|"))

            genre_counts = pd.Series(genres_list).value_counts().head(15)

            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation="h",
                title="Top 15 Movie Genres",
                labels={"x": "Number of Movies", "y": "Genre"},
                color=genre_counts.values,
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Extract year from release_date
            recommender.movies_df["year"] = pd.to_datetime(
                recommender.movies_df["release_date"], errors="coerce"
            ).dt.year

            year_counts = recommender.movies_df["year"].value_counts().sort_index()

            fig = px.line(
                x=year_counts.index,
                y=year_counts.values,
                title="Movies Released Per Year",
                labels={"x": "Year", "y": "Number of Movies"},
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Raw data
        with st.expander("📋 View Raw Dataset"):
            st.dataframe(
                recommender.movies_df[
                    ["title", "genres", "vote_average", "release_date", "popularity"]
                ],
                use_container_width=True,
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>Made with ❤️ using Streamlit | Powered by TMDb API</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
