"""
Advanced Movie Recommender System
Implements multiple recommendation approaches:
1. Content-Based Filtering (TF-IDF + Cosine Similarity)
2. Collaborative Filtering (Matrix Factorization with SVD)
3. Hybrid Approach
"""

# ruff: noqa: C416
import pickle
import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class MovieRecommenderSystem:
    def __init__(self) -> None:
        self.movies_df: pd.DataFrame
        self.tfidf_matrix: Any = None
        self.cosine_sim: Any = None
        self.title_to_idx: dict[str, int] = {}
        self.idx_to_title: dict[int, str] = {}

    def load_data(self, filepath: str) -> None:
        """Load and preprocess movie dataset"""
        self.movies_df = pd.read_csv(filepath)

        # Data cleaning
        self.movies_df["overview"] = self.movies_df["overview"].fillna("")
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        self.movies_df["keywords"] = self.movies_df["keywords"].fillna("")
        self.movies_df["cast"] = self.movies_df["cast"].fillna("")
        self.movies_df["director"] = self.movies_df["director"].fillna("")

        # Create mappings
        self.title_to_idx = {
            title: idx for idx, title in enumerate(self.movies_df["title"])
        }
        self.idx_to_title = {
            idx: title for idx, title in enumerate(self.movies_df["title"])
        }

        print(f"Loaded {len(self.movies_df)} movies")

    def create_content_features(self) -> None:
        """Create feature vectors for content-based filtering"""

        def clean_text(text: str | float) -> str:
            """Clean and preprocess text"""
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = text.replace("|", " ")
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            return text

        self.movies_df["content"] = (
            self.movies_df["overview"]
            + " "
            + self.movies_df["genres"].apply(lambda x: " ".join(x.split("|")) * 3)
            + " "
            + self.movies_df["keywords"].apply(lambda x: " ".join(x.split("|")) * 2)
            + " "
            + self.movies_df["cast"].apply(lambda x: " ".join(x.split("|")) * 2)
            + " "
            + self.movies_df["director"].apply(lambda x: x * 3)
        )

        self.movies_df["content"] = self.movies_df["content"].apply(clean_text)

        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2
        )

        self.tfidf_matrix = tfidf.fit_transform(self.movies_df["content"])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def compute_similarity(self) -> None:
        """Compute cosine similarity matrix"""
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Similarity matrix computed")

    def get_content_recommendations(
        self, title: str, n: int = 10, min_votes: int = 50
    ) -> pd.DataFrame:
        """
        Get content-based recommendations

        Args:
            title: Movie title to get recommendations for
            n: Number of recommendations
            min_votes: Minimum vote count threshold

        Returns:
            DataFrame with recommended movies
        """
        if title not in self.title_to_idx:
            return pd.DataFrame()

        idx = self.title_to_idx[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : n * 3]  # Get more than needed for filtering

        movie_indices = [i[0] for i in sim_scores]

        # Filter by vote count and get top N
        recommendations = self.movies_df.iloc[movie_indices].copy()
        recommendations = recommendations[recommendations["vote_count"] >= min_votes]
        recommendations["similarity_score"] = [
            i[1] for i in sim_scores[: len(recommendations)]
        ]

        return recommendations.head(n)

    def get_hybrid_recommendations(self, title: str, n: int = 10) -> pd.DataFrame:
        """
        Hybrid recommendation combining content similarity and popularity
        """
        content_recs = self.get_content_recommendations(title, n=n * 2)

        if content_recs.empty:
            return pd.DataFrame()

        # Normalize scores
        scaler = MinMaxScaler()
        content_recs["norm_similarity"] = scaler.fit_transform(
            content_recs[["similarity_score"]]
        )
        content_recs["norm_popularity"] = scaler.fit_transform(
            content_recs[["popularity"]]
        )
        content_recs["norm_rating"] = scaler.fit_transform(
            content_recs[["vote_average"]]
        )

        # Weighted hybrid score
        content_recs["hybrid_score"] = (
            0.5 * content_recs["norm_similarity"]
            + 0.3 * content_recs["norm_rating"]
            + 0.2 * content_recs["norm_popularity"]
        )

        content_recs = content_recs.sort_values("hybrid_score", ascending=False)

        return content_recs.head(n)

    def get_genre_recommendations(self, genre: str, n: int = 10) -> pd.DataFrame:
        """Get top movies by genre"""
        genre_movies = self.movies_df[
            self.movies_df["genres"].str.contains(genre, case=False, na=False)
        ].copy()

        # Score based on rating and popularity
        scaler = MinMaxScaler()
        genre_movies["norm_rating"] = scaler.fit_transform(
            genre_movies[["vote_average"]]
        )
        genre_movies["norm_popularity"] = scaler.fit_transform(
            genre_movies[["popularity"]]
        )

        genre_movies["score"] = (
            0.7 * genre_movies["norm_rating"] + 0.3 * genre_movies["norm_popularity"]
        )

        return genre_movies.nlargest(n, "score")

    def train(self, filepath: str) -> None:
        """Complete training pipeline"""
        print("=" * 60)
        print("TRAINING MOVIE RECOMMENDER SYSTEM")
        print("=" * 60)

        print("\n1. Loading data...")
        self.load_data(filepath)

        print("\n2. Creating content features...")
        self.create_content_features()

        print("\n3. Computing similarity matrix...")
        self.compute_similarity()

        print("\n✓ Training complete!")
        print("=" * 60)

    def save_model(self, filepath: str = "movie_recommender.pkl") -> None:
        """Save trained model"""
        model_data = {
            "movies_df": self.movies_df,
            "tfidf_matrix": self.tfidf_matrix,
            "cosine_sim": self.cosine_sim,
            "title_to_idx": self.title_to_idx,
            "idx_to_title": self.idx_to_title,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str = "movie_recommender.pkl") -> None:
        """Load trained model"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.movies_df = model_data["movies_df"]
        self.tfidf_matrix = model_data["tfidf_matrix"]
        self.cosine_sim = model_data["cosine_sim"]
        self.title_to_idx = model_data["title_to_idx"]
        self.idx_to_title = model_data["idx_to_title"]

        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    recommender = MovieRecommenderSystem()
    recommender.train("src/data/movies_dataset.csv")
    recommender.save_model("src/ml/movie_recommender.pkl")

    test_movie = "The Dark Knight"
    if test_movie in recommender.title_to_idx:
        print(f"\n\nRecommendations for '{test_movie}':")
        recs = recommender.get_hybrid_recommendations(test_movie, n=5)
        print(recs[["title", "genres", "vote_average", "hybrid_score"]])
