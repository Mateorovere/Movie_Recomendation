"""
Pytest tests for Movie Recommender System
"""

import pandas as pd

from src.ml.training import MovieRecommenderSystem


def test_data_loading(fresh_recommender: MovieRecommenderSystem) -> None:
    """Test data loading functionality"""
    fresh_recommender.load_data("src/data/movies_dataset.csv")

    assert fresh_recommender.movies_df is not None
    assert len(fresh_recommender.movies_df) > 0
    assert "title" in fresh_recommender.movies_df.columns
    assert "overview" in fresh_recommender.movies_df.columns
    assert "genres" in fresh_recommender.movies_df.columns
    assert "vote_average" in fresh_recommender.movies_df.columns
    assert "vote_count" in fresh_recommender.movies_df.columns

    # Check that data quality is reasonable
    assert fresh_recommender.movies_df["overview"].notna().sum() > 0
    assert fresh_recommender.movies_df["genres"].notna().sum() > 0


def test_feature_creation(fresh_recommender: MovieRecommenderSystem) -> None:
    """Test feature engineering"""
    fresh_recommender.load_data("src/data/movies_dataset.csv")
    fresh_recommender.create_content_features()

    assert fresh_recommender.tfidf_matrix is not None
    assert fresh_recommender.tfidf_matrix.shape[0] == len(fresh_recommender.movies_df)
    assert fresh_recommender.tfidf_matrix.shape[1] > 0
    assert "content" in fresh_recommender.movies_df.columns

    # Check that content was created for at least some movies
    assert fresh_recommender.movies_df["content"].notna().sum() > 0


def test_similarity_computation(recommender: MovieRecommenderSystem) -> None:
    """Test similarity matrix computation"""
    assert recommender.cosine_sim is not None
    assert recommender.cosine_sim.shape[0] == len(recommender.movies_df)
    assert recommender.cosine_sim.shape[1] == len(recommender.movies_df)

    # Check that diagonal is 1 (movie is identical to itself)
    import numpy as np

    diagonal = np.diag(recommender.cosine_sim)
    assert np.allclose(diagonal, 1.0, atol=1e-5)

    # Check that similarity values are in valid range [0, 1]
    assert recommender.cosine_sim.min() >= -0.01  # Allow small numerical errors
    assert recommender.cosine_sim.max() <= 1.01


def test_recommendations(recommender: MovieRecommenderSystem) -> None:
    """Test recommendation generation"""
    # Get a valid movie title from the dataset
    test_movie = recommender.movies_df.iloc[0]["title"]

    # Test content-based recommendations
    content_recs = recommender.get_content_recommendations(test_movie, n=5)
    assert isinstance(content_recs, pd.DataFrame)
    assert len(content_recs) <= 5
    assert len(content_recs) > 0
    assert "title" in content_recs.columns
    assert "similarity_score" in content_recs.columns

    # Verify the original movie is not in recommendations
    assert test_movie not in content_recs["title"].values

    # Test hybrid recommendations
    hybrid_recs = recommender.get_hybrid_recommendations(test_movie, n=5)
    assert isinstance(hybrid_recs, pd.DataFrame)
    assert len(hybrid_recs) <= 5
    assert len(hybrid_recs) > 0
    assert "title" in hybrid_recs.columns
    assert "hybrid_score" in hybrid_recs.columns

    # Verify scores are in valid range
    assert (hybrid_recs["hybrid_score"] >= 0).all()
    assert (hybrid_recs["hybrid_score"] <= 1).all()


def test_genre_recommendations(recommender: MovieRecommenderSystem) -> None:
    """Test genre-based recommendations"""
    test_genres = ["Action", "Comedy", "Drama"]

    for genre in test_genres:
        recs = recommender.get_genre_recommendations(genre, n=5)

        assert isinstance(recs, pd.DataFrame)
        assert len(recs) <= 5

        if len(recs) > 0:
            assert "title" in recs.columns
            assert "vote_average" in recs.columns

            # Verify all recommendations contain the requested genre
            for _, movie in recs.iterrows():
                assert genre in movie["genres"]


def test_invalid_movie_title(recommender: MovieRecommenderSystem) -> None:
    """Test handling of invalid movie titles"""
    result = recommender.get_content_recommendations("NonexistentMovie12345", n=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert result.empty


def test_edge_cases(recommender: MovieRecommenderSystem) -> None:
    """Test edge cases"""
    test_movie = recommender.movies_df.iloc[0]["title"]

    # Test with n=0
    recs = recommender.get_content_recommendations(test_movie, n=0)
    assert len(recs) == 0

    # Test with n=1
    recs = recommender.get_content_recommendations(test_movie, n=1)
    assert len(recs) <= 1

    # Test with very large n
    recs = recommender.get_content_recommendations(test_movie, n=10000)
    assert len(recs) < len(recommender.movies_df)
