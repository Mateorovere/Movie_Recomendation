"""
Pytest configuration and fixtures for Movie Recommender System tests
"""

import pytest

from src.ml.training import MovieRecommenderSystem


@pytest.fixture(scope="module")
def recommender() -> MovieRecommenderSystem:
    """
    Create and configure a MovieRecommenderSystem instance for testing.
    Uses module scope so the recommender is created once per test module.
    """
    rec = MovieRecommenderSystem()
    rec.load_data("src/data/movies_dataset.csv")
    rec.create_content_features()
    rec.compute_similarity()
    return rec


@pytest.fixture(scope="function")
def fresh_recommender() -> MovieRecommenderSystem:
    """
    Create a fresh MovieRecommenderSystem instance for each test.
    Use this when you need an unconfigured recommender.
    """
    return MovieRecommenderSystem()
