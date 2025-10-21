"""
Test and Demo Script for Movie Recommender System
Tests all components and generates sample recommendations
"""

import time

import pandas as pd

from src.ml.training import MovieRecommenderSystem


def print_section(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def test_data_loading(recommender: MovieRecommenderSystem) -> bool:
    """Test data loading functionality"""
    print_section("TEST 1: Data Loading")

    try:
        recommender.load_data("movies_dataset.csv")
        print("✓ Dataset loaded successfully")
        print(f"  - Total movies: {len(recommender.movies_df)}")
        print(f"  - Columns: {', '.join(recommender.movies_df.columns.tolist())}")

        # Data quality checks
        print("\n📊 Data Quality Checks:")
        print(
            f"  - Missing overviews: {recommender.movies_df['overview'].isna().sum()}"
        )
        print(f"  - Missing genres: {recommender.movies_df['genres'].isna().sum()}")
        print(
            f"  - Average rating: {recommender.movies_df['vote_average'].mean():.2f}/10"
        )
        print(
            f"  - Movies with 100+ votes: {(recommender.movies_df['vote_count'] >= 100).sum()}"
        )

        return True
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def test_feature_creation(recommender: MovieRecommenderSystem) -> bool:
    """Test feature engineering"""
    print_section("TEST 2: Feature Engineering")

    try:
        recommender.create_content_features()
        print("✓ Content features created successfully")
        print(f"  - TF-IDF matrix shape: {recommender.tfidf_matrix.shape}")
        print(f"  - Number of features: {recommender.tfidf_matrix.shape[1]}")
        print(
            f"  - Matrix density: {(recommender.tfidf_matrix.nnz / (recommender.tfidf_matrix.shape[0] * recommender.tfidf_matrix.shape[1]) * 100):.2f}%"
        )

        # Show sample content
        sample_idx = 0
        print(
            f"\n📝 Sample content for '{recommender.movies_df.iloc[sample_idx]['title']}':"
        )
        print(f"  {recommender.movies_df.iloc[sample_idx]['content'][:200]}...")

        return True
    except Exception as e:
        print(f"✗ Error creating features: {e}")
        return False


def test_similarity_computation(recommender: MovieRecommenderSystem) -> bool:
    """Test similarity matrix computation"""
    print_section("TEST 3: Similarity Matrix Computation")

    try:
        start_time = time.time()
        recommender.compute_similarity()
        elapsed = time.time() - start_time

        print("✓ Similarity matrix computed successfully")
        print(f"  - Matrix shape: {recommender.cosine_sim.shape}")
        print(f"  - Computation time: {elapsed:.2f} seconds")
        print(f"  - Memory size: {recommender.cosine_sim.nbytes / (1024*1024):.2f} MB")

        # Sample similarity scores
        sample_idx = 0
        sample_similarities = recommender.cosine_sim[sample_idx]
        print(f"\n🔍 Similarity stats for movie at index {sample_idx}:")
        print(f"  - Mean similarity: {sample_similarities.mean():.4f}")
        print(f"  - Max similarity: {sample_similarities.max():.4f}")
        print(f"  - Min similarity: {sample_similarities.min():.4f}")

        return True
    except Exception as e:
        print(f"✗ Error computing similarity: {e}")
        return False


def test_recommendations(recommender: MovieRecommenderSystem) -> bool:
    """Test recommendation generation"""
    print_section("TEST 4: Recommendation Generation")

    # Test movies
    test_movies = [
        recommender.movies_df.iloc[0]["title"],
        recommender.movies_df.iloc[10]["title"],
        recommender.movies_df.iloc[50]["title"],
    ]

    for test_movie in test_movies:
        if test_movie not in recommender.title_to_idx:
            continue

        print(f"\n🎬 Testing recommendations for: '{test_movie}'")

        try:
            # Content-based recommendations
            start_time = time.time()
            content_recs = recommender.get_content_recommendations(test_movie, n=5)
            content_time = time.time() - start_time

            print(
                f"  ✓ Content-based: Generated {len(content_recs)} recommendations in {content_time:.3f}s"
            )

            # Hybrid recommendations
            start_time = time.time()
            hybrid_recs = recommender.get_hybrid_recommendations(test_movie, n=5)
            hybrid_time = time.time() - start_time

            print(
                f"  ✓ Hybrid: Generated {len(hybrid_recs)} recommendations in {hybrid_time:.3f}s"
            )

            # Display top 3
            print("\n  📋 Top 3 Hybrid Recommendations:")
            for idx, (_, movie) in enumerate(hybrid_recs.head(3).iterrows(), 1):
                genres = movie["genres"].replace("|", ", ")
                print(f"    {idx}. {movie['title']} ({movie['vote_average']:.1f}/10)")
                print(f"       Genres: {genres}")
                print(f"       Match: {movie['hybrid_score']*100:.1f}%\n")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    return True


def test_genre_recommendations(recommender: MovieRecommenderSystem) -> bool:
    """Test genre-based recommendations"""
    print_section("TEST 5: Genre-Based Recommendations")

    test_genres = ["Action", "Comedy", "Drama", "Sci-Fi"]

    for genre in test_genres:
        try:
            recs = recommender.get_genre_recommendations(genre, n=5)
            print(f"\n🎭 Top {len(recs)} {genre} movies:")

            for idx, (_, movie) in enumerate(recs.head(3).iterrows(), 1):
                print(
                    f"  {idx}. {movie['title']} - ⭐ {movie['vote_average']:.1f}/10 ({int(movie['vote_count'])} votes)"
                )

        except Exception as e:
            print(f"  ✗ Error for genre {genre}: {e}")

    return True


def generate_demo_report(recommender: MovieRecommenderSystem) -> None:
    """Generate a comprehensive demo report"""
    print_section("DEMO REPORT: Sample Recommendations")

    # Pick a popular movie for demo
    popular_movies = recommender.movies_df.nlargest(10, "vote_count")
    demo_movie = popular_movies.iloc[0]["title"]

    print(f"🎯 DEMO: Recommendations based on '{demo_movie}'\n")

    # Get movie details
    movie_data = recommender.movies_df[
        recommender.movies_df["title"] == demo_movie
    ].iloc[0]
    print("📌 Selected Movie Details:")
    print(f"  Title: {movie_data['title']}")
    print(f"  Genres: {movie_data['genres'].replace('|', ', ')}")
    print(
        f"  Rating: ⭐ {movie_data['vote_average']:.1f}/10 ({int(movie_data['vote_count'])} votes)"
    )
    print(f"  Director: {movie_data['director']}")
    print(f"  Overview: {movie_data['overview'][:200]}...")

    # Generate recommendations
    recommendations = recommender.get_hybrid_recommendations(demo_movie, n=10)

    print("\n🎬 Top 10 Recommended Movies:\n")
    print(f"{'Rank':<6}{'Title':<40}{'Rating':<10}{'Match':<10}{'Genres'}")
    print("-" * 100)

    for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
        title = (
            movie["title"][:37] + "..." if len(movie["title"]) > 40 else movie["title"]
        )
        rating = f"⭐ {movie['vote_average']:.1f}"
        match = f"{movie['hybrid_score']*100:.1f}%"
        genres = movie["genres"].replace("|", ", ")[:30]

        print(f"{idx:<6}{title:<40}{rating:<10}{match:<10}{genres}")

    # Genre analysis
    print("\n📊 Recommendation Analytics:")

    all_genres = []
    for genres in recommendations["genres"]:
        all_genres.extend(genres.split("|"))

    genre_counts = pd.Series(all_genres).value_counts()
    print("\n  Genre Distribution:")
    for genre, count in genre_counts.head(5).items():
        print(f"    - {genre}: {count} movies")

    avg_rating = recommendations["vote_average"].mean()
    avg_votes = recommendations["vote_count"].mean()

    print(f"\n  Average Recommendation Rating: ⭐ {avg_rating:.2f}/10")
    print(f"  Average Vote Count: {int(avg_votes):,}")
    print(f"  Diversity Score: {len(genre_counts)} unique genres")


def run_full_test_suite() -> None:
    """Run complete test suite"""
    print("\n" + "█" * 70)
    print("  MOVIE RECOMMENDER SYSTEM - COMPREHENSIVE TEST SUITE")
    print("█" * 70)

    recommender = MovieRecommenderSystem()

    # Run tests
    tests = [
        ("Data Loading", lambda: test_data_loading(recommender)),
        ("Feature Engineering", lambda: test_feature_creation(recommender)),
        ("Similarity Computation", lambda: test_similarity_computation(recommender)),
        ("Recommendations", lambda: test_recommendations(recommender)),
        ("Genre Filtering", lambda: test_genre_recommendations(recommender)),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            results.append((test_name, f"ERROR: {str(e)[:50]}"))

    # Generate demo report
    try:
        generate_demo_report(recommender)
    except Exception as e:
        print(f"\n✗ Demo report generation failed: {e}")

    print_section("TEST SUMMARY")
    print(f"{'Test Name':<30}{'Result'}")
    print("-" * 50)
    for test_name, result in results:
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name:<28}{result}")

    passed = sum(1 for _, r in results if r == "PASSED")
    total = len(results)

    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    # Save model
    print_section("MODEL PERSISTENCE")
    try:
        recommender.save_model("movie_recommender.pkl")
        print("✓ Model saved successfully!")
        print("  You can now run: streamlit run streamlit_app.py")
    except Exception as e:
        print(f"✗ Error saving model: {e}")

    print("\n" + "█" * 70)
    print("  TEST SUITE COMPLETE")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    run_full_test_suite()
