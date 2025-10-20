"""
Movie Data Collection from TMDb API
Collects comprehensive movie data for the recommender system
"""

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


class TMDbDataCollector:
    def __init__(self, api_key: str):
        """
        Initialize TMDb data collector

        Args:
            api_key: Your TMDb API key (get free at https://www.themoviedb.org/settings/api)
        """
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def get_popular_movies(self, pages: int = 50) -> list[dict]:
        """Fetch popular movies across multiple pages"""
        movies = []

        for page in range(1, pages + 1):
            url = f"{self.base_url}/movie/popular"
            params = {"page": page, "language": "en-US"}

            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                movies.extend(data.get("results", []))
                print(f"Fetched page {page}/{pages}")
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"Error fetching page {page}: {e}")

        return movies

    def get_movie_details(self, movie_id: int) -> dict:
        """Fetch detailed information for a specific movie"""
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            "append_to_response": "credits,keywords,recommendations",
            "language": "en-US",
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching movie {movie_id}: {e}")
            return {}

    def enrich_movie_data(
        self, movies: list[dict], max_movies: int = 500
    ) -> pd.DataFrame:
        """Enrich basic movie data with detailed information"""
        enriched_data = []

        for idx, movie in enumerate(movies[:max_movies]):
            movie_id = movie.get("id")
            print(f"Enriching movie {idx+1}/{max_movies}: {movie.get('title')}")

            details = self.get_movie_details(movie_id)
            if not details:
                continue

            # Extract relevant information
            enriched_movie = {
                "movie_id": movie_id,
                "title": details.get("title"),
                "overview": details.get("overview", ""),
                "release_date": details.get("release_date", ""),
                "vote_average": details.get("vote_average", 0),
                "vote_count": details.get("vote_count", 0),
                "popularity": details.get("popularity", 0),
                "runtime": details.get("runtime", 0),
                "budget": details.get("budget", 0),
                "revenue": details.get("revenue", 0),
                "genres": "|".join([g["name"] for g in details.get("genres", [])]),
                "director": self._extract_director(details.get("credits", {})),
                "cast": "|".join(
                    self._extract_cast(details.get("credits", {}), top_n=5)
                ),
                "keywords": "|".join(
                    [
                        k["name"]
                        for k in details.get("keywords", {}).get("keywords", [])[:10]
                    ]
                ),
                "poster_path": details.get("poster_path", ""),
            }

            enriched_data.append(enriched_movie)
            time.sleep(0.2)  # Rate limiting

        return pd.DataFrame(enriched_data)

    def _extract_director(self, credits: dict) -> str:
        """Extract director name from credits"""
        crew = credits.get("crew", [])
        for person in crew:
            if person.get("job") == "Director":
                return person.get("name", "")
        return ""

    def _extract_cast(self, credits: dict, top_n: int = 5) -> list[str]:
        """Extract top cast members"""
        cast = credits.get("cast", [])[:top_n]
        return [actor.get("name", "") for actor in cast]

    def collect_and_save(
        self,
        output_file: str = "movies_dataset.csv",
        pages: int = 50,
        max_movies: int = 500,
    ):
        """Main collection pipeline"""
        print("Starting data collection...")

        # Step 1: Get popular movies
        print("\n1. Fetching popular movies...")
        movies = self.get_popular_movies(pages=pages)
        print(f"Collected {len(movies)} movies")

        # Step 2: Enrich with detailed data
        print("\n2. Enriching movie data...")
        df = self.enrich_movie_data(movies, max_movies=max_movies)

        # Step 3: Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\n✓ Dataset saved to {output_file}")
        print(f"Total movies: {len(df)}")
        print("\nDataset preview:")
        print(df.head())

        return df


if __name__ == "__main__":
    # Get your free API key at: https://www.themoviedb.org/settings/api

    API_KEY = os.getenv("TMDB-API")
    collector = TMDbDataCollector(API_KEY)

    df = collector.collect_and_save(
        output_file="src/data/movies_dataset.csv",
        pages=50,  # 50 pages = ~1000 movies
        max_movies=1000,  # Enrich first 500 movies (adjust based on time)
    )
