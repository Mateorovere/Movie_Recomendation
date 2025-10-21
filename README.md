# 🎬 AI Movie Recommender System

<div align="center">

**A production-ready movie recommendation system powered by machine learning and an interactive Streamlit interface.**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture)

</div>

---

## 🌟 Features

### 🤖 **Intelligent Recommendations**
- **Content-Based Filtering**: TF-IDF vectorization with cosine similarity
- **Hybrid Scoring System**: Combines content similarity, ratings, and popularity
- **Genre-Based Discovery**: Explore top movies by genre
- **Multi-Signal Ranking**: Weighted scoring algorithm for optimal results

### 🎨 **Interactive Web Interface**
- **Modern UI**: Gradient designs with responsive layout
- **Real-Time Search**: Instant movie recommendations
- **Visual Analytics**: Interactive charts and statistics
- **Movie Cards**: Rich metadata display with posters, ratings, and cast

### 📊 **Data Analytics Dashboard**
- Rating distribution histograms
- Genre frequency analysis
- Release year trends
- Dataset explorer with raw data view

### 🛠️ **Developer-Friendly**
- Modern Python tooling (UV, Ruff, Black, MyPy)
- Pre-commit hooks for code quality
- Type hints throughout codebase
- Comprehensive test setup with pytest
- Dev container support for consistent environments

---

## 🚀 Installation

### Prerequisites

- TMDb API key (free at [themoviedb.org](https://www.themoviedb.org/settings/api))


### Option 1: Dev Container (VS Code) (Recommended)

Simply open the project in VS Code and select **"Reopen in Container"** when prompted and open a new Terminal. Everything will be configured automatically!

### Option 2: Using UV

```bash
# Clone the repository
git clone https://github.com/Mateorovere/Movie_Recomendation.git
cd Movie_Recomendation

# Install uv
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --dev

# Install pre-commit hooks
pre-commit install
```

---

## ⚡ Quick Start

### Step 1: Configure API Key

Create a `.env` file in the project root:

```bash
TMDB-API=your_api_key_here
```

Get your free API key at [TMDb API Settings](https://www.themoviedb.org/settings/api)

### Step 2: Collect Movie Data

```bash
python src/data/collector.py
```

**What it does:**
- Fetches 1,000+ popular movies from TMDb API
- Enriches data with cast, crew, genres, and keywords
- Saves to `src/data/movies_dataset.csv`
- ⏱️ Duration: ~15-30 minutes (API rate limiting)

**Customization:**
```python
# Edit collector.py to adjust parameters
collector.collect_and_save(
    output_file="src/data/movies_dataset.csv",
    pages=50,        # 50 pages ≈ 1000 movies
    max_movies=1000  # Number to enrich
)
```

### Step 3: Train the Model

```bash
python src/ml/training.py
```

**Training Pipeline:**
1. Loads dataset (CSV)
2. Creates TF-IDF feature vectors from movie metadata
3. Computes cosine similarity matrix
4. Saves trained model to `src/ml/movie_recommender.pkl`

### Step 4: Launch the Web App

```bash
streamlit run src/api/st-interface.py
```

🌐 Opens automatically at `http://localhost:8501`

---

## 🏗️ Architecture

### Project Structure

```
movie-recommender/
│
├── .devcontainer/          # VS Code dev container config
│   ├── devcontainer.json
│   └── setup.sh
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI endpoints (future)
│   │   └── st-interface.py # Streamlit web app ⭐
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py    # TMDb data collection
│   │   └── movies_dataset.csv
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── training.py     # ML model training ⭐
│   │   └── movie_recommender.pkl
│   │
│   └── utils/
│       └── __init__.py
│
├── tests/                  # Pytest test suite
├── .env                    # Environment variables
├── .env.sample
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml          # Project dependencies & config
├── README.md
└── Makefile
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **ML Library** | scikit-learn | TF-IDF, cosine similarity |
| **Data Processing** | Pandas, NumPy | Dataset manipulation |
| **Visualization** | Plotly | Interactive charts |
| **API Client** | Requests | TMDb API integration |
| **Code Quality** | Ruff, Black, MyPy | Linting, formatting, type checking |
| **Testing** | Pytest | Unit and integration tests |
| **Environment** | UV | Fast dependency management |

---

## 🧠 Machine Learning Deep Dive

### 1. Content-Based Filtering

**Algorithm:** TF-IDF (Term Frequency-Inverse Document Frequency) + Cosine Similarity

**Feature Engineering:**
```python
content_features = (
    overview +                    # Movie plot
    genres × 3 +                  # Weighted genres (important!)
    keywords × 2 +                # Plot keywords
    cast × 2 +                    # Lead actors
    director × 3                  # Director (high weight)
)
```

**Similarity Calculation:**
```
similarity(A, B) = cos(θ) = (A · B) / (||A|| × ||B||)
```

**Advantages:**
- No cold-start problem for new users
- Explainable recommendations
- Works with limited data

### 2. Hybrid Recommendation System

**Scoring Formula:**
```python
hybrid_score = (
    0.5 × content_similarity +      # How similar is the content?
    0.3 × normalized_rating +       # How good is the movie?
    0.2 × normalized_popularity     # How popular is it?
)
```

**Normalization:** MinMax scaling (0-1 range) for fair comparison

**Why Hybrid?**
- Balances relevance with quality
- Prevents recommending obscure but similar movies
- Leverages multiple data signals

### 3. Genre-Based Ranking

**Scoring:**
```python
genre_score = 0.7 × normalized_rating + 0.3 × normalized_popularity
```

Prioritizes highly-rated movies within genre while considering popularity.

---

## 📊 Dataset Schema

### Movie Features

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `movie_id` | int | TMDb unique identifier | 550 |
| `title` | str | Movie title | "Fight Club" |
| `overview` | str | Plot summary | "An insomniac office worker..." |
| `genres` | str | Pipe-separated genres | "Drama\|Thriller" |
| `vote_average` | float | Average rating (0-10) | 8.4 |
| `vote_count` | int | Number of votes | 28847 |
| `popularity` | float | TMDb popularity metric | 62.315 |
| `release_date` | str | Release date | "1999-10-15" |
| `runtime` | int | Duration in minutes | 139 |
| `budget` | int | Production budget ($) | 63000000 |
| `revenue` | int | Box office revenue ($) | 100853753 |
| `director` | str | Film director | "David Fincher" |
| `cast` | str | Top 5 actors (pipe-separated) | "Brad Pitt\|Edward Norton..." |
| `keywords` | str | Movie tags | "support group\|dual identity..." |
| `poster_path` | str | Poster URL path | "/pB8BM7pdSp6B6Ih7QZ..." |

---

## 🎨 Web Interface Guide

### 1. Movie-Based Recommendations

**How to use:**
1. Select a movie from the dropdown
2. Click "Get Recommendations"
3. View personalized suggestions with match scores

**Example:**
```
Input: "The Dark Knight"
Output:
  #1 Batman Begins (94% match)
  #2 The Dark Knight Rises (92% match)
  #3 Inception (87% match)
  ...
```

### 2. Genre Explorer

Browse by genre (Action, Comedy, Drama, etc.) and discover top-rated movies in that category.

### 3. Top Rated Movies

Filter by minimum vote count to find critically acclaimed films with statistical significance.

### 4. Dataset Analytics

**Visualizations:**
- **Rating Distribution:** Histogram of movie ratings
- **Genre Analysis:** Bar chart of most common genres
- **Release Trends:** Time series of movies released per year

---

## 🔧 Development

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Lint and fix issues
ruff check --fix src/ tests/

# Type checking
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

```


---

## 🚀 Future Enhancements

### Phase 1: User Personalization
- [ ] User authentication system
- [ ] Personal rating storage (SQLite/PostgreSQL)
- [ ] Collaborative filtering (SVD, ALS algorithms)
- [ ] Viewing history tracking

### Phase 2: Advanced ML
- [ ] Neural network embeddings (Word2Vec, BERT)
- [ ] Deep learning recommendations (NCF, AutoEncoders)
- [ ] Real-time model retraining pipeline
- [ ] A/B testing framework

### Phase 3: Production Features
- [ ] FastAPI REST API endpoints
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Redis caching layer
- [ ] Rate limiting and authentication

### Phase 4: Data Expansion
- [ ] IMDb rating integration
- [ ] Rotten Tomatoes scores
- [ ] Streaming availability (Netflix, Prime, etc.)
- [ ] User reviews sentiment analysis


---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Install dev dependencies:** `uv sync --dev`
4. **Make changes** and add tests
5. **Run quality checks:** `pre-commit run --all-files`
6. **Commit changes:** `git commit -m 'Add amazing feature'`
7. **Push to branch:** `git push origin feature/amazing-feature`
8. **Open a Pull Request**


<div align="center">

⭐ Star this repo if you find it helpful!

</div>
