# 🎬 AI Movie Recommender System - Senior Level Project

A comprehensive movie recommendation system with machine learning and an interactive web interface.


### Step 1: Get TMDb API Key

1. Go to https://www.themoviedb.org/
2. Create a free account
3. Navigate to Settings → API
4. Request an API key
5. Copy your API Key

### Step 2: Collect Movie Data


This script will:
- Fetch 1,000+ popular movies from TMDb
- Enrich data with details (cast, crew, genres, keywords)
- Save to `movies_dataset.csv` (~500 movies enriched, customizable)
- Takes approximately 15-30 minutes (due to API rate limits)

### Step 3: Train the Model

This will:
- Load the dataset
- Create TF-IDF feature vectors
- Compute similarity matrices
- Save the trained model to `movie_recommender.pkl`
- Takes 2-5 minutes depending on dataset size

### Step 4: Launch the Web App

```bash
streamlit run src/api/st-interface.py
```

The app will open in your browser at `http://localhost:8501`

## 🧠 Machine Learning Algorithms

### 1. Content-Based Filtering
- **TF-IDF Vectorization**: Converts movie metadata into numerical features
- **Features Used**: Plot overview, genres, keywords, cast, director
- **Similarity Metric**: Cosine similarity
- **Advantage**: Works well for new users, doesn't need ratings

### 2. Hybrid Recommendation
Combines multiple signals:
```python
hybrid_score = 0.5 × content_similarity +
               0.3 × normalized_rating +
               0.2 × normalized_popularity
```

### 3. Genre-Based Filtering
- Filters movies by selected genre
- Ranks by weighted score of rating and popularity

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `movie_id` | Unique TMDb identifier |
| `title` | Movie title |
| `overview` | Plot summary |
| `genres` | Pipe-separated genre list |
| `vote_average` | Average user rating (0-10) |
| `vote_count` | Number of votes |
| `popularity` | TMDb popularity score |
| `release_date` | Release date |
| `runtime` | Movie duration (minutes) |
| `cast` | Top 5 actors |
| `director` | Film director |
| `keywords` | Movie keywords/tags |
| `poster_path` | Poster image path |

## 🎨 Streamlit Features

### 1. Movie-Based Recommendations
- Select any movie from the database
- Get personalized recommendations based on similarity
- Visual match percentage for each recommendation

### 2. Genre Explorer
- Browse movies by genre
- Smart ranking based on ratings and popularity

### 3. Top Rated Movies
- Configurable minimum vote threshold
- Discover critically acclaimed films

### 4. Dataset Analytics
- Interactive visualizations with Plotly
- Rating distributions
- Genre frequency analysis
- Release year trends


## 📚 Potential Enhancements

### 1. User Rating System
- Add user authentication
- Store user ratings in database
- Implement collaborative filtering (SVD, ALS)

### 2. Advanced Features
- Neural network embeddings (Word2Vec, BERT)
- Deep learning models (Neural Collaborative Filtering)
- Real-time model updates

### 3. Additional Data Sources
- IMDb integration
- Rotten Tomatoes scores
- Streaming availability



## 🤝 Contributing

Suggestions for improvement:
1. Fork the repository
2. Create a feature branch
3. Implement enhancements
4. Submit a pull request
