# Data loading & preprocessing

"""Data loading and preprocessing for MovieLens-1M"""

import pandas as pd
import numpy as np
from pathlib import Path

class MovieLensData:
    def __init__(self, data_path, sample_size=None):
        self.data_path = Path(data_path)
        self.sample_size = sample_size
        
    def load(self):
        """Load MovieLens-1M with gender labels"""
        # Load ratings
        ratings = pd.read_csv(
            self.data_path / "ratings.dat",
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python"
        )
        
        # Load movies
        movies = pd.read_csv(
            self.data_path / "movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1"
        )
        
        # Load users (contains gender)
        users = pd.read_csv(
            self.data_path / "users.dat",
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zip"],
            engine="python"
        )
        
        # Merge
        data = ratings.merge(users, on="user_id").merge(movies, on="movie_id")
        
        # Sample if needed
        if self.sample_size:
            data = data.sample(n=self.sample_size, random_state=42)
        
        return self._preprocess(data)
    
    def _preprocess(self, data):
        """Prepare features for prediction"""
        # Extract main genre
        data['main_genre'] = data['genres'].apply(lambda x: x.split('|')[0])
        
        # Clean title
        data['clean_title'] = data['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        
        # Binary gender (M/F)
        data['gender'] = data['gender'].map({'M': 'Male', 'F': 'Female'})
        
        return data[['user_id', 'movie_id', 'clean_title', 'main_genre', 'gender', 'rating']]

def load_data(config):
    """Convenience function"""
    loader = MovieLensData(config.DATA_PATH, config.SAMPLE_SIZE)
    return loader.load()