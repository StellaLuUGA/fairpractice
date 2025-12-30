"""
Data loading and preprocessing for MovieLens-1M
Files:
- ratings.dat: UserID::MovieID::Rating::Timestamp
- movies.dat : MovieID::Title::Genres
- users.dat  : UserID::Gender::Age::Occupation::Zip-code
"""

from pathlib import Path
import pandas as pd


class MovieLensData:
    def __init__(self, data_path, sample_size=5000, random_seed=42):
        self.data_path = Path(data_path)
        self.sample_size = sample_size
        self.random_seed = random_seed

        self.data = None       # merged + preprocessed dataframe
        self.item_db = None    # movie_id -> dict(title, genres, ...)

    def load(self):
        """Load MovieLens-1M and return a preprocessed dataframe."""
        # ---- basic path checks (fail fast with clear message) ----
        for fn in ["ratings.dat", "movies.dat", "users.dat"]:
            p = self.data_path / fn
            if not p.exists():
                raise FileNotFoundError(f"Missing file: {p}")

        # Load ratings
        ratings = pd.read_csv(
            self.data_path / "ratings.dat",
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

        # Load movies (use lowercase 'genres' to avoid KeyError)
        movies = pd.read_csv(
            self.data_path / "movies.dat",
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],   # <-- FIXED
            encoding="latin-1",
        )

        # Load users
        users = pd.read_csv(
            self.data_path / "users.dat",
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip"],
        )

        # Merge
        data = ratings.merge(users, on="user_id", how="inner").merge(
            movies, on="movie_id", how="inner"
        )

        # Sort for sequence-based tasks
        data = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        # Sample if requested (None means use all data)
        if self.sample_size is not None:
            data = data.sample(n=self.sample_size, random_state=self.random_seed).reset_index(drop=True)

        # Preprocess + store
        data = self._preprocess(data)
        self.data = data

        # Build item_db (movie lookup)  <-- FIXED column name to 'genres'
        movies_features = data[["movie_id", "clean_title", "main_genre", "genres"]].drop_duplicates("movie_id")
        self.item_db = movies_features.set_index("movie_id").to_dict(orient="index")

        return self.data

    @staticmethod
    def _preprocess(data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Extract main genre (first genre in pipe-separated list)
        data["main_genre"] = data["genres"].astype(str).apply(lambda x: x.split("|")[0] if x else "")

        # Remove "(1999)" year from title
        data["clean_title"] = (
            data["title"].astype(str)
            .str.replace(r"\(\d{4}\)", "", regex=True)
            .str.strip()
        )

        # Map gender
        data["gender"] = data["gender"].map({"M": "Male", "F": "Female"}).fillna(data["gender"])

        # Return a focused set (keep extra columns if you need them)
        # NOTE: include 'genres' so later code can use it safely
        return data[["user_id", "movie_id", "clean_title", "main_genre", "genres", "gender", "rating", "timestamp"]]


def load_data(config):
    """
    Convenience wrapper using your Config object.

    Returns:
      df: preprocessed dataframe
      item_db: dict movie_id -> {'clean_title','main_genre','genres'}
    """
    loader = MovieLensData(
        data_path=config.DATA_PATH,
        sample_size=config.SAMPLE_SIZE,          # None => use all data
        random_seed=getattr(config, "RANDOM_SEED", 42),
    )
    df = loader.load()
    return df, loader.item_db
