# data.py
import os
import pandas as pd

DATA_DIR = "/home/ubuntu/Aristella/FACTER/data/ml-1m"

MOVIES_PATH  = os.path.join(DATA_DIR, "movies.dat")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.dat")
USERS_PATH   = os.path.join(DATA_DIR, "users.dat")


TARGET_TITLES = [
    "Virgin Suicides, The (1999)",
    "Anna and the King (1999)",
    "Grosse Pointe Blank (1997)",
    "Go (1999)",
    "Alien: Resurrection (1997)",
    "One Flew Over the Cuckoo's Nest (1975)",
    "13th Warrior, The (1999)",
    "Innocents, The (1961)",
    "Down to You (2000)",
    "Basquiat (1996)",
    "Simple Plan, A (1998)",
    "Airplane! (1980)",
    "Just the Ticket (1999)",
    "Believers, The (1987)",
    "Fools Rush In (1997)",
    "French Kiss (1995)",
    "Four Weddings and a Funeral (1994)",
    "Last Temptation of Christ, The (1988)",
    "Strictly Ballroom (1992)",
    "Hunt for Red October, The (1990)",
    "U.S. Marshalls (1998)",
    "GoldenEye (1995)",
    "Some Folks Call It a Sling Blade (1993)",
    "Sound of Music, The (1965)",
    "Shadow of Angels (Schatten der Engel) (1976)",
    "Hear My Song (1991)",
    "Office Space (1999)",
    "After Life (1998)",
    "From Dusk Till Dawn (1996)",
    "Pretty Woman (1990)",
    "Hard Rain (1998)",
    "Thomas Crown Affair, The (1999)",
    "Crazy in Alabama (1999)",
]


def load_data(
    data_dir: str = DATA_DIR,
    target_titles: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set]:
    """
    Load MovieLens 1M movies/ratings/users and return a target title set.

    Returns:
      movies:  MovieID, Title, Genres
      ratings: UserID, MovieID, Rating, Timestamp
      users:   UserID, Gender, Age, Occupation, Zip-code
      target_set: set of target movie titles
    """
    movies_path = os.path.join(data_dir, "movies.dat")
    ratings_path = os.path.join(data_dir, "ratings.dat")
    users_path = os.path.join(data_dir, "users.dat")

    target_titles = target_titles if target_titles is not None else TARGET_TITLES
    target_set = set(target_titles)

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1",
    )

    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        header=None,
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding="latin-1",
    )

    return movies, ratings, users, target_set


def build_user_movie_df(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    target_set: set,
) -> pd.DataFrame:
    """
    Creates a user-movie interaction table restricted to target_set.

    Output columns for your predictor:
      clean_title, main_genre, gender
    """
    movies_sub = movies[movies["Title"].isin(target_set)].copy()

    merged = ratings.merge(
        movies_sub[["MovieID", "Title", "Genres"]],
        on="MovieID",
        how="inner",
    ).merge(
        users[["UserID", "Gender"]],
        on="UserID",
        how="inner",
    )

    merged["main_genre"] = merged["Genres"].astype(str).apply(
        lambda s: s.split("|")[0] if "|" in s else s
    )
    merged["clean_title"] = merged["Title"].astype(str)
    merged["gender"] = merged["Gender"].astype(str)

    return merged[["clean_title", "main_genre", "gender"]].copy()


def save_subset_csv(
    out_csv: str = "movielens_ratings_users_movies_subset_33.csv",
    data_dir: str = DATA_DIR,
    target_titles: list[str] | None = None,
) -> dict:
    """
    Optional utility: saves the merged subset CSV like your original script,
    but only runs when called.

    Returns:
      dict with matched count, missing titles, and row count saved.
    """
    movies, ratings, users, target_set = load_data(data_dir=data_dir, target_titles=target_titles)

    movies_sub = movies[movies["Title"].isin(target_set)].copy()
    found = set(movies_sub["Title"].tolist())
    missing = sorted(target_set - found)

    ratings_sub = ratings.merge(movies_sub[["MovieID", "Title", "Genres"]], on="MovieID", how="inner")
    full_merged = ratings_sub.merge(users, on="UserID", how="inner")

    full_merged.to_csv(out_csv, index=False)

    return {
        "matched": len(found),
        "total_targets": len(target_set),
        "missing": missing,
        "rows_saved": len(full_merged),
        "out_csv": out_csv,
    }


if __name__ == "__main__":
    # Only runs when you execute: python data.py
    info = save_subset_csv()
    print(f"Matched {info['matched']}/{info['total_targets']} target titles.")
    if info["missing"]:
        print("Missing titles:")
        for t in info["missing"]:
            print("  -", t)
    print(f"Saved {info['rows_saved']} rows to {info['out_csv']}")
