import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def load_movies():
    path = os.path.join(PROCESSED_DIR, 'movies_cleaned.csv')
    df = pd.read_csv(path)
    df['release_datetime'] = pd.to_datetime(df['release_datetime'], errors='coerce')
    return df

def load_movie_genres():
    path = os.path.join(PROCESSED_DIR, 'movies_genres.csv')
    return pd.read_csv(path)

def load_movie_companies():
    path = os.path.join(PROCESSED_DIR, 'movies_companies.csv')
    return pd.read_csv(path)

def load_all():
    return load_movies(), load_movie_genres(), load_movie_companies()
