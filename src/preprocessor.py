import os
import pandas as pd
import numpy as np

def clean_and_process_data(raw_path=r'data/raw/9616_UNIQUE_IMDB.csv', processed_dir=r'data/processed'):
    print(f"Loading raw data from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    print(f"Raw shape: {raw_df.shape}")

    # 1. Standardize language codes
    language_mapping = {
        'cn': 'Chinese',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'ko': 'Korean',
        'ru': 'Russian',
        'hi': 'Hindi',
        'pt': 'Portuguese'
    }
    raw_df['original_language'] = raw_df['original_language'].replace(language_mapping)

    # 2. Extract unique movies
    # Create genre list and company list per movie
    genres_per_movie = (
        raw_df[['id', 'genres']]
        .dropna()
        .drop_duplicates()
        .groupby('id')['genres']
        .apply(lambda x: [g for g in list(dict.fromkeys(x)) if g != 'Unspecified'])
        .reset_index()
    )
    genres_per_movie['genre_list'] = genres_per_movie['genres'].apply(lambda x: ", ".join(x) if x else "Unspecified")
    genres_per_movie['genre_count'] = genres_per_movie['genres'].apply(len)

    companies_per_movie = (
        raw_df[['id', 'production_companies']]
        .dropna()
        .drop_duplicates()
        .groupby('id')['production_companies']
        .apply(lambda x: list(dict.fromkeys(x)))
        .reset_index()
    )
    companies_per_movie['company_list'] = companies_per_movie['production_companies'].apply(lambda x: ", ".join(x) if x else "Independent/Unspecified")
    companies_per_movie['company_count'] = companies_per_movie['production_companies'].apply(len)

    # Core movie metadata (drop duplicate IDs)
    meta_cols = ['id', 'title', 'release_date', 'original_language', 'vote_average', 
                 'vote_count', 'popularity', 'budget', 'revenue', 'runtime', 'overview', 'tagline']
    movies = raw_df.drop_duplicates(subset=['id'])[meta_cols].copy()

    # Merge aggregated genres and companies
    movies = movies.merge(genres_per_movie[['id', 'genre_list', 'genre_count']], on='id', how='left')
    movies = movies.merge(companies_per_movie[['id', 'company_list', 'company_count']], on='id', how='left')

    movies['genre_list'] = movies['genre_list'].fillna('Unspecified')
    movies['genre_count'] = movies['genre_count'].fillna(0).astype(int)
    movies['company_list'] = movies['company_list'].fillna('Independent/Unspecified')
    movies['company_count'] = movies['company_count'].fillna(0).astype(int)

    # 3. Parse release dates & temporal features
    movies['release_datetime'] = pd.to_datetime(movies['release_date'], format='%d-%m-%Y %H:%M', errors='coerce')
    movies['release_year'] = movies['release_datetime'].dt.year
    movies['release_month'] = movies['release_datetime'].dt.month
    movies['release_day'] = movies['release_datetime'].dt.day
    movies['release_day_name'] = movies['release_datetime'].dt.day_name()
    movies['release_month_name'] = movies['release_datetime'].dt.month_name()

    # Decade feature
    movies['release_decade'] = (movies['release_year'] // 10) * 10
    movies['release_decade_label'] = movies['release_decade'].astype(str) + 's'

    # Season mapping
    month_to_season = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    movies['release_season'] = movies['release_month'].map(month_to_season)

    # 4. Financial features
    movies['has_financial_data'] = (movies['budget'] > 0) & (movies['revenue'] > 0)
    movies['profit'] = np.where(movies['has_financial_data'], movies['revenue'] - movies['budget'], np.nan)
    movies['roi_percent'] = np.where(movies['has_financial_data'], ((movies['revenue'] - movies['budget']) / movies['budget']) * 100, np.nan)
    
    # Financial tiers
    def categorize_success(row):
        if not row['has_financial_data']:
            return 'Unknown / Missing Data'
        if row['profit'] < 0:
            return 'Box Office Flop (Loss)'
        elif row['profit'] < row['budget']:
            return 'Moderate Earner'
        elif row['profit'] < 3 * row['budget']:
            return 'Commercial Hit'
        else:
            return 'Blockbuster Phenomenon (>3x ROI)'
            
    movies['commercial_success_tier'] = movies.apply(categorize_success, axis=1)

    # 5. Runtime categories
    def categorize_runtime(r):
        if r <= 0:
            return 'Unknown'
        elif r < 80:
            return 'Short (<80m)'
        elif r <= 130:
            return 'Standard (80-130m)'
        else:
            return 'Epic (>130m)'
            
    movies['runtime_category'] = movies['runtime'].apply(categorize_runtime)

    # 6. Create bridge tables for many-to-many relationships
    # Bridge table: Movie - Genre
    movie_genres = (
        raw_df[['id', 'genres']]
        .dropna()
        .drop_duplicates()
        .rename(columns={'id': 'movie_id', 'genres': 'genre'})
    )
    movie_genres = movie_genres[movie_genres['genre'] != 'Unspecified']
    # Merge core movie info for fast genre aggregation
    movie_genres = movie_genres.merge(
        movies[['id', 'title', 'release_year', 'vote_average', 'vote_count', 'popularity', 'budget', 'revenue', 'profit', 'roi_percent', 'has_financial_data']],
        left_on='movie_id', right_on='id', how='inner'
    ).drop(columns=['id'])

    # Bridge table: Movie - Production Company
    movie_companies = (
        raw_df[['id', 'production_companies']]
        .dropna()
        .drop_duplicates()
        .rename(columns={'id': 'movie_id', 'production_companies': 'company'})
    )
    movie_companies = movie_companies.merge(
        movies[['id', 'title', 'release_year', 'vote_average', 'vote_count', 'popularity', 'budget', 'revenue', 'profit', 'roi_percent', 'has_financial_data']],
        left_on='movie_id', right_on='id', how='inner'
    ).drop(columns=['id'])

    # 7. Save to processed directory
    os.makedirs(processed_dir, exist_ok=True)
    movies_out = os.path.join(processed_dir, 'movies_cleaned.csv')
    genres_out = os.path.join(processed_dir, 'movies_genres.csv')
    companies_out = os.path.join(processed_dir, 'movies_companies.csv')

    movies.to_csv(movies_out, index=False)
    movie_genres.to_csv(genres_out, index=False)
    movie_companies.to_csv(companies_out, index=False)

    print(f"Successfully processed {len(movies)} unique movies.")
    print(f"Saved cleaned movies to: {movies_out}")
    print(f"Saved movie-genres bridge ({len(movie_genres)} rows) to: {genres_out}")
    print(f"Saved movie-companies bridge ({len(movie_companies)} rows) to: {companies_out}")
    return movies, movie_genres, movie_companies

if __name__ == '__main__':
    clean_and_process_data()
