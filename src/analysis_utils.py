import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def analyze_q1_genres_finances(movie_genres_df):
    """
    Q1: Which film genres yield the highest financial returns (Revenue, Profit, and ROI)?
    Does a massive budget guarantee commercial success or high ratings?
    """
    fin = movie_genres_df[movie_genres_df['has_financial_data']].copy()
    
    # Filter genres with meaningful sample size (>= 30 movies)
    genre_counts = fin['genre'].value_counts()
    valid_genres = genre_counts[genre_counts >= 30].index
    fin = fin[fin['genre'].isin(valid_genres)]
    
    genre_stats = fin.groupby('genre').agg(
        movie_count=('title', 'count'),
        median_budget=('budget', 'median'),
        mean_budget=('budget', 'mean'),
        median_revenue=('revenue', 'median'),
        mean_revenue=('revenue', 'mean'),
        median_profit=('profit', 'median'),
        mean_profit=('profit', 'mean'),
        median_roi=('roi_percent', 'median'),
        mean_roi=('roi_percent', 'mean'),
        avg_vote=('vote_average', 'mean')
    ).reset_index()

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top genres by Mean Revenue
    top_rev = genre_stats.sort_values(by='mean_revenue', ascending=False)
    sns.barplot(data=top_rev, x='mean_revenue', y='genre', palette='Blues_r', ax=ax1)
    ax1.set_title("Average Box Office Revenue by Genre ($)", fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel("Mean Revenue ($)", fontsize=12)
    ax1.set_ylabel("Genre", fontsize=12)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-6:.0f}M"))

    # Top genres by Median ROI
    top_roi = genre_stats.sort_values(by='median_roi', ascending=False)
    sns.barplot(data=top_roi, x='median_roi', y='genre', palette='Greens_r', ax=ax2)
    ax2.set_title("Median Return on Investment (ROI %) by Genre", fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel("Median ROI (%)", fontsize=12)
    ax2.set_ylabel("")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'q1_genre_finances.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return genre_stats, fig_path


def analyze_q2_decade_trends(movies_df):
    """
    Q2: How has movie volume, budget, and runtime evolved across decades (1920-2023)?
    """
    valid = movies_df[(movies_df['release_year'] >= 1930) & (movies_df['release_year'] <= 2023)].copy()
    
    decade_stats = valid.groupby('release_decade_label').agg(
        movie_count=('id', 'count'),
        median_runtime=('runtime', lambda x: x[x > 0].median()),
        median_vote=('vote_average', 'median'),
        mean_vote=('vote_average', 'mean'),
        median_popularity=('popularity', 'median')
    ).reset_index()

    # Decade financial stats
    fin = valid[valid['has_financial_data']]
    decade_fin = fin.groupby('release_decade_label').agg(
        median_budget=('budget', 'median'),
        median_revenue=('revenue', 'median'),
        median_profit=('profit', 'median')
    ).reset_index()

    combined = decade_stats.merge(decade_fin, on='release_decade_label', how='left')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Movie releases count
    sns.barplot(data=combined, x='release_decade_label', y='movie_count', color='#4A90E2', ax=axes[0])
    axes[0].set_title("Movie Releases per Decade in Dataset", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Decade")
    axes[0].set_ylabel("Number of Movies")
    axes[0].tick_params(axis='x', rotation=45)

    # 2. Financial evolution (Budget & Revenue)
    axes[1].plot(combined['release_decade_label'], combined['median_budget'] * 1e-6, marker='o', label='Median Budget ($M)', color='#E74C3C', linewidth=2.5)
    axes[1].plot(combined['release_decade_label'], combined['median_revenue'] * 1e-6, marker='s', label='Median Revenue ($M)', color='#2ECC71', linewidth=2.5)
    axes[1].set_title("Median Budget vs. Revenue by Decade ($M)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Decade")
    axes[1].set_ylabel("Amount (Million $)")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    # 3. Runtime evolution
    sns.lineplot(data=combined, x='release_decade_label', y='median_runtime', marker='^', color='#9B59B6', linewidth=2.5, ax=axes[2])
    axes[2].set_title("Median Movie Runtime by Decade", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Decade")
    axes[2].set_ylabel("Runtime (Minutes)")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'q2_decade_trends.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return combined, fig_path


def analyze_q3_seasonality(movies_df):
    """
    Q3: Does release timing (Month / Season) affect Box Office Revenue and Popularity?
    """
    fin = movies_df[movies_df['has_financial_data']].copy()
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    monthly_stats = fin.groupby('release_month_name').agg(
        movie_count=('id', 'count'),
        mean_revenue=('revenue', 'mean'),
        median_revenue=('revenue', 'median'),
        mean_popularity=('popularity', 'mean'),
        mean_vote=('vote_average', 'mean')
    ).reindex(month_order).reset_index()

    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_stats = fin.groupby('release_season').agg(
        movie_count=('id', 'count'),
        mean_revenue=('revenue', 'mean'),
        median_revenue=('revenue', 'median'),
        mean_profit=('profit', 'mean'),
        median_profit=('profit', 'median'),
        mean_popularity=('popularity', 'mean')
    ).reindex(season_order).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Monthly revenue barplot
    sns.barplot(data=monthly_stats, x='release_month_name', y='mean_revenue', palette='coolwarm', ax=ax1)
    ax1.set_title("Average Box Office Revenue by Release Month", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean Revenue ($)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-6:.0f}M"))

    # Seasonal profit comparison
    sns.barplot(data=seasonal_stats, x='release_season', y='mean_profit', palette='Set2', ax=ax2)
    ax2.set_title("Average Net Profit by Release Season", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Mean Profit ($)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-6:.0f}M"))

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'q3_seasonality.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return monthly_stats, seasonal_stats, fig_path


def analyze_q4_ratings_and_finances(movies_df):
    """
    Q4: Is there a correlation between ratings, vote count, budget, popularity, and revenue?
    """
    fin = movies_df[movies_df['has_financial_data'] & (movies_df['vote_count'] >= 50)].copy()
    
    features = ['budget', 'revenue', 'profit', 'popularity', 'vote_average', 'vote_count', 'runtime']
    corr_matrix = fin[features].corr()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, ax=ax1)
    ax1.set_title("Correlation Matrix of Movie Attributes", fontsize=13, fontweight='bold')

    # Scatter plot: Budget vs Revenue with Vote Rating hue
    scatter = ax2.scatter(
        fin['budget'] * 1e-6, 
        fin['revenue'] * 1e-6, 
        c=fin['vote_average'], 
        cmap='viridis', 
        alpha=0.6, 
        edgecolors='none', 
        s=30
    )
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Vote Average (1-10)', fontsize=11)
    ax2.plot([0, 400], [0, 400], 'r--', label='Break-even Line (Revenue = Budget)')
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 3000)
    ax2.set_title("Budget vs. Revenue ($ Millions)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Budget ($M)")
    ax2.set_ylabel("Revenue ($M)")
    ax2.legend(loc='upper left')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'q4_correlations.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return corr_matrix, fig_path


def analyze_q5_top_studios(movie_companies_df):
    """
    Q5: Which production companies dominate the box office?
    """
    fin = movie_companies_df[movie_companies_df['has_financial_data']].copy()
    
    studio_stats = fin.groupby('company').agg(
        movie_count=('title', 'count'),
        total_revenue=('revenue', 'sum'),
        mean_revenue=('revenue', 'mean'),
        total_profit=('profit', 'sum'),
        mean_profit=('profit', 'mean'),
        median_roi=('roi_percent', 'median'),
        mean_vote=('vote_average', 'mean')
    ).reset_index()

    # Filter studios with at least 15 releases
    major_studios = studio_stats[studio_stats['movie_count'] >= 15].sort_values(by='total_revenue', ascending=False).head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Total Gross Revenue
    sns.barplot(data=major_studios, x='total_revenue', y='company', palette='crest_r', ax=ax1)
    ax1.set_title("Top 15 Studios by Cumulative Box Office Revenue ($ Billions)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Total Revenue ($)")
    ax1.set_ylabel("Production Studio")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-9:.1f}B"))

    # Average Profit per Movie
    top_avg_profit = major_studios.sort_values(by='mean_profit', ascending=False)
    sns.barplot(data=top_avg_profit, x='mean_profit', y='company', palette='mako_r', ax=ax2)
    ax2.set_title("Average Profit per Movie for Major Studios ($ Millions)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Mean Profit per Movie ($)")
    ax2.set_ylabel("")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x*1e-6:.0f}M"))

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'q5_top_studios.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return major_studios, fig_path


def analyze_q6_language_landscape(movies_df):
    """
    Q6: How do non-English films perform compared to English-language films?
    """
    valid = movies_df[movies_df['vote_count'] >= 30].copy()
    
    # Top 8 languages + 'Other'
    top_langs = valid['original_language'].value_counts().head(8).index.tolist()
    valid['lang_grouped'] = valid['original_language'].apply(lambda x: x if x in top_langs else 'Other')

    lang_stats = valid.groupby('lang_grouped').agg(
        movie_count=('id', 'count'),
        mean_vote=('vote_average', 'mean'),
        median_vote=('vote_average', 'median'),
        mean_popularity=('popularity', 'mean'),
        median_popularity=('popularity', 'median')
    ).sort_values(by='movie_count', ascending=False).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Average Vote by Language
    sns.barplot(data=lang_stats, x='mean_vote', y='lang_grouped', palette='rocket', ax=ax1)
    ax1.set_title("Average Audience Rating by Language (min 30 votes)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Mean Vote Average (1-10)")
    ax1.set_ylabel("Language")
    ax1.set_xlim(5.5, 8.0)

    # Boxplot of vote distribution for top languages
    sns.boxplot(data=valid, x='vote_average', y='lang_grouped', palette='Set3', ax=ax2)
    ax2.set_title("Rating Distribution Across Languages", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Vote Average")
    ax2.set_ylabel("")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'q6_language_landscape.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return lang_stats, fig_path

def run_all_analyses():
    from src.data_loader import load_all
    movies, genres, companies = load_all()
    print("Running Q1 Analysis...")
    analyze_q1_genres_finances(genres)
    print("Running Q2 Analysis...")
    analyze_q2_decade_trends(movies)
    print("Running Q3 Analysis...")
    analyze_q3_seasonality(movies)
    print("Running Q4 Analysis...")
    analyze_q4_ratings_and_finances(movies)
    print("Running Q5 Analysis...")
    analyze_q5_top_studios(companies)
    print("Running Q6 Analysis...")
    analyze_q6_language_landscape(movies)
    print(f"All analyses complete! Figures saved to {FIG_DIR}")

if __name__ == '__main__':
    run_all_analyses()
