import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import load_movies, load_movie_genres, load_movie_companies

# Page configuration
st.set_page_config(
    page_title="Movie Data Analysis | TMDB 9500+ Movies",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-hit { background-color: #DCFCE7; color: #166534; }
    .badge-flop { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)

# Data Caching
@st.cache_data
def get_data():
    movies = load_movies()
    genres = load_movie_genres()
    companies = load_movie_companies()
    return movies, genres, companies

movies_df, genres_df, companies_df = get_data()

# Sidebar Navigation & Filters
st.sidebar.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("🎬 TMDB Movie Analysis")
st.sidebar.markdown("**Group Final Project** | Course Assessment")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Summary",
        "🎯 Q1: Genre Financials & ROI",
        "⏳ Q2: Decades Evolution",
        "📅 Q3: Seasonality & Timing",
        "⭐ Q4: Ratings vs Commercial Appeal",
        "🏢 Q5: Studio Dominance",
        "🌍 Q6: Language Landscape",
        "🔍 Movie Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

min_year, max_year = int(movies_df['release_year'].min()), int(movies_df['release_year'].max())
selected_years = st.sidebar.slider(
    "Release Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(1970, 2023)
)

fin_only = st.sidebar.checkbox("Only movies with Box Office data", value=False)
min_votes = st.sidebar.number_input("Minimum Vote Count", min_value=0, max_value=5000, value=30, step=20)

# Filtered data
filtered_movies = movies_df[
    (movies_df['release_year'] >= selected_years[0]) &
    (movies_df['release_year'] <= selected_years[1]) &
    (movies_df['vote_count'] >= min_votes)
].copy()

if fin_only:
    filtered_movies = filtered_movies[filtered_movies['has_financial_data']].copy()

filtered_genres = genres_df[genres_df['movie_id'].isin(filtered_movies['id'])].copy()
filtered_companies = companies_df[companies_df['movie_id'].isin(filtered_movies['id'])].copy()

# Header
st.markdown('<div class="main-title">🎬 Cinema Industry Insights & Data Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Exploratory Data Analysis and Business Intelligence on 9,500+ TMDB Popular Movies</div>', unsafe_allow_html=True)

# Top KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(filtered_movies):,}</div>
        <div class="metric-label">Selected Movies</div>
    </div>
    """, unsafe_allow_html=True)

fin_sub = filtered_movies[filtered_movies['has_financial_data']]
with kpi2:
    med_budget = fin_sub['budget'].median() if len(fin_sub) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">${med_budget*1e-6:.1f}M</div>
        <div class="metric-label">Median Budget</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    med_rev = fin_sub['revenue'].median() if len(fin_sub) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">${med_rev*1e-6:.1f}M</div>
        <div class="metric-label">Median Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    med_roi = fin_sub['roi_percent'].median() if len(fin_sub) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{med_roi:.0f}%</div>
        <div class="metric-label">Median ROI</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    avg_score = filtered_movies['vote_average'].mean() if len(filtered_movies) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{avg_score:.2f} ★</div>
        <div class="metric-label">Avg Rating</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------- 1. Executive Summary -----------------
if nav_choice == "📊 Executive Summary":
    st.subheader("📌 Project Motivation & Assessment Background")
    st.info("""
    **Course Assessment Requirement (Group Final Project - 40% Grade):**
    - 2-3 students / group, utilizing Git & GitHub for collaborative version control.
    - Dataset: **9500+ Popular Movies TMDB** (from Kaggle).
    - Objective: Explore the dataset, discover critical structural anomalies, formulate meaningful analytical questions, apply robust data preprocessing, and derive actionable business & artistic insights.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### ⚠️ Data Discovery & Structural Quirks")
        st.markdown("""
        1. **Exploded Cross-Product**: Raw CSV contains **83,739 rows** representing **9,961 unique movies**. The dataset was flattened on `(genres × production_companies)`, artificially multiplying budgets and revenues up to 30x if uncleaned!
        2. **Unreported Financials (Zero Values)**: ~48% of movies have `budget = 0` and ~45% have `revenue = 0`. These represent unrecorded data rather than free movies.
        3. **Categorical Aliases**: Languages like Chinese were represented as both `'cn'` and `'Chinese'`.
        """)
    with col_b:
        st.markdown("### 💡 Strategic Research Questions")
        st.markdown("""
        - **Q1**: Which genres offer the highest Return on Investment (ROI) vs. total revenue?
        - **Q2**: How have production budgets, runtime, and volume shifted across decades?
        - **Q3**: Does release seasonality (Summer vs. Holiday) drive box office success?
        - **Q4**: Does big budget guarantee critical acclaim, or is there a divergence?
        - **Q5**: Which studios dominate total gross vs. average profitability?
        - **Q6**: How do international / non-English films compare against Hollywood?
        """)

# ----------------- 2. Q1: Genre Financials & ROI -----------------
elif nav_choice == "🎯 Q1: Genre Financials & ROI":
    st.subheader("🎯 Question 1: Genre Financial Performance & Return on Investment")
    st.write("Comparing Average Box Office Revenue with Median ROI across film genres.")

    fin_genres = filtered_genres[filtered_genres['has_financial_data']].copy()
    genre_summary = fin_genres.groupby('genre').agg(
        count=('title', 'count'),
        avg_budget=('budget', 'mean'),
        avg_revenue=('revenue', 'mean'),
        avg_profit=('profit', 'mean'),
        median_roi=('roi_percent', 'median'),
        avg_rating=('vote_average', 'mean')
    ).reset_index()
    genre_summary = genre_summary[genre_summary['count'] >= 15]

    tab1, tab2 = st.tabs(["💰 Revenue & Profit", "📈 ROI & Efficiency"])

    with tab1:
        fig_rev = px.bar(
            genre_summary.sort_values(by='avg_revenue', ascending=False),
            x='genre',
            y=['avg_revenue', 'avg_budget'],
            barmode='group',
            title="Average Revenue vs. Average Budget by Genre ($ USD)",
            labels={'value': 'Amount ($)', 'variable': 'Metric', 'genre': 'Genre'},
            color_discrete_map={'avg_revenue': '#2563EB', 'avg_budget': '#F87171'}
        )
        fig_rev.update_layout(yaxis_tickformat='$,.0f')
        st.plotly_chart(fig_rev, use_container_width=True)
        st.caption("Key Insight: Animation, Adventure, and Science Fiction demand the largest budgets and reap the highest total revenues.")

    with tab2:
        fig_roi = px.bar(
            genre_summary.sort_values(by='median_roi', ascending=False),
            x='genre',
            y='median_roi',
            title="Median Return on Investment (ROI %) by Genre",
            labels={'median_roi': 'Median ROI (%)', 'genre': 'Genre'},
            color='median_roi',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_roi, use_container_width=True)
        st.caption("Key Insight: **Horror** and **Mystery** consistently generate the highest ROI with modest production budgets and strong fan turnouts.")

# ----------------- 3. Q2: Decades Evolution -----------------
elif nav_choice == "⏳ Q2: Decades Evolution":
    st.subheader("⏳ Question 2: Evolution of Cinema Over Time (1930s - 2020s)")
    
    decade_agg = filtered_movies.groupby('release_decade_label').agg(
        movie_count=('id', 'count'),
        median_runtime=('runtime', lambda x: x[x > 0].median()),
        avg_rating=('vote_average', 'mean')
    ).reset_index()

    decade_fin = filtered_movies[filtered_movies['has_financial_data']].groupby('release_decade_label').agg(
        median_budget=('budget', 'median'),
        median_revenue=('revenue', 'median')
    ).reset_index()

    merged_decade = decade_agg.merge(decade_fin, on='release_decade_label', how='left')

    c1, c2 = st.columns(2)
    with c1:
        fig_fin_time = go.Figure()
        fig_fin_time.add_trace(go.Scatter(x=merged_decade['release_decade_label'], y=merged_decade['median_budget'], mode='lines+markers', name='Median Budget ($)', line=dict(color='#EF4444', width=3)))
        fig_fin_time.add_trace(go.Scatter(x=merged_decade['release_decade_label'], y=merged_decade['median_revenue'], mode='lines+markers', name='Median Revenue ($)', line=dict(color='#10B981', width=3)))
        fig_fin_time.update_layout(title="Median Budget & Revenue Growth by Decade", yaxis_tickformat='$,.0f')
        st.plotly_chart(fig_fin_time, use_container_width=True)

    with c2:
        fig_runtime = px.line(
            merged_decade,
            x='release_decade_label',
            y='median_runtime',
            markers=True,
            title="Median Movie Runtime Over Time (Minutes)",
            labels={'median_runtime': 'Runtime (mins)', 'release_decade_label': 'Decade'}
        )
        st.plotly_chart(fig_runtime, use_container_width=True)

# ----------------- 4. Q3: Seasonality -----------------
elif nav_choice == "📅 Q3: Seasonality & Timing":
    st.subheader("📅 Question 3: The Impact of Release Timing & Seasonality")
    
    fin_movies = filtered_movies[filtered_movies['has_financial_data']].copy()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    monthly = fin_movies.groupby('release_month_name').agg(
        count=('id', 'count'),
        avg_rev=('revenue', 'mean'),
        avg_profit=('profit', 'mean'),
        avg_pop=('popularity', 'mean')
    ).reindex(month_order).reset_index()

    fig_month = px.bar(
        monthly,
        x='release_month_name',
        y='avg_rev',
        color='avg_profit',
        title="Average Box Office Revenue & Profitability by Month",
        labels={'release_month_name': 'Month', 'avg_rev': 'Average Revenue ($)', 'avg_profit': 'Average Profit ($)'},
        color_continuous_scale='Viridis'
    )
    fig_month.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_month, use_container_width=True)
    st.markdown("""
    **Analytical Findings:**
    - **Summer Peak (May - July)**: Studio tentpoles and summer blockbusters yield peak revenue.
    - **Holiday Surge (November - December)**: Thanksgiving and Christmas windows capture strong family-driven box office.
    - **Dump Months (January - February & September)**: Lower revenues, frequently used for lower-budget releases.
    """)

# ----------------- 5. Q4: Ratings vs Commercial Appeal -----------------
elif nav_choice == "⭐ Q4: Ratings vs Commercial Appeal":
    st.subheader("⭐ Question 4: Critical Acclaim vs. Box Office Success")

    fin_sub = filtered_movies[filtered_movies['has_financial_data']].copy()
    
    fig_scatter = px.scatter(
        fin_sub,
        x='budget',
        y='revenue',
        color='vote_average',
        size='popularity',
        hover_name='title',
        hover_data=['release_year', 'genre_list'],
        title="Budget vs. Revenue (Colored by Rating, Sized by Popularity)",
        labels={'budget': 'Budget ($)', 'revenue': 'Revenue ($)', 'vote_average': 'Audience Rating'},
        color_continuous_scale='Plasma',
        opacity=0.7
    )
    fig_scatter.update_layout(xaxis_tickformat='$,.0f', yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_scatter, use_container_width=True)

    corr_val = fin_sub['budget'].corr(fin_sub['vote_average'])
    rev_vote_corr = fin_sub['revenue'].corr(fin_sub['vote_average'])
    bud_rev_corr = fin_sub['budget'].corr(fin_sub['revenue'])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Budget ⟷ Revenue Correlation", f"{bud_rev_corr:.2f}", "Strong Positive")
    c2.metric("Budget ⟷ Rating Correlation", f"{corr_val:.2f}", "Weak / Neutral")
    c3.metric("Revenue ⟷ Rating Correlation", f"{rev_vote_corr:.2f}", "Slight Positive")

# ----------------- 6. Q5: Studio Dominance -----------------
elif nav_choice == "🏢 Q5: Studio Dominance":
    st.subheader("🏢 Question 5: Production Studio Powerhouses")

    fin_comp = filtered_companies[filtered_companies['has_financial_data']].copy()
    studio_summary = fin_comp.groupby('company').agg(
        total_rev=('revenue', 'sum'),
        total_profit=('profit', 'sum'),
        avg_profit=('profit', 'mean'),
        movie_count=('title', 'count'),
        avg_rating=('vote_average', 'mean')
    ).reset_index()
    
    top_studios = studio_summary[studio_summary['movie_count'] >= 10].sort_values(by='total_rev', ascending=False).head(15)

    fig_studios = px.bar(
        top_studios,
        x='total_rev',
        y='company',
        orientation='h',
        title="Top 15 Studios by Total Box Office Gross ($ USD)",
        labels={'total_rev': 'Cumulative Gross ($)', 'company': 'Studio'},
        color='avg_profit',
        color_continuous_scale='Blues'
    )
    fig_studios.update_layout(yaxis=dict(autorange="reversed"), xaxis_tickformat='$,.0f')
    st.plotly_chart(fig_studios, use_container_width=True)

# ----------------- 7. Q6: Language Landscape -----------------
elif nav_choice == "🌍 Q6: Language Landscape":
    st.subheader("🌍 Question 6: Global & Non-English Cinema")

    top_langs = filtered_movies['original_language'].value_counts().head(10).index.tolist()
    lang_sub = filtered_movies[filtered_movies['original_language'].isin(top_langs)]

    fig_lang = px.box(
        lang_sub,
        x='original_language',
        y='vote_average',
        color='original_language',
        title="Vote Average Distribution Across Top Languages",
        labels={'original_language': 'Language', 'vote_average': 'Vote Rating'}
    )
    st.plotly_chart(fig_lang, use_container_width=True)
    st.caption("Observation: Japanese and Korean productions on TMDB exhibit high median audience ratings, outperforming English-language median scores.")

# ----------------- 8. Movie Explorer -----------------
elif nav_choice == "🔍 Movie Explorer":
    st.subheader("🔍 Interactive Movie Search & Exploration")

    search_query = st.text_input("Search movie title", placeholder="e.g. Inception, Avatar, Spirited Away...")
    
    display_df = filtered_movies.copy()
    if search_query:
        display_df = display_df[display_df['title'].str.contains(search_query, case=False, na=False)]

    st.write(f"Displaying **{len(display_df)}** matching movies:")
    
    st.dataframe(
        display_df[['title', 'release_year', 'genre_list', 'vote_average', 'vote_count', 'budget', 'revenue', 'profit', 'commercial_success_tier']]
        .head(100),
        use_container_width=True
    )
