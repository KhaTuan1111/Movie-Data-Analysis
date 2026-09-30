# 🎬 Exploratory Data Analysis & Business Intelligence on 9,500+ TMDB Popular Movies

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-TMDB%209500+-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/tushargoel04/9500plus-popular-movies-tmdb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎓 Course Final Project Assessment (40% of Grade)
- **Course**: Exploratory Data Analysis & Data Science
- **Group Assignment**: 2–3 Students per Group
- **Collaboration & Version Control**: Git & GitHub

### 👥 Team Members
| Student ID | Full Name | Assigned Responsibilities |
| :--- | :--- | :--- |
| `24120xxx` | **Member 1 (Leader)** | Data Ingestion, Profiling, Relational Preprocessing & Q1–Q2 Analysis |
| `24120xxx` | **Member 2** | Statistical Modeling, Feature Engineering & Q3–Q4 Analysis |
| `24120xxx` | **Member 3** | Studio/Language Analysis, Streamlit Interactive Dashboard & Q5–Q6 Analysis |

---

## 📌 Project Overview
The film industry is a high-stakes, multi-billion-dollar global entertainment market where investment decisions often range from tens of thousands of dollars for indie titles to over $350M for major franchise tentpoles. Understanding the determinants of commercial profitability, return on investment (ROI), critical audience reception, and release timing is paramount for directors, studios, and investors.

This project investigates the **[9500+ Popular Movies TMDB](https://www.kaggle.com/datasets/tushargoel04/9500plus-popular-movies-tmdb)** dataset sourced from Kaggle. Through an end-to-end data science pipeline, we uncover dataset quirks, engineer derived metrics, and provide empirical answers to 6 core industry questions.

---

## 🔍 Data Discovery & Structural Anomaly Detection
During initial data exploration, our team uncovered critical structural characteristics:
1. **The Exploded Cross-Product Quirk**:
   - The raw CSV has **83,739 rows**, but only **9,961 unique movie IDs**.
   - Each movie was unnested into a Cartesian product of its genres and production companies (e.g., *The Pope's Exorcist* appears across 18 rows).
   - *Direct aggregation without deduplication would artificially inflate revenues and budgets by up to 30x.*
2. **Missing / Unrecorded Financial Records**:
   - ~48% of movies have `budget = 0` and ~45% have `revenue = 0`. These represent unrecorded box office data rather than zero-cost films.
3. **Categorical Aliases**:
   - Language codes contained aliases (e.g., `'cn'` vs `'Chinese'`), which were mapped and standardized.

### Relational Data Model (Normalized Architecture)
To solve the cross-product distortion, we engineered a clean relational schema:
- **`movies_cleaned`** (9,961 rows): Exactly 1 record per movie, containing metadata, aggregated genres, aggregated studios, parsed dates, profit, and ROI.
- **`movies_genres`** (25,712 rows): 1-to-many bridge table (`movie_id` $\leftrightarrow$ `genre`).
- **`movies_companies`** (30,931 rows): 1-to-many bridge table (`movie_id` $\leftrightarrow$ `production_company`).

---

## ❓ Research Questions & Key Findings

### 🎯 Q1: Which genres yield the highest financial returns (Revenue vs. ROI)?
- **Box Office Titans**: **Animation** ($260M+ mean revenue), **Adventure** ($245M+), and **Science Fiction** lead in gross revenue.
- **Capital Efficiency Champions**: **Horror** and **Mystery** deliver the highest median ROI (**>200%–250%**). With low production budgets ($10M–$15M median), horror films consistently generate outsized profit multiples with minimal capital risk.

### ⏳ Q2: How have movie volume, budgets, and runtimes evolved over time (1920–2023)?
- **Volume Explosion**: Over 70% of popular films were released after 2000, accelerated by digital production and global streaming.
- **Escalating Budgets**: Median production budgets grew from <$5M in the 1960s to >$40M in recent decades.
- **Runtime Golden Ratio**: Across 9 decades, median theatrical runtime has remained firmly anchored between **98 and 108 minutes**, reflecting standard audience attention spans and theater scheduling cycles.

### 📅 Q3: Does release timing (Month / Season) dictate box office success?
- **The Dual-Peak Cycle**:
  - **Summer Window (May–July)**: Averages $150M–$175M gross, driven by school vacations and blockbuster tentpoles.
  - **Holiday Season (November–December)**: Surges due to Thanksgiving/Christmas family attendance and Academy Award campaigns.
- **The "Dump Months" (January & September)**: Generate the lowest revenues ($70M–$85M). Studios strategically release niche or low-confidence projects here.

### ⭐ Q4: Does a higher budget guarantee higher ratings or revenue?
- **Money Buys Reach, Not Acclaim**:
  - Budget strongly correlates with Revenue ($r = 0.72$), confirming that marketing and scale drive ticket sales.
  - Budget correlation with audience rating (`vote_average`) is virtually **zero** ($r = 0.05$). Massive spending does **not** ensure audience enjoyment.
  - True cinematic masterpieces frequently emerge from mid-budget or independent productions.

### 🏢 Q5: Which studios dominate total gross vs. average profitability?
- **Volume Leaders**: **Warner Bros.**, **Universal Pictures**, and **Walt Disney Pictures** lead in cumulative box office gross ($60B–$80B+ across their catalog).
- **Efficiency Leaders**: Specialized studios like **Marvel Studios** and **Pixar** boast average profits exceeding **$350M–$500M per release**.

### 🌍 Q6: How do international non-English films compare with Hollywood?
- **The Quality Premium**: Non-English films—specifically **Japanese** (mean rating ~7.30) and **Korean** (~7.15)—consistently score higher than English releases (mean ~6.48) on TMDB, driven by strong artistic merit, auteur storytelling, and passionate global fanbases.

---

## 📊 Visual Highlights

| Analysis | Chart Preview |
| :--- | :--- |
| **Q1: Genre Financials & ROI** | `reports/figures/q1_genre_finances.png` |
| **Q2: Decades Evolution** | `reports/figures/q2_decade_trends.png` |
| **Q3: Release Seasonality** | `reports/figures/q3_seasonality.png` |
| **Q4: Correlation Matrix & Scatter** | `reports/figures/q4_correlations.png` |
| **Q5: Studio Leaderboards** | `reports/figures/q5_top_studios.png` |
| **Q6: Global Language Landscape** | `reports/figures/q6_language_landscape.png` |

---

## 📁 Repository Structure

```
Movie-Data-Analysis/
├── data/
│   ├── raw/
│   │   └── 9616_UNIQUE_IMDB.csv          # Raw TMDB dataset (83,739 rows)
│   └── processed/
│       ├── movies_cleaned.csv            # Cleaned unique movies (9,961 records)
│       ├── movies_genres.csv             # Movie-Genre bridge table (25,712 records)
│       └── movies_companies.csv          # Movie-Company bridge table (30,931 records)
├── notebooks/
│   └── movie_data_analysis.ipynb         # Full executed Jupyter Notebook
├── reports/
│   └── figures/                          # High-resolution exported figures
│       ├── q1_genre_finances.png
│       ├── q2_decade_trends.png
│       ├── q3_seasonality.png
│       ├── q4_correlations.png
│       ├── q5_top_studios.png
│       └── q6_language_landscape.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Modular dataset loading utilities
│   ├── preprocessor.py                   # Automated cleaning and normalization
│   └── analysis_utils.py                 # Statistical aggregation and visualization
├── scripts/
│   └── build_notebook.py                 # Programmatic notebook builder
├── app.py                                # Streamlit interactive presentation dashboard
├── requirements.txt                      # Project dependency specification
├── .gitignore                            # Git ignore rules
└── README.md                             # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Clone the repository and install required packages:
```bash
git clone https://github.com/KhaTuan1111/Movie-Data-Analysis.git
cd Movie-Data-Analysis
python -m pip install -r requirements.txt
```

### 2. Run Data Preprocessing Pipeline
To regenerate the cleaned relational datasets:
```bash
python src/preprocessor.py
```

### 3. Generate Analytical Figures
To reproduce all figures in `reports/figures/`:
```bash
python src/analysis_utils.py
```

### 4. Open the Jupyter Notebook
Explore the fully executed analysis with markdown narratives:
```bash
jupyter notebook notebooks/movie_data_analysis.ipynb
```

### 5. Launch the Interactive Web Dashboard
Run the presentation dashboard for live demonstration to the teacher:
```bash
streamlit run app.py
```

---

## 💡 Strategic Takeaways for Filmmakers & Investors
1. **The Barbell Investment Strategy**: Combine low-budget, high-ROI **Horror** productions (downside protection) with high-upside **Animation/Adventure** tentpoles.
2. **Release Calendar Discipline**: Secure summer (May–July) or holiday (Nov–Dec) release dates for high-budget commercial titles.
3. **Budget Prudence**: Recognize that budget does not buy audience affection; invest in strong writing and directorial vision.

---
*Created for Academic Evaluation | Final Group Project.*