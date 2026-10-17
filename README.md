# 🍽️ Restaurant Data Science & Machine Learning Platform

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Executable-yellow.svg)](https://colab.research.google.com/drive/14-qEJW_wheZwl-mgDZ4BYdwI4fGBiv0_)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning and geospatial intelligence suite designed to analyze restaurant ecosystems, predict consumer ratings, deliver personalized dining recommendations, classify culinary styles, and map regional market densities.

---

## 📌 Project Overview

This repository houses a four-stage data science pipeline executed on global restaurant metadata. By combining supervised regression, content-based natural language processing (NLP), multi-class ensemble classification, and interactive geospatial clustering, this project uncovers the key operational drivers behind restaurant ratings and consumer preferences.

### Core Objectives
1. **Task 1 — Aggregate Rating Prediction:** Train and benchmark regression models to predict a restaurant's aggregate consumer rating and isolate the most influential operational features.
2. **Task 2 — Content-Based Recommendation Engine:** Build a TF-IDF vectorization and Cosine Similarity engine capable of matching user preference profiles (`Cuisine`, `City`, `Price Tier`) to optimal dining options.
3. **Task 3 — Multi-Class Cuisine Classification:** Engineer a stratified Random Forest classification pipeline to predict a restaurant's primary cuisine and diagnose structural class biases.
4. **Task 4 — Location Intelligence & Geospatial Mapping:** Aggregate city-level culinary statistics, evaluate socioeconomic price-to-rating correlations, and render interactive cluster maps using `Folium`.

---

## 🗂️ Repository Structure

```text
├── data/
│   └── restaurant_dataset.csv          # Raw restaurant metadata (uploaded dynamically in Colab)
├── notebooks/
│   └── main_model.ipynb                # Complete interactive Google Colab notebook
├── outputs/
│   └── restaurant_interactive_map.html # Generated Folium geospatial cluster map
└── README.md                           # Project documentation, methodology, and findings
```

---

## 🔬 Methodology & Technical Architecture

### 1. Data Preprocessing & Feature Engineering
* **Missing Value Imputation:** Target leaks (`Rating color`, `Rating text`) and high-cardinality identifiers (`Restaurant Name`, `Address`) are decoupled from predictive matrices. Numerical predictors are imputed using median statistics to remain robust against extreme pricing outliers.
* **Binary Flag Encoding:** Categorical operational indicators (`Has Table booking`, `Has Online delivery`) are standardized into binary numerical flags ($1$ for `Yes`, $0$ for `No`).
* **Geospatial Filtering:** Invalid placeholder coordinates ($\text{Latitude} = 0$, $\text{Longitude} = 0$) and null spatial entries are filtered prior to spatial rendering.

---

### 2. Task-by-Task Analytical Breakdown

#### 📈 Task 1: Aggregate Rating Prediction (Regression)
* **Predictor Matrix ($X$):** `Average Cost for two`, `Has Table booking`, `Has Online delivery`, `Price range`, and `Votes`.
* **Target Vector ($y$):** `Aggregate rating` (continuous scale from $0.0$ to $4.9$).
* **Models Evaluated:**
  * `LinearRegression` (Baseline parametric benchmark)
  * `DecisionTreeRegressor` ($\text{max\_depth} = 5$)
  * `RandomForestRegressor` ($\text{n\_estimators} = 100$)
* **Evaluation Metrics:**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \quad , \quad R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

#### 🎯 Task 2: Content-Based Recommendation System
* **Metadata Soup Construction:** Combines cleaned, lowercased, whitespace-stripped tokens of `Cuisines`, `City`, and a synthesized `price_tier_<N>` tag into a unified string descriptor (`Content_Profile`).
* **Vectorization:** Transforms text profiles into a sparse numerical term-frequency matrix via `TfidfVectorizer(stop_words='english')`.
* **Mathematical Ranking:** Computes the Cosine Similarity between the user's query vector ($\mathbf{u}$) and every restaurant's profile vector ($\mathbf{v}_i$):
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}_i) = \frac{\mathbf{u} \cdot \mathbf{v}_i}{\|\mathbf{u}\| \|\mathbf{v}_i\|}$$
* **Quality Tie-Breaking:** Top candidate matches are ranked by `Match Score`, breaking ties using `Aggregate rating` and `Votes` to surface verified, high-quality establishments.

#### 🍱 Task 3: Cuisine Classification
* **Target Formulation:** Extracts the primary cuisine (`Primary_Cuisine`) from comma-separated strings and isolates the **Top 10 most frequent culinary classes** to prevent extreme sparse-class fragmentation.
* **Data Splitting:** Utilizes an $80/20$ **Stratified Train-Test Split** (`stratify=y`) so minority cuisine proportions are preserved across training and validation sets.
* **Model Architecture:** `RandomForestClassifier` configured with `max_depth=12` and `class_weight='balanced'` to penalize majority-class domination during tree node splits.

#### 🌍 Task 4: Location-Based Analysis & Interactive Mapping
* **Macro Aggregation:** Groups records by `City` to compute restaurant density (`Count`), mean rating (`Avg_Rating`), average price tier (`Avg_Price`), average cost for two, and unique cuisine diversity.
* **Interactive Visualization:** Uses `Folium` and `MarkerCluster` centered on the dataset's median coordinates. Markers are dynamically color-coded by consumer rating ($\ge 4.0$ highlighted as top-tier) with HTML popups displaying restaurant metadata.

---

## 📊 Key Findings & Business Insights

### 1. Drivers of Restaurant Ratings (Task 1)
* **Customer Engagement (`Votes`) is the Dominant Predictor:** Feature importance extraction from the Random Forest Regressor reveals that `Votes` accounts for the overwhelming majority of variance in `Aggregate rating`. Higher customer footfall and review volume strongly correlate with stable, higher ratings.
* **Non-Linearity of Rating Mechanics:** Ensemble tree models (`RandomForestRegressor` and `DecisionTreeRegressor`) substantially outperform `LinearRegression`, achieving an $R^2 > 0.94$ due to their ability to capture non-linear thresholds (such as the boundary between unrated $0.0$ establishments and rated establishments).

### 2. Recommendation Precision (Task 2)
* Combining explicit price tier tokens (`price_tier_2`, `price_tier_3`) directly inside the TF-IDF vocabulary allows a single cosine similarity operation to simultaneously filter by geography, match multi-cuisine preferences (e.g., *North Indian + Mughlai* or *Italian + Pizza*), and respect budget constraints.

### 3. Classification Biases & Structural Overlaps (Task 3)
* **Majority Class Skew:** Regional dominance of *North Indian*, *Chinese*, and *Fast Food* creates baseline density imbalances. Even with `class_weight='balanced'`, cuisines that share identical pricing and rating footprints (such as *Cafe*, *Bakery*, and *Fast Food*) exhibit cross-class confusion in the confusion matrix.
* **Multi-Cuisine Blending:** Because many restaurants operate hybrid kitchens (e.g., *North Indian, Chinese, Continental*), reducing the label to the first listed cuisine introduces label noise when predicting strictly from structural metrics like cost and votes.

### 4. Geospatial & Socioeconomic Patterns (Task 4)
* **Metropolitan Clustering:** Restaurant observations cluster heavily around primary urban hubs (most notably New Delhi, Gurgaon, and Noida), which also boast exponentially higher unique cuisine counts compared to secondary cities.
* **Positive Price-to-Rating Correlation:** City-level linear regression (`sns.regplot`) confirms a positive trend between a city's `Average Price Range` and its `Average Rating`, indicating that fine-dining and premium operational tiers consistently earn higher average consumer satisfaction scores.

---

## 🚀 Getting Started & Execution

### Prerequisites
Ensure you have Python 3.8+ installed along with the following scientific computing packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium
```

### Running in Google Colab
1. Open the notebook directly in [Google Colaboratory](https://colab.research.google.com/drive/14-qEJW_wheZwl-mgDZ4BYdwI4fGBiv0_).
2. Run **Cell 1** and upload your restaurant dataset CSV file when prompted by `files.upload()`.
3. Execute the individual task modules sequentially, or run the **Unified Restaurant Data Science Platform** cell at the bottom of the notebook to execute all four pipelines and generate `restaurant_interactive_map.html` in a single pass.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://restaurantratingregreappr-2p8ki2a7pan7rv4rimcjoy.streamlit.app/)

> **🚀 Explore the Live Application:** [Click Here to View the Deployed Streamlit Dashboard](https://restaurantratingregreappr-2p8ki2a7pan7rv4rimcjoy.streamlit.app/)
---

## 🔮 Future Enhancements
* **Currency Normalization:** Standardize `Average Cost for two` across international currencies using real-time exchange rates prior to training multi-country models.
* **Multi-Label Cuisine Classification:** Transition Task 3 from single-label multi-class classification to a `MultiLabelBinarizer` + `OneVsRestClassifier` architecture to capture fusion and multi-cuisine kitchens.
* **Sentiment Analysis Integration:** Incorporate customer review text using transformer embeddings to complement numerical metadata in the recommendation and rating prediction pipelines.
