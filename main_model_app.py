import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Restaurant Intelligence & ML Suite",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍽️ Unified Restaurant Intelligence & Machine Learning Suite")
st.markdown(
    "An interactive end-to-end platform featuring **Aggregate Rating Regression**, "
    "**TF-IDF Content-Based Recommendations**, **Multi-Class Cuisine Classification**, "
    "and **Geospatial Folium Analytics**."
)

# ------------------------------------------------------------------------------
# DATA LOADING & GLOBAL PREPROCESSING
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_and_clean_data(uploaded_file=None):
    df = None

    # 1. Try reading user-uploaded file from the Streamlit sidebar first
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="utf-8", encoding_errors="replace")
        except pd.errors.EmptyDataError:
            st.sidebar.error("⚠️ The uploaded CSV file is empty (0 bytes). Please upload a valid CSV.")
            return None

    # 2. Otherwise, scan the repository directory for any valid, non-empty CSV file
    if df is None:
        preferred_files = ["Dataset .csv", "Dataset.csv", "zomato.csv", "restaurants.csv"]
        repo_csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
        candidate_files = preferred_files + [f for f in repo_csvs if f not in preferred_files]

        for fname in candidate_files:
            if os.path.exists(fname) and os.path.getsize(fname) > 10:
                try:
                    temp_df = pd.read_csv(fname, encoding="utf-8", encoding_errors="replace")
                    if not temp_df.empty and "Aggregate rating" in temp_df.columns:
                        df = temp_df
                        break
                except Exception:
                    continue

    if df is None or df.empty:
        return None

    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=["Aggregate rating"]).copy()

    # Map binary columns safely
    binary_map = {"Yes": 1, "No": 0, 1: 1, 0: 0}
    for col in ["Has Table booking", "Has Online delivery"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(binary_map).fillna(0).astype(int)
        else:
            df_clean[col] = 0

    # Ensure core numeric predictors are clean
    num_cols = [
        "Average Cost for two", "Has Table booking", "Has Online delivery",
        "Price range", "Votes", "Aggregate rating"
    ]
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Clean text descriptors
    df_clean["Cuisines"] = df_clean["Cuisines"].fillna("Unknown").astype(str)
    df_clean["City"] = df_clean["City"].fillna("Unknown").astype(str)
    if "Restaurant Name" not in df_clean.columns:
        df_clean["Restaurant Name"] = [f"Restaurant #{i+1}" for i in range(len(df_clean))]
    if "Restaurant ID" not in df_clean.columns:
        df_clean["Restaurant ID"] = np.arange(1, len(df_clean) + 1)

    return df_clean

# Sidebar Upload & Filters
st.sidebar.header("📂 Dataset Configuration")
uploaded_csv = st.sidebar.file_uploader("Upload Restaurant CSV (e.g., Dataset.csv)", type=["csv"])

df_clean = load_and_clean_data(uploaded_csv)

if df_clean is None:
    st.info(
        "👋 **Welcome!** Please upload your restaurant dataset CSV file in the **left sidebar** "
        "to launch the analytics suite."
    )
    st.stop()

st.sidebar.success(f"✅ Loaded **{len(df_clean):,}** restaurants across **{df_clean['City'].nunique()}** cities.")
exclude_unrated = st.sidebar.checkbox("Exclude Unrated Restaurants (Rating = 0.0) in Regression", value=False)

# ------------------------------------------------------------------------------
# CACHED MODEL TRAINING PIPELINES
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training Regression Models (Task 1)...")
def train_regression_models(df, drop_zero_ratings=False):
    df_reg = df[df["Aggregate rating"] > 0].copy() if drop_zero_ratings else df.copy()
    features = ["Average Cost for two", "Has Table booking", "Has Online delivery", "Price range", "Votes"]
    X = df_reg[features]
    y = df_reg["Aggregate rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree (max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest (100 Trees)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    metrics_list = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        metrics_list.append({
            "Model": name,
            "MSE": round(mse, 4),
            "RMSE": round(np.sqrt(mse), 4),
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "R² Score": round(r2_score(y_test, preds), 4)
        })
        trained_models[name] = model

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": trained_models["Random Forest (100 Trees)"].feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return trained_models, pd.DataFrame(metrics_list), importance_df, features


@st.cache_resource(show_spinner="Building TF-IDF Content Matrix (Task 2)...")
def build_recommender_engine(df):
    df_rec = df.copy()
    df_rec["Cuisines_Cleaned"] = df_rec["Cuisines"].apply(
        lambda x: " ".join([i.strip().lower().replace(" ", "") for i in str(x).split(",")])
    )
    df_rec["City_Cleaned"] = df_rec["City"].astype(str).str.lower().str.replace(" ", "")
    df_rec["Price_Cleaned"] = "price_tier_" + df_rec["Price range"].astype(int).astype(str)
    df_rec["Content_Profile"] = (
        df_rec["Cuisines_Cleaned"] + " " + df_rec["City_Cleaned"] + " " + df_rec["Price_Cleaned"]
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_rec["Content_Profile"])
    return tfidf, tfidf_matrix, df_rec


@st.cache_resource(show_spinner="Training Multi-Class Cuisine Classifier (Task 3)...")
def train_cuisine_classifier(df):
    df_cls = df.copy()
    df_cls["Primary_Cuisine"] = df_cls["Cuisines"].apply(lambda x: str(x).split(",")[0].strip())
    valid_cuisines = df_cls[df_cls["Primary_Cuisine"] != "Unknown"]["Primary_Cuisine"]
    top_10_cuisines = valid_cuisines.value_counts().nlargest(10).index
    df_filtered = df_cls[df_cls["Primary_Cuisine"].isin(top_10_cuisines)].copy()

    city_encoder = LabelEncoder()
    df_filtered["City_Encoded"] = city_encoder.fit_transform(df_filtered["City"].astype(str))

    features_cls = [
        "Average Cost for two", "Has Table booking", "Has Online delivery",
        "Price range", "Votes", "Aggregate rating", "City_Encoded"
    ]

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df_filtered["Primary_Cuisine"])
    X = df_filtered[features_cls]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(
        y_test, y_pred, target_names=target_encoder.classes_, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    cm = confusion_matrix(y_test, y_pred)

    return clf, target_encoder, city_encoder, acc, report_df, cm, features_cls


# ------------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# ------------------------------------------------------------------------------
tab_overview, tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Task 1: Rating Prediction",
    "🍲 Task 2: Recommendation Engine",
    "🏷️ Task 3: Cuisine Classification",
    "🗺️ Task 4: Geospatial Intelligence"
])

# === TAB 0: DATASET OVERVIEW ===
with tab_overview:
    st.subheader("📊 Executive Dataset Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Restaurants", f"{len(df_clean):,}")
    k2.metric("Unique Cities", f"{df_clean['City'].nunique():,}")
    k3.metric("Mean Aggregate Rating", f"{df_clean['Aggregate rating'].mean():.2f} ⭐")
    k4.metric("Median Votes", f"{int(df_clean['Votes'].median()):,}")

    st.markdown("#### Cleaned Dataset Preview")
    st.dataframe(df_clean.head(15), use_container_width=True)

# === TAB 1: RATING PREDICTION ===
with tab1:
    st.subheader("📈 Task 1: Predicting Aggregate Restaurant Ratings")
    trained_models, metrics_df, importance_df, reg_features = train_regression_models(df_clean, exclude_unrated)

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        st.markdown("#### Model Evaluation Benchmarks")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("#### Feature Importance Breakdown (Random Forest)")
        fig_imp, ax_imp = plt.subplots(figsize=(7, 3.8))
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            palette="viridis",
            ax=ax_imp
        )
        ax_imp.set_title("Structural Feature Weights")
        ax_imp.set_xlabel("Relative Importance Score")
        ax_imp.grid(axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig_imp)

    with col_right:
        st.markdown("#### 🔮 Live Restaurant Rating Simulator")
        st.caption("Adjust operational parameters below to predict the expected Zomato Aggregate Rating:")
        with st.form("rating_simulator_form"):
            sim_model_choice = st.selectbox("Select Regression Model", list(trained_models.keys()), index=2)
            sim_cost = st.number_input("Average Cost for Two", min_value=0, max_value=500000, value=int(df_clean["Average Cost for two"].median()), step=50)
            sim_price_tier = st.slider("Price Range Tier (1 = Budget, 4 = Luxury)", 1, 4, 2)
            sim_votes = st.number_input("Customer Votes Count", min_value=0, max_value=50000, value=int(df_clean["Votes"].median()), step=10)
            c_b1, c_b2 = st.columns(2)
            sim_table = c_b1.selectbox("Has Table Booking?", ["No", "Yes"])
            sim_delivery = c_b2.selectbox("Has Online Delivery?", ["Yes", "No"])
            submit_sim = st.form_submit_button("Predict Aggregate Rating 🚀")

        if submit_sim:
            input_df = pd.DataFrame([{
                "Average Cost for two": sim_cost,
                "Has Table booking": 1 if sim_table == "Yes" else 0,
                "Has Online delivery": 1 if sim_delivery == "Yes" else 0,
                "Price range": sim_price_tier,
                "Votes": sim_votes
            }])
            pred_rating = trained_models[sim_model_choice].predict(input_df)[0]
            pred_rating = float(np.clip(pred_rating, 0.0, 5.0))
            st.success(f"### Predicted Aggregate Rating: **{pred_rating:.2f} / 5.00** ⭐")

# === TAB 2: RECOMMENDATION ENGINE ===
with tab2:
    st.subheader("🍲 Task 2: Content-Based Restaurant Recommendation Engine")
    tfidf, tfidf_matrix, df_rec = build_recommender_engine(df_clean)

    all_cuisines = sorted(list({
        c.strip() for row in df_rec["Cuisines"].dropna() for c in str(row).split(",") if c.strip() and c.strip() != "Unknown"
    }))
    all_cities = sorted(df_rec["City"].unique().tolist())
    default_city_idx = all_cities.index("New Delhi") if "New Delhi" in all_cities else 0

    col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.2, 1, 1])
    selected_cuisines = col_f1.multiselect(
        "Preferred Cuisines",
        options=all_cuisines,
        default=[c for c in ["North Indian", "Mughlai"] if c in all_cuisines][:2]
    )
    selected_city = col_f2.selectbox("Target City", options=all_cities, index=default_city_idx)
    selected_price = col_f3.selectbox("Preferred Price Tier", options=[1, 2, 3, 4], index=1)
    top_n_rec = col_f4.slider("Number of Recommendations", min_value=3, max_value=20, value=5)

    if st.button("Generate Personalized Recommendations 🔍", type="primary"):
        cuisine_str = ", ".join(selected_cuisines) if selected_cuisines else "North Indian"
        user_cuisines_clean = " ".join([i.strip().lower().replace(" ", "") for i in cuisine_str.split(",")])
        user_city_clean = selected_city.lower().replace(" ", "")
        user_price_clean = f"price_tier_{selected_price}"
        user_profile_str = f"{user_cuisines_clean} {user_city_clean} {user_price_clean}"

        st.code(f"Generated TF-IDF Query Vector Profile: '{user_profile_str}'", language="text")

        user_vector = tfidf.transform([user_profile_str])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

        candidate_indices = similarity_scores.argsort()[-50:][::-1]
        recs = df_rec.iloc[candidate_indices].copy()
        recs["Match Score"] = np.round(similarity_scores
