import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Page configuration
st.set_page_config(page_title="Restaurant Analytics Platform", layout="wide")
st.title("🍽️ Integrated Restaurant Data Science Platform")

# --- DATA SEED & LAZY CACHING ---
@st.cache_data
@st.cache_data
def load_and_clean_data():
    backup_url = "https://githubusercontent.com"
    
    # Check if local file exists AND actually has content inside it
    if os.path.exists('zomato.csv') and os.path.getsize('zomato.csv') > 500:
        try:
            df = pd.read_csv('zomato.csv', encoding='latin-1')
            st.sidebar.success("Loaded dataset locally from repository!")
        except Exception:
            df = pd.read_csv(backup_url, encoding='latin-1')
            st.sidebar.info("Streaming dataset dynamically from cloud network backup...")
    else:
        # If the local file is empty or missing, stream it directly from the URL stream
        df = pd.read_csv(backup_url, encoding='latin-1')
        st.sidebar.info("Streaming dataset dynamically from cloud network backup...")
        
    df_clean = df.copy()
    df_clean['Cuisines'] = df_clean['Cuisines'].fillna('Unknown')
    df_clean['City'] = df_clean['City'].fillna('Unknown')
    return df_clean

df_clean = load_and_clean_data()

# --- PRE-COMPUTE SHARED ML COMPONENTS ---
binary_map = {'Yes': 1, 'No': 0}
df_predict = df_clean.dropna(subset=['Aggregate rating']).copy()
df_predict['Has Table booking'] = df_predict['Has Table booking'].map(binary_map).fillna(0)
df_predict['Has Online delivery'] = df_predict['Has Online delivery'].map(binary_map).fillna(0)
features_reg = ['Average Cost for two', 'Has Table booking', 'Has Online delivery', 'Price range', 'Votes']

# --- SIDEBAR INTERFACE NAVIGATION ---
st.sidebar.header("Navigation Control Center")
app_mode = st.sidebar.radio("Select Analytics Task Component", [
    "Overview & Spatial Maps", 
    "Rating Prediction Regressor", 
    "Content Recommendation Engine", 
    "Cuisine Pattern Classifier"
])

# ==============================================================================
# MODE 1: GEOGRAPHICAL INTERACTIVE MAPS
# ==============================================================================
if app_mode == "Overview & Spatial Maps":
    st.header("📍 Geographical Distribution & Hotspot Concentrations")
    
    city_stats = df_clean.groupby('City').agg(
        Total_Restaurants=('Restaurant ID', 'count'),
        Avg_Rating=('Aggregate rating', 'mean'),
        Avg_Price_Tier=('Price range', 'mean')
    ).sort_values(by='Total_Restaurants', ascending=False).reset_index()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Top Concentration Hubs")
        st.dataframe(city_stats.head(10), use_container_width=True)
        
    with col2:
        st.subheader("Interactive Cluster Map Explorer")
        geo_data = df_clean[(df_clean['Latitude'] != 0) & (df_clean['Longitude'] != 0) & (df_clean['Latitude'].notna())].copy()
        
        if not geo_data.empty:
            map_center = [geo_data['Latitude'].median(), geo_data['Longitude'].median()]
            restaurant_map = folium.Map(location=map_center, zoom_start=10, control_scale=True)
            marker_cluster = MarkerCluster().add_to(restaurant_map)

            for idx, row in geo_data.head(300).iterrows():
                popup_text = f"<b>{row['Restaurant Name']}</b><br>Rating: {row['Aggregate rating']}<br>Cuisines: {row['Cuisines']}"
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color="red" if row['Aggregate rating'] >= 4.0 else "blue", icon="cutlery")
                ).add_to(marker_cluster)
                
            st_folium(restaurant_map, width=800, height=500, returned_objects=[])
        else:
            st.info("No valid geographical coordinates found to render the map dashboard.")

# ==============================================================================
# MODE 2: RATING PREDICTION MODEL
# ==============================================================================
elif app_mode == "Rating Prediction Regressor":
    st.header("📊 Restaurant_Rating_Regressor Engine")
    
    X_reg = df_predict[features_reg].fillna(df_predict[features_reg].median())
    y_reg = df_predict['Aggregate rating']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_r, y_train_r)
    
    st.subheader("Simulate a Custom Restaurant Entry Profile")
    in_cost = st.number_input("Average Cost for Two", min_value=0, value=500, step=50)
    in_booking = st.selectbox("Offers Table Booking?", ["No", "Yes"])
    in_delivery = st.selectbox("Offers Online Delivery?", ["No", "Yes"])
    in_price = st.slider("Price Range Tier Descriptor", 1, 4, 2)
    in_votes = st.number_input("Accumulated Feedback Engagement Count (Votes)", min_value=0, value=150, step=10)
    
    if st.button("Calculate Predicted Performance Score"):
        input_data = pd.DataFrame([[
            in_cost, 
            1 if in_booking == "Yes" else 0, 
            1 if in_delivery == "Yes" else 0, 
            in_price, 
            in_votes
        ]], columns=features_reg)
        
        prediction = rf_regressor.predict(input_data)[0]
        st.metric(label="Predicted Aggregate Rating out of 5.0", value=f"⭐ {prediction:.2f}")

# ==============================================================================
# MODE 3: RECOMMENDATION ENGINE
# ==============================================================================
elif app_mode == "Content Recommendation Engine":
    st.header("🎯 Content_Based_Restaurant_Recommender Matrix")
    
    df_clean['Cuisines_Cleaned'] = df_clean['Cuisines'].apply(lambda x: " ".join([i.strip().lower().replace(" ", "") for i in x.split(',')]))
    df_clean['City_Cleaned'] = df_clean['City'].str.lower().str.replace(" ", "")
    df_clean['Price_Cleaned'] = "price_tier_" + df_clean['Price range'].astype(str)
    df_clean['Content_Profile'] = df_clean['Cuisines_Cleaned'] + " " + df_clean['City_Cleaned'] + " " + df_clean['Price_Cleaned']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_clean['Content_Profile'])
    
    unique_cities = sorted(df_clean['City'].unique())
    selected_city = st.selectbox("Target Destination City", unique_cities)
    selected_cuisine = st.text_input("Enter Preferred Cuisine Keywords (e.g. Italian, Fast Food)", "North Indian")
    selected_price = st.slider("Target Maximum Pricing Bracket Tier", 1, 4, 2)
    
    if st.button("Generate Tailored Recommendation Matches"):
        user_cuisines = " ".join([i.strip().lower().replace(" ", "") for i in selected_cuisine.split(',')])
        user_city = selected_city.lower().replace(" ", "")
        user_price = f"price_tier_{selected_price}"
        user_profile_str = f"{user_cuisines} {user_city} {user_price}"
        
        user_vector = tfidf.transform([user_profile_str])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-5:][::-1]
        
        recommendations = df_clean.iloc[top_indices].copy()
        recommendations['Match Score'] = similarity_scores[top_indices]
        
        st.dataframe(recommendations[['Restaurant Name', 'City', 'Cuisines', 'Price range', 'Aggregate rating', 'Match Score']], use_container_width=True)

# ==============================================================================
# MODE 4: CUISINE CLASSIFICATION
# ==============================================================================
elif app_mode == "Cuisine Pattern Classifier":
    st.header("🏷️ Primary_Cuisine_Classifier System")
    
    df_clean['Primary_Cuisine'] = df_clean['Cuisines'].apply(lambda x: x.split(',')[0].strip())
    top_cuisines = df_clean['Primary_Cuisine'].value_counts().nlargest(10).index
    df_cls = df_clean[df_clean['Primary_Cuisine'].isin(top_cuisines)].copy()
    
    if not df_cls.empty:
        target_encoder = LabelEncoder()
        y_cls = target_encoder.fit_transform(df_cls['Primary_Cuisine'])
        X_cls = df_cls[features_reg].fillna(df_cls[features_reg].median())
        
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
        
        rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
        rf_classifier.fit(X_train_c, y_train_c)
        
        st.subheader("Predict Cuisine Type Based on Operational Tiering")
        c_cost = st.number_input("Average Cost Setup for Two", min_value=0, value=700, key="cls_cost")
        c_booking = st.selectbox("Requires Table Booking reservation?", ["No", "Yes"], key="cls_book")
        c_delivery = st.selectbox("Supports Online Home Delivery distributions?", ["No", "Yes"], key="cls_del")
        c_price = st.slider("Pricing Tier Index Code", 1, 4, 3, key="cls_price")
        c_votes = st.number_input("Historical Engagement Interaction Volume (Votes)", min_value=0, value=400, key="cls_votes")
        
        if st.button("Execute Cuisine Class Inference Mapping"):
            input_cls_data = pd.DataFrame([[
                c_cost,
                1 if c_booking == "Yes" else 0,
                1 if c_delivery == "Yes" else 0,
                c_price,
                c_votes
            ]], columns=features_reg)
            
            class_idx = rf_classifier.predict(input_cls_data)[0]
            predicted_cuisine = target_encoder.classes_[class_idx]
