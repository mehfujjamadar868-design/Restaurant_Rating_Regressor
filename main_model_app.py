# ==============================================================================
# UNIFIED RESTAURANT DATA SCIENCE PLATFORM (ALL TASKS CONSOLIDATED)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# --- 0. INITIALIZATION & GLOBAL DATA CLEANING ---
# Ensure your data frame 'df' is loaded. If you need to re-read it, uncomment below:
# df = pd.read_csv('zomato.csv')

print("⚡ Starting Unified Pipeline Execution...\n")
df_clean = df.copy()

# ==============================================================================
# TASK 1: RATING PREDICTION MODEL
# ==============================================================================
print("--- [TASK 1] Executing Rating Prediction ---")
df_predict = df_clean.dropna(subset=['Aggregate rating']).copy()
binary_map = {'Yes': 1, 'No': 0}
df_predict['Has Table booking'] = df_predict['Has Table booking'].map(binary_map).fillna(0)
df_predict['Has Online delivery'] = df_predict['Has Online delivery'].map(binary_map).fillna(0)

features_reg = ['Average Cost for two', 'Has Table booking', 'Has Online delivery', 'Price range', 'Votes']
X_reg = df_predict[features_reg].fillna(df_predict[features_reg].median())
y_reg = df_predict['Aggregate rating']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_r, y_train_r)
reg_preds = rf_regressor.predict(X_test_r)

print(f"✅ Regression Complete! MSE: {mean_squared_error(y_test_r, reg_preds):.4f} | R2 Score: {r2_score(y_test_r, reg_preds):.4f}\n")


# ==============================================================================
# TASK 2: RECOMMENDATION SYSTEM ENGINES
# ==============================================================================
print("--- [TASK 2] Initializing Content-Based Recommendation Matrix ---")
df_clean['Cuisines'] = df_clean['Cuisines'].fillna('Unknown')
df_clean['City'] = df_clean['City'].fillna('Unknown')

df_clean['Cuisines_Cleaned'] = df_clean['Cuisines'].apply(lambda x: " ".join([i.strip().lower().replace(" ", "") for i in x.split(',')]))
df_clean['City_Cleaned'] = df_clean['City'].str.lower().str.replace(" ", "")
df_clean['Price_Cleaned'] = "price_tier_" + df_clean['Price range'].astype(str)
df_clean['Content_Profile'] = df_clean['Cuisines_Cleaned'] + " " + df_clean['City_Cleaned'] + " " + df_clean['Price_Cleaned']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_clean['Content_Profile'])

def recommend_restaurants(user_preferences, top_n=5):
    user_cuisines = " ".join([i.strip().lower().replace(" ", "") for i in user_preferences.get('cuisine', '').split(',')])
    user_city = user_preferences.get('city', '').lower().replace(" ", "")
    user_price = f"price_tier_{user_preferences.get('price_range', '')}"
    user_profile_str = f"{user_cuisines} {user_city} {user_price}"

    user_vector = tfidf.transform([user_profile_str])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    recommendations = df_clean.iloc[top_indices].copy()
    recommendations['Match Score'] = similarity_scores[top_indices]
    return recommendations[['Restaurant Name', 'City', 'Cuisines', 'Price range', 'Aggregate rating', 'Match Score']]

# Test execution
sample_pref = {'cuisine': 'North Indian, Mughlai', 'city': 'New Delhi', 'price_range': 2}
print("✅ Recommendation Matrix Generated! Test Recommendation Sample Output:")
display(recommend_restaurants(sample_pref, top_n=3))
print("\n")


# ==============================================================================
# TASK 3: CUISINE CLASSIFICATION MODEL
# ==============================================================================
print("--- [TASK 3] Executing Cuisine Classification ---")
df_clean['Primary_Cuisine'] = df_clean['Cuisines'].apply(lambda x: x.split(',')[0].strip())
top_cuisines = df_clean['Primary_Cuisine'].value_counts().nlargest(10).index
df_cls = df_clean[df_clean['Primary_Cuisine'].isin(top_cuisines)].copy()

target_encoder = LabelEncoder()
y_cls = target_encoder.fit_transform(df_cls['Primary_Cuisine'])
X_cls = df_cls[features_reg].fillna(df_cls[features_reg].median())

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train_c, y_train_c)
cls_preds = rf_classifier.predict(X_test_c)

print(f"✅ Classification Complete! Model Multi-Class Accuracy: {accuracy_score(y_test_c, cls_preds):.4f}\n")


# ==============================================================================
# TASK 4: LOCATION ANALYSIS & INTERACTIVE MAPPING
# ==============================================================================
print("--- [TASK 4] Building Location Dashboards & Interactive Maps ---")
# 1. Summary Aggregations
city_stats = df_clean.groupby('City').agg(
    Count=('Restaurant ID', 'count'),
    Avg_Rating=('Aggregate rating', 'mean'),
    Avg_Price=('Price range', 'mean')
).sort_values(by='Count', ascending=False).reset_index()

print("Top 5 Dense Restaurant Locations Found:")
display(city_stats.head(5))

# 2. Interactive Map Generation using Folium
geo_data = df_clean[(df_clean['Latitude'] != 0) & (df_clean['Longitude'] != 0) & (df_clean['Latitude'].notna())].copy()

# Focus map center on median coordinates of the dataset
map_center = [geo_data['Latitude'].median(), geo_data['Longitude'].median()]
restaurant_map = folium.Map(location=map_center, zoom_start=10, control_scale=True)
marker_cluster = MarkerCluster().add_to(restaurant_map)

# Sample map markers (limit to 500 to keep Colab notebook light and fast)
for idx, row in geo_data.head(500).iterrows():
    popup_text = f"<b>{row['Restaurant Name']}</b><br>Rating: {row['Aggregate rating']}<br>Cuisines: {row['Cuisines']}"
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="red" if row['Aggregate rating'] >= 4.0 else "blue", icon="cutlery")
    ).add_to(marker_cluster)

# Save the map locally
restaurant_map.save("restaurant_interactive_map.html")
print("✅ Location Analysis Finished! Interactive Map saved as 'restaurant_interactive_map.html'")

print("\n🎉 ALL TASKS ARE COMPLETED SUCCESSFULLY!")
