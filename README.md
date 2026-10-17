# 🍽️ Restaurant Analytics Platform: The Ultimate Foodie Data Stack

An absolute banger of an end-to-end data science platform that uses machine learning to decode restaurant trends, predict customer scores, drop tailored food recommendations, and map local dining hotspots. No cap, this project is hitting different.

Live demo link? Slap it right here 🚀 `[Insert Streamlit Link Here]`

---

## ⚡ The Vibe Check (TL;DR)
This isn't just a basic script; it’s a full ecosystem built to flex major predictive powers. We out here using tree-based ensembles and spatial text vectorization to transform chaotic restaurant metrics into clean, deployable UI features. 

---

## 🧠 The Brains Behind the App (Model Specifics)

We didn't come to play. The platform deploys three hyper-focused systems:

### 1. 📊 `Restaurant_Rating_Regressor`
- **The Workload:** Predicts continuous numerical scores (restaurant ratings out of 5.0).
- **The Flex:** Uses a **Random Forest Regressor** to analyze how price tiers and user engagement (`Votes`) shift the score baseline. 
- **The Tea:** Spills the fact that `Votes` holds a massive ~97% node importance weight. High engagement = high clout.

### 2. 🎯 `Content_Based_Restaurant_Recommender`
- **The Workload:** Unsupervised text-matching recommendation engine.
- **The Flex:** Turns cuisine types, cities, and price tags into a numeric matrix using **TF-IDF Vectorization**, then calculates the **Cosine Similarity** angle to drop the top 5 match choices tailored to user inputs. 

### 3. 🏷️ `Primary_Cuisine_Classifier`
- **The Workload:** Discrete multi-class categorical grouping.
- **The Flex:** Uses a **Random Forest Classifier** with balanced class weights to map structural metrics (cost, reservation options, delivery status) back to a dominant cuisine category.

## 🛡️ Key Insights & Reality Checks

- **Metropolitan Bias:** The data is heavily concentrated around capital city hubs. If a restaurant is outside the main grid, it's giving ghost town.
- **Price Trapped:** Higher price tiers show a direct positive correlation with higher average ratings. Boujee spots simply get better reviews, period.
- **Class Imbalance:** Massively popular categories like 'Fast Food' and 'North Indian' tend to dominate the classification boundaries, while niche cuisines get slept on unless balanced out.

---

## 💅 Built With
- **Python** (The absolute backbone)
- **Streamlit** (For the frontend UI glow-up)
- **Scikit-Learn** (For training the models)
- **Folium & Streamlit-Folium** (For mapping out interactive pins)
- **Matplotlib & Seaborn** (For the visual data checks)

---
Made with 💖, data, and too much caffeine. Leave a ⭐ if this repo passed the vibe check!
