# diet_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load Datasets
# ---------------------------
@st.cache_data
def load_data():
    data1 = pd.read_csv("https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/Food_and_Nutrition%20new.csv")
    data2 = pd.read_csv("https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/detailed_meals_macros_.csv")
    data3 = pd.read_csv("https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/diet_recommendations_dataset.csv")
    return pd.concat([data1, data2, data3], ignore_index=True)

data = load_data()

# Clean data
data.columns = [c.strip().title() for c in data.columns]
data['Disease'] = data['Disease'].astype(str).str.title()

meal_cols = ['Breakfast Suggestion', 'Lunch Suggestion', 'Dinner Suggestion', 'Snack Suggestion']

# ---------------------------
# UI Section
# ---------------------------
st.title("🥗 AI Diet Suggestion App")
st.markdown("Personalized diet plans based on age, weight, activity, and health condition.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 1, 100, 25)
    weight = st.slider("Weight (kg)", 30, 150, 70)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    activity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    preference = st.radio("Diet Preference", ["Vegetarian", "Non-Vegetarian"])
    cuisine = st.selectbox("Cuisine Preference", ["Indian", "Continental", "Asian", "Mediterranean", "Any"])

# Disease & allergy filters
diseases = sorted(set(d.title().strip() for d in data['Disease'].dropna() if d.lower() not in ['nan', 'none', 'unknown']))
selected_diseases = st.multiselect("Select any diseases (if applicable):", diseases)
allergies = st.text_input("Enter any food allergies (comma-separated):")

# ---------------------------
# Suggest Meals
# ---------------------------
def suggest_meals(selected_diseases, preference):
    if not selected_diseases:
        selected_diseases = ["General"]

    df = data[data['Disease'].apply(lambda x: any(d in str(x) for d in selected_diseases))]
    if df.empty:
        df = data[data['Disease'].str.lower() == "general"]
    if df.empty:
        df = data

    if preference == "Vegetarian":
        df = df[df.apply(lambda row: all("chicken" not in str(v).lower() and "fish" not in str(v).lower() and "egg" not in str(v).lower() for v in row.values), axis=1)]
        if df.empty:
            df = data  # fallback if no veg meals found

    return df.sample(min(2, len(df)))

# ---------------------------
# Main Button
# ---------------------------
if st.button("Get Meal Suggestions"):
    results = suggest_meals(selected_diseases, preference)

    if results.empty:
        st.warning("No exact matches found. Showing general healthy options.")
        results = data.sample(2)

    st.subheader(f"🍴 Suggested Meals for {', '.join(selected_diseases)}:")
    for i, (_, meal) in enumerate(results.iterrows(), start=1):
        st.markdown(f"**Option {i}:**")
        for col in meal_cols:
            st.write(f"- {col.replace(' Suggestion','')}: {meal.get(col, 'N/A')}")
        st.write("---")

    # ---------------------------
    # BMI & Macros Visualization
    # ---------------------------
    bmi = round(weight / ((1.7) ** 2), 2)  # assuming height ~1.7m
    st.metric("💪 Your BMI", bmi)

    st.markdown("### 📊 Average Daily Macros (sample view)")
    fig, ax = plt.subplots(figsize=(4, 2.5))  # compact graph size
    macros = {'Carbs': 50, 'Protein': 30, 'Fats': 20}
    ax.bar(macros.keys(), macros.values())
    ax.set_ylabel("%")
    ax.set_title("Macro Distribution")
    st.pyplot(fig)

    # ---------------------------
    # Precautions & Result Time
    # ---------------------------
    st.markdown("### ⚠️ Precautions:")
    st.markdown("""
    - Stay hydrated and avoid processed foods.
    - If you experience discomfort or allergic symptoms, **consult your doctor**.
    - Balance your portions and follow this plan for **consistent results**.
    """)

    st.success("⏱️ Expected visible results in about **3–4 weeks** with regular follow-up and physical activity.")

