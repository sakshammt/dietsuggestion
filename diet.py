# diet_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

# -----------------------------
# 1️⃣ Load your data
# -----------------------------
@st.cache_data
def load_data():
    data1 = pd.read_csv("https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/detailed_meals_macros_.csv")
    data2 = pd.read_csv("https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/Food_and_Nutrition%20new.csv")
    data3 = pd.read_csv("https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/diet_recommendations_dataset.csv")
    return pd.concat([data1, data2, data3], ignore_index=True)

combined_data_original = load_data()
meal_columns = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]

# -----------------------------
# 2️⃣ Title
# -----------------------------
st.title("🍎 AI Diet Suggestion System")
st.write("Personalized meal plans based on age, weight, activity level, and diseases.")

# -----------------------------
# 3️⃣ Input Section
# -----------------------------
age = st.slider("Age", 1, 100, 25)
weight = st.slider("Weight (kg)", 30, 150, 70)
gender = st.selectbox("Gender", ["Male", "Female"])
activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
preference = st.radio("Diet Preference", ["Vegetarian", "Non-Vegetarian"])

# Split diseases into unique values
all_diseases = set()
for d in combined_data_original['Disease'].dropna().unique():
    if isinstance(d, str):
        for disease in d.replace(";", ",").split(","):
            disease_clean = disease.strip().title()
            if disease_clean not in ["", "Nan", "Unknown", "None"]:
                all_diseases.add(disease_clean)
all_diseases = sorted(list(all_diseases))

# Checkboxes for diseases
st.subheader("Select Diseases (if any):")
selected_diseases = [d for d in all_diseases if st.checkbox(d, value=False)]

# -----------------------------
# 4️⃣ Suggestion Logic
# -----------------------------
def suggest_meals(selected_diseases, preference):
    if not selected_diseases:
        selected_diseases = ["General"]

    # Match diseases
    matching_rows = combined_data_original[
        combined_data_original['Disease'].apply(
            lambda x: any(d in str(x).title() for d in selected_diseases)
        )
    ]

    # Fallback to general
    if matching_rows.empty:
        matching_rows = combined_data_original[
            combined_data_original['Disease'].str.lower().eq("general")
        ]
        if matching_rows.empty:
            matching_rows = combined_data_original

    # Filter Veg preference
    if preference == "Vegetarian":
        matching_rows = matching_rows[
            ~matching_rows.apply(lambda row: any(meat in str(row).lower()
                                                 for meat in ["chicken", "egg", "fish", "mutton", "beef"]), axis=1)
        ]

    return matching_rows.sample(min(2, len(matching_rows)))

# -----------------------------
# 5️⃣ Button + Output
# -----------------------------
if st.button("Get Meal Suggestions"):
    suggestions = suggest_meals(selected_diseases, preference)

    st.subheader(f"🍽️ Suggested Meals for {', '.join(selected_diseases)}:")
    for i in range(min(2, len(suggestions))):
        meal = suggestions.iloc[i]
        st.markdown(f"**Option {i+1}:**")
        for col in meal_columns:
            meal_text = meal[col] if pd.notna(meal[col]) and str(meal[col]).strip().lower() != "nan" else "N/A"
            st.write(f"- {col.replace(' Suggestion','')}: {meal_text}")
        st.write("---")

    # -----------------------------
    # 6️⃣ BMI & Small Graph
    # -----------------------------
    height_m = 1.7  # assumed average height
    bmi = round(weight / (height_m ** 2), 2)
    st.metric("💪 Your BMI", bmi)

    # Small Macro chart
    st.markdown("### 📊 Average Daily Macros")
    fig, ax = plt.subplots(figsize=(4, 2.5))  # compact chart
    macros = {"Carbs": 50, "Protein": 30, "Fats": 20}
    ax.bar(macros.keys(), macros.values(), color=["#4CAF50", "#2196F3", "#FFC107"])
    ax.set_ylabel("%")
    ax.set_title("Macro Distribution")
    st.pyplot(fig)

    # -----------------------------
    # 7️⃣ Precautions
    # -----------------------------
    st.markdown("### ⚠️ Precautions:")
    st.markdown("""
    - Stay hydrated and eat at consistent times.
    - Avoid skipping meals.
    - If you experience any allergy or discomfort, **consult your doctor** immediately.
    """)

    st.success("✅ You can expect visible results within **3–4 weeks** if followed regularly.")

