# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="AI Diet Coach", layout="wide")

# ---------------------------
# Load Data from GitHub
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data():
    urls = {
        "data1": "https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/Food_and_Nutrition%20new.csv",
        "data2": "https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/detailed_meals_macros_.csv",
        "data3": "https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/diet_recommendations_dataset.csv",
    }
    dfs = []
    for k, u in urls.items():
        try:
            df = pd.read_csv(u)
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {k} from GitHub: {e}")
    if not dfs:
        st.stop()
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined.columns = [c.strip() for c in combined.columns]
    return combined

data = load_data()

MEAL_COLS = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
MACRO_MAP = {
    "Breakfast Suggestion": ["Breakfast Calories", "Breakfast Protein", "Breakfast Carbohydrates", "Breakfast Fats"],
    "Lunch Suggestion": ["Lunch Calories", "Lunch Protein", "Lunch Carbohydrates", "Lunch Fats"],
    "Dinner Suggestion": ["Dinner Calories", "Dinner Protein.1", "Dinner Carbohydrates.1", "Dinner Fats"],
    "Snack Suggestion": ["Snacks Calories", "Snacks Protein", "Snacks Carbohydrates", "Snacks Fats"]
}

# ---------------------------
# Extract Cuisines & Diseases
# ---------------------------
def get_unique_sorted(col):
    if col in data.columns:
        vals = data[col].dropna().astype(str).str.strip().unique().tolist()
        vals = sorted([v for v in vals if v and v.lower() not in ["nan", "none"]])
        return vals
    return []

cuisines = get_unique_sorted("Dietary Preference")

disease_set = set()
if "Disease" in data.columns:
    for v in data['Disease'].dropna().astype(str):
        for part in str(v).replace(";", ",").split(","):
            s = part.strip().title()
            if s and s.lower() not in ["nan", "none", "unknown"]:
                disease_set.add(s)
diseases_list = sorted(disease_set)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.title("🔧 User Inputs")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
height_cm = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0)
activity = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
diet_type = st.sidebar.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian"])
cuisine_pref = st.sidebar.selectbox("Cuisine Preference", ["Any"] + cuisines)
selected_diseases = st.sidebar.multiselect("Diseases (select all that apply)", options=diseases_list)
allergy_text = st.sidebar.text_input("Allergies (comma-separated)", value="")
plan_days = st.sidebar.slider("Plan duration (days)", min_value=7, max_value=30, value=7, step=1)

st.sidebar.markdown("---")
generate_btn = st.sidebar.button("Generate Meal Plan")
download_btn_placeholder = st.sidebar.empty()

# ---------------------------
# Helper Functions
# ---------------------------
def parse_allergies(text):
    return [a.strip().lower() for a in text.split(",") if a.strip()]

def meal_contains_any_keywords(meal_text, keywords):
    txt = str(meal_text).lower()
    return any(k in txt for k in keywords)

def row_is_safe_for_allergies(row, allergy_list):
    if not allergy_list:
        return True
    for col in MEAL_COLS:
        if col in row and meal_contains_any_keywords(row[col], allergy_list):
            return False
    return True

def filter_by_diet_and_cuisine(df, diet_type, cuisine_pref):
    df2 = df.copy()
    if 'Dietary Preference' in df2.columns:
        if diet_type == "Vegetarian":
            df2 = df2[df2['Dietary Preference'].str.contains('veg', case=False, na=False)]
    if cuisine_pref and cuisine_pref != "Any":
        if 'Dietary Preference' in df2.columns:
            df2 = df2[df2['Dietary Preference'].str.contains(cuisine_pref, case=False, na=False)]
    return df2

def filter_by_diseases(df, selected_diseases):
    if not selected_diseases:
        return df
    mask = df['Disease'].fillna("").astype(str).apply(lambda x: any(d.lower() in x.lower() for d in selected_diseases))
    return df[mask]

def apply_allergy_filter(df, allergy_list):
    if not allergy_list:
        return df
    mask = df.apply(lambda row: row_is_safe_for_allergies(row, allergy_list), axis=1)
    return df[mask]

def substitute_meal_if_allergy(meal_text, allergy_list, pool_texts):
    if not allergy_list:
        return meal_text, False
    if not meal_contains_any_keywords(meal_text, allergy_list):
        return meal_text, False
    for candidate in pool_texts:
        if not meal_contains_any_keywords(candidate, allergy_list):
            return candidate, True
    return meal_text, False

# ---------------------------
# Build Plan
# ---------------------------
def build_plan(df_available, days, diet_type, allergies):
    allergy_list = parse_allergies(allergies)
    plan_rows = []
    used_indices = set()
    pool_meal_texts = ["Mixed salad", "Oatmeal", "Grilled veggies", "Fruit bowl"]

    for day in range(days):
        choices = [i for i in df_available.index if i not in used_indices]
        if not choices:
            used_indices = set()
            choices = list(df_available.index)
        if not choices:
            sample_row = None
        else:
            idx_choice = random.choice(choices)
            used_indices.add(idx_choice)
            sample_row = df_available.loc[idx_choice]

        day_plan = {}
        if sample_row is None:
            for col in MEAL_COLS:
                day_plan[col] = "General healthy meal"
        else:
            for col in MEAL_COLS:
                meal_text = sample_row.get(col, "N/A")
                sub, _ = substitute_meal_if_allergy(meal_text, allergy_list, pool_meal_texts)
                day_plan[col] = sub

        day_macros = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        for col in MEAL_COLS:
            macros = MACRO_MAP.get(col, [])
            if sample_row is not None and all((m in sample_row) for m in macros):
                try:
                    day_macros["calories"] += float(sample_row[macros[0]] or 0)
                    day_macros["protein"] += float(sample_row[macros[1]] or 0)
                    day_macros["carbs"] += float(sample_row[macros[2]] or 0)
                    day_macros["fat"] += float(sample_row[macros[3]] or 0)
                except Exception:
                    pass

        plan_rows.append({
            "day": day + 1,
            **day_plan,
            "calories": round(day_macros["calories"], 1),
            "protein": round(day_macros["protein"], 1),
            "carbs": round(day_macros["carbs"], 1),
            "fat": round(day_macros["fat"], 1)
        })
    return pd.DataFrame(plan_rows)

# ---------------------------
# Weight Projection
# ---------------------------
def estimate_weight_change(current_weight_kg, avg_daily_cal, days, activity_level):
    activity_mult = {"Sedentary": 1.2, "Lightly Active": 1.375, "Moderately Active": 1.55, "Very Active": 1.725}
    bmr = 10 * current_weight_kg + 6.25 * height_cm - 5 * age + (5 if gender == "Male" else -161)
    maintenance = bmr * activity_mult.get(activity_level, 1.2)
    daily_deficit = maintenance - avg_daily_cal
    total_deficit = daily_deficit * days
    weight_change_kg = total_deficit / 7700.0
    return weight_change_kg, maintenance

# ---------------------------
# Main Logic
# ---------------------------
if generate_btn:
    st.title("🍎 AI Diet Coach — Generated Plan")
    with st.spinner("Generating your custom meal plan..."):
        allergy_list = parse_allergies(allergy_text)
        df = data.copy()

        df = filter_by_diseases(df, selected_diseases)
        df = filter_by_diet_and_cuisine(df, diet_type, cuisine_pref)
        df = apply_allergy_filter(df, allergy_list)

        if df.empty:
            df = data.copy()
            df = apply_allergy_filter(df, allergy_list)

        plan_df = build_plan(df, plan_days, diet_type, allergy_text)

        st.subheader(f"Meal Plan — {plan_days} days")
        st.dataframe(plan_df.style.format({
            "calories": "{:.0f}", "protein": "{:.1f}", "carbs": "{:.1f}", "fat": "{:.1f}"
        }), use_container_width=True)

        # ---------------------------
        # Compact Side-by-Side Graphs
        # ---------------------------
        avg_cal = plan_df["calories"].mean()
        avg_prot = plan_df["protein"].mean()
        avg_carbs = plan_df["carbs"].mean()
        avg_fat = plan_df["fat"].mean()
        bmi = weight / ((height_cm/100)**2)
        change_kg, maintenance = estimate_weight_change(weight, avg_cal, plan_days, activity)
        projected = weight + change_kg
        days_range = np.arange(0, plan_days+1)
        weight_trend = weight + (change_kg/plan_days) * days_range

        col_macro, col_weight = st.columns(2)
        with col_macro:
            st.markdown("### 📊 Average Daily Macros")
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            nutrients = ['Calories', 'Protein', 'Carbs', 'Fat']
            vals = [avg_cal, avg_prot, avg_carbs, avg_fat]
            ax.bar(nutrients, vals, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
            ax.set_ylabel("Amount")
            st.pyplot(fig)

        with col_weight:
            st.markdown("### ⚖️ Projected Weight Trend")
            fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))
            ax2.plot(days_range, weight_trend, marker='o')
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Weight (kg)")
            st.pyplot(fig2)

        # ---------------------------
        # BMI & Summary
        # ---------------------------
        st.markdown("### 🧍 BMI & Maintenance Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**BMI:** {bmi:.1f}")
            if bmi < 18.5:
                st.info("Underweight")
            elif bmi < 25:
                st.success("Normal")
            elif bmi < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")
        with col2:
            st.write(f"Estimated maintenance calories: {maintenance:.0f} kcal/day")
            st.write(f"Expected weight change in {plan_days} days: {change_kg:+.2f} kg → projected weight: {projected:.1f} kg")

        # ---------------------------
        # Precautions & Download
        # ---------------------------
        st.markdown("### ⚠️ Precautions & Tips")
        st.write("- If you experience any allergic reaction, stop immediately and consult a doctor.")
        st.write("- Follow the plan consistently for 2–4 weeks for visible results.")
        st.write(f"- Tip: {random.choice(['Stay hydrated', 'Include fiber-rich foods', 'Avoid sugary snacks', 'Prefer homemade meals'])}")

        csv_buf = io.StringIO()
        plan_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode()
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meal_plan_{now}.csv"
        download_btn_placeholder.download_button(label="⬇️ Download Meal Plan (CSV)", data=csv_bytes, file_name=filename, mime="text/csv")

        st.success("Meal plan generated successfully!")

else:
    st.title("🍎 AI Diet Coach")
    st.markdown("""
    Welcome! Configure your details in the sidebar and click **Generate Meal Plan**.
    This app provides balanced diet suggestions based on your:
    - Age, gender, activity level, and weight goals  
    - Diseases, allergies, and cuisine preferences  
    - Veg/Non-veg diet type  
    Includes **BMI tracking**, **macronutrient graphs**, and **downloadable meal plan**!
    """)
    st.info("💡 Tip: Enter allergy items like `peanut, shellfish` to exclude them from meal suggestions.")
