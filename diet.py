# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="AI Diet + Eco Coach", layout="wide")

# ---------------------------
# Helper: load datasets
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
    # Normalize column names trimming whitespace
    combined.columns = [c.strip() for c in combined.columns]
    return combined

data = load_data()

# Standard meal & macro columns mapping (adapt to your CSV names)
MEAL_COLS = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
MACRO_MAP = {
    "Breakfast Suggestion": ["Breakfast Calories", "Breakfast Protein", "Breakfast Carbohydrates", "Breakfast Fats"],
    "Lunch Suggestion": ["Lunch Calories", "Lunch Protein", "Lunch Carbohydrates", "Lunch Fats"],
    "Dinner Suggestion": ["Dinner Calories", "Dinner Protein.1", "Dinner Carbohydrates.1", "Dinner Fats"],
    "Snack Suggestion": ["Snacks Calories", "Snacks Protein", "Snacks Carbohydrates", "Snacks Fats"]
}

# ---------------------------
# Simple environmental footprint heuristic
# ---------------------------
# Score per ingredient keyword (lower is better)
ENV_FOOTPRINT = {
    "beef": 5.0,
    "mutton": 5.0,
    "lamb": 4.5,
    "pork": 4.0,
    "chicken": 2.5,
    "fish": 2.5,
    "egg": 2.0,
    "cheese": 3.5,
    "milk": 2.0,
    "lentil": 0.5,
    "beans": 0.5,
    "tofu": 0.6,
    "tempeh": 0.6,
    "vegetable": 0.2,
    "salad": 0.2,
    "rice": 1.0,
    "quinoa": 0.8,
    "pasta": 1.0,
    "potato": 0.3,
    "nuts": 1.2,
    "fruit": 0.2,
    "vegetarian": 0.3,
    "vegan": 0.2
}
def meal_env_score(meal_text):
    text = str(meal_text).lower()
    score = 0.0
    for k, v in ENV_FOOTPRINT.items():
        if k in text:
            score += v
    # If no keywords matched, give a mild default
    if score == 0:
        score = 0.8
    return round(score, 2)

# ---------------------------
# Extract cuisines and diseases
# ---------------------------
def get_unique_sorted(col):
    if col in data.columns:
        vals = data[col].dropna().astype(str).str.strip().unique().tolist()
        vals = sorted([v for v in vals if v and v.lower() not in ["nan", "none"]])
        return vals
    return []

cuisines = get_unique_sorted("Dietary Preference")
# Build disease list by splitting combined cells
disease_set = set()
if "Disease" in data.columns:
    for v in data['Disease'].dropna().astype(str):
        for part in str(v).replace(";", ",").split(","):
            s = part.strip().title()
            if s and s.lower() not in ["nan", "none", "unknown"]:
                disease_set.add(s)
diseases_list = sorted(disease_set)

# ---------------------------
# UI: sidebar inputs
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
allergy_text = st.sidebar.text_input("Allergies (comma-separated keywords)", value="")
env_pref_lowcarbon = st.sidebar.checkbox("Prioritise low-carbon meals", value=False)
env_pref_local = st.sidebar.checkbox("Prefer local/seasonal", value=False)  # placeholder - would require data
plan_days = st.sidebar.slider("Plan duration (days)", min_value=7, max_value=30, value=7, step=1)

# Buttons
st.sidebar.markdown("---")
generate_btn = st.sidebar.button("Generate Meal Plan")
download_btn_placeholder = st.sidebar.empty()

# ---------------------------
# Core filtering and substitution functions
# ---------------------------
def parse_allergies(text):
    return [a.strip().lower() for a in text.split(",") if a.strip()]

def meal_contains_any_keywords(meal_text, keywords):
    txt = str(meal_text).lower()
    return any(k in txt for k in keywords)

def row_is_safe_for_allergies(row, allergy_list):
    # Check across all meal suggestion columns
    if not allergy_list:
        return True
    for col in MEAL_COLS:
        if col in row and meal_contains_any_keywords(row[col], allergy_list):
            return False
    return True

def filter_by_diet_and_cuisine(df, diet_type, cuisine_pref):
    df2 = df.copy()
    # Diet filter: use 'Dietary Preference' column if available
    if 'Dietary Preference' in df2.columns:
        if diet_type == "Vegetarian":
            df2 = df2[df2['Dietary Preference'].str.contains('veg', case=False, na=False)]
        else:
            # for Non-Veg don't filter aggressively, keep all
            pass
    # cuisine
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

def apply_env_rank(df, lowcarbon):
    # Add an env score column computed from meal texts
    df = df.copy()
    df['_env_score'] = 0.0
    for col in MEAL_COLS:
        if col in df.columns:
            df['_env_score'] += df[col].fillna("").apply(meal_env_score)
    # lower is better, so sort ascending if lowcarbon True
    if lowcarbon:
        df = df.sort_values('_env_score', ascending=True)
    return df

def substitute_meal_if_allergy(meal_text, allergy_list, pool_texts):
    # If meal contains allergy, try to find alternative in pool_texts not containing allergy
    if not allergy_list:
        return meal_text, False
    if not meal_contains_any_keywords(meal_text, allergy_list):
        return meal_text, False
    # search for substitute
    for candidate in pool_texts:
        if not meal_contains_any_keywords(candidate, allergy_list):
            return candidate, True
    return meal_text, False

# ---------------------------
# Meal plan generation
# ---------------------------
def build_plan(df_available, days, diet_type, allergies):
    allergy_list = parse_allergies(allergies)
    plan_rows = []
    used_indices = set()
    pool_meal_texts = []
    # collect candidate meal lines for substitution
    for idx, row in df_available.iterrows():
        combined_meals = " | ".join([str(row.get(c, "")) for c in MEAL_COLS])
        pool_meal_texts.append(combined_meals)
    if not pool_meal_texts:
        pool_meal_texts = ["Mixed salad", "Oatmeal", "Grilled vegetables"]

    for day in range(days):
        # pick a row not used yet (to increase variety)
        choices = [i for i in df_available.index if i not in used_indices]
        if not choices:
            used_indices = set()
            choices = list(df_available.index)
        if not choices:
            # fallback random simple plan
            sample_row = None
        else:
            idx_choice = random.choice(choices)
            used_indices.add(idx_choice)
            sample_row = df_available.loc[idx_choice]

        day_plan = {}
        if sample_row is None:
            # default generic
            for col in MEAL_COLS:
                day_plan[col] = "General healthy meal"
        else:
            # for each meal, get text and substitute if necessary
            for col in MEAL_COLS:
                meal_text = sample_row.get(col, "N/A")
                sub, substituted = substitute_meal_if_allergy(meal_text, allergy_list, pool_meal_texts)
                # if substituted returns combined string, try to split and pick portion; for simplicity use candidate as snack
                day_plan[col] = sub

        # compute macros for the day by summing macros if available
        day_macros = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        for col in MEAL_COLS:
            macros = MACRO_MAP.get(col, [])
            if sample_row is not None and all((m in sample_row) for m in macros):
                try:
                    day_macros["calories"] += float(sample_row[macros[0]] if pd.notna(sample_row[macros[0]]) else 0)
                    day_macros["protein"] += float(sample_row[macros[1]] if pd.notna(sample_row[macros[1]]) else 0)
                    day_macros["carbs"] += float(sample_row[macros[2]] if pd.notna(sample_row[macros[2]]) else 0)
                    day_macros["fat"] += float(sample_row[macros[3]] if pd.notna(sample_row[macros[3]]) else 0)
                except Exception:
                    pass
        # calculate environmental score for day by averaging meal env scores
        day_env = np.mean([meal_env_score(day_plan[col]) for col in MEAL_COLS])
        plan_rows.append({
            "day": day + 1,
            **day_plan,
            "calories": round(day_macros["calories"], 1),
            "protein": round(day_macros["protein"], 1),
            "carbs": round(day_macros["carbs"], 1),
            "fat": round(day_macros["fat"], 1),
            "env_score": round(day_env, 2)
        })
    plan_df = pd.DataFrame(plan_rows)
    return plan_df

# ---------------------------
# Weight projection helper (very rough)
# ---------------------------
def estimate_weight_change(current_weight_kg, avg_daily_cal, days, activity_level):
    # Rough estimate: 7700 kcal per kg of fat
    # Adjust maintenance calories by activity multiplier (very rough)
    activity_mult = {"Sedentary": 1.2, "Lightly Active": 1.375, "Moderately Active": 1.55, "Very Active": 1.725}
    bmr = 10 * current_weight_kg + 6.25 * (height_cm) - 5 * age + (5 if gender == "Male" else -161)
    maintenance = bmr * activity_mult.get(activity_level, 1.2)
    daily_deficit = maintenance - avg_daily_cal
    total_deficit = daily_deficit * days
    weight_change_kg = total_deficit / 7700.0
    return weight_change_kg, maintenance

# ---------------------------
# Main action: generate plan
# ---------------------------
if generate_btn:
    st.title("🍃 AI Diet & Eco Coach — Generated Plan")
    with st.spinner("Filtering meals and generating plan..."):
        allergy_list = parse_allergies(allergy_text)
        df = data.copy()

        # Filter pipeline
        df = filter_by_diseases(df, selected_diseases)
        df = filter_by_diet_and_cuisine(df, diet_type, cuisine_pref)
        df = apply_allergy_filter(df, allergy_list)
        df = apply_env_rank(df, env_pref_lowcarbon)

        # If still empty, fallback to full data (but keep allergy filter)
        if df.empty:
            df = data.copy()
            df = apply_allergy_filter(df, allergy_list)

        # Generate plan
        plan_df = build_plan(df, plan_days, diet_type, allergy_text)

        # Display plan table
        st.subheader(f"Meal Plan — {plan_days} days")
        st.dataframe(plan_df.style.format({
            "calories": "{:.0f}", "protein": "{:.1f}", "carbs": "{:.1f}", "fat": "{:.1f}", "env_score": "{:.2f}"
        }), use_container_width=True)

        # Show aggregate macros chart
        avg_cal = plan_df["calories"].mean()
        avg_prot = plan_df["protein"].mean()
        avg_carbs = plan_df["carbs"].mean()
        avg_fat = plan_df["fat"].mean()
        st.markdown("### 📊 Average daily macros")
        fig, ax = plt.subplots(figsize=(6,3))
        nutrients = ['Calories','Protein','Carbs','Fat']
        vals = [avg_cal, avg_prot, avg_carbs, avg_fat]
        ax.bar(nutrients, vals, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        ax.set_ylabel("Amount")
        st.pyplot(fig)

        # Environmental summary
        total_env = plan_df["env_score"].sum()
        st.markdown(f"### 🌍 Environmental impact (lower is better)")
        st.metric("Average daily env score", f"{plan_df['env_score'].mean():.2f}")
        st.metric("Total plan env score", f"{total_env:.2f}")

        # BMI and weight projection
        bmi = weight / ((height_cm/100)**2)
        st.markdown("### ⚖️ BMI & Weight Projection")
        col1, col2 = st.columns([1,2])
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
            change_kg, maintenance = estimate_weight_change(weight, avg_cal, plan_days, activity)
            projected = weight + change_kg
            st.write(f"Estimated maintenance calories: {maintenance:.0f} kcal/day")
            st.write(f"Estimated weight change in {plan_days} days: {change_kg:+.2f} kg  → projected weight: {projected:.1f} kg")

            # Show simple line plot of projected weight trend
            days_range = np.arange(0, plan_days+1)
            weight_trend = weight + (change_kg/plan_days) * days_range
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.plot(days_range, weight_trend, marker='o')
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Weight (kg)")
            ax2.set_title("Projected weight trend (very rough)")
            st.pyplot(fig2)

        # Precautions and tips
        st.markdown("### ⚠️ Precautions & Tips")
        st.write("- If you experience allergic reactions or severe symptoms, stop immediately and consult a physician.")
        st.write("- Follow the plan consistently for at least 2–4 weeks to observe changes.")
        st.write(f"- Tip: {random.choice(['Stay hydrated','Include fiber-rich veggies','Prefer whole grains','Avoid processed snacks'])}")

        # ---------------------------
        # Downloadable CSV / printable
        # ---------------------------
        csv_buf = io.StringIO()
        plan_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode()
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meal_plan_{now}.csv"
        download_btn_placeholder.download_button(label="⬇️ Download meal plan (CSV)", data=csv_bytes, file_name=filename, mime="text/csv")

        st.success("Meal plan generated — download or print this page for offline use.")

else:
    st.title("🍃 AI Diet & Eco Coach")
    st.markdown(
        """
        Welcome — configure user inputs on the left sidebar and click **Generate Meal Plan**.
        This app suggests diet plans tailored to health (diseases/allergies), diet preferences, cuisine,
        and environmental considerations (low-carbon meals). It also provides macros, BMI and a printable CSV.
        """
    )
    st.info("Tip: Use the 'Allergies' field to enter ingredients to avoid (comma-separated), e.g. 'peanut, shellfish'.")

# End of app.py
