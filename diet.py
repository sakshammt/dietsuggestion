# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Diet Coach", layout="wide")

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
# Extract cuisines and diseases
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
plan_days = st.sidebar.slider("Plan duration (days)", min_value=7, max_value=30, value=7, step=1)

st.sidebar.markdown("---")
generate_btn = st.sidebar.button("Generate Meal Plan")
download_btn_placeholder = st.sidebar.empty()

# ---------------------------
# Core filtering
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
        df2['Dietary Preference'] = df2['Dietary Preference'].astype(str).str.lower()
        if diet_type == "Vegetarian":
            df2 = df2[df2['Dietary Preference'].str.contains('veg') & ~df2['Dietary Preference'].str.contains('non')]
        elif diet_type == "Non-Vegetarian":
            df2 = df2[df2['Dietary Preference'].str.contains('non')]
    if cuisine_pref and cuisine_pref != "Any":
        df2 = df2[df2['Dietary Preference'].str.contains(cuisine_pref.lower(), na=False)]
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

# ---------------------------
# Meal plan generation
# ---------------------------
def build_plan(df_available, days, diet_type, allergies):
    allergy_list = parse_allergies(allergies)
    plan_rows = []
    used_indices = set()
    pool_meal_texts = []
    for idx, row in df_available.iterrows():
        combined_meals = " | ".join([str(row.get(c, "")) for c in MEAL_COLS])
        pool_meal_texts.append(combined_meals)
    if not pool_meal_texts:
        pool_meal_texts = ["Mixed salad", "Oatmeal", "Grilled vegetables"]

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
                day_plan[col] = meal_text

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

        plan_rows.append({
            "Day": day + 1,
            **day_plan,
            "Calories": round(day_macros["calories"], 1),
            "Protein": round(day_macros["protein"], 1),
            "Carbs": round(day_macros["carbs"], 1),
            "Fat": round(day_macros["fat"], 1)
        })
    return pd.DataFrame(plan_rows)

# ---------------------------
# Weight projection
# ---------------------------
def estimate_weight_change(current_weight_kg, avg_daily_cal, days, activity_level):
    activity_mult = {"Sedentary": 1.2, "Lightly Active": 1.375, "Moderately Active": 1.55, "Very Active": 1.725}
    bmr = 10 * current_weight_kg + 6.25 * (height_cm) - 5 * age + (5 if gender == "Male" else -161)
    maintenance = bmr * activity_mult.get(activity_level, 1.2)
    daily_deficit = maintenance - avg_daily_cal
    total_deficit = daily_deficit * days
    weight_change_kg = total_deficit / 7700.0
    return weight_change_kg, maintenance

# ---------------------------
# PDF export
# ---------------------------
def create_pdf(plan_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("AI Diet Coach - Personalized Meal Plan", styles["Title"]), Spacer(1, 12)]

    data = [plan_df.columns.tolist()] + plan_df.values.tolist()
    t = Table(data)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8)
    ]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------
# Main action
# ---------------------------
if generate_btn:
    st.title("🍎 AI Diet Coach — Generated Plan")
    with st.spinner("Filtering meals and generating plan..."):
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
        st.dataframe(plan_df, use_container_width=True)

        # Compact Average Daily Macros Chart
        avg_cal = plan_df["Calories"].mean()
        avg_prot = plan_df["Protein"].mean()
        avg_carbs = plan_df["Carbs"].mean()
        avg_fat = plan_df["Fat"].mean()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Average Daily Macros")
            fig, ax = plt.subplots(figsize=(4.5, 2.5))
            nutrients = ['Calories','Protein','Carbs','Fat']
            vals = [avg_cal, avg_prot, avg_carbs, avg_fat]
            ax.bar(nutrients, vals, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
            ax.set_ylabel("Amount")
            st.pyplot(fig)

        with col2:
            bmi = weight / ((height_cm/100)**2)
            change_kg, maintenance = estimate_weight_change(weight, avg_cal, plan_days, activity)
            projected = weight + change_kg
            days_range = np.arange(0, plan_days+1)
            weight_trend = weight + (change_kg/plan_days) * days_range
            fig2, ax2 = plt.subplots(figsize=(4.5, 2.5))
            ax2.plot(days_range, weight_trend, marker='o')
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Weight (kg)")
            ax2.set_title("Projected Weight Trend (Estimate)")
            st.pyplot(fig2)

        # Precautions & Download
        st.markdown("### ⚠️ Precautions & Tips")
        st.write("- If you experience allergic reactions, stop immediately and consult a physician.")
        st.write("- Follow the plan consistently for 2–4 weeks to observe changes.")
        st.write(f"- Tip: {random.choice(['Stay hydrated','Add fiber-rich veggies','Prefer whole grains','Avoid processed snacks'])}")

        # CSV + PDF download
        csv_buf = io.StringIO()
        plan_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode()
        pdf_bytes = create_pdf(plan_df)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_btn_placeholder.download_button("⬇️ Download Meal Plan (CSV)", data=csv_bytes, file_name=f"meal_plan_{now}.csv", mime="text/csv")
        st.download_button("📄 Download Meal Plan (PDF)", data=pdf_bytes, file_name=f"meal_plan_{now}.pdf", mime="application/pdf")

        st.success("Meal plan generated — download or print this page for offline use.")
else:
    st.title("🍎 AI Diet Coach")
    st.markdown("""
    Welcome — configure user inputs on the left sidebar and click **Generate Meal Plan**.  
    This app suggests diet plans tailored to your health conditions, dietary preferences, and cuisine style.  
    It provides macros, BMI tracking, and downloadable PDF/CSV for easy use.
    """)
