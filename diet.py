# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Diet Coach", layout="wide")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data():
    urls = {
        "data1": "https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/Food_and_Nutrition%20new.csv",
        "data2": "https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/detailed_meals_macros_.csv",
        "data3": "https://raw.githubusercontent.com/sakshammt/meal-data/refs/heads/main/diet_recommendations_dataset.csv",
    }
    dfs = []
    for u in urls.values():
        try:
            df = pd.read_csv(u)
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading data: {e}")
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
# Sidebar
# ---------------------------
st.sidebar.title("🔧 User Inputs")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
height_cm = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0)
activity = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
diet_type = st.sidebar.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian"])
cuisine_pref = st.sidebar.selectbox("Cuisine Preference", ["Any"] + cuisines)
selected_diseases = st.sidebar.multiselect("Diseases", options=diseases_list)
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

def build_plan(df_available, days, allergies):
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
                day_plan[col] = meal_text

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
            "Day": day + 1,
            **day_plan,
            "Calories": round(day_macros["calories"], 1),
            "Protein (g)": round(day_macros["protein"], 1),
            "Carbs (g)": round(day_macros["carbs"], 1),
            "Fat (g)": round(day_macros["fat"], 1)
        })
    return pd.DataFrame(plan_rows)

def make_pdf_table(df):
    """Generate a structured PDF and return it as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("<b>AI Diet Coach — Personalized Meal Plan</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data, repeatRows=1)
    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ])
    table.setStyle(table_style)
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------
# Generate
# ---------------------------
if generate_btn:
    st.title("🍽️ AI Diet Coach — Generated Plan")

    with st.spinner("Generating your meal plan..."):
        df = data.copy()
        df = filter_by_diseases(df, selected_diseases)
        df = filter_by_diet_and_cuisine(df, diet_type, cuisine_pref)
        df = apply_allergy_filter(df, parse_allergies(allergy_text))

        if df.empty:
            df = data.copy()

        plan_df = build_plan(df, plan_days, allergy_text)

        st.subheader(f"Meal Plan — {plan_days} Days")
        st.dataframe(plan_df, use_container_width=True)

        # Compact graph
        col1, col2 = st.columns(2)
        avg = plan_df[["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]].mean()

        with col1:
            st.markdown("### 📊 Average Daily Macros")
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            ax.bar(avg.index, avg.values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
            st.pyplot(fig)

        # PDF and CSV downloads
        csv_buf = io.StringIO()
        plan_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode()
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_bytes = make_pdf_table(plan_df)

        st.download_button("⬇️ Download Meal Plan (CSV)", data=csv_bytes, file_name=f"meal_plan_{now}.csv", mime="text/csv")
        st.download_button("📄 Download Meal Plan (PDF)", data=pdf_bytes, file_name=f"meal_plan_{now}.pdf", mime="application/pdf")

        st.success("✅ Meal plan ready! Choose your preferred format to download.")
else:
    st.title("🍎 AI Diet Coach")
    st.markdown("""
    Configure your info in the sidebar and click **Generate Meal Plan**.
    
    You’ll get:
    - Personalized meals  
    - Compact macro chart  
    - Downloadable CSV **and** PDF table  
    - Suitable for printing & health tracking  
    """)
