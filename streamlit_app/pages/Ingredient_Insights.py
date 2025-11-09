import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from thefuzz import process

st.set_page_config(page_title="Ingredient Insights", layout="wide")
st.title("Ingredient Usage Insights")

# --- PARAMETERS ---
dataset_folder = "data"
ingredients_path = "data/MSY Data - Ingredient.csv"  # adjust as needed
MONTH_ORDER = ["May", "June", "July", "August", "September", "October"]

# Ingredients that are counts
count_ingredients = ['Egg(count)', 'Ramen (count)', 'Chicken Wings (pcs)', 'chicken thigh (pcs)', 'White onion']

@st.cache_data
def load_ingredient_totals():
    # --- LOAD INGREDIENTS CSV ---
    ingredients = pd.read_csv(ingredients_path)
    ingredients.columns = [c.strip() for c in ingredients.columns]

    # Convert grams to lbs for non-count ingredients
    for col in ingredients.columns[1:]:
        if col not in count_ingredients:
            ingredients[col] = pd.to_numeric(ingredients[col], errors='coerce') * 0.00220462
        else:
            ingredients[col] = pd.to_numeric(ingredients[col], errors='coerce')

    ingredient_names = ingredients.columns[1:]  # skip "Item name"
    totals = pd.DataFrame(0, index=ingredient_names, columns=MONTH_ORDER, dtype=float)

    # --- PROCESS MONTHLY SALES FILES ---
    for file in sorted(os.listdir(dataset_folder)):
        if not file.endswith(".xlsx"):
            continue
        month_name = file.split("_")[0]
        file_path = os.path.join(dataset_folder, file)
        sheet = "data 2" if "October" in file else "data 3"

        try:
            sales_df = pd.read_excel(file_path, sheet_name=sheet)
        except:
            continue

        sales_df.columns = [c.strip() for c in sales_df.columns]

        # Identify columns
        item_col = next((c for c in sales_df.columns if 'item' in c.lower() and 'name' in c.lower()), None)
        count_col = next((c for c in sales_df.columns if 'count' in c.lower()), None)
        if item_col is None or count_col is None:
            continue

        sales_df[item_col] = sales_df[item_col].astype(str).str.strip()
        sales_df[count_col] = pd.to_numeric(sales_df[count_col], errors='coerce').fillna(0)

        # --- COMPUTE TOTAL USAGE ---
        for _, row in sales_df.iterrows():
            item_name = row[item_col].lower()
            cnt = row[count_col]

            # Normalize item names for fuzzy matching
            if "fried chicken" in item_name:
                base_name = "Fried Wings"
            elif "cutlet" in item_name:
                base_name = "Chicken Cutlet"
            else:
                base_name = ''.join(c for c in item_name if c.isalpha() or c.isspace()).strip()

            # --- FUZZY MATCH TO CSV ITEM NAMES ---
            match = process.extractOne(base_name, ingredients['Item name'].str.lower())
            if match is None:
                continue
            matched_item, score, *_ = match
            ing_rows = ingredients[ingredients['Item name'].str.lower() == matched_item]
            if ing_rows.empty:
                continue

            for ing in ingredient_names:
                per_serving = pd.to_numeric(ing_rows.iloc[0][ing], errors='coerce')
                if pd.notna(per_serving):
                    totals.at[ing, month_name] += per_serving * cnt
                    # Removed debug logging

    return totals

# --- LOAD DATA ---
ingredient_totals = load_ingredient_totals()

# --- STREAMLIT INTERFACE ---
ingredient_selected = st.selectbox("Select ingredient to view usage", sorted(ingredient_totals.index))

values = ingredient_totals.loc[ingredient_selected, MONTH_ORDER].fillna(0)
grand_total = values.sum()

unit_label = "Count" if ingredient_selected in count_ingredients else "lbs"
st.markdown(f"**Grand Total {ingredient_selected}: {grand_total:.2f} {unit_label}**")

# --- PLOTLY BAR CHART ---
fig = go.Figure(go.Bar(
    x=MONTH_ORDER,
    y=values,
    text=[f"{v:.1f}" for v in values],
    textposition="auto",
    marker_color='darkred'
))
fig.update_layout(
    title=f"{ingredient_selected} Usage by Month",
    xaxis_title="Month",
    yaxis_title=unit_label,
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# --- RAW DATA EXPANDER ---
with st.expander("Show full ingredient usage table"):
    st.dataframe(ingredient_totals)

