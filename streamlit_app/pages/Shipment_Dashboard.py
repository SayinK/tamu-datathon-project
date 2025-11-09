import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Mai Shan Yan Shipments", layout="wide")
st.title("Ingredients Shipment Dashboard")
st.caption("Bars are all displays of monthly frequency per item!")

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "MSY Data - Shipment.csv"   # exact CSV filename
XLSX_PATH = DATA_DIR / "MSY Data - Shipment.xlsx" # fallback if it's Excel

if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
elif XLSX_PATH.exists():
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
else:
    st.error(f"Couldn’t find the data file.\nLooked for:\n- {CSV_PATH}\n- {XLSX_PATH}")
    st.stop()

freq_map = {"weekly": 4, "biweekly": 2, "monthly": 1}
freq = df["frequency"].astype(str).str.strip().str.lower().map(freq_map)

df["quantityshipment"] = df["Quantity per shipment"] * df["Number of shipments"]
df["Total monthly shipment"] = df["quantityshipment"] * freq

df["Total monthly shipment"] = df["Total monthly shipment"].fillna(0)

tab_monthly= st.tabs(["📊 Monthly Shipments"])

freq_options = ["All", "Weekly", "Biweekly", "Monthly"]
freq_selected = st.sidebar.selectbox(
    "frequency:",
    options=freq_options,
)

filt = df.copy()
if freq_selected != "All":
    filt = filt[filt["frequency"].astype(str).str.lower() == freq_selected.lower()]

sortable_map = {
    "Highest Monthly Total": ("Total monthly shipment", False),
    "Lowest Monthly Total": ("Total monthly shipment", True),
}
sort_choices = list(sortable_map.keys())

sort_selected = st.sidebar.selectbox(
    "Sort by (choose order top→bottom):",
    options=sort_choices,
)
if sort_selected:
    sort_col, ascending = sortable_map[sort_selected]
    filt = filt.sort_values(by=sort_col, ascending=ascending)

top_n = st.sidebar.slider(
    "Show top N rows",
    min_value=1,
    max_value=max(2, len(filt)),
    value=min(12, len(filt))
)

plot_df = filt.head(top_n)

sort_dir = "y" if ascending else "-y" # Reverses direction if Lowest Monthly Shipments

chart = (
    alt.Chart(plot_df)
    .mark_bar(color="#D41919")   # ← Not a redass TAMU maroon hex
    .encode(
        x=alt.X(
            "Ingredient:N",
            sort=sort_dir,
            title="Ingredient",
            axis=alt.Axis(labelAngle=0)   # <--- key line!
        ),
        y=alt.Y("Total monthly shipment:Q", title="Total Per Month"),
        tooltip=[
            alt.Tooltip("Ingredient:N"),
            alt.Tooltip("Unit of shipment:N", title="Unit of Shipment"),
            alt.Tooltip("Quantity per shipment:Q", title="Quantity per Shipment"),
            alt.Tooltip("Number of shipments:Q", title="Number of Shipments"),
            alt.Tooltip("frequency:N", title="Order Frequency"),
            alt.Tooltip("Total monthly shipment:Q", title="Total Per Month",format=",.0f"),
        ],
    )
    .properties(height=420)
)
st.altair_chart(chart, width='stretch')