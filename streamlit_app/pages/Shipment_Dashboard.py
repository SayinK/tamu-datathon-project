import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Mai Shan Yan Shipments", layout="wide")
st.title("🍗 Ingredients Dashboard")
st.caption("Bars are all displays of monthly frequency per item!")

# --- Load data ---
CSV_PATH = Path("../data/MSY Data - Shipment.csv")  # adjust if needed
df = pd.read_csv(CSV_PATH)

# --- Clean + compute ---
freq_map = {"weekly": 4, "biweekly": 2, "monthly": 1}
freq = df["frequency"].astype(str).str.strip().str.lower().map(freq_map)

df["quantity*shipment"] = df["Quantity per shipment"] * df["Number of shipments"]
df["Total monthly shipment"] = df["quantity*shipment"] * freq

# Optional: handle unexpected frequency values
df["Total monthly shipment"] = df["Total monthly shipment"].fillna(0)

# --- Navigation header (aka different tabs) ---
## tab_monthly, tab_itemized = st.tabs(["📊 Monthly Shipments", "🥣 Itemized Ingredients"])

## with tab_monthly:
# --- Sidebar controls ---
st.sidebar.header("Controls")

# Filters
freq_options = ["All", "Weekly", "Biweekly", "Monthly"]
freq_selected = st.sidebar.selectbox(
    "Frequency:",
    options=freq_options,
)

# Apply filters
filt = df.copy()
if freq_selected != "All":
    filt = filt[filt["frequency"].astype(str).str.lower() == freq_selected.lower()]

# Multi-sort
sortable_map = {
    "Highest Monthly Total": ("Total monthly shipment", False),
    "Lowest Monthly Total": ("Total monthly shipment", True),
}
sort_choices = list(sortable_map.keys())

sort_selected = st.sidebar.selectbox(
    "Sort by (choose order top→bottom):",
    options=sort_choices,
)

# Build sort parameters
if sort_selected:
    sort_col, ascending = sortable_map[sort_selected]
    filt = filt.sort_values(by=sort_col, ascending=ascending)

# How many bars
top_n = st.sidebar.slider(
    "Show top N rows",
    min_value=2,
    max_value=max(3, len(filt)),
    value=min(12, len(filt))
)

plot_df = filt.head(top_n)

sort_dir = "y" if ascending else "-y" # Reverses direction if Lowest Monthly Shipments

# --- Chart (Altair) ---
chart = (
    alt.Chart(plot_df)
    .mark_bar(color="#D41919")   # ← Not a redass TAMU maroon hex
    .encode(
        x=alt.X("Ingredient:N", sort=sort_dir, title="Ingredient"),
        y=alt.Y("Total monthly shipment:Q", title="Total per month"),
        tooltip=[
            alt.Tooltip("Ingredient:N"),
            alt.Tooltip("Unit of shipment:N"),
            alt.Tooltip("Quantity per shipment:Q"),
            alt.Tooltip("Number of shipments:Q"),
            alt.Tooltip("frequency:N"),
            alt.Tooltip("Total monthly shipment:Q", format=",.0f"),
        ],
    )
    .properties(height=420)
)
st.altair_chart(chart, use_container_width=True)
