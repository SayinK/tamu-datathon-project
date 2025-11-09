import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Mai Shan Yan Shipments", layout="wide")
st.title("📦 Monthly Shipments Dashboard")
st.caption("Quantity * Shipments * Frequency")

# --- Load data ---
CSV_PATH = Path("MSY Data - Shipment.csv")  # adjust if needed
df = pd.read_csv(CSV_PATH)

# --- Clean + compute ---
freq_map = {"weekly": 4, "biweekly": 2, "monthly": 1}
freq = df["frequency"].astype(str).str.strip().str.lower().map(freq_map)

df["quantity*shipment"] = df["Quantity per shipment"] * df["Number of shipments"]
df["Total monthly shipment"] = df["quantity*shipment"] * freq

# Optional: handle unexpected frequency values
df["Total monthly shipment"] = df["Total monthly shipment"].fillna(0)

# --- Sidebar controls ---
st.sidebar.header("Controls")
sort_by = st.sidebar.radio("Sort bars by", ["Total monthly shipment (desc)", "Ingredient (A→Z)"])
top_n = st.sidebar.slider("Show top N ingredients", min_value=3, max_value=len(df), value=min(12, len(df)))

if sort_by.startswith("Total"):
    plot_df = df.sort_values("Total monthly shipment", ascending=False).head(top_n)
else:
    plot_df = df.sort_values("Ingredient").head(top_n)

# --- Show table ---
with st.expander("View table data:"):
    st.dataframe(df)

# --- Chart (Altair) ---
chart = (
    alt.Chart(plot_df)
    .mark_bar()
    .encode(
        x=alt.X("Ingredient:N", sort="-y", title="Ingredient"),
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

