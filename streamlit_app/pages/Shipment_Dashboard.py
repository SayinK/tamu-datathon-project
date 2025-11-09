import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Expected vs Actual Shipments", layout="wide")
st.title("Expected vs Actual Shipments per Ingredient")
st.caption("Compare expected monthly shipments vs actual shipped quantities interactively.")

CSV_PATH = Path("data/MSY Data - Shipment.csv")
df = pd.read_csv(CSV_PATH)
df['frequency'] = df['frequency'].str.lower().str.strip()

freq_multiplier = {"weekly": 4, "biweekly": 2, "monthly": 1}
df['freq_mult'] = df['frequency'].map(freq_multiplier)
df['expected_monthly_quantity'] = df['Quantity per shipment'] * df['Number of shipments'] * df['freq_mult']
df['actual_quantity'] = df['Quantity per shipment'] * df['Number of shipments']
df['potential_delay'] = df['actual_quantity'] < df['expected_monthly_quantity']

top_n = 14
df = df.nlargest(top_n, 'expected_monthly_quantity')

chart_df = df.melt(
    id_vars=["Ingredient"], 
    value_vars=["expected_monthly_quantity", "actual_quantity"], 
    var_name="Type", 
    value_name="Quantity"
)

alt_chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("Ingredient:N", sort="-y", title="Ingredient"),
        y=alt.Y("Quantity:Q", title="Quantity"),
        color=alt.Color("Type:N", scale=alt.Scale(range=["#D41919", "#4C9AFF"]), title="Shipment Type"),
        tooltip=[
            alt.Tooltip("Ingredient:N"),
            alt.Tooltip("Type:N", title="Shipment Type"),
            alt.Tooltip("Quantity:Q", format=",")
        ]
    )
    .properties(width='container', height=450)
    .interactive()
)

st.altair_chart(alt_chart, use_container_width=True)

delayed = df[df['potential_delay']]
if not delayed.empty:
    st.warning("Ingredients potentially delayed:")
    st.table(delayed[['Ingredient', 'expected_monthly_quantity', 'actual_quantity']])
else:
    st.success("No potential delays detected for the top 14 ingredients.")
