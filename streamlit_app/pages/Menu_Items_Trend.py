import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Menu Item Trends", layout="wide")
st.title("Menu Item Popularity Trends")

dataset_folder = "data"
MONTH_ORDER = ["May", "June", "July", "August", "September", "October"]

@st.cache_data
def load_monthly_sales(dataset_folder):
    monthly_sales = {}

    for file in sorted(os.listdir(dataset_folder)):
        if not file.endswith(".xlsx"):
            continue
        file_path = os.path.join(dataset_folder, file)
        sheet = "data 2" if "October" in file else "data 3"
        try:
            sales_df = pd.read_excel(file_path, sheet_name=sheet)
        except:
            continue

        sales_df.columns = [c.strip() for c in sales_df.columns]
        item_col = next((c for c in sales_df.columns if 'item' in c.lower() and 'name' in c.lower()), None)
        count_col = next((c for c in sales_df.columns if 'count' in c.lower()), None)
        if item_col is None or count_col is None:
            continue

        sales_df[item_col] = sales_df[item_col].astype(str).str.strip().str.lower()
        sales_df[count_col] = pd.to_numeric(sales_df[count_col], errors='coerce').fillna(0)
        month_name = file.split("_")[0]

        for _, row in sales_df.iterrows():
            item = row[item_col]
            count = row[count_col]
            monthly_sales.setdefault(item, {})[month_name] = monthly_sales.get(item, {}).get(month_name, 0) + count

    if not monthly_sales:
        return None

    monthly_df = pd.DataFrame(monthly_sales).fillna(0).T
    monthly_df = monthly_df.reindex(columns=MONTH_ORDER, fill_value=0)
    return monthly_df

monthly_df = load_monthly_sales(dataset_folder)
if monthly_df is None or monthly_df.empty:
    st.error("No data loaded. Check your dataset folder.")
    st.stop()

monthly_df_diff = monthly_df.diff(axis=1)

total_increase = monthly_df_diff.clip(lower=0).sum(axis=1)
rising_items = total_increase.sort_values(ascending=False).head(5)

total_decrease = monthly_df_diff.clip(upper=0).sum(axis=1)
declining_items = total_decrease.sort_values().head(5)

st.sidebar.header("📊 Display Options")
max_items = len(monthly_df)
top_n = st.sidebar.slider("Number of top items to show", 1, max_items, min(10, max_items))
top_items = monthly_df.sum(axis=1).sort_values(ascending=False).head(top_n).index

colors = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880","#FF97FF","#FECB52"]
fig = go.Figure()
for i, item in enumerate(top_items):
    fig.add_trace(go.Scatter(
        x=monthly_df.columns,
        y=monthly_df.loc[item],
        mode='lines+markers',
        name=item.title(),
        line=dict(color=colors[i % len(colors)], width=3),
        marker=dict(size=8),
        hoverinfo="x+y+name",
        legendgroup=item
    ))

fig.update_layout(
    title="Menu Item Popularity Trends (Sales Count)",
    xaxis_title="Month",
    yaxis_title="Sales Count",
    height=600,
    legend_title="Top Items",
    hovermode="x unified",
    legend=dict(itemclick="toggleothers")
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Top 5 Rising Items (Overall May→October)")
for item in rising_items.index:
    st.markdown(f"**{item.title()}** (Total Increase: {rising_items[item]:.0f})")
    st.dataframe(monthly_df_diff.loc[item])

st.subheader("📉 Top 5 Declining Items (Overall May→October)")
for item in declining_items.index:
    st.markdown(f"**{item.title()}** (Total Decrease: {declining_items[item]:.0f})")
    st.dataframe(monthly_df_diff.loc[item])

with st.expander("📄 View Full Monthly Sales Table"):
    st.dataframe(monthly_df)
