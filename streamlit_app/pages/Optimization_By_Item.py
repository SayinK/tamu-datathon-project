# pages/Optimization_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
import os

st.set_page_config(page_title="Optimization Dashboard", layout="wide")

st.sidebar.title("⚙️ Optimization Mode")
mode = st.sidebar.selectbox(
    "Choose Optimization Type:",
    ["Item Optimization", "Ingredient Optimization"]
)

# ITEM OPTIMIZATION
@st.cache_data
def load_month_data(file_path, sheet_name, month_name):
    """Loads Excel data for one month and cleans it."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        st.warning(f"⚠️ Could not load {file_path}: {e}")
        return None

    item_col = next((col for col in df.columns if 'item' in col.lower()), None)
    amount_col = next((col for col in df.columns if 'amount' in col.lower()), None)
    if not item_col or not amount_col:
        st.warning(f"⚠️ Missing columns in {file_path}")
        return None

    df[amount_col] = pd.to_numeric(df[amount_col].replace('[\$,]', '', regex=True), errors='coerce')
    df = df[df[amount_col].notna() & (df[amount_col] != 0)]
    df = df[[item_col, amount_col]].rename(columns={item_col: 'Item Name', amount_col: 'Amount'})
    df['Month'] = month_name
    return df


files = [
    ("data/May_Data_Matrix.xlsx", "data 3", "May"),
    ("data/June_Data_Matrix.xlsx", "data 3", "June"),
    ("data/July_Data_Matrix.xlsx", "data 3", "July"),
    ("data/August_Data_Matrix.xlsx", "data 3", "August"),
    ("data/September_Data_Matrix.xlsx", "data 3", "September"),
    ("data/October_Data_Matrix.xlsx", "data 2", "October"),
]

dfs = [load_month_data(path, sheet, name) for path, sheet, name in files if os.path.exists(path)]
dfs = [df for df in dfs if df is not None]


# INGREDIENT OPTIMIZATION
@st.cache_data
def load_ingredient_data():
    """Loads and processes ingredient-level optimization."""
    ingredient_df = pd.read_csv("data/MSY Data - Ingredient.csv")
    ingredient_df.columns = ingredient_df.columns.str.strip()
    ingredient_df['Item name'] = ingredient_df['Item name'].str.strip().str.lower()

    month_files = [
        ("data/May_Data_Matrix.xlsx", "data 3", "May"),
        ("data/June_Data_Matrix.xlsx", "data 3", "June"),
        ("data/July_Data_Matrix.xlsx", "data 3", "July"),
        ("data/August_Data_Matrix.xlsx", "data 3", "August"),
        ("data/September_Data_Matrix.xlsx", "data 3", "September"),
        ("data/October_Data_Matrix.xlsx", "data 2", "October"),
    ]

    monthly_dfs = []
    for file_path, sheet_name, month_name in month_files:
        if not os.path.exists(file_path):
            continue
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        item_col = next((col for col in df.columns if 'item' in col.lower()), None)
        amount_col = next((col for col in df.columns if 'amount' in col.lower()), None)
        if not item_col or not amount_col:
            continue
        df[amount_col] = pd.to_numeric(df[amount_col].replace('[\$,]', '', regex=True), errors='coerce')
        df = df[df[amount_col].notna() & (df[amount_col] != 0)]
        df = df[[item_col, amount_col]].rename(columns={item_col: 'Item Name', amount_col: 'Amount'})
        df['Month'] = month_name
        df['Item Name'] = df['Item Name'].str.strip().str.lower()
        monthly_dfs.append(df)

    combined_df = pd.concat(monthly_dfs, ignore_index=True)

    special_matches = {
        "fried wings": ["chicken wings", "fried chicken", "crunch chicken"],
        "chicken cutlet": ["chicken cutlet"],
        "beef tossed rice noodles": ["beef tossed rice noodle"],
        "pork tossed rice noodles": ["pork tossed rice noodle"],
        "chicken tossed rice noodles": ["chicken tossed rice noodle"]
    }

    ingredient_profit_per_month = {month: {} for month in combined_df['Month'].unique()}
    month_total_profit = {}

    for month in combined_df['Month'].unique():
        month_df = combined_df[combined_df['Month'] == month]
        month_total = month_df['Amount'].sum()
        month_total_profit[month] = month_total

        for ingredient in ingredient_df.columns[1:]:
            ingredient_clean = ingredient.strip()
            used_in_items = ingredient_df.loc[
                ingredient_df[ingredient].notna() & (ingredient_df[ingredient] != 0),
                'Item name'
            ].dropna().tolist()

            total_profit = 0.0
            for item in used_in_items:
                patterns = special_matches.get(item, [item])
                for p in patterns:
                    matched = month_df[month_df['Item Name'].str.contains(p, case=False, na=False)]
                    if not matched.empty:
                        total_profit += matched['Amount'].sum()

            ingredient_profit_per_month[month][ingredient_clean] = total_profit

    return ingredient_profit_per_month, month_total_profit

if mode == "Item Optimization":
    st.header("Optimization by Item")

    if len(dfs) == 0:
        st.error("🚫 No item data could be loaded. Check file paths.")
        st.stop()

    combined_df = pd.concat(dfs, ignore_index=True)
    avg_df = combined_df.groupby('Item Name', as_index=False)['Amount'].mean()

    st.sidebar.header("📅 Filters")
    month_names = [name for _, _, name in files]
    selected_month = st.sidebar.selectbox("Select month:", month_names)
    top_n = 14  # fixed number of bars

    selected_tuple = next((f for f in files if f[2] == selected_month), None)
    if selected_tuple:
        file_path, sheet_name, month_name = selected_tuple
        month_df = load_month_data(file_path, sheet_name, month_name)
    else:
        st.stop()

    if month_df is not None and not month_df.empty:
        month_df = month_df.sort_values(by='Amount', ascending=False)
        df = month_df.head(top_n)
        avg_vals = avg_df.set_index('Item Name').reindex(df['Item Name'])['Amount'].fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Item Name'], y=df['Amount'], name=f"{month_name}", marker_color='#D41919'))
        fig.add_trace(go.Bar(x=df['Item Name'], y=avg_vals, name="Average Across Months", marker_color='lightgray'))

        fig.update_layout(
            title=f"Profit by Item — {month_name} vs Average",
            xaxis_title="Item Name",
            yaxis_title="Profit ($)",
            barmode='group',
            xaxis_tickangle=-45,
            legend=dict(x=0.02, y=0.98),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)

elif mode == "Ingredient Optimization":
    st.header("Optimization by Ingredient")

    ingredient_profit_per_month, month_total_profit = load_ingredient_data()

    month_names = list(ingredient_profit_per_month.keys())
    selected_month = st.sidebar.selectbox("Select month:", month_names)

    profits = ingredient_profit_per_month[selected_month]
    df_plot = pd.DataFrame(list(profits.items()), columns=['Ingredient', 'Total Profit'])
    total_profit = month_total_profit[selected_month]
    df_plot['Percentage'] = (df_plot['Total Profit'] / total_profit) * 100
    df_plot = df_plot.sort_values(by='Percentage', ascending=False).head(14)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_plot['Ingredient'],
        x=df_plot['Percentage'],
        orientation='h',
        marker_color='#FFFFFF',
        name='Profit %'
    ))

    fig.update_layout(
        title=f"Ingredient Profit Contribution — {selected_month}",
        xaxis_title="Percentage of Total Monthly Profit (%)",
        yaxis_title="Ingredient",
        height=700,
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_plot)
