import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os

st.set_page_config(page_title="Menu Ingredient Network", layout="wide")
st.title("Menu Item - Ingredient Network for May")

min_qty = 10
top_n_items = 10 

excel_file = "data/May_Data_Matrix.xlsx"
sheet_name = 'data 3'
ingredient_file = "data/MSY Data - Ingredient.csv"

sales_df = pd.read_excel(excel_file, sheet_name=sheet_name)
sales_df['Count'] = pd.to_numeric(sales_df['Count'], errors='coerce')
sales_df['item_name'] = sales_df['Item Name'].str.lower().str.strip()

top_items = sales_df.sort_values('Count', ascending=False).head(top_n_items)

ingredients_df = pd.read_csv(ingredient_file)
ingredients_df['item_name'] = ingredients_df['Item name'].str.lower().str.strip()

merged_df = pd.merge(top_items, ingredients_df, on='item_name', how='left')

ingredient_cols = [col for col in ingredients_df.columns if col.lower() not in ['item name', 'item_name']]

G = nx.Graph()

for _, row in merged_df.iterrows():
    item = row['item_name']
    G.add_node(item, color='orange', size=25, title=f"{item}")

    for ing in ingredient_cols:
        qty = pd.to_numeric(row[ing], errors='coerce')
        if pd.notnull(qty) and qty >= min_qty:
            if not G.has_node(ing):
                G.add_node(ing, color='lightblue', size=15, title=f"{ing}")
            G.add_edge(item, ing, value=qty, title=f"{qty} units")

net = Network(height="750px", width="100%", notebook=False, bgcolor="#ffffff", font_color="black")
net.from_nx(G)

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    tmp_path = tmp_file.name
    net.write_html(tmp_path)

    st.components.v1.html(
        open(tmp_path, 'r', encoding='utf-8').read(),
        height=750,
        scrolling=True
    )

os.remove(tmp_path)

