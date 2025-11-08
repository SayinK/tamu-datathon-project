import streamlit as st
import pandas as pd

st.title("My Streamlit App")

df = pd.read_csv("data.csv")  # same folder as this script
st.write("Preview:", df.head())

# Option A: plot all numeric columns
st.line_chart(df.select_dtypes("number"))