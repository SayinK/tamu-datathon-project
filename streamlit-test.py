import streamlit as st
import pandas as pd

st.title("Sales Dashboard")
st.header("Monthly Revenue Chart")

df = pd.read_csv("data.csv")
st.write("Here is the raw data:")
st.dataframe(df)

st.write("Revenue over time:")
st.line_chart(df.set_index('Month')['Revenue'])
