# predictive_analysis/ingredient_demand_forecast.py

import pandas as pd
from prophet import Prophet

def run_forecast():
    """
    Generates a basic Prophet forecast for ingredient demand using combined sales data.
    """
    df = pd.read_csv("cleaned_item_sales.csv")

    if "Date" not in df.columns:
        # Example: create a date column from Month name
        df["Date"] = pd.to_datetime(df["Month"] + "-01")

    # Aggregate sales per month
    grouped = df.groupby("Date")["Sales Count"].sum().reset_index()
    grouped.rename(columns={"Date": "ds", "Sales Count": "y"}, inplace=True)

    # Prophet forecast
    model = Prophet(yearly_seasonality=True)
    model.fit(grouped)

    future = model.make_future_dataframe(periods=3, freq="M")
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    result.to_csv("ingredient_demand_forecast.csv", index=False)

    return result
