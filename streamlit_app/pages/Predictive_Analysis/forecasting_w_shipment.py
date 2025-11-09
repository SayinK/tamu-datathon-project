# predictive_analysis/forecasting_w_shipment.py

import pandas as pd
import numpy as np
from prophet import Prophet
from rapidfuzz import process

def run_forecasting_with_shipments():
    """
    Forecasts ingredient demand and compares it with shipment data to estimate shortages/surpluses.
    """

    # --- CONSTANTS ---
    FUTURE_MONTHS = 3
    CHANGEPOINT_PRIOR_SCALE = 0.01
    CLIP_FACTOR = 5.0
    G_TO_LBS = 1 / 453.592
    LBS_TO_G = 453.592

    # --- LOAD DATA ---
    sales = pd.read_csv("cleaned_item_sales.csv")
    ingredients = pd.read_csv("data/MSY Data - Ingredient.csv")
    shipments = pd.read_csv("data/MSY Data - Shipment.csv")

    # --- CLEAN & PREP DATA ---
    sales["Date"] = pd.to_datetime(sales["Month"] + "-01")
    sales["Sales Count"] = pd.to_numeric(sales["Sales Count"], errors="coerce").fillna(0)

    # Aggregate per ingredient
    grouped = sales.groupby(["Date", "Item Name"])["Sales Count"].sum().reset_index()
    grouped.rename(columns={"Item Name": "Ingredient"}, inplace=True)

    final_forecast_list = []

    for ingredient, group in grouped.groupby("Ingredient"):
        if len(group) < 3:
            continue

        df = group[["Date", "Sales Count"]].rename(columns={"Date": "ds", "Sales Count": "y"})
        df["y"] = df["y"].clip(0, df["y"].mean() * CLIP_FACTOR)

        model = Prophet(changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE)
        model.fit(df)

        future = model.make_future_dataframe(periods=FUTURE_MONTHS, freq="M")
        forecast = model.predict(future)

        recent_shipments = shipments[shipments["Ingredient"].str.contains(ingredient, case=False, na=False)]
        avg_shipment = recent_shipments["Shipment Weight (lbs)"].mean() if not recent_shipments.empty else np.nan

        for _, row in forecast.tail(FUTURE_MONTHS).iterrows():
            predicted_demand = row["yhat"]
            shortfall = avg_shipment - predicted_demand if not np.isnan(avg_shipment) else np.nan
            final_forecast_list.append({
                "Ingredient": ingredient,
                "Month": row["ds"].strftime("%Y-%m"),
                "Predicted Demand": predicted_demand,
                "Avg Shipment (lbs)": avg_shipment,
                "Shortfall_Surplus": shortfall
            })

    final_forecast = pd.DataFrame(final_forecast_list)
    final_forecast.to_csv("ingredient_forecast_with_constraints.csv", index=False)
    return final_forecast
