import pandas as pd
import numpy as np
from prophet import Prophet
from rapidfuzz import process
import re

# Configuration for forecasting 
FUTURE_MONTHS = 3
CLIP_FACTOR = 5.0 # Clip predictions at 500% of the maximum historical usage // A low value prevents the model from extrapolating wildly from only 6 months of data.
CHANGEPOINT_PRIOR_SCALE = 0.01 

print(" @@@@ Starting Ingredient Demand Forecasting with Stability Fixes @@@@")

# Load Data
sales = pd.read_csv("cleaned_item_sales.csv")
ingredients = pd.read_csv("dataset/MSY Data - Ingredient.csv")
# shipments = pd.read_csv("MSY Data - Shipment.csv")


# Data Preprocessing !!!!
# Aggregate menu demand by month
sales["month"] = sales["month"].str.capitalize() + " 2025"
sales["month"] = pd.to_datetime(sales["month"], format="%B %Y")

# Standardize item names in BOTH datasets
sales['item name'] = sales['item name'].str.lower().str.strip()
ingredients['Item name'] = ingredients['Item name'].str.lower().str.strip()

# Remove quantity tags (e.g. "(8)") from item names
sales['item name'] = sales['item name'].apply(lambda x: re.sub(r"\(\d+\)", "", x))
ingredients['Item name'] = ingredients['Item name'].apply(lambda x: re.sub(r"\(\d+\)", "", x))

# Handle known special menu name mappings
manual_map = {
    "mai's bf chicken cutlet": "chicken cutlet",
    "mai og fried chicken wings": "fried wings",
}
sales['item name'] = sales['item name'].replace(manual_map)

# Fuzzy matching for remaining items :) fuzzyy sounds funny
ingredient_names = ingredients['Item name'].tolist()

def fuzzy_map(sales_item):
    match, score, idx = process.extractOne(sales_item, ingredient_names)
    if score >= 70:
        return match
    return sales_item

sales['item name'] = sales['item name'].apply(fuzzy_map)

menu_demand = (
    sales.groupby(['month', 'item name'])['count']
    .sum()
    .reset_index()
)

# Merge with ingredient requirements
ingredient_usage = menu_demand.merge(
    ingredients,
    left_on = 'item name',
    right_on = 'Item name'
)

# List ingredient columns
ignore_cols = ["month", "item name", "Item name", "count"]
ingredient_cols = [col for col in ingredient_usage.columns if col not in ignore_cols]

# Multiply count * qty for each ingredient column
for col in ingredient_cols:
    ingredient_usage[col] = pd.to_numeric(ingredient_usage[col], errors="coerce").fillna(0)
    ingredient_usage[col] = ingredient_usage[col] * ingredient_usage["count"]

# Melt and Aggregate total usage per month per ingredient
ingredient_long = ingredient_usage.melt(
    id_vars=["month", "item name"],
    value_vars=ingredient_cols,
    var_name="ingredient",
    value_name="total_qty_used"
).dropna(subset=['total_qty_used']) 

ingredient_timeseries = (
    ingredient_long.groupby(["month", "ingredient"])["total_qty_used"]
    .sum()
    .reset_index()
)

# DEBUGGING
print("\n" + "="*50)
print("RAW HISTORICAL INPUT DATA FOR RAMEN (BEFORE SMOOTHING/LOG TRANSFORMATION):")
boychoy_history = ingredient_timeseries[ingredient_timeseries["ingredient"] == "Boychoy(g)"]
print(boychoy_history)
print("="*50 + "\n")
# ---------------------------------



#  FORECASTING LOOP 
all_forecasts = []

for ing in ingredient_timeseries["ingredient"].unique():
    temp_df = ingredient_timeseries[ingredient_timeseries["ingredient"] == ing].copy()
    
    # 1. Calculate Max Historical Usage for clipping later
    max_historical_usage = temp_df["total_qty_used"].max()

    # Smooth spikes using 2-month rolling average
    temp_df["total_qty_used"] = temp_df["total_qty_used"].rolling(2, min_periods=1).mean()

    # Log-transform to stabilize variance
    temp_df["y"] = temp_df["total_qty_used"].apply(lambda x: max(x, 1))  # avoid log(0)
    temp_df["y"] = np.log(temp_df["y"])

    # Prepare for Prophet
    df = temp_df.rename(columns={"month": "ds"})
    df = df[["ds", "y"]]

    # Fit Prophet
    model = Prophet(
        growth='linear', 
        yearly_seasonality=False, #disabled yearly seasonality (yearly_seasonality=False) 
        changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE  #applied low changepoint_prior_scale for stable trend
    )
    model.fit(df)

    # Future dataframe
    future = model.make_future_dataframe(periods=FUTURE_MONTHS, freq="M")
    forecast = model.predict(future)

    # Back-transform to original scale
    forecast_df = forecast[["ds", "yhat"]].copy()
    forecast_df["yhat"] = np.exp(forecast_df["yhat"])
    
    forecast_df["yhat"] = forecast_df["yhat"].clip(
        lower=0, 
        upper=max_historical_usage * CLIP_FACTOR
    )

    forecast_df["ingredient"] = ing

    all_forecasts.append(forecast_df)

forecast_result = pd.concat(all_forecasts, ignore_index=True)


# preventing the "nanosecond timestamp" error.
forecast_result['Month_Label'] = forecast_result["ds"].dt.strftime('%b')

# Rename the original Prophet columns for clarity!!
forecast_result = forecast_result.rename(columns={
    'ds': 'Date', 
    'yhat': 'Forecasted_Usage', 
    'ingredient': 'Ingredient'
})

# Select and reorder columns
forecast_result = forecast_result[['Month_Label', 'Date', 'Forecasted_Usage', 'Ingredient']]


forecast_result.to_csv("ingredient_forecast_fixed.csv", index=False)

print("\nForecast completed — saved to ingredient_forecast.csv")
print("Stability fixes applied: trend flexibility was reduced and predictions were capped.")