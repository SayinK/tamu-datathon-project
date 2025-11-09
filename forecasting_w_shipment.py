import pandas as pd
import numpy as np
from prophet import Prophet
from rapidfuzz import process
import re

# Configuration for forecasting 
FUTURE_MONTHS = 3
CLIP_FACTOR = 5.0 # Clip predictions at 500% of the maximum historical usage 
CHANGEPOINT_PRIOR_SCALE = 0.01 

# Unit Conversion Constant: 1 pound = 453.592 grams
G_TO_LBS = 1 / 453.592
LBS_TO_G = 453.592


print(" @@@@ Starting Ingredient Demand Forecasting with Shipment Integration @@@@")

# Load Data
sales = pd.read_csv("cleaned_item_sales.csv")
ingredients = pd.read_csv("dataset/MSY Data - Ingredient.csv")
shipments = pd.read_csv("dataset/MSY Data - Shipment.csv") # <-- Shipment data is loaded

# --- Shipment Data Preprocessing ---
def calculate_monthly_supply(shipments_df):
    """
    Calculates the total monthly supply quantity for each ingredient based on 
    shipment frequency. Assumes 4.33 weeks per month.
    """
    
    # 1. Standardize Ingredient names (used as the primary merge key)
    shipments_df['Ingredient_Key'] = (
        shipments_df['Ingredient'].str.lower()
        .str.replace(' ', '')
        .str.replace('+', 'and', regex=False)
        .str.strip()
    )
    
    # 2. Map frequency to a multiplication factor (monthly total)
    def get_monthly_factor(freq):
        freq = freq.lower().strip()
        if 'weekly' in freq:
            return 4.33  # 4.33 weeks per month
        elif 'biweekly' in freq:
            return 2.16  # 2.16 bi-weeks per month
        elif 'monthly' in freq:
            return 1.0
        return 0

    shipments_df['Monthly_Factor'] = shipments_df['frequency'].apply(get_monthly_factor)
    
    # 3. Calculate Total Monthly Supply Quantity
    shipments_df['Monthly_Supply_Qty'] = (
        shipments_df['Quantity per shipment'] * shipments_df['Number of shipments'] * shipments_df['Monthly_Factor']
    )
    
    # Filter only necessary columns and return
    return shipments_df[['Ingredient_Key', 'Monthly_Supply_Qty', 'Unit of shipment']]

shipment_supply = calculate_monthly_supply(shipments)


# Data Preprocessing (Existing Logic for Demand Forecast) !!!!

# Aggregate menu demand by month
sales["month"] = sales["month"].str.capitalize() + " 2025"
sales["month"] = pd.to_datetime(sales["month"], format="%B %Y")

# Standardize item names in BOTH datasets
sales['item name'] = sales['item name'].str.lower().str.strip()
ingredients['Item name'] = ingredients['Item name'].str.lower().str.strip()

# Remove quantity tags ("(8)", "(10)") from item names
sales['item name'] = sales['item name'].apply(lambda x: re.sub(r"\(\d+\)", "", x))
ingredients['Item name'] = ingredients['Item name'].apply(lambda x: re.sub(r"\(\d+\)", "", x))

# Handle known special menu name mappings
manual_map = {
    "mai's bf chicken cutlet": "chicken cutlet",
    "mai og fried chicken wings": "fried wings",
}
sales['item name'] = sales['item name'].replace(manual_map)

# Fuzzy matching for remaining items
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
        yearly_seasonality=False,
        changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE 
    )
    model.fit(df)

    # Future dataframe
    future = model.make_future_dataframe(periods=FUTURE_MONTHS, freq="M")
    forecast = model.predict(future)

    # Back-transform to original scale
    forecast_df = forecast[["ds", "yhat"]].copy()
    forecast_df["yhat"] = np.exp(forecast_df["yhat"])
    
    # Clip the predictions
    forecast_df["yhat"] = forecast_df["yhat"].clip(
        lower=0, 
        upper=max_historical_usage * CLIP_FACTOR
    )

    forecast_df["ingredient"] = ing

    all_forecasts.append(forecast_df)

forecast_result = pd.concat(all_forecasts, ignore_index=True)


# Shipment Constraint Analysis ---

def get_ingredient_key(ingredient_name):
    """Maps the ingredient name from the usage data to the shipment data key."""
    name = ingredient_name.lower()
    
    # Specific mappings based on ingredient file column names
    if 'braised beef' in name: return 'beef'
    if 'braised chicken' in name: return 'chicken'
    if 'braised pork' in name: return 'pork' 
    if 'ramen (count)' in name: return 'ramen'
    if 'rice noodles' in name: return 'rice noodles'
    if 'flour' in name: return 'flour'
    if 'tapioca starch' in name: return 'tapioca starch'
    if 'rice(g)' in name: return 'rice'
    if 'green onion' in name: return 'green onion'
    if 'white onion' in name: return 'white onion'
    if 'cilantro' in name: return 'cilantro'
    if 'egg(count)' in name: return 'egg'
    if 'peas(g)' in name or 'carrot(g)' in name: return 'peasandcarrot'
    if 'boychoy(g)' in name: return 'bokchoy'
    if 'chicken wings' in name: return 'chicken wings'
    # Default to clean name if no specific match
    return name.split('(')[0].replace(' ', '').strip()


# Apply mapping key for merging
forecast_result['Ingredient_Key'] = forecast_result['ingredient'].apply(get_ingredient_key)

# Merge Forecast with Monthly Supply Data
final_forecast = forecast_result.merge(
    shipment_supply[['Ingredient_Key', 'Monthly_Supply_Qty', 'Unit of shipment']],
    on='Ingredient_Key',
    how='left'
)

# Standardize the Forecasted Usage to LBS or Count for comparison
def standardize_forecast_unit(row):
    """
    Converts grams to lbs if the shipment unit is lbs.
    Aligns count/piece units directly.
    """
    unit = row['Unit of shipment']
    original_qty = row['yhat']
    
    # Check if the ingredient was originally measured in grams and is shipped in lbs
    if 'g)' in row['ingredient'] and unit == 'lbs':
        return original_qty * G_TO_LBS
    # Check if the ingredient was originally measured in count and is shipped in a count unit
    elif 'count)' in row['ingredient'] and unit in ['eggs', 'rolls', 'pieces', 'whole onion']:
        return original_qty 
    # Handle the rest (usually already matched units or unmappable ones)
    return original_qty

final_forecast['Forecast_LBS_or_Count'] = final_forecast.apply(standardize_forecast_unit, axis=1)


# Calculate Shortfall/Surplus
final_forecast['Shortfall_Surplus'] = final_forecast['Monthly_Supply_Qty'] - final_forecast['Forecast_LBS_or_Count']


# Create Actionable Flag (Future months only)
final_forecast['Action_Required'] = final_forecast.apply(
    lambda row: '⚠️ SHORTFALL: Order More' if row['Shortfall_Surplus'] < 0 and row['ds'] > ingredient_timeseries['month'].max()
                else '✅ Sufficient Supply' if row['Shortfall_Surplus'] >= 0 and row['ds'] > ingredient_timeseries['month'].max()
                else 'Historical Data',
    axis=1
)


# Final Formatting
final_forecast['Month_Label'] = final_forecast["ds"].dt.strftime('%b')
final_forecast = final_forecast.rename(columns={
    'ds': 'Date', 
    'yhat': 'Forecasted_Usage_Original_Unit', # Now the original grams/count
    'ingredient': 'Ingredient',
    'Monthly_Supply_Qty': 'Monthly_Supply_Constraint',
    'Unit of shipment': 'Constraint_Unit'
})

# Select and reorder columns
final_forecast = final_forecast[[
    'Month_Label', 'Date', 'Ingredient', 'Forecasted_Usage_Original_Unit', 
    'Constraint_Unit', 'Forecast_LBS_or_Count', 'Monthly_Supply_Constraint', 
    'Shortfall_Surplus', 'Action_Required'
]]


final_forecast.to_csv("ingredient_forecast_with_constraints.csv", index=False)

print("\nForecast completed and constrained analysis added - saved to ingredient_forecast_with_constraints.csv")
print("\nCheck the 'Action_Required' column in the output file for actionable insights on ordering.")