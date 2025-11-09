import streamlit as st
import pandas as pd
import altair as alt
import re

# PAGE CONFIGURATION
st.set_page_config(layout="wide", page_title="Ingredient Demand Forecast Viewer")

# --- Configuration ---
# Updated to match the name of the file created by the forecasting script
CSV_FILEPATH = r"C:\Users\emily\Documents\Datathon\tamu-datathon-project\streamlit_app\pages\Predictive_Analysis\ingredient_forecast_with_constraints.csv"
# Assuming 6 historical data points (May through October 2025)
HISTORICAL_MONTHS = 6 


# --- DATA LOADING AND PREPROCESSING ---
@st.cache_data
def load_data():
    """Loads, cleans, and pre-processes the ingredient forecast data."""
    try:
        df = pd.read_csv(CSV_FILEPATH)

        # Rename columns to standardized, easier-to-use names
        df = df.rename(columns={
            'Date' : 'ds', # Prophet's date column
            'Forecast_LBS_or_Count': 'yhat', # Standardized forecast (LBS/Count) - Use this for plotting
            'Forecasted_Usage_Original_Unit': 'yhat_raw_unit', # Original forecast (Grams/Count)
            'Ingredient' : 'ingredient',
            'Monthly_Supply_Constraint': 'supply',
            'Constraint_Unit': 'unit',
            'Shortfall_Surplus': 'shortfall',
            'Action_Required': 'action_required'
        })
       
        # Drop columns not needed for visualization to keep dataframe clean
        df = df.drop(columns=['Month_Label'])

        # Convert date column to datetime objects
        df['ds'] = pd.to_datetime(df['ds'])

        # Determine the period for visualization
        df['period'] = df.groupby('ingredient').cumcount()
        df['period'] = df['period'].apply(lambda x: 'Historical Proxy' if x < HISTORICAL_MONTHS else 'Future Forecast')
        
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{CSV_FILEPATH}' was not found. Please ensure it is available.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return pd.DataFrame()


# --- CHART GENERATION FUNCTIONS ---

def create_trend_chart(df, ingredient_name, unit):
    """Creates the interactive time series Altair chart for a single ingredient."""
    df_filtered = df[df['ingredient'] == ingredient_name].sort_values('ds')
    
    # Base chart setup
    base = alt.Chart(df_filtered).encode(
        x=alt.X('ds', title='Date', axis=alt.Axis(format="%b", tickCount="month")),
        y=alt.Y('yhat', title=f'Forecasted Quantity', scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('ds', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('yhat', title='Quantity', format=',.0f'),
            'period'
        ]
    )

    # 1. Continuous Line
    line = base.mark_line(color="#750e2b", strokeWidth=3) 

    # 2. Colored Points: Used to visually encode the period (Historical vs Future).
    points = base.mark_circle(size=80).encode(
        color=alt.Color('period', 
                        scale=alt.Scale(domain=['Historical Proxy', 'Future Forecast'], 
                                        range=['#2563eb', '#ef4444']), # Blue for history, Red for future
                        legend=alt.Legend(title="Forecast Period")),
        opacity=alt.value(1)
    )

    
    # Add a rule/line for the transition point (end of historical data)
    transition_df = df_filtered[df_filtered['period'] == 'Future Forecast']
    if not transition_df.empty:
        # Get the date just before the forecast starts (end of historical period)
        historical_end_date = df_filtered[df_filtered['period'] == 'Historical Proxy']['ds'].max()
        rule = alt.Chart(pd.DataFrame({'date': [historical_end_date]})).mark_rule(
            strokeDash=[5, 5], 
            color='#94a3b8',
            size=2
        ).encode(
            x='date'
        )
        chart = rule + line
    else:
        chart = line

    return chart.properties(
        title=f'Demand Forecast Trend: {ingredient_name}'
    ).interactive()


def calculate_metrics(df_filtered):
    """Calculates key metrics for the selected ingredient."""
    if df_filtered.empty:
        return None

    historical_data = df_filtered[df_filtered['period'] == 'Historical Proxy']
    forecast_data = df_filtered[df_filtered['period'] == 'Future Forecast']
    
    metrics = {}
    
    # Historical Metrics
    metrics['Historical Average'] = historical_data['yhat'].mean()
    
    # Forecast Metrics (using the last forecast point)
    if not forecast_data.empty:
        last_forecast = forecast_data.iloc[-1]
        
        metrics['Last Forecast Month'] = last_forecast['ds'].strftime('%b %Y')
        metrics['Last Forecast Value'] = last_forecast['yhat']
        metrics['Current Monthly Supply'] = last_forecast['supply']
        metrics['Shortfall/Surplus'] = last_forecast['shortfall']
        metrics['Action Required'] = last_forecast['action_required']
        metrics['Unit'] = last_forecast['unit']
        
        # Calculate Percentage Change (Last Historical vs. Last Forecast)
        if not historical_data.empty:
            last_historical = historical_data.iloc[-1]['yhat']
            change = ((last_forecast['yhat'] - last_historical) / last_historical) * 100
            metrics['% Change (Last Hist to Forecast End)'] = change

    return metrics


# --- STREAMLIT APP LAYOUT ---
if __name__ == "__main__":
    df = load_data()

    st.title("Ingredient Demand Forecast & Constraint Analysis")
    st.markdown("Use this dashboard to check future demand for ingredients and see if your current shipment schedule is sufficient to cover it.")

    if not df.empty:
        # Ingredient Selection (The Dropdown) 
        default_ingredient = 'braised beef used (g)' if 'braised beef used (g)' in df['ingredient'].unique() else df['ingredient'].iloc[0]
        
        selected_ingredient = st.selectbox(
            "**Select Ingredient to Analyze:**", 
            df['ingredient'].unique().tolist(),
            index=df['ingredient'].unique().tolist().index(default_ingredient) if default_ingredient in df['ingredient'].unique() else 0
        )

        st.markdown("---")
        
        # Calculate Metrics and get Unit
        metrics_df = df[df['ingredient'] == selected_ingredient]
        metrics = calculate_metrics(metrics_df)
        unit = metrics.get('Unit', 'Units')

        if selected_ingredient and metrics:
            
            # --- Section 1: Forecasting Trend Chart ---
            clean_ingredient = re.sub(r"\s*\(.*?\)", "", selected_ingredient).strip().title()
            st.subheader(f"1. Demand Trend for {clean_ingredient}")
            st.markdown(f"Shows the monthly usage trend, standardized to **{unit}** for supply comparison.")
            trend_chart = create_trend_chart(df, selected_ingredient, unit)
            st.altair_chart(trend_chart, use_container_width=True)

            
            # --- Section 2: Constraint Summary and Metrics ---
            
            st.markdown("---")
            st.subheader(f"Inventory Constraint Summary")

            # Metrics Row 1: Demand & Supply
            col1, col2, col3, col4 = st.columns(4)
            
            # Historical Average
            col1.metric(
                label="Avg. Past Orders", 
                value=f"{metrics.get('Historical Average', 0):,.0f} {unit}"
            )

            # Monthly Supply Constraint
            col2.metric(
                label=f"Current Monthly Supply ({unit})", 
                value=f"{metrics.get('Current Monthly Supply', 0):,.0f}"
            )

            # Last Forecast Value
            col3.metric(
                label=f"Forecasted Demand ({metrics.get('Last Forecast Month')})", 
                value=f"{metrics.get('Last Forecast Value', 0):,.0f}" if metrics.get('Last Forecast Value') is not None else "N/A"
            )

            # Shortfall / Surplus
            shortfall_value = metrics.get('Shortfall/Surplus')
            delta_color = "inverse" if shortfall_value < 0 else "normal"
            col4.metric(
                label="Shortfall / Surplus", 
                value=f"{shortfall_value:,.0f}" if shortfall_value is not None else "N/A",
                delta_color=delta_color
            )
            
            # Metrics Row 2: Action Required
            st.markdown("#### **Action Required**")
            st.markdown(f"**{metrics.get('Action Required', 'N/A')}**")


            # --- Section 4: Raw Data ---
            st.markdown("---")
            st.subheader("Raw Data Points")
            st.dataframe(metrics_df[['ds', 'yhat', 'supply', 'shortfall', 'action_required']].rename(
                columns={'ds': 'Date', 'yhat': f'Forecast Quantity ({unit})', 'supply': 'Monthly Supply', 'shortfall': 'Shortfall/Surplus', 'action_required': 'Action'}), 
                use_container_width=True
            )

    elif df.empty:
        st.warning(f"Data could not be loaded. Please ensure the required CSV file ('{CSV_FILEPATH}') is correctly formatted and available.")
    else:
        st.info("The loaded data set appears to be empty or missing sufficient ingredient information to perform the analysis.")