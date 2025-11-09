# tamu-datathon-project 2025 
**Mai-Shan-Yun Vision** <br/>
Mai-Shan-Yun project is to design and build an interactive dashboard that transforms raw restaurant data into actionable intelligence.

<br/>

## Description:
- Developed a dashboard, using Streamlit, that integrates item sales, recipe matrices, and supply data to generate precise ingredient usage forecasts for every month.
- Implemented a predictive analytics pipeline (Python/Prophet) to convert forecasted demand into specific ingredient requirements.
- Automated a constraint analysis module to proactively identify potential supply shortfall risks by comparing projected demand against planned inventory capacity, facilitating optimized procurement and inventory management.

<br/>

## Dashboard purpose and Key Insights 
- Forecasting Ingredient Analysis: Predict future ingredient demand to guide procurement and planning
- Ingredient Insights: Track ingredient usage trends and identify patterns for better inventory management
- Menu Items Trend: Analyze the popularity of menu items over time 
- Monthly Category Income: Review revenue breakdown by category for financial visibility
- Optimization by Item: Suggest profit optimization and performance improvements per item and ingredient
- Shipments Dashboard: Compare expected versus actual shipments to monitor supply chain accuracy

<br/>

## Datasets used and How They were integrated 
- Shipments: Tracked total quantities of ingredients received each period of time (by unit) and compared them with monthly ingredient usage to analyze inventory flow
- Ingredients: Combined with monthly data to identify the most frequently used ingredients and their contribution to profit generation
- Months: Used to evaluate and visualize item profitability by month and to compare monthly performance with overall averages


## Technologies used 
- Frontend: Streamlit, CSS, HTML
- Backend: Python, Pandas, Prophet (Forecasting Model), Altair 


## Potential Improvements for the Future 
To further increase its accuracy, scope, and strategic value, the following improvements could be implemented:

1. Model & Data Enhancements
- Incorporate external variables to improve demand predictions (events, holidays, weather).
- Factor in seasonality and local trends to make short-term forecasts more accurate.

2. User Experience & Operations
- Interactive "What-If" dashboard to simulate ingredient demand changes.
- Mobile alerts for critical supply shortfalls or excess inventory.

3. Platform Expansion
- Deploy the Streamlit app for cloud access and multi-user functionality.
- Connect to live databases or APIs for real-time data.
- Automated alerts for upcoming shortages or excess stock.
