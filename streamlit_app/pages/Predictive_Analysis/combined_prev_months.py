# predictive_analysis/combined_prev_months.py

import pandas as pd
import os

def combine_previous_months():
    """
    Combines all monthly sales CSV files from the 'dataset' folder into one cleaned DataFrame.
    """
    dataset_folder = "data"
    output_file = "cleaned_item_sales.csv"

    all_data = []

    for file in sorted(os.listdir(dataset_folder)):
        if file.endswith(".csv"):
            month_name = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(dataset_folder, file))
            df["Month"] = month_name
            all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # Basic cleaning
    combined.dropna(subset=["Item Name"], inplace=True)
    combined["Sales Count"] = pd.to_numeric(combined["Sales Count"], errors="coerce").fillna(0)

    # Save
    combined.to_csv(output_file, index=False)
    return combined
