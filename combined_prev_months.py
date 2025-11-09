#Predictive analytics: Forecast future ingredient needs based on menu item demand.

#future ingredient
    # menu item demand (analyze which ingrdeints used most and used less)

#Data Prep
#   -> Combine previous months of items
import pandas as pd
import os
import openpyxl

# List of files to process
files = [
  "/Users/kara/tamu-datathon-project/dataset/May_Data_Matrix (1).xlsx",
  "/Users/kara/tamu-datathon-project/dataset/June_Data_Matrix.xlsx",
  "/Users/kara/tamu-datathon-project/dataset/July_Data_Matrix (1).xlsx",
  "/Users/kara/tamu-datathon-project/dataset/August_Data_Matrix (1).xlsx",
  "/Users/kara/tamu-datathon-project/dataset/September_Data_Matrix.xlsx",
  "/Users/kara/tamu-datathon-project/dataset/October_Data_Matrix_20251103_214000.xlsx"
  
]

# Initialize an empty list to collect cleaned dataframes
cleaned_dfs = []

for file in files:
  # Decide which sheet to read (October has different data sheet : data 2 is total menus ordered)
    if "October" in file:
        sheet_to_read = "data 2"
    else:
        sheet_to_read = "data 3"

    # Read item-level sales data
    df = pd.read_excel(file, sheet_name=sheet_to_read)

    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Clean item names: strip whitespace and convert to lowercase
    # Determine which column contains the item name
    if 'item_name' in df.columns:
        name_column = 'item_name'
    elif 'group' in df.columns:
        name_column = 'group'
    else:
        print(f"No item/group column found in {file}, skipping.")
        continue

# Clean item names
    df[name_column] = df[name_column].astype(str).str.strip().str.lower()

# Rename to a consistent name
    df.rename(columns={name_column: 'item name'}, inplace=True)   

    # Remove rows with missing or zero counts
    df['count'] = pd.to_numeric(df['count'], errors='coerce')
    df = df[df['count'].notna() & (df['count'] > 0)]

    # Add month based on filename
    month = os.path.basename(file).split('_')[0].lower()
    df['month'] = month

    # Append cleaned dataframe to list
    cleaned_dfs.append(df)

# Combine all cleaned dataframes
combined_cleaned_data = pd.concat(cleaned_dfs, ignore_index=True)

# Save to CSV (optional)
combined_cleaned_data.to_csv("cleaned_item_sales.csv", index=False)

# Display summary
print("Cleaned item-level sales data from July, August, and September:")
print(combined_cleaned_data.head())
print("\nTotal rows:", len(combined_cleaned_data))

df = pd.read_csv("cleaned_item_sales.csv")
