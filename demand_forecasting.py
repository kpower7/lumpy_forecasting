import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the daily demand data
demand_df = pd.read_excel("C:/Users/k_pow/Downloads/updated_file.xlsx")

# Load the external variable data
external_df = pd.read_excel("C:/Users/k_pow/Downloads/External_Variables.xlsx")

# Create a DateTime column in the demand data if not already created
demand_df['Date'] = pd.to_datetime(demand_df[['Year', 'Month', 'Day']])

# Extract month and year from the Date column in the demand data
demand_df['Month'] = demand_df['Date'].dt.month
demand_df['Year'] = demand_df['Date'].dt.year

# Convert the 'Date' column in the external data to datetime
external_df['DATE'] = pd.to_datetime(external_df['DATE'], format='%b-%y')

# Extract month and year from the Date column in the external data
external_df['Month'] = external_df['DATE'].dt.month
external_df['Year'] = external_df['DATE'].dt.year

# Merge the datasets on 'Month' and 'Year'
merged_df = pd.merge(demand_df, external_df, on=['Month', 'Year'], how='left')

# Drop the Month and Year columns if they are no longer needed
merged_df.drop(columns=['Month', 'Year'], inplace=True)

# Now you can proceed with forecasting or any other analysis
print(merged_df)

# Optionally, save the merged DataFrame if needed
merged_df.to_excel("C:/Users/k_pow/Downloads/merged_file.xlsx", index=False)
