# %% [markdown]
# # Enhanced Demand Forecasting using Prophet with Daily Data
# This script performs demand forecasting using daily demand data combined with monthly economic indicators.

# %% [markdown]
# ## Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# Set base path
BASE_PATH = r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC"

# Load daily data
daily_df = pd.read_excel(
    os.path.join(BASE_PATH, "Customer Order Quantity_Dispatched Quantity.xlsx")
)

# Filter for Material_4
material_id = '000161032'  # Material_4
material_no = 'Material_4'
daily_df = daily_df[daily_df['Product ID'] == material_id].copy()

print(f"\nDEBUG: Initial data shape for {material_no}: {daily_df.shape}")
print(f"DEBUG: Date range: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
print(f"DEBUG: Total demand: {daily_df['Dispatched Quantity'].sum()}")

# Convert date column
daily_df['Date'] = pd.to_datetime(daily_df['Date'], format='%d.%m.%Y')

# Create a complete date range from the minimum to the maximum date
full_date_range = pd.date_range(start='2022-01-01', end='2024-12-05', freq='D')

# Prepare data for Prophet (requires 'ds' for dates and 'y' for values)
prophet_df = pd.DataFrame({
    'ds': full_date_range,
    'y': daily_df.set_index('Date').reindex(full_date_range)['Dispatched Quantity'].fillna(0)
})

print(f"DEBUG: Prophet data shape: {prophet_df.shape}")
print(f"DEBUG: Non-zero demand days: {(prophet_df['y'] > 0).sum()}")

# Load monthly data with external variables
monthly_df = pd.read_csv(os.path.join(BASE_PATH, "Data_files", f"{material_id}_{material_no}.csv"))
monthly_df['YearMonth'] = pd.to_datetime(monthly_df['YearMonth'])

# Add external regressors from monthly data
external_vars = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
for var in external_vars:
    if var in monthly_df.columns:
        # Convert monthly data to daily by forward filling
        daily_values = monthly_df.set_index('YearMonth')[var].reindex(full_date_range, method='ffill')
        # Fill any remaining NaN values with the mean
        daily_values = daily_values.fillna(daily_values.mean())
        prophet_df[var] = daily_values

print("Data loaded successfully.")
print("Daily data shape:", prophet_df.shape)
print("Monthly data shape:", monthly_df.shape)

# Split data based on date
train_cutoff = pd.to_datetime('2024-06-01')
train_mask = prophet_df['ds'] < train_cutoff

train_data = prophet_df[train_mask].copy()
test_data = prophet_df[~train_mask].copy()

# Create validation set from last month of training data
val_cutoff = train_cutoff - pd.DateOffset(months=1)
val_mask = (prophet_df['ds'] >= val_cutoff) & (prophet_df['ds'] < train_cutoff)
val_data = prophet_df[val_mask].copy()
train_data = prophet_df[prophet_df['ds'] < val_cutoff].copy()

print("Training data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)
print("Test data shape:", test_data.shape)

# %% [markdown]
# ## Train Model

# %%
# Initialize and train Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode='multiplicative',
    interval_width=0.95,
    changepoint_prior_scale=0.05
)

# Add external regressors
for var in external_vars:
    if var in train_data.columns:
        model.add_regressor(var)

# Fit the model
model.fit(train_data)

# Make predictions
future_dates = pd.concat([train_data, test_data])['ds'].reset_index(drop=True)
future_df = pd.DataFrame({'ds': future_dates})

# Add external regressors to future dataframe
for var in external_vars:
    if var in train_data.columns:
        values = pd.concat([train_data[var], test_data[var]]).reset_index(drop=True)
        # Fill any NaN values with the mean
        values = values.fillna(values.mean())
        future_df[var] = values

forecast = model.predict(future_df)

print("Model training completed")

# %% [markdown]
# ## Evaluate Model at Multiple Time Scales

# %%
# Create results DataFrame for daily predictions
test_start = pd.Timestamp('2024-06-01')
test_end = pd.Timestamp('2024-12-05')
daily_results = pd.DataFrame({
    'Date': test_data['ds'],
    'Actual': test_data['y'],
    'Predicted': forecast[forecast['ds'].isin(test_data['ds'])]['yhat']
})
daily_results = daily_results[(daily_results['Date'] >= test_start) & (daily_results['Date'] <= test_end)]

# Aggregate to weekly - start from first Monday in June
weekly_df = daily_results.copy()
weekly_df.set_index('Date', inplace=True)
weekly_results = weekly_df.resample('W-MON').sum()
weekly_results = weekly_results.loc[test_start:test_end]
weekly_results = weekly_results.reset_index()

# Aggregate to monthly - start from June 1st
monthly_df = daily_results.copy()
monthly_df.set_index('Date', inplace=True)
monthly_results = monthly_df.resample('MS').sum()
monthly_results = monthly_results.loc[test_start:test_end]
monthly_results = monthly_results.reset_index()

print("\nDaily Results Date Range:")
print(daily_results['Date'].min(), "to", daily_results['Date'].max())
print("\nMonthly Results Date Range:")
print(monthly_results['Date'].min(), "to", monthly_results['Date'].max())

print("\nAggregation Info:")
print(f"Daily predictions: {len(daily_results)} days")
print(f"Weeks: {len(weekly_results)} weeks")
print(f"Months: {len(monthly_results)} months")

# Function to calculate metrics including Excel-style R²
def calculate_metrics(actual, predicted):
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {
            'MSE': None,
            'RMSE': None,
            'MAE': None,
            'R2': None
        }
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    # Calculate R² as the square of correlation coefficient (Excel method)
    r2 = np.corrcoef(actual, predicted)[0, 1] ** 2 if len(actual) > 1 else None
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Calculate metrics for daily predictions
daily_metrics = calculate_metrics(daily_results['Actual'], daily_results['Predicted'])
print("\nDEBUG: Metric Calculation Details")
print(f"DEBUG: Daily predictions shape: {len(daily_results)}")
print(f"DEBUG: Daily actual range: {daily_results['Actual'].min()} to {daily_results['Actual'].max()}")
print(f"DEBUG: Daily predicted range: {daily_results['Predicted'].min()} to {daily_results['Predicted'].max()}")

print("\n=== METRICS START ===")
print("Daily Metrics:")
print(f"DAILY_R2={daily_metrics['R2']:.6f}" if daily_metrics['R2'] is not None else "DAILY_R2=N/A")

# Calculate metrics for weekly predictions
weekly_metrics = calculate_metrics(weekly_results['Actual'], weekly_results['Predicted'])
print(f"\nDEBUG: Weekly predictions shape: {len(weekly_results)}")
print("\nWeekly Metrics:")
print(f"WEEKLY_R2={weekly_metrics['R2']:.6f}" if weekly_metrics['R2'] is not None else "WEEKLY_R2=N/A")

# Calculate metrics for monthly predictions
monthly_metrics = calculate_metrics(monthly_results['Actual'], monthly_results['Predicted'])
print(f"\nDEBUG: Monthly predictions shape: {len(monthly_results)}")
print("\nMonthly Metrics:")
print(f"MONTHLY_R2={monthly_metrics['R2']:.6f}" if monthly_metrics['R2'] is not None else "MONTHLY_R2=N/A")
print("=== METRICS END ===\n")

# Save metrics to CSV for verification
metrics_df = pd.DataFrame({
    'Timeframe': ['Daily', 'Weekly', 'Monthly'],
    'RMSE': [daily_metrics['RMSE'], weekly_metrics['RMSE'], monthly_metrics['RMSE']],
    'MAE': [daily_metrics['MAE'], weekly_metrics['MAE'], monthly_metrics['MAE']],
    'R2': [daily_metrics['R2'], weekly_metrics['R2'], monthly_metrics['R2']]
})

output_dir = os.path.join(os.path.dirname(__file__), 'predictions_prophet')
os.makedirs(output_dir, exist_ok=True)

metrics_output = os.path.join(output_dir, 'prediction_metrics.csv')
metrics_df.to_csv(metrics_output, index=False)
print(f"\nSaved metrics to: {metrics_output}")

# Save predictions to CSV files
daily_output = os.path.join(output_dir, 'daily_predictions.csv')
daily_results.to_csv(daily_output, index=True)
print(f"\nSaved daily predictions to: {daily_output}")

weekly_output = os.path.join(output_dir, 'weekly_predictions.csv')
weekly_results.to_csv(weekly_output, index=False)
print(f"Saved weekly predictions to: {weekly_output}")

monthly_output = os.path.join(output_dir, 'monthly_predictions.csv')
monthly_results.to_csv(monthly_output, index=False)
print(f"Saved monthly predictions to: {monthly_output}")

# Print summary statistics
print("\nSummary of predictions:")
print(f"Daily predictions: {len(daily_results)} days")
print(f"Weekly predictions: {len(weekly_results)} weeks")
print(f"Monthly predictions: {len(monthly_results)} months")

# Calculate and print total demand for each timeframe
print("\nTotal Demand:")
print(f"Daily - Actual: {daily_results['Actual'].sum():,.0f}, Predicted: {daily_results['Predicted'].sum():,.0f}")
print(f"Weekly - Actual: {weekly_results['Actual'].sum():,.0f}, Predicted: {weekly_results['Predicted'].sum():,.0f}")
print(f"Monthly - Actual: {monthly_results['Actual'].sum():,.0f}, Predicted: {monthly_results['Predicted'].sum():,.0f}")

def plot_predictions(results, title, freq):
    """Plotting function kept but commented out for reference"""
    pass
    # Print actual dates and values
    print(f"\n{freq} Results:")
    print("Date range:", results['Date'].min(), "to", results['Date'].max())
    print(results[['Date', 'Actual', 'Predicted']].to_string())

def analyze_residuals(results):
    """Residuals analysis function kept but commented out for reference"""
    pass
