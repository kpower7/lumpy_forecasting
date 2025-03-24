# %% [markdown]
# # Enhanced Demand Forecasting using LightGBM with Daily Data
# This script performs demand forecasting using daily demand data combined with monthly economic indicators.

# %% [markdown]
# ## Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# Load daily data
daily_df = pd.read_excel(
    r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx"
)

# Filter for Material_4
material_id = '000161032'  # Material_4
material_no = 'Material_4'
daily_df = daily_df[daily_df['Product ID'] == material_id].copy()

# Convert date column
daily_df['Date'] = pd.to_datetime(daily_df['Date'], format='%d.%m.%Y')
daily_df.set_index('Date', inplace=True)
daily_df.sort_index(inplace=True)

# Load monthly data with external variables
monthly_df = pd.read_csv(f'C:\\Users\\k_pow\\OneDrive\\Documents\\MIT\\MITx SCM\\IAP 2025\\SCC\\Data_files\\{material_id}_{material_no}.csv')
monthly_df['YearMonth'] = pd.to_datetime(monthly_df['YearMonth'])
monthly_df.set_index('YearMonth', inplace=True)

print("Data loaded successfully.")
print("Daily data shape:", daily_df.shape)
print("Monthly data shape:", monthly_df.shape)

# %%
# Create a complete date range from the minimum to the maximum date
full_date_range = pd.date_range(start='2022-01-01', end='2024-12-05', freq='D')

# Reindex the dataframe to include all dates, filling missing values with 0
daily_df = daily_df.reindex(full_date_range, fill_value=0)

# Re-add the static 'Product ID' and 'Product Name' since they remain constant
daily_df['Product ID'] = material_id
daily_df['Product Name'] = material_no

# Re-add Year, Month, Day columns
daily_df['Year'] = daily_df.index.year
daily_df['Month'] = daily_df.index.month
daily_df['Day'] = daily_df.index.day

# Comment out scatter plot
# sns.scatterplot(x=daily_df.index, y=daily_df['Customer Order Quantity'])

# %% [markdown]
# ## Prepare Features

# %%
def prepare_features(data, monthly_data, is_training=True):
    """
    Prepare features combining daily and monthly data.
    is_training: bool, if False will not use future data for feature creation
    """
    features = pd.DataFrame(index=data.index)
    
    # Basic demand lags - for test data, we need to be careful about the boundary
    for lag in [1, 2, 3, 7, 14, 30]:  # Removed longer lags that might leak across train/test boundary
        features[f'Demand_lag_{lag}d'] = data['Dispatched Quantity'].shift(lag)


    # Rolling statistics - using only past data
    for window in [7, 14, 30]:  # Reduced windows to avoid boundary issues
        roll = data['Dispatched Quantity'].rolling(window=window, min_periods=1, closed='left')
        features[f'Demand_roll_mean_{window}d'] = roll.mean()
        features[f'Demand_roll_std_{window}d'] = roll.std()
    
    # External variables - careful with monthly data
    external_vars = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    for var in external_vars:
        if var in monthly_data.columns:
            # Monthly lags - only use data from before current month
            for lag in range(1, 4):  # Reduced lag range to avoid leakage
                lagged = monthly_data[var].shift(lag)
                # For test data, we need to ensure we're not using future information
                if not is_training:
                    lagged = monthly_data[var].shift(lag + 1)  # Shift by (lag+1) in test mode
                else:
                    lagged = monthly_data[var].shift(lag)
                features[f'{var}_lag_{lag}m'] = lagged.reindex(data.index).fillna(method='ffill', limit=30)
    
    # Seasonality features - these are safe as they're based on the date
    features['Month'] = data.index.month
    features['Quarter'] = data.index.quarter
    features['WeekOfYear'] = data.index.isocalendar().week
    features['DayOfWeek'] = data.index.dayofweek
    features['DayOfMonth'] = data.index.day
    
    # Cyclical encoding - also safe as they're based on the date
    features['Month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    features['Month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    features['Day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    features['Day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    
    # Customer order features - only use recent lags
    for lag in [1, 2, 3, 7, 14]:
        features[f'Customer_Order_Lag{lag}'] = data['Customer Order Quantity'].shift(lag)
        if lag <= 7:  # Only compute ratios for short lags to avoid division issues
            ratio = (data['Customer Order Quantity'].shift(lag) / 
                    data['Dispatched Quantity'].shift(lag))
            features[f'Order_Demand_Ratio_Lag{lag}'] = ratio.replace([np.inf, -np.inf], np.nan)
    

    if not is_training:
        features = features.shift(1)  # Extra shift to prevent future info leak

    return features.fillna(0)

# Split data based on date first
train_cutoff = pd.to_datetime('2024-06-01')
train_mask = daily_df.index < train_cutoff

train_data = daily_df[train_mask].copy()
test_data = daily_df[~train_mask].copy()

# Split monthly data using the same cutoff
monthly_train_mask = monthly_df.index < train_cutoff
monthly_train_data = monthly_df[monthly_train_mask].copy()
monthly_test_data = monthly_df[~monthly_train_mask].copy()

# Create validation set from last month of training data
val_cutoff = train_cutoff - pd.DateOffset(months=1)
val_mask = (daily_df.index >= val_cutoff) & (daily_df.index < train_cutoff)
val_data = daily_df[val_mask].copy()
train_data = daily_df[daily_df.index < val_cutoff].copy()

# Split monthly data for validation
monthly_val_data = monthly_df[(monthly_df.index >= val_cutoff) & (monthly_df.index < train_cutoff)].copy()
monthly_train_data = monthly_df[monthly_df.index < val_cutoff].copy()

# Prepare features
X_train = prepare_features(train_data, monthly_train_data, is_training=True)
y_train = train_data['Dispatched Quantity']

X_val = prepare_features(val_data, monthly_val_data, is_training=False)
y_val = val_data['Dispatched Quantity']

X_test = prepare_features(test_data, monthly_test_data, is_training=False)
y_test = test_data['Dispatched Quantity']

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)

# %% [markdown]
# ## Train Model

# %%
# Create LightGBM datasets
train_data_lgb = lgb.Dataset(X_train, label=y_train)
val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

# Set parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Train the model
model = lgb.train(
    params,
    train_data_lgb,
    num_boost_round=100,
    valid_sets=[train_data_lgb, val_data_lgb],
    callbacks=[lgb.early_stopping(10)],  # Use callbacks instead of early_stopping_rounds
    verbose_eval=False
)

print("Model training completed")

# %% [markdown]
# ## Evaluate Model at Multiple Time Scales

# %%
# Make daily predictions
daily_pred = model.predict(X_test)

# Create results DataFrame for daily predictions
test_start = pd.Timestamp('2024-06-01')
test_end = pd.Timestamp('2024-12-05')
daily_results = pd.DataFrame({
    'Date': X_test.index,
    'Actual': y_test,
    'Predicted': daily_pred
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
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    # Calculate R² as the square of correlation coefficient (Excel method)
    r2 = np.corrcoef(actual, predicted)[0, 1] ** 2
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Calculate metrics for daily predictions
daily_metrics = calculate_metrics(daily_results['Actual'], daily_results['Predicted'])
print("\nDaily Metrics:")
print(f"RMSE: {daily_metrics['RMSE']:.2f}")
print(f"MAE: {daily_metrics['MAE']:.2f}")
print(f"R2 (correlation squared): {daily_metrics['R2']:.3f}")

# Calculate metrics for weekly predictions
weekly_metrics = calculate_metrics(weekly_results['Actual'], weekly_results['Predicted'])
print("\nWeekly Metrics:")
print(f"RMSE: {weekly_metrics['RMSE']:.2f}")
print(f"MAE: {weekly_metrics['MAE']:.2f}")
print(f"R2 (correlation squared): {weekly_metrics['R2']:.3f}")

# Calculate metrics for monthly predictions
monthly_metrics = calculate_metrics(monthly_results['Actual'], monthly_results['Predicted'])
print("\nMonthly Metrics:")
print(f"RMSE: {monthly_metrics['RMSE']:.2f}")
print(f"MAE: {monthly_metrics['MAE']:.2f}")
print(f"R2 (correlation squared): {monthly_metrics['R2']:.3f}")

# Save metrics to CSV for verification
metrics_df = pd.DataFrame({
    'Timeframe': ['Daily', 'Weekly', 'Monthly'],
    'RMSE': [daily_metrics['RMSE'], weekly_metrics['RMSE'], monthly_metrics['RMSE']],
    'MAE': [daily_metrics['MAE'], weekly_metrics['MAE'], monthly_metrics['MAE']],
    'R2': [daily_metrics['R2'], weekly_metrics['R2'], monthly_metrics['R2']]
})

output_dir = os.path.join(os.path.dirname(__file__), 'predictions_lgbm')
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

# Get feature importance from LightGBM model
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10))
