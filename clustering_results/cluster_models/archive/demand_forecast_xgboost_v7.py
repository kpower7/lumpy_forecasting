# %% [markdown]
# # Enhanced Demand Forecasting using XGBoost with Daily Data
# This script performs demand forecasting using daily demand data combined with monthly economic indicators.

# %% [markdown]
# ## Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt

# Load daily data
daily_df = pd.read_excel(
    r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx"
)

# Filter for Material_51
material_id = '000161032'  # Material_51
daily_df = daily_df[daily_df['Product ID'] == material_id].copy()

# Convert date column
daily_df['Date'] = pd.to_datetime(daily_df['Date'], format='%d.%m.%Y')
daily_df.set_index('Date', inplace=True)
daily_df.sort_index(inplace=True)

# Print date range information
print("\nDaily data date range:")
print("Start date:", daily_df.index.min())
print("End date:", daily_df.index.max())

# Load monthly data with external variables
monthly_df = pd.read_csv(r'C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files\0001O1010_Material_51.csv')
monthly_df['YearMonth'] = pd.to_datetime(monthly_df['YearMonth'])
monthly_df.set_index('YearMonth', inplace=True)

# Print monthly data range
print("\nMonthly data date range:")
print("Start date:", monthly_df.index.min())
print("End date:", monthly_df.index.max())

print("\nData loaded successfully.")
print("Daily data shape:", daily_df.shape)
print("Monthly data shape:", monthly_df.shape)

# %% [markdown]
# ## Split Data and Apply Feature Engineering

# %%
# First split the raw data based on date
train_cutoff = pd.to_datetime('2024-06-01')

# Create train and test masks
train_mask = daily_df.index < train_cutoff
test_mask = daily_df.index >= train_cutoff

# Split the data
daily_train = daily_df[train_mask].copy()
daily_test = daily_df[test_mask].copy()
monthly_train = monthly_df[monthly_df.index < train_cutoff].copy()
monthly_test = monthly_df[monthly_df.index >= train_cutoff].copy()

def prepare_features(daily_data, monthly_data, is_training=True):
    """Prepare features combining daily and monthly data."""
    features = pd.DataFrame(index=daily_data.index)
    
    # Basic demand lags
    for lag in [1, 2, 3, 7, 14, 30, 60, 90, 180]:
        features[f'Demand_lag_{lag}d'] = daily_data['Dispatched Quantity'].shift(lag)
    
    # Rolling statistics - only use past data
    for window in [7, 14, 30, 60, 90]:
        if is_training:
            roll = daily_data['Dispatched Quantity'].rolling(window=window, min_periods=1)
        else:
            roll = daily_data['Dispatched Quantity'].rolling(window=window, min_periods=1, closed='left')
        features[f'Demand_roll_mean_{window}d'] = roll.mean()
        features[f'Demand_roll_std_{window}d'] = roll.std()
        features[f'Demand_roll_max_{window}d'] = roll.max()
        features[f'Demand_roll_min_{window}d'] = roll.min()
    
    # External variables
    external_vars = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    for var in external_vars:
        if var in monthly_data.columns:
            # Monthly lags
            for lag in range(1, 7):
                features[f'{var}_lag_{lag}m'] = monthly_data[var].shift(lag).reindex(daily_data.index).ffill()
            
            # Rolling means
            if is_training:
                roll3m = monthly_data[var].rolling(window=3, min_periods=1)
                roll6m = monthly_data[var].rolling(window=6, min_periods=1)
            else:
                roll3m = monthly_data[var].rolling(window=3, min_periods=1, closed='left')
                roll6m = monthly_data[var].rolling(window=6, min_periods=1, closed='left')
            features[f'{var}_roll_mean_3m'] = roll3m.mean().reindex(daily_data.index).ffill()
            features[f'{var}_roll_mean_6m'] = roll6m.mean().reindex(daily_data.index).ffill()
    
    # Seasonality features (these don't cause leakage as they're based on date only)
    features['Month'] = daily_data.index.month
    features['Quarter'] = daily_data.index.quarter
    features['WeekOfYear'] = daily_data.index.isocalendar().week
    features['DayOfWeek'] = daily_data.index.dayofweek
    features['DayOfMonth'] = daily_data.index.day
    
    # Cyclical encoding (no leakage - based on date only)
    features['Month_sin'] = np.sin(2 * np.pi * daily_data.index.month / 12)
    features['Month_cos'] = np.cos(2 * np.pi * daily_data.index.month / 12)
    features['Week_sin'] = np.sin(2 * np.pi * daily_data.index.isocalendar().week / 52)
    features['Week_cos'] = np.cos(2 * np.pi * daily_data.index.isocalendar().week / 52)
    features['Day_sin'] = np.sin(2 * np.pi * daily_data.index.dayofweek / 7)
    features['Day_cos'] = np.cos(2 * np.pi * daily_data.index.dayofweek / 7)
    
    # Customer order features
    for lag in [1, 2, 3, 7, 14, 30]:
        features[f'Customer_Order_Lag{lag}'] = daily_data['Customer Order Quantity'].shift(lag)
        ratio = (daily_data['Customer Order Quantity'].shift(lag) / 
                daily_data['Dispatched Quantity'].shift(lag))
        features[f'Order_Demand_Ratio_Lag{lag}'] = ratio.replace([np.inf, -np.inf], np.nan)
    
    return features.fillna(0)

# Prepare features separately for train and test
X_train = prepare_features(daily_train, monthly_train, is_training=True)
y_train = daily_train['Dispatched Quantity']

X_test = prepare_features(daily_test, monthly_test, is_training=False)
y_test = daily_test['Dispatched Quantity']

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Train XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("Model training completed")

# %% [markdown]
# ## Evaluate Model at Multiple Time Scales

# %%
# Make predictions for both train and test sets
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Create Series with datetime index
train_dates = X_train.index
test_dates = X_test.index
train_actuals = pd.Series(y_train, index=train_dates)
test_actuals = pd.Series(y_test, index=test_dates)
train_predictions = pd.Series(train_pred, index=train_dates)
test_predictions = pd.Series(test_pred, index=test_dates)

# Combine all data
all_actuals = pd.concat([train_actuals, test_actuals]).sort_index()
all_predictions = pd.concat([train_predictions, test_predictions]).sort_index()

# Create daily results DataFrame
daily_results = pd.DataFrame({
    'Date': all_actuals.index,
    'Actual': all_actuals.values,
    'Predicted': all_predictions.values
})

# Create weekly results
weekly_results = pd.DataFrame({
    'Date': pd.date_range(start=daily_results['Date'].min(), end=daily_results['Date'].max(), freq='W-MON'),
})
weekly_results['Actual'] = all_actuals.resample('W-MON').mean()
weekly_results['Predicted'] = all_predictions.resample('W-MON').mean()
weekly_results = weekly_results.dropna()

# Create monthly results
monthly_results = pd.DataFrame({
    'Date': pd.date_range(start=daily_results['Date'].min(), end=daily_results['Date'].max(), freq='MS'),
})
monthly_results['Actual'] = all_actuals.resample('MS').mean()
monthly_results['Predicted'] = all_predictions.resample('MS').mean()
monthly_results = monthly_results.dropna()

# Calculate metrics for each time scale
def calculate_metrics(actual, pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(actual, pred)),
        'MAE': mean_absolute_error(actual, pred),
        'R2': r2_score(actual, pred)
    }

daily_metrics = calculate_metrics(daily_results['Actual'], daily_results['Predicted'])
weekly_metrics = calculate_metrics(weekly_results['Actual'], weekly_results['Predicted'])
monthly_metrics = calculate_metrics(monthly_results['Actual'], monthly_results['Predicted'])

print("\nModel Performance Metrics:")
print("\nDaily Metrics:")
print(f"RMSE: {daily_metrics['RMSE']:.2f}")
print(f"MAE: {daily_metrics['MAE']:.2f}")
print(f"R2: {daily_metrics['R2']:.3f}")

print("\nWeekly Metrics:")
print(f"RMSE: {weekly_metrics['RMSE']:.2f}")
print(f"MAE: {weekly_metrics['MAE']:.2f}")
print(f"R2: {weekly_metrics['R2']:.3f}")

print("\nMonthly Metrics:")
print(f"RMSE: {monthly_metrics['RMSE']:.2f}")
print(f"MAE: {monthly_metrics['MAE']:.2f}")
print(f"R2: {monthly_metrics['R2']:.3f}")

# %% [markdown]
# ## Visualize Results at Different Time Scales

# %%
def plot_predictions(results, title, freq):
    plt.figure(figsize=(12, 6))
    plt.plot(results['Date'], results['Actual'], label='Actual', marker='o')
    plt.plot(results['Date'], results['Predicted'], label='Predicted', marker='s')
    
    # Add vertical line at train-test split
    plt.axvline(x=pd.to_datetime('2024-06-01'), color='r', linestyle='--', label='Train-Test Split')
    
    plt.title(f'Material 51: Actual vs Predicted Demand ({freq})')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot results at different time scales
plot_predictions(daily_results, 'Daily Predictions', 'Daily')
plot_predictions(weekly_results, 'Weekly Predictions', 'Weekly')
plot_predictions(monthly_results, 'Monthly Predictions', 'Monthly')
