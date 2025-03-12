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

# Load monthly data with external variables
monthly_df = pd.read_csv(r'C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files\0001O1010_Material_51.csv')
monthly_df['YearMonth'] = pd.to_datetime(monthly_df['YearMonth'])
monthly_df.set_index('YearMonth', inplace=True)

print("Data loaded successfully.")
print("Daily data shape:", daily_df.shape)
print("Monthly data shape:", monthly_df.shape)

# %% [markdown]
# ## Prepare Features

# %%
def prepare_features(daily_data, monthly_data):
    """Prepare features combining daily and monthly data."""
    features = pd.DataFrame(index=daily_data.index)
    
    # Basic demand lags
    for lag in [1, 2, 3, 7, 14, 30, 60, 90, 180]:
        features[f'Demand_lag_{lag}d'] = daily_data['Dispatched Quantity'].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30, 60, 90]:
        roll = daily_data['Dispatched Quantity'].rolling(window=window, min_periods=1)
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
            roll3m = monthly_data[var].rolling(window=3, min_periods=1)
            roll6m = monthly_data[var].rolling(window=6, min_periods=1)
            features[f'{var}_roll_mean_3m'] = roll3m.mean().reindex(daily_data.index).ffill()
            features[f'{var}_roll_mean_6m'] = roll6m.mean().reindex(daily_data.index).ffill()
    
    # Seasonality features
    features['Month'] = daily_data.index.month
    features['Quarter'] = daily_data.index.quarter
    features['WeekOfYear'] = daily_data.index.isocalendar().week
    features['DayOfWeek'] = daily_data.index.dayofweek
    features['DayOfMonth'] = daily_data.index.day
    
    # Cyclical encoding
    features['Month_sin'] = np.sin(2 * np.pi * daily_data.index.month / 12)
    features['Month_cos'] = np.cos(2 * np.pi * daily_data.index.month / 12)
    features['Week_sin'] = np.sin(2 * np.pi * daily_data.index.isocalendar().week / 52)
    features['Week_cos'] = np.cos(2 * np.pi * daily_data.index.isocalendar().week / 52)
    features['Day_sin'] = np.sin(2 * np.pi * daily_data.index.dayofweek / 7)
    features['Day_cos'] = np.cos(2 * np.pi * daily_data.index.dayofweek / 7)
    
    # Year-over-year features
    features['Demand_SameDayLastYear'] = daily_data['Dispatched Quantity'].shift(365)
    features['Demand_SameWeekLastYear'] = daily_data['Dispatched Quantity'].shift(364).rolling(7).mean()
    features['Demand_SameMonthLastYear'] = daily_data['Dispatched Quantity'].shift(365).rolling(30).mean()
    
    # Customer order features
    for lag in [1, 2, 3, 7, 14, 30]:
        features[f'Customer_Order_Lag{lag}'] = daily_data['Customer Order Quantity'].shift(lag)
        ratio = (daily_data['Customer Order Quantity'].shift(lag) / 
                daily_data['Dispatched Quantity'].shift(lag))
        features[f'Order_Demand_Ratio_Lag{lag}'] = ratio.replace([np.inf, -np.inf], np.nan)
    
    return features.fillna(0)

# Prepare features
features_df = prepare_features(daily_df, monthly_df)
target = daily_df['Dispatched Quantity']

# %% [markdown]
# ## Split Data and Train Model

# %%
# Split data based on date
train_cutoff = pd.to_datetime('2024-06-01')
train_mask = features_df.index < train_cutoff

X_train = features_df[train_mask]
y_train = target[train_mask]
X_test = features_df[~train_mask]
y_test = target[~train_mask]

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
# Make daily predictions
daily_pred = model.predict(X_test)

# Create DataFrames with actual and predicted values at different scales
daily_results = pd.DataFrame({
    'Date': X_test.index,
    'Actual': y_test,
    'Predicted': daily_pred
})

# Aggregate to weekly
weekly_actual = y_test.resample('W-MON').mean()
weekly_pred = pd.Series(daily_pred, index=X_test.index).resample('W-MON').mean()
weekly_results = pd.DataFrame({
    'Date': weekly_actual.index,
    'Actual': weekly_actual,
    'Predicted': weekly_pred
})

# Aggregate to monthly
monthly_actual = y_test.resample('MS').mean()
monthly_pred = pd.Series(daily_pred, index=X_test.index).resample('MS').mean()
monthly_results = pd.DataFrame({
    'Date': monthly_actual.index,
    'Actual': monthly_actual,
    'Predicted': monthly_pred
})

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

# %% [markdown]
# ## Feature Importance Analysis

# %%
# Get feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 6))
plt.barh(importance.head(20)['feature'], importance.head(20)['importance'])
plt.title('Top 20 Most Important Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(importance.head(10))
