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
import matplotlib.dates as mdates
import seaborn as sns
import os

# Load daily data
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
daily_df = pd.read_excel(
    os.path.join(base_dir, 'Customer Order Quantity_Dispatched Quantity.xlsx')
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
monthly_df = pd.read_csv(os.path.join(base_dir, 'Data_files', f'{material_id}_{material_no}.csv'))
monthly_df['YearMonth'] = pd.to_datetime(monthly_df['YearMonth'])
monthly_df.set_index('YearMonth', inplace=True)

print("Data loaded successfully.")
print("Daily data shape:", daily_df.shape)
print("Monthly data shape:", monthly_df.shape)

# %%
daily_df

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

daily_df.head(60)

# %%
sns.scatterplot(x=daily_df.index, y=daily_df['Customer Order Quantity'])

# %%
monthly_df

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
                    lagged = lagged.shift(1)  # Extra shift for test to ensure we only use known data
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
# ## Split Data and Train Model

# %%
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
    eval_set=[(X_val, y_val)],  # Use validation set instead of test set
    verbose=False
)

print("Model training completed")

# %% [markdown]
# ## Evaluate Model at Multiple Time Scales

# %%
# Make daily predictions
daily_pred = model.predict(X_test)

# Create daily results DataFrame for daily predictions
daily_results = pd.DataFrame({
    'Date': X_test.index,
    'Actual': y_test,
    'Predicted': daily_pred
})

# Export daily results to CSV
output_path = os.path.join(os.path.dirname(__file__), 'daily_predictions_vs_actual.csv')
daily_results.to_csv(output_path, index=False)
print(f"\nExported daily predictions vs actuals to: {output_path}")

# Filter for actual test period
test_start = pd.Timestamp('2024-06-01')  # Changed from July 1st to June 1st
test_end = pd.Timestamp('2024-12-05')
daily_results = daily_results[(daily_results['Date'] >= test_start) & (daily_results['Date'] <= test_end)]

# Add month column for aggregation
daily_results['Year'] = daily_results['Date'].dt.year
daily_results['Month'] = daily_results['Date'].dt.month

# Aggregate to monthly using groupby (this should match Excel's behavior)
monthly_results = daily_results.groupby(['Year', 'Month']).agg({
    'Actual': 'sum',
    'Predicted': 'sum'
}).reset_index()

# Create proper datetime for the first of each month
monthly_results['Date'] = pd.to_datetime(monthly_results[['Year', 'Month']].assign(Day=1))

# Sort by date
monthly_results = monthly_results.sort_values('Date')

# Print the monthly totals for verification
print("\nMonthly Totals:")
print(monthly_results[['Date', 'Actual', 'Predicted']].to_string())

# Print the monthly total sums
total_actual = monthly_results['Actual'].sum()
total_predicted = monthly_results['Predicted'].sum()
print(f"\nMonthly Aggregated Totals:")
print(f"Total Actual: {total_actual:,.0f}")
print(f"Total Predicted: {total_predicted:,.0f}")
print(f"Difference (Predicted - Actual): {total_predicted - total_actual:,.0f}")
print(f"Percentage Error: {((total_predicted - total_actual) / total_actual * 100):.1f}%")

# Aggregate to weekly
weekly_df = daily_results.copy()
weekly_df.set_index('Date', inplace=True)
weekly_results = weekly_df.resample('W-MON').sum()
weekly_results = weekly_results.loc[test_start:test_end]
weekly_results = weekly_results.reset_index()

# Print weekly totals
print("\nWeekly Totals:")
print(weekly_results[['Date', 'Actual', 'Predicted']].to_string())

# Print the weekly total sums
total_actual_weekly = weekly_results['Actual'].sum()
total_predicted_weekly = weekly_results['Predicted'].sum()
print(f"\nWeekly Aggregated Totals:")
print(f"Total Actual: {total_actual_weekly:,.0f}")
print(f"Total Predicted: {total_predicted_weekly:,.0f}")
print(f"Difference (Predicted - Actual): {total_predicted_weekly - total_actual_weekly:,.0f}")
print(f"Percentage Error: {((total_predicted_weekly - total_actual_weekly) / total_actual_weekly * 100):.1f}%")

print("\nAggregation Info:")
print(f"Daily predictions: {len(daily_results)} days")
print(f"Weeks: {len(weekly_results)} weeks")
print(f"Months: {len(monthly_results)} months")

# Calculate metrics for each time scale
def calculate_metrics(actual, pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(actual, pred)),
        'MAE': mean_absolute_error(actual, pred),
        'R2': r2_score(actual, pred)
    }

# Calculate metrics for daily, weekly, and monthly
daily_metrics = calculate_metrics(daily_results['Actual'], daily_results['Predicted'])
weekly_metrics = calculate_metrics(weekly_results['Actual'], weekly_results['Predicted'])
monthly_metrics = calculate_metrics(monthly_results['Actual'], monthly_results['Predicted'])

# Debugging: Print actual vs predicted for daily results
print("\nDaily Actual vs Predicted:")
print(daily_results[['Actual', 'Predicted']].head(10))  # Print first 10 for inspection

# Additional Debugging: Print distribution of actual and predicted values
print("\nDistribution of Actual Values:")
print(daily_results['Actual'].describe())
print("\nDistribution of Predicted Values:")
print(daily_results['Predicted'].describe())

# Print metrics
print("\nModel Performance Metrics:")
print("\nDaily Metrics:")
print(f"RMSE: {daily_metrics['RMSE']:.2f}")
print(f"MAE: {daily_metrics['MAE']:.2f}")
print(f"R2: {daily_metrics['R2']:.3f}")

# Residuals analysis
residuals = daily_results['Actual'] - daily_results['Predicted']
print("\nResiduals Analysis:")
print(residuals.describe())

print("\nWeekly Metrics:")
print(f"RMSE: {weekly_metrics['RMSE']:.2f}")
print(f"MAE: {weekly_metrics['MAE']:.2f}")
print(f"R2: {weekly_metrics['R2']:.3f}")

print("\nMonthly Metrics:")
print(f"RMSE: {monthly_metrics['RMSE']:.2f}")
print(f"MAE: {monthly_metrics['MAE']:.2f}")
print(f"R2: {monthly_metrics['R2']:.3f}")

# %% [markdown]
# ## Visualize Predictions at Different Time Scales

# %%
def plot_predictions(results, title, freq):
    plt.figure(figsize=(12, 6))
    plt.plot(results['Date'], results['Actual'], label='Actual', marker='o')
    plt.plot(results['Date'], results['Predicted'], label='Predicted', marker='s')
    plt.title(f'{material_no}: Actual vs Predicted Demand ({freq})')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    
    # Just rotate the existing dates for readability
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print actual dates and values
    print(f"\n{freq} Results:")
    print("Date range:", results['Date'].min(), "to", results['Date'].max())
    print(results[['Date', 'Actual', 'Predicted']].to_string())

# Plot results at different time scales
plot_predictions(daily_results, 'Daily Predictions', 'Daily')
plot_predictions(weekly_results, 'Weekly Predictions', 'Weekly')
plot_predictions(monthly_results, 'Monthly Predictions', 'Monthly')

# %% [markdown]
# ## Analyze Residuals

# %%
def analyze_residuals(results):
    residuals = results['Actual'] - results['Predicted']
    plt.figure(figsize=(12, 6))
    plt.plot(results['Date'], residuals, label='Residuals', marker='o')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals Analysis')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Analyze residuals for each time scale
analyze_residuals(daily_results)
analyze_residuals(weekly_results)
analyze_residuals(monthly_results)

# %% [markdown]
# ## Simplify the Model (Optional)

# %%
# Train a simpler XGBoost model
simple_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50,  # Reduced number of estimators
    learning_rate=0.1,  # Adjusted learning rate
    max_depth=3,  # Reduced max depth
    min_child_weight=1,  # Adjusted min child weight
    subsample=0.9,  # Adjusted subsample
    colsample_bytree=0.9,  # Adjusted colsample
    random_state=42
)

# Train the simpler model
simple_model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],  # Use validation set instead of test set
    verbose=False
)

# Make predictions with the simpler model
simple_daily_pred = simple_model.predict(X_test)

# Create DataFrames with actual and predicted values for the simpler model
simple_daily_results = pd.DataFrame({
    'Date': X_test.index,
    'Actual': y_test,
    'Predicted': simple_daily_pred
})

# Filter for actual test period
test_start = pd.Timestamp('2024-06-01')  # Changed from July 1st to June 1st
test_end = pd.Timestamp('2024-12-05')
simple_daily_results = simple_daily_results[(simple_daily_results['Date'] >= test_start) & (simple_daily_results['Date'] <= test_end)]

# Calculate metrics for the simpler model
simple_daily_metrics = calculate_metrics(simple_daily_results['Actual'], simple_daily_results['Predicted'])

print("\nSimpler Model Daily Metrics:")
print(f"RMSE: {simple_daily_metrics['RMSE']:.2f}")
print(f"MAE: {simple_daily_metrics['MAE']:.2f}")
print(f"R2: {simple_daily_metrics['R2']:.3f}")

# Plot predictions for the simpler model
plot_predictions(simple_daily_results, 'Daily Predictions (Simpler Model)', 'Daily')

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
