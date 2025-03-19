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
import seaborn as sns   

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
daily_df

# %%
# Create a complete date range from the minimum to the maximum date
full_date_range = pd.date_range(start='2022-01-01', end='2024-12-05', freq='D')

# Reindex the dataframe to include all dates, filling missing values with 0
daily_df = daily_df.reindex(full_date_range, fill_value=0)

# Re-add the static 'Product ID' and 'Product Name' since they remain constant
daily_df['Product ID'] = material_no  
daily_df['Product Name'] = material_id  

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
    
    # Basic demand lags
    for lag in [1, 2, 3, 7, 14, 30, 60, 90, 180, 360, 720]:
        features[f'Demand_lag_{lag}d'] = data['Dispatched Quantity'].shift(lag)
    
    # Rolling statistics - using only past data
    for window in [7, 14, 30, 60, 90]:
        if is_training:
            roll = data['Dispatched Quantity'].rolling(window=window, min_periods=1)
        else:
            roll = data['Dispatched Quantity'].rolling(window=window, min_periods=1, closed='left')
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
                features[f'{var}_lag_{lag}m'] = monthly_data[var].shift(lag).reindex(data.index).ffill()
            
            # Rolling means - using only past data
            if is_training:
                roll3m = monthly_data[var].rolling(window=3, min_periods=1)
                roll6m = monthly_data[var].rolling(window=6, min_periods=1)
            else:
                roll3m = monthly_data[var].rolling(window=3, min_periods=1, closed='left')
                roll6m = monthly_data[var].rolling(window=6, min_periods=1, closed='left')
            features[f'{var}_roll_mean_3m'] = roll3m.mean().reindex(data.index).ffill()
            features[f'{var}_roll_mean_6m'] = roll6m.mean().reindex(data.index).ffill()
    
    # Seasonality features
    features['Month'] = data.index.month
    features['Quarter'] = data.index.quarter
    features['WeekOfYear'] = data.index.isocalendar().week
    features['DayOfWeek'] = data.index.dayofweek
    features['DayOfMonth'] = data.index.day
    
    # Cyclical encoding
    features['Month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    features['Month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    features['Week_sin'] = np.sin(2 * np.pi * data.index.isocalendar().week / 52)
    features['Week_cos'] = np.cos(2 * np.pi * data.index.isocalendar().week / 52)
    features['Day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    features['Day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    
    # Year-over-year features
    features['Demand_SameDayLastYear'] = data['Dispatched Quantity'].shift(365)
    if is_training:
        features['Demand_SameWeekLastYear'] = data['Dispatched Quantity'].shift(364).rolling(7).mean()
        features['Demand_SameMonthLastYear'] = data['Dispatched Quantity'].shift(365).rolling(30).mean()
    else:
        features['Demand_SameWeekLastYear'] = data['Dispatched Quantity'].shift(364).rolling(7, closed='left').mean()
        features['Demand_SameMonthLastYear'] = data['Dispatched Quantity'].shift(365).rolling(30, closed='left').mean()
    
    # Customer order features
    for lag in [1, 2, 3, 7, 14, 30]:
        features[f'Customer_Order_Lag{lag}'] = data['Customer Order Quantity'].shift(lag)
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

# Prepare features separately for train and test using corresponding monthly data
X_train = prepare_features(train_data, monthly_train_data, is_training=True)
y_train = train_data['Dispatched Quantity']

X_test = prepare_features(test_data, monthly_test_data, is_training=False)
y_test = test_data['Dispatched Quantity']

print("Training data shape:", X_train.shape)
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
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("Model training completed")

# %% [markdown]
# ## Evaluate Model at Multiple Time Scales

# %%
# Make daily predictions
daily_pred = model.predict(X_test)

# Create results DataFrame for daily predictions
daily_results = pd.DataFrame({
    'Date': X_test.index,
    'Actual': y_test,
    'Predicted': daily_pred
})

# Aggregate to weekly - exclude partial weeks at start/end
weekly_df = daily_results.copy()
weekly_df['WeekStart'] = weekly_df['Date'].dt.to_period('W-MON').dt.start_time
first_complete_week = weekly_df['WeekStart'].min() + pd.Timedelta(days=7)
last_complete_week = weekly_df['WeekStart'].max() - pd.Timedelta(days=7)
weekly_mask = (weekly_df['WeekStart'] >= first_complete_week) & (weekly_df['WeekStart'] <= last_complete_week)
weekly_df = weekly_df[weekly_mask]

weekly_actual = weekly_df['Actual'].resample('W-MON').sum()
weekly_pred = weekly_df['Predicted'].resample('W-MON').sum()
weekly_results = pd.DataFrame({
    'Date': weekly_actual.index,
    'Actual': weekly_actual,
    'Predicted': weekly_pred
})

# Aggregate to monthly - exclude partial months at start/end
monthly_df = daily_results.copy()
monthly_df['MonthStart'] = monthly_df['Date'].dt.to_period('M').dt.start_time
first_complete_month = monthly_df['MonthStart'].min() + pd.Timedelta(days=32)
first_complete_month = first_complete_month.to_period('M').start_time
last_complete_month = monthly_df['MonthStart'].max() - pd.Timedelta(days=32)
last_complete_month = last_complete_month.to_period('M').start_time
monthly_mask = (monthly_df['MonthStart'] >= first_complete_month) & (monthly_df['MonthStart'] <= last_complete_month)
monthly_df = monthly_df[monthly_mask]

monthly_actual = monthly_df['Actual'].resample('M').sum()
monthly_pred = monthly_df['Predicted'].resample('M').sum()
monthly_results = pd.DataFrame({
    'Date': monthly_actual.index,
    'Actual': monthly_actual,
    'Predicted': monthly_pred
})

print("\nAggregation Info:")
print(f"Daily predictions: {len(daily_results)} days")
print(f"Complete weeks: {len(weekly_results)} weeks")
print(f"Complete months: {len(monthly_results)} months")

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

# Print metrics
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
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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
    eval_set=[(X_test, y_test)],
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
