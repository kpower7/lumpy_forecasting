# %% [markdown]
# # Enhanced Demand Forecasting using XGBoost for Cluster 1 (Market-Sensitive Products)
# This script is optimized for products with high market sensitivity and moderate volume.

# %% [markdown]
# ## Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

def load_data(material_id, material_no):
    # Load daily data
    daily_df = pd.read_excel(
        r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx"
    )

    # Filter for specific material
    daily_df = daily_df[daily_df['Product ID'] == material_id].copy()

    # Convert date column
    daily_df['Date'] = pd.to_datetime(daily_df['Date'], format='%d.%m.%Y')
    daily_df.set_index('Date', inplace=True)
    daily_df.sort_index(inplace=True)

    # Load monthly data with external variables
    monthly_df = pd.read_csv(f'C:\\Users\\k_pow\\OneDrive\\Documents\\MIT\\MITx SCM\\IAP 2025\\SCC\\Data_files\\{material_id}_{material_no}.csv')
    monthly_df['YearMonth'] = pd.to_datetime(monthly_df['YearMonth'])
    monthly_df.set_index('YearMonth', inplace=True)

    return daily_df, monthly_df

def prepare_features(data, monthly_data, is_training=True):
    """
    Prepare features optimized for Cluster 1 - Market Sensitive Products
    """
    features = pd.DataFrame(index=data.index)
    
    # Ensure data is properly sorted by date
    data = data.sort_index()
    
    # Basic demand patterns with focus on market sensitivity
    for lag in [1, 2, 3, 7, 14]:  # Shorter lags for market-sensitive patterns
        features[f'Demand_lag_{lag}d'] = data['Dispatched Quantity'].shift(lag)
    
    # Market-focused rolling statistics
    for window in [7, 14, 30]:
        roll = data['Dispatched Quantity'].rolling(window=window, min_periods=1)
        features[f'Demand_mean_{window}d'] = roll.mean()
        features[f'Demand_std_{window}d'] = roll.std()
        
    # Enhanced market indicators (critical for Cluster 1)
    market_vars = ['TUAV', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    if monthly_data is not None:
        for var in market_vars:
            if var in monthly_data.columns:
                # Current and lagged values
                for lag in range(0, 3):  # Include current month
                    lagged = monthly_data[var].shift(lag)
                    if not is_training:
                        lagged = monthly_data[var].shift(lag + 1)
                    features[f'{var}_lag_{lag}m'] = lagged.reindex(data.index, method='ffill')
                
                # Market momentum (month-over-month changes)
                features[f'{var}_mom'] = monthly_data[var].diff().reindex(data.index, method='ffill')
                features[f'{var}_mom_3m'] = monthly_data[var].diff(3).reindex(data.index, method='ffill')
    
    # Enhanced seasonality features
    features['Month'] = data.index.month
    features['Quarter'] = data.index.quarter
    features['DayOfWeek'] = data.index.dayofweek
    
    # Cyclical encoding for better seasonal patterns
    features['Month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    features['Month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    features['Day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    features['Day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    
    # Order vs Demand features
    if 'Customer Order Quantity' in data.columns:
        for lag in [1, 2, 3, 7]:
            features[f'Order_lag_{lag}d'] = data['Customer Order Quantity'].shift(lag)
            ratio = (data['Customer Order Quantity'].shift(lag) / 
                    (data['Dispatched Quantity'].shift(lag) + 1e-6))  # Add epsilon to avoid division by zero
            features[f'Order_Demand_Ratio_{lag}d'] = ratio.replace([np.inf, -np.inf], np.nan)
    
    if not is_training:
        features = features.shift(1)  # Ensure no future data leakage
    
    # Fill NaN values with 0 and ensure all features are numeric
    features = features.fillna(0)
    return features.astype(float)

def train_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model optimized for Cluster 1 - Market Sensitive
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,  # Increased for better market pattern capture
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Ensure inputs are properly formatted
    X_train = X_train.astype(float)
    y_train = y_train.astype(float)
    X_val = X_val.astype(float)
    y_val = y_val.astype(float)
    
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

def calculate_metrics(actual, predicted):
    """Calculate forecast accuracy metrics"""
    # Remove any zero values to avoid division by zero in MAPE
    mask = actual != 0
    actual_filtered = actual[mask]
    predicted_filtered = predicted[mask]
    
    # Calculate R² as the square of correlation coefficient (Excel method)
    r2 = np.corrcoef(actual, predicted)[0, 1] ** 2
    
    # Handle edge cases for MAPE calculation
    if len(actual_filtered) > 0:
        mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
    else:
        mape = np.nan
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'MAPE': mape,
        'R2': r2
    }
    return metrics

def main(material_id, material_no):
    """Main function to run the forecasting process for market-sensitive products"""
    # Load data
    daily_df, monthly_df = load_data(material_id, material_no)
    
    # Create a complete date range
    full_date_range = pd.date_range(start='2022-01-01', end='2024-12-05', freq='D')
    daily_df = daily_df.reindex(full_date_range, fill_value=0)
    
    # Re-add the static columns
    daily_df['Product ID'] = material_id
    daily_df['Product Name'] = material_no
    daily_df['Year'] = daily_df.index.year
    daily_df['Month'] = daily_df.index.month
    daily_df['Day'] = daily_df.index.day
    
    # Split data
    train_cutoff = pd.to_datetime('2024-06-01')
    val_cutoff = train_cutoff - pd.DateOffset(months=1)
    
    # Create train/val/test splits
    train_data = daily_df[daily_df.index < val_cutoff].copy()
    val_data = daily_df[(daily_df.index >= val_cutoff) & (daily_df.index < train_cutoff)].copy()
    test_data = daily_df[daily_df.index >= train_cutoff].copy()
    
    monthly_train_data = monthly_df[monthly_df.index < val_cutoff].copy()
    monthly_val_data = monthly_df[(monthly_df.index >= val_cutoff) & (monthly_df.index < train_cutoff)].copy()
    monthly_test_data = monthly_df[monthly_df.index >= train_cutoff].copy()
    
    # Prepare features
    X_train = prepare_features(train_data, monthly_train_data, is_training=True)
    y_train = train_data['Dispatched Quantity']
    
    X_val = prepare_features(val_data, monthly_val_data, is_training=False)
    y_val = val_data['Dispatched Quantity']
    
    X_test = prepare_features(test_data, monthly_test_data, is_training=False)
    y_test = test_data['Dispatched Quantity']
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Make predictions
    daily_pred = model.predict(X_test)
    
    # Create results DataFrame
    daily_results = pd.DataFrame({
        'Date': X_test.index,
        'Actual': y_test,
        'Predicted': daily_pred
    })
    
    # Filter for actual test period
    test_start = pd.Timestamp('2024-06-01')
    test_end = pd.Timestamp('2024-12-05')
    daily_results = daily_results[(daily_results['Date'] >= test_start) & (daily_results['Date'] <= test_end)]
    
    # Create weekly and monthly aggregations
    weekly_results = daily_results.set_index('Date').resample('W-MON').sum().reset_index()
    monthly_results = daily_results.set_index('Date').resample('MS').sum().reset_index()
    
    # Calculate metrics
    metrics = {}
    metrics['Daily'] = calculate_metrics(daily_results['Actual'], daily_results['Predicted'])
    metrics['Weekly'] = calculate_metrics(weekly_results['Actual'], weekly_results['Predicted'])
    metrics['Monthly'] = calculate_metrics(monthly_results['Actual'], monthly_results['Predicted'])
    
    # Print metrics
    for timeframe in ['Daily', 'Weekly', 'Monthly']:
        print(f"\n{timeframe} Metrics:")
        print(f"RMSE: {metrics[timeframe]['RMSE']:.2f}")
        print(f"MAE: {metrics[timeframe]['MAE']:.2f}")
        print(f"R2 (correlation squared): {metrics[timeframe]['R2']:.3f}")
        print(f"MAPE: {metrics[timeframe]['MAPE']:.2f}%")
    
    return model, metrics

if __name__ == "__main__":
    # Example usage for a specific material
    material_id = '000161032'  # Replace with actual material ID
    material_no = 'Material_4'  # Replace with actual material number
    model, metrics = main(material_id, material_no)
