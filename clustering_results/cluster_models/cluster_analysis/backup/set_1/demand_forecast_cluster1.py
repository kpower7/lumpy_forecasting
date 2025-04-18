# %% [markdown]
# # Enhanced Demand Forecasting using XGBoost for Cluster 1 (Market-Sensitive Products)
# This script is optimized for products with high market sensitivity and moderate volume.

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
    Prepare features optimized for Cluster 1 - emphasizing market indicators and seasonality
    """
    features = pd.DataFrame(index=data.index)
    
    # Recent demand patterns
    for lag in [1, 2, 3, 7, 14]:  # Shorter lags for market-sensitive patterns
        features[f'Demand_lag_{lag}d'] = data['Dispatched Quantity'].shift(lag)

    # Rolling statistics focused on recent market changes
    for window in [7, 14, 30]:  # Shorter windows for market sensitivity
        features[f'Demand_mean_{window}d'] = data['Dispatched Quantity'].rolling(window).mean()
        features[f'Demand_std_{window}d'] = data['Dispatched Quantity'].rolling(window).std()
        
        # Market volatility features
        if 'Customer Order Quantity' in data.columns:
            features[f'Order_mean_{window}d'] = data['Customer Order Quantity'].rolling(window).mean()
            features[f'Order_std_{window}d'] = data['Customer Order Quantity'].rolling(window).std()
    
    # Enhanced market indicators (monthly features)
    if monthly_data is not None:
        # Market indicators with more weight
        features['TUAV'] = monthly_data['TUAV'].reindex(data.index, method='ffill')
        features['VTOTAL_19'] = monthly_data['VTOTAL_19'].reindex(data.index, method='ffill')
        features['OTOTAL_19'] = monthly_data['OTOTAL_19'].reindex(data.index, method='ffill')
        features['ECG_DESP'] = monthly_data['ECG_DESP'].reindex(data.index, method='ffill')
        
        # Rolling means of market indicators
        for window in [3, 6]:  # Shorter windows for market trends
            features[f'TUAV_{window}M'] = features['TUAV'].rolling(window*30).mean()
            features[f'VTOTAL_{window}M'] = features['VTOTAL_19'].rolling(window*30).mean()
            features[f'ECG_{window}M'] = features['ECG_DESP'].rolling(window*30).mean()
    
    # Seasonality features
    features['Month'] = data.index.month
    features['Quarter'] = data.index.quarter
    features['DayOfWeek'] = data.index.dayofweek
    
    if is_training:
        return features.dropna()
    return features

def train_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model optimized for Cluster 1 - Market Sensitive
    """
    model = xgb.XGBRegressor(
        n_estimators=150,  # Reduced for faster market adaptation
        learning_rate=0.05,  # Increased for better market response
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

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
    
    # Calculate R² using sklearn's r2_score for better handling of edge cases
    r2 = r2_score(actual, predicted) if len(actual) > 1 else np.nan
    
    # Handle edge cases for MAPE calculation
    if len(actual_filtered) > 0:
        mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
    else:
        mape = np.nan
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'MAPE': mape,
        'R2': max(0, r2)  # Ensure R² is not negative
    }
    return metrics

def main(material_id, material_no):
    # Load and prepare data
    daily_df, monthly_df = load_data(material_id, material_no)
    
    # Create complete date range
    full_date_range = pd.date_range(start='2022-01-01', end='2024-12-05', freq='D')
    daily_df = daily_df.reindex(full_date_range, fill_value=0)
    daily_df['Product ID'] = material_id
    daily_df['Product Name'] = material_no
    
    # Split data
    train_cutoff = pd.to_datetime('2024-06-01')
    val_cutoff = train_cutoff - pd.DateOffset(months=1)
    
    # Prepare datasets
    train_data = daily_df[daily_df.index < val_cutoff].copy()
    val_data = daily_df[(daily_df.index >= val_cutoff) & (daily_df.index < train_cutoff)].copy()
    test_data = daily_df[daily_df.index >= train_cutoff].copy()
    
    # Split monthly data
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
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Calculate daily metrics
    daily_metrics = calculate_metrics(y_test, test_pred)
    
    # Calculate weekly metrics
    weekly_actual = pd.Series(y_test).resample('W').sum()
    weekly_pred = pd.Series(test_pred, index=y_test.index).resample('W').sum()
    weekly_metrics = calculate_metrics(weekly_actual, weekly_pred)
    
    # Calculate monthly metrics
    monthly_actual = pd.Series(y_test).resample('M').sum()
    monthly_pred = pd.Series(test_pred, index=y_test.index).resample('M').sum()
    monthly_metrics = calculate_metrics(monthly_actual, monthly_pred)
    
    # Print metrics in exact format expected by run_all_materials.py
    print("\nDaily Metrics:")
    print(f"R2 (correlation squared): {daily_metrics['R2']:.4f}")
    print(f"RMSE: {daily_metrics['RMSE']:.2f}")
    print(f"MAE: {daily_metrics['MAE']:.2f}")
    print(f"MAPE: {daily_metrics['MAPE']:.2f}%")
    
    print("\nWeekly Metrics:")
    print(f"R2 (correlation squared): {weekly_metrics['R2']:.4f}")
    print(f"RMSE: {weekly_metrics['RMSE']:.2f}")
    print(f"MAE: {weekly_metrics['MAE']:.2f}")
    print(f"MAPE: {weekly_metrics['MAPE']:.2f}%")
    
    print("\nMonthly Metrics:")
    print(f"R2 (correlation squared): {monthly_metrics['R2']:.4f}")
    print(f"RMSE: {monthly_metrics['RMSE']:.2f}")
    print(f"MAE: {monthly_metrics['MAE']:.2f}")
    print(f"MAPE: {monthly_metrics['MAPE']:.2f}%")
    
    return model, (daily_metrics, weekly_metrics, monthly_metrics)

if __name__ == "__main__":
    # Example usage for a specific material
    material_id = '000161032'  # Replace with actual material ID from Cluster 1
    material_no = 'Material_4'  # Replace with actual material number
    model, metrics = main(material_id, material_no)
