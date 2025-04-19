# %% [markdown]
# # Enhanced Demand Forecasting using XGBoost for Cluster 3 (Volatile Products)
# This script is optimized for products with highly volatile demand patterns and strong market sensitivity.

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
    Prepare features optimized for Cluster 3 - emphasizing volatility and rapid changes
    """
    features = pd.DataFrame(index=data.index)
    
    # Very recent demand patterns
    for lag in [1, 2, 3, 5]:  # Very short lags for volatile patterns
        features[f'Demand_lag_{lag}d'] = data['Dispatched Quantity'].shift(lag)
    
    # Short-term volatility features
    for window in [3, 5, 7, 14]:  # Short windows for capturing rapid changes
        roll = data['Dispatched Quantity'].rolling(window=window, min_periods=1)
        features[f'Demand_mean_{window}d'] = roll.mean()
        features[f'Demand_std_{window}d'] = roll.std()
        features[f'Demand_max_{window}d'] = roll.max()
        features[f'Demand_min_{window}d'] = roll.min()
        
        # Volatility indicators
        features[f'Demand_range_{window}d'] = features[f'Demand_max_{window}d'] - features[f'Demand_min_{window}d']
        features[f'Demand_cv_{window}d'] = features[f'Demand_std_{window}d'] / (features[f'Demand_mean_{window}d'] + 1e-6)
    
    # Order volatility features
    if 'Customer Order Quantity' in data.columns:
        for window in [3, 5, 7]:
            roll_order = data['Customer Order Quantity'].rolling(window=window, min_periods=1)
            features[f'Order_mean_{window}d'] = roll_order.mean()
            features[f'Order_std_{window}d'] = roll_order.std()
            features[f'Order_range_{window}d'] = roll_order.max() - roll_order.min()
    
    # Market sensitivity features (monthly data)
    if monthly_data is not None:
        market_vars = ['ECG_DESP', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19']
        for var in market_vars:
            if var in monthly_data.columns:
                # Current value
                features[var] = monthly_data[var].reindex(data.index, method='ffill')
                
                # Short-term changes
                features[f'{var}_1m_change'] = monthly_data[var].diff().reindex(data.index, method='ffill')
                features[f'{var}_2m_change'] = monthly_data[var].diff(2).reindex(data.index, method='ffill')
                
                # Volatility
                roll_market = monthly_data[var].rolling(window=3)
                features[f'{var}_volatility'] = roll_market.std().reindex(data.index, method='ffill')
    
    # Enhanced seasonality features
    features['Month'] = data.index.month
    features['Quarter'] = data.index.quarter
    features['DayOfWeek'] = data.index.dayofweek
    
    # Demand shock indicators
    mean_demand = data['Dispatched Quantity'].rolling(30, min_periods=1).mean()
    std_demand = data['Dispatched Quantity'].rolling(30, min_periods=1).std()
    features['Demand_Shock'] = ((data['Dispatched Quantity'] - mean_demand) / (std_demand + 1e-6)).abs()
    
    if is_training:
        return features.dropna()
    return features

def train_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model optimized for Cluster 3 - Volatile Patterns
    """
    model = xgb.XGBRegressor(
        n_estimators=300,  # Increased for better pattern capture
        learning_rate=0.1,  # Higher learning rate for rapid adaptation
        max_depth=7,  # Deeper trees for complex patterns
        min_child_weight=1,  # Lower to capture rare but important patterns
        subsample=0.9,  # Higher subsample for better pattern capture
        colsample_bytree=0.9,
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
    material_id = '000161032'  # Replace with actual material ID from Cluster 3
    material_no = 'Material_4'  # Replace with actual material number
    model, metrics = main(material_id, material_no)
