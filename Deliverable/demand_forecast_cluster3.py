# %% [markdown]
# # Generic CatBoost Demand Forecasting Method
# This script implements a general-purpose CatBoost-based demand forecasting approach that can be adapted to various material types.

# %% [markdown]
"""
Generic CatBoost Demand Forecasting
================================

This implementation provides a flexible, data-driven approach to demand forecasting
using CatBoost. The model automatically learns patterns
from historical data, considering both time-based features and external variables.

Key Features:
- Automated feature engineering
- Robust handling of multiple input variables
- Configurable hyperparameters
- Comprehensive error metrics
- Built-in cross-validation

Model Characteristics:
- Handles non-linear relationships
- Captures complex interactions between features
- Resistant to overfitting through regularization
- Efficient handling of missing values
- Fast training and prediction

Usage Strategy:
- Suitable for materials with various demand patterns
- Adaptable to different seasonality patterns
- Can incorporate multiple external variables
- Provides feature importance analysis
- Supports both short and long-term forecasting
"""

# %% [markdown]
# ## Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from pathlib import Path

def load_data(material_id: str, material_name: str) -> tuple:
    """
    Load and prepare data for a specific material in Cluster 3 (Generic Model)
    """
    try:
        # Load daily data
        daily_df = pd.read_excel('https://www.dropbox.com/scl/fi/pw5717sy9bsfxru4vggf0/Customer-Order-Quantity_Dispatched-Quantity.xlsx?rlkey=bexjc34bevu4yz3y3t2efciv1&st=e78bctop&dl=1')
        
        # Filter for specific material
        daily_df = daily_df[daily_df['Product ID'] == material_id].copy()
        
        # Convert date column
        daily_df['Date'] = pd.to_datetime(daily_df['Date'], format='%d.%m.%Y')
        daily_df.set_index('Date', inplace=True)
        daily_df.sort_index(inplace=True)
        
        # Load master external variables file
        monthly_df = pd.read_excel('https://www.dropbox.com/scl/fi/z4byjz6aamkutje77pc1b/External_Variables.xlsx?rlkey=qcqat0lddovdcwjvqhv6cz43b&st=1wbhpcy5&dl=1', sheet_name=0)
        
        # Convert DATE column to datetime, handling the 'jan-17' format
        monthly_df['YearMonth'] = pd.to_datetime(monthly_df['DATE'], format='%b-%y')
        monthly_df.set_index('YearMonth', inplace=True)
        monthly_df.drop('DATE', axis=1, inplace=True)  # Remove the original DATE column
        
        return daily_df, monthly_df
            
    except Exception as e:
        print(f"Error loading data for material {material_id}: {str(e)}")
        return None, None

def prepare_features(data, monthly_data, is_training=True):
    """
    Prepare features optimized for Cluster 3 - emphasizing stability and economic indicators
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
            for lag in range(1, 4):  # Reduced lag range to avoid leakage
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
    
    # Cyclical encoding
    features['Month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    features['Month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    features['Day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    features['Day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    
    # Customer order features
    for lag in [1, 2, 3, 7, 14]:
        features[f'Customer_Order_Lag{lag}'] = data['Customer Order Quantity'].shift(lag)
        if lag <= 7:  # Only compute ratios for short lags
            ratio = (data['Customer Order Quantity'].shift(lag) / 
                    data['Dispatched Quantity'].shift(lag)).replace([np.inf, -np.inf], np.nan)
            features[f'Order_Dispatch_Ratio_Lag{lag}'] = ratio

    # For test data, shift all features by 1 to prevent future information leakage
    if not is_training:
        features = features.shift(1)

    return features.fillna(0), data['Dispatched Quantity']

def train_model(X_train, y_train, X_val, y_val):
    """
    Train CatBoost model optimized for Cluster 3
    """
    model = CatBoostRegressor(
        loss_function='RMSE',
        iterations=150,              # Fewer iterations due to sparse demand
        learning_rate=0.02,          # Conservative learning rate for smooth adjustments
        depth=5,                     # Moderate depth to prevent overfitting on rare demand spikes
        min_data_in_leaf=5,          # Higher minimum to smooth fluctuations
        subsample=0.6,               # Reduce overfitting risk
        colsample_bylevel=0.6,       # Reduce noise in economic variables
        l2_leaf_reg=7,               # Strongest regularization to avoid reacting to short-term noise
        random_seed=42,
        verbose=False
    )

    model.fit(
        X_train, 
        y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    return model

def evaluate_model(model, X, y, period='All'):
    """
    Evaluate model performance with multiple metrics
    
    Parameters:
    - model: Trained CatBoost model
    - X: Features to evaluate
    - y: Actual values
    - period: String identifier for the evaluation period
    
    Returns:
    - Dictionary of evaluation metrics
    """
    y_pred = model.predict(X)
    
    return {
        f'{period} R2': r2_score(y, y_pred),
        f'{period} RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        f'{period} MAE': mean_absolute_error(y, y_pred),
        f'{period} MAPE': np.mean(np.abs((y - y_pred) / y)) * 100
    }

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

def plot_monthly_forecast(monthly_df, material_id, material_no, output_path, monthly_metrics):
    """Plot monthly forecast vs actual with R² score"""
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_df.index, monthly_df['Actual'], label='Actual', marker='o')
    plt.plot(monthly_df.index, monthly_df['Predicted'], label='Predicted', marker='o')
    
    # Calculate R² for the plot title
    r2 = monthly_metrics['R2']
    plt.title(f'Monthly Forecast vs Actual for Material {material_id}\nR² = {r2:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to the material directory
    plt.savefig(output_path / 'monthly_forecast_plot.png')
    plt.close()

def main(material_id, material_no, output_path=None):
    """Main function to run the forecasting process"""
    try:
        # Load data
        daily_df, external_df = load_data(material_id, material_no)
        
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
        
        monthly_train_data = external_df[external_df.index < val_cutoff].copy()
        monthly_val_data = external_df[(external_df.index >= val_cutoff) & (external_df.index < train_cutoff)].copy()
        monthly_test_data = external_df[external_df.index >= train_cutoff].copy()
        
        # Prepare features
        X_train, y_train = prepare_features(train_data, monthly_train_data, is_training=True)
        X_val, y_val = prepare_features(val_data, monthly_val_data, is_training=False)
        X_test, y_test = prepare_features(test_data, monthly_test_data, is_training=False)
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val)
        
        # Make predictions
        test_pred = model.predict(X_test)
        
        # Calculate daily metrics
        daily_metrics = calculate_metrics(y_test, test_pred)
        
        # Calculate weekly and monthly aggregates
        weekly_actual = y_test.resample('W').sum()
        weekly_pred = pd.Series(test_pred, index=test_data.index).resample('W').sum()
        
        monthly_actual = y_test.resample('ME').sum()
        monthly_pred = pd.Series(test_pred, index=test_data.index).resample('ME').sum()
        
        # Create DataFrames for results
        results_df = pd.DataFrame({
            'Date': test_data.index,
            'Material': material_id,
            'Material Description': material_no,
            'Actual': y_test,
            'Predicted': test_pred
        })
        
        weekly_df = pd.DataFrame({
            'Date': weekly_actual.index,
            'Material': material_id,
            'Material Description': material_no,
            'Actual': weekly_actual,
            'Predicted': weekly_pred
        })
        
        monthly_df = pd.DataFrame({
            'Actual': monthly_actual,
            'Predicted': monthly_pred
        }, index=monthly_actual.index)
        
        # Calculate weekly metrics
        weekly_metrics = calculate_metrics(weekly_actual, weekly_pred)
        
        # Calculate monthly metrics
        monthly_metrics = calculate_metrics(monthly_actual, monthly_pred)
        
        # Plot monthly forecast
        if output_path is None:
            script_dir = Path(__file__).parent.absolute()
            output_path = script_dir / 'plots' / material_id
            output_path.mkdir(parents=True, exist_ok=True)
        plot_monthly_forecast(monthly_df, material_id, material_no, output_path, monthly_metrics)
        
        # Return both metrics and forecasts
        return {
            'metrics': {
                'Daily R2': daily_metrics['R2'],
                'Daily RMSE': daily_metrics['RMSE'],
                'Daily MAE': daily_metrics['MAE'],
                'Daily MAPE': daily_metrics['MAPE'],
                'Weekly R2': weekly_metrics['R2'],
                'Weekly RMSE': weekly_metrics['RMSE'],
                'Weekly MAE': weekly_metrics['MAE'],
                'Weekly MAPE': weekly_metrics['MAPE'],
                'Monthly R2': monthly_metrics['R2'],
                'Monthly RMSE': monthly_metrics['RMSE'],
                'Monthly MAE': monthly_metrics['MAE'],
                'Monthly MAPE': monthly_metrics['MAPE']
            },
            'forecasts': {
                'daily': results_df,
                'weekly': weekly_df,
                'monthly': monthly_df
            }
        }
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage for a specific material
    material_id = '000161032'
    material_no = 'Material_4'
    results = main(material_id, material_no)
    
    if results:
        metrics = results['metrics']
        # Print metrics for standalone usage
        print("\nDaily Metrics:")
        print(f"R2 (correlation squared): {metrics['Daily R2']:.4f}")
        print(f"RMSE: {metrics['Daily RMSE']:.4f}")
        print(f"MAE: {metrics['Daily MAE']:.4f}")
        print(f"MAPE: {metrics['Daily MAPE']:.4f}%")
        
        print("\nWeekly Metrics:")
        print(f"R2 (correlation squared): {metrics['Weekly R2']:.4f}")
        print(f"RMSE: {metrics['Weekly RMSE']:.4f}")
        print(f"MAE: {metrics['Weekly MAE']:.4f}")
        print(f"MAPE: {metrics['Weekly MAPE']:.4f}%")
        
        print("\nMonthly Metrics:")
        print(f"R2 (correlation squared): {metrics['Monthly R2']:.4f}")
        print(f"RMSE: {metrics['Monthly RMSE']:.4f}")
        print(f"MAE: {metrics['Monthly MAE']:.4f}")
        print(f"MAPE: {metrics['Monthly MAPE']:.4f}%")
