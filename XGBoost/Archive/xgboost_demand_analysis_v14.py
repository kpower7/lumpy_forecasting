# Monthly Demand Forecasting Analysis
# This version works with monthly aggregated demand data

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Configure matplotlib for better looking plots
plt.style.use('seaborn-v0_8')

def create_features(data, scaler=None, is_training=True):
    """
    Create features for monthly demand forecasting
    """
    features = data.copy()
    
    # Drop non-numeric columns
    features = features.drop(['Product Name', 'Product ID'], axis=1, errors='ignore')
    
    # Ensure date is datetime
    if not isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index)
    
    # Basic time features
    features['Month'] = features.index.month
    features['Month_Sin'] = np.sin(2 * np.pi * features['Month']/12)
    features['Month_Cos'] = np.cos(2 * np.pi * features['Month']/12)
    features['Quarter'] = features.index.quarter
    features['Year'] = features.index.year
    features['IsQuarterStart'] = features.index.is_quarter_start.astype(int)
    features['IsQuarterEnd'] = features.index.is_quarter_end.astype(int)
    features['IsYearStart'] = features.index.is_year_start.astype(int)
    features['IsYearEnd'] = features.index.is_year_end.astype(int)
    
    # Lag features (using only past data)
    for lag in [1, 2, 3, 6, 12]:  # Include yearly seasonality
        features[f'Demand_Lag_{lag}'] = features['Demand'].shift(lag)
        # Add binary indicator for non-zero demand
        features[f'NonZero_Demand_Lag_{lag}'] = (features[f'Demand_Lag_{lag}'] > 0).astype(int)
    
    # Rolling statistics (using only past data)
    for window in [3, 6, 12]:
        # Rolling demand statistics
        features[f'Rolling_Mean_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).mean()
        features[f'Rolling_Std_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).std()
        features[f'Rolling_Max_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).max()
        
        # Non-zero demand statistics
        features[f'NonZero_Mean_{window}m'] = (
            features['Demand']
            .shift(1)
            .where(features['Demand'].shift(1) > 0)
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(0)
        )
        
        # Demand frequency
        features[f'Demand_Frequency_{window}m'] = (
            (features['Demand'].shift(1) > 0)
            .rolling(window=window, min_periods=1)
            .mean()
        )
    
    # Year-over-year features
    features['YoY_Change'] = features['Demand'].pct_change(periods=12)
    features['YoY_Ratio'] = features['Demand'] / features['Demand'].shift(12)
    
    # Drop the target column
    y = features['Demand'].copy()
    features = features.drop('Demand', axis=1)
    
    # Drop rows with NaN values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)  # Fill remaining NaN with 0
    
    # Get feature columns
    feature_cols = features.columns.tolist()
    
    # Scale features if we have data
    if len(features) > 0:
        if is_training:
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for test data")
            features_scaled = scaler.transform(features)
        
        features_scaled = pd.DataFrame(features_scaled, columns=feature_cols, index=features.index)
        return features_scaled, feature_cols, scaler, y
    else:
        return None, None, None, None

def analyze_single_material(material_data, material_name):
    """
    Analyze monthly demand for a single material
    """
    try:
        # Ensure date is datetime and set as index
        material_data['YearMonth'] = pd.to_datetime(material_data['YearMonth'])
        material_data = material_data.set_index('YearMonth')
        material_data = material_data.rename(columns={'Customer Order Quantity': 'Demand'})
        material_data = material_data.sort_index()
        
        print(f"\nAnalyzing material: {material_name}")
        print(f"Data range: {material_data.index.min()} to {material_data.index.max()}")
        
        # Split into train (2022-2023) and test (2024 up to latest actual data)
        train_data = material_data[:'2023-12-31']
        test_data = material_data['2024-01-01':'2024-11-30']  # Only include up to November 2024
        
        # Skip if not enough training data
        if len(train_data) < 12:  # Require at least 12 months of training data
            print(f"Insufficient training data: {len(train_data)} months")
            return None
            
        # Skip if no test data
        if len(test_data) < 1:
            print(f"No test data available")
            return None
        
        print(f"Training data: {len(train_data)} months")
        print(f"Testing data: {len(test_data)} months")
        
        # Create features
        X_train, feature_cols, scaler, y_train = create_features(train_data, is_training=True)
        if X_train is None:
            print("No valid features could be created from training data")
            return None
            
        X_test, _, _, y_test = create_features(test_data, scaler=scaler, is_training=False)
        if X_test is None:
            print("No valid features could be created from test data")
            return None
        
        # Train model with parameters suitable for monthly data
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions for test period and future month
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Ensure no negative predictions
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate adjusted R² that accounts for scale
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate MAPE with protection against zero division
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if np.any(y_test != 0) else np.inf
        
        # Calculate RMSE percentage
        rmse_pct = (rmse / y_test.mean()) * 100 if y_test.mean() != 0 else np.inf
        
        # Calculate feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(importance.head(10))
        
        # Plot actual vs predicted values including December forecast
        plt.figure(figsize=(12, 6))
        
        # Plot actual values
        plt.plot(y_test.index, y_test.values, label='Actual', marker='o', color='blue')
        
        # Plot predictions
        plt.plot(y_test.index, y_pred, label='Predicted', marker='s', color='green')
        
        # Add prediction intervals (using std of residuals)
        residuals = y_test - y_pred
        residual_std = np.std(residuals)
        plt.fill_between(y_test.index, 
                        y_pred - 1.96 * residual_std,
                        y_pred + 1.96 * residual_std,
                        alpha=0.2, color='green',
                        label='95% Prediction Interval')
        
        plt.title(f'Monthly Demand Forecast - Material {material_name}\nR² = {r2:.2f}, RMSE% = {rmse_pct:.1f}%')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'forecast_plots/forecast_plot_{material_name}.png')
        plt.close()
        
        # Save detailed metrics
        metrics_df = pd.DataFrame({
            'Date': y_test.index,
            'Actual': y_test.values,
            'Predicted': y_pred,
            'AbsError': np.abs(y_test - y_pred),
            'PercError': np.abs((y_test - y_pred) / y_test) * 100
        })
        metrics_df.to_csv(f'forecast_metrics/metrics_{material_name}.csv')
        
        return {
            'material_id': material_name,
            'rmse': rmse,
            'rmse_pct': rmse_pct,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'mean_monthly_demand': y_train.mean(),
            'demand_variability': y_train.std() / y_train.mean() if y_train.mean() > 0 else np.nan,
            'test_mean': y_test.mean(),
            'test_std': y_test.std(),
            'prediction_interval': residual_std * 1.96
        }
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def main():
    print("Loading monthly demand data...")
    df = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Order_Quantity_Monthly.csv")
    
    # Create output directories if they don't exist
    os.makedirs('forecast_plots', exist_ok=True)
    os.makedirs('forecast_metrics', exist_ok=True)
    
    print("\nProcessing materials...")
    results = []
    materials = df['Product ID'].unique()
    print(f"Total materials to process: {len(materials)}")
    
    for i, material in enumerate(materials, 1):
        print(f"\nProcessing material {i}/{len(materials)}: {material}")
        material_data = df[df['Product ID'] == material].copy()
        result = analyze_single_material(material_data, material)
        if result is not None:
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Add demand pattern classification
        results_df['demand_pattern'] = pd.cut(
            results_df['demand_variability'],
            bins=[-np.inf, 0.5, 1.0, np.inf],
            labels=['Smooth', 'Erratic', 'Lumpy']
        )
        
        print("\nSummary Statistics:")
        print("===================")
        print("\nModel Performance Metrics:")
        metrics = ['rmse', 'rmse_pct', 'mae', 'r2', 'mape']
        print(results_df[metrics].describe())
        
        print("\nDemand Pattern Distribution:")
        print(results_df['demand_pattern'].value_counts())
        
        print("\nTop 5 Materials by R² Score:")
        print(results_df.nlargest(5, 'r2')[['material_id', 'r2', 'rmse', 'rmse_pct', 'mape', 'demand_pattern']])
        
        print("\nBottom 5 Materials by R² Score:")
        print(results_df.nsmallest(5, 'r2')[['material_id', 'r2', 'rmse', 'rmse_pct', 'mape', 'demand_pattern']])
        
        # Save results
        results_df.to_csv('monthly_forecast_results_v14.csv', index=False)
        print("\nResults saved to monthly_forecast_results_v14.csv")
        
        # Create performance visualization by demand pattern
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='demand_pattern', y='r2', data=results_df)
        plt.title('Model Performance by Demand Pattern')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('performance_by_pattern.png')
        plt.close()
        
    else:
        print("\nNo valid results to analyze")

if __name__ == "__main__":
    main()
