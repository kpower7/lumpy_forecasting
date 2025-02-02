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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings('ignore')

# Configure matplotlib for better looking plots
plt.style.use('seaborn-v0_8')

def is_smooth_demand(data, threshold_cv=0.5, min_nonzero_ratio=0.7):
    """
    Determine if a demand pattern is smooth based on:
    1. Coefficient of variation (CV)
    2. Ratio of non-zero demand periods
    """
    demand = data['Demand'].values
    nonzero_demand = demand[demand > 0]
    
    if len(nonzero_demand) == 0:
        return False
    
    # Calculate CV (standard deviation / mean)
    cv = np.std(nonzero_demand) / np.mean(nonzero_demand)
    
    # Calculate ratio of non-zero demand periods
    nonzero_ratio = len(nonzero_demand) / len(demand)
    
    return cv <= threshold_cv and nonzero_ratio >= min_nonzero_ratio

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
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        features[f'Demand_Lag_{lag}'] = features['Demand'].shift(lag)
        features[f'NonZero_Demand_Lag_{lag}'] = (features[f'Demand_Lag_{lag}'] > 0).astype(int)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        features[f'Rolling_Mean_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).mean()
        features[f'Rolling_Std_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).std()
        features[f'Rolling_Max_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).max()
        features[f'NonZero_Mean_{window}m'] = (
            features['Demand']
            .shift(1)
            .where(features['Demand'].shift(1) > 0)
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(0)
        )
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
    features = features.fillna(0)
    
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
    Analyze monthly demand for a single material using either XGBoost or Holt-Winters
    based on the demand pattern
    """
    try:
        # Create output directories if they don't exist
        os.makedirs('forecast_plots', exist_ok=True)
        os.makedirs('forecast_metrics', exist_ok=True)
        
        # Determine if the demand pattern is smooth
        smooth_demand = is_smooth_demand(material_data)
        
        # Split data into train and test sets (last 12 months for test)
        train_data = material_data[:-12].copy()
        test_data = material_data[-12:].copy()
        
        if smooth_demand and len(train_data) >= 24:  # Need at least 2 years of data for seasonal analysis
            # Use Holt-Winters for smooth demand
            model = ExponentialSmoothing(
                train_data['Demand'],
                seasonal_periods=12,
                trend='add',
                seasonal='add',
                damped_trend=True
            )
            fitted_model = model.fit(optimized=True)
            
            # Make predictions
            train_pred = fitted_model.fittedvalues
            test_pred = fitted_model.forecast(12)
            
            # Calculate prediction intervals
            residuals = train_data['Demand'] - train_pred
            residual_std = np.std(residuals)
            z_score = 1.96  # 95% confidence interval
            
            train_lower = train_pred - z_score * residual_std
            train_upper = train_pred + z_score * residual_std
            test_lower = test_pred - z_score * residual_std
            test_upper = test_pred + z_score * residual_std
            
            model_name = "Holt-Winters"
            
        else:
            # Use XGBoost for irregular demand
            X_train, feature_cols, scaler, y_train = create_features(train_data, is_training=True)
            X_test, _, _, y_test = create_features(test_data, scaler=scaler, is_training=False)
            
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate prediction intervals
            residuals = y_train - train_pred
            residual_std = np.std(residuals)
            z_score = 1.96
            
            train_lower = train_pred - z_score * residual_std
            train_upper = train_pred + z_score * residual_std
            test_lower = test_pred - z_score * residual_std
            test_upper = test_pred + z_score * residual_std
            
            model_name = "XGBoost"
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(train_data['Demand'], train_pred))
        test_rmse = np.sqrt(mean_squared_error(test_data['Demand'], test_pred))
        train_r2 = r2_score(train_data['Demand'], train_pred)
        test_r2 = r2_score(test_data['Demand'], test_pred)
        
        # Calculate RMSE as percentage of mean demand
        mean_demand = train_data['Demand'].mean()
        rmse_percentage = (test_rmse / mean_demand) * 100 if mean_demand > 0 else float('inf')
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        plt.plot(train_data.index, train_data['Demand'], label='Training Actual', color='blue', alpha=0.5)
        plt.plot(train_data.index, train_pred, label='Training Predicted', color='green', linestyle='--')
        plt.fill_between(train_data.index, train_lower, train_upper, color='green', alpha=0.1)
        
        plt.plot(test_data.index, test_data['Demand'], label='Test Actual', color='blue')
        plt.plot(test_data.index, test_pred, label='Test Predicted', color='red', linestyle='--')
        plt.fill_between(test_data.index, test_lower, test_upper, color='red', alpha=0.1)
        
        plt.title(f'{material_name} Demand Forecast\nModel: {model_name}, R² = {test_r2:.3f}, RMSE% = {rmse_percentage:.1f}%')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig(f'forecast_plots/{material_name}_forecast.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create metrics DataFrame
        train_metrics = pd.DataFrame({
            'Date': train_data.index,
            'Actual': train_data['Demand'],
            'Predicted': train_pred,
            'Lower_Bound': train_lower,
            'Upper_Bound': train_upper,
            'Dataset': 'Train'
        })
        
        test_metrics = pd.DataFrame({
            'Date': test_data.index,
            'Actual': test_data['Demand'],
            'Predicted': test_pred,
            'Lower_Bound': test_lower,
            'Upper_Bound': test_upper,
            'Dataset': 'Test'
        })
        
        # Combine train and test metrics
        metrics_df = pd.concat([train_metrics, test_metrics])
        metrics_df.to_csv(f'forecast_metrics/{material_name}_metrics.csv', index=False)
        
        return {
            'material_name': material_name,
            'model_type': model_name,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse_percentage': rmse_percentage
        }
    except Exception as e:
        print(f"Error analyzing {material_name}: {str(e)}")
        return None

def main():
    # Read the data
    data = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Order_Quantity_Monthly.csv")
    data['YearMonth'] = pd.to_datetime(data['YearMonth'])
    data.set_index('YearMonth', inplace=True)
    
    # Rename Dispatched Quantity to Demand for consistency
    data = data.rename(columns={'Dispatched Quantity': 'Demand'})
    
    # Get unique materials
    materials = data['Product ID'].unique()
    
    # Store results for each material
    results = []
    
    for material in materials:
        material_data = data[data['Product ID'] == material].copy()
        material_name = material_data['Product Name'].iloc[0]
        
        result = analyze_single_material(material_data, material_name)
        if result is not None:
            results.append(result)
            print(f"Completed analysis for {material_name}")
    
    # Save overall results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('monthly_forecast_results_v15.csv', index=False)
        print("\nAnalysis complete. Results saved to monthly_forecast_results_v15.csv")
    else:
        print("\nNo successful analyses to save.")

if __name__ == "__main__":
    main()
