# Advanced Demand Forecasting Analysis for Lumpy Demand
# This version implements multiple models specialized for intermittent demand patterns

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better looking plots
plt.style.use('seaborn-v0_8')

def croston_forecast(data, alpha=0.4):
    """
    Implement Croston's method for intermittent demand forecasting
    """
    demand = np.array(data)
    q = 1  # Initialize interval
    p = demand[0]  # Initialize size
    forecast = np.zeros(len(demand))
    
    for t in range(1, len(demand)):
        if demand[t] > 0:
            p = alpha * demand[t] + (1 - alpha) * p
            q = alpha * q + (1 - alpha) * q
            forecast[t] = p / q
        else:
            q += 1
            forecast[t] = p / q
    
    return forecast

def create_lstm_model(input_shape):
    """
    Create LSTM model for time series forecasting
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_features(data, scaler=None, is_training=True):
    """
    Enhanced feature creation for lumpy demand
    """
    features = data.copy()
    
    if not isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index)
    
    # Basic time-based features
    features['Month'] = features.index.month
    features['Month_Sin'] = np.sin(2 * np.pi * features['Month']/12)
    features['Month_Cos'] = np.cos(2 * np.pi * features['Month']/12)
    features['DayOfWeek'] = features.index.dayofweek
    features['DayOfWeek_Sin'] = np.sin(2 * np.pi * features['DayOfWeek']/7)
    features['DayOfWeek_Cos'] = np.cos(2 * np.pi * features['DayOfWeek']/7)
    features['Quarter'] = features.index.quarter
    features['WeekOfYear'] = features.index.isocalendar().week
    features['DayOfYear'] = features.index.dayofyear
    features['DayOfYear_Sin'] = np.sin(2 * np.pi * features['DayOfYear']/365)
    features['DayOfYear_Cos'] = np.cos(2 * np.pi * features['DayOfYear']/365)
    features['IsWeekend'] = features.index.dayofweek.isin([5, 6]).astype(int)
    features['IsMonthStart'] = features.index.is_month_start.astype(int)
    features['IsMonthEnd'] = features.index.is_month_end.astype(int)
    features['IsQuarterStart'] = features.index.is_quarter_start.astype(int)
    features['IsQuarterEnd'] = features.index.is_quarter_end.astype(int)
    
    # Enhanced lag features for lumpy demand
    for lag in [1, 2, 3, 7, 14, 30]:
        features[f'Demand_Lag_{lag}'] = features['Customer Order Quantity'].shift(lag)
        # Add binary indicator for non-zero demand
        features[f'NonZero_Demand_Lag_{lag}'] = (features[f'Demand_Lag_{lag}'] > 0).astype(int)
    
    # Lumpy demand specific features
    order_mask = features['Customer Order Quantity'] > 0
    features['Days_Since_Last_Order'] = (~order_mask).cumsum()
    features.loc[order_mask, 'Days_Since_Last_Order'] = 0
    
    # Calculate order frequency features
    for window in [7, 14, 30, 60, 90]:
        # Order frequency
        features[f'Order_Frequency_{window}d'] = (
            features['Customer Order Quantity'] > 0
        ).shift(1).rolling(window=window, min_periods=1).mean()
        
        # Zero frequency
        features[f'Zero_Frequency_{window}d'] = (
            features['Customer Order Quantity'] == 0
        ).shift(1).rolling(window=window, min_periods=1).mean()
        
        # Demand variability
        features[f'Demand_Variability_{window}d'] = (
            features['Customer Order Quantity']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)
        )
        
        # Average non-zero order size
        features[f'NonZero_Avg_Order_{window}d'] = (
            features['Customer Order Quantity']
            .shift(1)
            .where(features['Customer Order Quantity'].shift(1) > 0)
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(0)
        )
        
        # Order size variance (non-zero orders only)
        features[f'NonZero_Order_Variance_{window}d'] = (
            features['Customer Order Quantity']
            .shift(1)
            .where(features['Customer Order Quantity'].shift(1) > 0)
            .rolling(window=window, min_periods=1)
            .var()
            .fillna(0)
        )
        
        # Exponential moving averages
        features[f'EMA_{window}d'] = (
            features['Customer Order Quantity']
            .shift(1)
            .ewm(span=window, min_periods=1)
            .mean()
            .fillna(0)
        )
    
    # Add dispatched quantity features
    if 'Dispatched Quantity' in features.columns:
        mask = features['Customer Order Quantity'].shift(1) != 0
        features['Order_Fulfillment_Rate'] = np.where(
            mask,
            features['Dispatched Quantity'].shift(1) / features['Customer Order Quantity'].shift(1),
            1.0
        )
        features['Order_Fulfillment_Rate'] = features['Order_Fulfillment_Rate'].clip(0, 1)
    
    # Drop rows with NaN values
    features = features.dropna()
    
    # Replace infinite values
    features = features.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Prepare feature columns
    feature_cols = [col for col in features.columns 
                   if col not in ['Customer Order Quantity', 'Dispatched Quantity'] 
                   and not col.startswith('Date')]
    
    # Scale features using RobustScaler (better for outliers)
    if is_training:
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features[feature_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for test data")
        features_scaled = scaler.transform(features[feature_cols])
    
    features_scaled = pd.DataFrame(features_scaled, columns=feature_cols, index=features.index)
    
    return features_scaled, features['Customer Order Quantity'], feature_cols, scaler

def analyze_single_material(material_data, material_name):
    """
    Analyze a single material using multiple models
    """
    try:
        # Data preparation
        material_data['Date'] = pd.to_datetime(material_data['Date'], format='%d.%m.%Y')
        material_data = material_data.drop(['Product Name', 'Product ID'], axis=1)
        material_data = material_data.set_index('Date')
        material_data = material_data.sort_index()
        
        print(f"\nAnalyzing material: {material_name}")
        print(f"Data range: {material_data.index.min()} to {material_data.index.max()}")
        
        # Split data
        split_date = material_data.index[int(len(material_data) * 0.8)]
        train_data = material_data[:split_date]
        test_data = material_data[split_date:]
        
        # Create features
        X_train, y_train, feature_cols, scaler = create_features(train_data, is_training=True)
        X_test, y_test, _, _ = create_features(test_data, scaler=scaler, is_training=False)
        
        # Initialize models
        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.01,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42
            ),
            'CatBoost': CatBoostRegressor(
                iterations=500,
                learning_rate=0.01,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
        }
        
        # Train and evaluate models
        results = {}
        best_rmse = float('inf')
        best_model = None
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_train_cv = X_train.iloc[train_idx]
                X_val_cv = X_train.iloc[val_idx]
                y_train_cv = y_train.iloc[train_idx]
                y_val_cv = y_train.iloc[val_idx]
                
                model.fit(X_train_cv, y_train_cv)
                y_pred_cv = model.predict(X_val_cv)
                cv_scores.append(r2_score(y_val_cv, y_pred_cv))
            
            # Train on full training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            results[model_name] = {
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'cv_r2_mean': np.mean(cv_scores),
                'cv_r2_std': np.std(cv_scores)
            }
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
        
        # Add Croston's method
        croston_pred = croston_forecast(y_train)
        croston_rmse = np.sqrt(mean_squared_error(y_test, croston_pred[-len(y_test):]))
        croston_r2 = r2_score(y_test, croston_pred[-len(y_test):])
        
        results['Croston'] = {
            'rmse': croston_rmse,
            'r2': croston_r2,
            'mape': mean_absolute_percentage_error(y_test, croston_pred[-len(y_test):]),
            'cv_r2_mean': None,
            'cv_r2_std': None
        }
        
        # Print results
        print("\nModel Performance Comparison:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"{metric}: {value:.4f}")
        
        print(f"\nBest performing model: {best_model}")
        
        return {
            'material_id': material_name,
            'best_model': best_model,
            **results[best_model]
        }
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def main():
    """Main function to run the analysis"""
    print("Loading data...")
    df_orders = pd.read_excel(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx")
    
    print("\nProcessing materials...")
    results = []
    materials = df_orders['Product ID'].unique()
    print(f"Total materials to process: {len(materials)}")
    
    for i, material in enumerate(materials, 1):
        print(f"\nProcessing material {i}/{len(materials)}: {material}")
        material_data = df_orders[df_orders['Product ID'] == material].copy()
        result = analyze_single_material(material_data, material)
        if result is not None:
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        
        print("\nSummary Statistics:")
        print("===================")
        print("\nModel Performance Metrics:")
        metrics = ['rmse', 'r2', 'mape', 'cv_r2_mean', 'cv_r2_std']
        print(results_df[metrics].describe())
        
        # Model distribution
        print("\nBest Model Distribution:")
        print(results_df['best_model'].value_counts())
        
        print("\nTop 5 Materials by R² Score:")
        print(results_df.nlargest(5, 'r2')[['material_id', 'best_model', 'r2', 'rmse', 'mape']])
        
        print("\nBottom 5 Materials by R² Score:")
        print(results_df.nsmallest(5, 'r2')[['material_id', 'best_model', 'r2', 'rmse', 'mape']])
        
        results_df.to_csv('forecast_results_all_materials_v13.csv', index=False)
        print("\nResults saved to forecast_results_all_materials_v13.csv")
    else:
        print("\nNo valid results to analyze")

if __name__ == "__main__":
    main()
