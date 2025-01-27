# XGBoost Demand Forecasting Analysis
# This notebook analyzes demand forecasting using XGBoost for all materials, including detailed performance metrics and visualizations.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better looking plots
plt.style.use('seaborn-v0_8')

def create_features(data, scaler=None, is_training=True):
    """
    Create features using only the data available at prediction time
    Parameters:
        data: DataFrame with time series data
        scaler: fitted StandardScaler (optional)
        is_training: boolean indicating if this is training data
    """
    features = data.copy()
    
    # Convert index to datetime if it's not already
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
    
    # Calculate lagged features (using only past data)
    for lag in [1, 2, 3, 7, 14, 30]:
        features[f'Demand_Lag_{lag}'] = features['Customer Order Quantity'].shift(lag)
    
    # Features specific for lumpy demand using only past data
    order_mask = features['Customer Order Quantity'] > 0
    features['Days_Since_Last_Order'] = (~order_mask).cumsum()
    features.loc[order_mask, 'Days_Since_Last_Order'] = 0
    
    # Calculate order frequency features with different windows (using only past data)
    for window in [7, 14, 30, 60, 90]:
        # Rolling order frequency
        features[f'Order_Frequency_{window}d'] = (
            features['Customer Order Quantity'] > 0
        ).shift(1).rolling(window=window, min_periods=1).mean()
        
        # Rolling zero frequency
        features[f'Zero_Frequency_{window}d'] = (
            features['Customer Order Quantity'] == 0
        ).shift(1).rolling(window=window, min_periods=1).mean()
        
        # Rolling demand variability
        features[f'Demand_Variability_{window}d'] = (
            features['Customer Order Quantity']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)
        )
        
        # Average order size when orders occur
        features[f'Avg_Order_Size_{window}d'] = (
            features['Customer Order Quantity']
            .shift(1)
            .where(features['Customer Order Quantity'].shift(1) > 0)
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(0)
        )
        
        # Maximum order size
        features[f'Max_Order_Size_{window}d'] = (
            features['Customer Order Quantity']
            .shift(1)
            .rolling(window=window, min_periods=1)
            .max()
            .fillna(0)
        )
        
        # Order size variance
        features[f'Order_Size_Variance_{window}d'] = (
            features['Customer Order Quantity']
            .shift(1)
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
    
    # Add dispatched quantity features if available (using only past data)
    if 'Dispatched Quantity' in features.columns:
        # Calculate fulfillment rate using lagged data
        mask = features['Customer Order Quantity'].shift(1) != 0
        features['Order_Fulfillment_Rate'] = np.where(
            mask,
            features['Dispatched Quantity'].shift(1) / features['Customer Order Quantity'].shift(1),
            1.0
        )
        features['Order_Fulfillment_Rate'] = features['Order_Fulfillment_Rate'].clip(0, 1)
    
    # Drop rows with NaN values
    features = features.dropna()
    
    # Replace infinite values with large finite values
    features = features.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Prepare feature columns
    feature_cols = [col for col in features.columns 
                   if col not in ['Customer Order Quantity', 'Dispatched Quantity'] 
                   and not col.startswith('Date')]
    
    # Scale features
    if is_training:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features[feature_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for test data")
        features_scaled = scaler.transform(features[feature_cols])
    
    # Convert back to DataFrame
    features_scaled = pd.DataFrame(features_scaled, columns=feature_cols, index=features.index)
    
    return features_scaled, features['Customer Order Quantity'], feature_cols, scaler

def analyze_single_material(material_data, material_name):
    """Analyze a single material's demand pattern with proper time series handling"""
    try:
        # Convert date and set as index
        material_data['Date'] = pd.to_datetime(material_data['Date'], format='%d.%m.%Y')
        
        # Drop unnecessary columns and set index to Date
        material_data = material_data.drop(['Product Name', 'Product ID'], axis=1)
        material_data = material_data.set_index('Date')
        material_data = material_data.sort_index()
        
        print(f"\nAnalyzing material: {material_name}")
        print(f"Data range: {material_data.index.min()} to {material_data.index.max()}")
        
        # Calculate split point (80% train, 20% test)
        split_date = material_data.index[int(len(material_data) * 0.8)]
        
        # Split data chronologically
        train_data = material_data[:split_date]
        test_data = material_data[split_date:]
        
        # Create features separately for train and test
        X_train, y_train, feature_cols, scaler = create_features(train_data, is_training=True)
        X_test, y_test, _, _ = create_features(test_data, scaler=scaler, is_training=False)
        
        print(f"\nNumber of features: {len(feature_cols)}")
        
        # Initialize TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize model with updated parameters
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method='hist',
            objective='reg:squarederror'
        )
        
        # Perform time series cross-validation
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_train_cv = X_train.iloc[train_idx]
            X_val_cv = X_train.iloc[val_idx]
            y_train_cv = y_train.iloc[train_idx]
            y_val_cv = y_train.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            cv_scores.append(r2_score(y_val_cv, y_pred_cv))
        
        print(f"\nCross-validation R² scores: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        # Train final model on full training data
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Ensure no negative predictions
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        bias = np.mean(y_pred - y_test)
        zero_accuracy = accuracy_score((y_test == 0), (y_pred < 0.5))
        
        # Print feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        print(importance.head(10))
        
        return {
            'material_id': material_name,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'bias': bias,
            'zero_accuracy': zero_accuracy,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores)
        }
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def main():
    """Main function to run the analysis"""
    print("Loading data...")
    df_orders = pd.read_excel(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx")
    
    print("\nAvailable columns:")
    print(df_orders.columns.tolist())
    
    print("\nProcessing materials...")
    results = []
    materials = df_orders['Product ID'].unique()
    print(f"Total materials to process: {len(materials)}")
    
    for i, material in enumerate(materials, 1):
        print(f"\nProcessing material {i}/{len(materials)}: {material}")
        
        # Get data for this material
        material_data = df_orders[df_orders['Product ID'] == material].copy()
        
        # Analyze the material
        result = analyze_single_material(material_data, material)
        if result is not None:
            results.append(result)
    
    if results:
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        print("\nSummary Statistics:")
        print("===================")
        print("\nModel Performance Metrics:")
        metrics = ['rmse', 'r2', 'mape', 'zero_accuracy', 'cv_r2_mean', 'cv_r2_std']
        print(results_df[metrics].describe())
        
        # Print materials with best and worst performance
        print("\nTop 5 Materials by R² Score:")
        print(results_df.nlargest(5, 'r2')[['material_id', 'r2', 'rmse', 'mape', 'zero_accuracy', 'cv_r2_mean']])
        
        print("\nBottom 5 Materials by R² Score:")
        print(results_df.nsmallest(5, 'r2')[['material_id', 'r2', 'rmse', 'mape', 'zero_accuracy', 'cv_r2_mean']])
        
        # Save results
        results_df.to_csv('forecast_results_all_materials_v12.csv', index=False)
        print("\nResults saved to forecast_results_all_materials_v12.csv")
    else:
        print("\nNo valid results to analyze")

if __name__ == "__main__":
    main()
