import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

def create_simple_features(df, is_training=True):
    """Create a minimal set of features with strict data leakage prevention"""
    features = pd.DataFrame(index=df.index)
    
    # Core demand features - only use short-term lags
    for lag in range(1, 4):  # Reduced from 6 to 3 lags
        features[f'Demand_lag_{lag}'] = df['Dispatched Quantity'].shift(lag)
    
    # Simple moving averages - only short windows
    for window in [3, 6]:  # Reduced from [3,6,12] to [3,6]
        if is_training:
            roll_obj = df['Dispatched Quantity'].rolling(window=window, min_periods=1)
        else:
            roll_obj = df['Dispatched Quantity'].expanding(min_periods=1)
        
        features[f'Demand_MA_{window}'] = roll_obj.mean()
    
    # Simplified market indicators - only use most recent lag and simple MA
    market_cols = ['TUAV', 'ECG_DESP']  # Reduced market indicators
    for col in market_cols:
        # Only most recent lag
        features[f'{col}_lag_1'] = df[col].shift(1)
        
        # Simple moving average
        if is_training:
            roll_obj = df[col].rolling(window=3, min_periods=1)
        else:
            roll_obj = df[col].expanding(min_periods=1)
        features[f'{col}_MA_3'] = roll_obj.mean()
    
    # Order patterns - simple lagged features
    features['Order_lag_1'] = df['Customer Order Quantity'].shift(1)
    features['Order_Demand_Ratio_lag_1'] = df['Customer Order Quantity'].shift(1) / (df['Dispatched Quantity'].shift(1) + 1e-8)
    
    # Basic seasonal components
    features['Month'] = df.index.month
    features['Quarter'] = df.index.quarter
    
    return features.fillna(0)

def preprocess_data(df, is_training=True, train_stats=None):
    """Simple preprocessing with proper train/test separation"""
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    if is_training:
        train_stats = {
            'means': df[numeric_cols].mean(),
            'upper_limits': df[numeric_cols].quantile(0.95),  # Less extreme outlier handling
            'lower_limits': df[numeric_cols].quantile(0.05)
        }
    
    # Apply preprocessing using training statistics
    for col in numeric_cols:
        # Handle infinities
        df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with training means
        df_processed[col] = df_processed[col].fillna(train_stats['means'][col])
        
        # Cap extreme values using training quantiles
        df_processed[col] = df_processed[col].clip(
            lower=train_stats['lower_limits'][col],
            upper=train_stats['upper_limits'][col]
        )
    
    return df_processed, train_stats

def train_simple_model(material_id, material_name):
    """Train simplified model with strict data leakage prevention"""
    # Load data
    df = pd.read_csv(fr"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files\{material_id}_Material_{material_name.split('_')[1]}.csv")
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])
    
    # Filter for date range
    df = df[(df['YearMonth'] >= '2022-01-01') & (df['YearMonth'] <= '2024-09-01')]
    df.set_index('YearMonth', inplace=True)
    
    # Split data first before any preprocessing
    train = df[df.index < '2024-07-01']
    test = df[(df.index >= '2024-07-01') & (df.index <= '2024-09-01')]
    
    # Preprocess train and test separately
    train_processed, train_stats = preprocess_data(train, is_training=True)
    test_processed, _ = preprocess_data(test, is_training=False, train_stats=train_stats)
    
    # Create features with proper separation
    X_train = create_simple_features(train_processed, is_training=True)
    X_test = create_simple_features(test_processed, is_training=False)
    
    y_train = train_processed['Dispatched Quantity']
    y_test = test_processed['Dispatched Quantity']
    
    # Simpler XGBoost model with stronger regularization
    model = XGBRegressor(
        n_estimators=100,  # Reduced from 150
        learning_rate=0.01,  # Reduced from 0.03
        max_depth=2,  # Reduced from 3
        min_child_weight=5,  # Increased from 3
        subsample=0.7,  # Reduced from 0.8
        colsample_bytree=0.7,  # Reduced from 0.8
        gamma=1,  # Added to prevent overfitting
        reg_alpha=1,  # L1 regularization
        reg_lambda=2,  # L2 regularization
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Feature selection
    selector = SelectFromModel(model, prefit=True, threshold='mean')
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Retrain on selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=2,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=1,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42
    )
    
    model.fit(X_train_selected, y_train)
    
    # Predictions
    train_pred = model.predict(X_train_selected)
    test_pred = model.predict(X_test_selected)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'train_mape': mean_absolute_percentage_error(y_train, train_pred),
        'test_mape': mean_absolute_percentage_error(y_test, test_pred)
    }
    
    # Feature importance (only for selected features)
    importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, y_train, label='Actual', marker='o')
    plt.plot(train.index, train_pred, label='Predicted', marker='s')
    plt.plot(test.index, y_test, label='Actual Test', marker='o')
    plt.plot(test.index, test_pred, label='Predicted Test', marker='s')
    plt.axvline(x=pd.to_datetime('2024-07-01'), color='r', linestyle='--', label='Train/Test Split')
    plt.title(f'Simple Model v3: {material_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return metrics, importance, selected_features

if __name__ == "__main__":
    # Test with the specific materials from performance analysis
    materials = [
        # Good performers
        ('555136414', 'Material_1'),   # R² = 0.66 (best)
        ('000161032', 'Material_4'),   # R² = 0.58
        ('000193598', 'Material_28'),  # R² = 0.26
        
        # Moderate performers
        ('0001O3160', 'Material_77'),  # R² = 0.06
        ('0001O1184', 'Material_56'),  # R² = -0.18
        
        # Poor performers
        ('0001O1010', 'Material_51'),  # R² = -0.82
        ('333180636', 'Material_21'),  # R² = -1.88
        ('000196706', 'Material_37'),  # R² = -4.68
        ('0001PZ701', 'Material_82'),  # R² = -8.93
        ('000179754', 'Material_12'),  # R² = -54.69 (worst)
    ]
    
    # Store results for comparison
    all_metrics = []
    all_importance = []
    all_selected_features = {}
    
    for material_id, material_name in materials:
        print(f"\nTesting {material_name}")
        print("=" * 50)
        metrics, importance, selected_features = train_simple_model(material_id, material_name)
        
        # Store results
        metrics_df = pd.Series(metrics)
        metrics_df.name = material_name
        all_metrics.append(metrics_df)
        
        importance['material'] = material_name
        all_importance.append(importance)
        
        all_selected_features[material_name] = selected_features
        
        print("\nPerformance Metrics:")
        print(metrics_df)
        print("\nSelected Features:")
        print(importance)
        plt.show()
    
    # Print summary comparison
    print("\nSummary of All Materials")
    print("=" * 50)
    metrics_comparison = pd.DataFrame(all_metrics)
    print("\nMetrics Comparison:")
    print(metrics_comparison)
    
    print("\nSelected Features by Material:")
    for material, features in all_selected_features.items():
        print(f"\n{material}:")
        print(features)
