import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler  # Better for outliers
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    """Class to handle feature engineering with focus on high volatility"""
    def __init__(self):
        self.scalers = {}
        self.train_stats = {}
        
    def fit_transform(self, df, is_training=True):
        """Create features with focus on volatility handling"""
        features = pd.DataFrame(index=df.index)
        
        if is_training:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.train_stats = {
                'means': df[numeric_cols].mean(),
                'medians': df[numeric_cols].median(),  # More robust central tendency
                'upper_limits': df[numeric_cols].quantile(0.95),  # Less extreme quantiles
                'lower_limits': df[numeric_cols].quantile(0.05)
            }
        
        # Core demand features - shorter lag range
        for lag in range(1, 4):  # Reduced from 6 to 3
            features[f'Demand_lag_{lag}'] = df['Dispatched Quantity'].shift(lag)
        
        # Demand patterns - shorter windows
        for window in [2, 3, 4]:  # Shorter windows
            if is_training:
                roll_obj = df['Dispatched Quantity'].rolling(window, min_periods=1)
            else:
                roll_obj = df['Dispatched Quantity'].expanding(min_periods=1)
            
            roll_mean = roll_obj.mean()
            roll_median = roll_obj.median()  # More robust to outliers
            roll_std = roll_obj.std()
            
            features[f'Demand_roll_mean_{window}'] = roll_mean
            features[f'Demand_roll_median_{window}'] = roll_median
            features[f'Demand_roll_std_{window}'] = roll_std
            
            # Volatility features
            features[f'Demand_volatility_{window}'] = roll_std / (roll_median + 1e-8)
            
            # Direction changes
            if is_training:
                diff_sign = np.sign(df['Dispatched Quantity'].diff())
                features[f'Direction_changes_{window}'] = diff_sign.rolling(window, min_periods=1).apply(
                    lambda x: (x != x.shift(1)).sum() / len(x)
                )
            else:
                diff_sign = np.sign(df['Dispatched Quantity'].diff())
                features[f'Direction_changes_{window}'] = diff_sign.expanding(min_periods=1).apply(
                    lambda x: (x != x.shift(1)).sum() / len(x)
                )
        
        # Market indicators - focus on most important
        market_cols = ['TUAV', 'ECG_DESP']  # Reduced market indicators
        for col in market_cols:
            # Recent signals
            features[f'{col}_lag_1'] = df[col].shift(1)
            
            # Market trends - shorter windows
            for window in [2, 3]:
                if is_training:
                    roll_obj = df[col].rolling(window, min_periods=1)
                else:
                    roll_obj = df[col].expanding(min_periods=1)
                    
                roll_mean = roll_obj.mean()
                features[f'{col}_roll_mean_{window}'] = roll_mean
                # Relative change instead of momentum
                features[f'{col}_rel_change_{window}'] = (df[col].shift(1) - roll_mean) / (roll_mean + 1e-8)
        
        # Order patterns - simplified
        features['Order_lag_1'] = df['Customer Order Quantity'].shift(1)
        features['Order_Demand_Ratio_lag_1'] = df['Customer Order Quantity'].shift(1) / (df['Dispatched Quantity'].shift(1) + 1e-8)
        
        # Add recent volatility in orders
        if is_training:
            order_std = df['Customer Order Quantity'].rolling(3, min_periods=1).std()
            order_mean = df['Customer Order Quantity'].rolling(3, min_periods=1).mean()
        else:
            order_std = df['Customer Order Quantity'].expanding(min_periods=1).std()
            order_mean = df['Customer Order Quantity'].expanding(min_periods=1).mean()
        
        features['Order_volatility'] = order_std / (order_mean + 1e-8)
        
        # Seasonal components - simplified
        features['Month'] = df.index.month
        features['Quarter'] = df.index.quarter
        
        # Handle missing values using robust statistics
        for col in features.select_dtypes(include=[np.number]).columns:
            if col not in ['Month', 'Quarter']:
                features[col] = features[col].fillna(0)
        
        # Scale numerical features using RobustScaler for better outlier handling
        if is_training:
            for col in features.select_dtypes(include=[np.number]).columns:
                if col not in ['Month', 'Quarter']:
                    self.scalers[col] = RobustScaler()
                    features[col] = self.scalers[col].fit_transform(features[[col]])
        else:
            for col in features.select_dtypes(include=[np.number]).columns:
                if col in self.scalers and col not in ['Month', 'Quarter']:
                    features[col] = self.scalers[col].transform(features[[col]])
        
        return features

def preprocess_data(df, is_training=True, train_stats=None):
    """Preprocess data with robust statistics"""
    df_processed = df.copy()
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    if is_training:
        train_stats = {
            'medians': df[numeric_cols].median(),
            'upper_limits': df[numeric_cols].quantile(0.95),
            'lower_limits': df[numeric_cols].quantile(0.05)
        }
    
    # Apply preprocessing using robust statistics
    for col in numeric_cols:
        # Handle infinities
        df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with training medians
        df_processed[col] = df_processed[col].fillna(train_stats['medians'][col])
        
        # Cap extreme values using less extreme quantiles
        df_processed[col] = df_processed[col].clip(
            lower=train_stats['lower_limits'][col],
            upper=train_stats['upper_limits'][col]
        )
    
    return df_processed, train_stats

def train_hybrid_model(material_id, material_name):
    """Train hybrid model with focus on volatility handling"""
    # Load data
    df = pd.read_csv(fr"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files\{material_id}_Material_{material_name.split('_')[1]}.csv")
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])
    
    # Filter for date range
    df = df[(df['YearMonth'] >= '2022-01-01') & (df['YearMonth'] <= '2024-09-01')]
    df.set_index('YearMonth', inplace=True)
    
    # Split data
    train = df[df.index < '2024-07-01']
    test = df[(df.index >= '2024-07-01') & (df.index <= '2024-09-01')]
    
    # Preprocess with robust statistics
    train_processed, train_stats = preprocess_data(train, is_training=True)
    test_processed, _ = preprocess_data(test, is_training=False, train_stats=train_stats)
    
    # Create features
    feature_engineer = FeatureEngineering()
    X_train = feature_engineer.fit_transform(train_processed, is_training=True)
    X_test = feature_engineer.fit_transform(test_processed, is_training=False)
    
    y_train = train_processed['Dispatched Quantity']
    y_test = test_processed['Dispatched Quantity']
    
    # Train model with stronger regularization
    model = XGBRegressor(
        n_estimators=100,  # Reduced from 150
        learning_rate=0.01,  # Reduced from 0.03
        max_depth=2,  # Reduced from 3
        min_child_weight=5,  # Increased from 3
        subsample=0.7,  # More aggressive subsampling
        colsample_bytree=0.7,
        gamma=1,  # Added minimum loss reduction
        reg_alpha=0.5,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'train_mape': mean_absolute_percentage_error(y_train, train_pred),
        'test_mape': mean_absolute_percentage_error(y_test, test_pred)
    }
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, y_train, label='Actual', marker='o')
    plt.plot(train.index, train_pred, label='Predicted', marker='s')
    plt.plot(test.index, y_test, label='Actual Test', marker='o')
    plt.plot(test.index, test_pred, label='Predicted Test', marker='s')
    plt.axvline(x=pd.to_datetime('2024-07-01'), color='r', linestyle='--', label='Train/Test Split')
    plt.title(f'Hybrid Model v2.2 (Volatility Focus): {material_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return metrics, importance

if __name__ == "__main__":
    # Test with materials that showed high volatility
    materials = [
        # Most volatile materials from v2.1
        ('000193598', 'Material_28'),  # R² = -5.841
        ('333180636', 'Material_21'),  # R² = -23.926
        ('000196706', 'Material_37'),  # R² = -0.057
        ('0001PZ701', 'Material_82'),  # R² = -1.837
        ('000179754', 'Material_12'),  # R² = -1.228
    ]
    
    # Store results
    all_metrics = []
    all_importance = []
    
    for material_id, material_name in materials:
        print(f"\nTesting {material_name}")
        print("=" * 50)
        metrics, importance = train_hybrid_model(material_id, material_name)
        
        metrics_df = pd.Series(metrics)
        metrics_df.name = material_name
        all_metrics.append(metrics_df)
        
        importance['material'] = material_name
        all_importance.append(importance.head(5))
        
        print("\nPerformance Metrics:")
        print(metrics_df)
        print("\nTop 5 Features:")
        print(importance.head())
        plt.show()
    
    # Print summary
    print("\nSummary of All Materials")
    print("=" * 50)
    metrics_comparison = pd.DataFrame(all_metrics)
    print("\nMetrics Comparison:")
    print(metrics_comparison)
    
    print("\nTop Features by Material:")
    top_features = pd.concat(all_importance)
