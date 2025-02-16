import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    """Class to handle feature engineering with proper train/test separation"""
    def __init__(self):
        self.scalers = {}
        self.train_stats = {}
        
    def fit_transform(self, df, is_training=True):
        """Create features for training data and store transformation parameters"""
        features = pd.DataFrame(index=df.index)
        
        if is_training:
            # Store statistics from training data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.train_stats = {
                'means': df[numeric_cols].mean(),
                'upper_limits': df[numeric_cols].quantile(0.99),
                'lower_limits': df[numeric_cols].quantile(0.01)
            }
        
        # Core demand features (from demand-driven model)
        for lag in range(1, 7):
            features[f'Demand_lag_{lag}'] = df['Dispatched Quantity'].shift(lag)
        
        # Demand patterns (proven most effective)
        for window in [3, 6, 12]:
            # Use expanding window for test set to avoid future data leakage
            if is_training:
                roll_mean = df['Dispatched Quantity'].rolling(window).mean()
                roll_std = df['Dispatched Quantity'].rolling(window).std()
            else:
                roll_mean = df['Dispatched Quantity'].expanding(min_periods=1).mean()
                roll_std = df['Dispatched Quantity'].expanding(min_periods=1).std()
            
            features[f'Demand_roll_mean_{window}'] = roll_mean
            features[f'Demand_roll_std_{window}'] = roll_std
            features[f'Demand_roll_cv_{window}'] = roll_std / (roll_mean + 1e-8)
            
            # Acceleration using appropriate window
            if is_training:
                features[f'Demand_accel_{window}'] = df['Dispatched Quantity'].diff().rolling(window).mean()
            else:
                features[f'Demand_accel_{window}'] = df['Dispatched Quantity'].diff().expanding(min_periods=1).mean()
        
        # Market indicators
        market_cols = ['TUAV', 'ECG_DESP', 'ISE_CO']
        for col in market_cols:
            # Recent market signals
            for lag in range(1, 4):
                features[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Market trends
            for window in [3, 6]:
                if is_training:
                    roll_mean = df[col].rolling(window).mean()
                else:
                    roll_mean = df[col].expanding(min_periods=1).mean()
                    
                features[f'{col}_roll_mean_{window}'] = roll_mean
                features[f'{col}_momentum_{window}'] = df[col] - roll_mean
        
        # Economic indicators
        econ_cols = ['ICI', 'PIB_CO']
        for col in econ_cols:
            features[f'{col}_lag_3'] = df[col].shift(3)
            if is_training:
                features[f'{col}_roll_mean_6'] = df[col].rolling(6).mean()
            else:
                features[f'{col}_roll_mean_6'] = df[col].expanding(min_periods=1).mean()
        
        # Order patterns
        features['Customer_Order_Lag1'] = df['Customer Order Quantity'].shift(1)
        features['Order_Demand_Ratio_Lag1'] = df['Customer Order Quantity'].shift(1) / (df['Dispatched Quantity'].shift(1) + 1e-8)
        
        # Add lagged order patterns
        if is_training:
            roll_obj = df['Customer Order Quantity'].rolling(window=3, min_periods=1)
        else:
            roll_obj = df['Customer Order Quantity'].expanding(min_periods=1)
        features['Customer_Order_Mean3'] = roll_obj.mean()
        
        # Seasonal components
        features['Month'] = df.index.month
        features['Quarter'] = df.index.quarter
        features['Year'] = df.index.year - 2022
        
        # Handle missing values using training statistics
        features = features.fillna(0)
        
        # Scale numerical features
        if is_training:
            for col in features.select_dtypes(include=[np.number]).columns:
                self.scalers[col] = StandardScaler()
                if col not in ['Month', 'Quarter', 'Year']:  # Don't scale categorical features
                    features[col] = self.scalers[col].fit_transform(features[[col]])
        else:
            for col in features.select_dtypes(include=[np.number]).columns:
                if col in self.scalers and col not in ['Month', 'Quarter', 'Year']:
                    features[col] = self.scalers[col].transform(features[[col]])
        
        return features

def preprocess_data(df, is_training=True, train_stats=None):
    """Preprocess data with proper train/test separation"""
    df_processed = df.copy()
    
    # Get only numeric columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    if is_training:
        # Calculate statistics only from training data
        train_stats = {
            'means': df[numeric_cols].mean(),
            'upper_limits': df[numeric_cols].quantile(0.99),
            'lower_limits': df[numeric_cols].quantile(0.01)
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

def create_hybrid_features(df, is_training=True):
    """Create features combining successful elements from all models"""
    features = pd.DataFrame(index=df.index)
    
    # Core demand features (from demand-driven model)
    for lag in range(1, 7):  # Shorter lag range for precision
        features[f'Demand_lag_{lag}'] = df['Dispatched Quantity'].shift(lag)
    
    # Demand patterns (proven most effective)
    for window in [3, 6, 12]:
        # Use expanding window for test data to prevent leakage
        if is_training:
            roll_obj = df['Dispatched Quantity'].rolling(window, min_periods=1)
        else:
            roll_obj = df['Dispatched Quantity'].expanding(min_periods=1)
        
        roll_mean = roll_obj.mean()
        roll_std = roll_obj.std()
        
        features[f'Demand_roll_mean_{window}'] = roll_mean
        features[f'Demand_roll_std_{window}'] = roll_std
        features[f'Demand_roll_cv_{window}'] = roll_std / (roll_mean + 1e-8)
        
        # Acceleration using appropriate window
        if is_training:
            features[f'Demand_accel_{window}'] = df['Dispatched Quantity'].diff().rolling(window, min_periods=1).mean()
        else:
            features[f'Demand_accel_{window}'] = df['Dispatched Quantity'].diff().expanding(min_periods=1).mean()
    
    # Market indicators (selective from market model)
    market_cols = ['TUAV', 'ECG_DESP', 'ISE_CO']
    for col in market_cols:
        # Recent market signals
        for lag in range(1, 4):
            features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Market trends (simplified)
        for window in [3, 6]:
            if is_training:
                roll_obj = df[col].rolling(window, min_periods=1)
            else:
                roll_obj = df[col].expanding(min_periods=1)
            
            roll_mean = roll_obj.mean()
            features[f'{col}_roll_mean_{window}'] = roll_mean
            # Momentum calculation
            features[f'{col}_momentum_{window}'] = df[col] - roll_mean
    
    # Economic indicators (minimal, from economic model)
    econ_cols = ['ICI', 'PIB_CO']
    for col in econ_cols:
        features[f'{col}_lag_3'] = df[col].shift(3)  # Quarter lag
        if is_training:
            features[f'{col}_roll_mean_6'] = df[col].rolling(6, min_periods=1).mean()
        else:
            features[f'{col}_roll_mean_6'] = df[col].expanding(min_periods=1).mean()
    
    # Order patterns (important across models)
    features['Customer_Order_Lag1'] = df['Customer Order Quantity'].shift(1)
    features['Order_Demand_Ratio_Lag1'] = df['Customer Order Quantity'].shift(1) / (df['Dispatched Quantity'].shift(1) + 1e-8)
    
    # Add lagged order patterns
    if is_training:
        roll_obj = df['Customer Order Quantity'].rolling(window=3, min_periods=1)
    else:
        roll_obj = df['Customer Order Quantity'].expanding(min_periods=1)
    features['Customer_Order_Mean3'] = roll_obj.mean()
    
    # Seasonal components
    features['Month'] = df.index.month
    features['Quarter'] = df.index.quarter
    features['Year'] = df.index.year - 2022
    
    return features.fillna(0)

def train_hybrid_model(material_id, material_name):
    """Train hybrid model combining successful elements from all models"""
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
    X_train = create_hybrid_features(train_processed, is_training=True)
    X_test = create_hybrid_features(test_processed, is_training=False)
    
    y_train = train_processed['Dispatched Quantity']
    y_test = test_processed['Dispatched Quantity']
    
    # Train model with balanced parameters
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
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
    plt.title(f'Hybrid Model v2: {material_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return metrics, importance

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
    
    for material_id, material_name in materials:
        print(f"\nTesting {material_name}")
        print("=" * 50)
        metrics, importance = train_hybrid_model(material_id, material_name)
        
        # Store results
        metrics_df = pd.Series(metrics)
        metrics_df.name = material_name
        all_metrics.append(metrics_df)
        
        importance['material'] = material_name
        all_importance.append(importance.head(5))  # Top 5 features
        
        print("\nPerformance Metrics:")
        print(metrics_df)
        print("\nTop 5 Features:")
        print(importance.head())
        plt.show()
    
    # Print summary comparison
    print("\nSummary of All Materials")
    print("=" * 50)
    metrics_comparison = pd.DataFrame(all_metrics)
    print("\nMetrics Comparison:")
    print(metrics_comparison)
    
    print("\nTop Features by Material:")
    top_features = pd.concat(all_importance)
