import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def create_demand_features(df):
    """Create features optimized for demand-driven materials"""
    features = pd.DataFrame(index=df.index)
    
    # Extended demand features (crucial for Material_56, Material_77)
    for lag in range(1, 13):  # Extended lag range for better pattern capture
        features[f'Demand_lag_{lag}'] = df['Dispatched Quantity'].shift(lag)
    
    # Rolling statistics on demand (proven effective)
    for window in [3, 6, 12]:
        features[f'Demand_roll_mean_{window}'] = df['Dispatched Quantity'].rolling(window).mean()
        features[f'Demand_roll_std_{window}'] = df['Dispatched Quantity'].rolling(window).std()
        features[f'Demand_roll_cv_{window}'] = features[f'Demand_roll_std_{window}'] / features[f'Demand_roll_mean_{window}']
        
        # Demand acceleration (key predictor)
        features[f'Demand_accel_{window}'] = (
            df['Dispatched Quantity'].diff() - 
            df['Dispatched Quantity'].diff().rolling(window).mean()
        )
    
    # Customer order features (important relationship)
    features['Customer_Order'] = df['Customer Order Quantity']
    features['Order_Demand_Ratio'] = df['Customer Order Quantity'] / df['Dispatched Quantity']
    
    # Seasonal features
    features['Month'] = df.index.month
    features['Quarter'] = df.index.quarter
    features['Year'] = df.index.year - 2022
    
    # Basic market indicators (minimal context)
    market_cols = ['TUAV', 'ECG_DESP']
    for col in market_cols:
        features[f'{col}_lag_1'] = df[col].shift(1)
        features[f'{col}_lag_2'] = df[col].shift(2)
    
    return features.fillna(0)

def train_demand_model(material_id, material_name):
    """Train model optimized for demand-driven materials"""
    # Load data
    df = pd.read_csv(fr"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files\{material_id}_Material_{material_name.split('_')[1]}.csv")
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])
    
    # Filter for date range 2022-01 to 2024-09
    df = df[(df['YearMonth'] >= '2022-01-01') & (df['YearMonth'] <= '2024-09-01')]
    df.set_index('YearMonth', inplace=True)
    
    # Create features
    features = create_demand_features(df)
    
    # Split data
    train = df[df.index < '2024-07-01']
    test = df[(df.index >= '2024-07-01') & (df.index <= '2024-09-01')]
    
    X_train = features[features.index < '2024-07-01']
    X_test = features[(features.index >= '2024-07-01') & (features.index <= '2024-09-01')]
    y_train = train['Dispatched Quantity']
    y_test = test['Dispatched Quantity']
    
    # Train model with previous successful parameters
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
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
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, y_train, label='Actual', marker='o')
    plt.plot(train.index, train_pred, label='Predicted', marker='s')
    plt.plot(test.index, y_test, label='Actual Test', marker='o')
    plt.plot(test.index, test_pred, label='Predicted Test', marker='s')
    plt.axvline(x=pd.to_datetime('2024-07-01'), color='r', linestyle='--', label='Train/Test Split')
    plt.title(f'Demand-Driven Model: {material_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return metrics, importance

if __name__ == "__main__":
    # Test with demand-driven materials
    materials = [
        ('0001O1184', 'Material_56'),  # Original test material
        ('0001O3160', 'Material_77')   # Additional demand-driven material
    ]
    
    for material_id, material_name in materials:
        print(f"\nTesting {material_name}")
        print("=" * 50)
        metrics, importance = train_demand_model(material_id, material_name)
        print("\nPerformance Metrics:")
        print(pd.Series(metrics))
        print("\nTop 10 Features:")
        print(importance.head(10))
        plt.show()
