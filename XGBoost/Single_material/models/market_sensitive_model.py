import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def create_market_features(df):
    """Create features optimized for market-sensitive materials"""
    features = pd.DataFrame(index=df.index)
    
    # Core market indicators (focused set)
    market_cols = ['ECG_DESP', 'TUAV', 'ISE_CO']  # Most important market indicators
    for col in market_cols:
        # Market lags (shorter timeframe)
        for lag in range(1, 7):  # 6 months of lags
            features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Market trends
        for window in [3, 6]:
            features[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
            features[f'{col}_momentum_{window}'] = (
                df[col] - df[col].rolling(window).mean()
            ) / df[col].rolling(window).std()
    
    # Demand context (important for stability)
    for i in range(1, 4):  # Shorter lag range
        features[f'Demand_lag_{i}'] = df['Dispatched Quantity'].shift(i)
    
    for window in [3, 6]:
        features[f'Demand_roll_mean_{window}'] = df['Dispatched Quantity'].rolling(window).mean()
    
    # Market interactions (selective)
    features['market_correlation'] = df['TUAV'].rolling(6).corr(df['ECG_DESP'])
    features['Customer_Order'] = df['Customer Order Quantity']
    
    # Time features
    features['Month'] = df.index.month
    features['Quarter'] = df.index.quarter
    features['Year'] = df.index.year - 2022
    
    return features.fillna(0)

def train_market_model(material_id, material_name):
    """Train model optimized for market-sensitive materials"""
    # Load data
    df = pd.read_csv(fr"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files\{material_id}_Material_{material_name.split('_')[1]}.csv")
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])
    
    # Filter for date range 2022-01 to 2024-09
    df = df[(df['YearMonth'] >= '2022-01-01') & (df['YearMonth'] <= '2024-09-01')]
    df.set_index('YearMonth', inplace=True)
    
    # Create features
    features = create_market_features(df)
    
    # Split data
    train = df[df.index < '2024-07-01']
    test = df[(df.index >= '2024-07-01') & (df.index <= '2024-09-01')]
    
    X_train = features[features.index < '2024-07-01']
    X_test = features[(features.index >= '2024-07-01') & (features.index <= '2024-09-01')]
    y_train = train['Dispatched Quantity']
    y_test = test['Dispatched Quantity']
    
    # Train model with more conservative parameters
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
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
    plt.title(f'Market-Sensitive Model: {material_name}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return metrics, importance

if __name__ == "__main__":
    # Test with market-sensitive materials
    materials = [
        ('555136414', 'Material_1'),  # Original test material
        ('0001PZ701', 'Material_82')  # Additional market-sensitive material
    ]
    
    for material_id, material_name in materials:
        print(f"\nTesting {material_name}")
        print("=" * 50)
        metrics, importance = train_market_model(material_id, material_name)
        print("\nPerformance Metrics:")
        print(pd.Series(metrics))
        print("\nTop 10 Features:")
        print(importance.head(10))
        plt.show()
