#%% [markdown]
# XGBoost Model with Balanced Economic Features
# This version balances between long-term economic indicators and data availability

#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

#%% [markdown]
# Helper Functions

#%%
def create_features(df):
    df = df.copy()
    
    # Time-based features
    df['Year'] = df['YearMonth'].dt.year - 2022  # Normalize year
    df['Month'] = df['YearMonth'].dt.month
    df['Quarter'] = df['YearMonth'].dt.quarter
    
    # Economic indicators (longer timeframes)
    economic_vars = ['PIB_CO', 'ICI']
    market_vars = ['ECG_DESP', 'TUAV', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19']
    
    # Extended lags for economic indicators (up to 12 months)
    for col in economic_vars:
        for i in range(1, 13):  # 12 months of lags
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    
    # Regular lags for market indicators (up to 6 months)
    for col in market_vars:
        for i in range(1, 7):  # 6 months of lags
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    
    # Extended rolling means for economic indicators
    for col in economic_vars:
        for window in [3, 6, 12]:  # Up to 12 months
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
    
    # Regular rolling means for market indicators
    for col in market_vars:
        for window in [3, 6]:  # Up to 6 months
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
    
    # Target variable features (shorter lags to preserve data)
    for i in range(1, 7):  # 6 months of lags
        df[f'Demand_lag_{i}'] = df['Dispatched Quantity'].shift(i)
    
    for window in [3, 6]:
        df[f'Demand_roll_mean_{window}'] = df['Dispatched Quantity'].rolling(window=window).mean()
        df[f'Demand_roll_std_{window}'] = df['Dispatched Quantity'].rolling(window=window).std()
    
    return df

def train_and_evaluate(material_id, material_name):
    print(f"\nAnalyzing {material_name} (ID: {material_id})")
    print("=" * 50)
    
    # Read the data
    material_data = pd.read_csv(f"C:\\Users\\k_pow\\OneDrive\\Documents\\MIT\\MITx SCM\\IAP 2025\\SCC\\Data_files\\{material_id}_{material_name}.csv")
    material_data['YearMonth'] = pd.to_datetime(material_data['YearMonth'])
    
    # Focus on actual demand period (2022-2024)
    train_data = material_data[material_data['YearMonth'] >= '2022-01-01'].copy()
    
    print(f"Date Range: {train_data['YearMonth'].min().strftime('%Y-%m')} to {train_data['YearMonth'].max().strftime('%Y-%m')}")
    print(f"Initial records: {len(train_data)}")
    
    # Create features
    train_data = create_features(train_data)
    train_data = train_data.dropna()
    
    print(f"Records after feature creation: {len(train_data)}")
    print(f"Total features created: {len(train_data.columns)}")
    
    # Split data
    train = train_data[train_data['YearMonth'] < '2024-07-01']
    test = train_data[(train_data['YearMonth'] >= '2024-07-01') & (train_data['YearMonth'] <= '2024-09-01')]
    
    # Define features (exclude YearMonth)
    feature_columns = [col for col in train_data.columns 
                      if col not in ['YearMonth', 'Product ID', 'Product Name', 'Dispatched Quantity']]
    
    # Prepare X and y
    X_train = train[feature_columns]
    y_train = train['Dispatched Quantity']
    X_test = test[feature_columns]
    y_test = test['Dispatched Quantity']
    
    print(f"\nFeature columns: {len(feature_columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model with all features
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'train_mape': mean_absolute_percentage_error(y_train, train_pred),
        'test_mape': mean_absolute_percentage_error(y_test, test_pred)
    }
    
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance analysis
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Categorize features
    def categorize_feature(feature_name):
        if any(ext in feature_name for ext in ['PIB_CO', 'ICI']):
            if 'lag' in feature_name:
                return 'Economic Indicator Lag'
            elif 'roll_mean' in feature_name:
                return 'Economic Indicator Rolling Mean'
            elif 'roll_std' in feature_name:
                return 'Economic Indicator Rolling Std'
            else:
                return 'Economic Indicator Direct'
        elif any(ext in feature_name for ext in ['ECG_DESP', 'TUAV', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19']):
            if 'lag' in feature_name:
                return 'Market Indicator Lag'
            elif 'roll_mean' in feature_name:
                return 'Market Indicator Rolling Mean'
            elif 'roll_std' in feature_name:
                return 'Market Indicator Rolling Std'
            else:
                return 'Market Indicator Direct'
        elif 'Demand' in feature_name:
            if 'roll_std' in feature_name:
                return 'Demand Rolling Std'
            elif 'roll_mean' in feature_name:
                return 'Demand Rolling Mean'
            else:
                return 'Demand Lag'
        else:
            return 'Time Features'
    
    feature_importance['Category'] = feature_importance['Feature'].apply(categorize_feature)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(pd.concat([train['YearMonth'], test['YearMonth']]), 
             pd.concat([y_train, y_test]), label='Actual', marker='o')
    plt.plot(pd.concat([train['YearMonth'], test['YearMonth']]), 
             np.concatenate([train_pred, test_pred]), label='Predicted', marker='s')
    plt.axvline(x=pd.to_datetime('2024-07-01'), color='r', linestyle='--', label='Train/Test Split')
    plt.title(f'Forecast vs Actual Demand ({material_name})')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(20), x='Importance', y='Feature', hue='Category', dodge=False)
    plt.title(f'Top 20 Most Important Features ({material_name})')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Plot importance by category
    category_importance = feature_importance.groupby('Category')['Importance'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    category_importance.plot(kind='bar')
    plt.title(f'Feature Importance by Category ({material_name})')
    plt.xlabel('Feature Category')
    plt.ylabel('Total Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Save results
    results = pd.DataFrame({
        'Date': pd.concat([train['YearMonth'], test['YearMonth']]),
        'Actual': pd.concat([y_train, y_test]),
        'Predicted': np.concatenate([train_pred, test_pred])
    })
    
    results.to_csv(f"C:\\Users\\k_pow\\OneDrive\\Documents\\MIT\\MITx SCM\\IAP 2025\\SCC\\XGBoost\\{material_name}_predictions_economic.csv", index=False)
    feature_importance.to_csv(f"C:\\Users\\k_pow\\OneDrive\\Documents\\MIT\\MITx SCM\\IAP 2025\\SCC\\XGBoost\\{material_name}_feature_importance_economic.csv", index=False)
    
    print("\nTop Features by Category:")
    print("-" * 30)
    for category in category_importance.index:
        print(f"\n{category}:")
        top_features = feature_importance[feature_importance['Category'] == category].head(3)
        for _, row in top_features.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    return metrics, feature_importance

#%% [markdown]
# Run analysis for both materials

#%%
# Analyze Material 4
metrics_4, importance_4 = train_and_evaluate('000161032', 'Material_4')

# Analyze Material 12
metrics_12, importance_12 = train_and_evaluate('000179754', 'Material_12')

#%% [markdown]
# Compare results between materials

#%%
# Create comparison DataFrame
comparison = pd.DataFrame({
    'Material_4': pd.Series(metrics_4),
    'Material_12': pd.Series(metrics_12)
})

print("\nModel Performance Comparison:")
print("=" * 50)
print(comparison)

# Save comparison
comparison.to_csv(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\XGBoost\material_comparison_economic.csv")
