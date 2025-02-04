#%% [markdown]
# XGBoost Forecasting Model for Single Material
# This notebook demonstrates XGBoost forecasting for a single material using external variables

#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

#%% [markdown]
# Load and prepare the data for Material_4

#%%
# Read the data
material_data = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files\000161032_Material_4.csv")
material_data['YearMonth'] = pd.to_datetime(material_data['YearMonth'])

# We'll focus on the actual demand period (2022-2024)
train_data = material_data[material_data['YearMonth'] >= '2022-01-01'].copy()

print("Data Summary:")
print("=" * 50)
print(f"Date Range: {train_data['YearMonth'].min().strftime('%Y-%m')} to {train_data['YearMonth'].max().strftime('%Y-%m')}")
print(f"Number of records: {len(train_data)}")

#%% [markdown]
# Create features for the model

#%%
def create_features(df):
    df = df.copy()
    
    # Time-based features
    df['Year'] = df['YearMonth'].dt.year
    df['Month'] = df['YearMonth'].dt.month
    df['Quarter'] = df['YearMonth'].dt.quarter
    df['IsQuarterEnd'] = df['YearMonth'].dt.is_quarter_end.astype(int)
    df['IsQuarterStart'] = df['YearMonth'].dt.is_quarter_start.astype(int)
    
    # Lag features for external variables (3 months)
    for col in ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']:
        for i in range(1, 4):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
    
    # Rolling means for external variables (3 and 6 months)
    for col in ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']:
        for window in [3, 6]:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
    
    # Target variable lags and rolling stats
    df['Demand_lag_1'] = df['Dispatched Quantity'].shift(1)
    df['Demand_lag_2'] = df['Dispatched Quantity'].shift(2)
    df['Demand_lag_3'] = df['Dispatched Quantity'].shift(3)
    df['Demand_roll_mean_3'] = df['Dispatched Quantity'].rolling(window=3).mean()
    df['Demand_roll_mean_6'] = df['Dispatched Quantity'].rolling(window=6).mean()
    
    return df

# Create features
train_data = create_features(train_data)

# Drop rows with NaN values (due to lag features)
train_data = train_data.dropna()

print("\nFeature Creation:")
print("=" * 50)
print(f"Number of features created: {len(train_data.columns)}")
print(f"Data points after removing NaN: {len(train_data)}")

#%% [markdown]
# Split the data into training and testing sets

#%%
# Use data up to May 2024 for training (leaving 4 months for testing)
train = train_data[train_data['YearMonth'] < '2024-05-01']
test = train_data[train_data['YearMonth'] >= '2024-05-01']

# Define features
feature_columns = ['Year', 'Month', 'Quarter', 'IsQuarterEnd', 'IsQuarterStart',
                  'ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI',
                  'Demand_lag_1', 'Demand_lag_2', 'Demand_lag_3',
                  'Demand_roll_mean_3', 'Demand_roll_mean_6']

# Add lag and rolling mean features
feature_columns.extend([col for col in train.columns if ('lag_' in col or 'roll_mean_' in col) 
                       and not col.startswith('Demand')])

# Prepare X and y
X_train = train[feature_columns]
y_train = train['Dispatched Quantity']
X_test = test[feature_columns]
y_test = test['Dispatched Quantity']

print("\nTrain/Test Split:")
print("=" * 50)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

#%% [markdown]
# Train the XGBoost model

#%%
# Initialize and train the model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=False
)

#%% [markdown]
# Evaluate the model

#%%
# Make predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_mape = mean_absolute_percentage_error(y_train, train_pred)
test_mape = mean_absolute_percentage_error(y_test, test_pred)

print("\nModel Performance Metrics:")
print("=" * 50)
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Training MAPE: {train_mape:.4f}")
print(f"Testing MAPE: {test_mape:.4f}")

#%% [markdown]
# Visualize the results

#%%
# Create a DataFrame with actual and predicted values
results = pd.DataFrame({
    'Date': pd.concat([train['YearMonth'], test['YearMonth']]),
    'Actual': pd.concat([y_train, y_test]),
    'Predicted': np.concatenate([train_pred, test_pred])
})

plt.figure(figsize=(12, 6))
plt.plot(results['Date'], results['Actual'], label='Actual', marker='o')
plt.plot(results['Date'], results['Predicted'], label='Predicted', marker='s')
plt.axvline(x=pd.to_datetime('2024-05-01'), color='r', linestyle='--', label='Train/Test Split')
plt.title('XGBoost Forecast vs Actual Demand')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% [markdown]
# Feature Importance Analysis

#%%
# Get feature importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
plt.title('Top 10 Most Important Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

#%% [markdown]
# Save the results

#%%
# Save predictions to CSV
results.to_csv(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\XGBoost\material_4_predictions.csv", index=False)

# Save model metrics
metrics = pd.DataFrame({
    'Metric': ['Train RMSE', 'Test RMSE', 'Train R²', 'Test R²', 'Train MAPE', 'Test MAPE'],
    'Value': [train_rmse, test_rmse, train_r2, test_r2, train_mape, test_mape]
})
metrics.to_csv(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\XGBoost\material_4_metrics.csv", index=False)

print("\nResults have been saved to files.")
