# %% [markdown]
# # Demand Forecasting using XGBoost
# This notebook performs demand forecasting for Material_51 using various economic indicators as features.

# %% [markdown]
# ## Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Read the data
df = pd.read_csv(r'C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files\0001O1010_Material_51.csv')
# Convert YearMonth to datetime
df['YearMonth'] = pd.to_datetime(df['YearMonth'])
print("Data loaded successfully. Shape:", df.shape)

# %% [markdown]
# ## Split Data into Train and Test Sets

# %%
# Define features and target
features = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
target = 'Customer Order Quantity'

# Split data based on date
train_data = df[df['YearMonth'] < '2024-06-01']
test_data = df[(df['YearMonth'] >= '2024-06-01') & (df['YearMonth'] <= '2024-10-01')]

# Prepare train and test sets
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# %% [markdown]
# ## Train XGBoost Model

# %%
# Initialize and train XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)
print("Model training completed")

# %% [markdown]
# ## Make Predictions and Evaluate Model

# %%
# Make predictions
y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Root Mean Square Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.3f}")

# Create a DataFrame with actual vs predicted values
results = pd.DataFrame({
    'Date': test_data['YearMonth'],
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nPredictions for test period:")
print(results)

# %% [markdown]
# ## Visualize Forecasting Results

# %%
# Create visualization of actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(results['Date'], results['Actual'], label='Actual', marker='o')
plt.plot(results['Date'], results['Predicted'], label='Predicted', marker='s')
plt.title('Material 51: Actual vs Predicted Demand')
plt.xlabel('Date')
plt.ylabel('Order Quantity')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Show feature importance
plt.figure(figsize=(10, 5))
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
importance_df = importance_df.sort_values('Importance', ascending=True)
plt.barh(y=importance_df['Feature'], width=importance_df['Importance'])
plt.title('Feature Importance in Demand Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()

# Calculate and plot prediction error
plt.figure(figsize=(12, 4))
results['Error'] = results['Actual'] - results['Predicted']
plt.plot(results['Date'], results['Error'], marker='o', color='red', linestyle='-')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.title('Prediction Error Over Time')
plt.xlabel('Date')
plt.ylabel('Error (Actual - Predicted)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
