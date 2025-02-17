# %% [markdown]
# # Demand Forecasting using CatBoost
# This notebook performs demand forecasting for Material_51 using various economic indicators as features.

# %% [markdown]
# ## Import Libraries and Load Data

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
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
# ## Train CatBoost Model

# %%
# Initialize and train CatBoost model
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    verbose=100,
    random_seed=42
)

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_test))
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
plt.title('Material 51: Actual vs Predicted Demand (CatBoost)')
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
plt.title('Feature Importance in Demand Prediction (CatBoost)')
plt.xlabel('Importance Score')
plt.tight_layout()

# Calculate and plot prediction error
plt.figure(figsize=(12, 4))
results['Error'] = results['Actual'] - results['Predicted']
plt.plot(results['Date'], results['Error'], marker='o', color='red', linestyle='-')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.title('Prediction Error Over Time (CatBoost)')
plt.xlabel('Date')
plt.ylabel('Error (Actual - Predicted)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(model.get_evals_result()['learn']['RMSE'], label='Train')
plt.plot(model.get_evals_result()['validation']['RMSE'], label='Validation')
plt.title('Learning Curves (CatBoost)')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# %% [markdown]
# ## Compare Feature Importance Between Models
# To compare this with the XGBoost model, you can run both scripts and compare the feature importance plots and error metrics.
