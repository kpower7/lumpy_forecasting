import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the merged data
data = pd.read_excel("C:/Users/k_pow/Downloads/merged_file.xlsx")

# Ensure the Date column is in datetime format
# data['Date'] = pd.to_datetime(data['Date'])

# Create features for daily, weekly, and monthly forecasts
# data['Day'] = data['Date'].dt.day
# data['Month'] = data['Date'].dt.month
# data['Year'] = data['Date'].dt.year
# data['Week'] = data['Date'].dt.isocalendar().week

# Drop non-numeric columns
data = data.drop(columns=['Product ID', 'Product Name', 'Date', 'DATE'], errors='ignore')

# Set the target variable
target = 'Customer Order Quantity'

# Train/Test Split
train_data = data[data['Date'] <= '2024-05-31']
test_data = data[(data['Date'] > '2024-05-31') & (data['Date'] <= '2024-10-31')]

# Features and target for training
# Include all external variables and the target variable
X_train = train_data.drop(columns=[target, 'Date', 'DATE'])
y_train = train_data[target]

# Features and target for testing
X_test = test_data.drop(columns=[target, 'Date','DATE'])
y_test = test_data[target]

# Initialize and train the XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Optionally, save the predictions to a new file
predictions_df = test_data.copy()
predictions_df['Predicted'] = y_pred
predictions_df.to_excel("C:/Users/k_pow/Downloads/xgboost_predictions.xlsx", index=False)

# Train/Test Split for Daily Forecast
train_data_daily = data[data['Date'] <= '2024-05-31']
test_data_daily = data[(data['Date'] > '2024-05-31') & (data['Date'] <= '2024-10-31')]

# Features and target for daily training
X_train_daily = train_data_daily.drop(columns=['Customer Order Quantity'])
y_train_daily = train_data_daily['Customer Order Quantity']

# Initialize and train the XGBoost model for daily forecast
model_daily = XGBRegressor()
model_daily.fit(X_train_daily, y_train_daily)

# Make daily predictions
y_pred_daily = model_daily.predict(test_data_daily.drop(columns=['Customer Order Quantity']))

# Evaluate daily forecast
mse_daily = mean_squared_error(test_data_daily['Customer Order Quantity'], y_pred_daily)
print(f'Daily Mean Squared Error: {mse_daily}')

# Repeat similar steps for weekly and monthly forecasts... 