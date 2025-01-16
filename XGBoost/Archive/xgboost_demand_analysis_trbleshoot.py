# %% [markdown]
# # XGBoost Demand Forecasting Analysis
# This notebook analyzes demand forecasting using XGBoost for all materials, including detailed performance metrics and visualizations.

# %%
# Import required libraries
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better looking plots
plt.style.use('default')  

# %% [markdown]
# ## 1. Data Loading and Preprocessing

# %%
def load_and_process_data():
    """Load and process order data"""
    print("Loading data...")
    
    # Read the data files
    df_orders = pd.read_excel("../Customer Order Quantity_Dispatched Quantity.xlsx")
    
    # Print data info
    print("\nData Info:")
    print(df_orders.info())
    print("\nSample Data:")
    print(df_orders.head())
    
    # Convert date column
    df_orders['Date'] = pd.to_datetime(df_orders['Date'], format='%d.%m.%Y')
    
    return df_orders

# %% [markdown]
# ## 2. XGBoost Model Implementation

# %%
def xgboost_forecast(df):
    """Implement XGBoost model with feature engineering"""
    try:
        print("\nPreparing features...")
        df_features = df.copy()
        
        # Create lag features (previous days' demand)
        for i in range(1, 8):  
            df_features[f'order_lag_{i}'] = df_features['Customer Order Quantity'].shift(i)
        
        # Create rolling mean features with different windows
        for window in [3, 7, 14, 30]:  
            df_features[f'order_rolling_mean_{window}'] = df_features['Customer Order Quantity'].rolling(window=window).mean()
        
        # Create rolling std features to capture volatility
        for window in [7, 14]:
            df_features[f'order_rolling_std_{window}'] = df_features['Customer Order Quantity'].rolling(window=window).std()
        
        # Add day of week and month cyclical features
        df_features['DayOfWeek'] = df_features.index.dayofweek
        df_features['Month'] = df_features.index.month
        
        # Convert to cyclical features
        df_features['Month_Sin'] = np.sin(2 * np.pi * df_features['Month']/12)
        df_features['Month_Cos'] = np.cos(2 * np.pi * df_features['Month']/12)
        df_features['DayOfWeek_Sin'] = np.sin(2 * np.pi * df_features['DayOfWeek']/7)
        df_features['DayOfWeek_Cos'] = np.cos(2 * np.pi * df_features['DayOfWeek']/7)
        
        # Drop rows with NaN values from the lag features
        df_features = df_features.dropna()
        
        # Select features for modeling
        feature_cols = [col for col in df_features.columns if col not in 
                       ['Customer Order Quantity', 'Product ID', 'Product Name', 
                        'Dispatched Quantity', 'Year', 'Month', 'Day', 'DayOfWeek']]
        
        print("\nFeature columns:", feature_cols)
        print("Number of features:", len(feature_cols))
        
        # Prepare X and y
        X = df_features[feature_cols]
        y = df_features['Customer Order Quantity']
        
        # Split data
        split_idx = int(len(df_features) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print("\nData split information:")
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print("\nTarget variable statistics:")
        print("Training set:")
        print(y_train.describe())
        print("\nTest set:")
        print(y_test.describe())
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with adjusted parameters
        model = XGBRegressor(
            n_estimators=200,  
            learning_rate=0.05,  
            max_depth=5,  
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=42
        )
        
        # Train model with evaluation
        model.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            verbose=True
        )
        
        # Generate predictions
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print("\nDetailed metrics:")
        print(f"R² Score: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        
        # Print feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        print(importance_df.head(10))
        
        # Generate predictions for full dataset
        X_full_scaled = scaler.transform(X)
        y_pred = model.predict(X_full_scaled)
        
        # Create series with predictions aligned to original data
        predictions = pd.Series(index=df_features.index, data=y_pred)
        predictions = predictions.reindex(df.index)
        
        print("\nPrediction Statistics:")
        print(f"Actual values range: {int(y_test.min())} to {int(y_test.max())}")
        print(f"Predicted values range: {y_pred_test.min():.6f} to {y_pred_test.max():.6f}")
        
        return predictions, r2, {
            'MAE': mae, 
            'RMSE': rmse, 
            'Feature_Importance': dict(zip(feature_cols, model.feature_importances_))
        }
        
    except Exception as e:
        print(f"Error in XGBoost: {str(e)}")
        return None, None, None

# %%
def plot_forecast(df, predictions, material_name, r2, metrics):
    """Plot actual vs predicted values with detailed metrics"""
    print("\nPlotting data for", material_name)
    print("Data shape:", df.shape)
    
    # Print data around September 2024
    sept_2024_mask = (df.index >= '2024-08-01') & (df.index <= '2024-10-31')
    print("\nData around September 2024:")
    print("\nActual values:")
    print(df.loc[sept_2024_mask, 'Customer Order Quantity'].sort_values(ascending=False).head(10))
    print("\nPredicted values:")
    print(predictions[sept_2024_mask].sort_values(ascending=False).head(10))
    
    plt.figure(figsize=(15, 7))
    
    # Plot actual values
    plt.plot(df.index, df['Customer Order Quantity'], 
             label='Actual', color='blue', alpha=0.7, linewidth=2)
    
    # Plot predictions (only where we have them)
    mask = ~predictions.isna()
    plt.plot(df.index[mask], predictions[mask],
            label=f'XGBoost (R² = {r2:.3f})',
            color='red', alpha=0.7, linewidth=2)
    
    # Add metrics text box
    metrics_text = f"MAE: {metrics['MAE']:.2f}\nRMSE: {metrics['RMSE']:.2f}"
    plt.text(0.02, 0.98, metrics_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f'XGBoost Forecast vs Actual for {material_name}')
    plt.xlabel('Date')
    plt.ylabel('Order Quantity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add shaded background for test set
    split_idx = int(len(df) * 0.8)
    plt.axvspan(df.index[split_idx], df.index[-1], 
                color='gray', alpha=0.1, label='Test Set')
    
    plt.tight_layout()
    plt.show()

# %%
def main():
    # Load data
    order_data = load_and_process_data()
    
    # Process Material_1 (555136414)
    material_id = '555136414'  # Material_1's Product ID
    material_orders = order_data[order_data['Product ID'] == material_id].copy()
    
    print(f"\nProcessing Material_1...")
    print("\nRaw data for this material:")
    print(material_orders.sort_values('Date', ascending=True).head(20))
    
    # Set date as index and sort
    material_orders.set_index('Date', inplace=True)
    material_orders.sort_index(inplace=True)
    
    # Run XGBoost
    predictions, r2, metrics = xgboost_forecast(material_orders)
    
    if predictions is not None:
        # Create plots
        plot_forecast(material_orders, predictions, "Material_1", r2, metrics)

if __name__ == "__main__":
    main()
