# %% [markdown]
# # XGBoost Demand Forecasting Analysis
# This notebook analyzes demand forecasting using XGBoost for all materials, including detailed performance metrics and visualizations.

# %%
# Import required libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better looking plots
plt.style.use('seaborn-v0_8')  # Updated style name for newer versions

# %% [markdown]
# ## 1. Data Loading and Preprocessing

# %%
def load_and_process_data():
    """Load and process order data"""
    print("Loading data...")
    
    # Read the data files
    df_orders = pd.read_excel(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx")
    
    # Print column names to debug
    print("\nAvailable columns:")
    print(df_orders.columns.tolist())
    print("\nSample of first few rows:")
    print(df_orders.head())
    
    # Convert date column
    df_orders['Date'] = pd.to_datetime(df_orders['Date'], format='%d.%m.%Y')
    
    return df_orders

# %% [markdown]
# ## 2. XGBoost Model Implementation

# %%
def create_features(data):
    """Create features using only the data available at prediction time"""
    features = data.copy()
    
    # Convert index to datetime if it's not already
    if not isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index)
    
    # Basic time-based features
    features['Month'] = features.index.month
    features['Month_Sin'] = np.sin(2 * np.pi * features['Month']/12)
    features['Month_Cos'] = np.cos(2 * np.pi * features['Month']/12)
    features['DayOfWeek'] = features.index.dayofweek
    features['DayOfWeek_Sin'] = np.sin(2 * np.pi * features['DayOfWeek']/7)
    features['DayOfWeek_Cos'] = np.cos(2 * np.pi * features['DayOfWeek']/7)
    features['Quarter'] = features.index.quarter
    features['WeekOfYear'] = features.index.isocalendar().week
    features['DayOfYear'] = features.index.dayofyear
    features['DayOfYear_Sin'] = np.sin(2 * np.pi * features['DayOfYear']/365)
    features['DayOfYear_Cos'] = np.cos(2 * np.pi * features['DayOfYear']/365)
    features['IsWeekend'] = features.index.dayofweek.isin([5, 6]).astype(int)
    features['IsMonthStart'] = features.index.is_month_start.astype(int)
    features['IsMonthEnd'] = features.index.is_month_end.astype(int)
    features['IsQuarterStart'] = features.index.is_quarter_start.astype(int)
    features['IsQuarterEnd'] = features.index.is_quarter_end.astype(int)
    
    # Features specific for lumpy demand
    # Calculate intervals between orders
    order_mask = features['Customer Order Quantity'] > 0
    features['Days_Since_Last_Order'] = (~order_mask).cumsum()
    features['Days_Since_Last_Order'][order_mask] = 0
    
    # Calculate order frequency features with different windows
    for window in [7, 14, 30, 60, 90]:
        # Rolling order frequency
        features[f'Order_Frequency_{window}d'] = (
            features['Customer Order Quantity'] > 0
        ).rolling(window=window, min_periods=1).mean()
        
        # Rolling zero frequency
        features[f'Zero_Frequency_{window}d'] = (
            features['Customer Order Quantity'] == 0
        ).rolling(window=window, min_periods=1).mean()
        
        # Rolling demand variability
        features[f'Demand_Variability_{window}d'] = (
            features['Customer Order Quantity']
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)  # Fill NaN with 0
        )
        
        # Average order size when orders occur
        features[f'Avg_Order_Size_{window}d'] = (
            features['Customer Order Quantity']
            .where(features['Customer Order Quantity'] > 0)
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(0)  # Fill NaN with 0
        )
        
        # Maximum order size
        features[f'Max_Order_Size_{window}d'] = (
            features['Customer Order Quantity']
            .rolling(window=window, min_periods=1)
            .max()
            .fillna(0)  # Fill NaN with 0
        )
        
        # Order size variance
        features[f'Order_Size_Variance_{window}d'] = (
            features['Customer Order Quantity']
            .rolling(window=window, min_periods=1)
            .var()
            .fillna(0)  # Fill NaN with 0
        )
        
        # Exponential moving averages
        features[f'EMA_{window}d'] = (
            features['Customer Order Quantity']
            .ewm(span=window, min_periods=1)
            .mean()
            .fillna(0)  # Fill NaN with 0
        )
    
    # Add dispatched quantity features if available
    if 'Dispatched Quantity' in features.columns:
        # Handle division by zero
        mask = features['Customer Order Quantity'] != 0
        features['Order_Fulfillment_Rate'] = np.where(
            mask,
            features['Dispatched Quantity'] / features['Customer Order Quantity'],
            1.0  # When no order, assume 100% fulfillment
        )
        features['Order_Fulfillment_Rate'] = features['Order_Fulfillment_Rate'].clip(0, 1)
        
        # Calculate lag between order and dispatch
        features['Order_Dispatch_Lag'] = (
            features['Dispatched Quantity'].shift(-1) - 
            features['Customer Order Quantity']
        ).fillna(0)
    
    # Drop rows with NaN values
    features = features.dropna()
    
    # Replace infinite values with large finite values
    features = features.replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    # Prepare feature columns
    feature_cols = [col for col in features.columns 
                   if col not in ['Customer Order Quantity', 'Dispatched Quantity'] 
                   and not col.startswith('Date')]
    
    return features[feature_cols], features['Customer Order Quantity'], feature_cols

def xgboost_forecast(df):
    """Implement XGBoost model with feature engineering"""
    try:
        # Ensure data is sorted chronologically
        df = df.sort_index()
        
        # Calculate the split point to ensure test set is the most recent 20% of data
        split_date = df.index[int(len(df) * 0.8)]
        
        # Split data ensuring chronological order
        train_df = df[:split_date].copy()
        test_df = df[split_date:].copy()
        
        print(f"Training data from {train_df.index.min()} to {train_df.index.max()}")
        print(f"Testing data from {test_df.index.min()} to {test_df.index.max()}")
        
        # Create features for training data
        X_train, y_train, feature_cols = create_features(train_df)
        
        # Create features for test data
        X_test, y_test, _ = create_features(test_df)
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Ensure no negative predictions
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        bias = np.mean(y_pred - y_test)
        zero_accuracy = accuracy_score((y_test == 0), (y_pred < 0.5))
        
        return {
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'bias': bias,
            'zero_accuracy': zero_accuracy,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
    except Exception as e:
        print(f"Error in XGBoost: {str(e)}")
        return None

# %% [markdown]
# ## 3. Visualization Functions

# %%
def plot_forecast(df, predictions, material_name, r2, metrics):
    """Plot actual vs predicted values with detailed metrics"""
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
    metrics_text = f"MAE: {metrics['MAE']:.2f}\nRMSE: {metrics['RMSE']:.2f}\nMAPE: {metrics['MAPE']:.2f}%\nBias: {metrics['Bias']:.2f}\nZero/Non-zero Accuracy: {metrics['Binary_Accuracy']:.2f}"
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
def plot_feature_importance(feature_importance, material_name):
    """Plot feature importance for a material"""
    plt.figure(figsize=(10, 6))
    
    # Sort feature importance
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    # Create barh plot
    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance for {material_name}')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Analysis Functions

# %%
def analyze_performance_distribution(results_df):
    """Analyze the distribution of model performance"""
    plt.figure(figsize=(12, 6))
    
    # Create histogram of R2 scores
    plt.hist(results_df['R2_Score'], bins=20, edgecolor='black')
    plt.axvline(results_df['R2_Score'].mean(), color='red', linestyle='dashed', 
                label=f'Mean R² = {results_df["R2_Score"].mean():.3f}')
    plt.axvline(results_df['R2_Score'].median(), color='green', linestyle='dashed', 
                label=f'Median R² = {results_df["R2_Score"].median():.3f}')
    
    plt.title('Distribution of R² Scores Across Materials')
    plt.xlabel('R² Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
def analyze_error_patterns(results_df):
    """Analyze patterns in prediction errors"""
    error_analysis = pd.DataFrame({
        'Material': results_df['Material'],
        'R2_Score': results_df['R2_Score'],
        'MAE': results_df['MAE'],
        'RMSE': results_df['RMSE'],
        'MAPE': results_df['MAPE'],
        'Bias': results_df['Bias'],
        'Binary_Accuracy': results_df['Binary_Accuracy']
    }).sort_values('R2_Score', ascending=False)
    
    return error_analysis

# %% [markdown]
# ## 5. Main Analysis

# %%
def analyze_single_material(material_data, material_name):
    """Analyze a single material's demand pattern"""
    try:
        # Convert date and set as index
        material_data['Date'] = pd.to_datetime(material_data['Date'], format='%d.%m.%Y')
        
        # Drop unnecessary columns and set index to Date
        material_data = material_data.drop(['Product Name', 'Product ID'], axis=1)
        material_data = material_data.set_index('Date')
        material_data = material_data.sort_index()
        
        print(f"Training data from {material_data.index.min()} to {material_data.index.max()}")
        
        # Create features and split data
        X, y, feature_cols = create_features(material_data)
        
        # Print feature information
        print(f"\nNumber of features: {len(feature_cols)}")
        print("Features:", feature_cols)
        
        # Split into training and testing sets
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model with updated parameters
        model = xgb.XGBRegressor(
            n_estimators=500,          # More trees for better learning
            learning_rate=0.01,        # Lower learning rate for better generalization
            max_depth=6,               # Slightly deeper trees
            min_child_weight=5,        # Helps prevent overfitting
            subsample=0.8,             # Use 80% of data per tree
            colsample_bytree=0.8,      # Use 80% of features per tree
            gamma=1,                   # Minimum loss reduction for split
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=1.0,            # L2 regularization
            random_state=42,           # For reproducibility
            tree_method='hist',        # Faster histogram-based algorithm
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Ensure no negative predictions
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        bias = np.mean(y_pred - y_test)
        zero_accuracy = accuracy_score((y_test == 0), (y_pred < 0.5))
        
        # Print feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        print("\nTop 10 most important features:")
        print(importance.head(10))
        
        return {
            'material_id': material_name,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'bias': bias,
            'zero_accuracy': zero_accuracy,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def main():
    """Main function to run the analysis"""
    print("Loading data...")
    df_orders = pd.read_excel(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx")
    
    print("\nAvailable columns:")
    print(df_orders.columns.tolist())
    print("\nSample data:")
    print(df_orders.head())
    
    print("\nProcessing materials...")
    results = []
    materials = df_orders['Product ID'].unique()
    print(f"Total materials to process: {len(materials)}")
    
    for i, material in enumerate(materials, 1):
        print(f"\nProcessing material {i}/{len(materials)}: {material}")
        
        # Get data for this material
        material_data = df_orders[df_orders['Product ID'] == material].copy()
        
        # Analyze the material
        result = analyze_single_material(material_data, material)
        if result is not None:
            results.append(result)
    
    if results:
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        print("\nSummary Statistics:")
        print("===================")
        print("\nModel Performance Metrics:")
        metrics = ['rmse', 'r2', 'mape', 'zero_accuracy']
        print(results_df[metrics].describe())
        
        # Print materials with best and worst performance
        print("\nTop 5 Materials by R² Score:")
        print(results_df.nlargest(5, 'r2')[['material_id', 'r2', 'rmse', 'mape', 'zero_accuracy']])
        
        print("\nBottom 5 Materials by R² Score:")
        print(results_df.nsmallest(5, 'r2')[['material_id', 'r2', 'rmse', 'mape', 'zero_accuracy']])
        
        # Save results
        results_df.to_csv('forecast_results_all_materials.csv', index=False)
        print("\nResults saved to forecast_results_all_materials.csv")
    else:
        print("\nNo valid results to analyze")

if __name__ == "__main__":
    main()
