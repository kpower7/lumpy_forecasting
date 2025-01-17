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
import traceback

# Configure matplotlib for better looking plots
plt.style.use('seaborn-v0_8')

# %% [markdown]
# ## 1. Data Loading and Preprocessing

# %%
def load_and_process_data(file_path):
    """Load and process data from Excel file"""
    try:
        # Read data
        print("Loading data...\n")
        df = pd.read_excel(file_path)
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
        
        # Sort by Product ID and Date
        df = df.sort_values(['Product ID', 'Date']).copy()
        
        print("Data Info:")
        print(df.info())
        print("\nSample Data:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# %% [markdown]
# ## 2. XGBoost Model Implementation

# %%
def xgboost_forecast(df):
    """Implement XGBoost model with simplified feature engineering"""
    try:
        # Check if we have enough data points
        if len(df) < 60:  # Require at least 60 days of data
            print("Not enough data points for reliable forecasting")
            return None, None, None
            
        # Calculate sparsity
        sparsity = (df['Customer Order Quantity'] == 0).mean()
        print(f"Order sparsity: {sparsity:.2%}")
        
        # Create a clean copy
        df_clean = df.copy()
        
        # First split the data BEFORE any feature engineering
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx].copy()
        test_df = df_clean.iloc[split_idx:].copy()
        
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        def create_features(data):
            """Create simplified time series features"""
            df_features = data.copy()
            
            # Basic date features
            df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
            df_features['Month'] = df_features['Date'].dt.month
            
            # Simple lag features (just last 3 days)
            for i in range(1, 4):
                df_features[f'order_lag_{i}'] = df_features.groupby('Product ID')['Customer Order Quantity'].shift(i)
            
            # Simple moving average (7 days)
            df_features['order_ma_7'] = df_features.groupby('Product ID')['Customer Order Quantity'].rolling(
                window=7, min_periods=1).mean().reset_index(0, drop=True).shift(1)
            
            # Fill NaN values with 0
            df_features = df_features.fillna(0)
            
            return df_features
        
        # Create features
        train_features = create_features(train_df)
        test_features = create_features(test_df)
        
        # Select features for modeling
        feature_cols = ['DayOfWeek', 'Month', 
                       'order_lag_1', 'order_lag_2', 'order_lag_3', 
                       'order_ma_7']
        
        # Prepare X and y
        X_train = train_features[feature_cols]
        y_train = train_features['Customer Order Quantity']
        X_test = test_features[feature_cols]
        y_test = test_features['Customer Order Quantity']
        
        # Simplified XGBoost parameters
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        # Train model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred_test = model.predict(X_test)
        y_pred_test = np.maximum(0, y_pred_test)  # Ensure non-negative predictions
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'Sparsity': sparsity,
            'Feature_Importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        return y_pred_test, r2, metrics
        
    except Exception as e:
        print(f"Error in XGBoost: {str(e)}")
        return None, None, None

# %% [markdown]
# ## 3. Visualization Functions

# %%
def plot_actual_vs_predicted(df, predictions, material_name):
    """Plot actual vs predicted values"""
    # Get test set (last 20% of data)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()  # Create a copy to avoid SettingWithCopyWarning
    test_df = test_df.reset_index(drop=True)  # Reset index to avoid plotting issues
    
    # Clear any existing plots
    plt.clf()
    plt.close('all')
    
    # Create new figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot with dates
    ax.plot(test_df['Date'], test_df['Customer Order Quantity'], 
            label='Actual', alpha=0.7, color='blue', marker='o', markersize=4)
    ax.plot(test_df['Date'], predictions, 
            label='Predicted', alpha=0.7, color='red', marker='x', markersize=4)
    
    ax.set_title(f'Actual vs Predicted Demand for {material_name} (Test Set)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Quantity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()
    plt.close()

# %%
def plot_feature_importance(feature_importance, material_name):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title(f'Feature Importance for {material_name}')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Analysis Functions

# %%
def analyze_performance_distribution(results_df):
    """Analyze the distribution of model performance metrics"""
    try:
        if results_df is None or len(results_df) == 0:
            print("No results to analyze")
            return
            
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['R2'], bins=20, edgecolor='black')
        plt.title('Distribution of R² Scores Across Materials')
        plt.xlabel('R² Score')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print summary statistics
        print("\nPerformance Summary:")
        print(f"Average R²: {results_df['R2'].mean():.3f}")
        print(f"Median R²: {results_df['R2'].median():.3f}")
        print(f"Std R²: {results_df['R2'].std():.3f}")
        print(f"Min R²: {results_df['R2'].min():.3f}")
        print(f"Max R²: {results_df['R2'].max():.3f}")
        
        # Count materials by R² ranges
        ranges = [(-np.inf, 0), (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        counts = pd.cut(results_df['R2'], bins=[-np.inf, 0, 0.2, 0.4, 0.6, 0.8, 1.0]).value_counts().sort_index()
        
        print("\nR² Score Distribution:")
        for i, count in enumerate(counts):
            start, end = ranges[i]
            if start == -np.inf:
                print(f"R² < 0: {count} materials")
            else:
                print(f"{start:.1f} ≤ R² < {end:.1f}: {count} materials")
                
    except Exception as e:
        print(f"Error in performance analysis: {str(e)}")
        
def analyze_error_patterns(results_df):
    """Analyze patterns in prediction errors"""
    try:
        if results_df is None or len(results_df) == 0:
            print("No results to analyze")
            return
            
        # Analyze relationship between sparsity and performance
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Sparsity'], results_df['R2'], alpha=0.6)
        plt.title('R² Score vs Order Sparsity')
        plt.xlabel('Order Sparsity')
        plt.ylabel('R² Score')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Calculate correlation
        correlation = results_df['Sparsity'].corr(results_df['R2'])
        print(f"\nCorrelation between Sparsity and R²: {correlation:.3f}")
        
        # Group materials by performance
        poor_performers = results_df[results_df['R2'] < 0.4]
        good_performers = results_df[results_df['R2'] >= 0.4]
        
        print("\nPerformance Analysis:")
        print(f"Poor performing materials (R² < 0.4): {len(poor_performers)}")
        print(f"Good performing materials (R² ≥ 0.4): {len(good_performers)}")
        
        if len(poor_performers) > 0:
            print("\nPoor Performing Materials Analysis:")
            print(f"Average sparsity: {poor_performers['Sparsity'].mean():.2%}")
            print(f"Average MAE: {poor_performers['MAE'].mean():.2f}")
            print(f"Average RMSE: {poor_performers['RMSE'].mean():.2f}")
        
        if len(good_performers) > 0:
            print("\nGood Performing Materials Analysis:")
            print(f"Average sparsity: {good_performers['Sparsity'].mean():.2%}")
            print(f"Average MAE: {good_performers['MAE'].mean():.2f}")
            print(f"Average RMSE: {good_performers['RMSE'].mean():.2f}")
            
    except Exception as e:
        print(f"Error in error pattern analysis: {str(e)}")
        
# %% [markdown]
# ## 5. Main Analysis

# %%
def main():
    """Main analysis function"""
    # Load data
    order_data = load_and_process_data("../Customer Order Quantity_Dispatched Quantity.xlsx")
    
    # Process each material
    results = []
    
    # Process each unique material
    for material_id in order_data['Product ID'].unique():
        material_orders = order_data[order_data['Product ID'] == material_id].copy()
        material_name = material_orders['Product Name'].iloc[0]
        
        print(f"\nProcessing {material_name}...")
        
        # Run XGBoost
        predictions, r2, metrics = xgboost_forecast(material_orders)
        
        if predictions is not None:
            # Store results
            results.append({
                'Material': material_name,
                'R2': r2,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'Sparsity': metrics['Sparsity']
            })
            print(f"R² Score: {r2:.4f}")
            print(f"MAE: {metrics['MAE']:.2f}")
            print(f"RMSE: {metrics['RMSE']:.2f}")
    
    # Create summary DataFrame
    if results:
        results_df = pd.DataFrame(results)
        print("\nOverall Summary Statistics:")
        print(results_df[['R2', 'MAE', 'RMSE']].describe())
        
        # Sort by R2 score and display top/bottom performers
        print("\nTop 5 Materials by R² Score:")
        print(results_df.nlargest(5, 'R2')[['Material', 'R2', 'MAE', 'RMSE']])
        
        print("\nBottom 5 Materials by R² Score:")
        print(results_df.nsmallest(5, 'R2')[['Material', 'R2', 'MAE', 'RMSE']])
        
        # Save results
        results_df.to_csv("xgboost_performance_metrics.csv", index=False)
        
    else:
        print("No results to analyze")

# %%
if __name__ == "__main__":
    main()
