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
    """Implement XGBoost model with feature engineering"""
    try:
        # Check if we have enough data points
        if len(df) < 60:  # Require at least 60 days of data
            print("Not enough data points for reliable forecasting")
            return None, None, None
            
        # Calculate sparsity
        sparsity = (df['Customer Order Quantity'] == 0).mean()
        print(f"Order sparsity: {sparsity:.2%}")
        
        # Handle outliers
        Q1 = df['Customer Order Quantity'].quantile(0.25)
        Q3 = df['Customer Order Quantity'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df.copy()
        outliers = df_clean['Customer Order Quantity'] > upper_bound
        if outliers.any():
            print(f"Found {outliers.sum()} outliers ({outliers.sum()/len(df)*100:.1f}% of data)")
            df_clean.loc[outliers, 'Customer Order Quantity'] = upper_bound
        
        # First split the data BEFORE any feature engineering
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx].copy()
        test_df = df_clean.iloc[split_idx:].copy()
        
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        def create_features(data, is_training=True):
            """Create time series features from datetime index"""
            df_features = data.copy()
            
            # Create date features
            df_features['DayOfWeek'] = df_features['Date'].dt.dayofweek
            df_features['Month'] = df_features['Date'].dt.month
            df_features['Year'] = df_features['Date'].dt.year
            
            # Cyclical encoding of time features
            df_features['Month_Sin'] = np.sin(2 * np.pi * df_features['Month']/12)
            df_features['Month_Cos'] = np.cos(2 * np.pi * df_features['Month']/12)
            df_features['DayOfWeek_Sin'] = np.sin(2 * np.pi * df_features['DayOfWeek']/7)
            df_features['DayOfWeek_Cos'] = np.cos(2 * np.pi * df_features['DayOfWeek']/7)
            
            # Forward-looking features only
            if is_training:
                # For training data, create all features
                for i in range(1, 8):  # Previous 7 days
                    df_features[f'order_lag_{i}'] = df_features.groupby('Product ID')['Customer Order Quantity'].shift(i)
                
                for window in [7, 14, 30]:
                    df_features[f'order_rolling_mean_{window}'] = df_features.groupby('Product ID')['Customer Order Quantity'].rolling(
                        window=window, min_periods=1).mean().reset_index(0, drop=True).shift(1)
                    df_features[f'order_rolling_std_{window}'] = df_features.groupby('Product ID')['Customer Order Quantity'].rolling(
                        window=window, min_periods=1).std().reset_index(0, drop=True).shift(1)
                    df_features[f'days_since_order_{window}'] = df_features.groupby('Product ID')['Customer Order Quantity'].rolling(
                        window=window, min_periods=1).apply(lambda x: (x == 0).sum()).reset_index(0, drop=True).shift(1)
            else:
                # For test data, ensure we only use past data
                for i in range(1, 8):
                    df_features[f'order_lag_{i}'] = df_features.groupby('Product ID')['Customer Order Quantity'].shift(i)
                
                for window in [7, 14, 30]:
                    df_features[f'order_rolling_mean_{window}'] = df_features.groupby('Product ID')['Customer Order Quantity'].rolling(
                        window=window, min_periods=1).mean().reset_index(0, drop=True).shift(1)
                    df_features[f'order_rolling_std_{window}'] = df_features.groupby('Product ID')['Customer Order Quantity'].rolling(
                        window=window, min_periods=1).std().reset_index(0, drop=True).shift(1)
                    df_features[f'days_since_order_{window}'] = df_features.groupby('Product ID')['Customer Order Quantity'].rolling(
                        window=window, min_periods=1).apply(lambda x: (x == 0).sum()).reset_index(0, drop=True).shift(1)
            
            # Fill NaN values with appropriate defaults
            for col in df_features.columns:
                if col.startswith('order_lag_'):
                    df_features[col] = df_features[col].fillna(0)
                elif col.startswith('order_rolling_mean_'):
                    df_features[col] = df_features[col].fillna(df_features['Customer Order Quantity'].mean())
                elif col.startswith('order_rolling_std_'):
                    df_features[col] = df_features[col].fillna(0)
                elif col.startswith('days_since_order_'):
                    df_features[col] = df_features[col].fillna(0)
            
            return df_features
        
        # Create features separately for train and test
        train_features = create_features(train_df, is_training=True)
        test_features = create_features(test_df, is_training=False)
        
        # Select features for modeling
        feature_cols = [col for col in train_features.columns if col not in 
                       ['Customer Order Quantity', 'Product ID', 'Product Name', 
                        'Dispatched Quantity', 'Date']]
        
        # Prepare X and y for training
        X_train = train_features[feature_cols]
        y_train = train_features['Customer Order Quantity']
        
        # Prepare X and y for testing
        X_test = test_features[feature_cols]
        y_test = test_features['Customer Order Quantity']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Adjust model parameters based on sparsity
        if sparsity > 0.7:  # If more than 70% zeros
            params = {
                'n_estimators': 300,
                'learning_rate': 0.03,
                'max_depth': 4,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        else:
            params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        
        # Train model
        model = XGBRegressor(**params)
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        model.fit(
            X_train_scaled, 
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Generate predictions only for test set
        y_pred_test = model.predict(X_test_scaled)
        
        # Ensure predictions are non-negative
        y_pred_test = np.maximum(0, y_pred_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-6))) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Sparsity': sparsity
        }
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['Feature_Importance'] = feature_importance
        
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
            # Create plots
            plot_actual_vs_predicted(material_orders, predictions, material_name)
            plot_feature_importance(metrics['Feature_Importance'], material_name)
            
            # Store results
            results.append({
                'Material': material_name,
                'R2': r2,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'Sparsity': metrics['Sparsity']
            })
    
    # Create summary DataFrame
    if results:
        results_df = pd.DataFrame(results)
        print("\nSummary Statistics:")
        print(results_df.describe())
        
        # Save results
        results_df.to_csv("xgboost_detailed_results.csv", index=False)
        
        # Calculate and save error analysis
        error_analysis = pd.DataFrame({
            'Metric': ['R2', 'MAE', 'RMSE'],
            'Mean': [results_df['R2'].mean(), results_df['MAE'].mean(), results_df['RMSE'].mean()],
            'Std': [results_df['R2'].std(), results_df['MAE'].std(), results_df['RMSE'].std()],
            'Min': [results_df['R2'].min(), results_df['MAE'].min(), results_df['RMSE'].min()],
            'Max': [results_df['R2'].max(), results_df['MAE'].max(), results_df['RMSE'].max()]
        })
        error_analysis.to_csv("xgboost_error_analysis.csv", index=False)
    else:
        print("No results to analyze")

# %%
if __name__ == "__main__":
    main()
