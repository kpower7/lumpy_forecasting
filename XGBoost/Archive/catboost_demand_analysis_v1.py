import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Configure matplotlib for better looking plots
plt.style.use('seaborn-v0_8')

def create_features(data, is_training=True):
    """
    Create features for monthly demand forecasting
    """
    features = data.copy()
    
    # Drop non-numeric columns but keep Product ID as categorical
    features = features.drop(['Product Name'], axis=1, errors='ignore')
    
    # Ensure date is datetime
    if not isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index)
    
    # Basic time features
    features['Month'] = features.index.month.astype(str)
    features['Quarter'] = features.index.quarter.astype(str)
    features['Year'] = features.index.year.astype(str)
    features['YearMonth'] = features.index.strftime('%Y%m')
    features['IsQuarterStart'] = features.index.is_quarter_start.astype(int)
    features['IsQuarterEnd'] = features.index.is_quarter_end.astype(int)
    features['IsYearStart'] = features.index.is_year_start.astype(int)
    features['IsYearEnd'] = features.index.is_year_end.astype(int)
    
    # Convert Product ID to string if it's not already
    features['Product ID'] = features['Product ID'].astype(str)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        features[f'Demand_Lag_{lag}'] = features['Demand'].shift(lag)
        features[f'NonZero_Demand_Lag_{lag}'] = (features[f'Demand_Lag_{lag}'] > 0).astype(int)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        features[f'Rolling_Mean_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).mean()
        features[f'Rolling_Std_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).std()
        features[f'Rolling_Max_{window}m'] = features['Demand'].shift(1).rolling(window=window, min_periods=1).max()
        features[f'NonZero_Mean_{window}m'] = (
            features['Demand']
            .shift(1)
            .where(features['Demand'].shift(1) > 0)
            .rolling(window=window, min_periods=1)
            .mean()
            .fillna(0)
        )
        features[f'Demand_Frequency_{window}m'] = (
            (features['Demand'].shift(1) > 0)
            .rolling(window=window, min_periods=1)
            .mean()
        )
    
    # Year-over-year features
    features['YoY_Change'] = features['Demand'].pct_change(periods=12)
    features['YoY_Ratio'] = features['Demand'] / features['Demand'].shift(12)
    
    # Drop the target column
    y = features['Demand'].copy()
    features = features.drop('Demand', axis=1)
    
    # Define categorical features - ensure they're all strings
    cat_features = ['Product ID', 'YearMonth', 'Month', 'Quarter', 'Year']
    
    # Handle missing values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    if len(features) > 0:
        return features, cat_features, y
    else:
        return None, None, None

def analyze_single_material(material_data, material_name):
    """
    Analyze monthly demand for a single material using CatBoost
    """
    try:
        # Create output directories if they don't exist
        os.makedirs('forecast_plots', exist_ok=True)
        os.makedirs('forecast_metrics', exist_ok=True)
        
        # Split data into train and test sets (last 12 months for test)
        train_data = material_data[:-12].copy()
        test_data = material_data[-12:].copy()
        
        # Create features
        X_train, cat_features, y_train = create_features(train_data, is_training=True)
        X_test, _, y_test = create_features(test_data, is_training=False)
        
        # Initialize CatBoost model
        model = CatBoostRegressor(
            iterations=200,           # Reduced from 1000
            learning_rate=0.1,
            depth=4,                  # Reduced from 6
            l2_leaf_reg=3,
            loss_function='RMSE',
            early_stopping_rounds=20,  # Added early stopping
            verbose=False,
            cat_features=cat_features
        )
        
        # Train the model with eval set for early stopping
        model.fit(
            X_train, 
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate prediction intervals using residuals
        residuals = y_train - train_pred
        residual_std = np.std(residuals)
        z_score = 1.96  # 95% confidence interval
        
        train_lower = train_pred - z_score * residual_std
        train_upper = train_pred + z_score * residual_std
        test_lower = test_pred - z_score * residual_std
        test_upper = test_pred + z_score * residual_std
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Calculate RMSE as percentage of mean demand
        mean_demand = y_train.mean()
        rmse_percentage = (test_rmse / mean_demand) * 100 if mean_demand > 0 else float('inf')
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        plt.plot(train_data.index, y_train, label='Training Actual', color='blue', alpha=0.5)
        plt.plot(train_data.index, train_pred, label='Training Predicted', color='green', linestyle='--')
        plt.fill_between(train_data.index, train_lower, train_upper, color='green', alpha=0.1)
        
        plt.plot(test_data.index, y_test, label='Test Actual', color='blue')
        plt.plot(test_data.index, test_pred, label='Test Predicted', color='red', linestyle='--')
        plt.fill_between(test_data.index, test_lower, test_upper, color='red', alpha=0.1)
        
        plt.title(f'{material_name} Demand Forecast\nCatBoost Model, R² = {test_r2:.3f}, RMSE% = {rmse_percentage:.1f}%')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig(f'forecast_plots/{material_name}_forecast.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create metrics DataFrame
        train_metrics = pd.DataFrame({
            'Date': train_data.index,
            'Actual': y_train,
            'Predicted': train_pred,
            'Lower_Bound': train_lower,
            'Upper_Bound': train_upper,
            'Dataset': 'Train'
        })
        
        test_metrics = pd.DataFrame({
            'Date': test_data.index,
            'Actual': y_test,
            'Predicted': test_pred,
            'Lower_Bound': test_lower,
            'Upper_Bound': test_upper,
            'Dataset': 'Test'
        })
        
        # Combine train and test metrics
        metrics_df = pd.concat([train_metrics, test_metrics])
        metrics_df.to_csv(f'forecast_metrics/{material_name}_metrics.csv', index=False)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': model.feature_names_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        feature_importance.to_csv(f'forecast_metrics/{material_name}_feature_importance.csv', index=False)
        
        return {
            'material_name': material_name,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse_percentage': rmse_percentage,
            'top_features': feature_importance['Feature'].head(5).tolist(),
            'top_importance': feature_importance['Importance'].head(5).tolist()
        }
        
    except Exception as e:
        print(f"Error analyzing {material_name}: {str(e)}")
        return None

def main():
    # Read the data
    data = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Order_Quantity_Monthly.csv")
    data['YearMonth'] = pd.to_datetime(data['YearMonth'])
    data.set_index('YearMonth', inplace=True)
    
    # Rename Dispatched Quantity to Demand for consistency
    data = data.rename(columns={'Dispatched Quantity': 'Demand'})
    
    # Get unique materials
    materials = data['Product ID'].unique()
    
    # Store results for each material
    results = []
    
    total_materials = len(materials)
    print(f"Starting analysis of {total_materials} materials...")
    
    for i, material in enumerate(materials, 1):
        print(f"\nProcessing material {i}/{total_materials}")
        material_data = data[data['Product ID'] == material].copy()
        material_name = material_data['Product Name'].iloc[0]
        print(f"Material: {material_name}")
        
        result = analyze_single_material(material_data, material_name)
        if result is not None:
            results.append(result)
            print(f"[SUCCESS] Completed analysis for {material_name}")
        else:
            print(f"[FAILED] Could not analyze {material_name}")
    
    # Save overall results
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate and add summary statistics
        summary_stats = {
            'mean_test_r2': results_df['test_r2'].mean(),
            'median_test_r2': results_df['test_r2'].median(),
            'mean_rmse_pct': results_df['rmse_percentage'].mean(),
            'median_rmse_pct': results_df['rmse_percentage'].median()
        }
        
        # Save results
        results_df.to_csv('catboost_forecast_results.csv', index=False)
        
        # Save summary
        with open('catboost_summary.txt', 'w') as f:
            f.write("CatBoost Model Performance Summary\n")
            f.write("=================================\n\n")
            f.write(f"Total materials analyzed: {len(results_df)}\n")
            f.write(f"Mean Test R²: {summary_stats['mean_test_r2']:.3f}\n")
            f.write(f"Median Test R²: {summary_stats['median_test_r2']:.3f}\n")
            f.write(f"Mean RMSE%: {summary_stats['mean_rmse_pct']:.1f}%\n")
            f.write(f"Median RMSE%: {summary_stats['median_rmse_pct']:.1f}%\n")
        
        print("\nAnalysis complete!")
        print(f"Results saved to catboost_forecast_results.csv")
        print(f"Summary saved to catboost_summary.txt")
        
        # Print top features across all materials
        all_top_features = []
        for features in results_df['top_features']:
            all_top_features.extend(features)
        
        top_feature_counts = pd.Series(all_top_features).value_counts()
        print("\nMost Important Features Overall:")
        print(top_feature_counts.head())
        
    else:
        print("\nNo successful analyses to save.")

if __name__ == "__main__":
    main()
