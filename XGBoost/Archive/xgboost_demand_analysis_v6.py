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
plt.style.use('seaborn-v0_8')  # Updated style name for newer versions

# %% [markdown]
# ## 1. Data Loading and Preprocessing

# %%
def load_and_process_data():
    """Load and process order data"""
    print("Loading data...")
    
    # Read the data files
    df_orders = pd.read_excel(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx")
    
    # Convert date column
    df_orders['Date'] = pd.to_datetime(df_orders['Date'], format='%d.%m.%Y')
    
    return df_orders

# %% [markdown]
# ## 2. XGBoost Model Implementation

# %%
def xgboost_forecast(df):
    """Implement XGBoost model with feature engineering"""
    try:
        # Ensure data is sorted chronologically
        df = df.sort_index()
        
        def create_features(data, is_training=True, scaler=None, last_train_values=None):
            """Create features using only the data available at prediction time"""
            features = data.copy()
            
            # Add time-based features (these are safe as they don't use target variable)
            features['Month'] = features.index.month
            features['Month_Sin'] = np.sin(2 * np.pi * features['Month']/12)
            features['Month_Cos'] = np.cos(2 * np.pi * features['Month']/12)
            features['DayOfWeek'] = features.index.dayofweek
            features['Quarter'] = features.index.quarter
            
            if is_training:
                # For training data, we can calculate lags and rolling means normally
                for lag in [1, 2, 3]:
                    features[f'order_lag_{lag}'] = features['Customer Order Quantity'].shift(lag)
                
                for window in [3, 6]:
                    features[f'order_rolling_mean_{window}'] = features['Customer Order Quantity'].rolling(
                        window=window, min_periods=1
                    ).mean()
                
                # Store the last values needed for test set feature creation
                last_values = {
                    'last_orders': features['Customer Order Quantity'].iloc[-3:].values,  # Store last 3 values for lags
                    'last_rolling_3': features['Customer Order Quantity'].rolling(window=3).mean().iloc[-1],
                    'last_rolling_6': features['Customer Order Quantity'].rolling(window=6).mean().iloc[-1]
                }
            else:
                # For test data, use only historical information
                last_orders = last_train_values['last_orders']
                for lag in [1, 2, 3]:
                    features[f'order_lag_{lag}'] = pd.Series(index=features.index)
                    features.loc[features.index[0], f'order_lag_{lag}'] = last_orders[-lag]
                    features[f'order_lag_{lag}'].fillna(method='ffill', inplace=True)
                
                # Initialize rolling means with training values
                features['order_rolling_mean_3'] = last_train_values['last_rolling_3']
                features['order_rolling_mean_6'] = last_train_values['last_rolling_6']
                
                # Update rolling means one step at a time to avoid lookahead
                for i in range(1, len(features)):
                    prev_data = features['Customer Order Quantity'].iloc[:i]
                    if len(prev_data) >= 3:
                        features.loc[features.index[i], 'order_rolling_mean_3'] = prev_data[-3:].mean()
                    if len(prev_data) >= 6:
                        features.loc[features.index[i], 'order_rolling_mean_6'] = prev_data[-6:].mean()
            
            # Drop rows with NaN values
            features = features.dropna()
            
            # Prepare feature columns
            feature_cols = [col for col in features.columns 
                          if col.startswith(('order_lag_', 'order_rolling_', 'Month_', 'DayOfWeek', 'Quarter'))]
            
            if is_training:
                return features[feature_cols], features['Customer Order Quantity'], feature_cols, last_values
            return features[feature_cols], features['Customer Order Quantity'], feature_cols
        
        # Calculate the split point to ensure test set is the most recent 20% of data
        split_date = df.index[int(len(df) * 0.8)]
        
        # Split data ensuring chronological order
        train_df = df[:split_date].copy()
        test_df = df[split_date:].copy()
        
        print(f"Training data from {train_df.index.min()} to {train_df.index.max()}")
        print(f"Testing data from {test_df.index.min()} to {test_df.index.max()}")
        
        # Create features for training data and get last values for test set
        X_train, y_train, feature_cols, last_train_values = create_features(train_df, is_training=True)
        
        # Scale features using only training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Create features for test data using only historical information
        X_test, y_test, _ = create_features(test_df, is_training=False, 
                                          scaler=scaler, 
                                          last_train_values=last_train_values)
        X_test_scaled = scaler.transform(X_test)
        
        if len(X_train_scaled) < 2:
            print("Not enough data for XGBoost model")
            return None, None, None
        
        # Train model with conservative parameters and non-negative constraint
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=5,  # Increased to reduce overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            base_score=0.5,  # Start predictions from middle of range
            gamma=1  # Add minimum loss reduction for split
        )
        
        # Train model
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # Generate predictions
        predictions = pd.Series(index=df.index, data=np.nan)
        
        # Fill in predictions where we have features and clip at 0
        train_preds = np.clip(model.predict(X_train_scaled), 0, None)
        test_preds = np.clip(model.predict(X_test_scaled), 0, None)
        
        predictions[X_train.index] = train_preds
        predictions[X_test.index] = test_preds
        
        # Calculate metrics on test set only
        def custom_r2(y_true, y_pred):
            """Calculate R² score with detailed logging"""
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            print("\nDetailed R² Calculation:")
            print(f"Mean of true values: {np.mean(y_true):.2f}")
            print(f"Sum of squared residuals (ss_res): {ss_res:.2f}")
            print(f"Total sum of squares (ss_tot): {ss_tot:.2f}")
            print(f"R² components: 1 - ({ss_res:.2f} / {ss_tot:.2f})")
            print(f"Calculated R²: {r2:.3f}")
            print(f"SKLearn R²: {r2_score(y_true, y_pred):.3f}")
            
            # Additional diagnostics
            print("\nPrediction Statistics:")
            print(f"Mean of predictions: {np.mean(y_pred):.2f}")
            print(f"Std of true values: {np.std(y_true):.2f}")
            print(f"Std of predictions: {np.std(y_pred):.2f}")
            print(f"Min prediction: {np.min(y_pred):.2f}")
            print(f"Max prediction: {np.max(y_pred):.2f}")
            
            return r2
        
        r2 = custom_r2(y_test, test_preds)
        mae = mean_absolute_error(y_test, test_preds)
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
        return predictions, r2, {'MAE': mae, 'RMSE': rmse, 'Feature_Importance': dict(zip(feature_cols, model.feature_importances_))}
        
    except Exception as e:
        print(f"Error in XGBoost: {str(e)}")
        return None, None, None

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
        'RMSE': results_df['RMSE']
    }).sort_values('R2_Score', ascending=False)
    
    return error_analysis

# %% [markdown]
# ## 5. Main Analysis

# %%
def main():
    # Load data
    order_data = load_and_process_data()
    
    # Process each material
    results = []
    all_feature_importance = []
    
    # Define the correct material ID mapping
    material_ids = [
        '555136414',   # Material_1
        '555138033',   # Material_2
        '555138034',   # Material_3
        '000161032',   # Material_4
        '555171963',   # Material_5
        '555176702',   # Material_6
        '555179704',   # Material_7
        '555179719',   # Material_8
        '555179740',   # Material_9
        '555179748',   # Material_10
        '555179752',   # Material_11
        '000179754',   # Material_12
        '555179754',   # Material_13
        '000179758',   # Material_14
        '555179777',   # Material_15
        '555179781',   # Material_16
        '000180631',   # Material_17
        '555180631',   # Material_18
        '000180632',   # Material_19
        '555180632',   # Material_20
        '333180636',   # Material_21
        '333181685',   # Material_22
        '555193112',   # Material_23
        '333193136',   # Material_24
        '000193143',   # Material_25
        '555193169',   # Material_26
        '000193597',   # Material_27
        '000193598',   # Material_28
        '000193610',   # Material_29
        '333193714',   # Material_30
        '555193726',   # Material_31
        '555193750',   # Material_32
        '333193751',   # Material_33
        '333193796',   # Material_34
        '555194506',   # Material_35
        '555196704',   # Material_36
        '000196706',   # Material_37
        '333196762',   # Material_38
        '333196767',   # Material_39
        '000196768',   # Material_40
        '000196769',   # Material_41
        '000196770',   # Material_42
        '555196770',   # Material_43
        '555196772',   # Material_44
        '555197704',   # Material_45
        '555197707',   # Material_46
        '000197792',   # Material_47
        '333197792',   # Material_48
        '000197793',   # Material_49
        '333197793',   # Material_50
        '0001O1010',   # Material_51
        '5551O1032',   # Material_52
        '5551O1063',   # Material_53
        '5551O1138',   # Material_54
        '0001O1142',   # Material_55
        '0001O1184',   # Material_56
        '5551O1184',   # Material_57
        '5551O1239',   # Material_58
        '5551O1335',   # Material_59
        '5551O1350',   # Material_60
        '5551O1475',   # Material_61
        '5551O1476',   # Material_62
        '0001O1809',   # Material_63
        '0001O1864',   # Material_64
        '0001O2127',   # Material_65
        '0001O2130',   # Material_66
        '0001O2153',   # Material_67
        '0001O2290',   # Material_68
        '3331O2298',   # Material_69
        '0001O2471',   # Material_70
        '0001O2475',   # Material_71
        '0001O2477',   # Material_72
        '5551O2494',   # Material_73
        '0001O2502',   # Material_74
        '5551O2515',   # Material_75
        '5551O2518',   # Material_76
        '0001O3160',   # Material_77
        '0001O3245',   # Material_78
        '5551O3273',   # Material_79
        '0001O3401',   # Material_80
        '1000O3770',   # Material_81
        '0001PZ701',   # Material_82
        '0001PZ702',   # Material_83
        '0001PZ706',   # Material_84
    ]
    material_mapping = {id: f"Material_{i+1}" for i, id in enumerate(material_ids)}
    
    print("\nProcessing materials...")
    for material_id in material_ids:
        # Filter data for the specific material
        material_orders = order_data[order_data['Product ID'] == material_id].copy()
        material_name = material_mapping[material_id]
        
        if len(material_orders) == 0:
            continue
            
        # Set date as index and sort
        material_orders.set_index('Date', inplace=True)
        material_orders.sort_index(inplace=True)
        
        print(f"\nProcessing {material_name}...")
        
        # Run XGBoost
        predictions, r2, metrics = xgboost_forecast(material_orders)
        
        if predictions is not None:
            # Create plots
            plot_forecast(material_orders, predictions, material_name, r2, metrics)
            plot_feature_importance(metrics['Feature_Importance'], material_name)
            
            # Store results
            results.append({
                'Material': material_name,
                'R2_Score': r2,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'Feature_Importance': metrics['Feature_Importance']
            })
            
            print(f"R² = {r2:.3f}, MAE = {metrics['MAE']:.2f}, RMSE = {metrics['RMSE']:.2f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate performance distribution plot
    analyze_performance_distribution(results_df)
    
    # Analyze error patterns
    error_analysis = analyze_error_patterns(results_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average R² Score: {results_df['R2_Score'].mean():.3f}")
    print(f"Median R² Score: {results_df['R2_Score'].median():.3f}")
    print(f"Standard Deviation of R² Score: {results_df['R2_Score'].std():.3f}")
    print(f"\nTop 5 Best Performing Materials:")
    print(error_analysis.head().to_string())
    print(f"\nBottom 5 Performing Materials:")
    print(error_analysis.tail().to_string())
    
    # Save detailed results
    results_df.to_csv('xgboost_detailed_results.csv', index=False)
    error_analysis.to_csv('xgboost_error_analysis.csv', index=False)

# %%
if __name__ == "__main__":
    main()
