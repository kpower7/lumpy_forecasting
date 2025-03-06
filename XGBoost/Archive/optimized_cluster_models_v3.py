import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df, cluster_num):
    """Prepare features based on cluster characteristics."""
    features = pd.DataFrame(index=df.index)
    
    # Demand features
    for lag in range(1, 13):
        features[f'Demand_lag_{lag}'] = df['Dispatched Quantity'].shift(lag)
    
    for window in [3, 6, 12]:
        features[f'Demand_roll_mean_{window}'] = df['Dispatched Quantity'].rolling(window).mean()
        features[f'Demand_roll_std_{window}'] = df['Dispatched Quantity'].rolling(window).std()
    
    # Market indicators
    market_cols = ['TUAV', 'ECG_DESP', 'ISE_CO']
    for col in market_cols:
        features[f'{col}_lag_1'] = df[col].shift(1)
        features[f'{col}_lag_3'] = df[col].shift(3)
        features[f'{col}_roll_mean_3'] = df[col].rolling(3).mean()
    
    # Economic indicators
    if cluster_num == 0:  # Economic focus
        econ_cols = ['ICI', 'PIB_CO']
        for col in econ_cols:
            for lag in range(1, 7):
                features[f'{col}_lag_{lag}'] = df[col].shift(lag)
            features[f'{col}_roll_mean_6'] = df[col].rolling(6).mean()
    
    # Customer order features
    features['Customer_Order'] = df['Customer Order Quantity']
    features['Order_Demand_Ratio'] = df['Customer Order Quantity'] / df['Dispatched Quantity']
    
    # Time features
    features['Month'] = df.index.month
    features['Quarter'] = df.index.quarter
    features['Year'] = df.index.year - 2022
    
    return features.fillna(0)

def get_param_grid(cluster_num):
    """Get parameter grid based on cluster characteristics."""
    base_grid = {
        'random_state': [42],
        'objective': ['reg:squarederror']
    }
    
    if cluster_num == 0:  # High Volume - Economic focus
        param_grid = {
            'n_estimators': [80, 100, 120],
            'learning_rate': [0.08, 0.1, 0.12],
            'max_depth': [3, 4, 5],
            'min_child_weight': [2, 3],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.05, 0.1, 0.2],
            'reg_lambda': [0.8, 1.0, 1.2]
        }
    
    elif cluster_num == 1:  # Medium Volume - Market focus
        param_grid = {
            'n_estimators': [80, 100, 120],
            'learning_rate': [0.03, 0.05, 0.07],
            'max_depth': [2, 3, 4],
            'min_child_weight': [1, 2, 3],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.05, 0.1, 0.15],
            'reg_lambda': [0.8, 1.0, 1.2]
        }
    
    elif cluster_num == 2:  # Low Volume - Simple model
        param_grid = {
            'n_estimators': [120, 150, 180],
            'learning_rate': [0.02, 0.03, 0.04],
            'max_depth': [2, 3],
            'min_child_weight': [2, 3, 4],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'reg_alpha': [0.1, 0.2, 0.3],
            'reg_lambda': [0.8, 1.0, 1.2]
        }
    
    else:  # Very High Volume - Complex model
        param_grid = {
            'n_estimators': [150, 200, 250],
            'learning_rate': [0.03, 0.05, 0.07],
            'max_depth': [3, 4, 5],
            'min_child_weight': [2, 3, 4],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.1, 0.2, 0.3],
            'reg_lambda': [1.0, 1.5, 2.0]
        }
    
    param_grid.update(base_grid)
    return param_grid

def train_and_evaluate_material(args):
    """Train and evaluate model for a specific material using grid search."""
    material_id, cluster_num, data_dir = args
    try:
        # Load material data
        material_files = list(data_dir.glob(f"{str(material_id).zfill(9)}*.csv"))
        if not material_files:
            return None
        
        # Read and prepare data
        df = pd.read_csv(material_files[0])
        df['YearMonth'] = pd.to_datetime(df['YearMonth'])
        df.set_index('YearMonth', inplace=True)
        
        # Convert numeric columns to float64
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float64)
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        # Prepare features
        X = prepare_features(df, cluster_num)
        y = df['Customer Order Quantity'].astype(np.float64)
        
        # Remove rows with missing values
        mask = y.notna() & X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            return None
        
        # Split data
        train_cutoff = pd.to_datetime('2024-06-01')
        train_mask = df.index[mask] < train_cutoff
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[~train_mask]
        y_test = y[~train_mask]
        
        if len(X_train) < 5 or len(X_test) < 5:
            return None
        
        # Get parameter grid
        param_grid = get_param_grid(cluster_num)
        
        # Initialize base model
        base_model = XGBRegressor()
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='r2',
            cv=min(5, len(X_train) // 10),
            n_jobs=-1,
            verbose=0
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Material_ID': material_id,
            'Cluster': cluster_num,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'Train_Size': len(X_train),
            'Test_Size': len(X_test),
            'Features': list(X.columns),
            'Best_Params': grid_search.best_params_,
            'CV_Score': grid_search.best_score_
        }
        
        # Save best parameters
        params_dir = Path('clustering_results/cluster_models/best_params')
        params_dir.mkdir(parents=True, exist_ok=True)
        
        with open(params_dir / f"{material_id}_params.json", 'w') as f:
            json.dump(grid_search.best_params_, f, indent=4)
        
        return metrics
        
    except Exception as e:
        print(f"Error processing material {material_id}: {str(e)}")
        return None

def main():
    """Main function to train and evaluate models."""
    # Setup paths
    data_dir = Path("Data_files")
    results_file = Path("clustering_results/cluster_models/material_model_results.csv")
    cluster_file = Path("clustering_results/material_clusters.csv")
    
    # Load cluster assignments
    clusters_df = pd.read_csv(cluster_file)
    
    print(f"Processing {len(clusters_df)} materials...")
    
    # Prepare arguments for parallel processing
    args_list = [(row['Material_ID'], row['Cluster'], data_dir) 
                 for _, row in clusters_df.iterrows()]
    
    # Process materials in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(train_and_evaluate_material, args_list))
    
    # Filter out None results and convert to DataFrame
    results = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Save results
        results_df.to_csv(results_file, index=False)
        
        # Print summary statistics
        print("\nSummary Statistics by Cluster:")
        summary = results_df.groupby('Cluster').agg({
            'RMSE': ['mean', 'std', 'min', 'max'],
            'MAE': ['mean', 'std', 'min', 'max'],
            'R2': ['mean', 'std', 'min', 'max'],
            'Material_ID': 'count'
        })
        print(summary)
    else:
        print("No successful models to save")

if __name__ == "__main__":
    main()
