import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def add_rolling_features(df, col, windows=[3, 6]):
    """Add rolling average features for different windows."""
    for window in windows:
        df[f'{col}_{window}M'] = df[col].rolling(window, min_periods=1).mean()

def prepare_features(df, cluster_num):
    """Prepare features based on cluster-specific importance."""
    df = df.copy()
    
    # Common time features
    df['Month'] = df['YearMonth'].dt.month
    df['Quarter'] = df['YearMonth'].dt.quarter
    
    if cluster_num == 0:  # High Volume - Focus on economic indicators
        # Add rolling averages
        add_rolling_features(df, 'PIB_CO')
        add_rolling_features(df, 'ECG_DESP')
        
        return df[['PIB_CO', 'PIB_CO_3M', 'PIB_CO_6M', 
                  'ECG_DESP', 'ECG_DESP_3M', 'ECG_DESP_6M',
                  'ICI', 'Month', 'Quarter']]
        
    elif cluster_num == 1:  # Medium Volume - Focus on market indicators
        # Add rolling averages
        add_rolling_features(df, 'TUAV')
        add_rolling_features(df, 'VTOTAL_19')
        
        return df[['TUAV', 'TUAV_3M', 'TUAV_6M',
                  'VTOTAL_19', 'VTOTAL_19_3M', 'VTOTAL_19_6M',
                  'OTOTAL_19', 'Month', 'Quarter']]
        
    elif cluster_num == 2:  # Low Volume - Focus on basic patterns
        # Add rolling averages
        add_rolling_features(df, 'ECG_DESP', windows=[6, 12])
        
        return df[['ECG_DESP', 'ECG_DESP_6M', 'ECG_DESP_12M',
                  'Month', 'Quarter']]
    
    else:  # Very High Volume - All indicators
        # Add rolling averages
        add_rolling_features(df, 'PIB_CO')
        add_rolling_features(df, 'ECG_DESP')
        add_rolling_features(df, 'TUAV')
        
        return df[['PIB_CO', 'PIB_CO_3M', 'PIB_CO_6M',
                  'ECG_DESP', 'ECG_DESP_3M', 'ECG_DESP_6M',
                  'TUAV', 'TUAV_3M', 'TUAV_6M',
                  'ICI', 'ISE_CO', 'Month', 'Quarter']]

def get_optimized_model(cluster_num):
    """Get optimized XGBoost model based on cluster characteristics."""
    if cluster_num == 0:  # High Volume - Economic focus
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42
        )
    
    elif cluster_num == 1:  # Medium Volume - Market focus
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.05,
            random_state=42
        )
    
    elif cluster_num == 2:  # Low Volume - Simple model
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=4,
            subsample=1.0,
            colsample_bytree=1.0,
            gamma=0.2,
            random_state=42
        )
    
    else:  # Very High Volume - Complex model
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.05,
            random_state=42
        )

def train_and_evaluate_material(material_id, cluster_num, data_dir):
    """Train and evaluate model for a specific material."""
    try:
        # Load material data
        material_files = list(data_dir.glob(f"{str(material_id).zfill(9)}*.csv"))
        if not material_files:
            return None
        
        # Read and prepare data
        df = pd.read_csv(material_files[0])
        df['YearMonth'] = pd.to_datetime(df['YearMonth'])
        
        # Convert numeric columns to float64
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.float64)
        
        # Forward fill missing values
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        # Prepare features
        X = prepare_features(df, cluster_num)
        y = df['Customer Order Quantity'].astype(np.float64)
        
        # Remove rows with missing values
        mask = y.notna() & X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:  # Not enough data
            return None
        
        # Split data
        train_cutoff = pd.to_datetime('2024-06-01')
        train_mask = df['YearMonth'][mask] < train_cutoff
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[~train_mask]
        y_test = y[~train_mask]
        
        if len(X_train) < 5 or len(X_test) < 5:  # Not enough data for train/test
            return None
        
        # Train model
        model = get_optimized_model(cluster_num)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'Material_ID': material_id,
            'Cluster': cluster_num,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'Train_Size': len(X_train),
            'Test_Size': len(X_test),
            'Features': ', '.join(X.columns)
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error processing material {material_id}: {str(e)}")
        return None

def main():
    # Set up paths
    base_dir = Path(r'C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC')
    data_dir = base_dir / 'Data_files'
    output_dir = base_dir / 'clustering_results' / 'cluster_models'
    cluster_mapping_path = base_dir / 'cluster_analysis_plots' / 'material_cluster_mapping.csv'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load cluster mapping
        if not cluster_mapping_path.exists():
            raise FileNotFoundError(f"Cluster mapping file not found at: {cluster_mapping_path}")
        
        cluster_mapping = pd.read_csv(cluster_mapping_path)
        
        # Process each material
        results = []
        total_materials = len(cluster_mapping)
        
        print(f"Processing {total_materials} materials...")
        
        for idx, row in cluster_mapping.iterrows():
            material_id = row['Material_ID']
            cluster_num = row['Cluster']
            
            print(f"\rProcessing material {idx + 1}/{total_materials}: {material_id}", end='')
            
            metrics = train_and_evaluate_material(material_id, cluster_num, data_dir)
            if metrics:
                results.append(metrics)
        
        print("\nCompleted processing all materials")
        
        if results:
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Save results
            results_path = output_dir / 'material_model_results.csv'
            results_df.to_csv(results_path, index=False)
            
            # Print summary statistics
            print("\nSummary Statistics by Cluster:")
            summary = results_df.groupby('Cluster').agg({
                'RMSE': ['mean', 'std'],
                'MAE': ['mean', 'std'],
                'R2': ['mean', 'std'],
                'Material_ID': 'count'
            }).round(2)
            
            print(summary)
            print(f"\nDetailed results saved to: {results_path}")
        else:
            print("\nNo successful models to save")
            
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
