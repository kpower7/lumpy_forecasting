import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import os

def analyze_cluster_features(cluster_num):
    """Analyze feature importance for a specific cluster."""
    # Load cluster mapping
    cluster_mapping = pd.read_csv('cluster_analysis_plots/material_cluster_mapping.csv')
    
    # Get materials in this cluster
    cluster_materials = cluster_mapping[cluster_mapping['Cluster'] == cluster_num]['Material_ID'].tolist()
    
    # Load and combine data for all materials in cluster
    data_dir = r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Data_files"
    cluster_data = []
    
    for material_id in cluster_materials:
        material_id = str(material_id).zfill(9)
        material_files = [f for f in os.listdir(data_dir) if f.startswith(material_id)]
        
        if material_files:
            df = pd.read_csv(os.path.join(data_dir, material_files[0]))
            df['YearMonth'] = pd.to_datetime(df['YearMonth'])
            cluster_data.append(df)
    
    combined_data = pd.concat(cluster_data, ignore_index=True)
    
    # Create features based on cluster characteristics
    if cluster_num == 0:  # High Volume Cluster
        # Focus on recent trends and economic indicators
        features = {
            'ECG_DESP': combined_data['ECG_DESP'],
            'PIB_CO': combined_data['PIB_CO'],
            'ISE_CO': combined_data['ISE_CO'],
            'ICI': combined_data['ICI'],
            'Month': combined_data['YearMonth'].dt.month,
            'Quarter': combined_data['YearMonth'].dt.quarter,
            'ECG_DESP_3M': combined_data.groupby('Product ID')['ECG_DESP'].rolling(3).mean().reset_index(0, drop=True),
            'PIB_CO_3M': combined_data.groupby('Product ID')['PIB_CO'].rolling(3).mean().reset_index(0, drop=True),
        }
    
    elif cluster_num == 1:  # Medium Volume, Market Sensitive
        # Focus on market indicators and seasonality
        features = {
            'TUAV': combined_data['TUAV'],
            'VTOTAL_19': combined_data['VTOTAL_19'],
            'OTOTAL_19': combined_data['OTOTAL_19'],
            'Month': combined_data['YearMonth'].dt.month,
            'Quarter': combined_data['YearMonth'].dt.quarter,
            'TUAV_3M': combined_data.groupby('Product ID')['TUAV'].rolling(3).mean().reset_index(0, drop=True),
            'VTOTAL_19_3M': combined_data.groupby('Product ID')['VTOTAL_19'].rolling(3).mean().reset_index(0, drop=True),
        }
    
    elif cluster_num == 2:  # Low Volume, Stable
        # Focus on basic patterns and longer trends
        features = {
            'Month': combined_data['YearMonth'].dt.month,
            'Quarter': combined_data['YearMonth'].dt.quarter,
            'ECG_DESP': combined_data['ECG_DESP'],
            'TUAV': combined_data['TUAV'],
            'ECG_DESP_6M': combined_data.groupby('Product ID')['ECG_DESP'].rolling(6).mean().reset_index(0, drop=True),
            'TUAV_6M': combined_data.groupby('Product ID')['TUAV'].rolling(6).mean().reset_index(0, drop=True),
        }
    
    elif cluster_num == 3:  # Very High Volume
        # Focus on all indicators with emphasis on economic factors
        features = {
            'ECG_DESP': combined_data['ECG_DESP'],
            'TUAV': combined_data['TUAV'],
            'PIB_CO': combined_data['PIB_CO'],
            'ISE_CO': combined_data['ISE_CO'],
            'ICI': combined_data['ICI'],
            'Month': combined_data['YearMonth'].dt.month,
            'Quarter': combined_data['YearMonth'].dt.quarter,
            'ECG_DESP_3M': combined_data.groupby('Product ID')['ECG_DESP'].rolling(3).mean().reset_index(0, drop=True),
            'PIB_CO_3M': combined_data.groupby('Product ID')['PIB_CO'].rolling(3).mean().reset_index(0, drop=True),
            'TUAV_3M': combined_data.groupby('Product ID')['TUAV'].rolling(3).mean().reset_index(0, drop=True),
        }
    
    # Create feature DataFrame
    X = pd.DataFrame(features)
    y = combined_data['Customer Order Quantity']
    
    # Remove NaN rows
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    # Train a simple XGBoost model to get feature importance
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df, X.columns.tolist()

def get_cluster_specific_model(cluster_num, important_features):
    """Get cluster-specific XGBoost model configuration."""
    
    if cluster_num == 0:  # High Volume Cluster
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    elif cluster_num == 1:  # Medium Volume, Market Sensitive
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
    
    elif cluster_num == 2:  # Low Volume, Stable
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=4,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42
        )
    
    else:  # Very High Volume
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

def main():
    output_dir = Path(r'C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\clustering_results\cluster_models')
    output_dir.mkdir(exist_ok=True)
    
    # Analyze each cluster
    for cluster in range(4):
        print(f"\nAnalyzing Cluster {cluster}")
        
        # Get feature importance and features list
        importance_df, features = analyze_cluster_features(cluster)
        
        # Save feature importance
        importance_df.to_csv(output_dir / f'cluster_{cluster}_feature_importance.csv', index=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['Feature'], importance_df['Importance'])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Feature Importance for Cluster {cluster}')
        plt.tight_layout()
        plt.savefig(output_dir / f'cluster_{cluster}_feature_importance.png')
        plt.close()
        
        print(f"\nTop features for Cluster {cluster}:")
        print(importance_df.head())
        
        # Get cluster-specific model
        model = get_cluster_specific_model(cluster, features)
        print(f"\nModel parameters for Cluster {cluster}:")
        print(model.get_params())

if __name__ == "__main__":
    main()
