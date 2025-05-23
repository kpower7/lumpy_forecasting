import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import sys
from typing import Dict, Any
from tqdm import tqdm
from pathlib import Path
import datetime

# Import cluster-specific forecast modules
import demand_forecast_cluster0 as cluster0
import demand_forecast_cluster1 as cluster1
import demand_forecast_cluster2 as cluster2
import demand_forecast_cluster3 as cluster3

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('material_processing_clustered.log'),
        logging.StreamHandler()
    ]
)

# Map cluster numbers to their respective forecast functions
CLUSTER_FUNCTIONS = {
    0: cluster0.main,
    1: cluster1.main,
    2: cluster2.main,
    3: cluster3.main
}

def find_optimal_clusters(scaled_features):
    """Find optimal number of clusters using elbow method and silhouette analysis."""
    max_clusters = 10
    inertias = []
    silhouette_scores = []
    
    # KMeans parameters for better stability
    kmeans_params = {
        'init': 'k-means++',  # Use k-means++ initialization
        'n_init': 25,         # Number of times to run with different centroid seeds
        'max_iter': 300,      # Maximum number of iterations
        'tol': 1e-4,         # Tolerance for declaring convergence
        'random_state': 42    # For reproducibility
    }
    
    for k in range(2, max_clusters + 1):
        # K-means clustering with optimized parameters
        kmeans = KMeans(n_clusters=k, **kmeans_params)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
        
        # Silhouette score
        silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        print(f"\nk={k}:")
        print(f"Inertia: {kmeans.inertia_:.2f}")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Iterations to converge: {kmeans.n_iter_}")
    
    # Convert to numpy array for easier math
    inertias = np.array(inertias)
    
    # Calculate first and second derivatives
    first_derivative = np.gradient(inertias)
    second_derivative = np.gradient(first_derivative)
    
    # Find the elbow point - maximum of second derivative
    # Add 2 because we started with k=2
    elbow_k = np.argmax(np.abs(second_derivative)) + 2
    
    # Find k with highest silhouette score
    silhouette_k = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Plot results with derivatives
    plt.figure(figsize=(15, 5))
    
    # Plot elbow curve
    plt.subplot(1, 3, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.plot(range(2, max_clusters + 1), inertias, 'r--', alpha=0.3)
    plt.axvline(x=elbow_k, color='g', linestyle='--', label=f'Elbow k={elbow_k}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.legend()
    
    # Plot first derivative
    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_clusters + 1), first_derivative, marker='o')
    plt.axvline(x=elbow_k, color='g', linestyle='--', label=f'Elbow k={elbow_k}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('First Derivative')
    plt.title('Rate of Change')
    plt.legend()
    
    # Plot second derivative
    plt.subplot(1, 3, 3)
    plt.plot(range(2, max_clusters + 1), second_derivative, marker='o')
    plt.axvline(x=elbow_k, color='g', linestyle='--', label=f'Elbow k={elbow_k}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Second Derivative')
    plt.title('Rate of Change of Rate of Change')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'r--', alpha=0.3)
    plt.axvline(x=silhouette_k, color='b', linestyle='--', label=f'Best k={silhouette_k}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.legend()
    plt.show()
    
    print(f"\nAnalysis:")
    print(f"Silhouette analysis suggests {silhouette_k} clusters (score: {max(silhouette_scores):.3f})")
    print(f"Elbow method suggests {elbow_k} clusters (based on maximum curvature)")
    
    # Combine both methods - prefer elbow if silhouette score difference is small
    silhouette_scores_arr = np.array(silhouette_scores)
    elbow_score = silhouette_scores_arr[elbow_k - 2]  # -2 because we started at k=2
    best_score = silhouette_scores_arr[silhouette_k - 2]
    
    if best_score - elbow_score < 0.1:  # If scores are close, prefer elbow
        optimal_k = min(elbow_k, 3)
        print(f"\nUsing elbow method k={elbow_k} (silhouette scores similar)")
    else:
        optimal_k = min(silhouette_k, 3)
        print(f"\nUsing silhouette k={silhouette_k} (significantly better score)")
    
    return optimal_k

def perform_clustering(features_df):
    """Perform clustering on materials based on their features."""
    # Select numerical features for clustering with weights
    feature_groups = {
        'demand': {
            'cols': ['Avg_Monthly_Demand', 'Demand_Std', 'Demand_CV'],
            'weight': 1.0,
            'scaler': MinMaxScaler()  # Preserve zeros, scale to [0,1]
        },
        'fulfillment': {
            'cols': ['Order_Fulfillment'],
            'weight': 1.5,  # Give more weight to fulfillment rate
            'scaler': MinMaxScaler()
        },
        'economic': {
            'cols': ['Economic_Sensitivity_PIB', 'Economic_Sensitivity_ICI'],
            'weight': 2.0,  # Give more weight to economic sensitivity
            'scaler': StandardScaler()  # Use standard scaling for correlations
        },
        'market': {
            'cols': ['Market_Sensitivity_ECG', 'Market_Sensitivity_ISE'],
            'weight': 2.0,  # Give more weight to market sensitivity
            'scaler': StandardScaler()
        }
    }
    
    # Combine all column names
    numerical_cols = []
    for group in feature_groups.values():
        numerical_cols.extend(group['cols'])
    
    # Scale and weight features by group
    scaled_features = np.empty((len(features_df), 0))
    
    for group_name, group in feature_groups.items():
        # Scale the features
        X = features_df[group['cols']].values
        X_scaled = group['scaler'].fit_transform(X)
        
        # Apply weight
        X_weighted = X_scaled * group['weight']
        
        # Add to scaled features
        scaled_features = np.column_stack((scaled_features, X_weighted))
        
        # Print feature ranges after scaling
        print(f"\n{group_name} features after scaling:")
        for col, scaled_col in zip(group['cols'], X_weighted.T):
            print(f"{col}: min={scaled_col.min():.3f}, max={scaled_col.max():.3f}, mean={scaled_col.mean():.3f}")
    
    # Find optimal number of clusters
    n_clusters = find_optimal_clusters(scaled_features)
    print(f"\nUsing {n_clusters} clusters")
    
    # KMeans parameters for better stability
    kmeans_params = {
        'init': 'k-means++',    # Use k-means++ initialization
        'n_init': 100,          # Even more initializations
        'max_iter': 500,        # Keep high max iterations
        'tol': 1e-6,           # Keep tight tolerance
        'random_state': 42      # For reproducibility
    }
    
    # Perform K-means clustering with optimal k
    kmeans = KMeans(n_clusters=n_clusters, **kmeans_params)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add cluster assignments to the dataframe
    features_df['Cluster'] = clusters
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(scaled_features, clusters)
    print(f"\nFinal Silhouette Score: {silhouette_avg:.3f}")
    print(f"Number of iterations to converge: {kmeans.n_iter_}")
    
    # Print cluster sizes
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    print("\nCluster sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} materials")
    
    # Analyze feature importance for each cluster
    print("\nCluster characteristics:")
    cluster_means = []
    for i in range(n_clusters):
        cluster_data = features_df[features_df['Cluster'] == i]
        means = cluster_data[numerical_cols].mean()
        cluster_means.append(means)
        print(f"\nCluster {i} characteristics:")
        for col, mean in means.items():
            print(f"{col}: {mean:.3f}")
    
    # Analyze clusters
    analyze_clusters(features_df, numerical_cols)
    
    return features_df

def analyze_clusters(features_df, numerical_cols):
    """Analyze and visualize cluster characteristics."""
    # Create a clear mapping file
    cluster_mapping = features_df[['Material', 'Material Description', 'Cluster', 'Avg_Monthly_Demand']].copy()
    
    # Sort by cluster and demand for better readability
    cluster_mapping = cluster_mapping.sort_values(['Cluster', 'Avg_Monthly_Demand'], ascending=[True, False])
    
    # 1. Cluster sizes
    plt.figure(figsize=(10, 6))
    sns.countplot(data=features_df, x='Cluster')
    plt.title('Number of Materials per Cluster')
    plt.show()
    
    # 2. Feature distributions by cluster
    for col in numerical_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=features_df, x='Cluster', y=col)
        plt.title(f'{col} Distribution by Cluster')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 3. Cluster characteristics summary
    cluster_summary = features_df.groupby('Cluster')[numerical_cols].mean()
    print("\nCluster Summary:")
    print(cluster_summary)
    
    return cluster_summary

def process_material(material_row: pd.Series) -> Dict[str, Any]:
    """Process a single material through its appropriate cluster model"""
    material_id = material_row['Material']
    material_name = material_row['Material Description']
    cluster = material_row['Cluster']
    
    try:
        # Get the appropriate forecast function
        forecast_func = CLUSTER_FUNCTIONS.get(cluster)
        if forecast_func is None:
            logging.warning(f"No forecast function for cluster {cluster}, using generic model")
            forecast_func = cluster3.main
        
        # Create output directory if it doesn't exist
        script_dir = Path(__file__).parent.absolute()
        output_dir = script_dir / 'forecast_results'
        output_dir.mkdir(exist_ok=True)
        
        # Create material-specific directory
        material_dir = output_dir / material_id
        material_dir.mkdir(exist_ok=True)
        
        # Run forecast with material directory as output path
        results = forecast_func(material_id, material_name, material_dir)
        if results is None:
            logging.error(f"Failed to process material {material_id}")
            return None
        
        # Save forecasts
        forecasts = results['forecasts']
        
        # Save daily forecasts
        daily_file = material_dir / 'daily_forecast.csv'
        forecasts['daily'].to_csv(daily_file, index=True)
        
        # Save weekly forecasts
        weekly_file = material_dir / 'weekly_forecast.csv'
        forecasts['weekly'].to_csv(weekly_file, index=True)
        
        # Save monthly forecasts
        monthly_file = material_dir / 'monthly_forecast.csv'
        forecasts['monthly'].to_csv(monthly_file, index=True)
        
        # Create summary file
        summary_file = material_dir / 'forecast_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Forecast Summary for Material {material_id}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Material Description: {material_name}\n")
            f.write(f"Cluster: {cluster}\n\n")
            
            metrics = results['metrics']
            f.write("Daily Metrics:\n")
            f.write(f"R² (correlation squared): {metrics['Daily R2']:.4f}\n")
            f.write(f"RMSE: {metrics['Daily RMSE']:.4f}\n")
            f.write(f"MAE: {metrics['Daily MAE']:.4f}\n")
            f.write(f"MAPE: {metrics['Daily MAPE']:.4f}%\n\n")
            
            f.write("Weekly Metrics:\n")
            f.write(f"R² (correlation squared): {metrics['Weekly R2']:.4f}\n")
            f.write(f"RMSE: {metrics['Weekly RMSE']:.4f}\n")
            f.write(f"MAE: {metrics['Weekly MAE']:.4f}\n")
            f.write(f"MAPE: {metrics['Weekly MAPE']:.4f}%\n\n")
            
            f.write("Monthly Metrics:\n")
            f.write(f"R² (correlation squared): {metrics['Monthly R2']:.4f}\n")
            f.write(f"RMSE: {metrics['Monthly RMSE']:.4f}\n")
            f.write(f"MAE: {metrics['Monthly MAE']:.4f}\n")
            f.write(f"MAPE: {metrics['Monthly MAPE']:.4f}%\n")
        
        logging.info(f"Successfully processed material {material_id} (Cluster {cluster})")
        return results['metrics']
        
    except Exception as e:
        logging.error(f"Error processing material {material_id}: {str(e)}")
        return None

def main():
    # Load raw data
    print("\nLoading data...")
    raw_data = pd.read_excel('https://www.dropbox.com/scl/fi/pw5717sy9bsfxru4vggf0/Customer-Order-Quantity_Dispatched-Quantity.xlsx?rlkey=bexjc34bevu4yz3y3t2efciv1&st=e78bctop&dl=1')
    external_vars = pd.read_excel('https://www.dropbox.com/scl/fi/z4byjz6aamkutje77pc1b/External_Variables.xlsx?rlkey=qcqat0lddovdcwjvqhv6cz43b&st=1wbhpcy5&dl=1', sheet_name=0)
    
    # Convert dates with proper formats
    raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%d.%m.%Y')
    
    # Convert month-year format (e.g., 'jan-17' to '2017-01-01')
    external_vars['DATE'] = pd.to_datetime('20' + external_vars['DATE'].str[-2:] + '-' + external_vars['DATE'].str[:3] + '-01')
    external_vars = external_vars.rename(columns={'DATE': 'Date'})
    
    # Filter data to 2022 onwards
    raw_data = raw_data[raw_data['Date'].dt.year >= 2022]
    external_vars = external_vars[external_vars['Date'].dt.year >= 2022]
    
    # Add YearMonth column for aggregation
    raw_data['YearMonth'] = raw_data['Date'].dt.to_period('M').astype(str)
    external_vars['YearMonth'] = external_vars['Date'].dt.to_period('M').astype(str)
    
    # Aggregate raw data to monthly
    monthly_raw = raw_data.groupby(['Product ID', 'Product Name', 'YearMonth']).agg({
        'Dispatched Quantity': 'sum',
        'Customer Order Quantity': 'sum'
    }).reset_index()
    
    # Aggregate external vars to monthly (using mean for economic indicators)
    monthly_external = external_vars.groupby('YearMonth').agg({
        'PIB_CO': 'mean',
        'ICI': 'mean',
        'ECG_DESP': 'mean',
        'ISE_CO': 'mean'
    }).reset_index()
    
    # Merge monthly data
    data = pd.merge(monthly_raw, monthly_external, on='YearMonth', how='left')
    
    # Calculate features for clustering
    print("\nCalculating clustering features...")
    materials = []
    for material_id in raw_data['Product ID'].unique():
        material_data = data[data['Product ID'] == material_id]
        
        try:
            # Calculate correlations safely
            dispatched_qty = material_data['Dispatched Quantity']
            correlations = {
                'Economic_Sensitivity_PIB': dispatched_qty.corr(material_data['PIB_CO']),
                'Economic_Sensitivity_ICI': dispatched_qty.corr(material_data['ICI']),
                'Market_Sensitivity_ECG': dispatched_qty.corr(material_data['ECG_DESP']),
                'Market_Sensitivity_ISE': dispatched_qty.corr(material_data['ISE_CO'])
            }
            
            # Replace NaN correlations with 0
            correlations = {k: 0 if pd.isna(v) else v for k, v in correlations.items()}
            
            # Calculate demand metrics
            avg_demand = material_data['Dispatched Quantity'].mean()
            demand_std = material_data['Dispatched Quantity'].std()
            
            material_features = {
                'Material': material_id,
                'Material Description': material_data['Product Name'].iloc[0],
                'Avg_Monthly_Demand': avg_demand,
                'Demand_Std': demand_std,
                'Demand_CV': demand_std / (avg_demand + 1e-6),  # Add small constant to avoid division by zero
                'Order_Fulfillment': (material_data['Dispatched Quantity'].sum() / 
                                    (material_data['Customer Order Quantity'].sum() + 1e-6)),
                'Economic_Sensitivity_PIB': correlations['Economic_Sensitivity_PIB'],
                'Economic_Sensitivity_ICI': correlations['Economic_Sensitivity_ICI'],
                'Market_Sensitivity_ECG': correlations['Market_Sensitivity_ECG'],
                'Market_Sensitivity_ISE': correlations['Market_Sensitivity_ISE']
            }
            materials.append(material_features)
            
        except Exception as e:
            print(f"Error processing material {material_id}: {str(e)}")
            continue
    
    # Create features DataFrame
    features_df = pd.DataFrame(materials)
    features_df = features_df.dropna()
    
    # Sort by Material ID to ensure consistent order
    features_df = features_df.sort_values('Material')
    print(f"Processed {len(features_df)} materials successfully")
    
    # Perform clustering
    print("\nPerforming clustering...")
    features_df = perform_clustering(features_df)
    
    # Save clustering results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cluster_output_path = os.path.join(script_dir, 'material_clusters.csv')
    features_df.to_csv(cluster_output_path, index=False)
    print(f"\nClustering results saved to: {cluster_output_path}")
    
    # Process materials through forecast models
    results = []
    output_path = os.path.join(script_dir, 'material_metrics_clustered.csv')
    interim_path = os.path.join(script_dir, 'material_metrics_clustered_interim.csv')

    try:
        print("\nProcessing materials through forecast models...")
        for idx, row in tqdm(features_df.iterrows(), total=len(features_df)):
            metrics = process_material(row)
            if metrics:
                result = {
                    'Material': row['Material'],
                    'Material Description': row['Material Description'],
                    'Cluster': row['Cluster'],
                    **metrics  # Unpack metrics into result dict
                }
                results.append(result)
            
            # Save interim results
            if results:  # Only save if we have results
                interim_df = pd.DataFrame(results)
                interim_df.to_csv(interim_path, index=False)
            
        # Save final results
        if results:  # Only save if we have results
            final_df = pd.DataFrame(results)
            final_df.to_csv(output_path, index=False)
            interim_df.to_csv(output_path, index=False)  # Also save interim results to final location
            
            # Print summary statistics
            print("\nSummary Statistics:")
            for timeframe in ['Daily', 'Weekly', 'Monthly']:
                for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
                    col = f"{timeframe} {metric}"
                    valid_values = final_df[col].dropna()
                    if len(valid_values) > 0:
                        print(f"\n{col}:")
                        print(f"Mean: {valid_values.mean():.4f}")
                        print(f"Median: {valid_values.median():.4f}")
                        print(f"Std Dev: {valid_values.std():.4f}")
                    
            print(f"\nResults saved to: {output_path}")
        else:
            print("\nNo results were generated!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving partial results...")
        if results:
            pd.DataFrame(results).to_csv(interim_path, index=False)

if __name__ == "__main__":
    main()
