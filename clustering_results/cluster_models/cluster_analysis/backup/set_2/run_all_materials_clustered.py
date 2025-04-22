import pandas as pd
import os
from tqdm import tqdm
import logging
import subprocess
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('material_processing_clustered.log'),
        logging.StreamHandler()
    ]
)

# Load the data to get list of materials
daily_df = pd.read_excel(
    r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx"
)

# Load clustering results
cluster_df = pd.read_csv(
    r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\clustering_results\cluster_models\cluster_analysis\material_clusters.csv"
)

# Get unique materials with their cluster assignments
materials_list = daily_df[['Product ID', 'Product Name']].drop_duplicates().reset_index(drop=True)

# Clean up material IDs by removing leading zeros
materials_list['Clean_ID'] = materials_list['Product ID'].str.lstrip('0')
cluster_df['Clean_ID'] = cluster_df['Material_ID'].astype(str).str.lstrip('0')

# Merge using the cleaned IDs
materials_list = materials_list.merge(cluster_df[['Clean_ID', 'Cluster']], 
                                    left_on='Clean_ID',
                                    right_on='Clean_ID',
                                    how='left')
materials_list.drop('Clean_ID', axis=1, inplace=True)

print(f"Found {len(materials_list)} unique materials to process")
print(f"Materials with cluster assignments: {materials_list['Cluster'].notna().sum()}")

# Create results DataFrame
results = []

# Paths to the cluster-specific forecast scripts
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
forecast_scripts = {
    0: os.path.join(script_dir, "demand_forecast_cluster0.py"),
    1: os.path.join(script_dir, "demand_forecast_cluster1.py"),
    2: os.path.join(script_dir, "demand_forecast_cluster2.py"),
    3: os.path.join(script_dir, "demand_forecast_cluster3.py")
}

def extract_metrics(output):
    """Extract all metrics from script output"""
    metrics = {}
    
    # Regular expressions to find metric values
    patterns = {
        'Daily': {
            'R2': r"Daily Metrics:.*?R2 \(correlation squared\): ([\d.]+)",
            'RMSE': r"Daily Metrics:.*?RMSE: ([\d.]+)",
            'MAE': r"Daily Metrics:.*?MAE: ([\d.]+)",
            'MAPE': r"Daily Metrics:.*?MAPE: ([\d.]+)%"
        },
        'Weekly': {
            'R2': r"Weekly Metrics:.*?R2 \(correlation squared\): ([\d.]+)",
            'RMSE': r"Weekly Metrics:.*?RMSE: ([\d.]+)",
            'MAE': r"Weekly Metrics:.*?MAE: ([\d.]+)",
            'MAPE': r"Weekly Metrics:.*?MAPE: ([\d.]+)%"
        },
        'Monthly': {
            'R2': r"Monthly Metrics:.*?R2 \(correlation squared\): ([\d.]+)",
            'RMSE': r"Monthly Metrics:.*?RMSE: ([\d.]+)",
            'MAE': r"Monthly Metrics:.*?MAE: ([\d.]+)",
            'MAPE': r"Monthly Metrics:.*?MAPE: ([\d.]+)%"
        }
    }
    
    for timeframe, metric_patterns in patterns.items():
        metrics[timeframe] = {}
        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, output, re.DOTALL)
            if match:
                try:
                    metrics[timeframe][metric_name] = float(match.group(1))
                except ValueError:
                    metrics[timeframe][metric_name] = None
            else:
                metrics[timeframe][metric_name] = None
    
    return metrics

def create_temp_script(material_id, material_name, original_script_path):
    """Create temporary script for a specific material"""
    # Read the original script
    with open(original_script_path, 'r') as f:
        original_content = f.read()
    
    # Find the if __name__ == "__main__": block and replace the material_id and material_no
    main_block_pattern = r"if __name__ == \"__main__\":\s*#[^\n]*\s*material_id = '[^']*'[^\n]*\s*material_no = '[^']*'"
    replacement = f'''if __name__ == "__main__":
    # Run for specific material
    material_id = '{material_id}'
    material_no = '{material_name}'
    model, metrics = main(material_id, material_no)'''
    
    modified_content = re.sub(main_block_pattern, replacement, original_content)
    
    # Write to temporary file
    temp_script_path = os.path.join(script_dir, 'temp_forecast_script.py')
    with open(temp_script_path, 'w') as f:
        f.write(modified_content)
    
    return temp_script_path

# Process each material
for idx, row in tqdm(materials_list.iterrows(), total=len(materials_list)):
    material_id = row['Product ID']
    material_name = row['Product Name']
    cluster = row['Cluster']
    temp_script_path = None  # Initialize temp_script_path
    
    # Skip if cluster is not assigned
    if pd.isna(cluster):
        logging.warning(f"Skipping {material_name} (ID: {material_id}) - no cluster assigned")
        continue
        
    cluster = int(cluster)  # Convert to integer for dictionary lookup
    
    logging.info(f"Processing {material_name} (ID: {material_id}) from Cluster {cluster}")
    
    try:
        # Get the appropriate script for this cluster
        if cluster not in forecast_scripts:
            raise Exception(f"No forecast script found for cluster {cluster}")
            
        script_path = forecast_scripts[cluster]
        
        # Create temporary script for this material
        temp_script_path = create_temp_script(material_id, material_name, script_path)
        
        # Run the temporary script and capture output
        process = subprocess.Popen(['python', temp_script_path], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True,
                                 cwd=script_dir)  # Set working directory to script location
        output, error = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Script failed with error: {error}")
        
        # Extract metrics from output
        metrics = extract_metrics(output)
        
        # Add to results
        results.append({
            'Material ID': material_id,
            'Material Name': material_name,
            'Cluster': cluster,
            'Daily R2': metrics['Daily']['R2'],
            'Daily RMSE': metrics['Daily']['RMSE'],
            'Daily MAE': metrics['Daily']['MAE'],
            'Daily MAPE': metrics['Daily']['MAPE'],
            'Weekly R2': metrics['Weekly']['R2'],
            'Weekly RMSE': metrics['Weekly']['RMSE'],
            'Weekly MAE': metrics['Weekly']['MAE'],
            'Weekly MAPE': metrics['Weekly']['MAPE'],
            'Monthly R2': metrics['Monthly']['R2'],
            'Monthly RMSE': metrics['Monthly']['RMSE'],
            'Monthly MAE': metrics['Monthly']['MAE'],
            'Monthly MAPE': metrics['Monthly']['MAPE'],
            'Success': True,
            'Error': None
        })
        
        logging.info(f"Successfully processed {material_name}")
        
    except Exception as e:
        error_msg = f"Script failed with error: {str(e)}"
        logging.error(f"Error processing {material_name}: {error_msg}")
        results.append({
            'Material ID': material_id,
            'Material Name': material_name,
            'Cluster': cluster,
            'Daily R2': None,
            'Daily RMSE': None,
            'Daily MAE': None,
            'Daily MAPE': None,
            'Weekly R2': None,
            'Weekly RMSE': None,
            'Weekly MAE': None,
            'Weekly MAPE': None,
            'Monthly R2': None,
            'Monthly RMSE': None,
            'Monthly MAE': None,
            'Monthly MAPE': None,
            'Success': False,
            'Error': error_msg
        })
    
    finally:
        # Clean up temp script
        if temp_script_path and os.path.exists(temp_script_path):
            os.remove(temp_script_path)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
output_path = os.path.join(script_dir, 'material_metrics_clustered.csv')
results_df.to_csv(output_path, index=False)

# Print summary statistics for each metric
print("\nSummary Statistics:")
for timeframe in ['Daily', 'Weekly', 'Monthly']:
    for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
        col = f"{timeframe} {metric}"
        valid_values = results_df[col].dropna()
        if len(valid_values) > 0:
            print(f"\n{col}:")
            print(f"Mean: {valid_values.mean():.4f}")
            print(f"Median: {valid_values.median():.4f}")
            print(f"Std Dev: {valid_values.std():.4f}")

print(f"\nResults saved to: {output_path}")
