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

# Load clustering results to get material list
cluster_df = pd.read_csv(
    r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\clustering_results\cluster_models\cluster_analysis\material_clusters.csv"
)

# Create materials list with cluster assignments
materials_list = pd.DataFrame({
    'Product ID': cluster_df['Material_ID'].astype(str).str.zfill(9),  # Pad with zeros to match format
    'Product Name': cluster_df['Material_Name'],  # Use Material_Name from CSV
    'Cluster': cluster_df['Cluster']
}).dropna(subset=['Cluster'])  # Only keep materials with cluster assignments

print(f"Found {len(materials_list)} unique materials to process")
print(f"Materials with cluster assignments: {len(materials_list)}")

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

def parse_metrics(stdout):
    """Parse metrics from the script output"""
    metrics = {'Daily': {}, 'Weekly': {}, 'Monthly': {}}
    current_timeframe = None
    
    for line in stdout.split('\n'):
        if any(f"{t} Metrics:" in line for t in ['Daily', 'Weekly', 'Monthly']):
            current_timeframe = line.split()[0]  # Get 'Daily', 'Weekly', or 'Monthly'
        elif current_timeframe and any(m in line for m in ['RMSE:', 'MAE:', 'R2', 'MAPE:']):
            try:
                metric, value = line.strip().split(': ')
                metric = metric.strip()
                if 'R2' in metric:
                    metric = 'R2'
                elif 'MAPE' in metric:
                    value = value.rstrip('%')  # Remove % sign
                metrics[current_timeframe][metric] = float(value)
            except (ValueError, KeyError):
                continue  # Skip if line can't be parsed
    
    return metrics

def process_material(material_row):
    """Process a single material through its appropriate cluster model"""
    material_id = material_row['Product ID']
    material_no = material_row['Product Name']
    cluster = int(material_row['Cluster'])  # Convert to integer
    
    logging.info(f"Processing {material_no} (ID: {material_id}) from Cluster {cluster}")
    
    try:
        # Get the appropriate script for this cluster
        if cluster not in forecast_scripts:
            raise ValueError(f"No forecast script found for cluster {cluster}")
            
        script_path = forecast_scripts[cluster]
        
        # Create temporary script for this material
        temp_script_path = create_temp_script(material_id, material_no, script_path)
        
        # Run the temporary script and capture output
        process = subprocess.Popen(['python', temp_script_path], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(stderr)
            
        # Parse the metrics from stdout
        metrics = parse_metrics(stdout)
        
        # Add to results
        result = {
            'Material ID': material_id,
            'Material Name': material_no,
            'Cluster': cluster,
            'Success': True,
            'Error': None
        }
        
        # Add metrics for each timeframe
        for timeframe in ['Daily', 'Weekly', 'Monthly']:
            for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
                result[f'{timeframe} {metric}'] = metrics[timeframe].get(metric)
        
        logging.info(f"Successfully processed {material_no}")
        return result
        
    except Exception as e:
        error_msg = f"Script failed with error: {str(e)}"
        logging.error(f"Error processing {material_no}: {error_msg}")
        
        return {
            'Material ID': material_id,
            'Material Name': material_no,
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
        }
    finally:
        # Clean up temporary script
        if temp_script_path and os.path.exists(temp_script_path):
            os.remove(temp_script_path)

# Process each material
results = []
output_path = os.path.join(script_dir, 'material_metrics_clustered.csv')
interim_path = os.path.join(script_dir, 'material_metrics_clustered_interim.csv')

for idx, row in tqdm(materials_list.iterrows(), total=len(materials_list)):
    result = process_material(row)
    results.append(result)
    
    # Save intermediate results after each material
    pd.DataFrame(results).to_csv(interim_path, index=False)
    
# Save final results
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
logging.info(f"Results saved to {output_path}")

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
