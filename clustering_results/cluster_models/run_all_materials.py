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
        logging.FileHandler('material_processing.log'),
        logging.StreamHandler()
    ]
)

# Load the data to get list of materials
daily_df = pd.read_excel(
    r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx"
)

# Get unique materials
materials_list = daily_df[['Product ID', 'Product Name']].drop_duplicates().reset_index(drop=True)
print(f"Found {len(materials_list)} unique materials to process")

# Create results DataFrame
results = []

# Path to the forecast script
forecast_script = r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\clustering_results\cluster_models\demand_forecast_xgboost_new4.1.py"

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
                metrics[timeframe][metric_name] = float(match.group(1))
            else:
                metrics[timeframe][metric_name] = None
    
    return metrics

def create_temp_script(material_id, material_name, original_script_path):
    """Create temporary script for a specific material"""
    # Read the original script
    with open(original_script_path, 'r') as f:
        original_content = f.read()
    
    # Replace the material ID and name in the original content
    modified_content = original_content.replace(
        "material_id = '000161032'  # Material_4",
        f"material_id = '{material_id}'  # {material_name}"
    ).replace(
        "material_no = 'Material_4'",
        f"material_no = '{material_name}'"
    )
    
    # Write to temporary file
    output_dir = os.path.dirname(original_script_path)
    temp_script_path = os.path.join(output_dir, 'temp_forecast_script.py')
    with open(temp_script_path, 'w') as f:
        f.write(modified_content)
    
    return temp_script_path

# Process each material
for idx, row in tqdm(materials_list.iterrows(), total=len(materials_list)):
    material_id = row['Product ID']
    material_name = row['Product Name']
    
    logging.info(f"Processing {material_name} (ID: {material_id})")
    
    try:
        # Create temporary script for this material
        temp_script_path = create_temp_script(material_id, material_name, forecast_script)
        
        # Run the temporary script and capture output
        process = subprocess.Popen(['python', temp_script_path], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        output, error = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Script failed with error: {error}")
        
        # Extract metrics from output
        metrics = extract_metrics(output)
        
        # Add to results
        results.append({
            'Material ID': material_id,
            'Material Name': material_name,
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
            'Success': True
        })
        
        logging.info(f"Successfully processed {material_name}")
        
    except Exception as e:
        logging.error(f"Error processing {material_name}: {str(e)}")
        results.append({
            'Material ID': material_id,
            'Material Name': material_name,
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
            'Error': str(e)
        })
    
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
output_path = os.path.join(os.path.dirname(forecast_script), 'material_r2_results_v2.csv')
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")

# Print summary statistics
print("\nSummary of metrics:")
for timeframe in ['Daily', 'Weekly', 'Monthly']:
    for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
        col = f"{timeframe} {metric}"
        valid_values = results_df[col].dropna()
        if len(valid_values) > 0:
            print(f"\n{col}:")
            print(f"Mean: {valid_values.mean():.3f}")
            print(f"Median: {valid_values.median():.3f}")
            print(f"Min: {valid_values.min():.3f}")
            print(f"Max: {valid_values.max():.3f}")
            print(f"Number of valid results: {len(valid_values)}")

print(f"\nSuccessfully processed: {results_df['Success'].sum()} out of {len(results_df)} materials")
