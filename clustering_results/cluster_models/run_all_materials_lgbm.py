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
        logging.FileHandler('material_processing_lgbm.log'),
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
forecast_script = r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\clustering_results\cluster_models\demand_forecast_LGBM_new4.py"

def extract_r2_values(output):
    """Extract R2 values from script output"""
    r2_values = {}
    
    # Regular expressions to find R2 values
    patterns = {
        'Daily': r"Daily Metrics:.*?R2 \(correlation squared\): ([\d.]+)",
        'Weekly': r"Weekly Metrics:.*?R2 \(correlation squared\): ([\d.]+)",
        'Monthly': r"Monthly Metrics:.*?R2 \(correlation squared\): ([\d.]+)"
    }
    
    for timeframe, pattern in patterns.items():
        match = re.search(pattern, output, re.DOTALL)
        if match:
            r2_values[timeframe] = float(match.group(1))
        else:
            r2_values[timeframe] = None
    
    return r2_values

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
    temp_script_path = 'temp_forecast_script_lgbm.py'
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
        
        # Extract R2 values from output
        r2_values = extract_r2_values(output)
        
        # Add to results
        results.append({
            'Material ID': material_id,
            'Material Name': material_name,
            'Daily R2': r2_values.get('Daily'),
            'Weekly R2': r2_values.get('Weekly'),
            'Monthly R2': r2_values.get('Monthly'),
            'Success': True
        })
        
        logging.info(f"Successfully processed {material_name}")
        
    except Exception as e:
        logging.error(f"Error processing {material_name}: {str(e)}")
        results.append({
            'Material ID': material_id,
            'Material Name': material_name,
            'Daily R2': None,
            'Weekly R2': None,
            'Monthly R2': None,
            'Success': False,
            'Error': str(e)
        })
    
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
output_path = os.path.join(os.path.dirname(forecast_script), 'material_r2_results_lgbm.csv')
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")

# Print summary statistics
print("\nSummary of R2 values:")
for col in ['Daily R2', 'Weekly R2', 'Monthly R2']:
    valid_values = results_df[col].dropna()
    if len(valid_values) > 0:
        print(f"\n{col}:")
        print(f"Mean: {valid_values.mean():.3f}")
        print(f"Median: {valid_values.median():.3f}")
        print(f"Min: {valid_values.min():.3f}")
        print(f"Max: {valid_values.max():.3f}")
        print(f"Number of valid results: {len(valid_values)}")

print(f"\nSuccessfully processed: {results_df['Success'].sum()} out of {len(results_df)} materials")
