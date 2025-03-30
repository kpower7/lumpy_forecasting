import pandas as pd
import numpy as np
import os
import subprocess
import logging
from datetime import datetime
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(
    filename='material_processing_prophet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Read materials list
materials_list = pd.read_excel(
    r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\Customer Order Quantity_Dispatched Quantity.xlsx"
)

# Get unique materials
materials_list = materials_list[['Product ID', 'Product Name']].drop_duplicates()
print(f"Found {len(materials_list)} unique materials to process")

# Initialize results list
results = []

# Path to the forecast script
forecast_script = r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\clustering_results\cluster_models\demand_forecast_prophet_new4.py"

def extract_r2_values(output):
    """Extract R2 values from script output using explicit markers"""
    r2_values = {}
    
    # Look for metrics between markers
    start_marker = "=== METRICS START ==="
    end_marker = "=== METRICS END ==="
    
    try:
        # Extract metrics section
        metrics_start = output.find(start_marker)
        metrics_end = output.find(end_marker)
        
        if metrics_start == -1 or metrics_end == -1:
            logging.warning("Could not find metrics markers in output")
            return {'Daily': None, 'Weekly': None, 'Monthly': None}
            
        metrics_text = output[metrics_start:metrics_end]
        
        # Extract R2 values using simple key-value format
        patterns = {
            'Daily': r"DAILY_R2=([\d.]+|N/A)",
            'Weekly': r"WEEKLY_R2=([\d.]+|N/A)",
            'Monthly': r"MONTHLY_R2=([\d.]+|N/A)"
        }
        
        for timeframe, pattern in patterns.items():
            match = re.search(pattern, metrics_text)
            if match:
                value = match.group(1)
                try:
                    r2_values[timeframe] = float(value) if value != 'N/A' else None
                except ValueError:
                    r2_values[timeframe] = None
                logging.info(f"Found {timeframe} R2: {value}")
            else:
                r2_values[timeframe] = None
                logging.warning(f"Could not find {timeframe} R2 in output")
                
    except Exception as e:
        logging.error(f"Error extracting metrics: {str(e)}")
        r2_values = {'Daily': None, 'Weekly': None, 'Monthly': None}
    
    return r2_values

def create_temp_script(material_id, material_name, template_script):
    """Create a temporary script for the current material"""
    # Read the template script
    with open(template_script, 'r') as f:
        script_content = f.read()
    
    # Replace material ID and name
    script_content = script_content.replace("'000161032'", f"'{material_id}'")
    script_content = script_content.replace("'Material_4'", f"'{material_name}'")
    
    # Create temporary script
    temp_script = 'temp_forecast_script_prophet.py'
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    return temp_script

# Process each material
for idx, row in tqdm(materials_list.iterrows(), total=len(materials_list)):
    material_id = row['Product ID']
    material_name = row['Product Name']
    temp_script_path = None
    
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
        
        # Log the full output for debugging
        logging.info(f"Script output:\n{output}")
        
        # Extract R2 values from output
        r2_values = extract_r2_values(output)
        
        # Add to results
        result_entry = {
            'Material ID': material_id,
            'Material Name': material_name,
            'Daily R2': r2_values.get('Daily'),
            'Weekly R2': r2_values.get('Weekly'),
            'Monthly R2': r2_values.get('Monthly'),
            'Success': True,
            'Error': None
        }
        
        # Log the extracted values
        logging.info(f"Extracted R2 values for {material_name}: {r2_values}")
        
        results.append(result_entry)
        logging.info(f"Successfully processed {material_name}")
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing {material_name}: {error_msg}")
        result_entry = {
            'Material ID': material_id,
            'Material Name': material_name,
            'Daily R2': None,
            'Weekly R2': None,
            'Monthly R2': None,
            'Success': False,
            'Error': error_msg
        }
        results.append(result_entry)
    
    finally:
        # Save intermediate results after each run (success or failure)
        intermediate_df = pd.DataFrame(results)
        intermediate_path = os.path.join(os.path.dirname(forecast_script), 'material_r2_results_prophet_intermediate.csv')
        intermediate_df.to_csv(intermediate_path, index=False)
        
        # Clean up temporary script
        try:
            if temp_script_path and os.path.exists(temp_script_path):
                os.remove(temp_script_path)
        except Exception as e:
            logging.error(f"Error cleaning up temporary script: {str(e)}")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
output_path = os.path.join(os.path.dirname(forecast_script), 'material_r2_results_prophet.csv')
results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
print("\nSummary of results:")
print(f"Total materials processed: {len(results_df)}")
print(f"Successful runs: {results_df['Success'].sum()}")
print(f"Failed runs: {len(results_df) - results_df['Success'].sum()}")

# Calculate average R2 values for successful runs
successful_runs = results_df[results_df['Success']]
if len(successful_runs) > 0:
    print("\nAverage R2 values for successful runs:")
    print(f"Daily R2: {successful_runs['Daily R2'].mean():.3f}")
    print(f"Weekly R2: {successful_runs['Weekly R2'].mean():.3f}")
    print(f"Monthly R2: {successful_runs['Monthly R2'].mean():.3f}")
