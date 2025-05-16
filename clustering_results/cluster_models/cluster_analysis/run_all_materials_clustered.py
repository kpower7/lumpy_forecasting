import pandas as pd
import os
import subprocess
import logging

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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cluster_assignments.csv')
)

# Create materials list with cluster assignments
materials_list = pd.DataFrame({
    'Product ID': cluster_df['Product ID'],
    'Product Name': cluster_df['Product Name'],
    'Cluster': cluster_df['Cluster']
}).dropna(subset=['Cluster'])  # Only keep materials with cluster assignments

# Paths to the cluster-specific forecast scripts
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
forecast_scripts = {
    0: os.path.join(script_dir, "demand_forecast_cluster0.py"),
    1: os.path.join(script_dir, "demand_forecast_cluster1.py"),
    2: os.path.join(script_dir, "demand_forecast_cluster2.py"),
    3: os.path.join(script_dir, "demand_forecast_cluster3.py")
}

def create_temp_script(material_id, material_no, script_path):
    """Create a temporary script for a specific material"""
    # Read the original script
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Create temporary script path
    temp_script_path = os.path.join(script_dir, 'temp_forecast_script.py')
    
    # Replace the main execution part
    script_content = script_content.split('if __name__ == "__main__":')[0]
    script_content += '''
if __name__ == "__main__":
    material_id = '{}' 
    material_no = '{}'  
    model, metrics = main(material_id, material_no)
    
    print("\\nDaily Metrics:")
    print(f"R2: {metrics['Daily']['R2']}")
    print(f"RMSE: {metrics['Daily']['RMSE']}")
    print(f"MAE: {metrics['Daily']['MAE']}")
    print(f"MAPE: {metrics['Daily']['MAPE']}%")
    
    print("\\nWeekly Metrics:")
    print(f"R2: {metrics['Weekly']['R2']}")
    print(f"RMSE: {metrics['Weekly']['RMSE']}")
    print(f"MAE: {metrics['Weekly']['MAE']}")
    print(f"MAPE: {metrics['Weekly']['MAPE']}%")
    
    print("\\nMonthly Metrics:")
    print(f"R2: {metrics['Monthly']['R2']}")
    print(f"RMSE: {metrics['Monthly']['RMSE']}")
    print(f"MAE: {metrics['Monthly']['MAE']}")
    print(f"MAPE: {metrics['Monthly']['MAPE']}%")
'''.format(material_id, material_no)
    
    # Write the temporary script
    with open(temp_script_path, 'w') as f:
        f.write(script_content)
    
    return temp_script_path

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
            
        logging.info(f"Successfully processed {material_no}")
        
    except Exception as e:
        error_msg = f"Script failed with error: {str(e)}"
        logging.error(f"Error processing {material_no}: {error_msg}")
        
    finally:
        # Clean up temporary script
        if temp_script_path and os.path.exists(temp_script_path):
            os.remove(temp_script_path)

# Process each material
for idx, row in materials_list.iterrows():
    process_material(row)
