
import sys
import os

# Add the parent directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime

# Import the main function from cluster script
sys.path.append(os.path.dirname('c:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\clustering_results\cluster_models\cluster_analysis\demand_forecast_cluster1.py'))
from demand_forecast_cluster1 import main

# Run the model
material_id = '5551O3273'
material_no = 'Material_79'
model, metrics = main(material_id, material_no)
