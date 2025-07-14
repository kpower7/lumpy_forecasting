# Re-load the necessary libraries and dataset after execution state reset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('naive_forecast.log'),
        logging.StreamHandler()
    ]
)

def load_data(material_id):
    """Load and prepare data for a specific material"""
    try:
        # Load daily data
        daily_df = pd.read_excel('https://www.dropbox.com/scl/fi/pw5717sy9bsfxru4vggf0/Customer-Order-Quantity_Dispatched-Quantity.xlsx?rlkey=bexjc34bevu4yz3y3t2efciv1&st=e78bctop&dl=1')
        
        # Filter for specific material
        daily_df = daily_df[daily_df['Product ID'] == material_id].copy()
        
        # Convert date column
        daily_df['Date'] = pd.to_datetime(daily_df['Date'], format='%d.%m.%Y')
        daily_df.set_index('Date', inplace=True)
        daily_df.sort_index(inplace=True)
        
        return daily_df
            
    except Exception as e:
        logging.error(f"Error loading data for material {material_id}: {str(e)}")
        return None

def calculate_metrics(actual, predicted):
    """Calculate forecast accuracy metrics"""
    # Ensure we have matching indices
    df = pd.DataFrame({'actual': actual, 'predicted': predicted})
    df = df.dropna()  # Remove any rows with NaN values
    
    # Remove pairs where actual is zero
    mask = df['actual'] != 0
    df_filtered = df[mask]
    
    if len(df_filtered) <= 1:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan
        }
    
    actual_filtered = df_filtered['actual']
    predicted_filtered = df_filtered['predicted']
    
    mape = np.mean(np.abs((actual_filtered - predicted_filtered) / actual_filtered)) * 100
    rmse = np.sqrt(mean_squared_error(actual_filtered, predicted_filtered))
    mae = mean_absolute_error(actual_filtered, predicted_filtered)
    r2 = r2_score(actual_filtered, predicted_filtered)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }

def plot_monthly_forecast(monthly_df, material_id, material_no, output_path):
    """Plot monthly forecast vs actual"""
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_df.index, monthly_df['Actual'], label='Actual', marker='o')
    plt.plot(monthly_df.index, monthly_df['Predicted'], label='Predicted', marker='s')
    
    plt.title(f'Monthly Forecast vs Actual - Material {material_no} ({material_id})')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Save plot
    if output_path:
        plt.savefig(os.path.join(output_path, f'monthly_forecast_{material_id}.png'), 
                    bbox_inches='tight')
    plt.close()

def run_naive_forecast(material_id, material_name, output_path=None):
    """Run naive forecast for a specific material"""
    try:
        # Load data
        daily_data = load_data(material_id)
        if daily_data is None:
            return None
        
        # Split into train and test
        train_cutoff = pd.to_datetime("2024-06-01")
        test_data = daily_data[daily_data.index >= train_cutoff].copy()
        
        if len(test_data) == 0:
            logging.warning(f"No test data available for material {material_id}")
            return None
        
        demand_col = "Dispatched Quantity"
        
        # Naive forecast: Use previous day's value
        test_data["Naive_Predicted"] = test_data[demand_col].shift(1)
        
        # Create DataFrames for actual and predicted values
        daily_df = pd.DataFrame({
            'actual': test_data[demand_col],
            'predicted': test_data["Naive_Predicted"]
        })
        
        # Aggregate to weekly and monthly levels
        weekly_df = daily_df.resample("W").sum()
        monthly_df = daily_df.resample("ME").sum()
        
        # Calculate metrics
        daily_metrics = calculate_metrics(daily_df['actual'], daily_df['predicted'])
        weekly_metrics = calculate_metrics(weekly_df['actual'], weekly_df['predicted'])
        monthly_metrics = calculate_metrics(monthly_df['actual'], monthly_df['predicted'])
        
        # Create monthly DataFrame for plotting
        monthly_plot_df = pd.DataFrame({
            'Actual': monthly_df['actual'],
            'Predicted': monthly_df['predicted']
        })
        
        # Plot results if output path is provided
        if output_path:
            plot_monthly_forecast(monthly_plot_df, material_id, material_name, output_path)
        
        # Combine all metrics into a single dictionary
        all_metrics = {
            'Material': material_id,
            'Material Description': material_name,
            'Daily RMSE': daily_metrics['RMSE'],
            'Daily MAE': daily_metrics['MAE'],
            'Daily MAPE': daily_metrics['MAPE'],
            'Daily R2': daily_metrics['R2'],
            'Weekly RMSE': weekly_metrics['RMSE'],
            'Weekly MAE': weekly_metrics['MAE'],
            'Weekly MAPE': weekly_metrics['MAPE'],
            'Weekly R2': weekly_metrics['R2'],
            'Monthly RMSE': monthly_metrics['RMSE'],
            'Monthly MAE': monthly_metrics['MAE'],
            'Monthly MAPE': monthly_metrics['MAPE'],
            'Monthly R2': monthly_metrics['R2']
        }
        
        return all_metrics
        
    except Exception as e:
        logging.error(f"Error processing material {material_id}: {str(e)}")
        return None

def main():
    """Process all materials with naive forecast"""
    print("\nLoading data...")
    # Load raw data to get list of materials
    raw_data = pd.read_excel('https://www.dropbox.com/scl/fi/pw5717sy9bsfxru4vggf0/Customer-Order-Quantity_Dispatched-Quantity.xlsx?rlkey=bexjc34bevu4yz3y3t2efciv1&st=e78bctop&dl=1')
    
    # Get unique materials
    materials = raw_data[['Product ID', 'Product Name']].drop_duplicates()
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'naive_forecast_results')
    os.makedirs(output_path, exist_ok=True)
    
    # Process each material
    results = []
    interim_path = os.path.join(output_path, 'metrics_interim.csv')
    final_path = os.path.join(output_path, 'metrics_all_materials.csv')
    
    try:
        print("\nProcessing materials...")
        for idx, row in tqdm(materials.iterrows(), total=len(materials)):
            metrics = run_naive_forecast(row['Product ID'], row['Product Name'], output_path)
            if metrics:
                results.append(metrics)
                
                # Save interim results
                if results:
                    pd.DataFrame(results).to_csv(interim_path, index=False)
        
        # Save final results
        if results:
            final_df = pd.DataFrame(results)
            final_df.to_csv(final_path, index=False)
            
            # Print summary statistics
            print("\nSummary Statistics:")
            for timeframe in ['Daily', 'Weekly', 'Monthly']:
                for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
                    col = f'{timeframe} {metric}'
                    valid_values = final_df[col].dropna()
                    if len(valid_values) > 0:
                        print(f"\n{col}:")
                        print(f"Mean: {valid_values.mean():.4f}")
                        print(f"Median: {valid_values.median():.4f}")
                        print(f"Std Dev: {valid_values.std():.4f}")
            
            print(f"\nResults saved to: {final_path}")
        else:
            print("\nNo results were generated!")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving partial results...")
        if results:
            pd.DataFrame(results).to_csv(interim_path, index=False)

if __name__ == "__main__":
    main()
