import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the results
results_df = pd.read_csv('material_r2_results.csv')

# Print basic statistics
print("Basic Statistics:")
print("-" * 50)
print(f"Total materials processed: {len(results_df)}")
print(f"Successfully processed: {results_df['Success'].sum()}")
print(f"Failed: {len(results_df) - results_df['Success'].sum()}")
print("\n")

# Function to analyze R2 values
def analyze_r2(r2_values, timeframe):
    valid_r2 = r2_values.dropna()
    print(f"{timeframe} R2 Statistics:")
    print("-" * 50)
    print(f"Number of valid results: {len(valid_r2)}")
    print(f"Mean R2: {valid_r2.mean():.3f}")
    print(f"Median R2: {valid_r2.median():.3f}")
    print(f"Min R2: {valid_r2.min():.3f}")
    print(f"Max R2: {valid_r2.max():.3f}")
    print(f"Standard deviation: {valid_r2.std():.3f}")
    
    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"{p}th percentile: {valid_r2.quantile(p/100):.3f}")
    print("\n")
    
    return valid_r2

# Analyze each timeframe
daily_r2 = analyze_r2(results_df['Daily R2'], 'Daily')
weekly_r2 = analyze_r2(results_df['Weekly R2'], 'Weekly')
monthly_r2 = analyze_r2(results_df['Monthly R2'], 'Monthly')

# Create visualization
plt.figure(figsize=(15, 10))

# Box plots
plt.subplot(2, 2, 1)
data = [
    results_df['Daily R2'].dropna(),
    results_df['Weekly R2'].dropna(),
    results_df['Monthly R2'].dropna()
]
plt.boxplot(data, labels=['Daily', 'Weekly', 'Monthly'])
plt.title('R² Distribution by Timeframe')
plt.ylabel('R² Value')

# Histogram of R2 values
plt.subplot(2, 2, 2)
plt.hist(daily_r2, alpha=0.5, label='Daily', bins=20)
plt.hist(weekly_r2, alpha=0.5, label='Weekly', bins=20)
plt.hist(monthly_r2, alpha=0.5, label='Monthly', bins=20)
plt.title('Distribution of R² Values')
plt.xlabel('R² Value')
plt.ylabel('Count')
plt.legend()

# Scatter plot of R2 relationships
plt.subplot(2, 2, 3)
plt.scatter(results_df['Daily R2'], results_df['Weekly R2'], alpha=0.5)
plt.xlabel('Daily R²')
plt.ylabel('Weekly R²')
plt.title('Daily vs Weekly R²')

plt.subplot(2, 2, 4)
plt.scatter(results_df['Weekly R2'], results_df['Monthly R2'], alpha=0.5)
plt.xlabel('Weekly R²')
plt.ylabel('Monthly R²')
plt.title('Weekly vs Monthly R²')

plt.tight_layout()
plt.savefig('r2_analysis.png')
plt.close()

# Identify best and worst performing materials
def get_top_bottom_materials(df, r2_col, n=5):
    valid_df = df[df[r2_col].notna()].copy()
    top_n = valid_df.nlargest(n, r2_col)
    bottom_n = valid_df.nsmallest(n, r2_col)
    return top_n, bottom_n

print("Top and Bottom Performing Materials")
print("-" * 50)

for timeframe in ['Daily R2', 'Weekly R2', 'Monthly R2']:
    print(f"\n{timeframe.split()[0]} Timeframe:")
    print("-" * 30)
    
    top, bottom = get_top_bottom_materials(results_df, timeframe)
    
    print("\nTop 5 Materials:")
    for _, row in top.iterrows():
        print(f"Material: {row['Material Name']} (ID: {row['Material ID']}) - R²: {row[timeframe]:.3f}")
    
    print("\nBottom 5 Materials:")
    for _, row in bottom.iterrows():
        print(f"Material: {row['Material Name']} (ID: {row['Material ID']}) - R²: {row[timeframe]:.3f}")

# Calculate correlation between timeframes
r2_corr = results_df[['Daily R2', 'Weekly R2', 'Monthly R2']].corr()
print("\nCorrelation between timeframes:")
print("-" * 50)
print(r2_corr)

# Save detailed results to Excel
with pd.ExcelWriter('r2_analysis_detailed.xlsx') as writer:
    # Overall results
    results_df.to_excel(writer, sheet_name='All Results', index=False)
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Median', 'Min', 'Max', 'Std'],
        'Daily': [len(daily_r2), daily_r2.mean(), daily_r2.median(), daily_r2.min(), daily_r2.max(), daily_r2.std()],
        'Weekly': [len(weekly_r2), weekly_r2.mean(), weekly_r2.median(), weekly_r2.min(), weekly_r2.max(), weekly_r2.std()],
        'Monthly': [len(monthly_r2), monthly_r2.mean(), monthly_r2.median(), monthly_r2.min(), monthly_r2.max(), monthly_r2.std()]
    })
    summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
    
    # Correlation matrix
    r2_corr.to_excel(writer, sheet_name='Correlations')

print("\nDetailed analysis has been saved to 'r2_analysis_detailed.xlsx'")
print("Visualizations have been saved to 'r2_analysis.png'")
