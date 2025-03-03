import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
base_dir = Path(r'C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\clustering_results\cluster_models')
results_df = pd.read_csv(base_dir / 'material_model_results.csv')

# Create output directory for plots
output_dir = base_dir / 'analysis_plots'
output_dir.mkdir(exist_ok=True)

# 1. Basic statistics by cluster
print("\n=== Basic Statistics by Cluster ===")
cluster_stats = results_df.groupby('Cluster').agg({
    'RMSE': ['mean', 'std', 'min', 'max', 'count'],
    'MAE': ['mean', 'std', 'min', 'max'],
    'R2': ['mean', 'std', 'min', 'max']
}).round(2)

print("\nCluster Performance Summary:")
print(cluster_stats)

# 2. Identify best and worst performing materials
print("\n=== Best Performing Materials (by R2) ===")
print(results_df.nlargest(5, 'R2')[['Material_ID', 'Cluster', 'RMSE', 'MAE', 'R2']])

print("\n=== Worst Performing Materials (by R2) ===")
print(results_df.nsmallest(5, 'R2')[['Material_ID', 'Cluster', 'RMSE', 'MAE', 'R2']])

# 3. Distribution of metrics across clusters
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribution of Performance Metrics by Cluster')

# RMSE Distribution
sns.boxplot(data=results_df, x='Cluster', y='RMSE', ax=axes[0])
axes[0].set_title('RMSE Distribution')
axes[0].set_yscale('log')  # Log scale for better visualization

# MAE Distribution
sns.boxplot(data=results_df, x='Cluster', y='MAE', ax=axes[1])
axes[1].set_title('MAE Distribution')
axes[1].set_yscale('log')

# R2 Distribution
sns.boxplot(data=results_df, x='Cluster', y='R2', ax=axes[2])
axes[2].set_title('R² Distribution')

plt.tight_layout()
plt.savefig(output_dir / 'metric_distributions.png')
plt.close()

# 4. Correlation between metrics
correlation_matrix = results_df[['RMSE', 'MAE', 'R2']].corr()
print("\n=== Correlation between metrics ===")
print(correlation_matrix.round(3))

# 5. Performance relative to data size
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model Performance vs Dataset Size')

# R2 vs Training Size
sns.scatterplot(data=results_df, x='Train_Size', y='R2', hue='Cluster', ax=axes[0])
axes[0].set_title('R² vs Training Size')

# RMSE vs Training Size
sns.scatterplot(data=results_df, x='Train_Size', y='RMSE', hue='Cluster', ax=axes[1])
axes[1].set_title('RMSE vs Training Size')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(output_dir / 'performance_vs_size.png')
plt.close()

# 6. Calculate percentage of "good" models per cluster
r2_threshold = 0.3  # Define what constitutes a "good" model
good_models = results_df[results_df['R2'] > r2_threshold].groupby('Cluster').size()
total_models = results_df.groupby('Cluster').size()
success_rate = (good_models / total_models * 100).round(2)

print("\n=== Success Rate by Cluster (R² > 0.3) ===")
print(success_rate)

# 7. Summary insights
print("\n=== Summary Insights ===")
best_cluster = cluster_stats.loc[:, ('R2', 'mean')].idxmax()
worst_cluster = cluster_stats.loc[:, ('R2', 'mean')].idxmin()
most_consistent = cluster_stats.loc[:, ('R2', 'std')].idxmin()

print(f"- Best performing cluster (by mean R²): Cluster {best_cluster}")
print(f"- Worst performing cluster (by mean R²): Cluster {worst_cluster}")
print(f"- Most consistent cluster (by R² std): Cluster {most_consistent}")
print(f"- Overall success rate (R² > 0.3): {(len(results_df[results_df['R2'] > r2_threshold]) / len(results_df) * 100):.1f}%")

# Save detailed statistics to CSV
cluster_stats.to_csv(output_dir / 'cluster_performance_summary.csv')
