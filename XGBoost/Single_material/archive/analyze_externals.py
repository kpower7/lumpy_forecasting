import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the feature importance file
feature_importance = pd.read_csv(r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\IAP 2025\SCC\XGBoost\material_4_feature_importance.csv")

# Group features by type
def categorize_feature(feature_name):
    if any(ext in feature_name for ext in ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']):
        if 'lag' in feature_name:
            return 'External Variable Lag'
        elif 'roll_mean' in feature_name:
            return 'External Variable Rolling Mean'
        else:
            return 'External Variable Direct'
    elif 'Demand' in feature_name:
        return 'Demand History'
    else:
        return 'Time Features'

feature_importance['Category'] = feature_importance['Feature'].apply(categorize_feature)

# Calculate importance by category
category_importance = feature_importance.groupby('Category')['Importance'].sum().sort_values(ascending=False)

print("Feature Importance by Category:")
print("=" * 50)
print(category_importance)

print("\nTop 10 Individual Features:")
print("=" * 50)
print(feature_importance[['Feature', 'Category', 'Importance']].head(10))

# Plot category importance
plt.figure(figsize=(10, 6))
category_importance.plot(kind='bar')
plt.title('Feature Importance by Category')
plt.xlabel('Feature Category')
plt.ylabel('Total Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot top 20 individual features with color by category
plt.figure(figsize=(12, 8))
top_20 = feature_importance.head(20)
sns.barplot(data=top_20, x='Importance', y='Feature', hue='Category', dodge=False)
plt.title('Top 20 Most Important Features')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()
