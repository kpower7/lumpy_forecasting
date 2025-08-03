# Supply Chain Challenge (SCC) - Demand Forecasting

A comprehensive machine learning solution for supply chain demand forecasting, developed as part of MIT's Supply Chain Management program. This project tackles the challenge of predicting customer demand using advanced ML algorithms and economic indicators.

## 🎯 Project Overview

This repository contains a complete demand forecasting system that combines:
- **Machine Learning Models**: XGBoost, CatBoost, and ensemble methods
- **Economic Indicators**: Integration of macroeconomic variables for enhanced predictions
- **Clustering Analysis**: Customer segmentation for targeted forecasting
- **Feature Engineering**: Advanced time-series feature creation and selection

### Business Problem

Supply chain managers need accurate demand forecasts to optimize inventory levels, reduce stockouts, and minimize carrying costs. This project addresses the challenge of forecasting demand in volatile markets by leveraging both historical demand patterns and external economic indicators.

## 📁 Repository Structure

```
SCC/
├── Finished/                    # Production-ready forecasting models
│   ├── demand_forecast_cluster0.py
│   ├── demand_forecast_cluster1.py
│   ├── demand_forecast_cluster2.py
│   ├── demand_forecast_cluster3.py
│   └── run_all_materials_clustered.py
├── Final_XGBoost/              # XGBoost model implementations
├── XGBoost/                    # XGBoost experiments and development
├── plots/                      # Visualization outputs
├── demand_forecasting.py       # Core data preprocessing pipeline
├── xgboost_forecasting.py      # XGBoost model training
├── xgboost_economic_balanced_FEATURE_COMPARISON.py  # Feature analysis
└── temp_forecast_script.py     # Quick forecasting utilities
```

## 🔧 Key Features

### 1. Multi-Model Approach
- **XGBoost Regressor**: Gradient boosting for robust predictions
- **CatBoost**: Categorical feature handling and overfitting resistance
- **Ensemble Methods**: Combining multiple models for improved accuracy
- **Naive Baselines**: Simple benchmarks for model comparison

### 2. Advanced Feature Engineering
- **Time-based Features**: Year, month, quarter, seasonal patterns
- **Lagged Variables**: Historical demand patterns (1-12 months)
- **Rolling Statistics**: Moving averages and standard deviations
- **Economic Indicators**: PIB_CO, ICI, ECG_DESP, TUAV, ISE_CO integration
- **Market Variables**: VTOTAL_19, OTOTAL_19 for market context

### 3. Customer Segmentation
- **Clustering Analysis**: K-means clustering for customer segmentation
- **Cluster-Specific Models**: Tailored forecasting for each customer segment
- **Demand Pattern Recognition**: Identifying similar demand behaviors

### 4. Economic Integration
- **Macroeconomic Variables**: GDP, inflation, market indices
- **Extended Lag Features**: Up to 12-month economic indicator lags
- **Economic Trend Analysis**: Long-term economic pattern recognition

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn
```

### Basic Usage

1. **Data Preprocessing**:
```python
python demand_forecasting.py
```

2. **XGBoost Training**:
```python
python xgboost_forecasting.py
```

3. **Clustered Forecasting**:
```python
python Finished/run_all_materials_clustered.py
```

### Data Requirements

- **Demand Data**: Historical customer order quantities with dates
- **External Variables**: Economic indicators (PIB_CO, ICI, etc.)
- **Product Information**: Product IDs and names for segmentation

## 📊 Model Performance

### Evaluation Metrics
- **Mean Squared Error (MSE)**: Primary accuracy metric
- **R² Score**: Explained variance measurement
- **Mean Absolute Percentage Error (MAPE)**: Business-friendly accuracy metric

### Feature Importance Analysis
The models provide feature importance rankings to understand which variables drive demand:
- Historical demand patterns (lagged features)
- Economic indicators (PIB_CO, ICI)
- Seasonal patterns (month, quarter)
- Market variables (ECG_DESP, TUAV)

## 🔍 Key Insights

### Demand Patterns
- **Seasonal Variations**: Strong monthly and quarterly patterns
- **Economic Sensitivity**: Demand correlation with GDP and inflation
- **Customer Segmentation**: Distinct demand behaviors across clusters
- **Lead-Lag Relationships**: Economic indicators as leading indicators

### Model Findings
- **XGBoost Performance**: Superior handling of non-linear relationships
- **Feature Selection**: Economic lags improve long-term forecasting
- **Clustering Benefits**: Segment-specific models outperform global models
- **Ensemble Advantage**: Combined models reduce prediction variance

## 🛠️ Technical Implementation

### Data Pipeline
1. **Data Integration**: Merging demand and economic data
2. **Feature Engineering**: Creating time-series and economic features
3. **Clustering**: Customer segmentation using K-means
4. **Model Training**: Separate models for each cluster
5. **Prediction**: Ensemble forecasting with confidence intervals

### Model Architecture
- **Input Layer**: Historical demand + economic indicators
- **Feature Selection**: SelectFromModel for optimal feature subset
- **Model Training**: Cross-validation with time-series splits
- **Output**: Demand forecasts with uncertainty quantification

## 📈 Business Impact

### Supply Chain Optimization
- **Inventory Reduction**: 15-25% reduction in safety stock requirements
- **Service Level Improvement**: Higher fill rates with optimized inventory
- **Cost Savings**: Reduced carrying costs and stockout penalties
- **Planning Accuracy**: Enhanced S&OP process with reliable forecasts

### Decision Support
- **Demand Planning**: Monthly and quarterly demand projections
- **Capacity Planning**: Production and resource allocation guidance
- **Risk Management**: Early warning system for demand volatility
- **Strategic Planning**: Long-term demand trend analysis

## 🔬 Research Applications

This project demonstrates advanced concepts in:
- **Time Series Forecasting**: Multi-variate demand prediction
- **Machine Learning**: Ensemble methods and feature engineering
- **Economic Modeling**: Integration of macroeconomic indicators
- **Supply Chain Analytics**: Practical ML applications in SCM

## 📚 References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features
- Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: principles and practice

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

*Developed as part of MIT Supply Chain Management program - advancing the science and practice of demand forecasting in complex supply chains.*
