# Single Material Prediction Models

This directory contains various models for predicting material demand using XGBoost.

## Directory Structure

### `/models`
Contains the current best production models:
- `hybrid_model_v2.py`: Current best model with proper data leakage prevention and balanced feature set

### `/experiments`
Contains experimental versions and variations:
- `hybrid_model_v2_1.py`: Enhanced version with additional data leakage fixes
- `hybrid_model_v2_2.py`: Version optimized for high-volatility materials
- `hybrid_model_v3.py`: Simplified version with reduced feature set
- `xgboost_economic_balanced.py`: Model with balanced economic indicators

### `/archive`
Contains older model versions and approaches:
- `hybrid_model.py`: Original hybrid model
- `demand_driven_model.py`: Basic demand-based model
- `economic_indicator_model.py`: Economic indicators model
- `market_sensitive_model.py`: Market-based prediction model
- `xgboost_economic_features.py`: Original economic features model

### `/csv`
Contains CSV files with model results and analysis

## Model Performance

The current best model (`hybrid_model_v2.py`) shows good performance on materials with stable patterns while maintaining proper data leakage prevention. For high-volatility materials, consider using `hybrid_model_v2_2.py` which implements additional volatility handling features.
