"""
PROJECT: Predictive Maintenance & Financial Risk Pipeline
AUTHOR: [Your Name]
PURPOSE: Identifying industrial mechanical failures using Ensemble Learning 
         to mitigate operational downtime costs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# 1. Load and Verify
df = pd.read_csv('data/ai4i2020.csv')
print("✅ Data Loaded.")

# 2. Metadata Extraction
def extract_metadata(df):
    tier_mapping = {'L': 'Low (50%)', 'M': 'Medium (30%)', 'H': 'High (20%)'}
    df['Quality_Tier'] = df['Type'].map(tier_mapping)
    df['Serial_Number'] = df['Product ID'].str.extract(r'(\d+)')
    return df

df = extract_metadata(df)
print("✅ Metadata Extracted.")

# 3. Predictive Modeling
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X = df[features]
y = df['Machine failure']

# --- XGBOOST SANITIZATION FIX ---
# This removes the [ ] brackets that cause XGBoost to crash
X.columns = [c.replace('[', '').replace(']', '').replace('<', '') for c in X.columns]
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BASELINE: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = rf_model.score(X_test, y_test)

# --- CHALLENGER: XGBoost ---
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_acc = xgb_model.score(X_test, y_test)

print(f"✅ Random Forest Accuracy: {rf_acc:.4f}")
print(f"✅ XGBoost Accuracy: {xgb_acc:.4f}")

# 4. Economic ROI Engine
def calculate_roi(y_actual, y_preds):
    maintenance_cost = 500
    failure_cost = 5000
    savings = sum((y_actual == 1) & (y_preds == 1)) * (failure_cost - maintenance_cost)
    losses = sum((y_actual == 1) & (y_preds == 0)) * failure_cost
    return savings - losses

final_preds = xgb_model.predict(X_test) if xgb_acc > rf_acc else rf_model.predict(X_test)
total_impact = calculate_roi(y_test, final_preds)
print(f"💰 Total Financial Impact of Model: ${total_impact:,}")

# 5. Export for Dashboard
results_df = X_test.copy()
results_df['Actual_Failure'] = y_test
results_df['Predicted_Failure'] = final_preds
results_df['ROI_Impact'] = total_impact
results_df.to_csv('output/final_dashboard_data.csv', index=False)
print("✅ Final Data Exported!")