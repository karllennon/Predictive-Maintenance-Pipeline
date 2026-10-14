"""
PROJECT: Predictive Maintenance & Financial Risk Pipeline
AUTHOR: Karl Godel
PURPOSE: Identifying industrial mechanical failures using Ensemble Learning 
         to mitigate operational downtime costs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# 1. Load and Verify
df = pd.read_csv('data/ai4i2020.csv')
print(" Data Loaded.")

# 2. Metadata Extraction
def extract_metadata(df):
    tier_mapping = {'L': 'Low (50%)', 'M': 'Medium (30%)', 'H': 'High (20%)'}
    df['Quality_Tier'] = df['Type'].map(tier_mapping)
    df['Serial_Number'] = df['Product ID'].str.extract(r'(\d+)')
    return df

df = extract_metadata(df)
print(" Metadata Extracted.")

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

print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# 4. Economic ROI Engine
def calculate_roi(y_actual, y_preds):
    y_actual = y_actual.reset_index(drop=True)  # Align pandas Series index with numpy array     
    maintenance_cost = 500
    failure_cost = 5000
    savings = sum((y_actual == 1) & (y_preds == 1)) * (failure_cost - maintenance_cost)
    losses = sum((y_actual == 1) & (y_preds == 0)) * failure_cost
    false_alarms = sum((y_actual == 0) & (y_preds == 1)) * maintenance_cost
    return savings - losses - false_alarms

best_model = xgb_model if xgb_acc > rf_acc else rf_model
final_preds = best_model.predict(X_test)

print(f"\nBaseline (always predict no failure): {1 - y_test.mean():.4f}")
print("\n[Default 0.50 threshold]")
print(classification_report(y_test, final_preds, digits=3))
total_impact = calculate_roi(y_test, final_preds)
print(f"Net savings at 0.50 threshold: ${total_impact:,}")

# 4b. Cost-Optimal Alert Threshold
# Out-of-fold probabilities on the training set, so the threshold is never tuned on test data
train_probs = cross_val_predict(best_model, X_train, y_train, cv=5, method='predict_proba')[:, 1]
thresholds = np.arange(0.05, 0.95, 0.05)
best_t = max(thresholds, key=lambda t: calculate_roi(y_train, (train_probs >= t).astype(int)))

test_probs = best_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_t).astype(int)
optimized_impact = calculate_roi(y_test, test_preds)

print(f"\n[Cost-optimal threshold: {best_t:.2f}]")
print(classification_report(y_test, test_preds, digits=3))
print(f"Net savings at cost-optimal threshold: ${optimized_impact:,}")

# 5. Export for Dashboard
results_df = X_test.copy()
results_df['Actual_Failure'] = y_test
results_df['Failure_Probability'] = test_probs
results_df['Predicted_Failure'] = test_preds
results_df['Total_Model_ROI'] = optimized_impact
results_df.to_csv('output/final_dashboard_data.csv', index=False)
print("Final Data Exported!")