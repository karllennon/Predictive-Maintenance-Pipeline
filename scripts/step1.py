"""
PROJECT: Predictive Maintenance & Financial Risk Pipeline
MODULE: Data Reliability & Integrity Layer (Step 1)
PURPOSE: To perform initial data ingestion, sanity checks, and 
         outlier detection to ensure high-fidelity model inputs.
         
WORKFLOW:
1. Ingest raw sensor telemetry.
2. Validate data types and check for null values.
3. Establish data "Sanity Gates" (e.g., verifying realistic temperature/speed ranges).
"""

import pandas as pd
import numpy as np

class DataReliabilityLayer:
    def __init__(self, expected_schema):
        self.expected_schema = expected_schema

    def validate_data(self, df):
        reports = []
        
        # 1. Schema Check
        missing_cols = [col for col in self.expected_schema if col not in df.columns]
        if missing_cols:
            reports.append(f"CRITICAL: Missing columns {missing_cols}")

        # 2. Null Validation
        null_counts = df.isnull().sum()
        if null_counts.any():
            reports.append(f"WARNING: Null values found in: {null_counts[null_counts > 0].to_dict()}")

        # 3. Range Validation
        invalid_temp = df[(df['Air temperature [K]'] < 280) | (df['Air temperature [K]'] > 320)]
        if not invalid_temp.empty:
            reports.append(f"ALERT: {len(invalid_temp)} rows found with Air Temp outside normal range (280K-320K)")

        return reports

# --- Usage ---
expected_cols = ['Product ID', 'Air temperature [K]', 'Rotational speed [rpm]', 'Machine failure']
validator = DataReliabilityLayer(expected_cols)
df = pd.read_csv('data/ai4i2020.csv')
validation_issues = validator.validate_data(df)

if not validation_issues:
    print("✅ Data Integrity Verified. Proceeding to Metadata Extraction...")
else:
    print("❌ Validation Issues Found:")
    for issue in validation_issues:
        print(f" - {issue}")