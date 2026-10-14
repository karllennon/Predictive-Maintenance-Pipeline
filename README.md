# Predictive Maintenance & Financial Risk Pipeline

Predicts machine failures from sensor readings and chooses the alert threshold that minimizes maintenance cost, using the UCI AI4I 2020 dataset (10,000 records).

## Results (2,000-record test set, 61 failures)

| | Default threshold (0.50) | Cost-optimal threshold (0.10) |
|---|---|---|
| Failures caught (recall) | 59% | **85%** |
| Alerts that were real failures (precision) | 82% | 40% |
| Net savings | $33,000 | **$149,500** |

- **Model:** Random Forest (98.35% accuracy) narrowly beat XGBoost (98.30%). Accuracy is not a useful measure here: 96.95% of machines never fail, so a model that always predicts "no failure" scores almost as high.
- **Cost-optimal threshold:** a missed failure is assumed to cost $5,000 and a maintenance visit $500. The threshold is chosen on cross-validated predictions from the training set, so the test set is never used for tuning. At 0.10, the model accepts more false alarms because each costs one-tenth of a missed failure.
- **Limitation:** the test set has only 61 failures, so these percentages would shift by a few points on a different split. The costs are illustrative assumptions, not real maintenance data.

## How It Works
1. **Data checks (`step1.py`):** checks for required columns, missing values, and sensor readings outside expected ranges (air temperature, rotational speed, torque).
2. **Modeling (`complete_pipeline.py`):** trains Random Forest and XGBoost on five sensor features, keeps the more accurate model, then picks the cost-minimizing alert threshold.
3. **Cost model:** net savings = savings from caught failures, minus losses from missed failures, minus the cost of false alarms.
4. **Visuals (`visualize_results.py`):** a 3D scatter of test machines by air temperature, torque, and rotational speed, colored by actual failure, and a radar chart comparing average normalized sensor readings for failed vs. healthy machines.

<img width="726" height="726" alt="3D_Operational_Risk_Map" src="https://github.com/user-attachments/assets/7ac6c6d5-3f08-47d9-8170-34b1897dc21c" />
<img width="722" height="652" alt="Failure_DNA_Radar_Profile" src="https://github.com/user-attachments/assets/0e6361b0-7969-4d74-a911-3467cd5cd12b" />

## Tech Stack
Python 3.13, pandas, NumPy, scikit-learn, XGBoost, Plotly

## How to Run
1. Install packages: `pip install pandas numpy scikit-learn xgboost plotly`
2. From the repo root, run in order:
   - `python scripts/step1.py`
   - `python scripts/complete_pipeline.py`
   - `python scripts/visualize_results.py`

The dataset is included in `data/` under its CC BY 4.0 license.

---
*Data: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.*