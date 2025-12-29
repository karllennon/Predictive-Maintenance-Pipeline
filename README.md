# Predictive Maintenance & Financial Risk Pipeline

## Project Overview
This project addresses industrial downtime costs by applying **Ensemble Learning** to sensor telemetry. By identifying mechanical failure signatures before they occur, the system enables an early-intervention strategy that maximizes asset uptime and reduces catastrophic repair costs.

## Key Impact
- **Accuracy:** Achieved a peak predictive accuracy of **98.45%** using an optimized **XGBoost** model.
- **Financial ROI:** Identified a net operational savings of **$65,500** through a custom-built economic utility engine.
- **Reliability:** Implemented a multi-stage validation layer to ensure data integrity and prevent model drift.

## Technical Stack
- **Languages:** Python 3.13
- **ML Models:** XGBoost (Challenger), Random Forest (Baseline)
- **Analytics:** Pandas, NumPy, Scikit-learn
- **Visualization:** Plotly (3D Operational Maps & Radar DNA Profiles)

## System Architecture
1. **Reliability Layer (`step1.py`):** Automated schema validation and sensor range-gatekeeping.
2. **Modeling Engine (`complete_pipeline.py`):** Feature engineering and competitive model evaluation (RF vs. XGBoost).
3. **Economic Engine:** Integration of business logic to calculate ROI based on maintenance costs vs. failure losses.
4. **Visual Analytics (`visualize_results.py`):** Interactive stakeholder dashboards for risk signature identification.

## Visual Insights
*(Tip: Once you upload your screenshots to GitHub, link them here!)*
- **3D Risk Map:** Visualizes high-risk clusters across Torque, RPM, and Temperature.
- **Radar DNA Profile:** Identifies "Squeeze" patterns between Torque and Tool Wear that signal imminent failure.

- <img width="726" height="726" alt="3D_Operational_Risk_Map" src="https://github.com/user-attachments/assets/7ac6c6d5-3f08-47d9-8170-34b1897dc21c" />
<img width="722" height="652" alt="Failure_DNA_Radar_Profile" src="https://github.com/user-attachments/assets/0e6361b0-7969-4d74-a911-3467cd5cd12b" />


---
*Developed for professional portfolio use. Data sourced from UCI Machine Learning Repository.*
