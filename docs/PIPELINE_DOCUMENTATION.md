# PIPELINE DOCUMENTATION

## 1. Project Aim and Scope
The objective of this project is to build a practical, explainable, district-level Foot-and-Mouth Disease (FMD) early warning system for Sri Lanka.

The system is designed to answer three questions:
1. Will an outbreak occur? (Stage 1)
2. If outbreak occurs, how severe will it be? (Stage 2)
3. How confident are we in Stage 2 severity prediction? (Bootstrap uncertainty)

This project includes data engineering, model development, validation, explainability, and an interactive Streamlit demo for decision support.

## 2. End-to-End System Logic
High-level flow:
Raw data -> preprocessing -> feature engineering -> Stage 1 outbreak prediction -> Stage 2 severity classification -> SHAP explainability -> bootstrap uncertainty -> Streamlit interface

Operational logic in app:
1. User selects district, month, year.
2. Stage 1 predicts outbreak probability.
3. If probability >= 0.35, Stage 2 predicts severity class.
4. If bootstrap result exists for that district-month-year, show 95% interval and confidence.
5. App displays recommendation and key explainability charts.

## 3. Technologies Used
Core stack:
- Python
- Jupyter notebooks
- Streamlit

Data & modeling:
- pandas
- numpy
- scikit-learn
- xgboost
- joblib

Explainability & visualization:
- shap
- matplotlib
- seaborn

Geospatial/data support:
- geopandas
- openpyxl (Excel read support)

## 4. Data Sources and Final Dataset
Main source groups:
- DAPH outbreak and severity records
- CHIRPS rainfall data
- Climate variables (humidity, temperature, wind)
- Livestock density data (including buffalo density)
- Administrative district-level references

Key files:
- [data/processed/FMD_model_ready_main refined_final_dataset.csv](data/processed/FMD_model_ready_main%20refined_final_dataset.csv): final feature dataset, shape (2400, 26)
- [data/processed/daph_severity_dataset.xlsx](data/processed/daph_severity_dataset.xlsx): severity source, shape (115, 5)

## 5. Features Engineered
Number of model features used in both Stage 1 and Stage 2: 21

Feature list:
1. sin_month
2. cos_month
3. monsoon_phase_First_Inter_Monsoon
4. monsoon_phase_SW_Monsoon
5. monsoon_phase_Second_Inter_Monsoon
6. monsoon_phase_NE_Monsoon
7. rainfall_mm
8. r3h
9. rfq
10. rain_lag1
11. rain_lag2
12. rfq_lag1
13. lat
14. lon
15. humidity
16. wind_speed
17. temp_lag1
18. humidity_lag1
19. wind_lag1
20. buffalo_density
21. livestock_density

Why these features:
- Temporal seasonality (sin/cos month, monsoon phases)
- Weather and lagged weather (immediate and delayed effects)
- Spatial context (lat/lon)
- Host density (livestock, buffalo)

## 6. Stage 1: Outbreak Prediction
Model selected for deployment:
- Logistic Regression with class weighting

Decision reason:
- Best recall-oriented performance for rare-outbreak detection with stable temporal behavior in walk-forward testing.

Validation strategy:
- Expanding walk-forward by year:
  - Train 2017-2021 -> Test 2022
  - Train 2017-2022 -> Test 2023
  - Train 2017-2023 -> Test 2024

Key Stage 1 metrics from [data/processed/model_comparison_final_model result.csv](data/processed/model_comparison_final_model%20result.csv):
- 2022: Precision 0.205, Recall 0.750, F1 0.321, PR-AUC 0.278, ROC-AUC 0.720
- 2023: Precision 0.090, Recall 0.833, F1 0.163, PR-AUC 0.160, ROC-AUC 0.808
- 2024: Precision 0.528, Recall 0.731, F1 0.613, PR-AUC 0.648, ROC-AUC 0.820
- Mean across 2022-2024: Precision 0.274, Recall 0.771, F1 0.366, PR-AUC 0.362, ROC-AUC 0.783

Deployment thresholds in app:
- High risk: probability >= 0.60
- Medium risk: 0.35 <= probability < 0.60
- Low risk: probability < 0.35

## 7. Stage 2: Severity Classification
Severity scoring formula:
Severity score = Cases * (Outbreak_Months / 12) + Deaths * 10

Domain thresholds:
- LOW if score < 50
- MEDIUM if 50 <= score < 300
- HIGH if score >= 300

Class mapping (explicit):
- LOW = 0, MEDIUM = 1, HIGH = 2

Why explicit mapping:
- Prevents accidental class-order issues from automatic encoders.

Training subset:
- Only outbreak rows (Outbreak status == 1): 306 records

Models compared:
- Random Forest
- XGBoost

Validation strategy:
- Leave-One-Year-Out (LOYO) for years 2018, 2019, 2021, 2022

Results from [data/processed/Stage2_LOYO_Results.csv](data/processed/Stage2_LOYO_Results.csv):
- Random Forest mean: Accuracy 0.465, Macro F1 0.398
- XGBoost mean: Accuracy 0.463, Macro F1 0.385

Final decision:
- Random Forest selected (slight but consistent edge in Macro F1).

Important predictors from [data/processed/Stage2_Feature_Importances.csv](data/processed/Stage2_Feature_Importances.csv):
- buffalo_density, livestock_density, lat, lon, wind_speed, humidity_lag1, temp_lag1

## 8. Explainability (SHAP)
SHAP implemented in [notebooks/08_SHAP_Explainability.ipynb](notebooks/08_SHAP_Explainability.ipynb)

Stage 1 explainability:
- Global beeswarm
- Feature importance bar chart
- Local waterfall explanation (single case)

Stage 2 explainability:
- Global beeswarm
- Feature importance bar chart

Exports:
- [data/processed/stage1_shap_values.csv](data/processed/stage1_shap_values.csv)
- [data/processed/stage2_shap_values.csv](data/processed/stage2_shap_values.csv)

Why SHAP was used:
- To convert predictions into interpretable feature contributions for technical and non-technical stakeholders.

## 9. Bootstrap Uncertainty for Stage 2
Notebook:
- [notebooks/09_Bootstrap_Uncertainty.ipynb](notebooks/09_Bootstrap_Uncertainty.ipynb)

Method:
- 100 bootstrap iterations
- Resample training data with replacement
- Train fresh RandomForestClassifier each iteration
- Predict each sample repeatedly
- Use vote mode for point estimate
- Use 5th and 95th percentile class codes for interval bounds

What it adds:
- Point severity class + 95% prediction interval + confidence percentage

Generated outputs:
- [data/processed/bootstrap_intervals.csv](data/processed/bootstrap_intervals.csv), shape (306, 9)
- [models/bootstrap_intervals.pkl](models/bootstrap_intervals.pkl)
- [models/bootstrap_config.pkl](models/bootstrap_config.pkl)

Bootstrap summary from artifacts:
- Mean confidence on full outbreak rows: 81.14%
- High-confidence predictions (>70%): 227
- Low-confidence predictions (<50%): 6
- Interval widths:
  - Narrow [X, X]: 76
  - Medium [LOW, MEDIUM]: 171
  - Wide [LOW, HIGH]: 59

LOYO bootstrap config summary:
- Mean coverage: 63.6%
- Mean confidence: 85.7%

## 10. How We Took Decisions
Decision framework used throughout:
1. Temporal validity first (no random split leakage)
2. Macro-F1 and recall emphasis for rare-event usefulness
3. Prefer simpler, robust models unless complex model gives clear gain
4. Keep explicit, auditable feature and class mappings
5. Add interpretability and uncertainty for operational trust

Examples:
- Stage 1 logistic regression selected because recall and stability were operationally favorable.
- Stage 2 random forest selected over XGBoost based on LOYO Macro-F1.
- Bootstrap kept at 100 iterations to balance reliability and runtime.

## 11. Validations Used
1. Walk-forward temporal validation (Stage 1)
2. Leave-One-Year-Out validation (Stage 2)
3. Cross-model benchmarking (Logistic Regression, RF, Gradient Boosting, XGBoost)
4. SHAP consistency checks (global + local explanation)
5. Uncertainty calibration proxy via interval coverage (bootstrap)

## 12. Accuracy and Performance Snapshot
Stage 1 deployed model (Logistic Regression, mean over 2022-2024):
- Precision: 0.274
- Recall: 0.771
- F1: 0.366
- PR-AUC: 0.362
- ROC-AUC: 0.783

Stage 2 deployed model (Random Forest, LOYO mean):
- Accuracy: 0.465
- Macro F1: 0.398

Interpretation:
- Stage 1 is tuned toward high sensitivity/recall for early warning.
- Stage 2 is moderate performance due to multi-class imbalance and limited severe-class samples.

## 13. Main Challenges and Mitigations
Challenge 1: Class imbalance (especially HIGH severity)
- Mitigation: class_weight='balanced', Macro-F1 monitoring, uncertainty intervals instead of hard-only labels.

Challenge 2: Temporal drift across years
- Mitigation: year-wise validation (walk-forward + LOYO), no random shuffle splits.

Challenge 3: Data integration from multiple sources
- Mitigation: strict merge keys (district, year), quality checks, null handling.

Challenge 4: Stakeholder trust and interpretability
- Mitigation: SHAP plots + local case-level report + clear recommendation logic.

Challenge 5: Overconfidence risk in severity outputs
- Mitigation: bootstrap intervals and confidence display in Streamlit.

Challenge 6: Artifact compatibility issues (e.g., dict vs LabelEncoder object)
- Mitigation: robust loading and explicit fallback mappings in implementation.

## 14. How Best Scenarios Are Selected
Best scenario selection logic:
1. Use temporally valid folds only.
2. Select model by mean validation metrics, not single-year peak.
3. Prioritize operational objective:
   - Stage 1: sensitivity for early warning
   - Stage 2: balanced multi-class utility (Macro-F1)
4. Add confidence context (bootstrap interval width + confidence%).

## 15. Streamlit System Demonstration (Full Steps)
### A. Preparation before demo
1. Ensure environment and dependencies are installed.
2. Ensure required artifacts exist in `models/` and `data/processed/`.
3. Confirm `bootstrap_intervals.csv` exists for uncertainty display.

### B. Run commands
```powershell
cd D:\Projects\Research_Component
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
streamlit run app\streamlit_app.py
```

### C. What to show in app
1. Sidebar selection (district, month, year)
2. Stage 1 probability and risk level
3. Stage 2 severity (if Stage 1 >= 0.35)
4. 95% interval and confidence
5. Recommendation panel
6. January forecast tab

## 16. 3-Minute Viva Demonstration Script
Minute 0:00-0:40 (Problem + architecture)
- Explain two-stage logic and why it is needed.
- Mention data period (2017-2024) and district coverage.

Minute 0:40-1:30 (Model evidence)
- Show Stage 1 performance focus on recall.
- Show Stage 2 LOYO Macro-F1 and why RF selected.
- Mention top severity drivers: buffalo_density and livestock_density.

Minute 1:30-2:20 (Trust and interpretability)
- Show SHAP chart and explain one key feature contribution.
- Show bootstrap interval and confidence for one selection.

Minute 2:20-3:00 (Operational use and decisions)
- Show recommendation output.
- Switch to all-district forecast tab.
- Close with how confidence helps prioritize veterinary actions.

## 17. Future Enhancements
1. Improve uncertainty calibration to raise coverage toward >= 80%.
2. Explore conformal prediction for class-conditional guarantees.
3. Add district-level drift monitoring dashboard.
4. Incorporate additional mobility/trade network signals.
5. Upgrade Stage 2 minority-class handling (cost-sensitive + augmentation).
6. Add automated retraining pipeline for yearly updates.
7. Add API endpoint + logging for operational deployment.

## 18. Reproducibility Order
Run notebooks in sequence:
1. 01 Data collection and understanding
2. 02 Preprocessing and feature engineering
3. 03 EDA
4. 04 Stage 1 model training/evaluation
5. 05 Final model and predictions
6. 06 Interactive prediction setup
7. 07 Stage 2 severity model
8. 08 SHAP explainability
9. 09 Bootstrap uncertainty

## 19. Key Artifacts for Final Submission
Core models:
- [models/stage1_lr_model.pkl](models/stage1_lr_model.pkl)
- [models/stage1_scaler.pkl](models/stage1_scaler.pkl)
- [models/stage1_feature_cols.pkl](models/stage1_feature_cols.pkl)
- [models/stage2_rf_model.pkl](models/stage2_rf_model.pkl)
- [models/stage2_feature_cols.pkl](models/stage2_feature_cols.pkl)

Uncertainty:
- [data/processed/bootstrap_intervals.csv](data/processed/bootstrap_intervals.csv)
- [models/bootstrap_config.pkl](models/bootstrap_config.pkl)

Reports:
- [docs/Stage2_Results_Report.md](docs/Stage2_Results_Report.md)
- [docs/SHAP_Explainability_Results_Report.md](docs/SHAP_Explainability_Results_Report.md)

App:
- [app/streamlit_app.py](app/streamlit_app.py)
