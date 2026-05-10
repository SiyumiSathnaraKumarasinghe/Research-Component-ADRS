"""
Retrain Stage 1 model with 21 features (no district_enc).
Saves model artifacts to BOTH filename sets so Notebook 05
and the Streamlit app are consistent.
"""
import os
import sys
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed',
                         'FMD_model_ready_main refined_final_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- Load data ---
df = pd.read_csv(DATA_FILE)
print(f"Dataset: {df.shape}")

# --- Feature setup (21 features, NO district_enc) ---
TARGET = 'Outbreak status'
drop_cols = ['year', 'month_num', 'district', 'PCODE', 'district_enc', TARGET]
feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"district_enc included: {'district_enc' in feature_cols}")

# --- Train/Test split ---
train = df[df['year'] < 2024]
test = df[df['year'] == 2024].copy()

X_train, y_train = train[feature_cols], train[TARGET]
X_test, y_test = test[feature_cols], test[TARGET]

print(f"\nTraining set: {len(train)} rows (2017-2023)")
print(f"Test set:     {len(test)} rows (2024)")

# --- Scale and train ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_s, y_train)

test['pred'] = model.predict(X_test_s)
test['prob'] = model.predict_proba(X_test_s)[:, 1]

# --- Evaluation ---
print("\n" + "=" * 60)
print("  Classification Report — 2024 (21 features, no district_enc)")
print("=" * 60)
print(classification_report(y_test, test['pred'],
                            target_names=['No Outbreak', 'Outbreak'],
                            zero_division=0))
prec = precision_score(y_test, test['pred'], zero_division=0)
rec = recall_score(y_test, test['pred'], zero_division=0)
f1 = f1_score(y_test, test['pred'], zero_division=0)
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")

# --- Save to BOTH filename sets ---
# Set 1: Notebook 05 filenames (pickle)
path1_model = os.path.join(MODEL_DIR, 'final_model_stage1.pkl')
path1_scaler = os.path.join(MODEL_DIR, 'scaler_stage1.pkl')
with open(path1_model, 'wb') as f:
    pickle.dump(model, f)
with open(path1_scaler, 'wb') as f:
    pickle.dump(scaler, f)
print(f"\n[OK] Saved: {path1_model}")
print(f"[OK] Saved: {path1_scaler}")

# Set 2: Streamlit app filenames (joblib)
path2_model = os.path.join(MODEL_DIR, 'stage1_lr_model.pkl')
path2_scaler = os.path.join(MODEL_DIR, 'stage1_scaler.pkl')
path2_features = os.path.join(MODEL_DIR, 'stage1_feature_cols.pkl')
joblib.dump(model, path2_model)
joblib.dump(scaler, path2_scaler)
joblib.dump(feature_cols, path2_features)
print(f"[OK] Saved: {path2_model}")
print(f"[OK] Saved: {path2_scaler}")
print(f"[OK] Saved: {path2_features}")

print(f"\nModel coefficient shape: {model.coef_.shape}")
print("Done.")
