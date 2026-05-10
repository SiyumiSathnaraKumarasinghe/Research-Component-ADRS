"""
Regenerate all Notebook 05 plots using the 21-feature model.
"""
import os, sys
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed',
                         'FMD_model_ready_main refined_final_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR = os.path.join(BASE_DIR, 'plots', '05_final_predictions')
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Load model and data ---
model = joblib.load(os.path.join(MODEL_DIR, 'stage1_lr_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'stage1_scaler.pkl'))
feature_cols = list(joblib.load(os.path.join(MODEL_DIR, 'stage1_feature_cols.pkl')))

df = pd.read_csv(DATA_FILE)
TARGET = 'Outbreak status'

train = df[df['year'] < 2024]
test = df[df['year'] == 2024].copy()

X_test = test[feature_cols]
y_test = test[TARGET]
X_test_s = scaler.transform(X_test)

test['pred'] = model.predict(X_test_s)
test['prob'] = model.predict_proba(X_test_s)[:, 1]

print(f"Features: {len(feature_cols)}")
print(f"Model coef shape: {model.coef_.shape}")
print(f"Test set: {len(test)} rows (2024)")

# --- Metrics ---
prec = precision_score(y_test, test['pred'], zero_division=0)
rec = recall_score(y_test, test['pred'], zero_division=0)
f1 = f1_score(y_test, test['pred'], zero_division=0)
roc = roc_auc_score(y_test, test['prob'])
pr_auc = average_precision_score(y_test, test['prob'])
print(f"\nPrecision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"ROC-AUC:   {roc:.3f}")
print(f"PR-AUC:    {pr_auc:.3f}")

# ============================
# PLOT 1: Confusion Matrix
# ============================
cm = confusion_matrix(y_test, test['pred'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Outbreak', 'Outbreak'],
            yticklabels=['No Outbreak', 'Outbreak'], ax=ax,
            annot_kws={'size': 16})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix - Final Model (2024)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print("\n[OK] confusion_matrix.png saved")

# ============================
# PLOT 2: Actual vs Predicted Heatmap
# ============================
districts_sorted = sorted(test['district'].unique())
months = list(range(1, 13))
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

actual_mat = np.zeros((len(districts_sorted), 12))
pred_mat   = np.zeros((len(districts_sorted), 12))

for i, dist in enumerate(districts_sorted):
    for j, m in enumerate(months):
        row = test[(test['district'] == dist) & (test['month_num'] == m)]
        if len(row) > 0:
            actual_mat[i, j] = row.iloc[0][TARGET]
            pred_mat[i, j]   = row.iloc[0]['pred']

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(actual_mat, ax=axes[0], cmap='YlOrRd',
            cbar_kws={'label': 'Outbreak'},
            xticklabels=month_labels, yticklabels=districts_sorted,
            linewidths=0.5, vmin=0, vmax=1)
axes[0].set_title('Actual Outbreaks 2024', fontsize=13)
axes[0].set_xlabel('Month')
axes[0].set_ylabel('District')

sns.heatmap(pred_mat, ax=axes[1], cmap='YlOrRd',
            cbar_kws={'label': 'Outbreak'},
            xticklabels=month_labels, yticklabels=districts_sorted,
            linewidths=0.5, vmin=0, vmax=1)
axes[1].set_title('Predicted Outbreaks 2024', fontsize=13)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('District')

plt.suptitle('Actual vs Predicted - Final Model (2024)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'actual_vs_predicted.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[OK] actual_vs_predicted.png saved")

# ============================
# PLOT 3: Outbreak Probability Heatmap
# ============================
prob_mat = np.zeros((len(districts_sorted), 12))
for i, dist in enumerate(districts_sorted):
    for j, m in enumerate(months):
        row = test[(test['district'] == dist) & (test['month_num'] == m)]
        if len(row) > 0:
            prob_mat[i, j] = row.iloc[0]['prob']

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(prob_mat, ax=ax, cmap='YlOrRd',
            cbar_kws={'label': 'Outbreak Probability'},
            xticklabels=month_labels, yticklabels=districts_sorted,
            linewidths=0.5, vmin=0, vmax=1, annot=True, fmt='.2f',
            annot_kws={'size': 7})
ax.set_title('Outbreak Probability - Final Model (2024)', fontsize=14)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('District', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'outbreak_probability.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("[OK] outbreak_probability.png saved")

# ============================
# PLOT 4: District Risk Ranking
# ============================
district_risk = test.groupby('district')['prob'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.YlOrRd(district_risk.values / max(district_risk.values.max(), 0.01))
ax.barh(district_risk.index, district_risk.values,
        color=colors, edgecolor='white')
ax.set_xlabel('Mean Outbreak Probability', fontsize=12)
ax.set_title('District Risk Ranking - 2024', fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'district_risk_ranking.png'), dpi=150)
plt.close()
print("[OK] district_risk_ranking.png saved")

# ============================
# PLOT 5: Model Confidence Metrics
# ============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC-AUC bar
metrics = {'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': roc, 'PR-AUC': pr_auc}
bars = axes[0].bar(metrics.keys(), metrics.values(), color=['#3498db','#e74c3c','#2ecc71','#9b59b6','#f39c12'])
axes[0].set_ylim(0, 1)
axes[0].set_title('Model Performance Metrics (2024)', fontsize=13)
for bar, val in zip(bars, metrics.values()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=10)

# Probability distribution
axes[1].hist(test[test[TARGET]==0]['prob'], bins=20, alpha=0.7, label='No Outbreak', color='#3498db')
axes[1].hist(test[test[TARGET]==1]['prob'], bins=20, alpha=0.7, label='Outbreak', color='#e74c3c')
axes[1].set_xlabel('Predicted Probability', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Probability Distribution by Class', fontsize=13)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'model_confidence_metrics.png'), dpi=150)
plt.close()
print("[OK] model_confidence_metrics.png saved")

print("\nAll plots regenerated successfully.")
