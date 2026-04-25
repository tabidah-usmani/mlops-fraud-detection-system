"""
Test LightGBM on REAL fraud detection data
Run: python test_lgb_real.py
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# Configuration
BASE = r"C:\Users\HP\OneDrive - FAST National University\Semester8\MLOPS\Assignments\4"
DATA_DIR = os.path.join(BASE, "data")

print("="*70)
print("TESTING LIGHTGBM ON REAL FRAUD DATA")
print("="*70)

# Load real data
print("\n1. LOADING REAL DATA...")
train_txn = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
train_id = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))

# Merge
train_df = train_txn.merge(train_id, on="TransactionID", how="left")
print(f"   Raw data shape: {train_df.shape}")

# Check fraud rate in REAL data
fraud_rate_real = train_df["isFraud"].mean()
fraud_count_real = train_df["isFraud"].sum()
total_transactions = len(train_df)

print(f"\n2. REAL DATA STATISTICS:")
print(f"   Total transactions: {total_transactions:,}")
print(f"   Fraud cases: {fraud_count_real:,}")
print(f"   Fraud rate: {fraud_rate_real:.6%} ({fraud_rate_real*100:.4f}%)")
print(f"   Legitimate cases: {total_transactions - fraud_count_real:,}")
print(f"   Imbalance ratio: {(1-fraud_rate_real)/fraud_rate_real:.2f}")

# Check if this matches your synthetic test
print(f"\n   Comparison with synthetic test:")
print(f"   Synthetic fraud rate: 5.4000%")
print(f"   REAL fraud rate: {fraud_rate_real*100:.4f}%")
print(f"   Difference: {abs(5.4 - fraud_rate_real*100):.2f}%")

# Simple preprocessing (match your pipeline's approach)
print("\n3. PREPROCESSING DATA...")
y = train_df["isFraud"].copy()
X = train_df.drop(columns=["isFraud", "TransactionID"])

# Select only numeric columns to avoid complexity
X = X.select_dtypes(include=[np.number])
print(f"   Numeric features: {X.shape[1]}")

# Check missing values
missing_pct = X.isnull().mean().max()
print(f"   Max missing %: {missing_pct:.2%}")

# Simple median imputation
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Train-test split (stratified to maintain fraud rate)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n4. DATA SPLIT:")
print(f"   Training: {len(X_train):,} samples")
print(f"     Fraud in train: {y_train.sum():,} ({y_train.mean():.6%})")
print(f"   Test: {len(X_test):,} samples")
print(f"     Fraud in test: {y_test.sum():,} ({y_test.mean():.6%})")

# Test different configurations
print("\n" + "="*70)
print("TESTING LIGHTGBM CONFIGURATIONS ON REAL DATA")
print("="*70)

results = []

# Configuration 1: Default (no handling)
print("\n[1/5] Default (no class imbalance handling)...")
model1 = lgb.LGBMClassifier(random_state=42, verbose=-1)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
y_prob1 = model1.predict_proba(X_test)[:, 1]

results.append({
    "Config": "Default",
    "Unique": np.unique(y_pred1),
    "Fraud Pred": y_pred1.sum(),
    "Precision": precision_score(y_test, y_pred1, zero_division=0),
    "Recall": recall_score(y_test, y_pred1, zero_division=0),
    "F1": f1_score(y_test, y_pred1, zero_division=0),
    "AUC": roc_auc_score(y_test, y_prob1)
})

# Configuration 2: scale_pos_weight
neg, pos = (y_train==0).sum(), (y_train==1).sum()
spw = neg / pos
print(f"\n[2/5] scale_pos_weight = {spw:.2f}...")
model2 = lgb.LGBMClassifier(scale_pos_weight=spw, random_state=42, verbose=-1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
y_prob2 = model2.predict_proba(X_test)[:, 1]

results.append({
    "Config": f"scale_pos_weight",
    "Unique": np.unique(y_pred2),
    "Fraud Pred": y_pred2.sum(),
    "Precision": precision_score(y_test, y_pred2, zero_division=0),
    "Recall": recall_score(y_test, y_pred2, zero_division=0),
    "F1": f1_score(y_test, y_pred2, zero_division=0),
    "AUC": roc_auc_score(y_test, y_prob2)
})

# Configuration 3: Capped scale_pos_weight
capped_spw = min(spw, 50)
print(f"\n[3/5] Capped scale_pos_weight = {capped_spw:.2f}...")
model3 = lgb.LGBMClassifier(scale_pos_weight=capped_spw, random_state=42, verbose=-1)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
y_prob3 = model3.predict_proba(X_test)[:, 1]

results.append({
    "Config": f"scale_pos_weight(capped)",
    "Unique": np.unique(y_pred3),
    "Fraud Pred": y_pred3.sum(),
    "Precision": precision_score(y_test, y_pred3, zero_division=0),
    "Recall": recall_score(y_test, y_pred3, zero_division=0),
    "F1": f1_score(y_test, y_pred3, zero_division=0),
    "AUC": roc_auc_score(y_test, y_prob3)
})

# Configuration 4: class_weight='balanced'
print(f"\n[4/5] class_weight='balanced'...")
model4 = lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
y_prob4 = model4.predict_proba(X_test)[:, 1]

results.append({
    "Config": "class_weight='balanced'",
    "Unique": np.unique(y_pred4),
    "Fraud Pred": y_pred4.sum(),
    "Precision": precision_score(y_test, y_pred4, zero_division=0),
    "Recall": recall_score(y_test, y_pred4, zero_division=0),
    "F1": f1_score(y_test, y_pred4, zero_division=0),
    "AUC": roc_auc_score(y_test, y_prob4)
})

# Configuration 5: class_weight='balanced' + threshold tuning
print(f"\n[5/5] class_weight='balanced' with threshold tuning...")
# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob4)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
y_pred_tuned = (y_prob4 >= best_threshold).astype(int)

print(f"   Optimal threshold: {best_threshold:.4f}")

results.append({
    "Config": f"balanced + threshold({best_threshold:.3f})",
    "Unique": np.unique(y_pred_tuned),
    "Fraud Pred": y_pred_tuned.sum(),
    "Precision": precision_score(y_test, y_pred_tuned, zero_division=0),
    "Recall": recall_score(y_test, y_pred_tuned, zero_division=0),
    "F1": f1_score(y_test, y_pred_tuned, zero_division=0),
    "AUC": roc_auc_score(y_test, y_prob4)
})

# Display results
print("\n" + "="*70)
print("RESULTS ON REAL DATA")
print("="*70)

results_df = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(results_df.to_string(index=False))

# Detailed analysis of the best model
print("\n" + "="*70)
print("DETAILED ANALYSIS (Best Model: class_weight='balanced')")
print("="*70)

print(f"\nProbability Distribution on Test Set:")
print(f"  Min: {y_prob4.min():.8f}")
print(f"  Max: {y_prob4.max():.8f}")
print(f"  Mean: {y_prob4.mean():.6f}")
print(f"  Median: {np.median(y_prob4):.6f}")
print(f"  Std: {y_prob4.std():.6f}")

print(f"\nSamples at different thresholds:")
for thresh in [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]:
    count = (y_prob4 >= thresh).sum()
    pct = (y_prob4 >= thresh).mean() * 100
    print(f"  > {thresh:.2f}: {count:>6,} / {len(y_prob4):,} ({pct:.2f}%)")

# Confusion matrix at optimal threshold
print(f"\nConfusion Matrix (at threshold={best_threshold:.4f}):")
cm = confusion_matrix(y_test, y_pred_tuned)
tn, fp, fn, tp = cm.ravel()
print(f"  ┌─────────────┬─────────────┐")
print(f"  │             │  Predicted  │")
print(f"  │             │  Neg   Pos  │")
print(f"  ├─────────────┼──────┬──────┤")
print(f"  │ Actual Neg  │ {tn:>5,} │ {fp:>5,} │")
print(f"  │ Actual Pos  │ {fn:>5,} │ {tp:>5,} │")
print(f"  └─────────────┴──────┴──────┘")

# Check if model predicts both classes
if len(np.unique(y_pred4)) == 1:
    print("\n⚠️  WARNING: Model predicts only ONE class on REAL data!")
    print(f"   Unique predictions: {np.unique(y_pred4)}")
    print(f"   This explains why precision/recall are 0 in MLflow!")
else:
    print(f"\n✅ SUCCESS: Model predicts BOTH classes on REAL data!")
    print(f"   Unique predictions: {np.unique(y_pred4)}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if fraud_rate_real < 0.01:
    print(f"\n✓ Real fraud rate ({fraud_rate_real:.4%}) is MUCH lower than synthetic test (5.4%)")
    print(f"✓ With such extreme imbalance, LightGBM needs:")
    print(f"   1. class_weight='balanced' (BETTER than scale_pos_weight)")
    print(f"   2. Threshold tuning (optimal threshold likely < 0.5)")
    print(f"   3. Proper evaluation metrics (F1, Precision-Recall, not just AUC)")
    print(f"\n→ Update your fraud_pipeline.py with the fixes provided!")
else:
    print(f"\n✓ Real fraud rate ({fraud_rate_real:.4%}) is similar to synthetic test")
    print(f"→ The issue might be in preprocessing or feature engineering")

print("\n✅ Test complete!")