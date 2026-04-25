# ================================================================
# fraud_pipeline.py
# IEEE CIS Fraud Detection — Complete MLOps Assignment #4
# Tasks 1-9 in a single file (FULLY UPDATED & FIXED)
# Run: python fraud_pipeline.py
# ================================================================

import os, json, joblib, warnings, time
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models import infer_signature

from sklearn.model_selection    import train_test_split
from sklearn.impute             import SimpleImputer  # NO KNN - memory safe
from sklearn.ensemble           import RandomForestClassifier
from sklearn.feature_selection  import SelectFromModel
from sklearn.metrics            import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
)
from imblearn.over_sampling     import SMOTE
import xgboost  as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ================================================================
# GLOBAL CONFIGURATION
# ================================================================
BASE          = r"C:\Users\HP\OneDrive - FAST National University\Semester8\MLOPS\Assignments\4"
DATA_DIR      = os.path.join(BASE, "data")
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
MODELS_DIR    = os.path.join(BASE, "models")
MLFLOW_URI    = "http://localhost:5000"
EXPERIMENT    = "fraud-detection-master"
THRESHOLD     = 0.85       # deploy if AUC-ROC >= this
RANDOM_STATE  = 42

for d in [ARTIFACTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


# ================================================================
# SHARED HELPER — EVALUATE A MODEL
# ================================================================
def evaluate(model, X, y, selected_features=None):
    if selected_features is not None:
        X = X[selected_features]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    cm     = confusion_matrix(y, y_pred)
    return {
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y,    y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y,        y_pred, zero_division=0)), 4),
        "auc_roc":   round(float(roc_auc_score(y, y_prob)), 4),
        "cm":        cm,
        "y_prob":    y_prob,
        "y_pred":    y_pred,
    }


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ================================================================
# TASK 1 — STAGE 1: DATA INGESTION
# ================================================================
def stage1_data_ingestion():
    print("\n" + "="*60)
    print("  STAGE 1: DATA INGESTION")
    print("="*60)

    with mlflow.start_run(run_name="Stage1_DataIngestion", nested=True):
        train_txn = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
        train_id  = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))
        test_txn  = pd.read_csv(os.path.join(DATA_DIR, "test_transaction.csv"))
        test_id   = pd.read_csv(os.path.join(DATA_DIR, "test_identity.csv"))

        train_df = train_txn.merge(train_id, on="TransactionID", how="left")
        test_df  = test_txn.merge(test_id,   on="TransactionID", how="left")

        fraud_count = int(train_df["isFraud"].sum())
        fraud_pct   = round(float(train_df["isFraud"].mean()) * 100, 4)

        mlflow.log_param("train_rows",        int(train_df.shape[0]))
        mlflow.log_param("train_cols",        int(train_df.shape[1]))
        mlflow.log_metric("fraud_count",      fraud_count)
        mlflow.log_metric("fraud_percentage", fraud_pct)

        report = {"train_shape": list(train_df.shape),
                  "test_shape":  list(test_df.shape),
                  "fraud_count": fraud_count,
                  "fraud_pct":   fraud_pct}
        rp = os.path.join(ARTIFACTS_DIR, "s1_ingestion_report.json")
        save_json(report, rp);  mlflow.log_artifact(rp)

        print(f"  Train : {train_df.shape}  |  Test : {test_df.shape}")
        print(f"  Fraud rate : {fraud_pct}%")
        print("  STAGE 1 DONE ✓")
    return train_df, test_df


# ================================================================
# TASK 1 — STAGE 2: DATA VALIDATION
# ================================================================
def stage2_data_validation(train_df):
    print("\n" + "="*60)
    print("  STAGE 2: DATA VALIDATION")
    print("="*60)

    with mlflow.start_run(run_name="Stage2_DataValidation", nested=True):
        r      = {}
        passed = True

        r["has_target"]          = bool("isFraud" in train_df.columns)
        r["txn_id_unique"]       = bool(train_df["TransactionID"].is_unique)
        r["row_count"]           = int(len(train_df))
        r["min_rows_ok"]         = bool(len(train_df) >= 100_000)

        missing_pct              = train_df.isnull().mean()
        high_miss                = missing_pct[missing_pct > 0.80].index.tolist()
        r["high_missing_cols"]   = high_miss
        r["missing_check_ok"]    = bool(len(high_miss) == 0)

        fraud_rate               = float(train_df["isFraud"].mean())
        r["fraud_rate"]          = round(fraud_rate, 6)
        r["fraud_rate_ok"]       = bool(0.001 <= fraud_rate <= 0.10)

        unique_tgt               = sorted(train_df["isFraud"].dropna().unique().tolist())
        r["target_is_binary"]    = bool(set(unique_tgt).issubset({0, 1}))

        if not (r["has_target"] and r["min_rows_ok"] and r["target_is_binary"]):
            passed = False
        r["overall_passed"] = bool(passed)

        mlflow.log_metric("validation_passed",    int(passed))
        mlflow.log_metric("fraud_rate",           fraud_rate)
        mlflow.log_metric("high_missing_columns", len(high_miss))

        rp = os.path.join(ARTIFACTS_DIR, "s2_validation_report.json")
        save_json(r, rp);  mlflow.log_artifact(rp)

        print(f"  Status      : {'PASSED ✓' if passed else 'FAILED ✗'}")
        print(f"  Row count   : {r['row_count']:,}")
        print(f"  Fraud rate  : {fraud_rate*100:.4f}%")
        print(f"  High-miss   : {len(high_miss)} columns")
        print("  STAGE 2 DONE ✓")

    if not passed:
        raise ValueError("Validation failed — check s2_validation_report.json")
    return r


# ================================================================
# TASK 1 — STAGE 3: DATA PREPROCESSING (MEMORY EFFICIENT)
# ================================================================
def stage3_preprocessing(train_df, test_df):
    print("\n" + "="*60)
    print("  STAGE 3: DATA PREPROCESSING")
    print("="*60)

    with mlflow.start_run(run_name="Stage3_Preprocessing", nested=True):
        y_train  = train_df["isFraud"].copy()
        train_df = train_df.drop(columns=["isFraud"])

        common   = [c for c in train_df.columns if c in test_df.columns]
        train_df = train_df[common]
        test_df  = test_df[common]

        num_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns
                    if c != "TransactionID"]
        cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

        # Median imputation for numericals
        imp              = SimpleImputer(strategy="median")
        train_df[num_cols] = imp.fit_transform(train_df[num_cols])
        test_df[num_cols]  = imp.transform(test_df[num_cols])

        # Mode imputation for categoricals
        for col in cat_cols:
            m = train_df[col].mode()
            v = m[0] if len(m) > 0 else "Missing"
            train_df[col] = train_df[col].fillna(v)
            test_df[col]  = test_df[col].fillna(v)

        # Clip outliers 1st–99th percentile
        for col in num_cols:
            p01 = float(train_df[col].quantile(0.01))
            p99 = float(train_df[col].quantile(0.99))
            train_df[col] = train_df[col].clip(p01, p99)
            test_df[col]  = test_df[col].clip(p01, p99)

        train_df["isFraud"] = y_train.values

        mlflow.log_metric("missing_after", int(train_df.isnull().sum().sum()))
        mlflow.log_param("num_cols",       int(len(num_cols)))
        mlflow.log_param("cat_cols",       int(len(cat_cols)))
        mlflow.log_param("common_cols",    int(len(common)))

        print(f"  Common cols : {len(common)}")
        print(f"  Num cols    : {len(num_cols)}")
        print(f"  Cat cols    : {len(cat_cols)}")
        print(f"  Missing     : {int(train_df.isnull().sum().sum())}")
        print("  STAGE 3 DONE ✓")
    return train_df, test_df


# ================================================================
# TASK 2 — ADVANCED MISSING VALUE HANDLING (NO KNN - MEMORY SAFE)
# ================================================================
def task2_missing_values(train_df, test_df):
    print("\n" + "="*60)
    print("  TASK 2: ADVANCED MISSING VALUE HANDLING (No KNN)")
    print("="*60)

    with mlflow.start_run(run_name="T2_MissingValues", nested=True):

        # Missing indicator flags for columns with > 5% missing
        flag_cols = [c for c in train_df.columns
                     if c not in ["isFraud","TransactionID"]
                     and train_df[c].isnull().mean() > 0.05]

        for col in flag_cols:
            flag = f"{col}_was_missing"
            train_df[flag] = train_df[col].isnull().astype(int)
            test_df[flag]  = test_df[col].isnull().astype(int) \
                             if col in test_df.columns else 0

        # Use median imputation for ALL numeric columns (NO KNN - memory safe)
        num_cols    = train_df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ["isFraud","TransactionID"]]
        
        if num_cols:
            median_imp = SimpleImputer(strategy="median")
            train_df[num_cols] = median_imp.fit_transform(train_df[num_cols])
            test_tgts = [c for c in num_cols if c in test_df.columns]
            if test_tgts:
                test_df[test_tgts] = median_imp.transform(test_df[test_tgts])

        # Missing value distribution plot
        miss = train_df.isnull().mean().sort_values(ascending=False).head(20)
        if miss.max() > 0:
            fig, ax = plt.subplots(figsize=(10,5))
            miss.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Top 20 Columns — Missing Value %")
            ax.set_ylabel("Missing Fraction")
            ax.axhline(0.8, color="red",    ls="--", label="80% drop threshold")
            ax.axhline(0.5, color="orange", ls="--", label="50% threshold")
            ax.legend()
            plt.tight_layout()
            p = os.path.join(ARTIFACTS_DIR, "t2_missing_values.png")
            plt.savefig(p, dpi=150);  plt.close()
            mlflow.log_artifact(p)

        mlflow.log_param("missing_flag_cols", int(len(flag_cols)))
        mlflow.log_param("median_imputed_cols",  int(len(num_cols)))

        print(f"  Missing flags added : {len(flag_cols)}")
        print(f"  Median imputed cols : {len(num_cols)}")
        print("  TASK 2 MISSING DONE ✓")
    return train_df, test_df


# ================================================================
# TASK 1 — STAGE 4: FEATURE ENGINEERING
# ================================================================
def stage4_feature_engineering(train_df, test_df):
    print("\n" + "="*60)
    print("  STAGE 4: FEATURE ENGINEERING")
    print("="*60)

    with mlflow.start_run(run_name="Stage4_FeatureEngineering", nested=True):
        y_train  = train_df["isFraud"].copy()
        train_df = train_df.drop(columns=["isFraud"])

        # Time features
        if "TransactionDT" in train_df.columns:
            for df in [train_df, test_df]:
                df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
                df["day_of_week"] = (df["TransactionDT"] // (3600*24)) % 7
                df["is_night"]    = ((df["hour_of_day"] >= 22) |
                                     (df["hour_of_day"] <= 6)).astype(int)

        # Amount features
        if "TransactionAmt" in train_df.columns:
            for df in [train_df, test_df]:
                df["amt_log"]     = np.log1p(df["TransactionAmt"])
                df["amt_decimal"] = df["TransactionAmt"] % 1
                df["amt_cents"]   = (df["TransactionAmt"] % 1 * 100).round()

        # Target encoding (high cardinality > 20)
        cat_cols  = train_df.select_dtypes(include=["object"]).columns.tolist()
        high_card = [c for c in cat_cols if train_df[c].nunique() > 20]
        low_card  = [c for c in cat_cols if c not in high_card]

        global_mean = float(y_train.mean())
        for col in high_card:
            smoothing = 10
            stats     = y_train.groupby(train_df[col]).agg(["count","mean"])
            smooth    = (stats["count"]*stats["mean"] + smoothing*global_mean) \
                        / (stats["count"] + smoothing)
            train_df[col] = train_df[col].map(smooth).fillna(global_mean)
            test_df[col]  = test_df[col].map(smooth).fillna(global_mean)

        # Label encoding (low cardinality)
        for col in low_card:
            combined      = pd.concat([train_df[col], test_df[col]])
            cat_map       = {v: i for i, v in enumerate(combined.unique())}
            train_df[col] = train_df[col].map(cat_map).fillna(-1).astype(int)
            test_df[col]  = test_df[col].map(cat_map).fillna(-1).astype(int)

        train_df["isFraud"] = y_train.values

        mlflow.log_metric("total_features",   int(train_df.shape[1]))
        mlflow.log_param("high_card_enc",     int(len(high_card)))
        mlflow.log_param("low_card_enc",      int(len(low_card)))

        print(f"  Total features : {train_df.shape[1]}")
        print(f"  High-card enc  : {len(high_card)}")
        print(f"  Low-card enc   : {len(low_card)}")
        print("  STAGE 4 DONE ✓")
    return train_df, test_df


# ================================================================
# TASK 2 — IMBALANCE STRATEGY COMPARISON (FIXED)
# ================================================================
def task2_imbalance_comparison(X_train, X_val, y_train, y_val):
    print("\n" + "="*60)
    print("  TASK 2: IMBALANCE STRATEGY COMPARISON")
    print("="*60)

    with mlflow.start_run(run_name="T2_ImbalanceComparison", nested=True):
        neg, pos  = int((y_train==0).sum()), int((y_train==1).sum())
        spw       = neg / pos
        results   = {}

        # Strategy 1: SMOTE
        print("  [Strategy 1] SMOTE...")
        sm         = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        Xsm, ysm   = sm.fit_resample(X_train, y_train)
        m1 = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric="auc", tree_method="hist",
            random_state=RANDOM_STATE, verbosity=0,
            early_stopping_rounds=20
        )
        m1.fit(Xsm, ysm, eval_set=[(X_val, y_val)], verbose=False)
        r1 = evaluate(m1, X_val, y_val)
        results["SMOTE"] = {k: v for k, v in r1.items()
                            if k not in ["cm","y_prob","y_pred"]}
        results["SMOTE"]["cm"] = r1["cm"].tolist()

        # Strategy 2: Class Weighting
        print("  [Strategy 2] Class Weighting...")
        m2 = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric="auc", tree_method="hist",
            scale_pos_weight=spw, random_state=RANDOM_STATE, verbosity=0,
            early_stopping_rounds=20
        )
        m2.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        r2 = evaluate(m2, X_val, y_val)
        results["ClassWeight"] = {k: v for k, v in r2.items()
                                  if k not in ["cm","y_prob","y_pred"]}
        results["ClassWeight"]["cm"] = r2["cm"].tolist()

        # Log metrics
        for strat, res in results.items():
            mlflow.log_metric(f"{strat}_precision", res["precision"])
            mlflow.log_metric(f"{strat}_recall",    res["recall"])
            mlflow.log_metric(f"{strat}_f1",        res["f1_score"])
            mlflow.log_metric(f"{strat}_auc_roc",   res["auc_roc"])

        # Bar chart comparison
        labels  = ["Precision","Recall","F1-Score","AUC-ROC"]
        smote_v = [r1["precision"],r1["recall"],r1["f1_score"],r1["auc_roc"]]
        cw_v    = [r2["precision"],r2["recall"],r2["f1_score"],r2["auc_roc"]]
        x       = np.arange(len(labels))
        w       = 0.3
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(x-w/2, smote_v, w, label="SMOTE",          color="steelblue")
        ax.bar(x+w/2, cw_v,    w, label="Class Weighting", color="coral")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0,1.15); ax.set_ylabel("Score")
        ax.set_title("Task 2: Imbalance Strategy Comparison", fontsize=14)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        for i,(sv,cv) in enumerate(zip(smote_v,cw_v)):
            ax.text(i-w/2, sv+0.02, f"{sv:.3f}", ha="center", fontsize=9, fontweight="bold")
            ax.text(i+w/2, cv+0.02, f"{cv:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(ARTIFACTS_DIR, "t2_imbalance_comparison.png")
        plt.savefig(p, dpi=150); plt.close()
        mlflow.log_artifact(p)

        # ROC curves
        fig, ax = plt.subplots(figsize=(8,6))
        for label, prob in [("SMOTE", r1["y_prob"]), ("Class Weight", r2["y_prob"])]:
            fpr,tpr,_ = roc_curve(y_val, prob)
            ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_val, prob):.4f})", lw=2)
        ax.plot([0,1],[0,1],"k--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curves — Imbalance Comparison"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        p2 = os.path.join(ARTIFACTS_DIR, "t2_roc_imbalance.png")
        plt.savefig(p2, dpi=150); plt.close()
        mlflow.log_artifact(p2)

        rp = os.path.join(ARTIFACTS_DIR, "t2_imbalance_report.json")
        save_json(results, rp); mlflow.log_artifact(rp)

        print(f"\n  {'Strategy':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
        print(f"  {'-'*52}")
        for s,r in results.items():
            print(f"  {s:<16} {r['precision']:>10} {r['recall']:>10} "
                  f"{r['f1_score']:>10} {r['auc_roc']:>10}")
        print("  TASK 2 IMBALANCE DONE ✓")

    return results, r1["y_prob"], r2["y_prob"]

def evaluate_with_threshold_tuning(model, X, y, selected_features=None):
    """
    Evaluate model with optimal threshold tuning for imbalanced datasets.
    Finds the threshold that maximizes F1 score.
    """
    if selected_features is not None:
        X = X[selected_features]
    
    y_prob = model.predict_proba(X)[:, 1]
    
    # Find optimal threshold that maximizes F1 score
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    
    # Calculate F1 scores for all thresholds
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find best threshold (exclude the last point which has recall=0)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    best_f1 = f1_scores[best_idx]
    
    # Apply optimal threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # Also evaluate at default threshold for comparison
    y_pred_default = (y_prob >= 0.5).astype(int)
    default_recall = recall_score(y, y_pred_default, zero_division=0)
    default_precision = precision_score(y, y_pred_default, zero_division=0)
    
    print(f"    Threshold tuning: best={best_threshold:.4f} (F1={best_f1:.4f}) | "
          f"default=0.5 (P={default_precision:.3f}, R={default_recall:.3f})")
    
    cm = confusion_matrix(y, y_pred)
    
    return {
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y, y_prob)), 4),
        "cm": cm,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold": round(float(best_threshold), 4),
    }
# ================================================================
# TASK 3 — MODEL COMPLEXITY (XGBoost, LightGBM, Hybrid) - FIXED
# ================================================================
def task3_model_comparison(X_train, X_val, y_train, y_val):
    print("\n" + "="*60)
    print("  TASK 3: MODEL COMPLEXITY COMPARISON")
    print("  XGBoost | LightGBM | Hybrid (RF + XGBoost)")
    print("="*60)

    neg, pos = int((y_train==0).sum()), int((y_train==1).sum())
    spw = neg / pos
    actual_fraud_rate = y_train.mean()
    print(f"\n  [DEBUG] Training data stats:")
    print(f"    Total samples: {len(y_train):,}")
    print(f"    Fraud samples: {pos:,} ({actual_fraud_rate:.4%})")
    print(f"    Non-fraud: {neg:,}")
    print(f"    scale_pos_weight: {spw:.2f}")
    
    all_res = {}
    all_probs = {}
    sel_feats = None
    models_dict = {}

    # ============================================================
    # XGBoost
    # ============================================================
    with mlflow.start_run(run_name="T3_XGBoost", nested=True):
        print("\n  [XGBoost] Training...")
        xgb_m = xgb.XGBClassifier(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8, 
            colsample_bytree=0.8, 
            scale_pos_weight=spw,
            eval_metric="auc", 
            tree_method="hist", 
            random_state=RANDOM_STATE,
            verbosity=0, 
            early_stopping_rounds=30
        )
        xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Use threshold tuning for evaluation
        rx = evaluate_with_threshold_tuning(xgb_m, X_val, y_val)
        all_res["XGBoost"] = rx
        all_probs["XGBoost"] = rx["y_prob"]
        models_dict["XGBoost"] = xgb_m
        
        mlflow.log_metric("precision", rx["precision"])
        mlflow.log_metric("recall", rx["recall"])
        mlflow.log_metric("f1_score", rx["f1_score"])
        mlflow.log_metric("auc_roc", rx["auc_roc"])
        mlflow.log_metric("optimal_threshold", rx["threshold"])
        mlflow.xgboost.log_model(xgb_m, "xgboost_model")
        print(f"  XGBoost → P={rx['precision']} R={rx['recall']} "
              f"F1={rx['f1_score']} AUC={rx['auc_roc']} (threshold={rx['threshold']:.3f})")

    # ============================================================
    # LightGBM - FIXED VERSION
    # ============================================================
    with mlflow.start_run(run_name="T3_LightGBM", nested=True):
        print("\n  [LightGBM] Training...")
        
        # Cap scale_pos_weight to prevent extreme values
        capped_spw = min(spw, 50.0)
        print(f"    Capped scale_pos_weight: {capped_spw:.2f} (original: {spw:.2f})")
        
        lgb_m = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=capped_spw,  # Use capped value
            random_state=RANDOM_STATE,
            verbose=-1,
            min_child_samples=20,
            num_leaves=31,  # Reduced from 63 to prevent overfitting
            reg_lambda=0.1,  # L2 regularization
            reg_alpha=0.1,   # L1 regularization
            min_gain_to_split=0.01,
        )
        
        lgb_m.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc', 'logloss'],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        # Debug predictions
        y_probs = lgb_m.predict_proba(X_val)[:, 1]
        print(f"\n  [DEBUG] LightGBM raw predictions:")
        print(f"    Probability range: [{y_probs.min():.6f}, {y_probs.max():.6f}]")
        print(f"    Probabilities > 0.5: {(y_probs > 0.5).sum()} / {len(y_probs)}")
        print(f"    Probabilities > 0.1: {(y_probs > 0.1).sum()} / {len(y_probs)}")
        print(f"    Probabilities > 0.05: {(y_probs > 0.05).sum()} / {len(y_probs)}")
        
        # Use threshold tuning for evaluation
        rl = evaluate_with_threshold_tuning(lgb_m, X_val, y_val)
        all_res["LightGBM"] = rl
        all_probs["LightGBM"] = rl["y_prob"]
        models_dict["LightGBM"] = lgb_m
        
        mlflow.log_metric("precision", rl["precision"])
        mlflow.log_metric("recall", rl["recall"])
        mlflow.log_metric("f1_score", rl["f1_score"])
        mlflow.log_metric("auc_roc", rl["auc_roc"])
        mlflow.log_metric("optimal_threshold", rl["threshold"])
        mlflow.lightgbm.log_model(lgb_m, "lightgbm_model")
        print(f"\n  LightGBM → P={rl['precision']} R={rl['recall']} "
              f"F1={rl['f1_score']} AUC={rl['auc_roc']} (threshold={rl['threshold']:.3f})")

    # ============================================================
    # Hybrid: RF feature selection + XGBoost
    # ============================================================
    with mlflow.start_run(run_name="T3_Hybrid", nested=True):
        print("\n  [Hybrid] RF feature selection + XGBoost...")
        rf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        feat_imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        
        # Select top 50 features or fewer if less than 50
        n_features = min(50, X_train.shape[1])
        sel_feats = feat_imp.head(n_features)["feature"].tolist()
        print(f"    Selected {len(sel_feats)} / {X_train.shape[1]} features")
        
        hyb_m = xgb.XGBClassifier(
            n_estimators=300, 
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8, 
            colsample_bytree=0.8, 
            scale_pos_weight=spw,
            eval_metric="auc", 
            tree_method="hist", 
            random_state=RANDOM_STATE,
            verbosity=0, 
            early_stopping_rounds=30
        )
        hyb_m.fit(
            X_train[sel_feats], y_train,
            eval_set=[(X_val[sel_feats], y_val)], 
            verbose=False
        )
        
        rh = evaluate_with_threshold_tuning(hyb_m, X_val, y_val, sel_feats)
        all_res["Hybrid"] = rh
        all_probs["Hybrid"] = rh["y_prob"]
        models_dict["Hybrid"] = hyb_m
        
        mlflow.log_metric("precision", rh["precision"])
        mlflow.log_metric("recall", rh["recall"])
        mlflow.log_metric("f1_score", rh["f1_score"])
        mlflow.log_metric("auc_roc", rh["auc_roc"])
        mlflow.log_metric("optimal_threshold", rh["threshold"])
        mlflow.log_param("features_selected", len(sel_feats))
        mlflow.sklearn.log_model(hyb_m, "hybrid_model")
        
        fi_path = os.path.join(ARTIFACTS_DIR, "t3_feature_importance.csv")
        feat_imp.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)
        print(f"  Hybrid → P={rh['precision']} R={rh['recall']} "
              f"F1={rh['f1_score']} AUC={rh['auc_roc']} (threshold={rh['threshold']:.3f})")

    # ============================================================
    # COMPARISON PLOTS
    # ============================================================
    names = list(all_res.keys())
    prec_v = [all_res[n]["precision"] for n in names]
    rec_v = [all_res[n]["recall"] for n in names]
    f1_v = [all_res[n]["f1_score"] for n in names]
    auc_v = [all_res[n]["auc_roc"] for n in names]
    thresh_v = [all_res[n]["threshold"] for n in names]

    # Bar chart
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x-1.5*w, prec_v, w, label="Precision", color="steelblue")
    ax.bar(x-0.5*w, rec_v, w, label="Recall", color="coral")
    ax.bar(x+0.5*w, f1_v, w, label="F1-Score", color="green")
    ax.bar(x+1.5*w, auc_v, w, label="AUC-ROC", color="purple")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Task 3: Model Comparison", fontsize=14)
    ax.axhline(THRESHOLD, color="red", ls="--", label=f"Deploy threshold ({THRESHOLD})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Add threshold annotations
    for i, thresh in enumerate(thresh_v):
        ax.annotate(f"opt={thresh:.2f}", xy=(x[i], prec_v[i]+0.05), 
                   ha="center", fontsize=8, rotation=45)
    
    plt.tight_layout()
    p = os.path.join(ARTIFACTS_DIR, "t3_model_comparison.png")
    plt.savefig(p, dpi=150)
    plt.close()
    mlflow.log_artifact(p)

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "coral", "green"]
    for (n, prob), c in zip(all_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_val, prob)
        ax.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_val, prob):.4f})", 
                lw=2, color=c)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(ARTIFACTS_DIR, "t3_roc_curves.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    mlflow.log_artifact(p2)

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, names):
        cm = all_res[name]["cm"]
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["Not Fraud", "Fraud"],
               yticklabels=["Not Fraud", "Fraud"],
               xlabel="Predicted", ylabel="True",
               title=f"{name}\n(threshold={all_res[name]['threshold']:.3f})")
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.suptitle("Confusion Matrices — Fraud Class", fontsize=14)
    plt.tight_layout()
    p3 = os.path.join(ARTIFACTS_DIR, "t3_confusion_matrices.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(p3)

    # Precision-Recall curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for (n, prob), c in zip(all_probs.items(), colors):
        precision, recall, _ = precision_recall_curve(y_val, prob)
        ax.plot(recall, precision, label=f"{n}", lw=2, color=c)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p4 = os.path.join(ARTIFACTS_DIR, "t3_pr_curves.png")
    plt.savefig(p4, dpi=150)
    plt.close()
    mlflow.log_artifact(p4)

    # Save comparison CSV
    comp_rows = [{
        "Model": n,
        "Precision": all_res[n]["precision"],
        "Recall": all_res[n]["recall"],
        "F1": all_res[n]["f1_score"],
        "AUC-ROC": all_res[n]["auc_roc"],
        "Optimal_Threshold": all_res[n]["threshold"]
    } for n in names]
    comp_df = pd.DataFrame(comp_rows)
    cp = os.path.join(ARTIFACTS_DIR, "t3_model_comparison.csv")
    comp_df.to_csv(cp, index=False)
    mlflow.log_artifact(cp)

    best_name = max(all_res, key=lambda n: all_res[n]["auc_roc"])
    best_auc = all_res[best_name]["auc_roc"]

    print(f"\n  {'='*50}")
    print(f"  Model Comparison Summary:")
    print(comp_df.to_string(index=False))
    print(f"\n  Best Model: {best_name} (AUC={best_auc:.4f})")
    print("  TASK 3 DONE ✓")

    return all_res, models_dict, best_name, sel_feats, feat_imp


# ================================================================
# TASK 4 — COST-SENSITIVE LEARNING (FIXED)
# ================================================================
def task4_cost_sensitive(X_train, X_val, y_train, y_val):
    print("\n" + "="*60)
    print("  TASK 4: COST-SENSITIVE LEARNING")
    print("="*60)

    with mlflow.start_run(run_name="T4_CostSensitive", nested=True):

        neg, pos = int((y_train==0).sum()), int((y_train==1).sum())
        spw      = neg / pos

        # Standard training
        print("  [Standard] Training without cost-sensitivity...")
        m_std = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="auc", tree_method="hist", random_state=RANDOM_STATE,
            verbosity=0, early_stopping_rounds=30
        )
        m_std.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        r_std = evaluate(m_std, X_val, y_val)

        # Cost-sensitive: higher penalty for false negatives
        fn_cost = 10
        fp_cost = 1
        cost_spw = spw * (fn_cost / fp_cost)
        print(f"  [Cost-Sensitive] scale_pos_weight = {cost_spw:.2f} "
              f"(fn_cost={fn_cost}, fp_cost={fp_cost})...")
        m_cs = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="auc", tree_method="hist", scale_pos_weight=cost_spw,
            random_state=RANDOM_STATE, verbosity=0, early_stopping_rounds=30
        )
        m_cs.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        r_cs = evaluate(m_cs, X_val, y_val)

        # Business impact analysis
        cm_std = r_std["cm"]
        cm_cs  = r_cs["cm"]
        false_neg_cost     = 500   # cost per missed fraud ($)
        false_pos_cost     = 10    # cost per false alarm ($)

        fn_std = int(cm_std[1][0]);  fp_std = int(cm_std[0][1])
        fn_cs  = int(cm_cs[1][0]);   fp_cs  = int(cm_cs[0][1])

        loss_std = fn_std * false_neg_cost + fp_std * false_pos_cost
        loss_cs  = fn_cs  * false_neg_cost + fp_cs  * false_pos_cost
        savings  = loss_std - loss_cs

        # Log metrics
        for prefix, r in [("standard", r_std), ("cost_sensitive", r_cs)]:
            mlflow.log_metric(f"{prefix}_precision", r["precision"])
            mlflow.log_metric(f"{prefix}_recall",    r["recall"])
            mlflow.log_metric(f"{prefix}_f1",        r["f1_score"])
            mlflow.log_metric(f"{prefix}_auc_roc",   r["auc_roc"])
        mlflow.log_metric("business_savings_usd", float(savings))
        mlflow.log_param("fn_cost_weight",        fn_cost)
        mlflow.log_param("fp_cost_weight",        fp_cost)

        # Comparison bar chart
        labels  = ["Precision","Recall","F1-Score","AUC-ROC"]
        std_v   = [r_std["precision"],r_std["recall"],r_std["f1_score"],r_std["auc_roc"]]
        cs_v    = [r_cs["precision"], r_cs["recall"], r_cs["f1_score"], r_cs["auc_roc"]]
        x       = np.arange(len(labels)); w = 0.3
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(x-w/2, std_v, w, label="Standard",       color="steelblue")
        ax.bar(x+w/2, cs_v,  w, label="Cost-Sensitive", color="coral")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0,1.15); ax.set_ylabel("Score")
        ax.set_title("Task 4: Standard vs Cost-Sensitive Training", fontsize=14)
        ax.legend(); ax.grid(axis="y",alpha=0.3)
        for i,(sv,cv) in enumerate(zip(std_v,cs_v)):
            ax.text(i-w/2, sv+0.02, f"{sv:.3f}", ha="center", fontsize=9, fontweight="bold")
            ax.text(i+w/2, cv+0.02, f"{cv:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(ARTIFACTS_DIR, "t4_cost_sensitive_comparison.png")
        plt.savefig(p, dpi=150); plt.close()
        mlflow.log_artifact(p)

        # Business impact chart
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.bar(["Standard","Cost-Sensitive"],[loss_std,loss_cs],
                      color=["steelblue","coral"], edgecolor="black")
        ax.set_title("Task 4: Business Impact — Total Loss ($)", fontsize=13)
        ax.set_ylabel("Estimated Loss ($)")
        for bar, val in zip(bars,[loss_std,loss_cs]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                    f"${val:,}", ha="center", fontweight="bold")
        ax.annotate(f"Savings: ${savings:,}", xy=(0.5,0.95),
                    xycoords="axes fraction", ha="center",
                    fontsize=13, color="green", fontweight="bold")
        plt.tight_layout()
        p2 = os.path.join(ARTIFACTS_DIR, "t4_business_impact.png")
        plt.savefig(p2, dpi=150); plt.close()
        mlflow.log_artifact(p2)

        report = {
            "standard":       {k:v for k,v in r_std.items()
                                if k not in ["cm","y_prob","y_pred"]},
            "cost_sensitive":  {k:v for k,v in r_cs.items()
                                if k not in ["cm","y_prob","y_pred"]},
            "business_impact": {
                "fn_cost_per_case":  false_neg_cost,
                "fp_cost_per_case":  false_pos_cost,
                "standard_loss":     loss_std,
                "cost_sensitive_loss": loss_cs,
                "savings":           savings,
            }
        }
        rp = os.path.join(ARTIFACTS_DIR, "t4_cost_sensitive_report.json")
        save_json(report, rp); mlflow.log_artifact(rp)

        print(f"\n  {'Model':<20} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>10} {'AUC':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Standard':<20} {r_std['precision']:>10} {r_std['recall']:>10} "
              f"{r_std['f1_score']:>10} {r_std['auc_roc']:>10}")
        print(f"  {'Cost-Sensitive':<20} {r_cs['precision']:>10} {r_cs['recall']:>10} "
              f"{r_cs['f1_score']:>10} {r_cs['auc_roc']:>10}")
        print(f"\n  Business Impact:")
        print(f"    Standard loss      : ${loss_std:,}")
        print(f"    Cost-sensitive loss: ${loss_cs:,}")
        print(f"    Estimated savings  : ${savings:,}")
        print("  TASK 4 DONE ✓")

    return m_cs, report


# ================================================================
# TASK 1 — STAGE 5-6: FINAL MODEL TRAINING + EVALUATION
# ================================================================
def stage5_6_final_model(X_train, X_val, y_train, y_val,
                          best_model_name, all_models, sel_feats):
    print("\n" + "="*60)
    print(f"  STAGE 5-6: FINAL MODEL — {best_model_name}")
    print("="*60)

    with mlflow.start_run(run_name="Stage5_6_FinalModel", nested=True):
        model = all_models[best_model_name]
        feats = sel_feats if best_model_name == "Hybrid" else None
        r     = evaluate(model, X_val, y_val, feats)

        mlflow.log_metric("final_precision", r["precision"])
        mlflow.log_metric("final_recall",    r["recall"])
        mlflow.log_metric("final_f1",        r["f1_score"])
        mlflow.log_metric("final_auc_roc",   r["auc_roc"])

        cm = r["cm"]
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set(xticks=[0,1], yticks=[0,1],
               xticklabels=["Not Fraud","Fraud"],
               yticklabels=["Not Fraud","Fraud"],
               xlabel="Predicted", ylabel="True",
               title=f"Final Model: {best_model_name}")
        thresh = cm.max()/2
        for i in range(2):
            for j in range(2):
                ax.text(j,i,format(cm[i,j],"d"),ha="center",va="center",
                        color="white" if cm[i,j]>thresh else "black",fontsize=14)
        plt.tight_layout()
        p = os.path.join(ARTIFACTS_DIR, "s56_final_confusion_matrix.png")
        plt.savefig(p, dpi=150); plt.close()
        mlflow.log_artifact(p)

        # Save model bundle
        bundle      = {"model": model, "features": feats,
                       "model_name": best_model_name}
        bundle_path = os.path.join(MODELS_DIR, "final_model.pkl")
        joblib.dump(bundle, bundle_path)
        mlflow.log_artifact(bundle_path)
        
        # ============================================================
        # SAVE MODEL METRICS FOR API TO READ (ADD THIS!)
        # ============================================================
        metrics_file = os.path.join(ARTIFACTS_DIR, "final_model_metrics.json")
        model_metrics = {
            "recall": r["recall"],
            "auc": r["auc_roc"],
            "f1": r["f1_score"],
            "precision": r["precision"],
            "model_name": best_model_name,
            "timestamp": time.time(),
            "deployed": r["auc_roc"] >= THRESHOLD
        }
        save_json(model_metrics, metrics_file)
        mlflow.log_artifact(metrics_file)
        print(f"  Saved model metrics to {metrics_file}")

        print(f"  Precision : {r['precision']}")
        print(f"  Recall    : {r['recall']}")
        print(f"  F1-Score  : {r['f1_score']}")
        print(f"  AUC-ROC   : {r['auc_roc']}")
        print("  STAGE 5-6 DONE ✓")
    return r, bundle_path


# ================================================================
# TASK 1 — STAGE 7: CONDITIONAL DEPLOYMENT
# ================================================================
def stage7_conditional_deployment(best_name, auc_roc):
    print("\n" + "="*60)
    print("  STAGE 7: CONDITIONAL DEPLOYMENT")
    print("="*60)

    with mlflow.start_run(run_name="Stage7_Deployment", nested=True):
        mlflow.log_metric("auc_roc",         round(float(auc_roc),4))
        mlflow.log_param("deploy_threshold", float(THRESHOLD))
        mlflow.log_param("model",            best_name)

        if auc_roc >= THRESHOLD:
            print(f"  AUC-ROC {auc_roc:.4f} >= {THRESHOLD} → DEPLOYING {best_name} ✅")
            report = {"deployed":True,"model":best_name,
                      "auc_roc":round(float(auc_roc),4),
                      "threshold":THRESHOLD,"status":"Production"}
        else:
            print(f"  AUC-ROC {auc_roc:.4f} < {THRESHOLD} → NOT DEPLOYING ❌")
            report = {"deployed":False,"model":best_name,
                      "auc_roc":round(float(auc_roc),4),
                      "threshold":THRESHOLD,"reason":"Below threshold"}

        mlflow.log_param("deployed", bool(report["deployed"]))
        rp = os.path.join(ARTIFACTS_DIR, "s7_deployment_report.json")
        save_json(report, rp); mlflow.log_artifact(rp)
        print("  STAGE 7 DONE ✓")
    return report


# ================================================================
# TASK 7 — DRIFT SIMULATION
# ================================================================
def task7_drift_simulation(train_df):
    print("\n" + "="*60)
    print("  TASK 7: DRIFT SIMULATION (Time-Based)")
    print("="*60)

    with mlflow.start_run(run_name="T7_DriftSimulation", nested=True):

        X = train_df.drop(columns=["isFraud","TransactionID"], errors="ignore")
        X = X.select_dtypes(include=[np.number])
        y = train_df["isFraud"]

        # Split by time — earlier 70% = train, later 30% = test (drift)
        split_idx = int(len(X) * 0.70)
        X_early, y_early = X.iloc[:split_idx], y.iloc[:split_idx]
        X_late,  y_late  = X.iloc[split_idx:], y.iloc[split_idx:]

        neg, pos = int((y_early==0).sum()), int((y_early==1).sum())
        spw      = neg / pos

        # Train on early data
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            eval_metric="auc", tree_method="hist", scale_pos_weight=spw,
            random_state=RANDOM_STATE, verbosity=0
        )
        model.fit(X_early, y_early)

        # Evaluate on early (no drift)
        r_early = evaluate(model, X_early, y_early)

        # Evaluate on late (with drift)
        r_late  = evaluate(model, X_late, y_late)

        # Introduce new fraud patterns in late data
        X_late_drift = X_late.copy()
        if "TransactionAmt" in X_late_drift.columns:
            fraud_mask = y_late == 1
            X_late_drift.loc[fraud_mask, "TransactionAmt"] *= 2.5
        r_drift = evaluate(model, X_late_drift, y_late)

        # Feature distribution drift plot
        common_feats = [c for c in ["TransactionAmt","amt_log","hour_of_day"]
                        if c in X_early.columns]
        if common_feats:
            fig, axes = plt.subplots(1, len(common_feats), figsize=(14,4))
            if len(common_feats) == 1:
                axes = [axes]
            for ax, feat in zip(axes, common_feats):
                ax.hist(X_early[feat], bins=50, alpha=0.6,
                        label="Early (Train)", color="steelblue", density=True)
                ax.hist(X_late[feat],  bins=50, alpha=0.6,
                        label="Late (Drift)", color="coral",     density=True)
                ax.set_title(f"Distribution: {feat}")
                ax.set_xlabel(feat); ax.set_ylabel("Density")
                ax.legend()
            plt.suptitle("Task 7: Feature Distribution Drift", fontsize=13)
            plt.tight_layout()
            p = os.path.join(ARTIFACTS_DIR, "t7_feature_drift.png")
            plt.savefig(p, dpi=150); plt.close()
            mlflow.log_artifact(p)

        # Performance degradation plot
        metrics    = ["Precision","Recall","F1-Score","AUC-ROC"]
        early_vals = [r_early["precision"],r_early["recall"],
                      r_early["f1_score"],r_early["auc_roc"]]
        late_vals  = [r_late["precision"], r_late["recall"],
                      r_late["f1_score"],  r_late["auc_roc"]]
        drift_vals = [r_drift["precision"],r_drift["recall"],
                      r_drift["f1_score"], r_drift["auc_roc"]]

        x = np.arange(len(metrics)); w = 0.25
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(x-w, early_vals, w, label="Early data (train)", color="steelblue")
        ax.bar(x,   late_vals,  w, label="Late data (drift)",  color="coral")
        ax.bar(x+w, drift_vals, w, label="Late + new patterns",color="green")
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylim(0,1.15); ax.set_ylabel("Score")
        ax.set_title("Task 7: Performance Degradation Under Drift", fontsize=13)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p2 = os.path.join(ARTIFACTS_DIR, "t7_drift_performance.png")
        plt.savefig(p2, dpi=150); plt.close()
        mlflow.log_artifact(p2)

        mlflow.log_metric("early_auc",       r_early["auc_roc"])
        mlflow.log_metric("late_auc",        r_late["auc_roc"])
        mlflow.log_metric("drift_auc",       r_drift["auc_roc"])
        mlflow.log_metric("auc_degradation", round(r_early["auc_roc"]-r_late["auc_roc"],4))

        report = {
            "early_data":     {k:v for k,v in r_early.items()
                                if k not in ["cm","y_prob","y_pred"]},
            "late_data":      {k:v for k,v in r_late.items()
                                if k not in ["cm","y_prob","y_pred"]},
            "late_new_patterns": {k:v for k,v in r_drift.items()
                                   if k not in ["cm","y_prob","y_pred"]},
            "auc_degradation": round(r_early["auc_roc"]-r_late["auc_roc"],4),
        }
        rp = os.path.join(ARTIFACTS_DIR, "t7_drift_report.json")
        save_json(report, rp); mlflow.log_artifact(rp)

        print(f"  Early AUC  : {r_early['auc_roc']}")
        print(f"  Late AUC   : {r_late['auc_roc']}")
        print(f"  Drift AUC  : {r_drift['auc_roc']}")
        print(f"  Degradation: {report['auc_degradation']}")
        print("  TASK 7 DONE ✓")
    return report


# ================================================================
# TASK 8 — INTELLIGENT RETRAINING STRATEGY
# ================================================================
def task8_retraining_strategy(train_df):
    print("\n" + "="*60)
    print("  TASK 8: INTELLIGENT RETRAINING STRATEGY")
    print("="*60)

    with mlflow.start_run(run_name="T8_RetrainingStrategy", nested=True):

        X = train_df.drop(columns=["isFraud","TransactionID"], errors="ignore")
        X = X.select_dtypes(include=[np.number])
        y = train_df["isFraud"]
        n = len(X)

        neg, pos = int((y==0).sum()), int((y==1).sum())
        spw      = neg / pos

        def quick_train(Xt, yt, Xv, yv):
            m = xgb.XGBClassifier(n_estimators=100, max_depth=5,
                                   learning_rate=0.1, tree_method="hist",
                                   scale_pos_weight=spw,
                                   random_state=RANDOM_STATE, verbosity=0)
            m.fit(Xt, yt)
            return evaluate(m, Xv, yv)["auc_roc"]

        # Strategy A: Threshold-based retraining
        print("  [Strategy A] Threshold-based retraining...")
        retrain_threshold = 0.88
        chunks            = 5
        chunk_size        = n // chunks
        threshold_aucs    = []
        threshold_retrain_count = 0

        Xt = X.iloc[:chunk_size];  yt = y.iloc[:chunk_size]
        Xv = X.iloc[-chunk_size:]; yv = y.iloc[-chunk_size:]
        auc = quick_train(Xt, yt, Xv, yv)
        threshold_aucs.append(auc)

        for i in range(1, chunks-1):
            new_X = X.iloc[i*chunk_size:(i+1)*chunk_size]
            new_y = y.iloc[i*chunk_size:(i+1)*chunk_size]
            if auc < retrain_threshold:
                Xt = pd.concat([Xt, new_X])
                yt = pd.concat([yt, new_y])
                auc = quick_train(Xt, yt, Xv, yv)
                threshold_retrain_count += 1
            threshold_aucs.append(auc)

        # Strategy B: Periodic retraining (every 2 chunks)
        print("  [Strategy B] Periodic retraining (every 2 batches)...")
        periodic_aucs         = []
        periodic_retrain_count = 0

        Xt = X.iloc[:chunk_size]; yt = y.iloc[:chunk_size]
        auc = quick_train(Xt, yt, Xv, yv)
        periodic_aucs.append(auc)

        for i in range(1, chunks-1):
            new_X = X.iloc[i*chunk_size:(i+1)*chunk_size]
            new_y = y.iloc[i*chunk_size:(i+1)*chunk_size]
            if i % 2 == 0:
                Xt = pd.concat([Xt, new_X])
                yt = pd.concat([yt, new_y])
                auc = quick_train(Xt, yt, Xv, yv)
                periodic_retrain_count += 1
            periodic_aucs.append(auc)

        # Strategy C: Hybrid (threshold + periodic)
        print("  [Strategy C] Hybrid strategy...")
        hybrid_aucs          = []
        hybrid_retrain_count = 0

        Xt = X.iloc[:chunk_size]; yt = y.iloc[:chunk_size]
        auc = quick_train(Xt, yt, Xv, yv)
        hybrid_aucs.append(auc)

        for i in range(1, chunks-1):
            new_X = X.iloc[i*chunk_size:(i+1)*chunk_size]
            new_y = y.iloc[i*chunk_size:(i+1)*chunk_size]
            if auc < retrain_threshold or i % 3 == 0:
                Xt = pd.concat([Xt, new_X])
                yt = pd.concat([yt, new_y])
                auc = quick_train(Xt, yt, Xv, yv)
                hybrid_retrain_count += 1
            hybrid_aucs.append(auc)

        # Plot
        batches = list(range(1, len(threshold_aucs)+1))
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(batches, threshold_aucs, "o-", label="Threshold-based", color="steelblue", lw=2)
        ax.plot(batches, periodic_aucs,  "s-", label="Periodic",        color="coral",     lw=2)
        ax.plot(batches, hybrid_aucs,    "^-", label="Hybrid",          color="green",     lw=2)
        ax.axhline(retrain_threshold, color="red", ls="--",
                   label=f"Retrain threshold ({retrain_threshold})")
        ax.set_xlabel("Batch"); ax.set_ylabel("AUC-ROC")
        ax.set_title("Task 8: Retraining Strategy Comparison", fontsize=13)
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0.7,1.0)
        plt.tight_layout()
        p = os.path.join(ARTIFACTS_DIR, "t8_retraining_strategies.png")
        plt.savefig(p, dpi=150); plt.close()
        mlflow.log_artifact(p)

        mlflow.log_metric("threshold_retrains", threshold_retrain_count)
        mlflow.log_metric("periodic_retrains",  periodic_retrain_count)
        mlflow.log_metric("hybrid_retrains",    hybrid_retrain_count)
        mlflow.log_metric("threshold_final_auc",float(threshold_aucs[-1]))
        mlflow.log_metric("periodic_final_auc", float(periodic_aucs[-1]))
        mlflow.log_metric("hybrid_final_auc",   float(hybrid_aucs[-1]))

        report = {
            "threshold_strategy": {"retrains": threshold_retrain_count,
                                    "final_auc": float(threshold_aucs[-1]),
                                    "aucs": [float(a) for a in threshold_aucs]},
            "periodic_strategy":  {"retrains": periodic_retrain_count,
                                    "final_auc": float(periodic_aucs[-1]),
                                    "aucs": [float(a) for a in periodic_aucs]},
            "hybrid_strategy":    {"retrains": hybrid_retrain_count,
                                    "final_auc": float(hybrid_aucs[-1]),
                                    "aucs": [float(a) for a in hybrid_aucs]},
        }
        rp = os.path.join(ARTIFACTS_DIR, "t8_retraining_report.json")
        save_json(report, rp); mlflow.log_artifact(rp)

        print(f"\n  Strategy Comparison:")
        print(f"  {'Strategy':<20} {'Retrains':>10} {'Final AUC':>12}")
        print(f"  {'-'*45}")
        print(f"  {'Threshold':<20} {threshold_retrain_count:>10} {threshold_aucs[-1]:>12.4f}")
        print(f"  {'Periodic':<20} {periodic_retrain_count:>10} {periodic_aucs[-1]:>12.4f}")
        print(f"  {'Hybrid':<20} {hybrid_retrain_count:>10} {hybrid_aucs[-1]:>12.4f}")
        print("  TASK 8 DONE ✓")
    return report


# ================================================================
# TASK 9 — EXPLAINABILITY (SHAP) WITH FALLBACK
# ================================================================
def task9_explainability(model, X_train, X_val, sel_feats=None):
    print("\n" + "="*60)
    print("  TASK 9: EXPLAINABILITY (SHAP with fallback)")
    print("="*60)

    with mlflow.start_run(run_name="T9_Explainability", nested=True):

        # Use smaller sample for SHAP
        if sel_feats is not None:
            X_explain = X_val[sel_feats].head(200)
        else:
            X_explain = X_val.head(200)

        print(f"  Using {len(X_explain)} samples with {X_explain.shape[1]} features")

        # Try SHAP first, fallback to feature importance
        shap_success = False
        shap_df = None

        try:
            print("  Attempting SHAP TreeExplainer...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)

            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values

            # Plot SHAP summary
            fig, ax = plt.subplots(figsize=(10,8))
            shap.summary_plot(sv, X_explain, plot_type="bar", show=False, max_display=20)
            plt.title("Task 9: SHAP Feature Importance (Top 20)", fontsize=13)
            plt.tight_layout()
            p1 = os.path.join(ARTIFACTS_DIR, "t9_shap_importance.png")
            plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
            mlflow.log_artifact(p1)

            # SHAP beeswarm plot
            fig, ax = plt.subplots(figsize=(12,8))
            shap.summary_plot(sv, X_explain, show=False, max_display=20)
            plt.title("Task 9: SHAP Beeswarm — Feature Impact on Fraud", fontsize=13)
            plt.tight_layout()
            p2 = os.path.join(ARTIFACTS_DIR, "t9_shap_beeswarm.png")
            plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
            mlflow.log_artifact(p2)

            # Top features by mean SHAP
            mean_shap = np.abs(sv).mean(axis=0)
            feat_names = X_explain.columns.tolist()
            shap_df = pd.DataFrame({"feature": feat_names, "mean_shap": mean_shap})\
                        .sort_values("mean_shap", ascending=False)

            shap_csv = os.path.join(ARTIFACTS_DIR, "t9_shap_values.csv")
            shap_df.to_csv(shap_csv, index=False)
            mlflow.log_artifact(shap_csv)

            shap_success = True
            print("  ✅ SHAP analysis completed successfully")

        except Exception as e:
            print(f"  SHAP failed: {e}")
            print("  Using feature importance fallback...")

        # Fallback: Use model feature importance
        if not shap_success and hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_names = X_explain.columns.tolist()

            min_len = min(len(importance), len(feature_names))
            importance = importance[:min_len]
            feature_names = feature_names[:min_len]

            shap_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance
            }).sort_values("importance", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(shap_df["feature"].head(10).values[::-1],
                    shap_df["importance"].head(10).values[::-1], color="steelblue")
            ax.set_xlabel("Feature Importance")
            ax.set_title("Task 9: Top 10 Features (Model Importance Fallback)")
            plt.tight_layout()
            p_fallback = os.path.join(ARTIFACTS_DIR, "t9_feature_importance.png")
            plt.savefig(p_fallback, dpi=150)
            plt.close()
            mlflow.log_artifact(p_fallback)
            print("  ✅ Feature importance fallback completed")

        # Print top features
        print(f"\n  Top 10 Features by Importance:")
        print(f"  {'Feature':<35} {'Score':>12}")
        print(f"  {'-'*50}")
        for _, row in shap_df.head(10).iterrows():
            score_key = "mean_shap" if "mean_shap" in row else "importance"
            print(f"  {row['feature']:<35} {row[score_key]:>12.5f}")

        # Answer the question
        print(f"\n  🔍 WHY DOES THE MODEL PREDICT FRAUD?")
        print(f"  {'='*55}")
        print(f"  The model predicts fraud based on several key factors:")
        print(f"")
        for _, row in shap_df.head(5).iterrows():
            print(f"    • {row['feature']} (importance score: {row[score_key]:.4f})")
        print(f"")
        print(f"  These features help distinguish fraudulent transactions from")
        print(f"  legitimate ones by identifying anomalous patterns.")
        print("  TASK 9 DONE ✓")

    return shap_df


# ================================================================
# MAIN — RUN ALL TASKS
# ================================================================
if __name__ == "__main__":

    start_time = time.time()

    print("\n" + "#"*65)
    print("  IEEE CIS FRAUD DETECTION — COMPLETE MLOPS PIPELINE")
    print("  Tasks 1 through 9 (FULLY UPDATED)")
    print("#"*65)

    with mlflow.start_run(run_name="Master_Pipeline_All_Tasks"):

        mlflow.log_param("deploy_threshold", THRESHOLD)
        mlflow.log_param("random_state",     RANDOM_STATE)

        # ── TASK 1: STAGES 1–4 (Data Preparation) ──────────────
        print("\n>>> TASK 1: PIPELINE STAGES 1-4")
        train_df, test_df = stage1_data_ingestion()
        _                 = stage2_data_validation(train_df)
        train_df, test_df = stage3_preprocessing(train_df, test_df)

        # ── TASK 2: ADVANCED MISSING VALUES ─────────────────────
        print("\n>>> TASK 2: ADVANCED MISSING VALUE HANDLING")
        train_df, test_df = task2_missing_values(train_df, test_df)

        # ── TASK 1: STAGE 4 ─────────────────────────────────────
        train_df, test_df = stage4_feature_engineering(train_df, test_df)

        # ── PREPARE X / y ───────────────────────────────────────
        X = train_df.drop(columns=["isFraud","TransactionID"], errors="ignore")
        X = X.select_dtypes(include=[np.number])
        y = train_df["isFraud"]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # ── TASK 2: IMBALANCE COMPARISON ────────────────────────
        print("\n>>> TASK 2: IMBALANCE COMPARISON")
        imb_results, _, _ = task2_imbalance_comparison(
            X_train, X_val, y_train, y_val)

        # ── TASK 3: MODEL COMPLEXITY ─────────────────────────────
        print("\n>>> TASK 3: MODEL COMPLEXITY")
        t3_results, all_models, best_name, sel_feats, feat_imp = \
            task3_model_comparison(X_train, X_val, y_train, y_val)

        # ── TASK 4: COST-SENSITIVE LEARNING ─────────────────────
        print("\n>>> TASK 4: COST-SENSITIVE LEARNING")
        cs_model, t4_report = task4_cost_sensitive(
            X_train, X_val, y_train, y_val)

        # ── TASK 1: STAGES 5-6 (Final Model) ────────────────────
        print("\n>>> TASK 1: STAGES 5-6 FINAL MODEL & EVALUATION")
        final_eval, bundle_path = stage5_6_final_model(
            X_train, X_val, y_train, y_val,
            best_name, all_models, sel_feats)

        # ── TASK 1: STAGE 7 (Deployment) ────────────────────────
        print("\n>>> TASK 1: STAGE 7 CONDITIONAL DEPLOYMENT")
        deploy_report = stage7_conditional_deployment(
            best_name, final_eval["auc_roc"])

        # ── TASK 7: DRIFT SIMULATION ─────────────────────────────
        print("\n>>> TASK 7: DRIFT SIMULATION")
        drift_report = task7_drift_simulation(train_df)

        # ── TASK 8: RETRAINING STRATEGY ─────────────────────────
        print("\n>>> TASK 8: RETRAINING STRATEGY")
        retrain_report = task8_retraining_strategy(train_df)

        # ── TASK 9: EXPLAINABILITY ───────────────────────────────
        print("\n>>> TASK 9: EXPLAINABILITY")
        best_model = all_models[best_name]
        shap_df    = task9_explainability(
            best_model, X_train, X_val, sel_feats)

        # ── FINAL SUMMARY ────────────────────────────────────────
        elapsed = round(time.time() - start_time, 1)

        print("\n" + "#"*65)
        print("  ALL TASKS COMPLETE ✅")
        print("#"*65)
        print(f"\n  ⏱  Total time : {elapsed}s")
        print(f"\n  📊 TASK 2 — Imbalance Comparison:")
        for s,r in imb_results.items():
            print(f"     {s:<16} Recall={r['recall']}  AUC={r['auc_roc']}")
        print(f"\n  📊 TASK 3 — Model Comparison:")
        for n,r in t3_results.items():
            print(f"     {n:<10} Precision={r['precision']}  "
                  f"Recall={r['recall']}  AUC={r['auc_roc']}")
        print(f"\n  📊 TASK 4 — Cost-Sensitive:")
        print(f"     Savings: ${t4_report['business_impact']['savings']:,}")
        print(f"\n  📊 TASK 7 — Drift:")
        print(f"     AUC degradation: {drift_report['auc_degradation']}")
        print(f"\n  📊 TASK 8 — Best retraining strategy by final AUC:")
        best_strat = max(retrain_report,
                         key=lambda s: retrain_report[s]["final_auc"])
        print(f"     {best_strat} "
              f"(AUC={retrain_report[best_strat]['final_auc']:.4f})")
        print(f"\n  🏆 BEST MODEL  : {best_name} (AUC={final_eval['auc_roc']})")
        print(f"  🚀 DEPLOYED    : {deploy_report['deployed']}")
        print(f"  🔗 MLflow UI   : http://localhost:5000")
        print(f"\n  📁 Artifacts saved to: {ARTIFACTS_DIR}")
        print("#"*65 + "\n")