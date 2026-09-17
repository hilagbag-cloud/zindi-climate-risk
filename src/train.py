"""
train.py — Main training, 10-fold cross-validation, ensembling, and submission generation script.

Usage
-----
    # Run 10-fold CV ensemble and generate submission/best_submission.csv
    python src/train.py

    # Custom model selection
    python src/train.py --model ensemble --folds 10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import score_from_proba, find_best_threshold, zindi_score
from src.feature_engineering import (
    ID_COL,
    TARGET,
    load_raw_data,
    build_features,
    get_feature_columns,
)
from src.model import get_model


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
EXPERIMENTS_CSV = ROOT / "experiments.csv"


# ---------------------------------------------------------------------------
# Cross-Validation & Ensembling
# ---------------------------------------------------------------------------

def run_ensemble_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_folds: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
    """Run stratified K-fold CV using an ensemble of CatBoost, LightGBM, and XGBoost.

    Returns
    -------
    oof_proba : np.ndarray
        Out-of-fold predicted probabilities for training set.
    test_proba : np.ndarray
        Averaged test predicted probabilities across folds and models.
    metrics : dict
        Aggregated metrics (F1, AUC, Combined score).
    best_threshold : float
        Optimal threshold found on OOF probabilities to maximize combined score.
    """
    X = train_df.drop(columns=[ID_COL, TARGET])
    y = train_df[TARGET].values
    X_test = test_df.drop(columns=[ID_COL])

    numeric_features, categorical_features = get_feature_columns(train_df)

    # Convert categorical columns to category dtypes for CatBoost and LightGBM
    X_cat = X.copy()
    X_test_cat = X_test.copy()
    for col in categorical_features:
        X_cat[col] = X_cat[col].astype("category")
        X_test_cat[col] = X_test_cat[col].astype("category")

    # Consistent Ordinal Encoding fitted on train for XGBoost
    X_num = X.copy()
    X_test_num = X_test.copy()
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_num[categorical_features] = ord_enc.fit_transform(X[categorical_features].astype(str))
    X_test_num[categorical_features] = ord_enc.transform(X_test[categorical_features].astype(str))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_cat = np.zeros(len(train_df))
    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))

    test_cat = np.zeros(len(test_df))
    test_lgb = np.zeros(len(test_df))
    test_xgb = np.zeros(len(test_df))

    print(f"\nTraining 3-Model Ensemble with {n_folds}-Fold Stratified Cross-Validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr_cat, X_va_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        X_tr_num, X_va_num = X_num.iloc[train_idx], X_num.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        # 1. CatBoost
        model_cat = get_model(
            "catboost",
            categorical_features=categorical_features,
            random_state=random_state + fold_idx,
            train_dir="/tmp/catboost_info",
        )
        model_cat.fit(X_tr_cat, y_tr)
        oof_cat[val_idx] = model_cat.predict_proba(X_va_cat)[:, 1]
        test_cat += model_cat.predict_proba(X_test_cat)[:, 1] / n_folds

        # 2. LightGBM
        model_lgb = get_model("lgbm", random_state=random_state + fold_idx)
        model_lgb.fit(X_tr_cat, y_tr)
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va_cat)[:, 1]
        test_lgb += model_lgb.predict_proba(X_test_cat)[:, 1] / n_folds

        # 3. XGBoost
        model_xgb = get_model("xgboost", random_state=random_state + fold_idx)
        model_xgb.fit(X_tr_num, y_tr)
        oof_xgb[val_idx] = model_xgb.predict_proba(X_va_num)[:, 1]
        test_xgb += model_xgb.predict_proba(X_test_num)[:, 1] / n_folds

        fold_ens = 0.5 * oof_cat[val_idx] + 0.3 * oof_xgb[val_idx] + 0.2 * oof_lgb[val_idx]
        f_score = zindi_score(y_va, (fold_ens >= 0.5).astype(int), fold_ens)
        print(f"  Fold {fold_idx:2d}/{n_folds:2d} | F1: {f_score['f1']:.4f} | AUC: {f_score['roc_auc']:.4f} | Score: {f_score['combined_score']:.4f}")

    # Combine OOF predictions
    weights = (0.5, 0.3, 0.2)  # CatBoost, XGBoost, LightGBM
    oof_proba = weights[0] * oof_cat + weights[1] * oof_xgb + weights[2] * oof_lgb
    test_proba = weights[0] * test_cat + weights[1] * test_xgb + weights[2] * test_lgb

    # Evaluate default 0.5 threshold
    default_res = score_from_proba(y, oof_proba, threshold=0.5)

    # Search best threshold on OOF for F1 optimization
    best_res = find_best_threshold(y, oof_proba, low=0.20, high=0.80, step=0.005)
    best_threshold = best_res["threshold"]

    print(f"\n{'='*60}")
    print(f"  Overall OOF Evaluation ({n_folds} Folds)")
    print(f"{'='*60}")
    print(f"  Default Thresh (0.500) -> F1: {default_res['f1']:.6f} | AUC: {default_res['roc_auc']:.6f} | Combined: {default_res['combined_score']:.6f}")
    print(f"  Optimal Thresh ({best_threshold:.3f}) -> F1: {best_res['f1']:.6f} | AUC: {best_res['roc_auc']:.6f} | Combined: {best_res['combined_score']:.6f}")

    metrics = {
        "mean_f1": best_res["f1"],
        "mean_roc_auc": best_res["roc_auc"],
        "mean_combined_score": best_res["combined_score"],
        "threshold": best_threshold,
    }

    return oof_proba, test_proba, metrics, best_threshold


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

def log_experiment(
    model_name: str,
    n_folds: int,
    metrics: dict[str, float],
    notes: str = "",
) -> None:
    """Append experiment result to experiments.csv."""
    row = {
        "experiment_id": f"{model_name}_{int(time.time())}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_name,
        "features": "all_engineered_v2",
        "cv_folds": n_folds,
        "mean_f1": f"{metrics['mean_f1']:.6f}",
        "mean_auc": f"{metrics['mean_roc_auc']:.6f}",
        "mean_combined_score": f"{metrics['mean_combined_score']:.6f}",
        "notes": f"{notes} (Opt Thresh: {metrics.get('threshold', 0.5):.3f})",
    }

    file_exists = EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nExperiment logged to {EXPERIMENTS_CSV}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate climate risk model.")
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble", "catboost", "lgbm", "xgboost"],
        help="Model pipeline to train (default: ensemble)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of CV folds (default: 10)",
    )
    args = parser.parse_args()

    # 1. Load raw data
    print("Loading raw data...")
    train_raw, test_raw, climate = load_raw_data(DATA_DIR)
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}, Climate: {climate.shape}")

    # 2. Build enriched features
    print("\nBuilding features...")
    train_df, test_df = build_features(train_raw, test_raw, climate)
    print(f"  Train features: {train_df.shape}, Test features: {test_df.shape}")

    # 3. Cross-Validation and Ensembling
    oof_proba, test_proba, metrics, best_threshold = run_ensemble_cv(
        train_df, test_df, n_folds=args.folds
    )

    # 4. Log Experiment
    log_experiment("CatBoost+LGBM+XGBoost_Ensemble", args.folds, metrics)

    # 5. Generate Submission File
    print("\nGenerating final submission file...")
    test_labels = (test_proba >= best_threshold).astype(int)

    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        "TargetF1": test_labels,
        "TargetRAUC": test_proba,
    })

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSION_DIR / "best_submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"\nSubmission saved to {out_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  TargetF1 distribution: {submission['TargetF1'].value_counts().to_dict()}")
    print(f"  TargetRAUC stats: min={submission['TargetRAUC'].min():.4f}, "
          f"max={submission['TargetRAUC'].max():.4f}, "
          f"mean={submission['TargetRAUC'].mean():.4f}")


if __name__ == "__main__":
    main()
