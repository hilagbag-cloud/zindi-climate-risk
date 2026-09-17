"""
train.py — Main training, GroupKFold cross-validation, ensembling, and submission generation script.

Usage
-----
    # Default: 5-fold GroupKFold ensemble and generate submission/best_submission.csv & submission/submission_N.csv
    python src/train.py

    # CI run (LightGBM 5-fold CV without saving submission)
    python src/train.py --model lgbm --folds 5 --no-submit
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
from sklearn.model_selection import GroupKFold, StratifiedKFold
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
EXPERIMENTS_LOG = ROOT / "experiments.log"


# ---------------------------------------------------------------------------
# Logging & Submission Archiving
# ---------------------------------------------------------------------------

def get_best_historical_score() -> float:
    """Read experiments.log to find the highest CV score recorded so far."""
    if not EXPERIMENTS_LOG.exists():
        return -1.0

    best_score = -1.0
    with open(EXPERIMENTS_LOG, "r") as f:
        for line in f:
            if "CV Score:" in line:
                try:
                    parts = line.split("CV Score:")
                    score_str = parts[1].split("-")[0].strip()
                    score = float(score_str)
                    if score > best_score:
                        best_score = score
                except Exception:
                    continue
    return best_score


def log_experiment_to_file(
    exp_num: int,
    model_name: str,
    cv_score: float,
    status: str,
    submission_file: str = "",
) -> None:
    """Append experiment entry to experiments.log adhering to jules_guidelines."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    sub_str = f" - File: {submission_file}" if submission_file else ""
    log_line = f"[{now_str}] - Exp {exp_num:02d}: {model_name} - CV Score: {cv_score:.6f} - Status: [{status}]{sub_str}\n"

    with open(EXPERIMENTS_LOG, "a") as f:
        f.write(log_line)

    print(f"\nAppended to {EXPERIMENTS_LOG}:")
    print(f"  {log_line.strip()}")


def get_next_submission_index() -> int:
    """Find the next available submission_N.csv index in submission/."""
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = list(SUBMISSION_DIR.glob("submission_*.csv"))
    indices = []
    for f in existing_files:
        try:
            num = int(f.stem.split("submission_")[1])
            indices.append(num)
        except (IndexError, ValueError):
            continue
    return max(indices) + 1 if indices else 1


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
        "features": "all_engineered_v3",
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


# ---------------------------------------------------------------------------
# Cross-Validation Pipelines
# ---------------------------------------------------------------------------

def run_single_model_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    n_folds: int = 5,
    random_state: int = 42,
    use_group_kfold: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
    """Run CV for a single specified model."""
    X = train_df.drop(columns=[ID_COL, TARGET])
    y = train_df[TARGET].values
    X_test = test_df.drop(columns=[ID_COL])

    numeric_features, categorical_features = get_feature_columns(train_df)

    if use_group_kfold and "location" in train_df.columns:
        groups = train_df["location"].values
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(X, y, groups))
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(skf.split(X, y))

    oof_proba = np.zeros(len(train_df))
    test_proba = np.zeros(len(test_df))

    if model_name in ["catboost", "lgbm"]:
        X_cat = X.copy()
        X_test_cat = X_test.copy()
        for col in categorical_features:
            X_cat[col] = X_cat[col].astype("category")
            X_test_cat[col] = X_test_cat[col].astype("category")

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            X_tr, X_va = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            if model_name == "catboost":
                model = get_model(
                    model_name,
                    categorical_features=categorical_features,
                    random_state=random_state + fold_idx,
                    train_dir="/tmp/catboost_info",
                )
            else:
                model = get_model(
                    model_name,
                    random_state=random_state + fold_idx,
                )

            model.fit(X_tr, y_tr)
            oof_proba[val_idx] = model.predict_proba(X_va)[:, 1]
            test_proba += model.predict_proba(X_test_cat)[:, 1] / n_folds

            f_score = score_from_proba(y_va, oof_proba[val_idx], threshold=0.5)
            print(f"  Fold {fold_idx:2d}/{n_folds:2d} | F1: {f_score['f1']:.4f} | AUC: {f_score['roc_auc']:.4f} | Score: {f_score['combined_score']:.4f}")

    elif model_name == "xgboost":
        X_num = X.copy()
        X_test_num = X_test.copy()
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_num[categorical_features] = ord_enc.fit_transform(X[categorical_features].astype(str))
        X_test_num[categorical_features] = ord_enc.transform(X_test[categorical_features].astype(str))

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            X_tr, X_va = X_num.iloc[train_idx], X_num.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = get_model(model_name, random_state=random_state + fold_idx)
            model.fit(X_tr, y_tr)
            oof_proba[val_idx] = model.predict_proba(X_va)[:, 1]
            test_proba += model.predict_proba(X_test_num)[:, 1] / n_folds

            f_score = score_from_proba(y_va, oof_proba[val_idx], threshold=0.5)
            print(f"  Fold {fold_idx:2d}/{n_folds:2d} | F1: {f_score['f1']:.4f} | AUC: {f_score['roc_auc']:.4f} | Score: {f_score['combined_score']:.4f}")

    elif model_name == "logistic_regression":
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            model = get_model("logistic_regression", numeric_features=numeric_features, categorical_features=categorical_features)
            model.fit(X_tr, y_tr)
            oof_proba[val_idx] = model.predict_proba(X_va)[:, 1]
            test_proba += model.predict_proba(X_test)[:, 1] / n_folds

            f_score = score_from_proba(y_va, oof_proba[val_idx], threshold=0.5)
            print(f"  Fold {fold_idx:2d}/{n_folds:2d} | F1: {f_score['f1']:.4f} | AUC: {f_score['roc_auc']:.4f} | Score: {f_score['combined_score']:.4f}")

    best_res = find_best_threshold(y, oof_proba, low=0.20, high=0.80, step=0.005)
    best_threshold = best_res["threshold"]

    metrics = {
        "mean_f1": best_res["f1"],
        "mean_roc_auc": best_res["roc_auc"],
        "mean_combined_score": best_res["combined_score"],
        "threshold": best_threshold,
    }

    return oof_proba, test_proba, metrics, best_threshold


def run_ensemble_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
    use_group_kfold: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
    """Run CV using an ensemble of CatBoost, LightGBM, and XGBoost."""
    X = train_df.drop(columns=[ID_COL, TARGET])
    y = train_df[TARGET].values
    X_test = test_df.drop(columns=[ID_COL])

    numeric_features, categorical_features = get_feature_columns(train_df)

    X_cat = X.copy()
    X_test_cat = X_test.copy()
    for col in categorical_features:
        X_cat[col] = X_cat[col].astype("category")
        X_test_cat[col] = X_test_cat[col].astype("category")

    X_num = X.copy()
    X_test_num = X_test.copy()
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_num[categorical_features] = ord_enc.fit_transform(X[categorical_features].astype(str))
    X_test_num[categorical_features] = ord_enc.transform(X_test[categorical_features].astype(str))

    if use_group_kfold and "location" in train_df.columns:
        groups = train_df["location"].values
        gkf = GroupKFold(n_splits=n_folds)
        splits = list(gkf.split(X, y, groups))
        cv_name = "GroupKFold"
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(skf.split(X, y))
        cv_name = "StratifiedKFold"

    oof_cat = np.zeros(len(train_df))
    oof_lgb = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))

    test_cat = np.zeros(len(test_df))
    test_lgb = np.zeros(len(test_df))
    test_xgb = np.zeros(len(test_df))

    print(f"\nTraining 3-Model Ensemble with {n_folds}-Fold {cv_name}...")

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
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

    weights = (0.5, 0.3, 0.2)
    oof_proba = weights[0] * oof_cat + weights[1] * oof_xgb + weights[2] * oof_lgb
    test_proba = weights[0] * test_cat + weights[1] * test_xgb + weights[2] * test_lgb

    default_res = score_from_proba(y, oof_proba, threshold=0.5)
    best_res = find_best_threshold(y, oof_proba, low=0.20, high=0.80, step=0.005)
    best_threshold = best_res["threshold"]

    print(f"\n{'='*60}")
    print(f"  Overall OOF Evaluation ({n_folds} Folds {cv_name})")
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
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate climate risk model.")
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble", "catboost", "lgbm", "xgboost", "logistic_regression"],
        help="Model pipeline to train (default: ensemble)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed decision threshold for TargetF1 (default: optimal threshold found on OOF)",
    )
    parser.add_argument(
        "--no-group-kfold",
        action="store_true",
        help="Use StratifiedKFold instead of GroupKFold",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Skip generating submission CSV",
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

    # 3. Cross-Validation
    use_group_kfold = not args.no_group_kfold
    if args.model == "ensemble":
        oof_proba, test_proba, metrics, best_threshold = run_ensemble_cv(
            train_df, test_df, n_folds=args.folds, use_group_kfold=use_group_kfold
        )
        model_log_name = "CatBoost+LGBM+XGBoost_Ensemble"
    else:
        oof_proba, test_proba, metrics, best_threshold = run_single_model_cv(
            train_df, test_df, model_name=args.model, n_folds=args.folds, use_group_kfold=use_group_kfold
        )
        model_log_name = args.model

    # 4. Check historical best score and log experiment
    cv_score = metrics["mean_combined_score"]
    best_historical = get_best_historical_score()

    # 5. Handle submission archiving
    is_improvement = cv_score > best_historical
    sub_filename = ""

    if not args.no_submit:
        threshold_to_use = best_threshold if args.threshold is None else args.threshold
        print(f"\nGenerating submission file with threshold={threshold_to_use:.3f}...")
        test_labels = (test_proba >= threshold_to_use).astype(int)

        submission = pd.DataFrame({
            ID_COL: test_df[ID_COL],
            "TargetF1": test_labels,
            "TargetRAUC": test_proba,
        })

        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

        if is_improvement or best_historical < 0:
            next_idx = get_next_submission_index()
            sub_filename = f"submission_{next_idx}.csv"
            archive_path = SUBMISSION_DIR / sub_filename
            best_path = SUBMISSION_DIR / "best_submission.csv"

            submission.to_csv(archive_path, index=False)
            submission.to_csv(best_path, index=False)

            print(f"\nNEW BEST SCORE! Submission archived to {archive_path} and copied to {best_path}")
            print(f"  Shape: {submission.shape}")
            print(f"  TargetF1 distribution: {submission['TargetF1'].value_counts().to_dict()}")
            print(f"  TargetRAUC stats: min={submission['TargetRAUC'].min():.4f}, "
                  f"max={submission['TargetRAUC'].max():.4f}, "
                  f"mean={submission['TargetRAUC'].mean():.4f}")
            status = "Improved" if best_historical >= 0 else "Baseline"
        else:
            status = "Keep"
            print(f"\nCV score ({cv_score:.6f}) did not beat best historical score ({best_historical:.6f}). best_submission.csv unchanged.")
    else:
        status = "Baseline" if best_historical < 0 else "Keep"

    # Log to experiments.csv and experiments.log
    log_experiment(model_log_name, args.folds, metrics)
    exp_num = get_next_submission_index() - 1 if is_improvement else 1
    log_experiment_to_file(exp_num=exp_num, model_name=model_log_name, cv_score=cv_score, status=status, submission_file=sub_filename)


if __name__ == "__main__":
    main()
