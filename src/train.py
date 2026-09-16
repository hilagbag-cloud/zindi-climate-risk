"""
train.py — Main training, cross-validation and submission generation script.

Usage
-----
    # Default: 5-fold CV with LightGBM, writes submission/best_submission.csv
    python src/train.py

    # Baseline logistic regression
    python src/train.py --model logistic_regression --folds 5

    # XGBoost with custom threshold search
    python src/train.py --model xgboost --search-threshold
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

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import score_from_proba, find_best_threshold
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
# Cross-Validation
# ---------------------------------------------------------------------------

def run_cv(
    train_df: pd.DataFrame,
    model_name: str,
    n_folds: int = 5,
    search_threshold: bool = False,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run stratified K-fold cross-validation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Feature-engineered training DataFrame (with ID and target columns).
    model_name : str
        Name of the model to use (from model registry).
    n_folds : int
        Number of CV folds.
    search_threshold : bool
        If True, search for best threshold per fold; otherwise use 0.5.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    oof_proba : np.ndarray
        Out-of-fold predicted probabilities (indexed like train_df).
    metrics : dict
        Aggregated metrics across folds.
    """
    X = train_df.drop(columns=[ID_COL, TARGET])
    y = train_df[TARGET].values

    numeric_features, categorical_features = get_feature_columns(train_df)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_proba = np.zeros(len(train_df))
    fold_scores: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx}/{n_folds}")
        print(f"{'='*60}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline = get_model(model_name, numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)

        val_proba = pipeline.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba

        if search_threshold:
            fold_result = find_best_threshold(y_val, val_proba)
        else:
            fold_result = score_from_proba(y_val, val_proba, threshold=0.5)

        fold_scores.append(fold_result)
        print(f"  F1:       {fold_result['f1']:.6f}")
        print(f"  ROC-AUC:  {fold_result['roc_auc']:.6f}")
        print(f"  Combined: {fold_result['combined_score']:.6f}")
        if search_threshold:
            print(f"  Threshold: {fold_result['threshold']:.3f}")

    # Aggregate
    mean_f1 = np.mean([s["f1"] for s in fold_scores])
    mean_auc = np.mean([s["roc_auc"] for s in fold_scores])
    mean_combined = np.mean([s["combined_score"] for s in fold_scores])

    print(f"\n{'='*60}")
    print(f"  CV Summary ({n_folds} folds)")
    print(f"{'='*60}")
    print(f"  Mean F1:       {mean_f1:.6f}")
    print(f"  Mean ROC-AUC:  {mean_auc:.6f}")
    print(f"  Mean Combined: {mean_combined:.6f}")

    metrics = {
        "mean_f1": mean_f1,
        "mean_roc_auc": mean_auc,
        "mean_combined_score": mean_combined,
    }

    return oof_proba, metrics


# ---------------------------------------------------------------------------
# Full train + predict
# ---------------------------------------------------------------------------

def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Train on full training data and generate submission.

    Returns
    -------
    submission : pd.DataFrame
        DataFrame with columns [ID, TargetF1, TargetRAUC].
    """
    X_train = train_df.drop(columns=[ID_COL, TARGET])
    y_train = train_df[TARGET].values
    X_test = test_df.drop(columns=[ID_COL])

    numeric_features, categorical_features = get_feature_columns(train_df)

    pipeline = get_model(model_name, numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_labels = (test_proba >= threshold).astype(int)

    submission = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        "TargetF1": test_labels,
        "TargetRAUC": test_proba,
    })

    return submission


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
        "features": "all_engineered",
        "cv_folds": n_folds,
        "mean_f1": f"{metrics['mean_f1']:.6f}",
        "mean_auc": f"{metrics['mean_roc_auc']:.6f}",
        "mean_combined_score": f"{metrics['mean_combined_score']:.6f}",
        "notes": notes,
    }

    file_exists = EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nExperiment logged to {EXPERIMENTS_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate climate risk model.")
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["logistic_regression", "lgbm", "xgboost"],
        help="Model to train (default: lgbm)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--search-threshold",
        action="store_true",
        help="Grid-search optimal threshold for combined score",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed decision threshold for submission (default: 0.5)",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Skip submission generation (CV only)",
    )
    args = parser.parse_args()

    # 1. Load raw data
    print("Loading raw data...")
    train_raw, test_raw, climate = load_raw_data(DATA_DIR)
    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}, Climate: {climate.shape}")

    # 2. Feature engineering
    print("\nBuilding features...")
    train_df, test_df = build_features(train_raw, test_raw, climate)
    print(f"  Train features: {train_df.shape}, Test features: {test_df.shape}")

    # 3. Cross-validation
    print(f"\nRunning {args.folds}-fold CV with model={args.model}...")
    oof_proba, metrics = run_cv(
        train_df,
        model_name=args.model,
        n_folds=args.folds,
        search_threshold=args.search_threshold,
    )

    # 4. Log experiment
    log_experiment(args.model, args.folds, metrics)

    # 5. Generate submission
    if not args.no_submit:
        print("\nTraining on full data and generating submission...")
        submission = train_and_predict(
            train_df, test_df,
            model_name=args.model,
            threshold=args.threshold,
        )

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
