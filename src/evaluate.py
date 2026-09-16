"""
evaluate.py — Exact Zindi competition metric calculator.

Metric:
    Score = 0.6 × F1-Score + 0.4 × ROC-AUC

Where:
    - F1 is computed from the binary column `TargetF1` (threshold = 0.5 applied
      during prediction, NOT here — the column already contains 0/1).
    - ROC-AUC is computed from the continuous column `TargetRAUC`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

def zindi_score(
    y_true: np.ndarray,
    y_pred_label: np.ndarray,
    y_pred_proba: np.ndarray,
    f1_weight: float = 0.6,
    auc_weight: float = 0.4,
) -> dict[str, float]:
    """Compute the Zindi combined score.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth binary labels (0 or 1).
    y_pred_label : array-like of int
        Predicted binary labels for F1 (column ``TargetF1``).
    y_pred_proba : array-like of float
        Predicted probabilities for ROC-AUC (column ``TargetRAUC``).
    f1_weight : float
        Weight for the F1 component (default 0.6).
    auc_weight : float
        Weight for the ROC-AUC component (default 0.4).

    Returns
    -------
    dict with keys ``f1``, ``roc_auc``, ``combined_score``.
    """
    f1 = f1_score(y_true, y_pred_label)
    auc = roc_auc_score(y_true, y_pred_proba)
    combined = f1_weight * f1 + auc_weight * auc
    return {"f1": f1, "roc_auc": auc, "combined_score": combined}


def score_from_proba(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    f1_weight: float = 0.6,
    auc_weight: float = 0.4,
) -> dict[str, float]:
    """Convenience wrapper: derive TargetF1 from probabilities + threshold.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth binary labels (0 or 1).
    y_pred_proba : array-like of float
        Predicted probabilities.
    threshold : float
        Decision threshold to produce binary labels for F1.
    f1_weight : float
        Weight for the F1 component (default 0.6).
    auc_weight : float
        Weight for the ROC-AUC component (default 0.4).

    Returns
    -------
    dict with keys ``f1``, ``roc_auc``, ``combined_score``, ``threshold``.
    """
    y_pred_label = (np.asarray(y_pred_proba) >= threshold).astype(int)
    result = zindi_score(y_true, y_pred_label, y_pred_proba, f1_weight, auc_weight)
    result["threshold"] = threshold
    return result


def find_best_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    low: float = 0.20,
    high: float = 0.80,
    step: float = 0.01,
    f1_weight: float = 0.6,
    auc_weight: float = 0.4,
) -> dict[str, float]:
    """Grid-search the threshold that maximises the combined Zindi score.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth binary labels.
    y_pred_proba : array-like of float
        Predicted probabilities.
    low, high, step : float
        Range and step for the grid search.
    f1_weight, auc_weight : float
        Weights for the combined metric.

    Returns
    -------
    dict with keys ``f1``, ``roc_auc``, ``combined_score``, ``threshold``.
    """
    best: dict[str, float] = {"combined_score": -1.0}
    for t in np.arange(low, high + step, step):
        result = score_from_proba(y_true, y_pred_proba, t, f1_weight, auc_weight)
        if result["combined_score"] > best["combined_score"]:
            best = result
    return best


# ---------------------------------------------------------------------------
# CLI usage:  python src/evaluate.py ground_truth.csv submission.csv
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Zindi submission CSV.")
    parser.add_argument("ground_truth", type=Path, help="Path to ground-truth CSV with columns ID, is_climate_sensitive")
    parser.add_argument("submission", type=Path, help="Path to submission CSV with columns ID, TargetF1, TargetRAUC")
    args = parser.parse_args()

    gt = pd.read_csv(args.ground_truth)
    sub = pd.read_csv(args.submission)

    # Merge to ensure same ordering
    merged = gt.merge(sub, on="ID", how="inner")
    if len(merged) != len(gt):
        missing = len(gt) - len(merged)
        print(f"WARNING: {missing} IDs in ground truth not found in submission.", file=sys.stderr)

    result = zindi_score(
        y_true=merged["is_climate_sensitive"].values,
        y_pred_label=merged["TargetF1"].values,
        y_pred_proba=merged["TargetRAUC"].values,
    )

    print(f"F1-Score:       {result['f1']:.6f}")
    print(f"ROC-AUC:        {result['roc_auc']:.6f}")
    print(f"Combined Score: {result['combined_score']:.6f}")


if __name__ == "__main__":
    _cli()
