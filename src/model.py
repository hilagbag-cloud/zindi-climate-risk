"""
model.py — Model configuration and ensemble factory.

Provides constructors and wrappers for CatBoost, LightGBM, XGBoost, and
Ensemble pipelines used in the Zindi Climate Risk challenge.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Preprocessor builder
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


# ---------------------------------------------------------------------------
# Baseline Logistic Regression Pipeline
# ---------------------------------------------------------------------------

def get_baseline_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """Return LogisticRegression pipeline."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# LightGBM Classifier Pipeline
# ---------------------------------------------------------------------------

def get_lgbm_classifier(random_state: int = 42):
    """Return tuned LightGBM classifier."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.015,
        max_depth=5,
        num_leaves=16,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=20,
        random_state=random_state,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# CatBoost Classifier Pipeline
# ---------------------------------------------------------------------------

def get_catboost_classifier(
    cat_features: List[str] = None,
    random_state: int = 42,
    train_dir: str = "/tmp/catboost_info",
):
    """Return tuned CatBoost classifier."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=700,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=3.0,
        cat_features=cat_features,
        random_seed=random_state,
        train_dir=train_dir,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# XGBoost Classifier Pipeline
# ---------------------------------------------------------------------------

def get_xgb_classifier(random_state: int = 42):
    """Return tuned XGBoost classifier."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.015,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Any] = {
    "logistic_regression": get_baseline_pipeline,
    "lgbm": get_lgbm_classifier,
    "catboost": get_catboost_classifier,
    "xgboost": get_xgb_classifier,
}


def get_model(
    name: str,
    numeric_features: List[str] = None,
    categorical_features: List[str] = None,
    **kwargs,
) -> Any:
    """Get model constructor by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    if name == "logistic_regression":
        return MODEL_REGISTRY[name](numeric_features, categorical_features)
    elif name == "catboost":
        return MODEL_REGISTRY[name](cat_features=categorical_features, **kwargs)
    else:
        return MODEL_REGISTRY[name](**kwargs)
