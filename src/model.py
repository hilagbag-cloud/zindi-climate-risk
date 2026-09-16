"""
model.py — Model configuration and factory.

Provides model constructors for baseline and advanced models used in the
Climate Risk challenge.  Each function returns a scikit-learn–compatible
estimator (or Pipeline) ready for ``.fit()`` / ``.predict_proba()``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Preprocessor builder
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer for numeric + categorical features.

    Numeric pipeline:
        - Median imputation
        - StandardScaler

    Categorical pipeline:
        - Most-frequent imputation
        - OneHotEncoder (ignore unknown)
    """
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
# Baseline model (from starter notebook)
# ---------------------------------------------------------------------------

def get_baseline_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Return a LogisticRegression pipeline matching the starter notebook.

    Configuration:
        - max_iter=2000
        - class_weight='balanced'
    """
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
# LightGBM model (upgrade)
# ---------------------------------------------------------------------------

def get_lgbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Return a LightGBM pipeline for improved performance.

    Uses the scikit-learn API wrapper from lightgbm.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("lightgbm is required. Install it with: pip install lightgbm")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# XGBoost model (upgrade)
# ---------------------------------------------------------------------------

def get_xgb_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Return an XGBoost pipeline for improved performance."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost is required. Install it with: pip install xgboost")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Compute scale_pos_weight externally if needed; default to 1 here
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, callable] = {
    "logistic_regression": get_baseline_pipeline,
    "lgbm": get_lgbm_pipeline,
    "xgboost": get_xgb_pipeline,
}


def get_model(
    name: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Get a model pipeline by name.

    Parameters
    ----------
    name : str
        One of ``'logistic_regression'``, ``'lgbm'``, ``'xgboost'``.
    numeric_features : list[str]
        List of numeric column names.
    categorical_features : list[str]
        List of categorical column names.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](numeric_features, categorical_features)
