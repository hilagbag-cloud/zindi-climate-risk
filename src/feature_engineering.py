"""
feature_engineering.py — Feature engineering pipeline.

Converts the starter notebook logic into a reusable, production-quality
feature engineering module.  All transformations are applied identically
to train and test sets via a single ``build_features()`` entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID_COL = "ID"
TARGET = "is_climate_sensitive"
DATE_COL = "deathdate"

# Columns explicitly excluded in the starter notebook
COLS_TO_DROP = ["location", "latitude", "longitude"]

CATEGORICAL_FEATURES = ["zone", "gender"]

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_raw_data(
    data_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Train, Test and climate_features from ``data_dir``.

    Returns
    -------
    train, test, climate_features : tuple of DataFrames
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "Train.csv")
    test = pd.read_csv(data_dir / "Test.csv")
    climate = pd.read_csv(data_dir / "climate_features.csv")
    return train, test, climate


def merge_climate(
    df: pd.DataFrame,
    climate: pd.DataFrame,
) -> pd.DataFrame:
    """Merge climate features onto *df* by ID.

    Drops the duplicate ``deathdate`` column coming from the climate CSV.
    """
    climate_clean = climate.drop(columns=[DATE_COL], errors="ignore")
    return df.merge(climate_clean, on=ID_COL, how="left")


# ---------------------------------------------------------------------------
# Feature engineering (from starter notebook + new ideas)
# ---------------------------------------------------------------------------

def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from deathdate.

    Features added:
        - ``day_of_year``
        - ``month``
        - ``year``
        - ``day_of_year_sin``, ``day_of_year_cos`` (cyclical encoding)
    """
    dt = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["day_of_year"] = dt.dt.dayofyear
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year

    # Cyclical encoding
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


def _add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temperature-based interaction features."""
    df["temperature_range"] = df["max_temperature"] - df["min_temperature"]

    # Deviation of daily avg from 30-day avg
    if "tavg_30d" in df.columns:
        df["temp_deviation_30d"] = df["avg_temperature"] - df["tavg_30d"]

    # Deviation of daily avg from 90-day avg
    if "tavg_90d" in df.columns:
        df["temp_deviation_90d"] = df["avg_temperature"] - df["tavg_90d"]

    # Ratio of temperature range to mean range over 30d
    if "temp_range_mean_30d" in df.columns:
        df["temp_range_ratio_30d"] = df["temperature_range"] / (df["temp_range_mean_30d"] + 1e-6)

    return df


def _add_precipitation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive precipitation-based interaction features."""
    df["is_rainy_day_current"] = (df["precipitation"] > 0).astype(int)

    if "rain_sum_30d" in df.columns and "rain_sum_90d" in df.columns:
        df["rain_ratio_30d_90d"] = df["rain_sum_30d"] / (df["rain_sum_90d"] + 1e-6)

    if "rain_sum_7d" in df.columns and "rain_sum_30d" in df.columns:
        df["rain_ratio_7d_30d"] = df["rain_sum_7d"] / (df["rain_sum_30d"] + 1e-6)

    if "max_daily_rain_30d" in df.columns and "rain_sum_30d" in df.columns:
        df["max_rain_proportion_30d"] = df["max_daily_rain_30d"] / (df["rain_sum_30d"] + 1e-6)

    return df


def _add_vegetation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive vegetation-based features."""
    if "ndvi_30d" in df.columns and "ndvi_90d" in df.columns:
        df["ndvi_change_30d_90d"] = df["ndvi_30d"] - df["ndvi_90d"]

    return df


def _add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Categorise age into risk groups."""
    bins = [-1, 1, 5, 15, 45, 65, 200]
    labels = ["infant", "young_child", "child", "adult", "senior", "elderly"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    df["age_group"] = df["age_group"].astype(str)

    # Log-transform age (common for skewed distributions)
    df["age_log"] = np.log1p(df["age"])

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cross-domain interaction features."""
    # Temperature × Precipitation interactions
    df["temp_x_precip"] = df["avg_temperature"] * df["precipitation"]

    if "tavg_30d" in df.columns and "rain_sum_30d" in df.columns:
        df["tavg30d_x_rain30d"] = df["tavg_30d"] * df["rain_sum_30d"]

    # Age × Temperature interaction
    df["age_x_temp"] = df["age"] * df["avg_temperature"]

    return df


def _drop_and_clean(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Drop raw date and unnecessary columns; ensure correct dtypes."""
    cols_to_drop = [DATE_COL] + COLS_TO_DROP
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    climate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end feature engineering pipeline.

    Parameters
    ----------
    train : pd.DataFrame
        Raw train data (as loaded from ``Train.csv``).
    test : pd.DataFrame
        Raw test data (as loaded from ``Test.csv``).
    climate : pd.DataFrame
        Climate features (as loaded from ``climate_features.csv``).

    Returns
    -------
    train_df, test_df : tuple of DataFrames
        Feature-engineered DataFrames ready for modelling.
    """
    # 1. Merge climate features
    train_df = merge_climate(train, climate)
    test_df = merge_climate(test, climate)

    # 2. Apply feature engineering
    for transform_fn in [
        _add_date_features,
        _add_temperature_features,
        _add_precipitation_features,
        _add_vegetation_features,
        _add_age_features,
        _add_interaction_features,
    ]:
        train_df = transform_fn(train_df)
        test_df = transform_fn(test_df)

    # 3. Cleanup
    train_df = _drop_and_clean(train_df, is_train=True)
    test_df = _drop_and_clean(test_df, is_train=False)

    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) lists.

    Excludes ID and target columns.
    """
    exclude = {ID_COL, TARGET}
    cols = [c for c in df.columns if c not in exclude]

    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    return numeric, categorical

