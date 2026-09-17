"""
feature_engineering.py — Feature engineering pipeline.

Converts the raw datasets and climate features into an enriched representation
optimized for predicting climate-sensitive health outcomes.
All transformations are applied consistently across train and test sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID_COL = "ID"
TARGET = "is_climate_sensitive"
DATE_COL = "deathdate"

# Columns to drop if raw or redundant
COLS_TO_DROP = ["deathdate", "latitude", "longitude"]

CATEGORICAL_FEATURES = ["zone", "gender", "location", "zone_gender"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_raw_data(
    data_dir: str | Path = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


# ---------------------------------------------------------------------------
# Feature engineering pipeline
# ---------------------------------------------------------------------------

def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    climate: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    climate_clean = climate.drop(columns=[DATE_COL], errors="ignore")
    trn = train.merge(climate_clean, on=ID_COL, how="left")
    tst = test.merge(climate_clean, on=ID_COL, how="left")

    n_train = len(train)
    full = pd.concat([trn, tst], ignore_index=True)

    # 1. Temporal Features
    dt = pd.to_datetime(full[DATE_COL], errors="coerce")
    full["day_of_year"] = dt.dt.dayofyear
    full["month"] = dt.dt.month
    full["year"] = dt.dt.year
    full["day"] = dt.dt.day
    full["day_of_week"] = dt.dt.dayofweek
    full["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    full["quarter"] = dt.dt.quarter

    # Cyclical encodings
    full["doy_sin"] = np.sin(2 * np.pi * full["day_of_year"] / 365.25)
    full["doy_cos"] = np.cos(2 * np.pi * full["day_of_year"] / 365.25)
    full["month_sin"] = np.sin(2 * np.pi * full["month"] / 12)
    full["month_cos"] = np.cos(2 * np.pi * full["month"] / 12)

    # 2. Age Features (Age is strongly correlated with climate sensitivity)
    full["age_log"] = np.log1p(full["age"])
    full["age_sq"] = full["age"] ** 2
    full["age_sqrt"] = np.sqrt(full["age"])

    full["is_infant"] = (full["age"] == 0).astype(int)
    full["is_under_1"] = (full["age"] <= 1).astype(int)
    full["is_under_5"] = (full["age"] <= 5).astype(int)
    full["is_under_15"] = (full["age"] <= 15).astype(int)
    full["is_working_age"] = ((full["age"] > 15) & (full["age"] < 60)).astype(int)
    full["is_senior"] = (full["age"] >= 60).astype(int)
    full["is_elderly"] = (full["age"] >= 75).astype(int)

    age_bins = [-1, 0, 1, 5, 12, 18, 30, 45, 60, 75, 120]
    full["age_bin"] = pd.cut(full["age"], bins=age_bins, labels=False)

    # 3. Temperature Interaction & Anomaly Features
    full["temp_range"] = full["max_temperature"] - full["min_temperature"]
    full["temp_dev_30d"] = full["avg_temperature"] - full["tavg_30d"]
    full["temp_dev_90d"] = full["avg_temperature"] - full["tavg_90d"]
    full["temp_dev_7d"] = full["avg_temperature"] - full["tavg_7d"]
    full["tmax_dev_30d"] = full["max_temperature"] - full["tmax_30d"]
    full["tmin_dev_30d"] = full["min_temperature"] - full["tmin_30d"]
    full["hot_days_ratio"] = full["hot_days_30d"] / 30.0
    full["temp_anomaly_7_30"] = full["tavg_7d"] - full["tavg_30d"]
    full["temp_anomaly_30_90"] = full["tavg_30d"] - full["tavg_90d"]

    # 4. Precipitation & Moisture Features
    full["is_rainy"] = (full["precipitation"] > 0).astype(int)
    full["rain_days_ratio_30d"] = full["rain_days_30d"] / 30.0
    full["rain_dev_30d"] = full["precipitation"] - (full["rain_sum_30d"] / 30.0)
    full["rain_ratio_7_30"] = full["rain_sum_7d"] / (full["rain_sum_30d"] + 1e-5)
    full["rain_ratio_30_90"] = full["rain_sum_30d"] / (full["rain_sum_90d"] + 1e-5)
    full["max_rain_prop"] = full["max_daily_rain_30d"] / (full["rain_sum_30d"] + 1e-5)

    # 5. Vegetation & Terrain Features
    full["ndvi_diff"] = full["ndvi_30d"] - full["ndvi_90d"]
    full["ndvi_ratio"] = full["ndvi_30d"] / (full["ndvi_90d"] + 1e-5)

    # 6. Domain Interaction Features
    full["age_x_temp"] = full["age"] * full["avg_temperature"]
    full["age_x_rain30"] = full["age"] * full["rain_sum_30d"]
    full["age_x_ndvi30"] = full["age"] * full["ndvi_30d"]
    full["temp_x_precip"] = full["avg_temperature"] * full["precipitation"]
    full["temp30_x_rain30"] = full["tavg_30d"] * full["rain_sum_30d"]
    full["temp30_x_ndvi30"] = full["tavg_30d"] * full["ndvi_30d"]

    # 7. Spatial / Categorical Interactions
    loc_counts = full["location"].value_counts()
    full["location_freq"] = full["location"].map(loc_counts)

    for col in ["age", "avg_temperature", "elevation", "ndvi_30d"]:
        if col in full.columns:
            loc_mean = full.groupby("location")[col].transform("mean")
            full[f"{col}_mean_by_loc"] = loc_mean
            full[f"{col}_diff_loc_mean"] = full[col] - loc_mean

    full["zone_gender"] = full["zone"].astype(str) + "_" + full["gender"].astype(str)

    # Drop raw date and unnecessary columns
    drop_cols = [c for c in COLS_TO_DROP if c in full.columns]
    full = full.drop(columns=drop_cols)

    train_df = full.iloc[:n_train].copy()
    test_df = full.iloc[n_train:].copy()

    # Drop target from test if present
    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    return train_df, test_df


def get_feature_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) lists.

    Excludes ID and target columns.
    """
    exclude = {ID_COL, TARGET}
    cols = [c for c in df.columns if c not in exclude]

    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    return numeric, categorical
