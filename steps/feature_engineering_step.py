"""
steps/feature_engineering_step.py
═══════════════════════════════════
ZenML Step 3 of 6 — Feature Engineering

Creates domain-specific features identified in EDA:

  INTERACTION TERMS (most impactful):
    smoker_bmi   = smoker × bmi        captures obese-smoker cluster
    smoker_age   = smoker × age        older smokers cost much more

  BINARY FLAGS:
    bmi_obese    = 1 if bmi ≥ 30       clinical obesity threshold
    high_risk    = smoker AND obese    highest-cost segment

  RATIO FEATURES:
    bmi_age      = bmi × age / 100     ageing amplifies BMI effect
    family_size  = children + 1

  ORDINAL BINS:
    age_group    = cut(age, 4 groups)  young/middle/senior/elderly

  ENCODING:
    One-hot encode: sex, smoker, region, age_group  (drop_first=True)

  TARGET:
    log1p(charges) — reduces skew from 1.51 to ~0.05

No sklearn fitting occurs here — safe to apply identically to
both train and test without data leakage.

Template: Georginh0/house_price_prediction
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os

try:
    from typing import Annotated, Tuple
except ImportError:
    from typing_extensions import Annotated
    from typing import Tuple

from zenml import step

logger = logging.getLogger(__name__)
MODELS_PATH = "models"


# ── Pure transformation (no fitting) ────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transforms.
    Called identically on train and test — no fit/transform pattern needed.
    """
    df = df.copy()

    # Numeric proxy for smoker (needed for arithmetic only — dropped after)
    df["smoker_num"] = (df["smoker"] == "yes").astype(int)

    # ── Interaction terms ──────────────────────────────────────────────────
    df["smoker_bmi"] = df["smoker_num"] * df["bmi"]  
    df["smoker_age"] = df["smoker_num"] * df["age"]  

    # ── Binary flags ──────────────────────────────────────────────────────
    df["bmi_obese"] = (df["bmi"] >= 30).astype(int)
    df["high_risk"] = ((df["smoker_num"] == 1) & (df["bmi"] >= 30)).astype(int)

    # ── Ratio / composite ─────────────────────────────────────────────────
    df["bmi_age"] = (df["bmi"] * df["age"]) / 100
    df["family_size"] = df["children"] + 1

    # ── Ordinal bins ──────────────────────────────────────────────────────
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 45, 60, 100],
        labels=["young", "middle", "senior", "elderly"],
    ).astype(str)

    # ── One-hot encode ────────────────────────────────────────────────────
    df = pd.get_dummies(
        df,
        columns=["sex", "smoker", "region", "age_group"],
        drop_first=True,
    )

    # Drop arithmetic proxy
    df.drop(columns=["smoker_num"], errors="ignore", inplace=True)

    return df


# ── ZenML step ───────────────────────────────────────────────────────────────


@step(name="feature_engineering")
def feature_engineering_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train_eng"],
    Annotated[pd.DataFrame, "X_test_eng"],
    Annotated[pd.Series, "y_train_log"],
    Annotated[pd.Series, "y_test_log"],
]:
    """
    Apply feature engineering to both splits and log-transform target.

    Returns
    -------
    X_train_eng, X_test_eng : Engineered feature DataFrames
    y_train_log, y_test_log : log1p(charges) — normalised target
    """
    os.makedirs(MODELS_PATH, exist_ok=True)

    X_train_eng = build_features(X_train)
    X_test_eng = build_features(X_test)

    # Align test columns to training schema
    # (handles rare dummy categories absent from test split)
    X_test_eng = X_test_eng.reindex(columns=X_train_eng.columns, fill_value=0)

    # Persist column order — required for inference alignment
    joblib.dump(
        list(X_train_eng.columns),
        os.path.join(MODELS_PATH, "feature_names.pkl"),
    )

    # Log1p-transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    logger.info(
        f"[feature_engineering] {X_train_eng.shape[1]} features  |  "
        f"Target log range: [{y_train_log.min():.2f}, {y_train_log.max():.2f}]"
    )
    return X_train_eng, X_test_eng, y_train_log, y_test_log
