"""
steps/ingest_data_step.py
══════════════════════════
ZenML Step 1 of 6 — Data Ingestion

Loads insurance.csv, validates the schema, checks value ranges,
and removes the single duplicate row identified in EDA.

Template: Georginh0/house_price_prediction
"""

import logging
import pandas as pd

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from zenml import step

logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/insurance.csv"

EXPECTED_COLS = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
VALID_SEX = {"male", "female"}
VALID_SMOKER = {"yes", "no"}
VALID_REGIONS = {"northeast", "northwest", "southeast", "southwest"}


@step(name="ingest_data")
def ingest_data_step() -> Annotated[pd.DataFrame, "raw_df"]:
    """
    Load and validate raw insurance dataset.

    Checks performed:
    - File exists and is readable
    - All expected columns present
    - No missing values
    - Categorical values within valid set
    - Numeric ranges sensible (age 18-64, bmi 10-70, charges > 0)
    - Duplicate rows removed

    Returns
    -------
    raw_df : pd.DataFrame
        Clean DataFrame, 1337 rows × 7 columns.
    """
    # ── Load ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{RAW_PATH}'.\n"
            "Place insurance.csv in data/raw/ before running the pipeline."
        )

    logger.info(f"[ingest_data] Loaded raw data: {df.shape}")

    # ── Schema check ─────────────────────────────────────────────────
    missing_cols = EXPECTED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # ── Missing values ────────────────────────────────────────────────
    nulls = df.isnull().sum()
    if nulls.any():
        raise ValueError(f"Dataset contains missing values:\n{nulls[nulls > 0]}")

    # ── Categorical validity ──────────────────────────────────────────
    invalid_sex = set(df["sex"].unique()) - VALID_SEX
    invalid_smoker = set(df["smoker"].unique()) - VALID_SMOKER
    invalid_region = set(df["region"].unique()) - VALID_REGIONS
    for label, invalid in [
        ("sex", invalid_sex),
        ("smoker", invalid_smoker),
        ("region", invalid_region),
    ]:
        if invalid:
            raise ValueError(f"Unexpected values in '{label}': {invalid}")

    # ── Numeric ranges ────────────────────────────────────────────────
    assert df["age"].between(18, 64).all(), "age values outside [18, 64]"
    assert df["bmi"].between(10, 70).all(), "bmi values outside [10, 70]"
    assert (df["charges"] > 0).all(), "charges contains non-positive values"

    # ── Remove duplicates ─────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"[ingest_data] Removed {n_dropped} duplicate row(s)")

    logger.info(f"[ingest_data] Clean data ready: {df.shape}")
    return df
