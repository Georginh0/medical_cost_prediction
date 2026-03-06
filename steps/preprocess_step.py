"""
steps/preprocess_step.py
═════════════════════════
ZenML Step 2 of 6 — Data Preprocessing

Performs a stratified train/test split on the 'smoker' column.
Stratification is critical because smoker status drives ~79% of
the variance in charges — an imbalanced split would skew both
training and evaluation.


"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from typing import Annotated, Tuple
except ImportError:
    from typing_extensions import Annotated
    from typing import Tuple

from zenml import step

logger = logging.getLogger(__name__)


@step(name="preprocess")
def preprocess_step(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Split dataset into stratified train / test sets.

    Stratified on 'smoker' to preserve the ~20% smoker ratio
    in both splits, ensuring neither split is biased toward
    low-cost non-smoker or high-cost smoker profiles.

    Parameters
    ----------
    df           : Clean DataFrame from ingest_data_step
    test_size    : Fraction of data for test set (default 0.20)
    random_state : Reproducibility seed (default 42)

    Returns
    -------
    X_train, X_test : Feature DataFrames
    y_train, y_test : Target Series (raw charges in USD)
    """
    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=df["smoker"],  # ← key: preserve smoker ratio
    )

    # Reset indices for downstream concat/align safety
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Verify stratification held
    train_smoke_rate = (X_train["smoker"] == "yes").mean()
    test_smoke_rate = (X_test["smoker"] == "yes").mean()

    logger.info(
        f"[preprocess] Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows"
    )
    logger.info(
        f"[preprocess] Smoker rate — Train: {train_smoke_rate:.2%}  |  Test: {test_smoke_rate:.2%}"
    )

    return X_train, X_test, y_train, y_test
