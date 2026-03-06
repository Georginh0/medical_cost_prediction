"""
steps/scale_reduce_step.py
═══════════════════════════
ZenML Step 4 of 6 — Scaling & Dimensionality Reduction

1. StandardScaler — zero mean, unit variance.
   Fit on X_train only → transform both splits (no leakage).

2. PCA — retains 95 % of explained variance.
   Fit on scaled X_train → transform both splits.
   The PCA artifact is saved for optional inspection,
   but tree-based models use the scaled (pre-PCA) arrays directly
   since they do not benefit from PCA rotation.

Artifacts saved to models/:
  scaler.pkl  — fitted StandardScaler
  pca.pkl     — fitted PCA

"""

import logging
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from typing import Annotated, Tuple
except ImportError:
    from typing_extensions import Annotated
    from typing import Tuple

from zenml import step

logger = logging.getLogger(__name__)
MODELS_PATH = "models"


@step(name="scale_and_reduce")
def scale_reduce_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    pca_variance: float = 0.95,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train_scaled"],
    Annotated[pd.DataFrame, "X_test_scaled"],
]:
    """
    Scale features with StandardScaler, then fit PCA as a reference artifact.
    Returns scaled arrays (before PCA) — tree models perform best without PCA.

    Parameters
    ----------
    X_train      : Engineered training features
    X_test       : Engineered test features
    pca_variance : Fraction of variance PCA should retain (default 0.95)

    Returns
    -------
    X_train_scaled, X_test_scaled : StandardScaler-transformed DataFrames
    """
    os.makedirs(MODELS_PATH, exist_ok=True)

    # ── StandardScaler ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
    )
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))
    logger.info(
        f"[scale_reduce] StandardScaler fitted — {X_train_sc.shape[1]} features"
    )

    # ── PCA (reference artifact) ──────────────────────────────────────────
    pca = PCA(n_components=pca_variance, random_state=42)
    pca.fit(X_train_sc)
    joblib.dump(pca, os.path.join(MODELS_PATH, "pca.pkl"))
    logger.info(
        f"[scale_reduce] PCA fitted — {pca.n_components_} components "
        f"explain {pca_variance * 100:.0f}% variance"
    )

    return X_train_sc, X_test_sc
