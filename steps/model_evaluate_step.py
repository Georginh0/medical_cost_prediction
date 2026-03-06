"""
steps/model_evaluate_step.py
═════════════════════════════
ZenML Step 6 of 6 — Model Evaluation

Generates final diagnostic artifacts:
  reports/06_actual_vs_predicted.png  — scatter + residual plots
  reports/07_feature_importance.png   — top-15 feature importances
  extracted_data/test_predictions.csv — row-level predictions
  extracted_data/final_metrics.json   — summary metrics

All metrics are reported in original USD scale (expm1 inverse).

Template: Georginh0/house_price_prediction
"""

import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from zenml import step

logger = logging.getLogger(__name__)
REPORTS_DIR = "reports"
EXTRACT_DIR = "extracted_data"


@step(name="model_evaluate")
def model_evaluate_step(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Annotated[dict, "final_metrics"]:
    """
    Evaluate the trained model on the held-out test set.

    Parameters
    ----------
    model  : Fitted sklearn estimator from model_train_step
    X_test : Scaled test features
    y_test : Log-transformed test targets

    Returns
    -------
    final_metrics : dict — RMSE, MAE, MAPE, R2 in USD scale
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # ── Predictions (USD scale) ──────────────────────────────────────────
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    residuals = y_true - y_pred

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs(residuals / y_true)) * 100)

    # ── Plot 1: Actual vs Predicted + Residuals ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Evaluation — Test Set", fontsize=13, fontweight="bold")

    # Scatter
    axes[0].scatter(
        y_true, y_pred, alpha=0.40, color="#1a6b4a", s=18, edgecolors="none"
    )
    lim = max(float(y_true.max()), float(y_pred.max())) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual Charges (USD)")
    axes[0].set_ylabel("Predicted Charges (USD)")
    axes[0].set_title(f"Actual vs Predicted  (R²={r2:.3f})")
    axes[0].legend()

    # Residuals
    axes[1].scatter(
        y_pred, residuals, alpha=0.40, color="#e8b84b", s=18, edgecolors="none"
    )
    axes[1].axhline(0, color="red", lw=2, linestyle="--")
    axes[1].set_xlabel("Predicted Charges (USD)")
    axes[1].set_ylabel("Residuals (USD)")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    plt.savefig(
        os.path.join(REPORTS_DIR, "06_actual_vs_predicted.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # ── Plot 2: Feature Importance ───────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=X_test.columns)
        top15 = fi.sort_values(ascending=True).tail(15)

        plt.figure(figsize=(9, 6))
        top15.plot(kind="barh", color="#1a6b4a", alpha=0.85, edgecolor="white")
        plt.title("Top 15 Feature Importances", fontweight="bold")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(
            os.path.join(REPORTS_DIR, "07_feature_importance.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(
            f"[model_evaluate] Top 5 features: "
            f"{list(fi.sort_values(ascending=False).index[:5])}"
        )

    # ── Export predictions ───────────────────────────────────────────────
    pred_df = pd.DataFrame(
        {
            "actual_charges": y_true.values,
            "predicted_charges": y_pred,
            "residual": residuals.values,
            "abs_pct_error": np.abs(residuals.values / y_true.values) * 100,
        }
    )
    pred_df.to_csv(os.path.join(EXTRACT_DIR, "test_predictions.csv"), index=False)

    # ── Summary metrics ──────────────────────────────────────────────────
    final_metrics = {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
    }
    with open(os.path.join(EXTRACT_DIR, "final_metrics.json"), "w") as fh:
        json.dump(final_metrics, fh, indent=2)

    logger.info("[model_evaluate] ── Final Test Metrics ──")
    logger.info(f"  RMSE  : ${rmse:>10,.2f}")
    logger.info(f"  MAE   : ${mae:>10,.2f}")
    logger.info(f"  MAPE  : {mape:>9.2f}%")
    logger.info(f"  R²    : {r2:>10.4f}")

    return final_metrics
