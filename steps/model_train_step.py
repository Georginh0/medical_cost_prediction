"""
steps/model_train_step.py
══════════════════════════
ZenML Step 5 of 6 — Model Training

1. 5-Fold CV comparison of 6 regression models (log-scale target).
2. GridSearchCV hyperparameter tuning on the best model.
3. All experiments logged to MLflow (experiment: medical_cost_prediction).
4. Saves best_model.pkl and model_meta.json to models/.
5. Saves CV comparison table to extracted_data/.

Models compared:
  LinearRegression, Ridge, Lasso,
  RandomForest, GradientBoosting, KNN


"""

import json
import logging
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from zenml import step

logger = logging.getLogger(__name__)
MODELS_PATH = "models"
EXTRACT_DIR = "extracted_data"

MLFLOW_EXPERIMENT = "medical_cost_prediction"


# ── Candidate models ─────────────────────────────────────────────────────────


def _get_candidates() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10_000),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }


# ── Hyperparameter grids ─────────────────────────────────────────────────────

PARAM_GRIDS = {
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.15],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    },
    "Ridge": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1]},
    "KNN": {"n_neighbors": [3, 5, 7, 10], "weights": ["uniform", "distance"]},
    "LinearRegression": {},
}


# ── ZenML step ───────────────────────────────────────────────────────────────


@step(name="model_train")
def model_train_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Annotated[RegressorMixin, "best_model"]:
    """
    Select, tune, and train the best regression model.

    Workflow
    --------
    1. Run 5-Fold CV on 6 candidates
    2. Select winner by mean CV R²
    3. GridSearchCV on winner
    4. Final test-set evaluation (dollar scale via expm1)
    5. Log everything to MLflow
    6. Save artifacts

    Returns
    -------
    best_model : Fitted sklearn estimator (trained on full X_train)
    """
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    candidates = _get_candidates()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Step A: 5-Fold CV ────────────────────────────────────────────────
    cv_records = {}
    logger.info("[model_train] ── 5-Fold Cross-Validation ──")

    for name, model in candidates.items():
        r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
        rmse_scores = np.sqrt(
            -cross_val_score(
                model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"
            )
        )
        mae_scores = -cross_val_score(
            model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
        )
        cv_records[name] = {
            "cv_r2_mean": round(float(r2_scores.mean()), 4),
            "cv_r2_std": round(float(r2_scores.std()), 4),
            "cv_rmse_mean": round(float(rmse_scores.mean()), 4),
            "cv_mae_mean": round(float(mae_scores.mean()), 4),
        }
        logger.info(
            f"  {name:<22}  R²={r2_scores.mean():.4f}±{r2_scores.std():.4f}"
            f"  RMSE={rmse_scores.mean():.4f}"
        )

        with mlflow.start_run(run_name=f"cv_{name}", nested=True):
            mlflow.log_param("model", name)
            mlflow.log_metric("cv_r2_mean", r2_scores.mean())
            mlflow.log_metric("cv_r2_std", r2_scores.std())
            mlflow.log_metric("cv_rmse_mean", rmse_scores.mean())

    # Save CV comparison
    cv_df = pd.DataFrame(cv_records).T.sort_values("cv_r2_mean", ascending=False)
    cv_df.to_csv(os.path.join(EXTRACT_DIR, "model_comparison.csv"))
    logger.info(f"\n{cv_df.to_string()}\n")

    # ── Step B: Select best model ────────────────────────────────────────
    best_name = cv_df.index[0]
    best_model = candidates[best_name]
    logger.info(
        f"[model_train] Best model: {best_name}  (CV R²={cv_records[best_name]['cv_r2_mean']:.4f})"
    )

    # ── Step C: GridSearchCV ─────────────────────────────────────────────
    grid_params = PARAM_GRIDS.get(best_name, {})
    best_params = {}

    if grid_params:
        logger.info(f"[model_train] ── GridSearchCV: {best_name} ──")
        gs = GridSearchCV(
            best_model,
            grid_params,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        gs.fit(X_train, y_train)
        tuned_model = gs.best_estimator_
        best_params = gs.best_params_
        logger.info(f"  Best params : {best_params}")
        logger.info(f"  Best CV R²  : {gs.best_score_:.4f}")
    else:
        # No grid to search (e.g. LinearRegression)
        tuned_model = best_model
        tuned_model.fit(X_train, y_train)

    # ── Step D: Final test evaluation (dollar scale) ──────────────────────
    tuned_model.fit(X_train, y_train)  # refit on full train set
    y_pred_log = tuned_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # back to USD
    y_true = np.expm1(y_test)  # back to USD

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    logger.info("=" * 50)
    logger.info("  TEST SET RESULTS  (USD scale)")
    logger.info(f"  RMSE  : ${rmse:>10,.2f}")
    logger.info(f"  MAE   : ${mae:>10,.2f}")
    logger.info(f"  MAPE  : {mape:>9.2f}%")
    logger.info(f"  R²    : {r2:>10.4f}")
    logger.info("=" * 50)

    # ── Step E: MLflow final run ──────────────────────────────────────────
    with mlflow.start_run(run_name=f"{best_name}_tuned_final"):
        mlflow.log_param("model", best_name)
        mlflow.log_param("cv_r2", cv_records[best_name]["cv_r2_mean"])
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_mape", mape)
        mlflow.log_metric("test_r2", r2)
        mlflow.sklearn.log_model(tuned_model, artifact_path="model")

    # ── Step F: Persist artifacts ─────────────────────────────────────────
    joblib.dump(tuned_model, os.path.join(MODELS_PATH, "best_model.pkl"))

    meta = {
        "best_model": best_name,
        "best_params": best_params,
        "cv_r2": cv_records[best_name]["cv_r2_mean"],
        "metrics": {
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
            "R2": round(r2, 4),
        },
    }
    with open(os.path.join(MODELS_PATH, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(f"[model_train] Saved → {MODELS_PATH}/best_model.pkl")
    return tuned_model
