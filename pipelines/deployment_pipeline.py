"""
pipelines/deployment_pipeline.py
══════════════════════════════════
ZenML deployment pipeline.
Re-runs the full training pipeline, then conditionally deploys
the model if it meets the R² quality threshold.

Run via: python run_deployment.py [--min-r2 0.80]


"""

import json
import logging
import os

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from zenml import pipeline, step
from zenml.logger import get_logger

from steps.ingest_data_step          import ingest_data_step
from steps.preprocess_step           import preprocess_step
from steps.feature_engineering_step  import feature_engineering_step
from steps.scale_reduce_step         import scale_reduce_step
from steps.model_train_step          import model_train_step
from steps.model_evaluate_step       import model_evaluate_step

logger = get_logger(__name__)

DEFAULT_MIN_R2 = 0.80


# ── Deployment decision ───────────────────────────────────────────────────────

@step(name="deployment_decision")
def deployment_decision_step(
    metrics: dict,
    min_r2: float = DEFAULT_MIN_R2,
) -> Annotated[bool, "should_deploy"]:
    """
    Return True if model test R² meets the minimum threshold.
    """
    r2 = metrics.get("R2", 0.0)
    should_deploy = r2 >= min_r2
    if should_deploy:
        logger.info(f"[deployment_decision] R²={r2:.4f} ≥ {min_r2} → DEPLOY ✅")
    else:
        logger.warning(f"[deployment_decision] R²={r2:.4f} < {min_r2} → SKIP ❌")
    return should_deploy


# ── Deploy step ───────────────────────────────────────────────────────────────

@step(name="deploy_model")
def deploy_model_step(should_deploy: bool) -> Annotated[str, "deploy_status"]:
    """
    Mock deployment step — writes a manifest to extracted_data/.

    In production replace with:
      - mlflow.register_model(...)
      - Docker build + push
      - Cloud endpoint update (SageMaker, Vertex AI, Azure ML)
    """
    if not should_deploy:
        logger.warning("[deploy_model] Deployment skipped — model below quality threshold")
        return "skipped"

    # Read meta
    try:
        with open("models/model_meta.json") as fh:
            meta = json.load(fh)
    except FileNotFoundError:
        meta = {}

    os.makedirs("extracted_data", exist_ok=True)
    manifest = {
        "status":      "deployed",
        "model":       meta.get("best_model", "unknown"),
        "test_r2":     meta.get("metrics", {}).get("R2", "?"),
        "artifacts":   [
            "models/best_model.pkl",
            "models/scaler.pkl",
            "models/pca.pkl",
            "models/feature_names.pkl",
            "models/model_meta.json",
        ],
        "serve_cmd":   "uvicorn app.server:app --reload --port 8000",
        "mlflow_ui":   "mlflow ui --port 5000",
    }
    with open("extracted_data/deployment_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(f"[deploy_model] ✅  {manifest['model']}  R²={manifest['test_r2']}")
    logger.info("  To serve: uvicorn app.server:app --reload --port 8000")
    return "deployed"


# ── Pipeline ──────────────────────────────────────────────────────────────────

@pipeline(name="medical_cost_deployment_pipeline")
def deployment_pipeline(min_r2: float = DEFAULT_MIN_R2):
    """Train → evaluate → conditionally deploy."""
    raw_df                               = ingest_data_step()
    X_train, X_test, y_train, y_test     = preprocess_step(raw_df)
    X_tr_e, X_te_e, y_tr_l, y_te_l      = feature_engineering_step(
                                               X_train, X_test, y_train, y_test)
    X_tr_s, X_te_s                       = scale_reduce_step(X_tr_e, X_te_e)
    best_model                           = model_train_step(
                                               X_tr_s, X_te_s, y_tr_l, y_te_l)
    metrics                              = model_evaluate_step(best_model, X_te_s, y_te_l)
    should_deploy                        = deployment_decision_step(metrics, min_r2)
    status                               = deploy_model_step(should_deploy)
