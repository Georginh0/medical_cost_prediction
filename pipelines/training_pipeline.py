"""
pipelines/training_pipeline.py
════════════════════════════════


Run via: python run_pipeline.py

"""

from zenml import pipeline

from steps.ingest_data_step import ingest_data_step
from steps.preprocess_step import preprocess_step
from steps.feature_engineering_step import feature_engineering_step
from steps.scale_reduce_step import scale_reduce_step
from steps.model_train_step import model_train_step
from steps.model_evaluate_step import model_evaluate_step


@pipeline(name="medical_cost_training_pipeline")
def training_pipeline():
    """
    End-to-end medical cost prediction training pipeline.

    Step flow:
      ingest_data  →  preprocess  →  feature_engineering
          →  scale_and_reduce  →  model_train  →  model_evaluate
    """
    # 1 ── Load & validate
    raw_df = ingest_data_step()

    # 2 ── Stratified split
    X_train, X_test, y_train, y_test = preprocess_step(raw_df)

    # 3 ── Feature engineering + log target
    X_tr_eng, X_te_eng, y_tr_log, y_te_log = feature_engineering_step(
        X_train, X_test, y_train, y_test
    )

    # 4 ── Scale + fit PCA
    X_tr_sc, X_te_sc = scale_reduce_step(X_tr_eng, X_te_eng)

    # 5 ── Compare models, tune, train
    best_model = model_train_step(X_tr_sc, X_te_sc, y_tr_log, y_te_log)

    # 6 ── Evaluate, plot, export
    metrics = model_evaluate_step(best_model, X_te_sc, y_te_log)
