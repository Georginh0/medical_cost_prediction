"""
run_pipeline.py
════════════════
Entry point — runs the training pipeline.

Usage:
    conda activate medical_cost
    python run_pipeline.py


"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

from pipelines.training_pipeline import training_pipeline


def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   Medical Cost Prediction — Training Pipeline    ║")
    print("╚══════════════════════════════════════════════════╝")

    training_pipeline()

    print()
    print("✅  Pipeline complete")
    print()
    print("  Artifacts  →  models/")
    print("  Reports    →  reports/")
    print("  Extracts   →  extracted_data/")
    print()
    print("  Next steps:")
    print("    python run_deployment.py        # deploy model")
    print("    python sample_predict.py        # test predictions")
    print("    uvicorn app.server:app --reload --port 8000  # web UI")
    print("    mlflow ui --port 5000           # view experiments")


if __name__ == "__main__":
    main()
