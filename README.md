# 🏥 Medical Cost Prediction

> End-to-end ML pipeline predicting annual insurance charges from patient profiles.

[![Python](https://img.shields.io/badge/Python-3.8-blue)](https://python.org)
[![ZenML](https://img.shields.io/badge/ZenML-0.55.4-purple)](https://zenml.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.8.1-orange)](https://mlflow.org)
[![R²](https://img.shields.io/badge/R²-0.8633-brightgreen)]()

## Results

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| **Lasso (α=0.0001)** | **0.8633** | **$4,440** | **$2,359** |

## Quick Start

```bash
conda create -n medical_cost python=3.8 -y && conda activate medical_cost
pip install -r requirements.txt && pip install -e .
zenml init
python run_pipeline.py
```

## Usage

```bash
python run_pipeline.py                              # train
python run_deployment.py                            # deploy
python sample_predict.py                            # test predictions
uvicorn app.server:app --port 8000 --no-reload &    # web UI → localhost:8000
mlflow ui --port 5000 &                             # experiments → localhost:5000
```

## Structure

```
├── analyze_src/eda.ipynb        ← EDA notebook
├── steps/                       ← 6 ZenML steps (one per file)
├── pipelines/                   ← training + deployment pipelines
├── app/server.py + index.html   ← FastAPI + web UI
├── models/                      ← artifacts after training
├── extracted_data/              ← predictions, metrics CSVs
├── reports/                     ← evaluation plots
├── run_pipeline.py              ← entry: train
├── run_deployment.py            ← entry: deploy
└── sample_predict.py            ← entry: predict
```

## Pipeline

```
ingest_data → preprocess → feature_engineering → scale_reduce → model_train → model_evaluate
```

## Key Insight

Smoker status (r=0.79 with charges) dominates all other features. Three charge clusters exist:
non-smokers (low), smokers with lean BMI (medium), smokers with BMI ≥ 30 (very high).
The engineered feature `smoker_bmi = smoker × bmi` captures this directly.

## Dataset

1,338 rows · 7 columns · 0 missing values  
Features: `age`, `sex`, `bmi`, `children`, `smoker`, `region` → target: `charges` (USD)
