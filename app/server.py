"""
app/server.py
══════════════
FastAPI prediction server.

Endpoints:
  GET  /         → web UI (app/index.html)
  GET  /health   → model status + metrics
  POST /predict  → JSON prediction

Run:
    uvicorn app.server:app --reload --port 8000

"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Cost Prediction API",
    description="Predict annual insurance charges from patient profile.",
    version="1.0.0",
)

MODELS_PATH = "models"

# ── Lazy-load artifacts ───────────────────────────────────────────────────────
_MODEL = None
_SCALER = None
_FEATURE_NAMES = None
_META = {}


def _load():
    global _MODEL, _SCALER, _FEATURE_NAMES, _META
    if _MODEL is not None:
        return
    try:
        _MODEL = joblib.load(os.path.join(MODELS_PATH, "best_model.pkl"))
        _SCALER = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
        _FEATURE_NAMES = joblib.load(os.path.join(MODELS_PATH, "feature_names.pkl"))
        with open(os.path.join(MODELS_PATH, "model_meta.json")) as fh:
            _META = json.load(fh)
        logger.info(f"Model loaded: {_META.get('best_model')}")
    except Exception as exc:
        logger.warning(f"Model not loaded — running in fallback mode. ({exc})")


@app.on_event("startup")
def startup():
    _load()


# ── Schemas ───────────────────────────────────────────────────────────────────


class PatientInput(BaseModel):
    age: int = Field(..., ge=18, le=64)
    sex: str = Field(...)
    bmi: float = Field(..., ge=10.0, le=60.0)
    children: int = Field(..., ge=0, le=5)
    smoker: str = Field(...)
    region: str = Field(...)

    @validator("sex")
    def sex_valid(cls, v):
        if v not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        return v

    @validator("smoker")
    def smoker_valid(cls, v):
        if v not in ("yes", "no"):
            raise ValueError("smoker must be 'yes' or 'no'")
        return v

    @validator("region")
    def region_valid(cls, v):
        valid = {"northeast", "northwest", "southeast", "southwest"}
        if v not in valid:
            raise ValueError(f"region must be one of {sorted(valid)}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "sex": "male",
                "bmi": 28.5,
                "children": 1,
                "smoker": "no",
                "region": "northwest",
            }
        }


# ── Feature engineering ───────────────────────────────────────────────────────


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["smoker_num"] = (df["smoker"] == "yes").astype(int)
    df["smoker_bmi"] = df["smoker_num"] * df["bmi"]
    df["smoker_age"] = df["smoker_num"] * df["age"]
    df["bmi_obese"] = (df["bmi"] >= 30).astype(int)
    df["high_risk"] = ((df["smoker_num"] == 1) & (df["bmi"] >= 30)).astype(int)
    df["bmi_age"] = df["bmi"] * df["age"] / 100
    df["family_size"] = df["children"] + 1
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 45, 60, 100],
        labels=["young", "middle", "senior", "elderly"],
    ).astype(str)
    df = pd.get_dummies(
        df, columns=["sex", "smoker", "region", "age_group"], drop_first=True
    )
    df.drop(columns=["smoker_num"], errors="ignore", inplace=True)
    return df


def _fallback(d: dict) -> float:
    """Rule-based fallback when model not loaded."""
    pred = (
        3000
        + d["age"] * 120
        + max(0, (d["bmi"] - 25) * 200)
        + (15000 if d["smoker"] == "yes" else 0)
        + d["children"] * 500
    )
    return float(pred)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path).read())
    return HTMLResponse(
        "<h2>Medical Cost API</h2><p>Visit <a href='/docs'>/docs</a></p>"
    )


@app.get("/health")
async def health():
    _load()
    return {
        "status": "ok",
        "model": _META.get("best_model", "not loaded"),
        "metrics": _META.get("metrics", {}),
    }


@app.post("/predict")
async def predict(patient: PatientInput):
    _load()
    data = patient.dict()

    if _MODEL is None or _SCALER is None or _FEATURE_NAMES is None:
        charge = _fallback(data)
        model_name = "fallback_rules"
    else:
        try:
            df = pd.DataFrame([data])
            eng = _engineer(df).reindex(columns=_FEATURE_NAMES, fill_value=0)
            sc = _SCALER.transform(eng)
            charge = float(np.expm1(_MODEL.predict(sc)[0]))
            model_name = _META.get("best_model", "model")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    risk_n = (
        int(data["smoker"] == "yes") + int(data["bmi"] >= 30) + int(data["age"] >= 50)
    )
    risk_lv = ["Low", "Moderate", "High", "Very High"][min(risk_n, 3)]

    return {
        "predicted_charges": round(charge, 2),
        "confidence_range": [round(charge * 0.85, 2), round(charge * 1.15, 2)],
        "risk_level": risk_lv,
        "model_used": model_name,
    }
