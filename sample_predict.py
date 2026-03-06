import json, logging, os, sys
import joblib, numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
MODELS_PATH = "models"


def load_artifacts():
    for f in ["best_model.pkl", "scaler.pkl", "feature_names.pkl", "model_meta.json"]:
        if not os.path.exists(f"{MODELS_PATH}/{f}"):
            print(
                f"\n❌  Missing: {MODELS_PATH}/{f}\n   Run python run_pipeline.py first.\n"
            )
            sys.exit(1)
    model = joblib.load(f"{MODELS_PATH}/best_model.pkl")
    scaler = joblib.load(f"{MODELS_PATH}/scaler.pkl")
    feats = joblib.load(f"{MODELS_PATH}/feature_names.pkl")
    meta = json.load(open(f"{MODELS_PATH}/model_meta.json"))
    return model, scaler, feats, meta


def engineer(df):
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


def predict(patient, model, scaler, feats):
    df = pd.DataFrame([patient])
    eng = engineer(df).reindex(columns=feats, fill_value=0)
    charge = float(np.expm1(model.predict(scaler.transform(eng))[0]))
    risk_n = (
        int(patient["smoker"] == "yes")
        + int(patient["bmi"] >= 30)
        + int(patient["age"] >= 50)
    )
    return {
        "predicted_charges": round(charge, 2),
        "range": [round(charge * 0.85, 2), round(charge * 1.15, 2)],
        "risk_level": ["Low", "Moderate", "High", "Very High"][min(risk_n, 3)],
    }


if __name__ == "__main__":
    model, scaler, feats, meta = load_artifacts()
    print(
        f"\n  Model: {meta['best_model']}  |  R²={meta['metrics']['R2']}  |  RMSE=${meta['metrics']['RMSE']:,.0f}"
    )
    cases = [
        (
            "Young non-smoker",
            {
                "age": 25,
                "sex": "female",
                "bmi": 22.0,
                "children": 0,
                "smoker": "no",
                "region": "northwest",
            },
        ),
        (
            "Middle-aged smoker, obese",
            {
                "age": 45,
                "sex": "male",
                "bmi": 35.2,
                "children": 2,
                "smoker": "yes",
                "region": "southeast",
            },
        ),
        (
            "Senior smoker, very obese",
            {
                "age": 60,
                "sex": "male",
                "bmi": 42.0,
                "children": 0,
                "smoker": "yes",
                "region": "southwest",
            },
        ),
        (
            "Young smoker, lean",
            {
                "age": 28,
                "sex": "female",
                "bmi": 19.5,
                "children": 0,
                "smoker": "yes",
                "region": "northeast",
            },
        ),
        (
            "Non-smoker, 3 children",
            {
                "age": 40,
                "sex": "male",
                "bmi": 27.3,
                "children": 3,
                "smoker": "no",
                "region": "northeast",
            },
        ),
    ]
    print("\n" + "─" * 58)
    for label, d in cases:
        r = predict(d, model, scaler, feats)
        print(f"\n  {label}")
        print(f"    age={d['age']}  bmi={d['bmi']}  smoker={d['smoker']}")
        print(f"    Predicted : ${r['predicted_charges']:>10,.2f}")
        print(f"    Range     : ${r['range'][0]:>10,.2f} – ${r['range'][1]:,.2f}")
        print(f"    Risk      : {r['risk_level']}")
    print("\n" + "─" * 58)
