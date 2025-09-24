# src/app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from joblib import load
from scipy import sparse
import json
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from src.validate import load_latest_frame

app = FastAPI()

# --- 지원하는 스타일 ---
STYLES = ["reds", "whites", "sparkling", "rose", "port"]

# --- 아티팩트 로드 캐시 ---
artifacts = {}

def load_artifacts(style: str):
    """스타일별 아티팩트 로드 (캐싱)"""
    if style not in artifacts:
        vec = load(f"artifacts/tfidf_{style}.pkl")
        X = sparse.load_npz(f"artifacts/X_{style}.npz")
        ids = json.load(open(f"artifacts/ids_{style}.json", "r", encoding="utf-8"))
        df = load_latest_frame(style)
        df_idx = df.set_index("id", drop=False)
        artifacts[style] = (vec, X, ids, df_idx)
    return artifacts[style]

class QueryInput(BaseModel):
    terms: list[str]
    k: int = 5
    style: str = "reds"

@app.get("/")
def root():
    return {"message": "Wine Reco API Ready!", "styles_available": STYLES}

@app.post("/recommend")
def recommend(q: QueryInput):
    style = q.style
    if style not in STYLES:
        return {"error": f"Unsupported style '{style}'. Choose from {STYLES}"}

    vec, X, ids, df_idx = load_artifacts(style)

    query_text = " ".join(q.terms) if q.terms else "wine"
    qv = normalize(vec.transform([query_text]))
    scores = linear_kernel(X, qv).ravel()
    order = np.argsort(-scores)
    picks = [int(ids[i]) for i in order[:q.k]]
    wines = df_idx.loc[picks].to_dict(orient="records")
    return {"style": style, "recommendations": wines}

