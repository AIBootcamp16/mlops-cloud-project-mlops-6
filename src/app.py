from fastapi import FastAPI, Query
import wandb, joblib, scipy.sparse as sp, json, os
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

# ✅ 환경 변수 (팀원별 계정 가능)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "hwanseok0629-")
WANDB_PROJECT = "wine-reco"

# ✅ 캐시된 모델 보관
_models = {}

def load_model(style: str = "reds"):
    if style in _models:
        return _models[style]

    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="api_server", reinit=True)
    artifact = run.use_artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/tfidf-{style}:latest", type="model")
    artifact_dir = artifact.download()

    vec = joblib.load(f"{artifact_dir}/tfidf_{style}.pkl")
    X   = sp.load_npz(f"{artifact_dir}/X_{style}.npz")
    ids = json.load(open(f"{artifact_dir}/ids_{style}.json","r",encoding="utf-8"))

    _models[style] = (vec, X, ids)
    return vec, X, ids


@app.get("/")
def read_root():
    return {"message": "Hello, Wine Recommendation API!"}

@app.get("/info")
def get_info(style: str = "reds"):
    vec, X, ids = load_model(style)
    return {
        "style": style,
        "rows": X.shape[0],
        "dims": X.shape[1],
        "num_ids": len(ids),
        "artifact_cached": True
    }

@app.get("/recommend")
def recommend(style: str = "reds", query: str = Query(...), k: int = 5):
    vec, X, ids = load_model(style)

    qv = vec.transform([query])
    scores = linear_kernel(X, qv).ravel()
    order = np.argsort(-scores)[:k]

    results = [{"id": int(ids[i]), "score": float(scores[i])} for i in order]
    return {"query": query, "style": style, "top_k": results}
