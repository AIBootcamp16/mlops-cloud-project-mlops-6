"""
eval_report.py (final, fixed)
- 지표: Hit@K / Terms Hit@K / Country Hit@K / Diversity@K
- 목표: 텍스트 기반 추천의 핵심 지표를 산출하고, 모든 지표에 대한 차트를 생성한다.
- 결과물:
  - reports/eval_YYYYMMDD-HHMMSS.csv   (지표 표)
  - reports/eval_YYYYMMDD-HHMMSS.html  (요약 + 표 일부 + 모든 차트)
  - reports/eval_..._diversity.png     (다양성 히스토그램)
  - reports/eval_..._terms.png         (Terms Hit@K 히스토그램)
  - reports/eval_..._country.png       (Country Hit@K 히스토그램)
  - reports/eval_..._hit.png           (Hit@K 히스토그램)
"""

# 1. 표준/외부 모듈
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. 서드파티 유틸
from joblib import load
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel

# 3. 로컬 모듈
from src.validate import load_latest_frame
from src.reco.keywords import text_has_any_terms

# ✅ WandB
import wandb

# 4. 아티팩트 로더
def load_artifacts_any(style: str):
    if style == "all":
        vec  = load("artifacts/tfidf_all.pkl")
        X    = sparse.load_npz("artifacts/X_all.npz")
        keys = json.load(open("artifacts/keys_all.json","r",encoding="utf-8"))
        meta = json.load(open("artifacts/meta_all.json","r",encoding="utf-8"))
        frames = []
        for s in meta["styles"]:
            df_s = load_latest_frame(s).copy()
            df_s["style"] = s
            frames.append(df_s)
        df_idx = pd.concat(frames, ignore_index=True).set_index(["style","id"], drop=False)
        all_styles = meta["styles"]
        return vec, X, keys, df_idx, all_styles
    else:
        vec = load(f"artifacts/tfidf_{style}.pkl")
        X   = sparse.load_npz(f"artifacts/X_{style}.npz")
        ids = json.load(open(f"artifacts/ids_{style}.json","r",encoding="utf-8"))
        df  = load_latest_frame(style).copy()
        df["style"] = style
        df_idx = df.set_index(["style","id"], drop=False)
        keys = [{"style": style, "id": int(i)} for i in ids]
        all_styles = [style]
        return vec, X, keys, df_idx, all_styles

# 5. 유틸 함수들
def _item_text(row: dict) -> str:
    return f"{row.get('wine','')} {row.get('winery','')} {row.get('location','')}".lower()

def score_by_terms(vec, X, tokens):
    q = " ".join([t for t in (tokens or []) if t]).lower().strip() or "wine"
    qv = normalize(vec.transform([q]))
    return linear_kernel(X, qv).ravel()

def pick_topk(scores, keys, k):
    order = np.argsort(-scores)
    idxs = [i for i in order if scores[i] > -1e8][:k]
    picks = [keys[i] for i in idxs]
    return idxs, picks

def intra_list_similarity(X, picked_row_idxs):
    if len(picked_row_idxs) <= 1:
        return 0.0
    sub = X[picked_row_idxs]
    sim = linear_kernel(sub, sub)  # (k,k)
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = sim[mask]
    return float(np.mean(vals)) if vals.size else 0.0

# 6. 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="configs/users.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--style", default="all")
    args = ap.parse_args()

    Path("reports").mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    vec, X, keys, df_idx, all_styles = load_artifacts_any(args.style)
    users = json.load(open(args.users,"r",encoding="utf-8"))

    key_to_row = { (k["style"], int(k["id"])): i for i, k in enumerate(keys) }

    rows = []
    for u in users:
        uid = u["user_id"]

        # 쿼리 토큰
        q_tokens = (u.get("terms") or []) + (u.get("prefer_countries") or []) + (u.get("preferred_styles") or [])

        scores = score_by_terms(vec, X, q_tokens)
        idxs, picks = pick_topk(scores, keys, k=args.k)
        picked_row_idxs = [ key_to_row[(p["style"], int(p["id"]))] for p in picks ]

        # Hit / Terms Hit / Country Hit / Diversity
        if picks:
            sub = pd.DataFrame([df_idx.loc[(p["style"], int(p["id"]))].to_dict() for p in picks])
            base_terms = [t.lower() for t in (u.get("terms") or [])]
            pc = {x.lower() for x in (u.get("prefer_countries") or [])}

            terms_hit = sum(1 for _, r in sub.iterrows() if base_terms and any(t in _item_text(r.to_dict()) for t in base_terms)) / len(sub)
            country_hit = sum(1 for _, r in sub.iterrows() if (r.get("country") or "").lower() in pc) / len(sub) if pc else 0
            hit_at_k = int(any((any(t in _item_text(r.to_dict()) for t in base_terms)) or ((r.get("country") or "").lower() in pc) for _, r in sub.iterrows()))
            ils = intra_list_similarity(X, picked_row_idxs)
            diversity = 1.0 - ils
        else:
            terms_hit = country_hit = ils = diversity = 0.0
            hit_at_k = 0

        rows.append({
            "user_id": uid,
            "reco_count": len(picks),
            "hit@k": hit_at_k,
            "terms_hit@k": round(terms_hit, 3),
            "country_hit@k": round(country_hit, 3),
            "ILS@k": round(ils, 3),
            "Diversity@k": round(diversity, 3),
        })

    # CSV 저장
    ts_now = time.strftime("%Y%m%d-%H%M%S")
    out_csv = f"reports/eval_{ts_now}.csv"
    dfm = pd.DataFrame(rows)
    dfm.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 요약 통계
    summary = {
        "users": len(users),
        "k": args.k,
        "styles": ", ".join(sorted(all_styles)),
        "avg_hit@k": round(float(dfm["hit@k"].mean()), 3),
        "avg_terms_hit@k": round(float(dfm["terms_hit@k"].mean()), 3),
        "avg_country_hit@k": round(float(dfm["country_hit@k"].mean()), 3),
        "avg_diversity@k": round(float(dfm["Diversity@k"].mean()), 3),
    }

    # 차트 저장
    def _hist(col, fname, title, xlabel, bins=10):
        plt.figure()
        plt.hist(dfm[col].values, bins=bins)
        out = f"reports/eval_{ts_now}_{fname}.png"
        plt.title(title)
        plt.xlabel(xlabel); plt.ylabel("Count")
        plt.savefig(out, bbox_inches="tight"); plt.close()
        return out

    diversity_png = _hist("Diversity@k", "diversity", "Diversity distribution", "Diversity", bins=10)
    terms_png     = _hist("terms_hit@k", "terms", "Terms Hit@K distribution", "Terms Hit@K", bins=10)
    country_png   = _hist("country_hit@k", "country", "Country Hit@K distribution", "Country Hit@K", bins=10)
    hit_png       = _hist("hit@k", "hit", "Hit@K distribution", "Hit@K", bins=2)

    # ✅ WandB logging
    run = wandb.init(
        project="wine-reco",  # 팀원이 동일하게 사용해야 하는 프로젝트 이름
        job_type="eval_report",
        name=f"{args.style}_k{args.k}_{time.strftime('%Y%m%d-%H%M%S')}"
    )

    metrics = {
        "avg_hit_at_k": float(summary["avg_hit@k"]),
        "avg_terms_hit_at_k": float(summary["avg_terms_hit@k"]),
        "avg_country_hit_at_k": float(summary["avg_country_hit@k"]),
        "avg_diversity_at_k": float(summary["avg_diversity@k"])
    }

    # log + summary update
    wandb.log(metrics)
    run.summary.update(metrics)

    print(f"[EVAL] csv={out_csv}")

# 7. 엔트리
if __name__ == "__main__":
    main()
