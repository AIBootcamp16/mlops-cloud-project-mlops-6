"""
-eval_report.py (final)
- 지표: Hit@K / Terms Hit@K / Country Hit@K / Diversity@K
- 목표: 텍스트 기반 추천의 핵심 지표를 산출하고, 모든 지표에 대한 차트를 생성한다.
- 결과물:
  - reports/eval_YYYYMMDD-HHMMSS.csv   (지표 표)
  - reports/eval_YYYYMMDD-HHMMSS.html  (요약 + 표 일부 + 모든 차트)
  - reports/eval_..._diversity.png     (다양성 히스토그램)
  - reports/eval_..._terms.png         (Terms Hit@K 히스토그램)
  - reports/eval_..._country.png       (Country Hit@K 히스토그램)
  - reports/eval_..._hit.png           (Hit@K 히스토그램)

Usage:
  python -m src.pipelines.eval_report --users configs/users.json --k 5 --style all
  python -m src.pipelines.eval_report --users configs/users.json --k 5 --style reds
"""

# 1. 표준/외부 모듈
import argparse, json, os, time, re
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
from src.reco.keywords import canon_country, text_has_any_terms

# 4. 아티팩트 로더: all/per-style 모두 지원
def load_artifacts_any(style: str):
    # 4.1 all 모드: 통합 아티팩트 + 스타일 병합 인덱스
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
    # 4.2 per-style: 기존 방식
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


# 5. 유틸: 평문 텍스트/스코어링/필터
def _item_text(row: dict) -> str:
    return f"{row.get('wine','')} {row.get('winery','')} {row.get('location','')}".lower()

def score_by_terms(vec, X, tokens):
    # 5.1 쿼리 토큰: terms만 사용 (정밀도 유지)
    q = " ".join([t for t in (tokens or []) if t]).lower().strip() or "wine"
    qv = normalize(vec.transform([q]))
    return linear_kernel(X, qv).ravel()

def apply_soft_prefs(scores, df_idx, keys, u, boost=0.25, penalty=0.30):
    # 5.2 소프트 선호/회피 가중치 (terms/국가/와이너리)
    s = scores.copy().astype(float)
    pc = {x.lower() for x in (u.get("prefer_countries") or [])}
    ac = {x.lower() for x in (u.get("avoid_countries")  or [])}
    pw = {x.lower() for x in (u.get("prefer_wineries")  or [])}
    aw = {x.lower() for x in (u.get("avoid_wineries")   or [])}
    pt = {x.lower() for x in (u.get("terms")            or [])}
    at = {x.lower() for x in (u.get("avoid_terms")      or [])}
    for i, k in enumerate(keys):
        key = (k["style"], int(k["id"]))
        if key not in df_idx.index:
            continue
        row = df_idx.loc[key]
        country = (row.get("country") or "").lower()
        winery  = (row.get("winery")  or "").lower()
        text    = _item_text(row.to_dict())
        if pc and country in pc: s[i] += boost
        if ac and country in ac: s[i] -= penalty
        if pw and winery  in pw: s[i] += boost
        if aw and winery  in aw: s[i] -= penalty
        if pt and any(t in text for t in pt): s[i] += boost
        if at and any(t in text for t in at): s[i] -= penalty
    return s

def apply_hard_filters(scores, df_idx, keys, allowed_styles, min_reviews, min_rating):
    # 5.3 스타일/품질 하드 필터 (품질 필드는 현 리포트에 미노출)
    s = scores.copy().astype(float)
    mask_allowed = np.array([k["style"] in allowed_styles for k in keys], dtype=bool) if allowed_styles else np.ones(len(keys), dtype=bool)
    s[~mask_allowed] = -1e9
    if (min_reviews and min_reviews>0) or (min_rating and min_rating>0):
        keep = np.ones_like(s, dtype=bool)
        for i, k in enumerate(keys):
            if not mask_allowed[i]:
                keep[i] = False
                continue
            # 5.3.1 rating/reviews 체크는 메타 구조에 맞춰 확장 가능
            keep[i] = True
        s[~keep] = -1e9
    return s

def pick_topk(scores, keys, k):
    # 5.4 Top-K 선택
    order = np.argsort(-scores)
    idxs = [i for i in order if scores[i] > -1e8][:k]
    picks = [keys[i] for i in idxs]
    return idxs, picks


# 6. 다양성(ILS) 계산: 추천끼리 평균 유사도
def intra_list_similarity(X, picked_row_idxs):
    if len(picked_row_idxs) <= 1:
        return 0.0
    sub = X[picked_row_idxs]
    sim = linear_kernel(sub, sub)  # (k,k)
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = sim[mask]
    return float(np.mean(vals)) if vals.size else 0.0


# 7. 메인
def main():
    # 7.1 인자
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="configs/users.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--style", default="all")  # all 또는 reds/whites/...
    args = ap.parse_args()

    # 7.2 출력 폴더/타임스탬프
    Path("reports").mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")

    # 7.3 아티팩트/데이터 로드
    vec, X, keys, df_idx, all_styles = load_artifacts_any(args.style)
    users = json.load(open(args.users,"r",encoding="utf-8"))

    # 7.3.1 키→행 인덱스 매핑 (ILS 계산용)
    key_to_row = { (k["style"], int(k["id"])): i for i, k in enumerate(keys) }

    rows = []
    for u in users:
        uid = u["user_id"]

        # 7.4 허용 스타일 계산(선호 우선, 없으면 회피 제외)
        pref = set(u.get("preferred_styles") or [])
        avoid = set(u.get("avoid_styles") or [])
        if pref:
            allowed_styles = {s for s in all_styles if s in pref}
        else:
            allowed_styles = {s for s in all_styles if s not in avoid}

        # 7.5 쿼리 토큰 (terms만)
        q_tokens = (u.get("terms") or []) \
                + (u.get("prefer_countries") or []) \
                + (u.get("preferred_styles") or [])

        # 7.6 점수 계산
        scores = score_by_terms(vec, X, q_tokens)
        scores = apply_soft_prefs(scores, df_idx, keys, u, boost=0.12, penalty=0.25)

        # 7.7 Top-K 선택 → 슬롯 규칙 적용 → 인덱스 재계산
        idxs, picks = pick_topk(scores, keys, k=args.k)              # 7.7.1 기본 Top-K

        # 7.7.2 terms / country 최소할당 규칙
        def enforce_quota(picks, df_idx, user, k):
            # 1 후보 DF
            sub = [df_idx.loc[(p["style"], int(p["id"]))].to_dict() for p in picks]
            df = pd.DataFrame(sub)
            # 2 매칭 플래그
            base_terms = [t.lower() for t in (user.get("terms") or [])]
            pc = {x.lower() for x in (user.get("prefer_countries") or [])}
            def _tmask(r):
                if not base_terms: return False
                text = f"{r.get('wine','')} {r.get('winery','')} {r.get('location','')}".lower()
                return any(t in text for t in base_terms)
            term_mask = df.apply(_tmask, axis=1)
            country_mask = df.apply(lambda r: (str(r.get("country",""))).lower() in pc if pc else False, axis=1)
            # 3 최소 개수
            need_terms = 1
            need_country = 1 if pc else 0
            # 4 우선 선발
            idx_terms = list(np.where(term_mask.values)[0])
            idx_country = list(np.where(country_mask.values)[0])
            take = []
            for i in idx_terms:
                if len(take) < need_terms:
                    take.append(i)
            for i in idx_country:
                if len(take) < need_terms + need_country and i not in take:
                    take.append(i)
            # 4.3 나머지 상위 점수로 채우기
            for i in range(len(df)):
                if len(take) >= k: break
                if i not in take:
                    take.append(i)
            return [picks[i] for i in take[:k]]

        picks = enforce_quota(picks, df_idx, u, k=args.k)            # 7.7.3 슬롯 규칙 적용
        picked_row_idxs = [ key_to_row[(p["style"], int(p["id"]))] for p in picks ]  # 7.7.4 ILS용 인덱스

        # 7.8 지표 계산
        if picks:
            # 7.8.1 추천 아이템 DF
            sub = pd.DataFrame([df_idx.loc[(p["style"], int(p["id"]))].to_dict() for p in picks])

            # 7.8.2 Terms Hit@K
            base_terms = [t.lower() for t in (u.get("terms") or [])]
            term_hits = 0
            for _, r in sub.iterrows():
                text = _item_text(r.to_dict())
                if base_terms and any(t in text for t in base_terms):
                    term_hits += 1
            terms_hit = term_hits / len(sub) if len(sub)>0 else 0.0

            # 7.8.3 Country Hit@K
            pc = {x.lower() for x in (u.get("prefer_countries") or [])}
            country_hits = 0
            for _, r in sub.iterrows():
                c = (r.get("country") or "").lower()
                if pc and c in pc:
                    country_hits += 1
            country_hit = country_hits / len(sub) if len(sub)>0 else 0.0

            # 7.8.4 Hit@K (terms OR (country AND style))
            pref_styles = set(u.get("preferred_styles") or [])
            hit = 0
            for _, r in sub.iterrows():
                text = _item_text(r.to_dict())
                style_ok = (r.get("style") in pref_styles) if pref_styles else False
                term_ok  = (base_terms and any(t in text for t in base_terms))
                country_ok = False
                if pc:
                    c = (r.get("country") or "").lower()
                    country_ok = c in pc
                if term_ok or (country_ok and style_ok):
                    hit = 1
                    break
            hit_at_k = int(hit)

            # 7.8.5 ILS / Diversity
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

    # 7.9 CSV 저장
    ts_now = time.strftime("%Y%m%d-%H%M%S")
    out_csv = f"reports/eval_{ts_now}.csv"
    dfm = pd.DataFrame(rows)
    dfm.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 7.10 요약 통계
    zero_users = int((dfm["reco_count"]==0).sum())
    summary = {
        "users": len(users),
        "k": args.k,
        "styles": ", ".join(sorted(all_styles)),
        "zero_reco_users": zero_users,
        "avg_hit@k": round(float(dfm["hit@k"].mean()), 3),
        "avg_terms_hit@k": round(float(dfm["terms_hit@k"].mean()), 3),
        "avg_country_hit@k": round(float(dfm["country_hit@k"].mean()), 3),
        "avg_diversity@k": round(float(dfm["Diversity@k"].mean()), 3),
    }

    # 7.11 차트 저장 (모든 지표)
    def _hist(col, fname, title, xlabel, bins=10):
        plt.figure()
        plt.hist(dfm[col].values, bins=bins)
        out = f"reports/eval_{ts_now}_{fname}.png"
        plt.title(title)
        plt.xlabel(xlabel); plt.ylabel("Count")
        plt.savefig(out, bbox_inches="tight"); plt.close()
        return out

    diversity_png = _hist("Diversity@k", "diversity", "Diversity (1 - ILS) distribution", "Diversity", bins=10)
    terms_png     = _hist("terms_hit@k", "terms", "Terms Hit@K distribution", "Terms Hit@K", bins=10)
    country_png   = _hist("country_hit@k", "country", "Country Hit@K distribution", "Country Hit@K", bins=10)
    hit_png       = _hist("hit@k", "hit", "Hit@K distribution", "Hit@K", bins=2)

    # 7.12 HTML 요약 저장
    out_html = f"reports/eval_{ts_now}.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>")
        f.write("<style>body{font-family:system-ui,Arial;margin:24px} table{border-collapse:collapse} td,th{padding:6px 10px;border:1px solid #ddd}</style>")
        f.write("<h2>Recommendation Eval Summary</h2><pre>")
        for k,v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("</pre>")
        f.write("<h3>Charts</h3>")
        for img in [hit_png, terms_png, country_png, diversity_png]:
            f.write(f"<img src='{os.path.basename(img)}' width='420' style='margin-right:12px;margin-bottom:12px'>")
        f.write("<h3>Per-user Metrics (Top 20 rows)</h3>")
        f.write(dfm[[
            'user_id','reco_count','hit@k','terms_hit@k','country_hit@k','Diversity@k'
        ]].head(20).to_html(index=False))
    print(f"[EVAL] csv={out_csv}\n[EVAL] html={out_html}")


# 8. 엔트리
if __name__ == "__main__":
    main()
