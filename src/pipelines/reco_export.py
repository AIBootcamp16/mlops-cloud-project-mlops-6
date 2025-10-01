"""
reco_export.py
- 사용자별 추천을 HTML 한 장으로 내보낸다.
- per-style (스타일 하나)모드와 all 모드(전체 스타일)를 모두 지원한다.
- 스타일 종류: reds, whites, sparkling, rose, port
Usage:
  # 통합 모델에서 전 스타일 글로벌 Top-K (회피/선호 게이트 반영)
  python -m src.pipelines.reco_export --users configs/users.json --style all --k 5

  # 단일 스타일에서 Top-K
  python -m src.pipelines.reco_export --users configs/users.json --style whites --k 5
"""
from __future__ import annotations

# 1. 표준 라이브러리
import argparse, json, time, re
from pathlib import Path

# 2. 서드파티
import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel

# 3. 우리 모듈
from src.validate import load_latest_frame

# 4. 리뷰 수/평점 파싱
def _reviews_count(r):
    s = (r or {}).get("reviews") if isinstance(r, dict) else None
    if not s: return 0
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else 0

def _avg_rating(r):
    try: return float((r or {}).get("average")) if isinstance(r, dict) else 0.0
    except: return 0.0

# 5. 텍스트 쿼리 점수
def score_by_terms(vec, X, terms):
    q = " ".join(terms).lower().strip() if terms else "wine"
    qv = normalize(vec.transform([q]))
    return linear_kernel(X, qv).ravel()

# 6. 소프트 가감점
def apply_soft_prefs(scores: np.ndarray, df_idx: pd.DataFrame, keys: list[dict], u: dict,
                     boost=0.05, penalty=0.20) -> np.ndarray:
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
        text    = f"{row.get('wine','')} {row.get('winery','')} {row.get('location','')}".lower()
        if pc and country in pc: s[i] += boost
        if ac and country in ac: s[i] -= penalty
        if pw and winery  in pw: s[i] += boost
        if aw and winery  in aw: s[i] -= penalty
        if pt and any(t in text for t in pt): s[i] += boost
        if at and any(t in text for t in at): s[i] -= penalty
    return s

# 7. 하드 필터
def apply_hard_filters(scores: np.ndarray, df_idx: pd.DataFrame, keys: list[dict],
                       mask_allowed: np.ndarray, min_reviews: int, min_rating: float) -> np.ndarray:
    s = scores.copy().astype(float)
    s[~mask_allowed] = -1e9
    if min_reviews or min_rating:
        keep = np.ones_like(s, dtype=bool)
        for i, k in enumerate(keys):
            if not mask_allowed[i]:
                keep[i] = False
                continue
            row = df_idx.loc[(k["style"], int(k["id"]))]
            ok = True
            if min_reviews:
                ok &= _reviews_count(row.get("rating")) >= int(min_reviews)
            if min_rating:
                ok &= _avg_rating(row.get("rating")) >= float(min_rating)
            keep[i] = ok
        s[~keep] = -1e9
    return s

# 8. 아티팩트 로더
def load_artifacts_any(style: str):
    # 9. all 모드
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
        df_all = pd.concat(frames, ignore_index=True)
        df_idx = df_all.set_index(["style","id"], drop=False)
        styles = meta["styles"]
        return vec, X, keys, df_idx, styles
    # 10. per-style 모드
    else:
        vec = load(f"artifacts/tfidf_{style}.pkl")
        X   = sparse.load_npz(f"artifacts/X_{style}.npz")
        ids = json.load(open(f"artifacts/ids_{style}.json","r",encoding="utf-8"))
        df  = load_latest_frame(style).copy()
        df["style"] = style
        df_idx = df.set_index(["style","id"], drop=False)
        keys = [{"style": style, "id": int(i)} for i in ids]
        styles = [style]
        return vec, X, keys, df_idx, styles

# 11. HTML 프레임
def html_header() -> list[str]:
    return [
        "<!doctype html><meta charset='utf-8'>",
        "<style>",
        "body{font-family:system-ui,Arial;margin:24px;background:#fafafa}",
        ".user{margin:30px 0}",
        ".title{font-size:20px;font-weight:700;margin:8px 0}",
        ".style{font-size:16px;font-weight:600;margin:6px 0;color:#444}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}",
        ".card{background:#fff;border-radius:12px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,.06)}",
        ".img{width:100%;height:220px;object-fit:contain;background:#fff;border:1px solid #eee;border-radius:8px}",
        ".meta{font-size:13px;color:#333;margin-top:6px;line-height:1.35}",
        "</style>"
    ]

# 12. 메인
def main():
    # 13. 인자
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="configs/users.json")
    ap.add_argument("--style", default="all")  # all 또는 reds/whites/...
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 14. 데이터 로드
    vec, X, keys, df_idx, all_styles = load_artifacts_any(args.style)
    users = json.load(open(args.users, "r", encoding="utf-8"))

    # 15. 출력 경로
    Path("reports").mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or f"reports/reco_{args.style}_{ts}.html"

    # 16. HTML 시작
    html = html_header()

    # 17. 사용자별 루프
    for u in users:
        terms = ", ".join(u.get("terms", [])) or "키워드 없음"
        html.append(f"<div class='user'><div class='title'>👤 {u['user_id']} 님에게 추천 (Top {args.k}) "
                    f"<span style='font-weight:400;color:#666'>(키워드: {terms})</span></div>")

        # 18. 스타일 게이트
        pref = set(u.get("preferred_styles") or [])
        avoid = set(u.get("avoid_styles") or [])
        if pref:
            allowed_styles = {s for s in all_styles if s in pref}
        else:
            allowed_styles = {s for s in all_styles if s not in avoid}

        # 18. 스타일 게이트
        pref = set(u.get("preferred_styles") or [])
        avoid = set(u.get("avoid_styles") or [])
        if pref:
            allowed_styles = {s for s in all_styles if s in pref}
        else:
            allowed_styles = {s for s in all_styles if s not in avoid}

        # 18-1. 라벨: 'all' 대신 허용 스타일 나열(없으면 all)
        styles_label = ", ".join(sorted(allowed_styles)) if allowed_styles else "all"
        html.append(f"<div class='style'>• {styles_label}</div>")


        # 19. 기본 점수
        scores = score_by_terms(vec, X, u.get("terms"))

        # 20. 허용 스타일 마스크
        mask_allowed = np.array([k["style"] in allowed_styles for k in keys], dtype=bool)

        # 21. 소프트 가감점
        scores = apply_soft_prefs(scores, df_idx, keys, u)

        # 22. 하드 필터
        scores = apply_hard_filters(scores, df_idx, keys, mask_allowed,
                                    u.get("min_reviews", 0), u.get("min_rating", 0.0))

        # 23. 상위 K 선택
        order = np.argsort(-scores)
        picks = [keys[i] for i in order if scores[i] > -1e8][:args.k]

        # 24. 렌더
        # html.append(f"<div class='style'>• {args.style if args.style=='all' else list(allowed_styles)[0] if len(allowed_styles)==1 else 'selected styles'}</div>")
        html.append("<div class='cards'>")
        for k in picks:
            row = df_idx.loc[(k["style"], int(k["id"]))]
            name, winery, country = row.get("wine","?"), row.get("winery",""), row.get("country","")
            img = row.get("image","")
            html.append("<div class='card'>")
            if img: html.append(f"<img class='img' src='{img}' alt='label'>")
            html.append(f"<div class='meta'><b>{name}</b><br>{winery} — {country} <span style='color:#777'>({k['style']})</span></div>")
            html.append("</div>")
        html.append("</div>")  # .cards
        html.append("</div>")  # .user

    # 25. 저장
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[EXPORT] {out}")

# 26. 엔트리
if __name__ == "__main__":
    main()
