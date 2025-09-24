# src/pipelines/preview.py
# --------------------------------------------
# 간단 미리보기 스크립트
# - 쿼리 기반 Top-K:   --q pinot noir --k 5
# - 아이템 기반 Top-K: --like-id 417  --k 5
# --------------------------------------------

# 1. 표준/외부 모듈 임포트
import argparse, json
import numpy as np
from joblib import load
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from src.validate import load_latest_frame

# 2. 쿼리(단어들) → 점수 벡터
def score_by_query(vec, X, terms):
    # 3. 쿼리를 하나의 문자열로 합치고 소문자/트림
    query = " ".join(terms).lower().strip()
    # 4. 벡터라이저로 TF-IDF 벡터화
    qv = vec.transform([query])
    # 5. 코사인 유사도 계산을 위한 정규화
    qv = normalize(qv)
    # 6. linear_kernel == dot == cosine(정규화 가정) → (N, 1) → (N,)
    return linear_kernel(X, qv).ravel()

# 7. 특정 아이템(행)과 비슷한 아이템 점수
def score_by_item(X, row_idx):
    # 8. X[row_idx]와 모든 행의 코사인 유사도 → (N, 1) → (N,)
    sims = linear_kernel(X, X[row_idx]).ravel()
    # 9. 자기 자신은 제외(가장 유사해서 항상 1.0에 가깝기 때문)
    sims[row_idx] = -1.0
    return sims

# 10. 메인 진입
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", nargs="+", help="query terms, e.g., pinot noir")
    ap.add_argument("--like-id", type=int, help="recommend items similar to this wine id")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--style", default="reds")
    args = ap.parse_args()

    # 11. 아티팩트/데이터 로드
    vec  = load("artifacts/tfidf.pkl")                                  # 벡터라이저
    X    = sparse.load_npz(f"artifacts/X_{args.style}.npz")             # TF-IDF 행렬
    ids  = json.load(open(f"artifacts/ids_{args.style}.json","r",encoding="utf-8"))  # 행→id
    df   = load_latest_frame(args.style)                                 # 표시용 메타

    # 12. id ↔ index 빠른 조회를 위해 딕셔너리 생성
    id2idx = {wid: i for i, wid in enumerate(ids)}

    # 13. 실행 모드 분기
    if args.like_id is not None:
        # 14. 아이템 유사 모드: 입력 id가 ids에 존재하는지 확인
        if args.like_id not in id2idx:
            print(f"[PREVIEW] id={args.like_id} not found in artifacts.")
            return
        row_idx = id2idx[args.like_id]
        scores = score_by_item(X, row_idx)
        mode_desc = f"like-id={args.like_id}"
    elif args.q:
        # 15. 쿼리 모드
        scores = score_by_query(vec, X, args.q)
        mode_desc = f"query={' '.join(args.q)}"
    else:
        print("[PREVIEW] Provide either --q <terms...> or --like-id <id>.")
        return

    # 16. 상위 K 인덱스 (내림차순)
    top = np.argsort(-scores)[:args.k]

    # 17. 결과 표시
    print(f"[PREVIEW] {mode_desc!r}, top{args.k}")
    for rank, idx in enumerate(top, 1):
        wid = ids[idx] if idx < len(ids) else None
        row = df.loc[df["id"] == wid].head(1) if wid is not None else df.iloc[0:0]
        if not row.empty:
            name    = row.iloc[0].get("wine", "?")
            winery  = row.iloc[0].get("winery", "")
            country = row.iloc[0].get("country", "")
        else:
            name = winery = country = "?"
        print(f"{rank:2d}. id={wid} | {name} — {winery} ({country}) | score={scores[idx]:.4f}")

# 18. 스크립트 실행 진입점
if __name__ == "__main__":
    main()
