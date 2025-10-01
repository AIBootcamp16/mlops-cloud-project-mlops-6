# src/pipelines/users_generate.py
"""
users_generate.py
- 여러 스타일의 검증된 DF를 결합하여 자주 등장하는 국가/와이너리/단어 분포를 추출하고,
  이를 기반으로 가상 고객 N명을 생성하여 configs/users.json에 저장한다.
- 행동 로그는 사용하지 않고 텍스트 기반 선호만 반영한다.

Usage:
  python -m src.pipelines.users_generate --n 30 --styles reds,whites,sparkling,rose,port --out configs/users.json --minimal
"""

# 1. 표준/외부 모듈
import argparse
import json
import random
from collections import Counter
from typing import List

import numpy as np
import pandas as pd

# 2. 로컬 프로젝트 모듈
from src.validate import load_latest_frame
from src.reco.keywords import (
    clean_terms,
    extract_terms_from_text,
    canon_country,
)

# 3. 헬퍼: 가중치 고유 샘플링
def _weighted_unique_sample(pool: List[str], weights: np.ndarray, n: int, exclude: set[str] | None = None) -> List[str]:
    # 3.1 exclude 처리
    exclude = exclude or set()
    items = [c for c in pool if c not in exclude]
    if not items:
        return []
    w = np.array([weights[pool.index(c)] for c in items], dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    # 3.2 고유 샘플
    picks = []
    while items and len(picks) < n:
        choice = np.random.choice(items, p=w)
        picks.append(choice)
        j = items.index(choice)
        items.pop(j)
        w = np.delete(w, j)
        if w.size > 0:
            if w.sum() <= 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
    return picks

# 4. 메인
def main():
    # 4.1 인자
    ap = argparse.ArgumentParser(description="Generate synthetic users from wine metadata")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--styles", default="reds,whites,sparkling,rose,port")
    ap.add_argument("--out", default="configs/users.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p_pref_style", type=float, default=0.7)
    ap.add_argument("--p_avoid_style", type=float, default=0.3)
    ap.add_argument("--p_pref_winery", type=float, default=0.4)
    ap.add_argument("--p_avoid_country", type=float, default=0.2)
    ap.add_argument("--minimal", action="store_true")
    ap.add_argument("--min_terms", type=int, default=2)
    args = ap.parse_args()

    # 4.2 시드
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 4.3 스타일 로드
    style_list = [s.strip() for s in args.styles.split(",") if s.strip()]
    frames: List[pd.DataFrame] = []
    for style in style_list:
        df = load_latest_frame(style)
        if not df.empty:
            df = df.copy()
            df["style"] = style
            frames.append(df)
    if not frames:
        raise SystemExit("Empty DF pool. Run snapshot/embed first.")
    df_all = pd.concat(frames, ignore_index=True)

    # 4.4 텍스트/국가/와이너리
    countries_raw = df_all["country"].fillna("")
    countries = [canon_country(str(x)) for x in countries_raw]
    wineries  = [str(x) for x in df_all["winery"].fillna("")]
    texts = (df_all["wine"].fillna("") + " " + df_all["winery"].fillna("") + " " + df_all["location"].fillna(""))

    # 4.5 분포 추출 (나라/와이너리/키워드)
    top_countries = [c for c, _ in Counter([c for c in countries if c]).most_common(50)]
    top_wineries  = [w for w, _ in Counter([w for w in wineries if w]).most_common(40)]

    word_cnt = Counter()
    for s in texts:
        # 4.5.1 terms에서 국가는 제거
        word_cnt.update(extract_terms_from_text(s, max_terms=8, keep_countries=False))
    top_terms = [w for w, _ in word_cnt.most_common(200)]

    # 4.6 국가 가중치 풀 (공급 기반)
    country_counts = Counter([c for c in countries if c])
    pop_countries = [c for c, _ in country_counts.most_common(50)]
    pop_weights = np.array([country_counts[c] for c in pop_countries], dtype=float)
    if pop_weights.size > 0 and pop_weights.sum() > 0:
        pop_weights = pop_weights / pop_weights.sum()
    else:
        pop_weights = np.ones(len(pop_countries), dtype=float) / max(1, len(pop_countries))

    # 4.7 Top5 국가 (무조건 1개 포함 규칙에 사용)
    top5_countries = [c for c, _ in country_counts.most_common(5)]

    # 4.8 디버그 출력
    uniq_countries = sorted({c for c in countries if c})
    print(f"[USERS] unique countries: {len(uniq_countries)}")
    print(f"[USERS] top5 countries: {top5_countries}")

    # 4.9 사용자 생성
    users = []
    for i in range(args.n):
        uid = f"u{i+1:02d}"

        # 4.9.1 terms (국가 제거) 최소 k개 보장
        k_terms = max(args.min_terms, 2)
        if len(top_terms) >= k_terms:
            terms = random.sample(top_terms, k_terms)
        else:
            terms = top_terms[:]
        terms = clean_terms(terms, max_terms=k_terms, keep_countries=False)
        if len(terms) < k_terms:
            for w, _ in word_cnt.most_common(300):
                if w not in terms:
                    terms.append(w)
                    if len(terms) == k_terms:
                        break

        # 4.9.2 스타일 선호/회피
        preferred_styles = []
        avoid_styles = []
        if random.random() < args.p_pref_style:
            preferred_styles = random.sample(style_list, k=min(2, max(1, len(style_list)//3)))
        remaining = [s for s in style_list if s not in preferred_styles]
        if remaining and random.random() < args.p_avoid_style:
            avoid_styles = random.sample(remaining, k=1)

        # 4.9.3 선호 국가 (최소 2개, 가끔 3개) + Top5 중 1개 반드시 포함
        m = 2 if len(pop_countries) < 3 else np.random.choice([2, 3], p=[0.7, 0.3])
        prefer_countries: List[str] = []

        # 4.9.3.1 Top5에서 1개 고정
        if top5_countries:
            # 4.9.3.1.1 Top5 가중치 재계산
            t5 = [c for c in top5_countries if c in pop_countries]
            if t5:
                idx = [pop_countries.index(c) for c in t5]
                w = pop_weights[idx]
                if w.sum() <= 0:
                    w = np.ones_like(w) / len(w)
                else:
                    w = w / w.sum()
                prefer_countries.append(np.random.choice(t5, p=w))
            else:
                prefer_countries.append(random.choice(top5_countries))

        # 4.9.3.2 나머지 국가는 공급 가중치 기반 고유 샘플
        need_more = max(0, m - len(prefer_countries))
        if need_more > 0 and pop_countries:
            extra = _weighted_unique_sample(pop_countries, pop_weights, need_more, exclude=set(prefer_countries))
            prefer_countries.extend(extra)

        # 4.9.3.3 풀 부족 시 백업
        if len(prefer_countries) < m:
            pool_backup = [c for c in uniq_countries if c and c not in prefer_countries]
            random.shuffle(pool_backup)
            prefer_countries.extend(pool_backup[: (m - len(prefer_countries))])

        # 4.9.4 회피 국가 (있어도 되고 없어도 됨, 선호와 중복 금지)
        avoid_countries = []
        if pop_countries and random.random() < args.p_avoid_country:
            # 4.9.4.1 회피 후보는 선호에 없는 것만
            avoid_pool = [c for c in pop_countries if c not in prefer_countries]
            if avoid_pool:
                aw = np.array([country_counts[c] for c in avoid_pool], dtype=float)
                if aw.sum() <= 0:
                    aw = np.ones_like(aw) / len(avoid_pool)
                else:
                    aw = aw / aw.sum()
                avoid_countries = [np.random.choice(avoid_pool, p=aw)]

        # 4.9.5 품질 기준
        min_rating  = random.choice([0.0, 4.0, 4.2])
        min_reviews = random.choice([0, 5, 10])

        # 4.9.6 모험성
        adventurous = random.choice([0.2, 0.5, 0.8])

        # 4.9.7 레코드
        user_full = {
            "user_id": uid,
            "terms": terms,
            "avoid_terms": [],
            "preferred_styles": preferred_styles,
            "avoid_styles": avoid_styles,
            "prefer_countries": prefer_countries,
            "avoid_countries": avoid_countries,
            "prefer_regions": [],
            "avoid_regions": [],
            "prefer_wineries": [],
            "avoid_wineries": [],
            "min_rating": min_rating,
            "min_reviews": min_reviews,
            "adventurous": adventurous
        }

        # 4.9.8 미니멀 스키마
        if args.minimal:
            user_full = {
                "user_id": uid,
                "terms": terms,
                "preferred_styles": preferred_styles,
                "avoid_styles": avoid_styles,
                "prefer_countries": prefer_countries,
                "avoid_countries": avoid_countries,
                "min_reviews": min_reviews,
                "min_rating": min_rating
            }

        users.append(user_full)

    # 4.10 저장
    out = args.out
    with open(out, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    print(f"[USERS] generated {len(users)} → {out}")

# 5. 엔트리
if __name__ == "__main__":
    main()
