"""
validate.py
- Load latest snapshot → schema validate (pydantic) → minimal cleanup.
- Output: clean pandas.DataFrame ready for reco module.
- 실행 코드
    python -c "from src.validate import load_latest_frame_with_stats as f; import pprint; _,s=f('reds'); pprint.pprint(s)"
"""
# 1. 최신 스냅샷 로드 → 스키마 검증 → DataFrame 반환(+통계)
from __future__ import annotations
import glob, json, re
from typing import Optional, Literal, Tuple, Dict
import pandas as pd
from pydantic import BaseModel, HttpUrl, ValidationError

# 2. 평점/리뷰 스키마
class Rating(BaseModel):
    average: Optional[float] = None
    reviews: Optional[str]   = None

# 3. 와인 스키마(스타일 필수: reds/whites/…)
class Wine(BaseModel):
    id: int
    wine: str
    winery: Optional[str] = None
    location: Optional[str] = None
    image: Optional[HttpUrl] = None
    rating: Optional[Rating] = None
    style: Literal["reds","whites","sparkling","rose","dessert","port"]

# 4. location → country 추출(첫 토큰)
def _country_from_location(loc: Optional[str]) -> Optional[str]:
    if not loc: return None
    return re.split(r"[·|-|\n]", loc)[0].strip()

# 5. 최신 스냅샷 파일 경로
def _latest(style: str) -> str:
    dirs = sorted(glob.glob("data/snapshots/*"))
    if not dirs:
        raise FileNotFoundError("No snapshots. Run snapshot first.")
    return f"{dirs[-1]}/wines_{style}.json"

# 6. 검증 실행(통계 함께 반환)
def load_latest_frame_with_stats(style: str = "reds") -> tuple[pd.DataFrame, Dict[str,int|str]]:
    # 7. 원본 로드
    json_path = _latest(style)
    raw = json.load(open(json_path, "r", encoding="utf-8"))
    # 8. 초기 통계
    raw_total = len(raw)
    ok_rows, bad_log = [], []
    # 9. 레코드 루프
    for d in raw:
        d = dict(d)
        d["style"] = style  # 10. 스타일 필드 보강(엔드포인트 의미를 명시)
        try:
            v = Wine(**d)
            row = v.model_dump()
            row["country"] = _country_from_location(row.get("location"))
            ok_rows.append(row)
        except ValidationError as e:
            bad_log.append({"error": str(e), "item": d})
    # 11. 불량 로그 저장(있으면)
    if bad_log:
        with open(json_path.replace(".json","_bad.json"), "w", encoding="utf-8") as f:
            json.dump(bad_log, f, ensure_ascii=False, indent=2)
    # 12. DataFrame
    df = pd.DataFrame(ok_rows)
    dup = 0
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"], keep="last")
        dup = before - len(df)
    # 13. 요약 통계
    stats = {
        "style": style,
        "snapshot_path": json_path,
        "raw_total": raw_total,
        "validated_ok": len(ok_rows),
        "invalid_bad": len(bad_log),
        "duplicates_removed": dup,
        "final_rows": len(df)
    }
    # 14. 반환
    return df.reset_index(drop=True), stats

# 15. 호환 함수(예전 코드와 동일한 시그니처)
def load_latest_frame(style: str = "reds") -> pd.DataFrame:
    df, _ = load_latest_frame_with_stats(style)
    return df
