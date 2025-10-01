# 1. 멀티 스타일 스냅샷 파이프라인 (단일 --style도 호환)
#    - 스타일별 API 결과를 타임스탬프 폴더로 저장(JSON/CSV).
#    - 예)
#      python -m src.pipelines.snapshot --styles reds,whites,sparkling,rose,port
#      python -m src.pipelines.snapshot --style reds

from __future__ import annotations
# 2. 표준/로컬 임포트
import argparse
from src.io.wines_api import fetch_style              # 3. 외부 API 호출
from src.io.storage   import timestamp_dir, save_snapshot  # 4. 스냅샷 디렉터리/파일 저장

# 5. 핵심 실행: 스타일 리스트 순회 저장
def run(styles: list[str]) -> None:
    out = timestamp_dir()                            # 6. 타임스탬프 폴더 생성
    print(f"[SNAPSHOT] dir={out}")
    for s in styles:                                 # 7. 스타일별 API 호출 → 저장
        items = fetch_style(s)
        save_snapshot(items, s, out)
        print(f"  - {s:<10} count={len(items):<4} -> wines_{s}.json")
    print("[SNAPSHOT] done.")                        # 8. 완료 로그

# 9. CLI 엔트리
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch wine lists by style and snapshot to data/snapshots")
    ap.add_argument("--styles", default="", help="쉼표구분: reds,whites,sparkling,rose,port")
    ap.add_argument("--style",  default="", help="단일 스타일(호환용). 예: reds")
    args = ap.parse_args()

    # 10. 인자 해석: --styles 우선, 없으면 --style(기본 reds)
    if args.styles.strip():
        styles = [x.strip() for x in args.styles.split(",") if x.strip()]
    else:
        styles = [args.style.strip() or "reds"]

    # 11. 실행
    run(styles)
