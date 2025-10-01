"""
select_best.py
- W&B에서 기록된 run들 중 지정 metric 기준으로 최적 run 선택
- summary 타입이 깨진 경우를 대비해 history로 안전 폴백

사용 예:
  python -m src.pipelines.select_best --project wine-reco --entity hwanseok0629- --metric avg_hit@k --maximize
"""

from __future__ import annotations
import argparse, json
from collections.abc import Mapping
import wandb

def _is_num(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def _get_metric_from_summary(run, metric: str):
    """summary가 dict/HTTPSummary/str(JSON) 등 어떤 형태든 최대한 숫자 반환"""
    try:
        s = run.summary  # HTTPSummary | dict | str | 기타
        # 1) dict/매핑이면 바로 접근
        if isinstance(s, Mapping):
            v = s.get(metric, None)
            if _is_num(v):
                return float(v)
        # 2) HTTPSummary 내부 원시 딕셔너리 접근 시도
        raw = getattr(s, "_json_dict", None)
        if raw is not None:
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = None
            if isinstance(raw, Mapping):
                v = raw.get(metric, None)
                if _is_num(v):
                    return float(v)
        # 3) s 자체가 문자열(JSON)인 경우
        if isinstance(s, str):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, Mapping):
                    v = parsed.get(metric, None)
                    if _is_num(v):
                        return float(v)
            except Exception:
                pass
    except Exception:
        pass
    return None

def _get_metric_from_history(run, metric: str):
    """history 스캔해서 해당 metric의 '마지막 값'을 반환"""
    try:
        last_val = None
        # scan_history는 메모리 안전: 페이지네이션으로 dict들을 Yield
        for row in run.scan_history(keys=[metric]):
            v = row.get(metric)
            if _is_num(v):
                last_val = float(v)
        return last_val
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="W&B project name")
    ap.add_argument("--entity", required=True, help="W&B entity (username/team)")
    ap.add_argument("--metric", required=True, help="Target metric (e.g., avg_hit@k)")
    ap.add_argument("--maximize", action="store_true", help="클수록 좋으면 지정")
    ap.add_argument("--state", default="finished", help="run 상태 필터 (finished|running|crashed|... or all)")
    ap.add_argument("--download", action="store_true", help="최적 run의 첫 artifact 다운로드")
    args = ap.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    runs = api.runs(path)

    best_run = None
    best_val = None

    for run in runs:
        # 상태 필터 (선택)
        if args.state != "all":
            if getattr(run, "state", None) != args.state:
                continue

        # 1) summary 우선
        val = _get_metric_from_summary(run, args.metric)

        # 2) summary 실패 시 history 폴백
        if val is None:
            val = _get_metric_from_history(run, args.metric)

        if val is None:
            print(f"[WARN] {run.name} ({run.id}) → '{args.metric}' 없음/파싱 실패")
            continue

        # 최적값 갱신
        if best_val is None:
            best_val, best_run = val, run
        else:
            if (args.maximize and val > best_val) or ((not args.maximize) and val < best_val):
                best_val, best_run = val, run

    if not best_run:
        print("[ERROR] 적합한 run을 찾지 못했습니다.")
        return

    print(f"[BEST] name={best_run.name} id={best_run.id} {args.metric}={best_val}")
    print(f"[URL]  {best_run.url}")

    # 필요 시 artifact 다운로드
    if args.download:
        arts = list(best_run.logged_artifacts())
        if not arts:
            print("[INFO] 이 run에는 logged artifact가 없습니다.")
            return
        art = arts[0]
        target = art.download(root="artifacts/best")
        print(f"[DOWNLOAD] {art.name} → {target}")

if __name__ == "__main__":
    main()
