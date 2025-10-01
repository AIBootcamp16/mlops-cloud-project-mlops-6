"""
storage.py
- Snapshot utilities: timestamped folder, JSON+CSV dump.
- Snapshot = "data git tag" for reproducibility/audit/drift.
"""
from __future__ import annotations
import datetime as dt, json, os
from typing import List, Dict
import pandas as pd

def timestamp_dir(base: str = "data/snapshots") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = f"{base}/{ts}"
    os.makedirs(out, exist_ok=True)
    return out

def save_snapshot(items: List[Dict], style: str, out_dir: str) -> None:
    json_path = f"{out_dir}/wines_{style}.json"
    csv_path  = f"{out_dir}/wines_{style}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    pd.json_normalize(items).to_csv(csv_path, index=False)
