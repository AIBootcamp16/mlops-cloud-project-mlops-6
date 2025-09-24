"""
corpus.py (PURE)
- Compose weighted text per item: wine×6 + winery×3 + location/country×1
"""
from __future__ import annotations
from typing import List
import pandas as pd

def build_corpus(df: pd.DataFrame) -> List[str]:
    def row_text(r):
        wine   = (r.get("wine") or "")
        winery = (r.get("winery") or "")
        loc    = (r.get("location") or r.get("country") or "")
        return (f"{wine} " * 6 + f"{winery} " * 3 + f"{loc} ").strip().lower()
    return df.apply(row_text, axis=1).tolist()
