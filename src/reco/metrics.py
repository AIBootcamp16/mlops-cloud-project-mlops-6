"""
metrics.py (PURE)
- NDCG@K, RegionMatch@K.
"""
from __future__ import annotations
from typing import List, Set
import math

def ndcg_at_k(gains: List[int], k: int = 10) -> float:
    G = gains[:k]
    def dcg(gs): return sum((2**g - 1) / math.log2(i+2) for i, g in enumerate(gs))
    ideal = sorted(G, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(G) / idcg

def region_match_at_k(countries: List[str], preferred: Set[str], k: int = 10) -> float:
    top = countries[:k]
    if not top:
        return 0.0
    match = sum(1 for c in top if c in preferred)
    return match / len(top)
