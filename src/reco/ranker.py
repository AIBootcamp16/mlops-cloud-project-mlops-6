"""
ranker.py (PURE)
- Re-rank with user preferences (style/region) and adventurousness 'a'.
"""
from __future__ import annotations
from typing import List, Dict

def rerank(
    ids: List[int],
    sims: List[float],
    item_styles: List[str],
    item_countries: List[str],
    user_profile: Dict
) -> List[int]:
    a = float(user_profile.get("adventurousness", 0.5))
    pref_styles   = user_profile.get("pref_styles",   {})
    pref_regions  = user_profile.get("pref_regions",  {})

    scored = []
    for id_, sim, st, co in zip(ids, sims, item_styles, item_countries):
        cat = float(pref_styles.get(st, 0.0))
        reg = float(pref_regions.get(co, 0.0))
        score = (
            0.55 * sim
          + (0.35*(1-a) + 0.05*a) * cat
          + (0.20*(1-a))          * reg
        )
        if a > 0.6 and cat == 0.0:
            score += 0.05 * a
        scored.append((score, id_))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [id_ for score, id_ in scored]
