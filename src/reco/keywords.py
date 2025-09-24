# src/reco/keywords.py
# 1. 텍스트/키워드 정규화와 국가 캐노니컬을 한 곳에서 처리한다.

from __future__ import annotations
import re, unicodedata
from typing import Iterable, List, Set

# 2. 악센트 제거  ex) 'réserve' -> 'reserve', 'rosé' -> 'rose'
def strip_accents(text: str) -> str:
    if not text:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

# 3. 토큰 정규화  소문자 → 악센트 제거 → 비문자 제거 → 공백 정리
def norm_token(t: str) -> str:
    t = strip_accents((t or "").lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# 4. 프레이즈  공백 포함 표현은 먼저 보존한다
PHRASES: List[str] = [
    "blanc de blancs","gran reserva","vieilles vignes",
    "cote du rhone","cotes du rhone","new zealand","south africa",
    "pedro ximenez","domaines ott","jacques selosse"
]

# 5. 허용 단어  품종·스타일·산지·대표 하우스
ALLOW: Set[str] = {
    "merlot","pinot","noir","chardonnay","riesling","sauvignon","cabernet","syrah",
    "nebbiolo","sangiovese","tempranillo","barbera","grenache","garnacha","mourvedre",
    "zinfandel","malbec","chenin","viognier","semillon","vermentino","albarino","muscat",
    "champagne","prosecco","cava","sherry","tawny","vintage","rose",
    "bordeaux","burgundy","bourgogne","loire","rhone","alsace","provence","languedoc",
    "tuscany","piemonte","piedmont","veneto","sicily","rioja","priorat","ribera","toro",
    "douro","alentejo","mendoza","barossa","mosel","pfalz","napa","sonoma",
    "krug","selosse","bollinger","moet","veuve","aubert","kistler"
}

# 6. 제거 단어  관사·일반어·모호어
BAN: Set[str] = {
    "zan","lot","medium","botella","tour","del","traditional","nature","ambassadeur",
    "peninsula","eastern","petite"
}

# 7. 불용어  스타일명은 게이트에서 처리하므로 제거
STOP: Set[str] = {
    "wine","nv","reserve","grand","valley","estate","vineyard","cellars","winery",
    "domaine","cuvee","blanc","rouge","des","de","du","la","le","les","et",
    "red","white","sparkling","rose","port"
}

# 8. 국가 캐노니컬과 국가 세트
_COUNTRY_ALIAS = {
    "usa":"united states","u.s.a.":"united states","u.s.":"united states","us":"united states",
    "england":"united kingdom","uk":"united kingdom","u.k.":"united kingdom",
    "korea":"south korea","republic of korea":"south korea"
}
COUNTRIES = {
    "united states","france","italy","spain","portugal","germany","austria","switzerland",
    "argentina","chile","australia","new zealand","south africa","united kingdom"
}

def canon_country(s: str) -> str:
    s = norm_token(s)
    return _COUNTRY_ALIAS.get(s, s)

def is_country(tok: str) -> bool:
    return canon_country(tok) in COUNTRIES

# 9. 텍스트에서 의미 토큰 추출  프레이즈 보존 → 단일 토큰 필터
def extract_terms_from_text(text: str, max_terms: int = 8, keep_countries: bool = False) -> List[str]:
    t = norm_token(text)
    if not t:
        return []
    found: List[str] = []
    tmp = t
    for ph in PHRASES:
        pp = norm_token(ph)
        if pp and pp in tmp:
            if keep_countries or not is_country(pp):
                found.append(pp)
            tmp = tmp.replace(pp, " ")
    words = [w for w in tmp.split() if len(w) >= 3 and w not in STOP]
    return clean_terms(found + words, max_terms=max_terms, keep_countries=keep_countries)

# 10. 키워드 정제  입력 리스트에 대해 중복 제거·프레이즈 우선·허용만 유지
def clean_terms(raw: Iterable[str], max_terms: int = 6, keep_countries: bool = False) -> List[str]:
    toks: List[str] = []
    for t in (raw or []):
        nt = norm_token(t)
        if nt:
            toks.append(nt)
    toks = list(dict.fromkeys(toks))
    kept: List[str] = []
    joined = " ".join(toks)
    for ph in PHRASES:
        pp = norm_token(ph)
        if pp and pp in joined:
            if keep_countries or not is_country(pp):
                kept.append(pp)
    used = set(w for ph in kept for w in ph.split())
    for t in toks:
        if t in BAN or t in STOP:
            continue
        if t in used:
            continue
        if not keep_countries and is_country(t):
            continue
        if t in ALLOW:
            kept.append(t)
    if not kept:
        kept = [t for t in toks if len(t) >= 4 and t not in BAN and t not in STOP and (keep_countries or not is_country(t))]
    return kept[:max_terms]

# 11. 텍스트 배열에서 빈도 카운트용 토큰 제너레이터
def iter_clean_terms_from_rows(rows: Iterable[str]) -> Iterable[str]:
    for s in rows:
        for t in extract_terms_from_text(s):
            yield t

# 12. 간단 매칭 헬퍼  추천 카드 텍스트 vs terms
def text_has_any_terms(text: str, terms: Iterable[str]) -> bool:
    txt = norm_token(text)
    for t in (terms or []):
        nt = norm_token(t)
        if nt and nt in txt:
            return True
    return False
