"""
embed.py (PURE)
- TF-IDF fit/transform interface (no file/HTTP).
"""
from __future__ import annotations
from typing import List
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def fit_tfidf(corpus: List[str], ngram=(1,2), min_df=2) -> tuple[TfidfVectorizer, "sparse.csr_matrix"]:
    vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df)
    X = vec.fit_transform(corpus)
    return vec, X

def transform(vec: TfidfVectorizer, texts: List[str]) -> "sparse.csr_matrix":
    return vec.transform(texts)
