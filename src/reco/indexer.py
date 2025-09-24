"""
indexer.py (PURE)
- Cosine kNN over TF-IDF matrix.
"""
from __future__ import annotations
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

def topk_cosine(X: "sparse.csr_matrix", q_vec: "sparse.csr_matrix", k: int = 10) -> tuple[list[int], list[float]]:
    nn = NearestNeighbors(metric="cosine", n_neighbors=k)
    nn.fit(X)
    dist, idx = nn.kneighbors(q_vec, return_distance=True)
    return idx[0].tolist(), (1.0 - dist[0]).tolist()
