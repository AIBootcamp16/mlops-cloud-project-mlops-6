"""
wines_api.py
- HTTP client for SampleAPIs (Wines).
- Responsibility: external I/O only (no ML logic).
"""
from __future__ import annotations
from typing import List, Dict
import requests

API_BASE = "https://api.sampleapis.com/wines"
HEADERS  = {"User-Agent": "mlops-cloud-project-mlops-6/0.1 (+team)"}

def fetch_style(style: str, timeout: int = 20) -> List[Dict]:
    url = f"{API_BASE}/{style}"
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
