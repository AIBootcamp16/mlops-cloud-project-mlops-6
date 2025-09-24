# mlops-cloud-project-mlops-6

Minimal **MLOps-ready scaffold** for a **content-based wine recommender** using FastAPI + Docker.

---

## 📂 Layout



```
mlops-cloud-project-mlops-6/
├─ src/ # Code (packages below)
│ ├─ io_pkg/ # I/O adapters: external systems (HTTP/files/S3)
│ ├─ pipelines/ # Orchestration/entrypoints for each step (CLI)
│ ├─ validate.py # Schema & basic sanity checks for snapshots
│ └─ reco/ # Pure recommendation logic (no I/O)
├─ configs/ # Policies/config (e.g., users.json, gates.yaml) — tracked
├─ data/ # External raw snapshots (timestamped) — gitignored
├─ artifacts/ # Model/embeddings/index artifacts — gitignored
├─ tests/ # Unit/integration tests
└─ .gitignore
```


- **src/io_pkg**: HTTP clients, local/S3 storage helpers. No ML logic.
- **src/pipelines**: Thin CLIs that wire modules in order. No heavy algorithms.
- **src/reco**: Pure functions only (deterministic, testable). No network/files.
- **configs**: Team-reviewed config; safe to commit.
- **data/artifacts**: Large/ephemeral; excluded from Git.

---

## 🚀 Quickstart (Local Python)

```bash
# 가상환경 생성 (선택)
python -m venv .venv && source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 1. 데이터 스냅샷
python -m src.pipelines.snapshot --styles reds,whites,sparkling,rose,port

# 2. 임베딩 학습 (예: reds)
python -m src.pipelines.embed_fit --style reds --outdir artifacts

# 3. 유닛 테스트 실행
pytest -q


## Run with Docker
# 빌드
docker build -t wine-reco:latest .

# 실행
docker run -d -p 8000:8000 --name wine-reco-api wine-reco:latest

