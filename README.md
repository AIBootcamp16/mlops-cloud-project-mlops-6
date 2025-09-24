# 🎧 와인 추천 시스템  

## 💻 프로젝트 소개  
**Wine Recommendation System**  


---

## ✨ Features  


---

## 👥 팀 구성원  

| <img src="https://via.placeholder.com/180x180.png?text=+" width="180"> | <img src="https://via.placeholder.com/180x180.png?text=+" width="180"> | <img src="https://via.placeholder.com/180x180.png?text=+" width="180"> | <img src="https://via.placeholder.com/180x180.png?text=+" width="180"> | <img src="https://via.placeholder.com/180x180.png?text=+" width="180"> |
|:---:|:---:|:---:|:---:|:---:|
| [**최현**](#) | [**임환석**](#) | [**김종범**](#) | [**윤소영**](#) | [**권효주**](#) |
| 팀장, 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 |



---


## 🔨 개발 환경 및 기술 스택  
- **언어**: Python  
- **버전 및 이슈 관리**: GitHub  
- **협업 툴**: GitHub  
- **주요 라이브러리**: FastAPI, Streamlit, Scikit-learn, Spotipy, Pandas, NumPy  

---



## 📁 프로젝트 구조  

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

---


## 💻 구현 기능  




---

## 🛠️ 작품 아키텍처 (선택 사항)  
*예시 이미지 – 실제 아키텍처 다이어그램 추가 예정*  

---



## 🚨 트러블 슈팅  
1. **OOO 에러 발견**  
   - 설명: 프로젝트 진행 중 발생한 문제 상황  
   - 해결: 해결 방법 및 적용한 코드/설정


    
## 📌 프로젝트 회고  
- 🤔 배운 점  
- 🚀 개선할 점  
- 🙌 협업에서 느낀 점  

---



## 📰 참고자료  
















---

# mlops-cloud-project-mlops-6

Minimal **MLOps-ready scaffold** for a **content-based wine recommender** using FastAPI + Docker.

---

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

### 1. Build (GitHub Actions 자동 빌드 → DockerHub Push)
이미지가 자동으로 DockerHub(`hwan00/wine-reco`)에 올라갑니다.

### 2. Run on EC2
EC2 서버에서 **모델 아티팩트(`artifacts/`)**와 **스냅샷 데이터(`data/snapshots/`)**를 마운트해서 실행하세요:

```bash
docker rm -f wine-reco-api

docker run -d --name wine-reco-api \
  -p 8000:8000 \
  -v /home/ubuntu/mlops-cloud-project-mlops-6/artifacts:/app/artifacts \
  -v /home/ubuntu/mlops-cloud-project-mlops-6/data/snapshots:/app/data/snapshots \
  hwan00/wine-reco:latest
