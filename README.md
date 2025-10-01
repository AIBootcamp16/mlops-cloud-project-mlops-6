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


# mlops-cloud-project-mlops-6

Minimal MLOps-ready scaffold for a **content-based wine recommender**.

## Layout

```
mlops-cloud-project-mlops-6/
├─ src/                # Code (packages below)
│  ├─ io/              # I/O adapters: external systems (HTTP/files/S3)
│  ├─ pipelines/       # Orchestration/entrypoints for each step (CLI)
│  ├─ validate.py      # Schema & basic sanity checks for snapshots
│  └─ reco/            # Pure recommendation logic (no I/O)
├─ configs/            # Policies/config (e.g., users.json, gates.yaml) — tracked
├─ data/               # External raw snapshots (timestamped) — gitignored
├─ artifacts/          # Model/embeddings/index artifacts — gitignored
├─ tests/              # Unit/integration tests
└─ .gitignore
```

- **src/io**: HTTP clients, local/S3 storage helpers. No ML logic.
- **src/pipelines**: Thin CLIs that wire modules in order. No heavy algorithms.
- **src/reco**: Pure functions only (deterministic, testable). No network/files.
- **configs**: Team-reviewed config; safe to commit.
- **data/artifacts**: Large/ephemeral; excluded from Git.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.pipelines.snapshot --style reds        # take snapshot → data/snapshots/<ts>/
python -m src.pipelines.embed_fit --style reds       # fit TF-IDF → artifacts/
pytest -q                                            # run fast unit tests
```

## Evaluation Metrics

본 프로젝트의 추천 품질 평가는 다음 지표로 이루어집니다:

1. Hit@K

사용자의 선호 조건(스타일, 국가, 키워드)에 맞는 와인이 Top-K 추천 안에 최소 1개라도 포함되는지 여부.

적중률 지표 (있으면 1, 없으면 0).

2. Terms Hit@K

사용자가 지정한 키워드(terms) 와 추천 와인 설명(wine, winery, location)의 매칭 비율.

키워드 기반 취향 반영 정도.

3. Country Hit@K

사용자가 선호 국가로 지정한 값과 추천 와인의 국가가 일치하는 비율.

국가 취향 반영 정도.

4. Diversity@K

추천 리스트의 다양성을 측정 (1 - Intra-List Similarity).

값이 높을수록 유사한 와인만 몰리지 않고 다양한 와인을 추천.

📌 정리

Hit@K: 적중률

Terms Hit@K: 키워드 적합도

Country Hit@K: 국가 취향 반영도

Diversity@K: 추천 다양성
```
```
## 실행 순서
git clone https://github.com/<your-repo>/mlops-cloud-project-mlops-6.git

cd mlops-cloud-project-mlops-6

pip install -r requirements.txt

pip install wandb

wandb login <자신의_API_KEY>

WANDB_ENTITY=<팀원계정> uvicorn src.app:app --reload
```
```
## 📌 모델 학습 및 자동화 파이프라인 (W&B 연동)

### 1. 스냅샷 & 임베딩 학습
와인 데이터를 수집하고 스타일별 TF-IDF 모델을 학습합니다. 학습된 모델은 **W&B Artifact**로 자동 업로드됩니다.
```bash
python -m src.pipelines.embed_fit --style reds
python -m src.pipelines.embed_fit --style whites
python -m src.pipelines.embed_fit --style sparkling
```
### 2. 추천 성능 평가 (Eval Report)

사용자 프로필(configs/users.json) 기반으로 추천 품질 지표를 산출하고, 결과를 W&B에 로깅합니다.
python -m src.pipelines.eval_report --users configs/users.json --style reds --k 5
python -m src.pipelines.eval_report --users configs/users.json --style whites --k 5
python -m src.pipelines.eval_report --users configs/users.json --style sparkling --k 5
```
```
### 3. 최적 모델 자동 선택
python -m src.pipelines.select_best \
  --project wine-reco \
  --entity <wandb_username> \
  --metric avg_hit_at_k \
  --maximize
```

```
### 4. 최적 모델 다운로드
wandb artifact get <wandb_username>/wine-reco/tfidf-reds:latest
