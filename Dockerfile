FROM python:3.10-slim

# 환경 변수
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    WANDB_PROJECT=wine-reco \
    WANDB_ENTITY=default_user\
    PYTHONIOENCODING=utf-8 

# 작업 디렉토리
WORKDIR /app

# 빌드 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 복사
COPY . .

# FastAPI 실행 (gunicorn + uvicorn)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
