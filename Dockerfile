FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=8002

WORKDIR /app

# 1. 패키지 설치 후 즉시 캐시 및 불필요한 파일 삭제
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        ffmpeg \
        espeak-ng \
        libespeak-ng1 \
        libsndfile1 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev && \
    # [정리] 설치 직후 패키지 리스트 삭제 (레이어 용량 축소)
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 2. 가상환경 생성 및 pip 업그레이드
RUN python3.11 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt

# 3. PyTorch 및 라이브러리 설치 (캐시 미사용 강제)
# --no-cache-dir 옵션을 사용하여 빌드 중 임시 파일이 쌓이는 것을 방지합니다.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 \
        torchaudio==2.11.0 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    # [정리] pip 캐시 디렉토리 다시 한번 강제 삭제
    rm -rf /root/.cache/pip

COPY . /app

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--proxy-headers"]