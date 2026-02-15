# syntax=docker/dockerfile:1.4

FROM python:3.11-bookworm

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 2. Install PyTorch
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 3. Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy TrackNetV3 (Code + Weights)
COPY TrackNetV3/ /app/TrackNetV3/

# 5. Download Large Models
RUN mkdir -p /app/models /app/TrackNetV3/ckpts \
    && curl -L -o /app/models/slowfast_model.pt https://github.com/bryanlje/mds06-backend/releases/download/v1.0-models/slowfast_model.pt \
    && curl -L -o /app/models/yolo_weights.pt https://github.com/bryanlje/mds06-backend/releases/download/v1.0-models/yolo_weights.pt \
    && curl -L -o /app/models/contact_model.pth https://github.com/bryanlje/mds06-backend/releases/download/v1.0-models/contact_model.pth \
    && curl -L -o /app/models/osnet_x1_0_badminton.pt https://github.com/bryanlje/mds06-backend/releases/download/v1.0-models/osnet_x1_0_badminton.pt

# 6. Copy Refactored Application Code
# We copy the 'app' package and the 'main.py' entry point
COPY app/ /app/app/
COPY main.py .

# 7. Create temp directories
RUN mkdir -p /tmp/uploads /tmp/outputs /tmp/tracknet_outputs && \
    chmod 777 /tmp/uploads /tmp/outputs /tmp/tracknet_outputs

# 8. Set Python Path
ENV PYTHONPATH="/app:/app/TrackNetV3:${PYTHONPATH}"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# 9. Run 'main:app'
CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 0 --max-requests 50 --max-requests-jitter 10 --worker-class sync --access-logfile - --error-logfile - --log-level info main:app"]