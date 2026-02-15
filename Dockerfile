# syntax=docker/dockerfile:1.4

FROM python:3.11-bookworm

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY TrackNetV3/ /app/TrackNetV3/

COPY models/ /app/models/

# Copy application code
COPY app.py .

# Create necessary directories
RUN mkdir -p /tmp/uploads /tmp/outputs /tmp/tracknet_outputs && \
    chmod 777 /tmp/uploads /tmp/outputs /tmp/tracknet_outputs

# Set Python path
ENV PYTHONPATH="/app:/app/TrackNetV3:${PYTHONPATH}"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 0 --max-requests 50 --max-requests-jitter 10 --worker-class sync --access-logfile - --error-logfile - --log-level info app:app"]