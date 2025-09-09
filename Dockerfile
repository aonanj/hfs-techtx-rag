# Use a slim Python image (matches local development Python 3.12 artifacts)
FROM python:3.12-slim AS builder

# Metadata
LABEL maintainer="techtrans <phaethon@phaethon.llc>"

# Environment - Optimized for Cloud Run and Cloud Logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies required by some Python packages (Tesseract for OCR, libs for images/PDFs)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    wget \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (include gunicorn for production)
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    wget \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . .

USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash", "-lc", "exec gunicorn 'app:create_app()' --bind 0.0.0.0:${PORT} --workers 1 --threads 1 --timeout ${GUNICORN_TIMEOUT:-3500} --access-logfile - --error-logfile -"]