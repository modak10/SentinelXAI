# SentinelXAI — production image (Phase 7).
# One image, two roles (api / dashboard) selected via compose command override.
FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, configs, scripts (model/data artifacts are mounted at runtime).
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY docs ./docs
COPY pyproject.toml README.md ./

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    ENV=production \
    LOG_LEVEL=INFO

EXPOSE 8000 8501

# Default: API. Overridden by docker-compose for the dashboard service.
CMD ["python", "-m", "uvicorn", "sentinelxai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
