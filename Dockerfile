# Match this to the Python version you used locally (printed by src/print_versions.py).
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# LightGBM needs this system library, and the slim image doesn't include it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first: if only your code changes, Docker reuses the
# cached (slow) package-install step and rebuilds in seconds.
COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what the API needs at runtime
COPY src ./src
COPY api ./api
COPY models ./models
COPY outputs/policy_config.json ./outputs/policy_config.json
COPY data/processed/feature_cols.json ./data/processed/feature_cols.json

# Run as a normal user instead of root (safer if the service is ever compromised)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Docker will mark the container "unhealthy" if /health stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# 0.0.0.0 means "accept connections from outside the container"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]