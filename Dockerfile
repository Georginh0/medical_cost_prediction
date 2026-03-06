# ═══════════════════════════════════════════════════════════════════
# Dockerfile — Medical Cost Prediction
# ═══════════════════════════════════════════════════════════════════
# Build:  docker build -t medical-cost-predictor .
# Run:    docker run -p 8000:8000 medical-cost-predictor
# UI:     http://localhost:8000
# ═══════════════════════════════════════════════════════════════════

FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        pandas==2.0.3 \
        numpy==1.24.4 \
        scikit-learn==1.3.2 \
        scipy==1.10.1 \
        matplotlib==3.7.5 \
        seaborn==0.12.2 \
        mlflow==2.8.1 \
        joblib==1.3.2 \
        fastapi==0.103.2 \
        uvicorn==0.23.2 \
        pydantic==1.10.13 \
        typing_extensions

# ── Copy project files ─────────────────────────────────────────────
COPY setup.py .
COPY src/       src/
COPY steps/     steps/
COPY pipelines/ pipelines/
COPY app/       app/
COPY models/    models/

# ── Register local packages (makes 'import steps' work) ───────────
RUN pip install -e .

# ── Expose port ────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start server ───────────────────────────────────────────────────
# PYTHONPATH ensures 'app' package is importable
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
