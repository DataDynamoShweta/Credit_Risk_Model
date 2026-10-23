FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    ENVIRONMENT=container

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY api ./api
COPY pipelines ./pipelines
COPY scripts ./scripts
COPY governance ./governance
COPY reports ./reports

# Demo artefact: train on the canonical synthetic data during the build so the image can
# serve out of the box. Mount a different artefact at /app/artifacts (and set MODEL_SHA256)
# to serve your own model. The audit trail written here is discarded with the build.
RUN python pipelines/train.py \
    && rm -rf governance/audit_trail/*/*.jsonl \
    && chown -R appuser:appuser /app/governance /app/artifacts /app/reports

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json,urllib.request,sys; h=json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=4)); sys.exit(0 if h.get('model_loaded') else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
