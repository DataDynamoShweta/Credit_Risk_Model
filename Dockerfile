FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY api ./api
COPY pipelines ./pipelines
RUN pip install --no-cache-dir .
COPY reports ./reports
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
