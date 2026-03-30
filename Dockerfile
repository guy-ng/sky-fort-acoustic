# Stage 1: Build React frontend
FROM node:22-slim AS frontend-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# Stage 2: Python runtime with built frontend
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2-dev libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY --from=frontend-build /web/dist /app/web/dist

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "acoustic.main:app", "--host", "0.0.0.0", "--port", "8000"]
