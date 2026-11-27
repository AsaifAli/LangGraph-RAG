FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer libreoffice-impress curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN adduser --system --group --home /home/app --no-create-home app \
    && mkdir -p /app/chat_history /app/.cache/huggingface /home/app \
    && chown -R app:app /app/chat_history /app/.cache /home/app

ENV HF_HOME=/app/.cache/huggingface
ENV HOME=/home/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8000}/ready || exit 1

USER app
CMD ["sh", "-c", "uvicorn app.api_main:app --host 0.0.0.0 --port ${PORT:-8000}"]
