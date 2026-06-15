FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VMB_TRAINING_RUNS_DIR=/app/training_runs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.deploy.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.deploy.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.4.0" "torchvision>=0.19.0"

COPY app ./app
RUN mkdir -p /app/data /app/training_runs

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
