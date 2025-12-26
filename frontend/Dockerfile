FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY prompt_enhancer.py .
COPY workflow_api.json .
COPY workflow_video_t2v.json .
COPY workflow_video_i2v.json .

# Copy static files
COPY static/ ./static/

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
