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
COPY app.py .
COPY prompt_enhancer.py .
COPY workflow_api.json .

EXPOSE 7860

CMD ["python", "app.py"]
