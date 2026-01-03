FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with same UID/GID as host user (typically 1000)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -u ${UID} -g ${GID} -m appuser

# Install requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY prompt_enhancer.py .
COPY workflow_api.json .
COPY workflow_video_i2v.json .
COPY workflow_character.json .

# Copy static files
COPY static/ ./static/

# Create directories for downloads/sessions and set ownership
RUN mkdir -p /tmp/zimage_downloads /tmp/zimage_sessions && \
    chown -R appuser:appuser /tmp/zimage_downloads /tmp/zimage_sessions /app

# Switch to non-root user
USER appuser

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
