FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY workflow_api.json .

EXPOSE 7860

CMD ["python", "app.py"]
