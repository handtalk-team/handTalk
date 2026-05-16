FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ app/

# ONNX model (optional — if not present, rule-based fallback activates)
COPY ml/models/ ml/models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
