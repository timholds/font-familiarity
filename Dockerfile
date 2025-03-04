FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (for PyTorch and other ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    libpng-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY frontend_requirements.txt .
RUN pip install --no-cache-dir -r frontend_requirements.txt

# Copy application code
COPY frontend_app.py .
COPY templates/ ./templates/
COPY static/ ./static/
COPY ml/ ./ml/

# Create directories for model files (will be mounted as volumes)
RUN mkdir -p /app/model

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=frontend_app.py

# Command to run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "30", "--log-level", "info", "frontend_app:create_app('/app/model/fontCNN_BS64-ED512-IC32.pt', '/app/model', '/app/model/class_embeddings_512.npy')"]