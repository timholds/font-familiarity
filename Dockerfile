# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    libpng-dev \
    curl \
    git \
    libgl1-mesa-glx \  
    libglib2.0-0 \      
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --no-cache-dir numpy==1.26.4  # Install NumPy first
COPY frontend_requirements.txt .
RUN pip install --no-cache-dir -r frontend_requirements.txt
RUN git clone https://github.com/timholds/CRAFT-text-detection.git /tmp/CRAFT \
    && cd /tmp/CRAFT \
    && pip install .

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libpng16-16 \
    curl \
    libgl1-mesa-glx \ 
    libglib2.0-0 \  
    libsm6 \
    libxext6 \
    libxrender1 \   
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /tmp/CRAFT /tmp/CRAFT
# Also ensure Python can find it:
ENV PYTHONPATH="${PYTHONPATH}:/tmp"


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
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "30", "--log-level", "info", "frontend_app:create_app('/app/model/fontCNN_BS64-ED512-IC32.pt', '/app/model', '/app/model/class_embeddings_512.npy')"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "30", "--log-level", "info", "frontend_app:create_app('/app/model/fontCNN_BS64-ED1024-IC16.pt', '/app/model', '/app/model/class_embeddings_1024.npy', 'label_mapping.npy', 'True')"]