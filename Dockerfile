# Simple Dockerfile for TruthScan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=app.py
ENV PORT=5001

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download required NLTK data and spaCy model
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" && \
    python -c "import spacy.cli; spacy.cli.download('en_core_web_sm')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs results

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/api/health || exit 1

# Run the application
CMD ["python", "app.py"] 