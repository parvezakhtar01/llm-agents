# Use Python slim image for a smaller footprint
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (better caching)
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY .env .env.example ./

# Clean up cache files
RUN find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find . -type f -name "*.pyc" -delete && \
    find . -type f -name "*.pyo" -delete && \
    find . -type f -name "*.pyd" -delete

# Expose the port
EXPOSE 8000

# Command to run the application
#CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# Set the ENTRYPOINT and CMD for Lambda
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "app.main.handler" ]

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
# Since we copied
CMD ["main.handler"]