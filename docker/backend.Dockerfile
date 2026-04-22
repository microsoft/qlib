FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY backend/requirements.txt .

# Copy qlib source code and setup.py
COPY qlib /app/qlib
COPY setup.py /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Cython for building QLib C extensions
RUN pip install --no-cache-dir cython

# Build QLib C extensions
RUN cd /app && python setup.py build_ext --inplace

# Add qlib to Python path
ENV PYTHONPATH="$PYTHONPATH:/app"

# Set qlib version using environment variable to bypass setuptools-scm
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_QLIB="0.1.0"

# Copy application code
COPY backend/ .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
