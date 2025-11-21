# CMB Phase Transitions Analysis - Docker Container
# ================================================
#
# This Dockerfile creates a containerized environment for running
# the complete CMB phase transition analysis including gamma calculations
# and BAO validation across all datasets.
#
# Build: docker build -t cmb-analysis .
# Run:   docker run -v $(pwd)/results:/app/results cmb-analysis

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Create results directory
RUN mkdir -p results

# Set the default command to run the full analysis
CMD ["python", "main.py", "--gamma", "--bao", "--all-datasets", "--full-validation", "--output-dir", "/app/results"]
