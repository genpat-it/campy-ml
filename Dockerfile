FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY campyml_model.py .

# Copy models
COPY models/*.pkl ./models/

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default entrypoint
ENTRYPOINT ["python", "campyml_model.py"]

# Default arguments (show help)
CMD ["--help"]