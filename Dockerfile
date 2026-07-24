FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies
RUN uv sync

# Copy the remaining project files
COPY . .

CMD ["/bin/bash"]