# syntax=docker/dockerfile:1
# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

# The article's trick: copy uv directly from its official image! No curl or apt-get needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file
COPY requirements.txt .

# Use uv pip with caching enabled for subsequent ultra-fast builds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    torch==2.2.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Production stage - pure runtime
FROM python:3.9-slim
WORKDIR /app

# Install ONLY runtime dependencies for pygame and xvfb
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 \
    libxext6 \
    libx11-6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the fully built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the venv and add python optimizations
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy the application code last (this layer changes most frequently)
COPY . .

# Create mandatory output directories
RUN mkdir -p models logs media
