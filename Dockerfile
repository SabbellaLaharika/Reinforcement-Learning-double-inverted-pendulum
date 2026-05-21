# syntax=docker/dockerfile:1
# --- Build Stage ---
FROM python:3.9-slim AS builder

WORKDIR /app

# Install uv directly from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create a virtual environment (Best Practice)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV UV_HTTP_TIMEOUT=300

COPY requirements.txt .

# Use BuildKit caching for ultra-fast rebuilds AND the index strategy fix
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    torch==2.2.1+cpu \
    -r requirements.txt

# Aggressively prune the virtual environment to minimize size
RUN find /opt/venv -name "tests" -type d -exec rm -rf {} + && \
    find /opt/venv -name "test" -type d -exec rm -rf {} + && \
    rm -rf /opt/venv/lib/python3.9/site-packages/cv2/qt && \
    rm -rf /opt/venv/lib/python3.9/site-packages/ale_py/roms && \
    rm -rf /opt/venv/lib/python3.9/site-packages/torch/include && \
    rm -rf /opt/venv/lib/python3.9/site-packages/torch/share && \
    find /opt/venv/lib/python3.9/site-packages/torch/lib -name "*.a" -delete

# --- Production Stage ---
FROM python:3.9-slim
WORKDIR /app

# Install ONLY runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 \
    libxext6 \
    libx11-6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the fully built and PRUNED virtual environment
COPY --from=builder /opt/venv /opt/venv

# Activate the venv and add python optimizations
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY . .

RUN mkdir -p models logs media
