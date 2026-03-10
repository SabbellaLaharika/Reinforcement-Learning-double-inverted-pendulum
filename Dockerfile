FROM python:3.9-slim

# Install system dependencies for pygame and pymunk
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libxext-dev \
    libx11-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for models, logs, and media
RUN mkdir -p models logs media

ENV PYTHONUNBUFFERED=1
