# Use a verified, modern Runpod image with Python 3.11 and a compatible CUDA version
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# ✅ Install system dependencies (git, pkg-config, ffmpeg, build tools)
RUN apt-get update && apt-get install -y \
    git \
    pkg-config \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files
COPY . /app

# ✅ Upgrade pip first
RUN pip install --upgrade pip

# 1. Install the dependencies. This may temporarily break the torch installation.
RUN pip install --no-cache-dir -r requirements.txt

# 2. Force the re-installation of a known-good, CUDA-compatible PyTorch stack.
RUN pip install --no-cache-dir --force-reinstall torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Command to run your handler script when the worker starts
CMD ["python", "handler.py"]
