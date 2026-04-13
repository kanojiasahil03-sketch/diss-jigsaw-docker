FROM python:3.11-slim

LABEL maintainer="Sahil Kanojia"
LABEL project="MSc Dissertation — Context-Aware RL Jigsaw Assembly"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

EXPOSE 6006 8888

CMD ["/bin/bash"]
