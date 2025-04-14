# Choose a base image compatible with recent torch, diffusers, and CUDA 12.x if needed
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO

# Set HuggingFace cache directory (consistent path)
ENV HF_HOME=/cache/huggingface
ENV HF_HUB_CACHE=/cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/cache/huggingface/hub
# Set cache dir for diffusers models (used by model_fetcher.py and predict.py)
ENV DIFFUSERS_CACHE=/diffusers-cache

# Create cache directory
RUN mkdir -p /cache/huggingface/hub /diffusers-cache

# System packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        vim \
        git \
        wget \
        curl \
        ca-certificates \
        libgoogle-perftools-dev \
        libssl-dev \
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3.10-dev && \
    rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Model ID is hardcoded in scripts, ARG/ENV no longer needed here
# ARG MODEL_REPO_ID="black-forest-labs/FLUX.1-dev"
# ENV MODEL_TAG=${MODEL_REPO_ID}

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt && \
    rm /requirements.txt

# Fetch the model (Model ID is hardcoded in the script)
COPY builder/model_fetcher.py /model_fetcher.py
# Ensure this step has enough RAM allocated in Docker settings!
RUN python /model_fetcher.py # No need to pass model_repo_id argument
# Remove the fetcher script after use
RUN rm /model_fetcher.py

# Set ENV VAR so predictor knows to use local files after build
ENV RUNPOD_USE_LOCAL_FILES="true"

# Copy application source code
COPY src .

# Set the entrypoint command
CMD python -u /rp_handler.py