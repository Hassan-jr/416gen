# Choose a base image compatible with recent torch, diffusers, and CUDA 12.x if needed
# nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 is a good candidate
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=INFO
# Set HuggingFace cache directory (optional, but good practice)
ENV HF_HOME=/cache/huggingface
ENV HF_HUB_CACHE=/cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/cache/huggingface/hub
ENV DIFFUSERS_CACHE=/cache/huggingface/hub

# Create cache directory
RUN mkdir -p /cache/huggingface/hub

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
        # Add libssl-dev needed by some python versions/pip
        libssl-dev \
        # Install Python 3.10 or 3.11 (check compatibility with dependencies)
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3.10-dev && \
    # Link python3.10 to python and python3
    rm -f /usr/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    # Clean up
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Build arguments for the model
# Default to Flux Schnell base model
ARG MODEL_REPO_ID="black-forest-labs/FLUX.1-schnell"
ENV MODEL_TAG=${MODEL_REPO_ID} # Use the repo ID as the tag for consistency

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt && \
    rm /requirements.txt

# Fetch the model (Uses MODEL_REPO_ID passed during build or the default)
COPY builder/model_fetcher.py /model_fetcher.py
# Pass the build argument to the script
RUN python /model_fetcher.py --model_repo_id=${MODEL_REPO_ID}
# Remove the fetcher script after use
RUN rm /model_fetcher.py
# Ensure downloaded models are stored in the designated cache dir within the image
ENV DIFFUSERS_CACHE=/diffusers-cache
ENV TRANSFORMERS_CACHE=/diffusers-cache
ENV HF_HUB_CACHE=/diffusers-cache
# Set ENV VAR so predictor knows to use local files after build
ENV RUNPOD_USE_LOCAL_FILES="true"


# Copy application source code
COPY src .

# Expose port if needed (RunPod serverless doesn't strictly require this)
# EXPOSE 8080

# Set the entrypoint command
# Pass the MODEL_TAG (which is the repo ID) to the handler script
CMD python -u /rp_handler.py