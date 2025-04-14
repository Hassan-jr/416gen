'''
RunPod | serverless-flux-template | model_fetcher.py

Downloads the Flux model from HuggingFace using FluxPipeline.
'''
import os
import shutil
import argparse # Keep argparse import even if not used below, for consistency? Or remove. Let's remove.
# import argparse
from pathlib import Path
import traceback
import torch

# Import AutoPipeline class
from diffusers import FluxPipeline

# Define the model cache directory (should match Dockerfile ENV)
# MODEL_CACHE_DIR = "/diffusers-cache"
# Use bfloat16 if supported, float16 otherwise
TORCH_DTYPE_DOWNLOAD = torch.bfloat16

# Hardcode the model ID
MODEL_REPO_ID = "black-forest-labs/FLUX.1-dev"

def download_model():
    '''
    Downloads the specified Flux model pipeline using FluxPipeline.
    This handles downloading all necessary components (UNet, VAE, text encoders, tokenizers).
    '''
    # model_cache_path = Path(MODEL_CACHE_DIR)
    # model_cache_path.mkdir(parents=True, exist_ok=True)
    # print(f"Ensured cache directory exists: {model_cache_path}")

    print(f"Downloading Flux model pipeline: {MODEL_REPO_ID}...")
    print(f"Using dtype for download: {TORCH_DTYPE_DOWNLOAD}")

    try:
        # Use FluxPipeline to download everything needed
        _ = FluxPipeline.from_pretrained(
            MODEL_REPO_ID,
            torch_dtype=TORCH_DTYPE_DOWNLOAD,
            # cache_dir=model_cache_path,
        )
        print(f"Model components for {MODEL_REPO_ID} downloaded successfully.")

    except Exception as e:
        print(f"Error during model download: {e}")
        traceback.print_exc()
        exit(1) # Fail the build

# --- Main Execution ---
if __name__ == "__main__":
    # No argument parsing needed as model ID is hardcoded
    print("Starting model fetching process...")
    download_model()
    print("Model fetching script finished successfully.")