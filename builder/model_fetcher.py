'''
RunPod | serverless-flux-template | model_fetcher.py

Downloads the Flux model from HuggingFace.
'''

import os
import shutil
import argparse
from pathlib import Path
from urllib.parse import urlparse
import traceback
import torch

# Import Flux pipeline and necessary transformers components
from diffusers import FluxPipeline
from transformers import AutoTokenizer, CLIPTextModelWithProjection

# Define the model cache directory
MODEL_CACHE_DIR = "diffusers-cache"
# Recommended dtype for Flux download/cache (often stored in fp16 or bf16)
TORCH_DTYPE_DOWNLOAD = torch.float16 # Or float16

def download_model(model_repo_id: str):
    '''
    Downloads the Flux model and associated components from HuggingFace.
    '''
    model_cache_path = Path(MODEL_CACHE_DIR)
    if model_cache_path.exists():
        print(f"Cache directory {MODEL_CACHE_DIR} exists. Removing it.")
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Created cache directory: {model_cache_path}")

    print(f"Downloading Flux model: {model_repo_id}...")
    print(f"Using dtype for download: {TORCH_DTYPE_DOWNLOAD}")

    try:
        # 1. Download the main Flux pipeline
        # This will also download the UNet, VAE, and potentially schedulers
        FluxPipeline.from_pretrained(
            model_repo_id,
            torch_dtype=TORCH_DTYPE_DOWNLOAD,
            cache_dir=model_cache_path,
            # variant="fp16" # Specify variant if needed (e.g., "fp16" or "bf16")
        )
        print("Downloaded Flux pipeline components.")

        # 2. Download the required text encoders and tokenizers separately
        # These are often specified in the Flux model card
        # Example using CLIP-L for base Flux models
        text_encoder_id = "openai/clip-vit-large-patch14"
        # text_encoder_id = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" # Alternative if needed

        print(f"Downloading text encoder/tokenizer: {text_encoder_id}...")
        CLIPTextModelWithProjection.from_pretrained(
            text_encoder_id,
            torch_dtype=TORCH_DTYPE_DOWNLOAD,
            cache_dir=model_cache_path,
        )
        AutoTokenizer.from_pretrained(
            text_encoder_id,
            cache_dir=model_cache_path,
        )
        print("Downloaded text encoder and tokenizer.")

        print("Model download process complete.")
        # Set environment variable to signal that local files should be used on run
        os.environ["RUNPOD_USE_LOCAL_FILES"] = "true"


    except Exception as e:
        print(f"Error during model download: {e}")
        traceback.print_exc()
        # Exit with error to fail the build if download fails
        exit(1)


# --- Parse Arguments ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # Default to the Flux Schnell base model ID
    parser.add_argument(
        "--model_repo_id", type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="Hugging Face repository ID of the Flux model (e.g., 'black-forest-labs/FLUX.1-schnell')."
    )
    args = parser.parse_args()

    # Simple validation: ensure it looks like a repo ID
    if "/" not in args.model_repo_id or len(args.model_repo_id.split('/')) != 2:
         print(f"Error: --model_repo_id '{args.model_repo_id}' does not look like a valid Hugging Face repo ID (e.g., 'org/model').")
         exit(1)

    download_model(args.model_repo_id)
    print("Model fetching script finished.")