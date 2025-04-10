import os
import torch
import requests # Keep for LoRA download
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
import traceback

# Use FluxPipeline for simplified loading
from diffusers import FluxPipeline

# Use the cache directory defined in Dockerfile ENV VARS for consistency at runtime
# MODEL_CACHE = os.environ.get("DIFFUSERS_CACHE", "/diffusers-cache")
# Use bfloat16 if supported (Ampere+ GPUs), fall back to float16 if issues occur
TORCH_DTYPE = torch.bfloat16

# Hardcode the model ID
MODEL_REPO_ID = "black-forest-labs/FLUX.1-schnell"

class Predictor:
    ''' Predictor class for Flux, using FluxPipeline '''

    def __init__(self): # No model_tag needed
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_loaded = False
        self.loaded_lora_path = None

    def setup(self):
        ''' Load the Flux model into memory using AutoPipeline '''
        print(f"Loading pipeline for: {MODEL_REPO_ID}...")
        print(f"Using device: {self.device}")
        print(f"Using dtype: {TORCH_DTYPE}")

        local_files_only = self.should_use_local_files()
        print(f"Loading from local files only: {local_files_only}")

        try:
            # --- Load the pipeline using AutoPipeline ---
            # This automatically loads the correct pipeline class (FluxSchnellPipeline)
            # and its necessary components (UNet, VAE, Text Encoders, Tokenizers)
            # from the cache populated during build.
            self.pipe = FluxPipeline.from_pretrained(
                MODEL_REPO_ID,
                torch_dtype=TORCH_DTYPE,
                # cache_dir=MODEL_CACHE,
                local_files_only=local_files_only,
            )
            self.pipe.to(self.device)
            print("Pipeline components loaded via AutoPipeline.")

            # Optional: Enable memory-efficient attention if xformers installed
            try:
                 if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                     self.pipe.enable_xformers_memory_efficient_attention()
                     print("Enabled xformers memory efficient attention (if available).")
            except Exception as e:
                print(f"Could not enable xformers: {e}")

            print("Pipeline setup complete.")

        except Exception as e:
            print(f"Error loading pipeline: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model {MODEL_REPO_ID}") from e

    def should_use_local_files(self):
        """ Check env var to force local files only after build """
        return os.environ.get("RUNPOD_USE_LOCAL_FILES", "false").lower() == "true"

    def load_lora(self, lora_path: str):
        """ Loads LoRA weights into the pipeline """
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized.")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at {lora_path}")

        print(f"Loading LoRA weights from: {lora_path}...")
        try:
            self.unload_lora()
            # Use load_lora_weights directly on the pipeline loaded by AutoPipeline
            self.pipe.load_lora_weights(lora_path)
            self.lora_loaded = True
            self.loaded_lora_path = lora_path
            print("LoRA weights loaded.")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            self.lora_loaded = False
            self.loaded_lora_path = None
            raise

    def unload_lora(self):
        """ Unloads any currently loaded LoRA weights """
        if self.lora_loaded and self.pipe:
            # AutoPipeline returns the actual pipeline (e.g., FluxSchnellPipeline)
            # so unload_lora_weights should exist if supported by that pipeline.
            if hasattr(self.pipe, "unload_lora_weights"):
                print(f"Unloading LoRA weights ({self.loaded_lora_path})...")
                try:
                    self.pipe.unload_lora_weights()
                    self.lora_loaded = False
                    self.loaded_lora_path = None
                    print("LoRA weights unloaded.")
                except Exception as e:
                    print(f"Error unloading LoRA weights: {e}")
            else:
                print("Warning: Loaded pipeline does not support unload_lora_weights.")
                # Still reset flags
                self.lora_loaded = False
                self.loaded_lora_path = None


    @torch.inference_mode()
    def predict(self,
                prompt: str,
                negative_prompt: Optional[str] = None,
                width: int = 1024,
                height: int = 1024,
                num_inference_steps: int = 4,
                guidance_scale: float = 0.0,
                num_outputs: int = 1,
                seed: Optional[int] = None,
                lora_scale: float = 0.8,
                **kwargs) -> List[Tuple[str, int]]:
        ''' Run prediction using the loaded AutoPipeline '''
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        if seed is None:
            generator_seed = torch.Generator(device=self.device).seed()
        else:
            generator_seed = seed
        print(f"Initial seed: {generator_seed}")

        generators = [torch.Generator(device=self.device).manual_seed(generator_seed + i) for i in range(num_outputs)]
        output_paths = []
        print(f"Generating {num_outputs} images for prompt...")

        for i in range(num_outputs):
            current_seed = generator_seed + i
            print(f"  Generating image {i+1}/{num_outputs} with seed {current_seed}...")

            # Use lora_scale with cross_attention_kwargs if LoRA is loaded
            cross_attention_kwargs = {"scale": lora_scale} if self.lora_loaded else None

            try:
                # The call looks the same, AutoPipeline delegates to the underlying pipeline
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generators[i],
                    output_type="pil",
                    cross_attention_kwargs=cross_attention_kwargs,
                    # Pass other specific kwargs if needed by the underlying pipeline
                    # e.g., aesthetic_score=kwargs.get('aesthetic_score')
                )

                if not output.images:
                    print(f"Warning: No image generated for output {i+1}.")
                    continue

                image = output.images[0]
                os.makedirs("/tmp", exist_ok=True)
                output_path = f"/tmp/out-{i}-{current_seed}.png"
                image.save(output_path)
                output_paths.append((output_path, current_seed))
                print(f"  Saved image {i+1} to {output_path}")

            except Exception as e:
                 print(f"Error during generation for image {i+1}: {e}")
                 traceback.print_exc()

        if not output_paths:
            raise Exception("No images were generated successfully for this task.")

        return output_paths