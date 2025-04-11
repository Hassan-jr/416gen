# predict.py
import os
import torch
import requests # Keep for LoRA download in handler
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
import traceback
import time # For timing generations

# Use FluxPipeline for simplified loading
from diffusers import FluxPipeline, DPMSolverMultistepScheduler # Import scheduler if needed explicitly

# Use the cache directory defined in Dockerfile ENV VARS for consistency at runtime
# MODEL_CACHE = os.environ.get("DIFFUSERS_CACHE", "/diffusers-cache")
# Use bfloat16 for Ampere+ GPUs (L4) for better performance & memory
TORCH_DTYPE = torch.bfloat16

# Hardcode the model ID (or get from ENV)
MODEL_REPO_ID = "black-forest-labs/FLUX.1-schnell"

class Predictor:
    ''' Predictor class for Flux.1-schnell, using FluxPipeline '''

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_loaded = False
        self.loaded_lora_path = None
        print(f"Predictor initialized. Device: {self.device}, Dtype: {TORCH_DTYPE}")

    def setup(self):
        ''' Load the Flux model into memory using FluxPipeline '''
        if self.pipe is not None:
            print("Pipeline already loaded.")
            return

        print(f"Loading pipeline for: {MODEL_REPO_ID}...")
        start_time = time.time()

        # Check if model files should be loaded from local cache only
        # Useful after the image build has cached the model. Set ENV VAR in Dockerfile.
        local_files_only = os.environ.get("RUNPOD_USE_LOCAL_FILES", "false").lower() == "true"
        print(f"Using local files only: {local_files_only}")

        try:
            # Explicitly load the FluxPipeline with specified dtype
            # If the model was cached during build, this will be fast.
            self.pipe = FluxPipeline.from_pretrained(
                MODEL_REPO_ID,
                torch_dtype=TORCH_DTYPE,
                # cache_dir=MODEL_CACHE, # Only if customizing cache location
                local_files_only=local_files_only,
                # variant="fp16" # Optional: if fp16 variant exists and you prefer it
            )

            # Optional: Configure a scheduler if needed (often default is fine)
            # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

            self.pipe.to(self.device)
            print(f"Pipeline loaded to {self.device} successfully.")

            # Optional: Enable memory-efficient attention if xformers installed
            try:
                # Check if 'enable_xformers_memory_efficient_attention' exists and call it
                if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("Enabled xformers memory efficient attention (if available).")
                # For some pipelines, attention slicing might be an alternative
                # elif hasattr(self.pipe, 'enable_attention_slicing'):
                #     self.pipe.enable_attention_slicing()
                #     print("Enabled attention slicing.")
            except Exception as e:
                print(f"Could not enable memory optimization: {e}")

            end_time = time.time()
            print(f"Pipeline setup complete in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"Error loading pipeline: {e}")
            traceback.print_exc()
            # Clean up partially loaded model?
            self.pipe = None
            raise RuntimeError(f"Failed to load model {MODEL_REPO_ID}") from e


    def load_lora(self, lora_path: str):
        """ Loads LoRA weights into the pipeline's relevant components. """
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized. Cannot load LoRA.")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at {lora_path}")

        # Unload any existing LoRA first
        self.unload_lora()

        print(f"Loading LoRA weights from: {lora_path}...")
        try:
            start_time = time.time()
            # Load LoRA weights into the pipeline. This modifies the UNet/TextEncoders in-place.
            # The `adapter_name` argument is optional but good practice if managing multiple LoRAs.
            self.pipe.load_lora_weights(lora_path, adapter_name="default_lora")
            # Note: Some pipelines might require setting the adapter after loading,
            # but usually load_lora_weights handles activation for the first LoRA.
            # If issues arise: self.pipe.set_adapters(["default_lora"])
            self.lora_loaded = True
            self.loaded_lora_path = lora_path
            end_time = time.time()
            print(f"LoRA weights loaded and activated in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            traceback.print_exc()
            self.lora_loaded = False
            self.loaded_lora_path = None
            # Attempt to clean up potentially corrupted state if possible
            self.unload_lora() # Try unloading again
            raise # Re-raise the exception

    def unload_lora(self):
        """ Unloads any currently loaded LoRA weights """
        if self.lora_loaded and self.pipe:
            print(f"Unloading LoRA weights ({self.loaded_lora_path})...")
            try:
                start_time = time.time()
                # Check if unload_lora_weights exists (it should for modern Diffusers)
                if hasattr(self.pipe, "unload_lora_weights"):
                     # If using adapter names: self.pipe.unload_lora_weights(["default_lora"])
                     self.pipe.unload_lora_weights()
                # Alternative/Older method if unload doesn't exist or work: Reload the original weights?
                # More complex, usually unload_lora_weights is the way.

                # Or potentially disable adapters if unload isn't sufficient
                # if hasattr(self.pipe, "disable_lora"):
                #    self.pipe.disable_lora() # Might be needed depending on Diffusers version / pipe implementation

                self.lora_loaded = False
                self.loaded_lora_path = None
                end_time = time.time()
                print(f"LoRA weights unloaded in {end_time - start_time:.2f} seconds.")
            except Exception as e:
                print(f"Error unloading LoRA weights: {e}")
                # State might be uncertain here, but flags are reset
                self.lora_loaded = False
                self.loaded_lora_path = None
                # Consider raising the error if unloading is critical for correctness
                # raise
        elif not self.pipe:
             print("Cannot unload LoRA, pipeline not initialized.")
        # else: # Not loaded, nothing to do
        #    print("No LoRA currently loaded.")


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
                **kwargs) -> List[Tuple[str, int]]:
        ''' Run prediction using the loaded FluxPipeline '''
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        if seed is None:
            # Generate a random seed if none provided
            generator_seed = torch.Generator(device='cpu').seed() # Generate on CPU for reproducibility if needed later
        else:
            generator_seed = seed
        print(f"Using initial seed: {generator_seed} for {num_outputs} outputs.")

        # Create a generator for each output image, offset by index
        # Using CPU generator initially allows consistent seed reporting even if GPU generator behaves differently
        generators = [torch.Generator(device=self.device).manual_seed(generator_seed + i) for i in range(num_outputs)]
        output_paths = []

        print(f"Generating {num_outputs} images for prompt: '{prompt[:80]}...'")
        generation_start_time = time.time()

        for i in range(num_outputs):
            current_seed = generator_seed + i
            print(f"  Generating image {i+1}/{num_outputs} with seed {current_seed}...")
            iter_start_time = time.time()

            try:
                # Call the pipeline
                # NO cross_attention_kwargs here for FluxPipeline
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generators[i],
                    output_type="pil", # Get PIL images
                    # Any other Flux-specific args can be passed via kwargs if needed
                )

                if not output.images:
                    print(f"Warning: No image generated for output {i+1} (seed {current_seed}).")
                    continue # Skip saving/appending if no image

                # Should only be one image per call in this loop
                image = output.images[0]

                # Ensure temporary directory exists
                os.makedirs("/tmp", exist_ok=True)
                # Include seed in filename for easy identification
                output_path = f"/tmp/out_{current_seed}_{i}.png"
                image.save(output_path)

                iter_end_time = time.time()
                print(f"  Saved image {i+1} to {output_path} (took {iter_end_time - iter_start_time:.2f}s)")
                output_paths.append((output_path, current_seed)) # Append tuple (path, seed used)

            except Exception as e:
                 print(f"Error during generation for image {i+1} (seed {current_seed}): {e}")
                 traceback.print_exc()
                 # Continue to next image if one fails? Or raise immediately?
                 # For now, we just print and continue.

        generation_end_time = time.time()
        total_time = generation_end_time - generation_start_time
        print(f"Finished generating {len(output_paths)} images in {total_time:.2f} seconds.")

        if not output_paths:
            raise Exception("No images were generated successfully for this task.")

        return output_paths # List of (path, seed) tuples