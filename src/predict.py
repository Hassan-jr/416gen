import os
import torch
import requests
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
import traceback

# Use FluxPipeline
from diffusers import FluxPipeline # Add ControlNet if needed later
# Schedulers are usually handled internally by Flux pipelines, but import if needed
# from diffusers import EulerDiscreteScheduler # Example

# Import transformers components used by Flux
from transformers import AutoTokenizer, CLIPTextModelWithProjection

MODEL_CACHE = "diffusers-cache"
# Recommended dtype for Flux
TORCH_DTYPE = torch.float16 # or torch.float16 if bfloat16 not supported/problematic

class Predictor:
    ''' Predictor class for Flux.1 Schnell '''

    def __init__(self, model_tag="black-forest-labs/FLUX.1-schnell"):
        self.model_tag = model_tag
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lora_loaded = False
        self.loaded_lora_path = None

    def setup(self):
        ''' Load the Flux model into memory '''
        print(f"Loading Flux pipeline: {self.model_tag}...")
        print(f"Using device: {self.device}")
        print(f"Using dtype: {TORCH_DTYPE}")

        try:
            # FLUX.1 requires specific text encoder setup
            # Pre-load text encoders expected by Flux pipelines
            text_encoder = CLIPTextModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14", # Use the one specified by Flux model card
                # "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", # Alternative if needed
                torch_dtype=TORCH_DTYPE,
                cache_dir=MODEL_CACHE,
                local_files_only=self.should_use_local_files(),
            ).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(
                 "openai/clip-vit-large-patch14",
                # "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                cache_dir=MODEL_CACHE,
                local_files_only=self.should_use_local_files(),
            )


            self.pipe = FluxPipeline.from_pretrained(
                self.model_tag,
                # Pass the pre-loaded components if required by the specific diffusers version
                # text_encoder=text_encoder,
                # tokenizer=tokenizer,
                torch_dtype=TORCH_DTYPE,
                cache_dir=MODEL_CACHE,
                local_files_only=self.should_use_local_files(),
                # variant="fp16" # or "bf16" if available and preferred
            )
            # Check if text_encoder and tokenizer are correctly loaded by the pipeline
            # If not, assign the pre-loaded ones:
            if not hasattr(self.pipe, 'text_encoder') or self.pipe.text_encoder is None:
                 self.pipe.text_encoder = text_encoder
            if not hasattr(self.pipe, 'tokenizer') or self.pipe.tokenizer is None:
                 self.pipe.tokenizer = tokenizer

            self.pipe.to(self.device)

            # Optional: Enable memory-efficient attention if supported and beneficial for Flux
            # Needs xformers installed
            try:
                 self.pipe.enable_xformers_memory_efficient_attention()
                 print("Enabled xformers memory efficient attention.")
            except Exception as e:
                print(f"Could not enable xformers (might not be installed or compatible): {e}")

            print("Flux pipeline loaded successfully.")

        except Exception as e:
            print(f"Error loading Flux pipeline: {e}")
            traceback.print_exc()
            raise # Reraise the exception to prevent worker start if model fails

    def should_use_local_files(self):
        """ Check env var to force local files only after build """
        return os.environ.get("RUNPOD_USE_LOCAL_FILES", "false").lower() == "true"


    def load_lora(self, lora_path: str):
        """ Loads LoRA weights into the pipeline """
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at {lora_path}")

        print(f"Loading LoRA weights from: {lora_path}...")
        try:
            # Unload previous LoRA if one was loaded
            self.unload_lora()

            # Load the new LoRA
            self.pipe.load_lora_weights(lora_path) # Use the directory or file path directly
            # Note: Some diffusers versions might expect the directory containing the LoRA files
            # Adjust if necessary based on how you save/download the LoRA.

            self.lora_loaded = True
            self.loaded_lora_path = lora_path
            print("LoRA weights loaded.")

            # Fuse LoRA weights for potential performance improvement (optional)
            # self.pipe.fuse_lora()
            # print("Fused LoRA weights.")

        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            traceback.print_exc()
            self.lora_loaded = False # Ensure state is correct on failure
            self.loaded_lora_path = None
            raise # Reraise to signal failure

    def unload_lora(self):
        """ Unloads any currently loaded LoRA weights """
        if self.lora_loaded and self.pipe:
            print(f"Unloading LoRA weights ({self.loaded_lora_path})...")
            try:
                # Unfuse first if fused (optional, depends if you fused)
                # self.pipe.unfuse_lora()
                # print("Unfused LoRA weights.")

                self.pipe.unload_lora_weights()
                self.lora_loaded = False
                self.loaded_lora_path = None
                print("LoRA weights unloaded.")
            except Exception as e:
                print(f"Error unloading LoRA weights: {e}")
                # Don't raise here, unloading failure is less critical

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
        ''' Run prediction with Flux '''
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        if seed is None:
            generator_seed = torch.Generator(device=self.device).seed()
        else:
            generator_seed = seed

        # Create a generator for each output image to ensure distinct results when num_outputs > 1
        generators = [torch.Generator(device=self.device).manual_seed(generator_seed + i) for i in range(num_outputs)]

        output_paths = []

        # Flux pipelines might directly support `num_images_per_prompt`
        # Check the specific pipeline documentation. If it does, use it directly.
        # If not, loop as follows:

        print(f"Generating {num_outputs} images for prompt...")
        for i in range(num_outputs):
            current_seed = generator_seed + i
            print(f"  Generating image {i+1}/{num_outputs} with seed {current_seed}...")

            # Set cross_attention_kwargs for LoRA scale *only* if LoRA is loaded
            cross_attention_kwargs = {"scale": lora_scale} if self.lora_loaded else None

            try:
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generators[i], # Use the specific generator for this image
                    output_type="pil", # Get PIL images
                    cross_attention_kwargs=cross_attention_kwargs, # Apply LoRA scale
                    # Add any other relevant Flux parameters from kwargs here
                    # Example: aesthetic_score=kwargs.get('aesthetic_score', 6.0)
                )

                # Flux output format might vary, check documentation. Assuming output.images
                if not output.images:
                    print(f"Warning: No image generated for output {i+1}.")
                    continue

                # Save the image (assuming one image per call in the loop)
                image = output.images[0]
                output_path = f"/tmp/out-{i}-{current_seed}.png"
                image.save(output_path)
                output_paths.append((output_path, current_seed)) # Return path and seed used
                print(f"  Saved image {i+1} to {output_path}")

            except Exception as e:
                 print(f"Error during generation for image {i+1}: {e}")
                 traceback.print_exc()
                 # Optionally add an error marker or skip, depending on desired behavior


        if not output_paths:
            raise Exception("No images were generated successfully.")

        return output_paths