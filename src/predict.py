# # predict.py (Refactored based on documentation example)
# import os
# import torch
# from PIL import Image
# from typing import List, Tuple, Optional
# import traceback
# import uuid # Using uuid for unique temporary file names
# import gc

# # --- Diffusers Imports ---
# # Using FluxPipeline as AutoPipeline/FluxSchnellPipeline caused issues previously for the user
# from diffusers import FluxPipeline

# # --- Configuration ---
# # Use bfloat16 as per example, requires Ampere+ GPU usually
# TORCH_DTYPE = torch.bfloat16
# MODEL_REPO_ID = "black-forest-labs/FLUX.1-schnell"
# # Use the cache directory defined in Dockerfile ENV VARS or fall back to default Hugging Face cache
# MODEL_CACHE = os.environ.get("DIFFUSERS_CACHE", None) # None uses default ~/.cache/huggingface

# class Predictor:
#     '''
#     Predictor class for Flux, using FluxPipeline based on documentation examples.
#     Handles dynamic LoRA loading and multi-output generation.
#     '''
#     def __init__(self):
#         """Initializes the Predictor."""
#         self.pipe: Optional[FluxPipeline] = None # Type hint for clarity
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.lora_loaded = False
#         self.loaded_lora_path = None
#         print(f"Predictor initialized on device: {self.device}")

#     def setup(self):
#         '''
#         Loads the Flux model pipeline into memory.
#         '''
#         if self.pipe is not None:
#             print("Pipeline already loaded.")
#             return

#         print(f"Loading pipeline: {MODEL_REPO_ID}...")
#         print(f"Using device: {self.device}")
#         print(f"Using dtype: {TORCH_DTYPE}")
#         print(f"Using cache dir: {MODEL_CACHE if MODEL_CACHE else 'Default (~/.cache/huggingface)'}")

#         try:
#             # Load the entire pipeline at once
#             self.pipe = FluxPipeline.from_pretrained(
#                 MODEL_REPO_ID,
#                 torch_dtype=TORCH_DTYPE,
#                 cache_dir=MODEL_CACHE,
#                 # Set local_files_only based on ENV var if running in the built container context
#                 local_files_only=os.environ.get("RUNPOD_USE_LOCAL_FILES", "false").lower() == "true",
#             )

#             # Use CPU offloading for memory management if CUDA is available
#             if self.device == "cuda":
#                 print("Enabling model CPU offload...")
#                 self.pipe.enable_model_cpu_offload()
#             else:
#                 print("CUDA not available, cannot enable CPU offload.")
#                 self.pipe.to(self.device) # Move to CPU if needed

#             # Optional: Enable xformers if installed and supported
#             try:
#                  if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
#                      self.pipe.enable_xformers_memory_efficient_attention()
#                      print("Enabled xformers memory efficient attention (if available).")
#             except Exception as e:
#                 print(f"Could not enable xformers: {e}")


#             print("Pipeline setup complete.")

#         except Exception as e:
#             self.pipe = None # Ensure pipe is None if setup fails
#             print(f"Error loading pipeline: {e}")
#             traceback.print_exc()
#             raise RuntimeError(f"Failed to load model {MODEL_REPO_ID}") from e

#     def load_lora(self, lora_path: str):
#         """ Loads LoRA weights using the standard diffusers method. """
#         if not self.pipe:
#             raise RuntimeError("Pipeline not initialized. Call setup() first.")
#         if not os.path.exists(lora_path):
#             raise FileNotFoundError(f"LoRA file not found at {lora_path}")

#         print(f"Loading LoRA weights from: {lora_path}...")
#         try:
#             self.unload_lora() # Unload previous if any

#             # Load weights into the main pipeline
#             self.pipe.load_lora_weights(lora_path)
#             # Note: We don't explicitly fuse here, scale is applied at inference

#             self.lora_loaded = True
#             self.loaded_lora_path = lora_path
#             print("LoRA weights loaded into pipeline.")
#         except Exception as e:
#             print(f"Error loading LoRA weights: {e}")
#             traceback.print_exc()
#             self.lora_loaded = False
#             self.loaded_lora_path = None
#             raise # Reraise to signal failure

#     def unload_lora(self):
#         """ Unloads LoRA weights using the standard diffusers method. """
#         if self.lora_loaded and self.pipe:
#             if hasattr(self.pipe, "unload_lora_weights"):
#                 print(f"Unloading LoRA weights ({self.loaded_lora_path})...")
#                 try:
#                     self.pipe.unload_lora_weights()
#                     self.lora_loaded = False
#                     self.loaded_lora_path = None
#                     print("LoRA weights unloaded.")
#                 except Exception as e:
#                     print(f"Error unloading LoRA weights: {e}")
#                     # Reset state even if unload fails
#                     self.lora_loaded = False
#                     self.loaded_lora_path = None
#             else:
#                 print("Warning: Loaded pipeline does not support unload_lora_weights.")
#                 self.lora_loaded = False
#                 self.loaded_lora_path = None

#     def _flush_gpu_memory(self):
#         """ Utility to manually clear GPU memory if needed (less necessary with CPU offload). """
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     @torch.inference_mode()
#     def predict(self,
#                 prompt: str,
#                 negative_prompt: Optional[str] = None, # Added back negative prompt
#                 width: int = 1024,
#                 height: int = 1024,
#                 num_inference_steps: int = 4, # Use Flux defaults (often low)
#                 guidance_scale: float = 0.0,  # Use Flux defaults (often 0)
#                 num_outputs: int = 1,
#                 seed: Optional[int] = None,
#                 lora_scale: float = 0.8,
#                 max_sequence_length: int = 512, # Common parameter for text encoders
#                 **kwargs) -> List[Tuple[str, int]]:
#         '''
#         Runs prediction using the loaded Flux pipeline.
#         '''
#         if not self.pipe:
#              # Attempt setup if called directly without prior setup
#              print("Pipeline not initialized. Attempting setup...")
#              self.setup()
#              if not self.pipe:
#                   raise RuntimeError("Pipeline setup failed.")

#         if seed is None:
#             base_seed = torch.Generator(device="cpu").seed() # Generate seed on CPU
#         else:
#             base_seed = seed
#         print(f"Initial seed for task: {base_seed}")

#         # Create generators on CPU, they will be moved to device by the pipeline if needed
#         generators = [torch.Generator(device="cpu").manual_seed(base_seed + i) for i in range(num_outputs)]

#         output_paths = []
#         print(f"Generating {num_outputs} images for prompt...")

#         for i in range(num_outputs):
#             current_seed = base_seed + i
#             print(f"  Generating image {i+1}/{num_outputs} with seed {current_seed}...")

#             # Re-introducing cross_attention_kwargs for standard LoRA scaling
#             cross_attention_kwargs = {"scale": lora_scale} if self.lora_loaded else None
#             if self.lora_loaded:
#                 print(f"  Applying LoRA scale: {lora_scale}")

#             try:
#                 # Call the unified pipeline directly
#                 # Pass None for negative_prompt if empty string or None
#                 neg_prompt = negative_prompt if negative_prompt else None

#                 output = self.pipe(
#                     prompt=prompt,
#                     negative_prompt=neg_prompt,
#                     width=width,
#                     height=height,
#                     num_inference_steps=num_inference_steps,
#                     guidance_scale=guidance_scale,
#                     generator=generators[i], # Pass the specific generator
#                     output_type="pil",
#                     max_sequence_length=max_sequence_length, # Pass sequence length
#                     # cross_attention_kwargs=cross_attention_kwargs # Pass LoRA scale info
#                 )

#                 if not output.images:
#                     print(f"Warning: No image generated for output {i+1}.")
#                     continue

#                 image = output.images[0]
#                 os.makedirs("/tmp", exist_ok=True)
#                 # Unique file name using UUID to prevent collisions
#                 output_path = f"/tmp/output-{uuid.uuid4()}.png"
#                 image.save(output_path)
#                 output_paths.append((output_path, current_seed))
#                 print(f"  Saved image {i+1} to {output_path}")

#             except Exception as e:
#                  print(f"Error during generation for image {i+1} (Seed: {current_seed}): {e}")
#                  # Check specifically for the cross_attention_kwargs error again
#                  if isinstance(e, TypeError) and 'cross_attention_kwargs' in str(e):
#                       print("***********************************************************")
#                       print("Failed due to 'cross_attention_kwargs'.")
#                       print("This pipeline version might not support LoRA scaling via this argument.")
#                       print("Consider removing 'cross_attention_kwargs' from the pipe() call.")
#                       print("***********************************************************")
#                  traceback.print_exc()
#                  # Continue to next image or raise? Let's continue for now.

#             finally:
#                  # Attempt to clear memory after each generation if using offload/GPU
#                  if self.device == "cuda":
#                       self._flush_gpu_memory()


#         if not output_paths:
#             # If the loop finishes without generating any images (e.g., all failed)
#             raise Exception("No images were generated successfully for this task.")

#         return output_paths

# predict.py
import os
import torch
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
import traceback
import time
import gc

# --- Dependency Checks ---
try:
    import peft
    print(f"Successfully imported 'peft' library. Version: {peft.__version__}")
    PEFT_AVAILABLE = True
except ImportError:
    print("WARNING: Failed to import the 'peft' library.")
    print("         LoRA loading will be DISABLED.")
    PEFT_AVAILABLE = False

# --- Diffusers Import ---
from diffusers import FluxPipeline

# --- Configuration ---
TORCH_DTYPE = torch.float16 # Standard choice for compatibility and performance
SCHNELL_REPO_ID = "black-forest-labs/FLUX.1-schnell"
SCHNELL_REVISION = "refs/pr/1" # Or "main"
MIN_INFERENCE_STEPS = 4 # Schnell is designed for few steps

class Predictor:
    '''
    Predictor class for Flux.1-schnell using the full Diffusers pipeline.
    This sacrifices sequential loading VRAM optimization for correctness.
    '''

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[FluxPipeline] = None
        self.active_lora_adapter_name = "flux_lora_adapter"
        self.lora_loaded = False # Tracks if LoRA is currently fused/active
        print(f"Predictor initialized. Target Device: {self.device}, Base Dtype: {TORCH_DTYPE}")
        if not PEFT_AVAILABLE: print("PEFT not found, LoRA functionality disabled.")
        if self.device == "cpu": print("Warning: CUDA not available, running on CPU.")

    def _get_gpu_memory_usage(self) -> str:
        """ Helper to get formatted GPU memory usage """
        if not torch.cuda.is_available(): return "N/A (CPU)"
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        # Peak memory tracking can be useful for debugging OOMs
        # peak_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        # peak_reserved = torch.cuda.max_memory_reserved(self.device) / 1e9
        # return f"Alloc={allocated:.2f}GB, Res={reserved:.2f}GB | PeakAlloc={peak_allocated:.2f}GB, PeakRes={peak_reserved:.2f}GB"
        return f"Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB"


    def _flush_gpu_memory(self):
        """ Utility function to clear GPU memory """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # torch.cuda.reset_peak_memory_stats() # Optional reset if tracking peak memory

    def setup(self):
        ''' Load the full Flux pipeline. '''
        if self.pipe:
            print("Predictor already set up.")
            return

        print("Setting up predictor (loading FULL Flux pipeline)...")
        print(f"GPU Memory Before Load: {self._get_gpu_memory_usage()}")
        start_time = time.time()
        local_files_only = os.environ.get("RUNPOD_USE_LOCAL_FILES", "false").lower() == "true"
        print(f"Using local files only: {local_files_only}")

        try:
            # Load the entire pipeline onto the specified device and dtype
            self.pipe = FluxPipeline.from_pretrained(
                SCHNELL_REPO_ID,
                revision=SCHNELL_REVISION,
                torch_dtype=TORCH_DTYPE,
                # Quantization Notes (Requires bitsandbytes, adds complexity):
                # If VRAM is an issue on 24GB card, consider uncommenting ONE of these:
                # load_in_8bit=True,
                # load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, # 4-bit needs compute dtype
                local_files_only=local_files_only
            )
            self.pipe.to(self.device)

            # Optional: Compilation (Requires PyTorch >= 2.0)
            # Can speed up inference after the first run, but increases startup time/memory slightly.
            # Consider enabling if doing many inferences per job.
            # try:
            #     print("Attempting to compile pipeline components...")
            #     # Compile the most compute-intensive parts
            #     self.pipe.transformer = torch.compile(self.pipe.transformer, mode="reduce-overhead", fullgraph=True)
            #     self.pipe.vae.decoder = torch.compile(self.pipe.vae.decoder, mode="reduce-overhead", fullgraph=True)
            #     print("Compilation successful (or skipped).")
            # except Exception as compile_err:
            #     print(f"Warning: Compilation failed - {compile_err}. Continuing without compilation.")


            print(f"Full Flux pipeline loaded to {self.device} in {time.time() - start_time:.2f} seconds.")
            print(f"GPU Memory After Load: {self._get_gpu_memory_usage()}")

        except Exception as e:
            print(f"FATAL Error during predictor setup (loading full pipeline): {e}")
            traceback.print_exc()
            self.pipe = None # Ensure pipe is None if setup fails
            # Attempt cleanup before raising
            self._flush_gpu_memory()
            raise RuntimeError("Failed to setup predictor (full pipeline load)") from e

    def _load_lora(self, lora_path: str, lora_scale: float):
        """Loads and fuses LoRA weights using pipeline methods."""
        if not self.pipe:
            print("Error: Pipeline not loaded, cannot load LoRA.")
            return False
        if not PEFT_AVAILABLE:
            print("Warning: PEFT not available, skipping LoRA load.")
            return False
        if self.lora_loaded:
            print("Warning: Another LoRA is already loaded. Unloading previous one first.")
            if not self._unload_lora(): # Attempt to unload previous
                print("Error: Failed to unload previous LoRA. Cannot load new LoRA.")
                return False

        print(f"Loading LoRA from {os.path.basename(lora_path)} and fusing with scale {lora_scale}...")
        try:
            # 1. Load the LoRA weights into the pipeline with a specific adapter name.
            self.pipe.load_lora_weights(lora_path, adapter_name=self.active_lora_adapter_name)
            print("LoRA weights loaded.")

            # 2. Fuse the loaded LoRA weights into the base model layers with the specified scale.
            # This modifies the underlying model weights. `unfuse_lora` is needed to restore originals.
            # Ensure your diffusers version supports specifying adapter_names here.
            # If not, it might fuse *all* loaded adapters.
            self.pipe.fuse_lora(lora_scale=lora_scale, adapter_names=[self.active_lora_adapter_name])
            # If adapter_names isn't supported: self.pipe.fuse_lora(lora_scale=lora_scale)

            self.lora_loaded = True # Mark LoRA as active/fused
            print("LoRA weights fused successfully.")
            print(f"GPU Memory After LoRA Load/Fuse: {self._get_gpu_memory_usage()}")
            return True
        except Exception as e:
            print(f"ERROR applying/fusing LoRA via pipeline: {e}")
            traceback.print_exc()
            self.lora_loaded = False
            # Attempt to clean up potentially partially loaded state
            try:
                 self.pipe.unfuse_lora() # Try unfusing even if fuse failed midway
                 if hasattr(self.pipe, "delete_lora_weights"):
                      self.pipe.delete_lora_weights(self.active_lora_adapter_name)
            except Exception:
                 print("Exception during LoRA error cleanup.")
            return False

    def _unload_lora(self):
        """Unfuses LoRA weights using pipeline methods."""
        if not self.pipe or not self.lora_loaded:
            return False # Nothing to unload

        print("Unfusing LoRA weights...")
        try:
            # 1. Unfuse LoRA weights, restoring the original model weights.
            self.pipe.unfuse_lora()

            # 2. Optional but recommended: Delete the adapter weights reference
            # Check if method exists in your diffusers version.
            if hasattr(self.pipe, "delete_lora_weights"):
                 self.pipe.delete_lora_weights(adapter_name=self.active_lora_adapter_name)
            elif hasattr(self.pipe, "remove_adapter"): # Some versions might use this
                 self.pipe.remove_adapter(self.active_lora_adapter_name)


            self.lora_loaded = False # Mark LoRA as inactive
            print("LoRA unfused successfully.")
            self._flush_gpu_memory() # Clean up memory after unload
            print(f"GPU Memory After LoRA Unload: {self._get_gpu_memory_usage()}")
            return True
        except Exception as e:
            print(f"Warning: Error unloading/unfusing LoRA: {e}")
            traceback.print_exc()
            # Assume unloaded for safety, though state might be inconsistent
            self.lora_loaded = False
            return False

    @torch.inference_mode()
    def predict(self,
                prompt: str,
                negative_prompt: Optional[str] = None, # Ignored by Schnell
                width: int = 1024,
                height: int = 1024,
                num_inference_steps: int = 4, # Schnell's intended range
                guidance_scale: float = 0.0, # Ignored by Schnell
                num_outputs: int = 1,
                seed: Optional[int] = None,
                lora_path: Optional[str] = None, # Passed from handler
                lora_scale: float = 0.8,
                **kwargs) -> List[Tuple[str, int]]:
        ''' Run prediction using the loaded full FluxPipeline. '''

        if not self.pipe:
            raise RuntimeError("Predictor not set up (pipeline not loaded). Call setup() first.")

        # --- Argument Handling & Logging ---
        if seed is None: base_seed = torch.Generator(device='cpu').seed()
        else: base_seed = seed
        print(f"Using base seed: {base_seed} for {num_outputs} outputs.")

        # Schnell ignores these, set appropriate defaults
        guidance_scale = 0.0
        effective_negative_prompt = None # Pass None if ignored
        print(f"Info: FLUX.1-schnell ignores guidance_scale (using 0.0) and negative_prompt.")

        # Use provided steps, ensuring it's at least the minimum
        effective_steps = max(num_inference_steps, MIN_INFERENCE_STEPS)
        if effective_steps != num_inference_steps:
             print(f"Info: Using {effective_steps} inference steps (min {MIN_INFERENCE_STEPS})")

        lora_status = "Not Provided"
        if lora_path:
             if not PEFT_AVAILABLE: lora_status = f"Provided but PEFT missing"
             elif not os.path.exists(lora_path): lora_status = f"Provided but file not found"
             else: lora_status = f"Attempting load ('{os.path.basename(lora_path)}' scale={lora_scale})"

        print(f"\nStarting generation for prompt: '{prompt[:80]}...'")
        print(f"Params: W={width}, H={height}, Steps={effective_steps}, Num={num_outputs}, Seed={base_seed}, LoRA={lora_status}")

        output_paths = []
        overall_start_time = time.time()
        current_lora_applied = False # Track if LoRA applied *in this specific call*

        try:
            # --- LoRA Handling within predict call ---
            # Ensure any LoRA from a *previous* call is unloaded before potentially loading a new one.
            if self.lora_loaded:
                self._unload_lora()

            # Attempt to load LoRA if requested for *this* call
            if lora_path and PEFT_AVAILABLE and os.path.exists(lora_path):
                load_success = self._load_lora(lora_path, lora_scale)
                if load_success:
                    current_lora_applied = True
                else:
                    print("Warning: LoRA loading failed for this task, proceeding without LoRA.")
            # --- End LoRA Handling ---


            # Prepare generator(s) for reproducibility
            generators = [torch.Generator(device=self.device).manual_seed(base_seed + i) for i in range(num_outputs)]

            # --- Generate Images using the Full Pipeline ---
            # The pipeline handles all steps: encoding, denoising, decoding.
            print(f"GPU Memory Before Inference: {self._get_gpu_memory_usage()}")
            pipeline_output = self.pipe(
                prompt=prompt,
                negative_prompt=effective_negative_prompt, # Pass None
                height=height,
                width=width,
                num_inference_steps=effective_steps,
                guidance_scale=guidance_scale, # 0.0
                num_images_per_prompt=num_outputs,
                generator=generators,
                output_type="pil" # Request PIL images directly
            )
            images = pipeline_output.images
            print(f"GPU Memory After Inference: {self._get_gpu_memory_usage()}")


            # --- Save Images ---
            print(f"Generation complete. Saving {len(images)} image(s)...")
            os.makedirs("/tmp", exist_ok=True)
            for i, img in enumerate(images):
                current_seed = base_seed + i
                output_path = f"/tmp/out_{current_seed}_{i}.png"
                try:
                    if img is None: # Handle potential None output from pipeline if errors occurred
                        raise ValueError("Pipeline returned None image object.")
                    img.save(output_path)
                    output_paths.append((output_path, current_seed))
                    print(f"  Saved image {i+1} to {output_path}")
                except Exception as save_err:
                    print(f"    ERROR saving image {i+1} (Seed: {current_seed}): {save_err}")
                    traceback.print_exc()
                    # Add placeholder for failed saves, helps handler know an attempt was made
                    output_paths.append(("/tmp/error_save.png", current_seed))


        except Exception as e:
             print(f"ERROR during pipeline execution or saving: {e}")
             traceback.print_exc()
             # Create placeholders for all expected outputs on pipeline error
             output_paths = [("/tmp/error_pipeline.png", base_seed + i) for i in range(num_outputs)]
        finally:
            # --- Unload LoRA only if it was loaded specifically for *this* predict call ---
            if current_lora_applied:
                self._unload_lora()

            # --- Memory Flush ---
            # Flush memory at the end of the predict call regardless of LoRA
            self._flush_gpu_memory()


        overall_end_time = time.time()
        print(f"\nFinished generation attempt in {overall_end_time - overall_start_time:.2f} seconds.")
        # Filter out error placeholders before returning
        successful_images = [p for p in output_paths if not p[0].startswith("/tmp/error_")]
        num_errors = len(output_paths) - len(successful_images)
        print(f"Successfully generated {len(successful_images)} images out of {num_outputs} requested ({num_errors} save/pipeline errors).")

        if not successful_images and num_outputs > 0:
            print("Warning: No images were successfully generated in this task.")

        return successful_images # Return list of (path, seed) for successful images only