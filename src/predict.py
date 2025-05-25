# # predict.py (Simplified - Mimics Example Script's LoRA Behavior)
# import os
# import torch
# from PIL import Image
# from typing import List, Tuple, Optional, Dict, Any
# import traceback
# import time
# import gc
# import uuid # For unique filenames

# # NOTE: Removed explicit PEFT import/check. Relies on try/except during load.
# # Ensure 'peft' is installed if LoRA is used.

# # --- Diffusers Import ---
# from diffusers import FluxPipeline

# from dotenv import load_dotenv

# # Automatically load the .env file from the current directory
# load_dotenv()

# # --- Configuration ---
# TORCH_DTYPE = torch.bfloat16 # As per example script
# MODEL_REPO_ID = "black-forest-labs/FLUX.1-dev" # As per example script

# class Predictor:
#     '''
#     Simplified Predictor class for Flux.1-dev.
#     Mimics the simple example script's LoRA behavior (load once, persists).
#     Designed for rp_handler.py structure.
#     '''
#     def __init__(self):
#         """Initializes the predictor."""
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.pipe: Optional[FluxPipeline] = None
#         self._lora_loaded_once = False # Tracks if LoRA has been loaded *at all*
#         self._lora_path_used = None   # Store the path of the first loaded LoRA
#         print(f"Predictor initialized. Device: {self.device}")

#     def setup(self):
#         ''' Loads the Flux pipeline model. Expected to be called once. '''
#         if self.pipe:
#             return # Avoid reloading

#         print(f"Loading model: {MODEL_REPO_ID} ({TORCH_DTYPE})...")
#         try:
#             self.pipe = FluxPipeline.from_pretrained(
#                 MODEL_REPO_ID,
#                 torch_dtype=TORCH_DTYPE
#             )
#             self.pipe.to(self.device)
#             print("Model loaded successfully.")
#         except Exception as e:
#             self.pipe = None
#             print(f"FATAL: Model loading failed: {e}")
#             traceback.print_exc()
#             raise RuntimeError(f"Failed to load model '{MODEL_REPO_ID}'") from e

#     # Removed internal _load/_unload helpers for extreme simplicity. Logic is inline.

#     # Keep the full signature for compatibility
#     # Use @torch.inference_mode() for efficiency
#     # Add torch.autocast like the example
#     @torch.inference_mode()
#     def predict(self,
#                 prompt: str,
#                 negative_prompt: Optional[str] = "",
#                 width: int = 1024,
#                 height: int = 1024,
#                 num_inference_steps: int = 30,
#                 guidance_scale: float = 4.0,
#                 num_outputs: int = 1,
#                 seed: Optional[int] = None,
#                 lora_path: Optional[str] = None, # Path provided by handler
#                 lora_scale: float = 0.8,
#                 **kwargs) -> List[Tuple[str, int]]:
#         '''
#         Performs inference. Attempts LoRA load ONLY the first time lora_path is provided.
#         Uses previously loaded LoRA (if any) on subsequent calls, ignoring new lora_path.
#         Returns a list of (local_image_path, seed) tuples for successes.
#         '''
#         if not self.pipe:
#             raise RuntimeError("Predictor is not set up. Call setup() first.")

#         start_time = time.time()

#         # Try to load LoRA only if one hasn't been loaded before AND a path is given now
#         if not self._lora_loaded_once and lora_path:
#             print(f"First time LoRA request. Attempting load: {os.path.basename(lora_path)}")
#             if os.path.exists(lora_path):
#                 try:
#                     # Attempt load (requires 'peft' to be installed)
#                     self.pipe.load_lora_weights(lora_path)
#                     self._lora_loaded_once = True # Mark LoRA as loaded
#                     self._lora_path_used = lora_path # Store which LoRA was loaded
#                     print(f"Successfully loaded LoRA: {os.path.basename(lora_path)}")
#                 except Exception as e:
#                     # This includes errors if 'peft' is missing or load fails
#                     print(f"Error loading LoRA weights from {lora_path}: {e}")
#                     # traceback.print_exc() # Optional debug
#                     # Proceed without LoRA for this and future calls
#             else:
#                 print(f"Warning: LoRA file not found at: {lora_path}. Proceeding without LoRA.")

#         # --- Seed Generation ---
#         if seed is None:
#             base_seed = torch.Generator(device='cpu').seed()
#         else:
#             base_seed = seed
#         generators = [torch.Generator(device=self.device).manual_seed(base_seed + i) for i in range(num_outputs)]

#         # --- Inference ---
#         output_paths = []
#         try:
#             # Using autocast like the example script
#             with torch.autocast(device_type=self.device, dtype=TORCH_DTYPE, enabled=self.device=='cuda'):
#                 pipeline_output = self.pipe(
#                     prompt=prompt,
#                     negative_prompt=negative_prompt if negative_prompt else None,
#                     height=height,
#                     width=width,
#                     num_inference_steps=num_inference_steps,
#                     guidance_scale=guidance_scale,
#                     num_images_per_prompt=num_outputs,
#                     generator=generators,
#                     output_type="pil",
#                 )
#             images = pipeline_output.images

#             # --- Save Images ---
#             os.makedirs("/tmp", exist_ok=True)
#             for i, img in enumerate(images):
#                 current_seed = base_seed + i
#                 output_path = f"/tmp/gen_{current_seed}_{uuid.uuid4()}.png"
#                 try:
#                     if img is None: continue
#                     img.save(output_path)
#                     output_paths.append((output_path, current_seed))
#                 except Exception as save_err:
#                     print(f"Error saving image {i+1} (Seed: {current_seed}): {save_err}")

#         except Exception as predict_err:
#             print(f"Error during pipeline execution: {predict_err}")
#             traceback.print_exc()
#             output_paths = [] # Return empty on error
#         finally:
#             if torch.cuda.is_available():
#                 gc.collect()
#                 torch.cuda.empty_cache()

#         # print(f"Predict call finished in {time.time() - start_time:.2f}s") # Optional debug
#         return output_paths


# predict.py

import os
import torch
from PIL import Image
from typing import List, Tuple, Optional
import traceback
import time
import gc
import uuid

from diffusers import FluxPipeline
from dotenv import load_dotenv

load_dotenv()

# --- Configuration Constants ---
TORCH_DTYPE = torch.bfloat16
MODEL_REPO_ID = "black-forest-labs/FLUX.1-dev"
MIN_INFERENCE_STEPS = 10 # Define the minimum inference steps here

class Predictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[FluxPipeline] = None
        self.current_lora_adapter_name = "flux_active_lora" # Consistent name for management
        print(f"Predictor initialized. Device: {self.device}")

    def setup(self):
        ''' Loads the Flux pipeline model. Expected to be called once per instance. '''
        if self.pipe:
            print("Model (self.pipe) already loaded. Skipping setup.")
            return

        print(f"Loading model: {MODEL_REPO_ID} (Torch DType: {TORCH_DTYPE}, Device: {self.device})...")
        try:
            self.pipe = FluxPipeline.from_pretrained(
                MODEL_REPO_ID,
                torch_dtype=TORCH_DTYPE
            )
            self.pipe.to(self.device)
            # Optional: if you need model CPU offloading and have less VRAM
            # if self.device == "cuda":
            #     try:
            #         self.pipe.enable_model_cpu_offload()
            #         print("Model CPU offload enabled.")
            #     except AttributeError:
            #         print("Warning: enable_model_cpu_offload not available for this pipeline version.")
            #     except Exception as offload_e:
            #         print(f"Warning: Could not enable model CPU offload: {offload_e}")

            print(f"Model '{MODEL_REPO_ID}' loaded successfully to {self.device}.")
        except Exception as e:
            self.pipe = None # Ensure pipe is None if setup fails
            print(f"FATAL: Model loading failed during setup: {e}")
            traceback.print_exc()
            # This error will be caught by the calling code (rp_handler.py)
            raise RuntimeError(f"Failed to load model '{MODEL_REPO_ID}' during setup") from e

    @torch.inference_mode()
    def predict(self,
                prompt: str,
                negative_prompt: Optional[str] = "",
                width: int = 1024,
                height: int = 1024,
                num_inference_steps: int = 30, # Default, will be overridden by handler if provided & valid
                guidance_scale: float = 4.0,
                num_outputs: int = 1,
                seed: Optional[int] = None,
                lora_path: Optional[str] = None,
                lora_scale: float = 0.8,
                **kwargs) -> List[Tuple[str, int]]:

        if not self.pipe:
            # This check is crucial. If self.pipe is None, setup() failed or was not called.
            raise RuntimeError("Predictor is not set up. Call setup() first or check for setup errors.")

        predict_start_time = time.time()
        print(f"\n--- Starting Predict Call ---")
        print(f"  Prompt: '{prompt[:100]}...'")
        print(f"  LoRA Path: {lora_path}, LoRA Scale: {lora_scale}")

        # --- LoRA Handling (Dynamic: Unload old, load new if requested) ---
        cross_attention_kwargs = None # Default to no LoRA effect / no scale

        # 1. Unload/Delete any previously loaded LoRA adapter
        try:
            # Check if PEFT integration is used and if our adapter exists
            if hasattr(self.pipe, "delete_adapters") and \
               hasattr(self.pipe, "peft_config") and \
               self.pipe.peft_config is not None and \
               self.current_lora_adapter_name in self.pipe.peft_config:
                print(f"  Attempting to delete previously loaded adapter: {self.current_lora_adapter_name}")
                self.pipe.delete_adapters(self.current_lora_adapter_name)
                print(f"  Adapter '{self.current_lora_adapter_name}' deleted.")
            # Generic unload_lora_weights might be needed if not using named PEFT adapters
            # or as a broader cleanup. Be cautious as it might error if no LoRAs are loaded.
            # elif hasattr(self.pipe, "unload_lora_weights"):
            # print(f"  Attempting to unload_lora_weights (generic)...")
            # self.pipe.unload_lora_weights()
            # print(f"  unload_lora_weights called.")
        except Exception as e:
            print(f"  Info: Issue during LoRA unloading (possibly none to unload or adapter not found): {e}")
            # traceback.print_exc() # Uncomment for detailed debugging of unload issues

        # 2. Load the new LoRA if a path is provided for the current request
        if lora_path:
            print(f"  Current request has LoRA: {os.path.basename(lora_path)}. Attempting load...")
            if os.path.exists(lora_path):
                try:
                    # Load LoRA with our consistent adapter name.
                    self.pipe.load_lora_weights(lora_path, adapter_name=self.current_lora_adapter_name)

                    # Activation and Scaling:
                    # FluxPipeline might activate on load. set_adapters is common for PEFT.
                    # If scale is applied via cross_attention_kwargs, adapter_weights should be 1.0.
                    if hasattr(self.pipe, "set_adapters"):
                         self.pipe.set_adapters([self.current_lora_adapter_name], adapter_weights=[1.0]) # Activate with base weight
                         print(f"  Adapter '{self.current_lora_adapter_name}' activated using set_adapters.")
                    else:
                         print(f"  Warning: pipe does not have set_adapters. Assuming LoRA loaded by load_lora_weights is active by default.")

                    cross_attention_kwargs = {"scale": lora_scale}
                    print(f"  Successfully loaded LoRA: {os.path.basename(lora_path)} as adapter '{self.current_lora_adapter_name}'. Scale {lora_scale} will be applied via cross_attention_kwargs.")

                except Exception as e:
                    print(f"  ERROR loading LoRA weights from {lora_path}: {e}")
                    traceback.print_exc()
                    print("  Proceeding without LoRA for this request due to load error.")
                    cross_attention_kwargs = None # Ensure no LoRA effect

                    # Attempt to clean up the failed/partially loaded adapter
                    if hasattr(self.pipe, "delete_adapters") and \
                       hasattr(self.pipe, "peft_config") and \
                       self.pipe.peft_config is not None and \
                       self.current_lora_adapter_name in self.pipe.peft_config:
                        try:
                            self.pipe.delete_adapters(self.current_lora_adapter_name)
                            print(f"  Cleaned up failed LoRA adapter '{self.current_lora_adapter_name}'.")
                        except Exception as cleanup_e:
                            print(f"  Error cleaning up failed LoRA adapter: {cleanup_e}")
            else:
                print(f"  Warning: LoRA file not found at: {lora_path}. Proceeding without LoRA for this request.")
                cross_attention_kwargs = None
        else:
            print("  No LoRA path provided for this request. Proceeding without LoRA.")
            cross_attention_kwargs = None # Explicitly ensure no LoRA effect
        # --- End LoRA Handling ---

        # --- Seed Generation ---
        if seed is None:
            # Use a CPU generator for the base seed to ensure reproducibility across devices if needed
            base_seed = torch.Generator(device='cpu').manual_seed(int.from_bytes(os.urandom(4), "big")).initial_seed()
        else:
            base_seed = seed
        # Create per-output generators on the target device
        generators = [torch.Generator(device=self.device).manual_seed(base_seed + i) for i in range(num_outputs)]
        print(f"  Base Seed: {base_seed}, Num Outputs: {num_outputs}")

        # --- Inference ---
        output_paths = []
        try:
            print(f"  Running pipeline with {num_inference_steps} steps...")
            # Using autocast for mixed precision if on CUDA
            with torch.autocast(device_type=self.device, dtype=TORCH_DTYPE, enabled=(self.device=='cuda')):
                pipeline_output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps, # Value from rp_handler (already validated)
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_outputs,
                    generator=generators,
                    output_type="pil", # Request PIL images
                    cross_attention_kwargs=cross_attention_kwargs
                )
            images: List[Image.Image] = pipeline_output.images
            print(f"  Pipeline execution completed. Received {len(images)} image(s).")

            # --- Save Images ---
            save_dir = "/tmp/generated_images" # Save to a specific subdirectory
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(images):
                current_seed = base_seed + i # The actual seed used for this specific image
                # Create a unique filename for the image
                img_filename = f"gen_seed{current_seed}_{uuid.uuid4()}.png"
                output_path = os.path.join(save_dir, img_filename)
                try:
                    if img is None:
                        print(f"  Warning: Image {i+1} (Seed: {current_seed}) is None, skipping save.")
                        continue
                    img.save(output_path)
                    output_paths.append((output_path, current_seed)) # Return path and the seed used for it
                    print(f"  Saved image {i+1} to {output_path} (Seed: {current_seed})")
                except Exception as save_err:
                    print(f"  Error saving image {i+1} (Seed: {current_seed}) to {output_path}: {save_err}")
                    # Optionally, decide if you want to append a failure placeholder here

        except Exception as predict_err:
            print(f"  ERROR during pipeline execution or image saving: {predict_err}")
            traceback.print_exc()
            output_paths = [] # Ensure empty list on major prediction error
        finally:
            # CUDA cache cleanup
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                print("  CUDA cache cleared.")

        print(f"--- Predict Call Finished in {time.time() - predict_start_time:.2f}s. Returning {len(output_paths)} image paths. ---\n")
        return output_paths