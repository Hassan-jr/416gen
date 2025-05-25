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

TORCH_DTYPE = torch.bfloat16
MODEL_REPO_ID = "black-forest-labs/FLUX.1-dev"
MIN_INFERENCE_STEPS = 10

class Predictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[FluxPipeline] = None
        self.lora_adapter_name = "active_lora" # Consistent name used for PEFT

        # Track currently active LoRA to avoid redundant operations
        self.active_lora_path: Optional[str] = None
        self.active_lora_scale: Optional[float] = None
        print(f"Predictor initialized. Device: {self.device}")

    def _unload_active_lora(self):
        """Helper to unload the currently tracked active LoRA."""
        if self.active_lora_path is None: # No LoRA is tracked as active
            # As a safeguard, check if the adapter name still exists in peft_config
            # and try to delete it if it does (e.g. if state tracking got out of sync)
            if hasattr(self.pipe, "delete_adapters") and \
               hasattr(self.pipe, "peft_config") and \
               self.pipe.peft_config is not None and \
               self.lora_adapter_name in self.pipe.peft_config:
                print(f"  Warning: active_lora_path was None, but adapter '{self.lora_adapter_name}' found in peft_config. Attempting delete.")
                try:
                    self.pipe.delete_adapters(self.lora_adapter_name)
                    print(f"  Adapter '{self.lora_adapter_name}' deleted (safeguard).")
                except Exception as e:
                    print(f"  Error during safeguard delete of adapter '{self.lora_adapter_name}': {e}")
            return

        print(f"  Unloading active LoRA: {os.path.basename(self.active_lora_path)} (Adapter: {self.lora_adapter_name})")
        try:
            if hasattr(self.pipe, "delete_adapters") and \
               hasattr(self.pipe, "peft_config") and \
               self.pipe.peft_config is not None and \
               self.lora_adapter_name in self.pipe.peft_config:
                self.pipe.delete_adapters(self.lora_adapter_name)
                print(f"  Adapter '{self.lora_adapter_name}' deleted successfully.")
            # Optional: A more general disable if the above doesn't fully clear
            # elif hasattr(self.pipe, "disable_lora"):
            #     self.pipe.disable_lora()
            #     print(f"  Called pipe.disable_lora().")
            else:
                print(f"  Info: Could not specifically delete adapter '{self.lora_adapter_name}' (not found or method unavailable).")

            self.active_lora_path = None
            self.active_lora_scale = None
        except Exception as e:
            print(f"  Error during LoRA unload for {self.lora_adapter_name}: {e}")
            # Attempt to reset tracking even if unload had issues
            self.active_lora_path = None
            self.active_lora_scale = None


    def setup(self):
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
            print(f"Model '{MODEL_REPO_ID}' loaded successfully to {self.device}.")
        except Exception as e:
            self.pipe = None
            print(f"FATAL: Model loading failed during setup: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model '{MODEL_REPO_ID}' during setup") from e

    @torch.inference_mode()
    def predict(self,
                prompt: str,
                negative_prompt: Optional[str] = "",
                width: int = 1024,
                height: int = 1024,
                num_inference_steps: int = 30,
                guidance_scale: float = 4.0,
                num_outputs: int = 1,
                seed: Optional[int] = None,
                lora_path: Optional[str] = None, # Path from rp_handler
                lora_scale: float = 0.8,
                **kwargs) -> List[Tuple[str, int]]:

        if not self.pipe:
            raise RuntimeError("Predictor is not set up. Call setup() first or check for setup errors.")

        predict_start_time = time.time()
        print(f"\n--- Starting Predict Call ---")
        print(f"  Prompt: '{prompt[:100]}...'")
        print(f"  Requested LoRA Path: {lora_path}, Requested LoRA Scale: {lora_scale}")
        print(f"  Currently Active LoRA Path: {self.active_lora_path}, Scale: {self.active_lora_scale}")

        # --- Smarter LoRA Management ---
        if lora_path: # A LoRA is requested for this prediction
            if not os.path.exists(lora_path):
                print(f"  Warning: Requested LoRA file not found at: {lora_path}. Unloading any active LoRA.")
                if self.active_lora_path:
                    self._unload_active_lora()
            elif self.active_lora_path != lora_path: # New LoRA or switching LoRA
                print(f"  Switching LoRA. Old: {self.active_lora_path}, New: {os.path.basename(lora_path)}")
                if self.active_lora_path: # Unload previous if one was active
                    self._unload_active_lora()
                try:
                    print(f"  Loading new LoRA: {os.path.basename(lora_path)} with adapter name '{self.lora_adapter_name}'")
                    self.pipe.load_lora_weights(lora_path, adapter_name=self.lora_adapter_name)
                    if hasattr(self.pipe, "set_adapters"):
                        self.pipe.set_adapters([self.lora_adapter_name], adapter_weights=[lora_scale])
                        print(f"  Adapter '{self.lora_adapter_name}' loaded and set with scale {lora_scale}.")
                    else:
                        print(f"  Warning: pipe does not have set_adapters. LoRA scaling might not be applied as expected.")
                    self.active_lora_path = lora_path
                    self.active_lora_scale = lora_scale
                except Exception as e:
                    print(f"  ERROR loading new LoRA weights from {lora_path}: {e}")
                    traceback.print_exc()
                    self._unload_active_lora() # Attempt to clean up if load failed
            elif self.active_lora_scale != lora_scale: # Same LoRA path, but different scale
                print(f"  Updating scale for active LoRA: {os.path.basename(self.active_lora_path)}. Old scale: {self.active_lora_scale}, New scale: {lora_scale}")
                if hasattr(self.pipe, "set_adapters"):
                    self.pipe.set_adapters([self.lora_adapter_name], adapter_weights=[lora_scale])
                    self.active_lora_scale = lora_scale
                    print(f"  Adapter '{self.lora_adapter_name}' scale updated to {lora_scale}.")
                else:
                    print(f"  Warning: pipe does not have set_adapters. Cannot update LoRA scale dynamically.")
            else: # LoRA path and scale are already active
                print(f"  LoRA {os.path.basename(lora_path)} with scale {lora_scale} is already active. No change needed.")
        else: # No LoRA is requested for this prediction
            if self.active_lora_path is not None: # A LoRA is active, but not requested now
                print(f"  No LoRA requested. Unloading currently active LoRA: {os.path.basename(self.active_lora_path)}")
                self._unload_active_lora()
            else:
                print("  No LoRA requested, and no LoRA is currently active.")
        # --- End Smarter LoRA Management ---

        if seed is None:
            base_seed = torch.Generator(device='cpu').manual_seed(int.from_bytes(os.urandom(4), "big")).initial_seed()
        else:
            base_seed = seed
        generators = [torch.Generator(device=self.device).manual_seed(base_seed + i) for i in range(num_outputs)]
        print(f"  Base Seed: {base_seed}, Num Outputs: {num_outputs}, Steps: {num_inference_steps}, Guidance: {guidance_scale}")

        output_paths = []
        try:
            print(f"  Running pipeline...")
            with torch.autocast(device_type=self.device, dtype=TORCH_DTYPE, enabled=(self.device=='cuda')):
                pipeline_output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_outputs,
                    generator=generators,
                    output_type="pil"
                )
            images: List[Image.Image] = pipeline_output.images
            print(f"  Pipeline execution completed. Received {len(images)} image(s).")

            save_dir = "/tmp/generated_images"
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(images):
                current_seed = base_seed + i
                img_filename = f"gen_seed{current_seed}_{uuid.uuid4()}.png"
                output_path = os.path.join(save_dir, img_filename)
                try:
                    if img is None:
                        print(f"  Warning: Image {i+1} (Seed: {current_seed}) is None, skipping save.")
                        continue
                    img.save(output_path)
                    output_paths.append((output_path, current_seed))
                    print(f"  Saved image {i+1} to {output_path} (Seed: {current_seed})")
                except Exception as save_err:
                    print(f"  Error saving image {i+1} (Seed: {current_seed}) to {output_path}: {save_err}")

        except Exception as predict_err:
            print(f"  ERROR during pipeline execution or image saving: {predict_err}")
            traceback.print_exc()
            output_paths = []
        finally:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                print("  CUDA cache cleared.")

        print(f"--- Predict Call Finished in {time.time() - predict_start_time:.2f}s. Returning {len(output_paths)} image paths. ---\n")
        return output_paths