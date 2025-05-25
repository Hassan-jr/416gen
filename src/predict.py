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
#             print("@@@@@@@@@@@@Unloading Lora@@@@@@@@@@@@@@@@@@")
#             self.pipe.unload_lora_weights()
#             if torch.cuda.is_available():
#                 gc.collect()
#                 torch.cuda.empty_cache()

#         # print(f"Predict call finished in {time.time() - start_time:.2f}s") # Optional debug
#         return output_paths

# predict.py (Fixed LoRA Management)
import os
import torch
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import traceback
import time
import gc
import uuid # For unique filenames

# NOTE: Removed explicit PEFT import/check. Relies on try/except during load.
# Ensure 'peft' is installed if LoRA is used.

# --- Diffusers Import ---
from diffusers import FluxPipeline

from dotenv import load_dotenv

# Automatically load the .env file from the current directory
load_dotenv()

# --- Configuration ---
TORCH_DTYPE = torch.bfloat16 # As per example script
MODEL_REPO_ID = "black-forest-labs/FLUX.1-dev" # As per example script

class Predictor:
    '''
    Fixed Predictor class for Flux.1-dev with proper LoRA management.
    Handles loading different LoRAs correctly and ensures clean state between calls.
    '''
    def __init__(self):
        """Initializes the predictor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[FluxPipeline] = None
        self._current_lora_path = None   # Track currently loaded LoRA
        print(f"Predictor initialized. Device: {self.device}")

    def setup(self):
        ''' Loads the Flux pipeline model. Expected to be called once. '''
        if self.pipe:
            return # Avoid reloading

        print(f"Loading model: {MODEL_REPO_ID} ({TORCH_DTYPE})...")
        try:
            self.pipe = FluxPipeline.from_pretrained(
                MODEL_REPO_ID,
                torch_dtype=TORCH_DTYPE
            )
            self.pipe.to(self.device)
            print("Model loaded successfully.")
        except Exception as e:
            self.pipe = None
            print(f"FATAL: Model loading failed: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model '{MODEL_REPO_ID}'") from e

    def _ensure_lora_state(self, lora_path: Optional[str]) -> bool:
        """
        Ensures the correct LoRA state:
        - If lora_path is None: unloads any existing LoRA
        - If lora_path is different from current: unloads current and loads new
        - If lora_path is same as current: does nothing
        
        Returns True if a LoRA is loaded after this call, False otherwise.
        """
        # If no LoRA requested, ensure none is loaded
        if lora_path is None:
            if self._current_lora_path is not None:
                print("Unloading current LoRA (none requested)")
                try:
                    self.pipe.unload_lora_weights()
                    self._current_lora_path = None
                except Exception as e:
                    print(f"Warning: Error unloading LoRA: {e}")
                    self._current_lora_path = None
            return False
        
        # If same LoRA is already loaded, nothing to do
        if self._current_lora_path == lora_path:
            print(f"LoRA already loaded: {os.path.basename(lora_path)}")
            return True
        
        # Need to change LoRA (either load new or switch)
        
        # First, unload current LoRA if any
        if self._current_lora_path is not None:
            print(f"Unloading current LoRA: {os.path.basename(self._current_lora_path)}")
            try:
                self.pipe.unload_lora_weights()
            except Exception as e:
                print(f"Warning: Error unloading current LoRA: {e}")
            self._current_lora_path = None
        
        # Now load the new LoRA
        if not os.path.exists(lora_path):
            print(f"Warning: LoRA file not found at: {lora_path}")
            return False
        
        print(f"Loading new LoRA: {os.path.basename(lora_path)}")
        try:
            self.pipe.load_lora_weights(lora_path)
            self._current_lora_path = lora_path
            print(f"Successfully loaded LoRA: {os.path.basename(lora_path)}")
            return True
        except Exception as e:
            print(f"Error loading LoRA weights from {lora_path}: {e}")
            self._current_lora_path = None
            return False

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
                lora_path: Optional[str] = None, # Path provided by handler
                lora_scale: float = 0.8,
                **kwargs) -> List[Tuple[str, int]]:
        '''
        Performs inference with proper LoRA management.
        Loads/unloads LoRAs as needed to ensure clean state.
        Returns a list of (local_image_path, seed) tuples for successes.
        '''
        if not self.pipe:
            raise RuntimeError("Predictor is not set up. Call setup() first.")

        start_time = time.time()

        # Ensure correct LoRA state
        lora_loaded = self._ensure_lora_state(lora_path)
        
        if lora_path and not lora_loaded:
            print("Warning: Requested LoRA could not be loaded, proceeding without LoRA")

        # --- Seed Generation ---
        if seed is None:
            base_seed = torch.Generator(device='cpu').seed()
        else:
            base_seed = seed
        generators = [torch.Generator(device=self.device).manual_seed(base_seed + i) for i in range(num_outputs)]

        # --- Inference ---
        output_paths = []
        try:
            # Using autocast like the example script
            with torch.autocast(device_type=self.device, dtype=TORCH_DTYPE, enabled=self.device=='cuda'):
                pipeline_output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_outputs,
                    generator=generators,
                    output_type="pil",
                )
            images = pipeline_output.images

            # --- Save Images ---
            os.makedirs("/tmp", exist_ok=True)
            for i, img in enumerate(images):
                current_seed = base_seed + i
                output_path = f"/tmp/gen_{current_seed}_{uuid.uuid4()}.png"
                try:
                    if img is None: continue
                    img.save(output_path)
                    output_paths.append((output_path, current_seed))
                except Exception as save_err:
                    print(f"Error saving image {i+1} (Seed: {current_seed}): {save_err}")

        except Exception as predict_err:
            print(f"Error during pipeline execution: {predict_err}")
            traceback.print_exc()
            output_paths = [] # Return empty on error
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

        print(f"Predict call finished in {time.time() - start_time:.2f}s")
        return output_paths

    def cleanup(self):
        """Clean up resources and unload any loaded LoRA"""
        if self._current_lora_path is not None:
            print("Cleaning up: Unloading current LoRA")
            try:
                self.pipe.unload_lora_weights()
            except Exception as e:
                print(f"Warning: Error during cleanup unload: {e}")
            self._current_lora_path = None
        
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()