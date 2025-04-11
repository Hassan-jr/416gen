# predict.py
import os
import torch
# requests is no longer needed here, moved to handler
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import traceback
import time
import gc # Garbage Collector
import sys # For exit on critical failure

# --- Dependency Checks ---
try:
    import peft
    print(f"Successfully imported 'peft' library. Version: {peft.__version__}")
    PEFT_AVAILABLE = True
except ImportError:
    print("WARNING: Failed to import the 'peft' library.")
    print("         Ensure 'peft' is correctly listed and installed from requirements.txt.")
    print("         LoRA loading will be DISABLED.")
    PEFT_AVAILABLE = False

# --- Diffusers/Transformers Imports ---
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.utils import is_peft_available # Keep the check function
from diffusers.image_processor import VaeImageProcessor # Needed for decoding

# --- Quantization Imports ---
# Keep the check, but disable usage by default for this refactor
try:
    from optimum.quanto import freeze, qfloat8, quantize
    OPTIMUM_AVAILABLE = True
    print("Found optimum.quanto library. Quantization dependency is present.")
except ImportError:
    print("WARNING: optimum.quanto not found. Quantization will be DISABLED.")
    OPTIMUM_AVAILABLE = False

# --- Configuration ---
TORCH_DTYPE = torch.float16 # Stick with float16 for broader compatibility
# TORCH_DTYPE = torch.bfloat16 # Use bfloat16 if your target GPUs support it (Ampere+)
SCHNELL_REPO_ID = "black-forest-labs/FLUX.1-schnell"
SCHNELL_REVISION = "refs/pr/1" # Use specific revision if needed, often `main` is fine too
CLIP_REPO_ID = "openai/clip-vit-large-patch14"
# --- !!! DISABLE QUANTIZATION FOR MEMORY OPTIMIZATION REFACTOR !!! ---
USE_QUANTIZATION = False # Set to False to use the sequential loading pattern effectively
# QUANT_TYPE = qfloat8 # Keep for reference if re-enabling

class Predictor:
    '''
    Predictor class for Flux.1-schnell optimized for lower VRAM by
    loading components sequentially. Quantization is disabled for this pattern.
    '''

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheduler = None # Will be loaded once
        self.vae_scale_factor = None # Will be calculated once
        self.image_processor = None # Will be created once
        self.loaded_lora_adapter_name = "default_lora" # Consistent name for PEFT
        print(f"Predictor initialized. Target Device: {self.device}, Base Dtype: {TORCH_DTYPE}")
        if not PEFT_AVAILABLE: print("PEFT not found, LoRA functionality disabled.")
        if USE_QUANTIZATION: print("WARNING: Quantization is enabled but incompatible with this sequential loading predictor. Set USE_QUANTIZATION=False.")
        if self.device == "cpu": print("Warning: CUDA not available, running on CPU.")
        elif torch.cuda.is_available(): print(f"Initial GPU Memory: {self._get_gpu_memory_usage()}")

    def _get_gpu_memory_usage(self) -> str:
        """ Helper to get formatted GPU memory usage """
        if not torch.cuda.is_available(): return "N/A (CPU)"
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        return f"Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB"

    def _flush_gpu_memory(self):
        """ Utility function to clear GPU memory """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # torch.cuda.reset_peak_memory_stats() # Optional: Reset peak stats too

    def setup(self):
        ''' Load minimal, persistent components: Scheduler and VAE config '''
        if self.scheduler and self.vae_scale_factor:
            print("Predictor already set up.")
            return

        print("Setting up predictor...")
        start_time = time.time()
        local_files_only = os.environ.get("RUNPOD_USE_LOCAL_FILES", "false").lower() == "true"
        print(f"Using local files only: {local_files_only}")

        try:
            # Load scheduler (small, keep loaded)
            print("Loading scheduler...")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                SCHNELL_REPO_ID,
                subfolder="scheduler",
                revision=SCHNELL_REVISION,
                local_files_only=local_files_only
            )

            # Get VAE config to determine scale factor (don't load full VAE yet)
            print("Loading VAE config...")
            vae_config = AutoencoderKL.load_config(
                SCHNELL_REPO_ID,
                subfolder="vae",
                revision=SCHNELL_REVISION,
                local_files_only=local_files_only
            )
            self.vae_scale_factor = 2 ** (len(vae_config.get("block_out_channels", [])) - 1) # Diffusers 0.28+ calculation
            # self.vae_scale_factor = 2 ** (len(vae_config.block_out_channels) - 1) # Older diffusers
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            print(f"VAE Scale Factor: {self.vae_scale_factor}")

            print(f"Predictor setup complete in {time.time() - start_time:.2f} seconds.")
            print(f"GPU Memory after setup: {self._get_gpu_memory_usage()}")

        except Exception as e:
            print(f"FATAL Error during predictor setup: {e}")
            traceback.print_exc()
            self.scheduler = None
            self.vae_scale_factor = None
            self.image_processor = None
            raise RuntimeError("Failed to setup predictor") from e
        finally:
            self._flush_gpu_memory()

    # Note: load_lora and unload_lora are removed as public methods.
    # LoRA is handled internally within the predict method.

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
                lora_path: Optional[str] = None, # Added LoRA path parameter
                lora_scale: float = 0.8, # Added LoRA scale parameter
                **kwargs) -> List[Tuple[str, int]]:
        ''' Run prediction using sequential loading of components. '''

        if not self.scheduler or not self.image_processor:
            raise RuntimeError("Predictor not set up. Call setup() first.")

        if USE_QUANTIZATION:
             print("WARNING: Quantization is enabled but not used in this sequential predictor.")

        if lora_path and not PEFT_AVAILABLE:
            print("Warning: LoRA path provided, but PEFT library is not available. Ignoring LoRA.")
            lora_path = None # Ensure LoRA is not used

        # --- Argument Handling ---
        if seed is None:
            base_seed = torch.Generator(device='cpu').seed()
        else:
            base_seed = seed
        print(f"Using base seed: {base_seed} for {num_outputs} outputs.")

        if guidance_scale != 0.0:
            print(f"Info: FLUX.1-schnell ignores guidance_scale. Using 0.0.")
            guidance_scale = 0.0 # Force 0.0
        if negative_prompt:
            print(f"Info: FLUX.1-schnell ignores negative prompts.")
            negative_prompt = "" # Force empty

        local_files_only = os.environ.get("RUNPOD_USE_LOCAL_FILES", "false").lower() == "true"
        output_paths = []
        overall_start_time = time.time()
        print(f"\nStarting generation for prompt: '{prompt[:80]}...'")
        print(f"Params: W={width}, H={height}, Steps={num_inference_steps}, Num={num_outputs}, Seed={base_seed}, LoRA='{lora_path if lora_path else 'None'}'")

        # ================== PHASE 1: PROMPT ENCODING ==================
        print(f"\n--- Phase 1: Prompt Encoding ---")
        phase1_start = time.time()
        prompt_embeds, pooled_prompt_embeds = None, None
        text_encoder, text_encoder_2, tokenizer, tokenizer_2 = None, None, None, None
        temp_pipeline_stub = None # Temporary object to use encode_prompt

        try:
            print(f"Loading text encoders and tokenizers... (dtype={TORCH_DTYPE}, local_only={local_files_only})")
            tokenizer = CLIPTokenizer.from_pretrained(CLIP_REPO_ID, local_files_only=local_files_only)
            tokenizer_2 = T5TokenizerFast.from_pretrained(SCHNELL_REPO_ID, subfolder="tokenizer_2", revision=SCHNELL_REVISION, local_files_only=local_files_only)
            text_encoder = CLIPTextModel.from_pretrained(CLIP_REPO_ID, torch_dtype=TORCH_DTYPE, local_files_only=local_files_only).to(self.device)
            text_encoder_2 = T5EncoderModel.from_pretrained(SCHNELL_REPO_ID, subfolder="text_encoder_2", revision=SCHNELL_REVISION, torch_dtype=TORCH_DTYPE, local_files_only=local_files_only).to(self.device)
            print(f"Text encoders loaded. GPU Memory: {self._get_gpu_memory_usage()}")

            # Use a temporary pipeline stub to access the encode_prompt method easily
            temp_pipeline_stub = FluxPipeline(
                 scheduler=self.scheduler, # Need a scheduler object
                 tokenizer=tokenizer, text_encoder=text_encoder,
                 tokenizer_2=tokenizer_2, text_encoder_2=text_encoder_2,
                 vae=None, transformer=None # These are not needed for encoding
            )
            # Match Flux's default max_sequence_length if needed, check pipeline config or defaults
            max_length = 512 # Common default, check if Flux uses something different

            print("Encoding prompt...")
            prompt_embeds, pooled_prompt_embeds, _ = temp_pipeline_stub.encode_prompt(
                prompt=prompt,
                prompt_2=None, # Schnell doesn't use prompt_2
                max_sequence_length=max_length,
                device=self.device, # Ensure embeds are created on the correct device
                num_images_per_prompt=1, # Encode once per prompt
                # do_classifier_free_guidance=False, # Schnell specific
            )
            print("Prompt encoded successfully.")
            print(f"Phase 1 duration: {time.time() - phase1_start:.2f}s")

        except Exception as e:
            print(f"ERROR during prompt encoding: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed during prompt encoding phase.") from e
        finally:
            print("Unloading text encoders and tokenizers...")
            del text_encoder, text_encoder_2, tokenizer, tokenizer_2, temp_pipeline_stub
            self._flush_gpu_memory()
            print(f"GPU Memory after Phase 1 cleanup: {self._get_gpu_memory_usage()}")

        if prompt_embeds is None or pooled_prompt_embeds is None:
             raise RuntimeError("Prompt embeddings were not generated.")

        # ================== PHASE 2: LATENT GENERATION (DENOISING) ==================
        print(f"\n--- Phase 2: Latent Generation (Denoising) ---")
        transformer = None
        temp_pipeline = None # Pipeline stub for denoising
        latents_list = [] # Store latents for each output
        lora_applied = False
        try:
            # Expand embeds for num_outputs
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(num_outputs, 1, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_outputs, 1)

            print(f"Loading transformer... (dtype={TORCH_DTYPE}, local_only={local_files_only})")
            phase2_start = time.time()
            transformer = FluxTransformer2DModel.from_pretrained(
                SCHNELL_REPO_ID,
                subfolder="transformer",
                revision=SCHNELL_REVISION,
                torch_dtype=TORCH_DTYPE,
                local_files_only=local_files_only
            ).to(self.device)
            print(f"Transformer loaded. GPU Memory: {self._get_gpu_memory_usage()}")

            # --- LoRA Handling ---
            
            if lora_path:
                 if not PEFT_AVAILABLE:
                     print("Skipping LoRA: PEFT library not available.")
                 elif not os.path.exists(lora_path):
                     print(f"Skipping LoRA: File not found at {lora_path}")
                 else:
                     print(f"Applying LoRA from {lora_path} with scale {lora_scale}...")
                     try:
                         # Apply LoRA directly to the transformer module
                         # We might need to wrap it in a temporary pipeline to use `load_lora_weights` easily
                         # Or use peft functions directly if `load_lora_weights` targets specific modules
                         # Let's try using a temporary pipeline context
                         temp_lora_pipe = FluxPipeline(
                             scheduler=self.scheduler, transformer=transformer,
                             # Include minimal other components if required by load_lora_weights
                             tokenizer=None, text_encoder=None, tokenizer_2=None, text_encoder_2=None, vae=None
                         )
                         temp_lora_pipe.load_lora_weights(lora_path, adapter_name=self.loaded_lora_adapter_name)
                         # Set the LoRA scale - Requires diffusers >= 0.27 ? Check documentation
                         # temp_lora_pipe.set_adapters([self.loaded_lora_adapter_name], adapter_weights=[lora_scale]) # Newer diffusers
                         # If the above doesn't work or isn't available, scale is often applied differently or baked in.
                         # For now, we load it, scale application might need adjustment based on diffusers version.
                         print(f"LoRA loaded onto transformer. Scale {lora_scale} requested (application method depends on diffusers version).")
                         lora_applied = True
                         # Keep the transformer modified by LoRA
                         transformer = temp_lora_pipe.transformer
                         del temp_lora_pipe # Remove the temporary pipe wrapper

                     except Exception as lora_err:
                         print(f"ERROR applying LoRA: {lora_err}")
                         traceback.print_exc()
                         # Ensure transformer is still the original one if LoRA failed
                         # (Reloading might be safer but costly, assume transformer object is okay if load_lora_weights fails cleanly)

            # Create the minimal pipeline for denoising
            temp_pipeline = FluxPipeline(
                scheduler=self.scheduler,
                transformer=transformer, # Use the (potentially LoRA-modified) transformer
                # Other components are not needed for latent generation with pre-computed embeds
                tokenizer=None, text_encoder=None, tokenizer_2=None, text_encoder_2=None, vae=None
            )
            # No need for .to(self.device) if transformer is already there

            print("Generating latents...")
            # Prepare generator for reproducibility across num_outputs
            generators = [torch.Generator(device=self.device).manual_seed(base_seed + i) for i in range(num_outputs)]

            # Note: Flux pipeline call handles iterating internally if num_images_per_prompt > 1
            # We pass num_outputs via repeated embeds and provide multiple generators
            latents = temp_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generators, # Pass list of generators
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, # Should be 0.0
                height=height,
                width=width,
                output_type="latent",
            ).images # .images contains the latents when output_type="latent"

            # latents should have shape (num_outputs, C, H // scale_factor, W // scale_factor)
            print(f"Latents generated, shape: {latents.shape}")
            latents_list = [latents[i:i+1] for i in range(num_outputs)] # Split into list of single latents
            print(f"Phase 2 duration: {time.time() - phase2_start:.2f}s")

        except Exception as e:
             print(f"ERROR during latent generation: {e}")
             traceback.print_exc()
             raise RuntimeError("Failed during latent generation phase.") from e
        finally:
             print("Unloading transformer and LoRA (if applied)...")
             if lora_applied and temp_pipeline and hasattr(temp_pipeline, "unload_lora_weights"):
                 try:
                      print(f"Unloading LoRA adapter: {self.loaded_lora_adapter_name}")
                      # Ensure PEFT state is cleaned from the transformer
                      temp_pipeline.unload_lora_weights() # Unload from the pipeline context
                      print("LoRA unloaded.")
                 except Exception as unload_err:
                      print(f"Warning: Error unloading LoRA weights: {unload_err}")
             del transformer, temp_pipeline # Delete objects
             self._flush_gpu_memory()
             print(f"GPU Memory after Phase 2 cleanup: {self._get_gpu_memory_usage()}")

        if not latents_list:
            raise RuntimeError("Latents were not generated.")

        # ================== PHASE 3: LATENT DECODING ==================
        print(f"\n--- Phase 3: Latent Decoding ---")
        vae = None
        try:
            print(f"Loading VAE... (dtype={TORCH_DTYPE}, local_only={local_files_only})")
            phase3_start = time.time()
            # Load VAE with the same dtype as others for consistency
            vae = AutoencoderKL.from_pretrained(
                 SCHNELL_REPO_ID,
                 subfolder="vae",
                 revision=SCHNELL_REVISION,
                 torch_dtype=TORCH_DTYPE,
                 local_files_only=local_files_only
             ).to(self.device)
            print(f"VAE loaded. GPU Memory: {self._get_gpu_memory_usage()}")

            # Get VAE scale factor and shift factor directly from loaded VAE config
            vae_scale_factor = self.vae_scale_factor # Already calculated
            scaling_factor = vae.config.scaling_factor
            shift_factor = vae.config.shift_factor if hasattr(vae.config, 'shift_factor') else 0.0 # Handle older diffusers/configs

            print(f"Decoding {len(latents_list)} latents...")
            os.makedirs("/tmp", exist_ok=True) # Ensure temp dir exists

            for i, latent_batch in enumerate(latents_list):
                 current_seed = base_seed + i
                 print(f"  Decoding image {i+1}/{len(latents_list)} (Seed: {current_seed})...")
                 iter_start_time = time.time()

                 # Prepare latents for VAE
                 # Based on FluxPipeline._unpack_latents and standard VAE usage
                 # No need to unpack if latents are already correct shape from pipeline
                 # Just scale and shift
                 prepared_latents = (latent_batch / scaling_factor) + shift_factor

                 # Decode
                 with torch.no_grad(): # Ensure inference mode for VAE decode
                      image = vae.decode(prepared_latents.to(self.device, dtype=TORCH_DTYPE), return_dict=False)[0]

                 # Post-process
                 image = self.image_processor.postprocess(image, output_type="pil")[0]

                 # Save
                 output_path = f"/tmp/out_{current_seed}_{i}.png"
                 image.save(output_path)
                 output_paths.append((output_path, current_seed))
                 print(f"  Saved image {i+1} to {output_path} (Decode took {time.time() - iter_start_time:.2f}s)")

            print(f"Phase 3 duration: {time.time() - phase3_start:.2f}s")

        except Exception as e:
             print(f"ERROR during latent decoding: {e}")
             traceback.print_exc()
             # Don't raise here, allow cleanup, return paths generated so far
             print("Attempting to return any successfully generated images.")
        finally:
             print("Unloading VAE...")
             del vae
             self._flush_gpu_memory()
             print(f"GPU Memory after Phase 3 cleanup: {self._get_gpu_memory_usage()}")


        overall_end_time = time.time()
        print(f"\nFinished generating {len(output_paths)} images in {overall_end_time - overall_start_time:.2f} seconds.")
        if not output_paths and num_outputs > 0:
            print("Warning: No images were successfully generated.")
            # Optionally raise an error if zero images is critical failure
            # raise RuntimeError("Failed to generate any images.")

        return output_paths