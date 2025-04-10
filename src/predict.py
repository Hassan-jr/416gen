import os
import torch
from PIL import Image
from typing import List, Tuple, Optional
import traceback
from diffusers import FluxPipeline

# MODEL_CACHE = os.environ.get("DIFFUSERS_CACHE", "/diffusers-cache")
MODEL_REPO_ID = "black-forest-labs/FLUX.1-schnell"

class Predictor:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32
        self.lora_loaded = False

    def setup(self):
        print(f"Initializing pipeline for {MODEL_REPO_ID}")
        
        # Determine optimal dtype
        if self.device == "cuda":
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            self.pipe = FluxPipeline.from_pretrained(
                MODEL_REPO_ID,
                torch_dtype=self.torch_dtype,
                # cache_dir=MODEL_CACHE,
                local_files_only=self._use_local_files()
            )
            
            # Memory optimization techniques
            self.pipe.enable_model_cpu_offload()
            
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                self.pipe.enable_xformers_memory_efficient_attention()

            print("Pipeline ready")

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Pipeline initialization failed: {e}")

    def _use_local_files(self):
        return os.environ.get("RUNPOD_USE_LOCAL_FILES", "false").lower() == "true"

    def load_lora(self, lora_path: str):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file missing: {lora_path}")
        
        try:
            self.pipe.load_lora_weights(lora_path)
            self.lora_loaded = True
            print(f"LoRA loaded from {lora_path}")
        except Exception as e:
            self.lora_loaded = False
            raise RuntimeError(f"LoRA loading failed: {e}")

    def unload_lora(self):
        if self.lora_loaded:
            self.pipe.unload_lora_weights()
            torch.cuda.empty_cache()
            self.lora_loaded = False
            print("LoRA unloaded")

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
                lora_scale: float = 0.8) -> List[Tuple[Image.Image, int]]:
        
        generator = torch.Generator(self.device)
        seed = seed or generator.seed()
        generators = [torch.Generator(self.device).manual_seed(seed + i) for i in range(num_outputs)]
        
        images = []
        for i in range(num_outputs):
            try:
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generators[i],
                    cross_attention_kwargs={"scale": lora_scale} if self.lora_loaded else None
                )
                images.append((result.images[0], seed + i))
            except Exception as e:
                print(f"Error generating image {i+1}: {str(e)}")
                traceback.print_exc()
        
        if not images:
            raise RuntimeError("No images generated")
        return images