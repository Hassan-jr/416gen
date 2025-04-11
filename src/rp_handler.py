# rp_handler.py
import os
import uuid
import io
import traceback
import requests
from pathlib import Path
import time # For basic timing if needed

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_cleanup

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from rp_schema import INPUT_SCHEMA
import predict # Import your predictor class

# Global predictor instance
MODEL = None

def download_file(url, save_path):
    """Downloads a file from a URL to a local path."""
    print(f"Downloading {url} to {save_path}...")
    try:
        response = requests.get(url, stream=True, timeout=300) # 5 min timeout
        response.raise_for_status() # Raise an exception for bad status codes
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return False

def upload_to_r2(image_bytes, bucket_name, object_key, r2_config):
    """Uploads image bytes to Cloudflare R2."""
    print(f"Uploading to R2: {bucket_name}/{object_key}")
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=r2_config['endpoint_url'],
            aws_access_key_id=r2_config['access_key_id'],
            aws_secret_access_key=r2_config['secret_access_key'],
            config=Config(s3={'addressing_style': 'virtual'}),
            region_name='auto'
        )

        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=image_bytes,
            ContentType='image/png'
            # ACL='public-read' # Uncomment if you want public read access
        )
        # Construct the public URL (ensure endpoint_url doesn't have trailing slash)
        public_url = f"{r2_config['endpoint_url']}/{bucket_name}/{object_key}"
        print(f"Upload successful. URL: {public_url}")
        return public_url

    except ClientError as e:
        print(f"Boto3 ClientError uploading to R2: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error uploading to R2: {e}")
        traceback.print_exc()
        return None

def run(job):
    '''
    Run inference on the Flux.1 Schnell model for multiple generation requests.
    '''
    global MODEL # Ensure we're using the global model instance

    job_input = job['input']

    # --- Input Validation ---
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # --- Extract Config ---
    instance_id = validated_input['instanceId']
    lora_url = validated_input.get('lora_url')
    # Keep lora_scale from input for logging, even if not used as inference multiplier
    lora_scale_input = validated_input.get('lora_scale', 0.8)
    r2_config = {
        'bucket_name': validated_input['r2_bucket_name'],
        'access_key_id': validated_input['r2_access_key_id'],
        'secret_access_key': validated_input['r2_secret_access_key'],
        'endpoint_url': validated_input['r2_endpoint_url'].rstrip('/'),
        'path_prefix': validated_input.get('r2_path_in_bucket', '').strip('/')
    }
    generation_tasks = validated_input['generations']

    # --- Initialize Predictor (if not already) ---
    if MODEL is None:
         print("Error: Model predictor not initialized.")
         # Attempt to initialize if running locally? Usually done in main guard for serverless.
         try:
             print("Attempting late model initialization...")
             MODEL = predict.Predictor()
             MODEL.setup()
             print("Model initialized.")
         except Exception as e:
            print(f"Fatal: Failed to initialize model late: {e}")
            traceback.print_exc()
            return {"error": "Model predictor failed to initialize."}

    # --- Download and Load LoRA (if provided) ---
    lora_download_path = "/tmp/downloaded_lora.safetensors" # Temporary path
    lora_loaded_successfully = False
    if lora_url:
        print("LoRA URL provided, attempting download...")
        if download_file(lora_url, lora_download_path):
            try:
                # Pass the file path to the loader
                MODEL.load_lora(lora_download_path)
                lora_loaded_successfully = True
                print("LoRA loaded successfully.")
            except Exception as e:
                print(f"Error loading LoRA from {lora_download_path}: {e}")
                traceback.print_exc()
                # Decide how to proceed: return error or continue without LoRA
                # return {"error": f"Failed to load LoRA: {e}"} # Option 1: Fail job
                print("Warning: Proceeding without LoRA due to loading error.") # Option 2: Continue
        else:
            print("LoRA download failed. Proceeding without LoRA.")
            # return {"error": "Failed to download LoRA."} # Option 1: Fail job
    else:
        print("No LoRA URL provided. Ensuring any previous LoRA is unloaded.")
        # Ensure any previously loaded LoRA is unloaded
        try:
            MODEL.unload_lora() # unload_lora handles the check if it's loaded
            # print("Unloaded any previous LoRAs.") # unload_lora prints messages
        except Exception as e:
             # This shouldn't normally error unless unload_lora itself fails badly
             print(f"Error during explicit LoRA unload: {e}")


    # --- Process Generation Tasks ---
    job_output = {
        "instanceId": instance_id,
        "generations": []
    }
    temp_image_paths = [] # Keep track of local files

    try:
        overall_start_time = time.time()
        for index, task in enumerate(generation_tasks):
            task_start_time = time.time()
            print(f"\nProcessing generation task {index + 1}/{len(generation_tasks)}...")
            prompt = task['prompt']
            num_outputs = task.get('num_outputs', 1)
            # Use provided seed or generate a random one
            seed = task.get('seed') if task.get('seed') is not None else int.from_bytes(os.urandom(4), "big")

            print(f"  Prompt: {prompt[:100]}...") # Log truncated prompt
            print(f"  Num Outputs: {num_outputs}, Seed: {seed}")
            print(f"  LoRA Active: {lora_loaded_successfully}, Requested LoRA Scale (Informational): {lora_scale_input}")

            # Prepare arguments for the predictor
            # REMOVED: lora_scale is not passed to predict anymore
            predict_args = {
                "prompt": prompt,
                "negative_prompt": task.get("negative_prompt", ""),
                "width": task.get('width', 1024),
                "height": task.get('height', 1024),
                "num_inference_steps": task.get('num_inference_steps', 4),
                "guidance_scale": task.get('guidance_scale', 0.0),
                "num_outputs": num_outputs,
                "seed": seed,
                # Add other Flux parameters from the task if needed
            }

            task_results = {
                "prompt": prompt,
                "negative_prompt": predict_args["negative_prompt"],
                "width": predict_args["width"],
                "height": predict_args["height"],
                "num_outputs": num_outputs,
                "num_inference_steps": predict_args["num_inference_steps"],
                "guidance_scale": predict_args["guidance_scale"],
                "seed": seed, # Record the initial seed used
                "lora_url": lora_url if lora_loaded_successfully else None, # Record LoRA URL only if loaded
                "lora_scale_requested": lora_scale_input, # Record user's requested scale
                "images": [],
                "task_duration_seconds": None,
                "error": None
            }

            try:
                # Generate images - expect predict to return a list of tuples: (image_path, final_seed)
                generated_images = MODEL.predict(**predict_args) # Returns list of (path, final_seed)

                # Upload each generated image
                for img_path, img_seed in generated_images:
                    temp_image_paths.append(img_path) # Track for cleanup
                    try:
                        with open(img_path, 'rb') as f_img:
                            image_bytes = f_img.read()

                        # Create a unique name for the image in R2
                        # Include instanceId and task index for better traceability
                        image_filename = f"{instance_id}_task{index}_seed{img_seed}_{uuid.uuid4()}.png"
                        object_key = image_filename
                        if r2_config['path_prefix']:
                           object_key = f"{r2_config['path_prefix']}/{image_filename}" # Add prefix

                        # Upload and get URL
                        r2_url = upload_to_r2(
                            image_bytes=image_bytes,
                            bucket_name=r2_config['bucket_name'],
                            object_key=object_key,
                            r2_config=r2_config
                        )

                        if r2_url:
                            task_results["images"].append({
                                "url": r2_url,
                                "seed": img_seed # Store the specific seed for this image
                            })
                        else:
                            print(f"Warning: Failed to upload image {img_path} to R2.")
                            task_results["images"].append({
                                "url": None,
                                "seed": img_seed,
                                "error": "Upload failed"
                             })
                    except Exception as upload_err:
                         print(f"Error reading or uploading image {img_path}: {upload_err}")
                         traceback.print_exc()
                         task_results["images"].append({
                            "url": None,
                            "seed": img_seed,
                            "error": f"Image processing/upload failed: {upload_err}"
                         })


            except Exception as e:
                print(f"Error during prediction for task {index}: {e}")
                traceback.print_exc()
                task_results["error"] = f"Prediction failed: {str(e)}" # Add error to the specific task result

            task_end_time = time.time()
            task_results["task_duration_seconds"] = round(task_end_time - task_start_time, 2)
            job_output["generations"].append(task_results)

        overall_end_time = time.time()
        print(f"\nFinished all tasks in {round(overall_end_time - overall_start_time, 2)} seconds.")

    finally:
        # --- Cleanup ---
        print("Starting cleanup...")
        # Remove downloaded LoRA file
        if lora_url and os.path.exists(lora_download_path):
            try:
                os.remove(lora_download_path)
                print(f"Removed temporary LoRA file: {lora_download_path}")
            except OSError as e:
                print(f"Error removing temporary LoRA file {lora_download_path}: {e}")

        # Remove generated temporary image files
        cleaned_images = 0
        for path in temp_image_paths:
             if os.path.exists(path):
                try:
                    os.remove(path)
                    cleaned_images += 1
                except OSError as e:
                    print(f"Error removing temporary image file {path}: {e}")
        print(f"Removed {cleaned_images} temporary image files.")

        # Unload LoRA weights from the model if it was loaded
        if lora_loaded_successfully:
             try:
                 MODEL.unload_lora() # Handles check and prints messages
             except Exception as e:
                 print(f"Error unloading LoRA from model: {e}") # Should be rare

        # RunPod cleanup (optional, usually for downloaded inputs)
        # rp_cleanup.clean(['input_objects']) # Not strictly needed here as we download manually
        print("Cleanup finished.")


    return job_output

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting worker...")
    # Fetch the model tag from environment variable set in Dockerfile
    # Default to Flux.1 Schnell if not set (should match Dockerfile ARG)
    print(f"Initializing predictor for Flux...")

    try:
        MODEL = predict.Predictor()
        MODEL.setup() # Load the model into memory
        print("Predictor initialized successfully.")
    except Exception as e:
        print(f"FATAL: Model initialization failed: {e}")
        traceback.print_exc()
        # If the model fails to load, the worker can't function.
        # You might exit or handle this state appropriately depending on RunPod's behavior.
        # For now, it will likely error out when a job comes in.sc
        # Consider adding a health check endpoint if needed.

    print("Starting RunPod serverless...")
    runpod.serverless.start({"handler": run})