# rp_handler.py
import os
import uuid
import io
import traceback
import requests
from pathlib import Path
import time
from typing import Optional # For type hinting

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_cleanup
from runpod.serverless.utils.rp_cleanup import clean


import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from rp_schema import INPUT_SCHEMA
# Import the *new* Predictor class that uses the full pipeline
import predict

# Global predictor instance
MODEL: Optional[predict.Predictor] = None # Type hint

# --- download_file function ---
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

# --- upload_to_r2 function (with previous fixes) ---
def upload_to_r2(image_bytes, bucket_name, object_key, r2_config):
    """
    Uploads image bytes to Cloudflare R2.
    Ensures the endpoint_url is the BASE URL (e.g., https://<ACCOUNT_ID>.r2.cloudflarestorage.com)
    """
    base_endpoint_url = r2_config['endpoint_url']
    print(f"Uploading to R2 -> Bucket: {bucket_name}, Key: {object_key}, Base Endpoint: {base_endpoint_url}")

    # Basic sanity check for the endpoint URL format
    # Allow common R2 URLs and custom domains if flagged
    is_likely_r2 = ".r2.cloudflarestorage.com" in base_endpoint_url or "r2.dev" in base_endpoint_url
    is_custom_domain = r2_config.get('is_custom_domain', False)
    if not base_endpoint_url.startswith("https://") or (not is_likely_r2 and not is_custom_domain):
         print(f"WARNING: R2 endpoint URL '{base_endpoint_url}' doesn't look like a standard R2 format or custom domain wasn't flagged.")

    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=base_endpoint_url, # Use the base endpoint
            aws_access_key_id=r2_config['access_key_id'],
            aws_secret_access_key=r2_config['secret_access_key'],
            config=Config(
                s3={'addressing_style': 'virtual'},
                retries = {'max_attempts': 3}
                ),
            region_name='auto' # R2 requires 'auto'
        )

        s3_client.put_object(
            Bucket=bucket_name, # Bucket name passed separately
            Key=object_key,
            Body=image_bytes,
            ContentType='image/png'
        )

        # Construct the public URL
        public_url = f"{object_key}"
        print(f"Upload successful. URL: {public_url}")
        return public_url

    except EndpointConnectionError as e:
        print(f"ERROR: Boto3 EndpointConnectionError - Could not connect to R2 endpoint '{base_endpoint_url}'.")
        print(f"       Check the endpoint URL, DNS, and worker network connectivity.")
        print(f"       Error Details: {e}")
        return None
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        print(f"ERROR: Boto3 ClientError uploading to R2 (Code: {error_code}): {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error uploading to R2: {e}")
        traceback.print_exc()
        return None


# --- Main RunPod Handler ---
def run(job):
    '''
    Run inference using the full FluxPipeline via the simplified Predictor class.
    '''
    global MODEL

    job_input = job['input']

    # --- Input Validation ---
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # --- Extract Config ---
    instance_id = validated_input['instanceId']
    lora_url = validated_input.get('lora_url')
    lora_scale_input = validated_input.get('lora_scale', 0.8)

    # --- R2 Config ---
    r2_endpoint_url_input = validated_input['r2_endpoint_url']
    if not r2_endpoint_url_input or not r2_endpoint_url_input.startswith("https"):
         return {"error": "Invalid or missing 'r2_endpoint_url'. It must be the base HTTPS URL for your R2 account (e.g., https://<accountid>.r2.cloudflarestorage.com)."}

    r2_config = {
        'bucket_name': validated_input['r2_bucket_name'],
        'access_key_id': validated_input['r2_access_key_id'],
        'secret_access_key': validated_input['r2_secret_access_key'],
        'endpoint_url': r2_endpoint_url_input.rstrip('/'),
        'path_prefix': validated_input.get('r2_path_in_bucket', '').strip('/'),
        # 'is_custom_domain': True # Set if using a custom domain endpoint
    }
    generation_tasks = validated_input['generations']

    # --- Initialize Predictor (Loads full pipeline) ---
    if MODEL is None:
         print("Predictor not initialized. Initializing now (loading full pipeline)...")
         try:
             MODEL = predict.Predictor()
             MODEL.setup() # This now loads the full pipeline
             print("Predictor initialized and pipeline loaded.")
         except Exception as e:
            # If setup fails, the worker is unusable. Log and return error.
            print(f"FATAL: Failed to initialize predictor/load pipeline: {e}")
            traceback.print_exc()
            return {"error": f"Model predictor failed to initialize: {e}"}
    # --- End Initialization ---


    # --- Download LoRA (if needed) ---
    # LoRA is loaded/unloaded *inside* the predict method based on the path
    lora_download_path = None
    lora_file_available = False
    temp_lora_file_to_clean = None

    if lora_url:
        print("LoRA URL provided, attempting download...")
        # Use a unique temp filename
        temp_lora_filename = f"/tmp/downloaded_lora_{uuid.uuid4()}.safetensors"
        if download_file(lora_url, temp_lora_filename):
            lora_download_path = temp_lora_filename # Store path for predict()
            lora_file_available = True
            temp_lora_file_to_clean = temp_lora_filename # Mark for cleanup
            print(f"LoRA downloaded successfully to {lora_download_path}")
        else:
            print("Warning: LoRA download failed. Will proceed without LoRA.")
            # Consider returning an error if LoRA is essential for the job
            # return {"error": f"Failed to download required LoRA from {lora_url}"}
    else:
        print("No LoRA URL provided.")
    # --- End LoRA Download ---


    # --- Process Generation Tasks ---
    job_output = {
        "instanceId": instance_id,
        "generations": []
    }
    temp_image_paths_to_clean = [] # Track only successfully saved local images

    try:
        overall_start_time = time.time()
        for index, task in enumerate(generation_tasks):
            task_start_time = time.time()
            print(f"\nProcessing generation task {index + 1}/{len(generation_tasks)}...")
            prompt = task['prompt']
            num_inference_steps = task['num_inference_steps']
            num_outputs = task.get('num_outputs', 1)
            seed = task.get('seed') if task.get('seed') is not None else int.from_bytes(os.urandom(4), "big")

            # Ensure minimum steps, default to predictor's minimum if not specified
            # num_inference_steps_input = task.get('num_inference_steps', predict.MIN_INFERENCE_STEPS)
            # num_inference_steps = max(num_inference_steps_input, predict.MIN_INFERENCE_STEPS)
            # if num_inference_steps != num_inference_steps_input:
            #      print(f"  Adjusted inference steps from {num_inference_steps_input} to {num_inference_steps}")

            print(f"  Prompt: {prompt[:100]}...")
            print(f"  Num Outputs: {num_outputs}, Seed: {seed}, Steps: {num_inference_steps}")
            print(f"  LoRA File Available: {lora_file_available}, Requested Scale: {lora_scale_input}")

            # Prepare arguments for the simplified predict method
            predict_args = {
                "prompt": prompt,
                # negative_prompt and guidance_scale are ignored by predictor now
                "width": task.get('width', 1024),
                "height": task.get('height', 1024),
                "num_inference_steps": num_inference_steps,
                "num_outputs": num_outputs,
                "seed": seed,
                "lora_path": lora_download_path if lora_file_available else None,
                "lora_scale": lora_scale_input,
            }

            # Setup results dict, recording input params
            task_results = {
                "prompt": prompt,
                "negative_prompt": task.get("negative_prompt", ""), # Record input np
                "width": predict_args["width"],
                "height": predict_args["height"],
                "num_outputs": num_outputs,
                "num_inference_steps": num_inference_steps, # Record actual steps used
                "guidance_scale": task.get('guidance_scale', 0.0), # Record input gs
                "seed": seed, # Record base seed for the task
                "lora_url": lora_url if lora_file_available else None,
                "lora_scale_requested": lora_scale_input,
                "images": [], # Will contain {url, seed, error} dicts
                "task_duration_seconds": None,
                "error": None # For errors *before* or *during* predict call itself
            }

            try:
                # Call the predictor's predict method
                # It now returns only successfully generated (path, seed) tuples
                generated_images = MODEL.predict(**predict_args)

                if not generated_images and num_outputs > 0:
                     print("Warning: Predictor returned no successful images for this task.")
                     # If no predict error, record generation failure.
                     if task_results["error"] is None:
                        task_results["error"] = "Image generation pipeline failed or produced no output."
                     # Add placeholders for expected outputs
                     for i in range(num_outputs):
                          task_results["images"].append({
                                "url": None,
                                "seed": seed + i, # Approximate seed
                                "error": task_results["error"]
                          })

                # --- Upload Loop for Successfully Generated Images ---
                for img_path, img_seed in generated_images:
                    # Add path to cleanup list *only if* successful generation
                    temp_image_paths_to_clean.append(img_path)
                    try:
                        # Read the generated image file
                        with open(img_path, 'rb') as f_img:
                            image_bytes = f_img.read()

                        # Create a unique name for the image in R2
                        image_filename = f"{instance_id}_task{index}_seed{img_seed}_{uuid.uuid4()}.png"
                        object_key = image_filename
                        if r2_config['path_prefix']:
                           object_key = f"{r2_config['path_prefix']}/{image_filename}"

                        # Upload and get URL
                        r2_url = upload_to_r2(
                            image_bytes=image_bytes,
                            bucket_name=r2_config['bucket_name'],
                            object_key=object_key,
                            r2_config=r2_config # Pass the whole config dict
                        )

                        if r2_url:
                            # Success: Add URL and specific seed to results
                            task_results["images"].append({"url": r2_url, "seed": img_seed})
                        else:
                            # Upload failure (upload_to_r2 logs details)
                            print(f"Upload failed for image from seed {img_seed}.")
                            task_results["images"].append({"url": None, "seed": img_seed, "error": "Upload failed"})

                    except FileNotFoundError:
                         # Should ideally not happen if predict returns valid paths
                         print(f"Error: Predictor returned path {img_path} but file not found.")
                         task_results["images"].append({"url": None, "seed": img_seed, "error": "Generated file not found post-generation"})
                    except Exception as upload_err:
                         # Catch errors during file reading or upload call preparation
                         print(f"Error processing/uploading image from seed {img_seed}: {upload_err}")
                         traceback.print_exc()
                         task_results["images"].append({"url": None, "seed": img_seed, "error": f"Upload process failed: {upload_err}"})
                # --- End Upload Loop ---

            except Exception as e:
                # Catch errors during the MODEL.predict() call itself
                print(f"Error during prediction execution for task {index}: {e}")
                traceback.print_exc()
                task_results["error"] = f"Prediction failed: {str(e)}"
                 # Add placeholders for expected outputs if predict fails entirely
                if not task_results["images"]: # Only if no images were added yet
                    for i in range(num_outputs):
                          task_results["images"].append({
                                "url": None, "seed": seed + i, "error": task_results["error"]
                          })


            task_end_time = time.time()
            task_results["task_duration_seconds"] = round(task_end_time - task_start_time, 2)
            job_output["generations"].append(task_results)
            print(f"Task {index+1} processing finished.")

        overall_end_time = time.time()
        print(f"\nFinished all tasks in {round(overall_end_time - overall_start_time, 2)} seconds.")

    finally:
        # --- Cleanup ---
        print("Starting cleanup...")
        # Cleanup downloaded LoRA file
        if temp_lora_file_to_clean and os.path.exists(temp_lora_file_to_clean):
            try:
                os.remove(temp_lora_file_to_clean)
                print(f"Removed temporary LoRA file: {temp_lora_file_to_clean}")
            except OSError as e:
                print(f"Error removing temporary LoRA file {temp_lora_file_to_clean}: {e}")

        # Cleanup successfully generated local image files
        cleaned_images = 0
        for path in temp_image_paths_to_clean:
            if os.path.exists(path): # Check existence again just in case
                try:
                    os.remove(path)
                    cleaned_images += 1
                except OSError as e:
                    print(f"Error removing temporary image file {path}: {e}")
        print(f"Removed {cleaned_images} successfully generated temporary image files.")
        # RunPod cleanup (usually not needed with manual handling)
        # rp_cleanup.clean(['input_objects'])
        print("Cleanup finished.")
        print("@@@ Removing TEMP AGAIN @@@")
        clean(folder_list=["/tmp/"])
        # --- End Cleanup ---

    return job_output

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting worker...")
    print(f"Initializing predictor (loading FULL pipeline)...") # Updated message

    # Initialize and setup the predictor globally
    try:
        MODEL = predict.Predictor()
        MODEL.setup() # This loads the full pipeline now
        print("Predictor initialized and pipeline loaded successfully.")
    except Exception as e:
        print(f"FATAL: Model initialization failed during setup: {e}")
        traceback.print_exc()
        # If the model fails to load, the worker cannot function.
        # Option: Exit or rely on RunPod termination.
        # sys.exit(1) # Uncomment to force exit on init failure

    # Start the RunPod serverless handler
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": run})