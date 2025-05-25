# # rp_handler.py
# import os
# import uuid
# import io
# import traceback
# import requests
# from pathlib import Path
# import time
# from typing import Optional # For type hinting

# import runpod
# from runpod.serverless.utils.rp_validator import validate
# from runpod.serverless.utils import rp_cleanup

# import boto3
# from botocore.config import Config
# from botocore.exceptions import ClientError, EndpointConnectionError

# from rp_schema import INPUT_SCHEMA
# # Import the *new* Predictor class that uses the full pipeline
# import predict

# # Global predictor instance
# MODEL: Optional[predict.Predictor] = None # Type hint

# # --- download_file function ---
# def download_file(url, save_path):
#     """Downloads a file from a URL to a local path."""
#     print(f"Downloading {url} to {save_path}...")
#     try:
#         response = requests.get(url, stream=True, timeout=300) # 5 min timeout
#         response.raise_for_status() # Raise an exception for bad status codes
#         with open(save_path, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print("Download complete.")
#         return True
#     except requests.exceptions.RequestException as e:
#         print(f"Error downloading {url}: {e}")
#         return False
#     except Exception as e:
#         print(f"An unexpected error occurred during download: {e}")
#         return False

# # --- upload_to_r2 function (with previous fixes) ---
# def upload_to_r2(image_bytes, bucket_name, object_key, r2_config):
#     """
#     Uploads image bytes to Cloudflare R2.
#     Ensures the endpoint_url is the BASE URL (e.g., https://<ACCOUNT_ID>.r2.cloudflarestorage.com)
#     """
#     base_endpoint_url = r2_config['endpoint_url']
#     print(f"Uploading to R2 -> Bucket: {bucket_name}, Key: {object_key}, Base Endpoint: {base_endpoint_url}")

#     # Basic sanity check for the endpoint URL format
#     # Allow common R2 URLs and custom domains if flagged
#     is_likely_r2 = ".r2.cloudflarestorage.com" in base_endpoint_url or "r2.dev" in base_endpoint_url
#     is_custom_domain = r2_config.get('is_custom_domain', False)
#     if not base_endpoint_url.startswith("https://") or (not is_likely_r2 and not is_custom_domain):
#          print(f"WARNING: R2 endpoint URL '{base_endpoint_url}' doesn't look like a standard R2 format or custom domain wasn't flagged.")

#     try:
#         s3_client = boto3.client(
#             's3',
#             endpoint_url=base_endpoint_url, # Use the base endpoint
#             aws_access_key_id=r2_config['access_key_id'],
#             aws_secret_access_key=r2_config['secret_access_key'],
#             config=Config(
#                 s3={'addressing_style': 'virtual'},
#                 retries = {'max_attempts': 3}
#                 ),
#             region_name='auto' # R2 requires 'auto'
#         )

#         s3_client.put_object(
#             Bucket=bucket_name, # Bucket name passed separately
#             Key=object_key,
#             Body=image_bytes,
#             ContentType='image/png'
#         )

#         # Construct the public URL
#         public_url = f"{object_key}"
#         print(f"Upload successful. URL: {public_url}")
#         return public_url

#     except EndpointConnectionError as e:
#         print(f"ERROR: Boto3 EndpointConnectionError - Could not connect to R2 endpoint '{base_endpoint_url}'.")
#         print(f"       Check the endpoint URL, DNS, and worker network connectivity.")
#         print(f"       Error Details: {e}")
#         return None
#     except ClientError as e:
#         error_code = e.response.get('Error', {}).get('Code')
#         print(f"ERROR: Boto3 ClientError uploading to R2 (Code: {error_code}): {e}")
#         return None
#     except Exception as e:
#         print(f"ERROR: Unexpected error uploading to R2: {e}")
#         traceback.print_exc()
#         return None


# # --- Main RunPod Handler ---
# def run(job):
#     '''
#     Run inference using the full FluxPipeline via the simplified Predictor class.
#     '''
#     global MODEL

#     job_input = job['input']

#     # --- Input Validation ---
#     validated_input = validate(job_input, INPUT_SCHEMA)
#     if 'errors' in validated_input:
#         return {"error": validated_input['errors']}
#     validated_input = validated_input['validated_input']

#     # --- Extract Config ---
#     instance_id = validated_input['instanceId']
#     lora_url = validated_input.get('lora_url')
#     lora_scale_input = validated_input.get('lora_scale', 0.8)

#     # --- R2 Config ---
#     r2_endpoint_url_input = validated_input['r2_endpoint_url']
#     if not r2_endpoint_url_input or not r2_endpoint_url_input.startswith("https"):
#          return {"error": "Invalid or missing 'r2_endpoint_url'. It must be the base HTTPS URL for your R2 account (e.g., https://<accountid>.r2.cloudflarestorage.com)."}

#     r2_config = {
#         'bucket_name': validated_input['r2_bucket_name'],
#         'access_key_id': validated_input['r2_access_key_id'],
#         'secret_access_key': validated_input['r2_secret_access_key'],
#         'endpoint_url': r2_endpoint_url_input.rstrip('/'),
#         'path_prefix': validated_input.get('r2_path_in_bucket', '').strip('/'),
#         # 'is_custom_domain': True # Set if using a custom domain endpoint
#     }
#     generation_tasks = validated_input['generations']

#     # --- Initialize Predictor (Loads full pipeline) ---
#     if MODEL is None:
#          print("Predictor not initialized. Initializing now (loading full pipeline)...")
#          try:
#              MODEL = predict.Predictor()
#              MODEL.setup() # This now loads the full pipeline
#              print("Predictor initialized and pipeline loaded.")
#          except Exception as e:
#             # If setup fails, the worker is unusable. Log and return error.
#             print(f"FATAL: Failed to initialize predictor/load pipeline: {e}")
#             traceback.print_exc()
#             return {"error": f"Model predictor failed to initialize: {e}"}
#     # --- End Initialization ---


#     # --- Download LoRA (if needed) ---
#     # LoRA is loaded/unloaded *inside* the predict method based on the path
#     lora_download_path = None
#     lora_file_available = False
#     temp_lora_file_to_clean = None

#     if lora_url:
#         print("LoRA URL provided, attempting download...")
#         # Use a unique temp filename
#         temp_lora_filename = f"/tmp/downloaded_lora_{uuid.uuid4()}.safetensors"
#         if download_file(lora_url, temp_lora_filename):
#             lora_download_path = temp_lora_filename # Store path for predict()
#             lora_file_available = True
#             temp_lora_file_to_clean = temp_lora_filename # Mark for cleanup
#             print(f"LoRA downloaded successfully to {lora_download_path}")
#         else:
#             print("Warning: LoRA download failed. Will proceed without LoRA.")
#             # Consider returning an error if LoRA is essential for the job
#             # return {"error": f"Failed to download required LoRA from {lora_url}"}
#     else:
#         print("No LoRA URL provided.")
#     # --- End LoRA Download ---


#     # --- Process Generation Tasks ---
#     job_output = {
#         "instanceId": instance_id,
#         "generations": []
#     }
#     temp_image_paths_to_clean = [] # Track only successfully saved local images

#     try:
#         overall_start_time = time.time()
#         for index, task in enumerate(generation_tasks):
#             task_start_time = time.time()
#             print(f"\nProcessing generation task {index + 1}/{len(generation_tasks)}...")
#             prompt = task['prompt']
#             num_inference_steps = task['num_inference_steps']
#             num_outputs = task.get('num_outputs', 1)
#             seed = task.get('seed') if task.get('seed') is not None else int.from_bytes(os.urandom(4), "big")

#             # Ensure minimum steps, default to predictor's minimum if not specified
#             # num_inference_steps_input = task.get('num_inference_steps', predict.MIN_INFERENCE_STEPS)
#             # num_inference_steps = max(num_inference_steps_input, predict.MIN_INFERENCE_STEPS)
#             # if num_inference_steps != num_inference_steps_input:
#             #      print(f"  Adjusted inference steps from {num_inference_steps_input} to {num_inference_steps}")

#             print(f"  Prompt: {prompt[:100]}...")
#             print(f"  Num Outputs: {num_outputs}, Seed: {seed}, Steps: {num_inference_steps}")
#             print(f"  LoRA File Available: {lora_file_available}, Requested Scale: {lora_scale_input}")

#             # Prepare arguments for the simplified predict method
#             predict_args = {
#                 "prompt": prompt,
#                 # negative_prompt and guidance_scale are ignored by predictor now
#                 "width": task.get('width', 1024),
#                 "height": task.get('height', 1024),
#                 "num_inference_steps": num_inference_steps,
#                 "num_outputs": num_outputs,
#                 "seed": seed,
#                 "lora_path": lora_download_path if lora_file_available else None,
#                 "lora_scale": lora_scale_input,
#             }

#             # Setup results dict, recording input params
#             task_results = {
#                 "prompt": prompt,
#                 "negative_prompt": task.get("negative_prompt", ""), # Record input np
#                 "width": predict_args["width"],
#                 "height": predict_args["height"],
#                 "num_outputs": num_outputs,
#                 "num_inference_steps": num_inference_steps, # Record actual steps used
#                 "guidance_scale": task.get('guidance_scale', 0.0), # Record input gs
#                 "seed": seed, # Record base seed for the task
#                 "lora_url": lora_url if lora_file_available else None,
#                 "lora_scale_requested": lora_scale_input,
#                 "images": [], # Will contain {url, seed, error} dicts
#                 "task_duration_seconds": None,
#                 "error": None # For errors *before* or *during* predict call itself
#             }

#             try:
#                 # Call the predictor's predict method
#                 # It now returns only successfully generated (path, seed) tuples
#                 generated_images = MODEL.predict(**predict_args)

#                 if not generated_images and num_outputs > 0:
#                      print("Warning: Predictor returned no successful images for this task.")
#                      # If no predict error, record generation failure.
#                      if task_results["error"] is None:
#                         task_results["error"] = "Image generation pipeline failed or produced no output."
#                      # Add placeholders for expected outputs
#                      for i in range(num_outputs):
#                           task_results["images"].append({
#                                 "url": None,
#                                 "seed": seed + i, # Approximate seed
#                                 "error": task_results["error"]
#                           })

#                 # --- Upload Loop for Successfully Generated Images ---
#                 for img_path, img_seed in generated_images:
#                     # Add path to cleanup list *only if* successful generation
#                     temp_image_paths_to_clean.append(img_path)
#                     try:
#                         # Read the generated image file
#                         with open(img_path, 'rb') as f_img:
#                             image_bytes = f_img.read()

#                         # Create a unique name for the image in R2
#                         image_filename = f"{instance_id}_task{index}_seed{img_seed}_{uuid.uuid4()}.png"
#                         object_key = image_filename
#                         if r2_config['path_prefix']:
#                            object_key = f"{r2_config['path_prefix']}/{image_filename}"

#                         # Upload and get URL
#                         r2_url = upload_to_r2(
#                             image_bytes=image_bytes,
#                             bucket_name=r2_config['bucket_name'],
#                             object_key=object_key,
#                             r2_config=r2_config # Pass the whole config dict
#                         )

#                         if r2_url:
#                             # Success: Add URL and specific seed to results
#                             task_results["images"].append({"url": r2_url, "seed": img_seed})
#                         else:
#                             # Upload failure (upload_to_r2 logs details)
#                             print(f"Upload failed for image from seed {img_seed}.")
#                             task_results["images"].append({"url": None, "seed": img_seed, "error": "Upload failed"})

#                     except FileNotFoundError:
#                          # Should ideally not happen if predict returns valid paths
#                          print(f"Error: Predictor returned path {img_path} but file not found.")
#                          task_results["images"].append({"url": None, "seed": img_seed, "error": "Generated file not found post-generation"})
#                     except Exception as upload_err:
#                          # Catch errors during file reading or upload call preparation
#                          print(f"Error processing/uploading image from seed {img_seed}: {upload_err}")
#                          traceback.print_exc()
#                          task_results["images"].append({"url": None, "seed": img_seed, "error": f"Upload process failed: {upload_err}"})
#                 # --- End Upload Loop ---

#             except Exception as e:
#                 # Catch errors during the MODEL.predict() call itself
#                 print(f"Error during prediction execution for task {index}: {e}")
#                 traceback.print_exc()
#                 task_results["error"] = f"Prediction failed: {str(e)}"
#                  # Add placeholders for expected outputs if predict fails entirely
#                 if not task_results["images"]: # Only if no images were added yet
#                     for i in range(num_outputs):
#                           task_results["images"].append({
#                                 "url": None, "seed": seed + i, "error": task_results["error"]
#                           })


#             task_end_time = time.time()
#             task_results["task_duration_seconds"] = round(task_end_time - task_start_time, 2)
#             job_output["generations"].append(task_results)
#             print(f"Task {index+1} processing finished.")

#         overall_end_time = time.time()
#         print(f"\nFinished all tasks in {round(overall_end_time - overall_start_time, 2)} seconds.")

#     finally:
#         # --- Cleanup ---
#         print("Starting cleanup...")
#         # Cleanup downloaded LoRA file
#         if temp_lora_file_to_clean and os.path.exists(temp_lora_file_to_clean):
#             try:
#                 os.remove(temp_lora_file_to_clean)
#                 print(f"Removed temporary LoRA file: {temp_lora_file_to_clean}")
#             except OSError as e:
#                 print(f"Error removing temporary LoRA file {temp_lora_file_to_clean}: {e}")

#         # Cleanup successfully generated local image files
#         cleaned_images = 0
#         for path in temp_image_paths_to_clean:
#             if os.path.exists(path): # Check existence again just in case
#                 try:
#                     os.remove(path)
#                     cleaned_images += 1
#                 except OSError as e:
#                     print(f"Error removing temporary image file {path}: {e}")
#         print(f"Removed {cleaned_images} successfully generated temporary image files.")
#         # RunPod cleanup (usually not needed with manual handling)
#         # rp_cleanup.clean(['input_objects'])
#         print("Cleanup finished.")
#         # --- End Cleanup ---

#     return job_output

# # --- Main Execution ---
# if __name__ == "__main__":
#     print("Starting worker...")
#     print(f"Initializing predictor (loading FULL pipeline)...") # Updated message

#     # Initialize and setup the predictor globally
#     try:
#         MODEL = predict.Predictor()
#         MODEL.setup() # This loads the full pipeline now
#         print("Predictor initialized and pipeline loaded successfully.")
#     except Exception as e:
#         print(f"FATAL: Model initialization failed during setup: {e}")
#         traceback.print_exc()
#         # If the model fails to load, the worker cannot function.
#         # Option: Exit or rely on RunPod termination.
#         # sys.exit(1) # Uncomment to force exit on init failure

#     # Start the RunPod serverless handler
#     print("Starting RunPod serverless handler...")
#     runpod.serverless.start({"handler": run})

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
# from runpod.serverless.utils import rp_cleanup # Not strictly needed with manual file handling

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from rp_schema import INPUT_SCHEMA # Make sure you have this file with your schema
import predict # Your predict.py

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
        print(f"An unexpected error occurred during download of {url}: {e}")
        return False

# --- upload_to_r2 function ---
def upload_to_r2(image_bytes, bucket_name, object_key, r2_config):
    """Uploads image bytes to Cloudflare R2."""
    base_endpoint_url = r2_config['endpoint_url']
    print(f"Uploading to R2 -> Bucket: {bucket_name}, Key: {object_key}, Base Endpoint: {base_endpoint_url}")

    is_likely_r2 = ".r2.cloudflarestorage.com" in base_endpoint_url or "r2.dev" in base_endpoint_url
    is_custom_domain = r2_config.get('is_custom_domain', False)
    if not base_endpoint_url.startswith("https://") or (not is_likely_r2 and not is_custom_domain):
         print(f"WARNING: R2 endpoint URL '{base_endpoint_url}' doesn't look like a standard R2 format or custom domain wasn't flagged.")

    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=base_endpoint_url,
            aws_access_key_id=r2_config['access_key_id'],
            aws_secret_access_key=r2_config['secret_access_key'],
            config=Config(s3={'addressing_style': 'virtual'}, retries={'max_attempts': 3}),
            region_name='auto' # R2 specific
        )
        s3_client.put_object(
            Bucket=bucket_name, Key=object_key, Body=image_bytes, ContentType='image/png'
        )
        # Construct the public URL (ensure your R2 bucket has public access configured if needed)
        public_url = f"{base_endpoint_url}/{bucket_name}/{object_key}"
        print(f"Upload successful. URL: {public_url}")
        return public_url
    except EndpointConnectionError as e:
        print(f"ERROR: Boto3 EndpointConnectionError connecting to R2 '{base_endpoint_url}'. Details: {e}")
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
    global MODEL

    job_input = job['input']
    job_id = job.get('id', f"local_job_{uuid.uuid4()}") # Get job ID for logging or generate one
    print(f"\n--- Received Job ID: {job_id} ---")

    # --- Input Validation ---
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        print(f"Job {job_id}: Input validation failed: {validated_input['errors']}")
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']
    print(f"Job {job_id}: Input validated successfully.")

    # --- Extract Config ---
    instance_id = validated_input['instanceId']
    lora_url = validated_input.get('lora_url')
    lora_scale_input = validated_input.get('lora_scale', 0.8) # Default LoRA scale if not provided

    # --- R2 Config ---
    r2_endpoint_url_input = validated_input['r2_endpoint_url']
    if not r2_endpoint_url_input or not r2_endpoint_url_input.startswith("https"):
         print(f"Job {job_id}: Invalid R2 endpoint URL: {r2_endpoint_url_input}")
         return {"error": "Invalid or missing 'r2_endpoint_url'. Must be base HTTPS URL."}
    r2_config = {
        'bucket_name': validated_input['r2_bucket_name'],
        'access_key_id': validated_input['r2_access_key_id'],
        'secret_access_key': validated_input['r2_secret_access_key'],
        'endpoint_url': r2_endpoint_url_input.rstrip('/'),
        'path_prefix': validated_input.get('r2_path_in_bucket', '').strip('/'),
    }
    generation_tasks = validated_input['generations']

    # --- Initialize Predictor (if not already done or if initial setup failed) ---
    if MODEL is None:
         print(f"Job {job_id}: Global MODEL is None. Attempting lazy initialization...")
         try:
             temp_model_instance = predict.Predictor()
             temp_model_instance.setup() # This now loads the full pipeline
             MODEL = temp_model_instance # Assign to global MODEL only on success
             print(f"Job {job_id}: Predictor lazy initialized and pipeline loaded.")
         except Exception as e:
            print(f"Job {job_id}: FATAL - Lazy initialization of predictor/pipeline FAILED: {e}")
            traceback.print_exc()
            return {"error": f"Model predictor failed to initialize during job execution: {e}"}
    else:
        # Optional: Check if the existing MODEL.pipe is valid, though Predictor.predict() will do this.
        if MODEL.pipe is None:
            print(f"Job {job_id}: Global MODEL exists but its pipe is None. This indicates a past setup failure. Attempting re-initialization...")
            try:
                MODEL.setup() # Try to re-run setup on the existing instance
                if MODEL.pipe is None: # If still None after re-setup
                    raise RuntimeError("Re-setup of existing MODEL failed to initialize the pipe.")
                print(f"Job {job_id}: Re-setup of MODEL successful.")
            except Exception as e:
                print(f"Job {job_id}: FATAL - Re-initialization of existing MODEL FAILED: {e}")
                traceback.print_exc()
                return {"error": f"Model predictor re-initialization failed: {e}"}
        else:
            print(f"Job {job_id}: Global MODEL already initialized and pipe seems valid. Using existing instance.")
    # --- End Initialization ---

    # --- Download LoRA (if needed) ---
    lora_download_path = None
    lora_file_available = False
    temp_lora_file_to_clean = None

    if lora_url:
        print(f"Job {job_id}: LoRA URL provided ({lora_url}), attempting download...")
        # Use a unique temp filename in /tmp
        temp_lora_filename = f"/tmp/downloaded_lora_{uuid.uuid4()}.safetensors" # Ensure /tmp exists
        if download_file(lora_url, temp_lora_filename):
            lora_download_path = temp_lora_filename
            lora_file_available = True
            temp_lora_file_to_clean = temp_lora_filename
            print(f"Job {job_id}: LoRA downloaded successfully to {lora_download_path}")
        else:
            print(f"Job {job_id}: Warning - LoRA download failed from {lora_url}. Will proceed without LoRA.")
    else:
        print(f"Job {job_id}: No LoRA URL provided.")
    # --- End LoRA Download ---

    # --- Process Generation Tasks ---
    job_output = {"instanceId": instance_id, "generations": []}
    temp_image_paths_to_clean = [] # Track successfully saved local images for cleanup

    try:
        overall_start_time = time.time()
        for index, task in enumerate(generation_tasks):
            task_start_time = time.time()
            print(f"\nJob {job_id}: Processing generation task {index + 1}/{len(generation_tasks)}...")
            prompt = task['prompt']
            num_outputs = task.get('num_outputs', 1)
            raw_seed = task.get('seed')
            seed = int(raw_seed) if raw_seed is not None else None # Predictor handles None seed by generating one

            # predict.MIN_INFERENCE_STEPS must be defined in predict.py
            num_inference_steps_input = task.get('num_inference_steps', predict.MIN_INFERENCE_STEPS)
            num_inference_steps = max(num_inference_steps_input, predict.MIN_INFERENCE_STEPS)
            if num_inference_steps != num_inference_steps_input:
                 print(f"  Job {job_id}: Adjusted inference steps from {num_inference_steps_input} to {num_inference_steps} (min: {predict.MIN_INFERENCE_STEPS})")

            print(f"  Job {job_id}: Task Params - Prompt: '{prompt[:50]}...', Outputs: {num_outputs}, Seed: {seed}, Steps: {num_inference_steps}")
            print(f"  Job {job_id}: LoRA File Available: {lora_file_available}, Requested LoRA Scale: {lora_scale_input}")

            predict_args = {
                "prompt": prompt,
                "negative_prompt": task.get("negative_prompt", ""),
                "width": task.get('width', 1024),
                "height": task.get('height', 1024),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": task.get('guidance_scale', 4.0), # Default for Flux, adjust if needed
                "num_outputs": num_outputs,
                "seed": seed,
                "lora_path": lora_download_path if lora_file_available else None,
                "lora_scale": lora_scale_input,
            }

            task_results = {
                "prompt": prompt, "negative_prompt": predict_args["negative_prompt"],
                "width": predict_args["width"], "height": predict_args["height"],
                "num_outputs": num_outputs, "num_inference_steps": num_inference_steps,
                "guidance_scale": predict_args["guidance_scale"], "seed": seed, # Base seed for the task
                "lora_url": lora_url if lora_file_available else None,
                "lora_scale_requested": lora_scale_input,
                "images": [], "task_duration_seconds": None, "error": None
            }

            try:
                generated_images_info = MODEL.predict(**predict_args) # Returns list of (path, actual_seed)

                if not generated_images_info and num_outputs > 0:
                     print(f"  Job {job_id}: Warning - Predictor returned no successful images for task {index+1}.")
                     task_results["error"] = "Image generation produced no output or failed internally."
                     for i in range(num_outputs): # Add placeholders
                          task_results["images"].append({
                                "url": None, "seed": (seed if seed is not None else i), "error": task_results["error"]
                          })

                for img_path, img_actual_seed in generated_images_info:
                    temp_image_paths_to_clean.append(img_path) # Mark for cleanup
                    try:
                        with open(img_path, 'rb') as f_img: image_bytes = f_img.read()
                        image_filename = f"{instance_id}_task{index}_seed{img_actual_seed}_{uuid.uuid4()}.png"
                        object_key = f"{r2_config['path_prefix']}/{image_filename}" if r2_config['path_prefix'] else image_filename
                        r2_url = upload_to_r2(image_bytes, r2_config['bucket_name'], object_key, r2_config)
                        if r2_url:
                            task_results["images"].append({"url": r2_url, "seed": img_actual_seed})
                        else:
                            print(f"  Job {job_id}: Upload to R2 failed for image with seed {img_actual_seed}.")
                            task_results["images"].append({"url": None, "seed": img_actual_seed, "error": "Upload to R2 failed"})
                    except FileNotFoundError:
                         print(f"  Job {job_id}: Generated file {img_path} (seed {img_actual_seed}) not found for upload.")
                         task_results["images"].append({"url": None, "seed": img_actual_seed, "error": f"Generated file {img_path} not found."})
                    except Exception as upload_err:
                         print(f"  Job {job_id}: Error during upload process for image with seed {img_actual_seed}: {upload_err}")
                         traceback.print_exc()
                         task_results["images"].append({"url": None, "seed": img_actual_seed, "error": f"Upload process failed: {upload_err}"})

            except RuntimeError as e: # Catch "Predictor not set up" specifically or other critical runtime issues
                print(f"  Job {job_id}: CRITICAL ERROR during MODEL.predict() for task {index+1}: {e}")
                traceback.print_exc()
                task_results["error"] = f"Prediction failed critically: {str(e)}"
                for i in range(num_outputs):
                      task_results["images"].append({
                            "url": None, "seed": (seed if seed is not None else i), "error": task_results["error"]
                      })
            except Exception as e:
                print(f"  Job {job_id}: Error during MODEL.predict() execution for task {index+1}: {e}")
                traceback.print_exc()
                task_results["error"] = f"Prediction failed: {str(e)}"
                for i in range(num_outputs): # Add placeholders
                      task_results["images"].append({
                            "url": None, "seed": (seed if seed is not None else i), "error": task_results["error"]
                      })

            task_end_time = time.time()
            task_results["task_duration_seconds"] = round(task_end_time - task_start_time, 2)
            job_output["generations"].append(task_results)
            print(f"  Job {job_id}: Task {index+1} processing finished in {task_results['task_duration_seconds']}s.")

        overall_end_time = time.time()
        print(f"\nJob {job_id}: Finished all {len(generation_tasks)} tasks in {round(overall_end_time - overall_start_time, 2)} seconds.")

    finally:
        print(f"\nJob {job_id}: Starting cleanup...")
        if temp_lora_file_to_clean and os.path.exists(temp_lora_file_to_clean):
            try:
                os.remove(temp_lora_file_to_clean)
                print(f"  Job {job_id}: Removed temporary LoRA file: {temp_lora_file_to_clean}")
            except OSError as e:
                print(f"  Job {job_id}: Error removing temp LoRA {temp_lora_file_to_clean}: {e}")
        cleaned_images = 0
        for path in temp_image_paths_to_clean:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    cleaned_images += 1
                except OSError as e:
                    print(f"  Job {job_id}: Error removing temp image {path}: {e}")
        print(f"  Job {job_id}: Removed {cleaned_images} temporary image files.")
        print(f"Job {job_id}: Cleanup finished.")
    print(f"--- Job ID: {job_id} Completed ---")
    return job_output

# --- Main Execution (for worker startup) ---
if __name__ == "__main__":
    print("--- Worker Starting ---")
    # Attempt to initialize the model globally at startup
    initialized_model_successfully = False
    try:
        print("Attempting to initialize global predictor (loading FULL pipeline)...")
        temp_model_instance = predict.Predictor()
        temp_model_instance.setup() # This loads the model and assigns to temp_model_instance.pipe
        MODEL = temp_model_instance # Assign to global MODEL only on success
        print("Global predictor initialized and pipeline loaded successfully.")
        initialized_model_successfully = True
    except Exception as e:
        print(f"FATAL: Global model initialization FAILED during startup: {e}")
        traceback.print_exc()
        MODEL = None # Explicitly ensure MODEL is None if global setup fails

    if not initialized_model_successfully:
        print("WARNING: Global model initialization failed. Worker will attempt lazy load on first job.")
        # If the model is critical for the worker to even start, you might want to exit:
        # import sys
        # print("Exiting due to critical model initialization failure at startup.")
        # sys.exit(1)

    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": run, "concurrency_modifier": lambda x:1}) # Set concurrency to 1 if model is large and not thread-safe for setup
    print("--- Worker Exited ---") # This line might not be reached if runpod.serverless.start() blocks indefinitely