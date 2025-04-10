import os
import uuid
import io
import traceback
import requests
from pathlib import Path

import runpod
from runpod.serverless.utils.rp_validator import validate
# Using standard libraries for download/upload now
# from runpod.serverless.utils import rp_download, rp_cleanup # No longer needed for init/mask
from runpod.serverless.utils import rp_cleanup # Still use for cleanup

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
            config=Config(s3={'addressing_style': 'virtual'}), # Often needed for R2
            region_name='auto' # R2 specific
        )

        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=image_bytes,
            ContentType='image/png' # Assuming PNG output
            # Add ACL='public-read' if you want images publicly accessible directly
            # ACL='public-read'
        )
        # Construct the public URL (adjust if you have a custom domain)
        # Basic URL structure: https://<bucket>.<accountid>.r2.cloudflarestorage.com/<key>
        # Extract account ID from endpoint URL if possible, otherwise a simpler structure might be needed
        # Or just return the object key and let the caller construct the URL
        # Simplified approach: return object key + bucket
        # A more robust URL might be needed depending on R2 setup
        # public_url = f"{r2_config['endpoint_url']}/{bucket_name}/{object_key}"
        print("Upload successful.")
        # Return the object key, the caller can decide how to build the URL if needed
        return f"{r2_config['endpoint_url']}/{bucket_name}/{object_key}"

    except ClientError as e:
        print(f"Boto3 ClientError uploading to R2: {e}")
        # Consider logging e.response['Error'] for more details
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
    lora_scale = validated_input.get('lora_scale', 0.8)
    r2_config = {
        'bucket_name': validated_input['r2_bucket_name'],
        'access_key_id': validated_input['r2_access_key_id'],
        'secret_access_key': validated_input['r2_secret_access_key'],
        'endpoint_url': validated_input['r2_endpoint_url'].rstrip('/'), # Ensure no trailing slash
        'path_prefix': validated_input.get('r2_path_in_bucket', '').strip('/') # Ensure no surrounding slashes
    }
    generation_tasks = validated_input['generations']

    # --- Initialize Predictor (if not already) ---
    # This is usually done outside the handler in __main__ for serverless workers
    if MODEL is None:
         print("Error: Model predictor not initialized.")
         return {"error": "Model predictor not initialized."}

    # --- Download and Load LoRA (if provided) ---
    lora_download_path = "/tmp/downloaded_lora.safetensors" # Temporary path
    lora_loaded_successfully = False
    if lora_url:
        print("LoRA URL provided, attempting download...")
        if download_file(lora_url, lora_download_path):
            try:
                MODEL.load_lora(lora_download_path)
                lora_loaded_successfully = True
                print("LoRA loaded successfully.")
            except Exception as e:
                print(f"Error loading LoRA from {lora_download_path}: {e}")
                traceback.print_exc()
                # Decide if you want to proceed without LoRA or return an error
                # return {"error": f"Failed to load LoRA: {e}"}
        else:
            print("LoRA download failed. Proceeding without LoRA.")
            # return {"error": "Failed to download LoRA."} # Or proceed without
    else:
        print("No LoRA URL provided.")
        # Ensure any previously loaded LoRA is unloaded
        try:
            MODEL.unload_lora()
            print("Unloaded any previous LoRAs.")
        except Exception as e:
             print(f"Minor error unloading LoRA (might be none loaded): {e}")


    # --- Process Generation Tasks ---
    job_output = {
        "instanceId": instance_id,
        "generations": []
    }
    temp_image_paths = [] # Keep track of local files

    try:
        for index, task in enumerate(generation_tasks):
            print(f"\nProcessing generation task {index + 1}/{len(generation_tasks)}...")
            prompt = task['prompt']
            num_outputs = task.get('num_outputs', 1)
            seed = task.get('seed') or int.from_bytes(os.urandom(4), "big") # Use 4 bytes for larger seed space

            print(f"  Prompt: {prompt[:100]}...") # Log truncated prompt
            print(f"  Num Outputs: {num_outputs}, Seed: {seed}")

            # Prepare arguments for the predictor
            predict_args = {
                "prompt": prompt,
                "negative_prompt": task.get("negative_prompt", ""),
                "width": task.get('width', 1024),
                "height": task.get('height', 1024),
                "num_inference_steps": task.get('num_inference_steps', 4),
                "guidance_scale": task.get('guidance_scale', 0.0),
                "num_outputs": num_outputs, # Pass num_outputs to predictor
                "seed": seed,
                "lora_scale": lora_scale if lora_loaded_successfully else 0.0 # Apply scale only if LoRA loaded
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
                "lora_url": lora_url, # Record used lora url
                "lora_scale": predict_args["lora_scale"],
                "images": []
            }

            try:
                # Generate images - predictor should handle num_outputs > 1
                # Expect predict to return a list of tuples: (image_path, final_seed)
                generated_images = MODEL.predict(**predict_args) # Returns list of (path, final_seed)

                # Upload each generated image
                for img_path, img_seed in generated_images:
                    temp_image_paths.append(img_path) # Track for cleanup
                    with open(img_path, 'rb') as f_img:
                        image_bytes = f_img.read()

                    # Create a unique name for the image in R2
                    image_filename = f"{job['id']}_task{index}_{uuid.uuid4()}.png"
                    object_key = image_filename
                    if r2_config['path_prefix']:
                       object_key = f"{r2_config['path_prefix']}/{image_filename}" # Add prefix if specified

                    # Upload and get URL/Key
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
                        # Decide how to handle failed uploads (e.g., add an error marker)
                        task_results["images"].append({
                            "url": None,
                            "seed": img_seed,
                            "error": "Upload failed"
                         })

            except Exception as e:
                print(f"Error during prediction or upload for task {index}: {e}")
                traceback.print_exc()
                task_results["error"] = str(e) # Add error to the specific task result

            job_output["generations"].append(task_results)

    finally:
        # --- Cleanup ---
        # Remove downloaded LoRA file
        if lora_url and os.path.exists(lora_download_path):
            try:
                os.remove(lora_download_path)
                print(f"Removed temporary LoRA file: {lora_download_path}")
            except OSError as e:
                print(f"Error removing temporary LoRA file {lora_download_path}: {e}")

        # Remove generated temporary image files
        for path in temp_image_paths:
             if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Error removing temporary image file {path}: {e}")

        # RunPod cleanup (e.g., input objects if they were downloaded, though not used here)
        rp_cleanup.clean(['input_objects'])

        # Unload LoRA weights from the model if loaded
        if lora_loaded_successfully:
             try:
                 MODEL.unload_lora()
                 print("Unloaded LoRA weights from model.")
             except Exception as e:
                 print(f"Error unloading LoRA from model: {e}")


    return job_output

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting worker...")
    # Fetch the model tag from environment variable set in Dockerfile
    # Default to Flux.1 Schnell if not set (should match Dockerfile ARG)
    model_tag = os.environ.get("MODEL_TAG", "black-forest-labs/FLUX.1-schnell")
    print(f"Loading model: {model_tag}")

    MODEL = predict.Predictor(model_tag=model_tag)
    MODEL.setup() # Load the model into memory

    print("Starting RunPod serverless...")
    runpod.serverless.start({"handler": run})