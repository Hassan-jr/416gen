import io
import os
import uuid
import runpod
from runpod.serverless.utils import rp_cleanup
from runpod.serverless.utils.rp_validator import validate
import requests
import boto3
from botocore.config import Config

from rp_schema import INPUT_SCHEMA
import predict

from PIL import Image

MODEL = predict.Predictor()

def download_lora(url: str) -> str:
    path = f"/tmp/{uuid.uuid4()}.safetensors"
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        return path
    except Exception as e:
        print(f"LoRA download failed: {str(e)}")
        raise

def upload_image(image: Image.Image, config: dict, job_id: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    
    s3 = boto3.client(
        's3',
        endpoint_url=config['endpoint_url'],
        aws_access_key_id=config['access_key_id'],
        aws_secret_access_key=config['secret_access_key'],
        config=Config(s3={'addressing_style': 'virtual'}),
        region_name='auto'
    )
    
    key = f"{config.get('path_prefix', '')}/{job_id}_{uuid.uuid4()}.png".lstrip('/')
    s3.put_object(
        Bucket=config['bucket_name'],
        Key=key,
        Body=buffer.getvalue(),
        ContentType='image/png'
    )
    return f"{config['endpoint_url']}/{config['bucket_name']}/{key}"

def run(job):
    job_input = job['input']
    validated = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated:
        return {"error": validated['errors']}
    data = validated['validated_input']
    
    # Handle LoRA
    lora_path = None
    if data.get('lora_url'):
        try:
            lora_path = download_lora(data['lora_url'])
            MODEL.load_lora(lora_path)
        except Exception as e:
            if lora_path and os.path.exists(lora_path):
                os.remove(lora_path)
            return {"error": f"LoRA handling failed: {str(e)}"}
    
    # Process generations
    results = []
    for task in data['generations']:
        try:
            images = MODEL.predict(
                prompt=task['prompt'],
                negative_prompt=task.get('negative_prompt'),
                width=task.get('width', 1024),
                height=task.get('height', 1024),
                num_inference_steps=task.get('num_inference_steps', 4),
                guidance_scale=task.get('guidance_scale', 0.0),
                num_outputs=task.get('num_outputs', 1),
                seed=task.get('seed'),
                lora_scale=data.get('lora_scale', 0.8)
            )
            
            task_result = {
                "prompt": task['prompt'],
                "images": [
                    {"url": upload_image(img, data, job['id']), "seed": seed}
                    for img, seed in images
                ]
            }
            results.append(task_result)
            
        except Exception as e:
            results.append({"error": str(e), "input": task})
    
    # Cleanup
    if lora_path and os.path.exists(lora_path):
        MODEL.unload_lora()
        os.remove(lora_path)
    rp_cleanup.clean(['input_objects'])
    
    return {"instanceId": data['instanceId'], "generations": results}

if __name__ == "__main__":
    MODEL.setup()
    runpod.serverless.start({"handler": run})