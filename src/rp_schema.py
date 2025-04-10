INPUT_SCHEMA = {
    'instanceId': {
        'type': str,
        'required': True
    },
    'lora_url': {
        'type': str,
        'required': False, # Make it optional if no LoRA is needed
        'default': None
    },
    'lora_scale': {
        'type': float,
        'required': False,
        'default': 0.8, # Adjust default as needed for Flux
        'constraints': lambda scale: 0.0 <= scale <= 2.0 # Adjust range if needed
    },
    'r2_bucket_name': {
        'type': str,
        'required': True
    },
    'r2_access_key_id': {
        'type': str,
        'required': True
    },
    'r2_secret_access_key': {
        'type': str,
        'required': True
    },
    'r2_endpoint_url': {
        'type': str,
        'required': True
    },
    'r2_path_in_bucket': { # Optional path prefix within the bucket
        'type': str,
        'required': False,
        'default': ''
    },
    'generations': {
        'type': list,
        'required': True,
        'schema': { # Schema for each item in the generations list
            'prompt': {
                'type': str,
                'required': True
            },
            'negative_prompt': {
                'type': str,
                'required': False,
                'default': "" # Flux often uses an empty string
            },
            'width': {
                'type': int,
                'required': False,
                'default': 1024,
                # Add constraints based on Flux capabilities if known, e.g. multiple of 8 or 64
                'constraints': lambda width: width >= 64 and width <= 2048 and width % 8 == 0
            },
            'height': {
                'type': int,
                'required': False,
                'default': 1024,
                'constraints': lambda height: height >= 64 and height <= 2048 and height % 8 == 0
            },
            'num_outputs': {
                'type': int,
                'required': False,
                'default': 1,
                'constraints': lambda num: 1 <= num <= 10 # Adjust max if needed
            },
            'num_inference_steps': {
                'type': int,
                'required': False,
                'default': 4, # Flux Schnell is fast! Default is often low.
                'constraints': lambda steps: 1 <= steps <= 50 # Adjust range
            },
            'guidance_scale': {
                'type': float,
                'required': False,
                'default': 0.0, # Flux often uses 0 for no guidance or low values
                'constraints': lambda scale: 0.0 <= scale <= 10.0 # Adjust range
            },
            'seed': {
                'type': int,
                'required': False,
                'default': None # Will generate random if None
            },
            # Add any other Flux-specific parameters you want to control here
            # e.g., 'guidance_rescale', 'aesthetic_score', etc.
        }
    }
}