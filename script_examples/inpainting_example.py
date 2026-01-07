"""
Inpainting Example - Demonstrates how to inpaint an image using a mask

This example shows how to:
1. Load an existing image
2. Load or create a mask to define the area to inpaint
3. Use VAEEncodeForInpaint to prepare the image and mask
4. Run the diffusion process to inpaint the masked area
5. Save the result

Requirements:
- A checkpoint model (e.g., SD 1.5 or SDXL)
- An input image in the input folder
- A mask image (white areas will be inpainted)
"""

import json
from urllib import request

# ComfyUI API server address
server_address = "127.0.0.1:8188"

def queue_prompt(prompt):
    """Send a prompt to the ComfyUI server"""
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"http://{server_address}/prompt", data=data)
    return request.urlopen(req)


# Inpainting workflow using VAEEncodeForInpaint
inpainting_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "example.png"  # Replace with your image filename
        }
    },
    "3": {
        "class_type": "LoadImageMask",
        "inputs": {
            "image": "mask.png",  # Replace with your mask filename
            "channel": "alpha"
        }
    },
    "4": {
        "class_type": "VAEEncodeForInpaint",
        "inputs": {
            "pixels": ["2", 0],
            "vae": ["1", 2],
            "mask": ["3", 0],
            "grow_mask_by": 6
        }
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "beautiful landscape with mountains and trees",
            "clip": ["1", 1]
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, distorted",
            "clip": ["1", 1]
        }
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 8.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["4", 0]
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["1", 2]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "inpaint_result",
            "images": ["8", 0]
        }
    }
}


def run_inpainting(
    image_filename,
    mask_filename,
    positive_prompt,
    negative_prompt="blurry, low quality",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    steps=20,
    cfg=8.0,
    seed=42,
    denoise=1.0
):
    """
    Run an inpainting workflow
    
    Args:
        image_filename: Name of the image file in the input folder
        mask_filename: Name of the mask file (white areas will be inpainted)
        positive_prompt: Text describing what to generate
        negative_prompt: Text describing what to avoid
        checkpoint: Model checkpoint to use
        steps: Number of sampling steps
        cfg: CFG scale
        seed: Random seed
        denoise: Denoise strength (0.0-1.0)
    """
    workflow = inpainting_workflow.copy()
    
    # Update parameters
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = image_filename
    workflow["3"]["inputs"]["image"] = mask_filename
    workflow["5"]["inputs"]["text"] = positive_prompt
    workflow["6"]["inputs"]["text"] = negative_prompt
    workflow["7"]["inputs"]["seed"] = seed
    workflow["7"]["inputs"]["steps"] = steps
    workflow["7"]["inputs"]["cfg"] = cfg
    workflow["7"]["inputs"]["denoise"] = denoise
    
    # Queue the prompt
    queue_prompt(workflow)
    print(f"Inpainting queued: {image_filename} with mask {mask_filename}")


if __name__ == "__main__":
    # Example usage
    run_inpainting(
        image_filename="example.png",
        mask_filename="mask.png",
        positive_prompt="a cat sitting on a chair, photorealistic",
        negative_prompt="blurry, distorted, low quality",
        steps=30,
        cfg=7.5,
        seed=12345
    )
