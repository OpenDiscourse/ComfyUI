"""
Image Enhancement Examples - Upscaling, img2img, and face enhancement

This example demonstrates:
1. Image upscaling with ESRGAN/RealESRGAN
2. Image-to-Image transformation
3. Face restoration/enhancement
4. Multi-pass upscaling workflows

Requirements:
- A checkpoint model for img2img
- Upscaling models (ESRGAN, RealESRGAN, etc.)
- Input images to enhance
"""

import json
import copy
from urllib import request

server_address = "127.0.0.1:8188"


def queue_prompt(prompt):
    """Send a prompt to the ComfyUI server"""
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"http://{server_address}/prompt", data=data)
    return request.urlopen(req)


# Simple upscaling workflow
upscaling_workflow = {
    "1": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "input.png"
        }
    },
    "2": {
        "class_type": "UpscaleModelLoader",
        "inputs": {
            "model_name": "RealESRGAN_x4plus.pth"
        }
    },
    "3": {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": ["2", 0],
            "image": ["1", 0]
        }
    },
    "4": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "upscaled",
            "images": ["3", 0]
        }
    }
}


# Image-to-Image workflow
img2img_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "input.png"
        }
    },
    "3": {
        "class_type": "VAEEncode",
        "inputs": {
            "pixels": ["2", 0],
            "vae": ["1", 2]
        }
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a beautiful landscape painting, vibrant colors, masterpiece",
            "clip": ["1", 1]
        }
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality",
            "clip": ["1", 1]
        }
    },
    "6": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.7,  # Lower denoise preserves more of original
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["3", 0]
        }
    },
    "7": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["6", 0],
            "vae": ["1", 2]
        }
    },
    "8": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "img2img",
            "images": ["7", 0]
        }
    }
}


# High-resolution fix workflow (2-pass txt2img/img2img)
hires_fix_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a detailed portrait, professional photography",
            "clip": ["1", 1]
        }
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality",
            "clip": ["1", 1]
        }
    },
    # First pass - generate at base resolution
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1
        }
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0]
        }
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["5", 0],
            "vae": ["1", 2]
        }
    },
    # Upscale
    "7": {
        "class_type": "UpscaleModelLoader",
        "inputs": {
            "model_name": "RealESRGAN_x4plus.pth"
        }
    },
    "8": {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": ["7", 0],
            "image": ["6", 0]
        }
    },
    # Second pass - refine at high resolution
    "9": {
        "class_type": "VAEEncode",
        "inputs": {
            "pixels": ["8", 0],
            "vae": ["1", 2]
        }
    },
    "10": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 15,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.4,  # Light refinement pass
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["9", 0]
        }
    },
    "11": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["10", 0],
            "vae": ["1", 2]
        }
    },
    "12": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "hires_fix",
            "images": ["11", 0]
        }
    }
}


# Combined upscale and enhance workflow
upscale_and_enhance_workflow = {
    "1": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "input.png"
        }
    },
    # First upscale with model
    "2": {
        "class_type": "UpscaleModelLoader",
        "inputs": {
            "model_name": "RealESRGAN_x4plus.pth"
        }
    },
    "3": {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {
            "upscale_model": ["2", 0],
            "image": ["1", 0]
        }
    },
    # Then enhance with AI
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "5": {
        "class_type": "VAEEncode",
        "inputs": {
            "pixels": ["3", 0],
            "vae": ["4", 2]
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "high quality, detailed, sharp, professional",
            "clip": ["4", 1]
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, artifacts",
            "clip": ["4", 1]
        }
    },
    "8": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 15,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.3,  # Light touch to preserve details
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        }
    },
    "9": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["8", 0],
            "vae": ["4", 2]
        }
    },
    "10": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "upscaled_enhanced",
            "images": ["9", 0]
        }
    }
}


def upscale_image(
    image_filename,
    upscale_model="RealESRGAN_x4plus.pth",
    output_prefix="upscaled"
):
    """
    Simple image upscaling using an upscale model
    
    Args:
        image_filename: Input image filename
        upscale_model: Upscaling model (RealESRGAN_x4plus.pth, ESRGAN_4x.pth, etc.)
        output_prefix: Prefix for output filename
    """
    workflow = copy.deepcopy(upscaling_workflow)
    
    workflow["1"]["inputs"]["image"] = image_filename
    workflow["2"]["inputs"]["model_name"] = upscale_model
    workflow["4"]["inputs"]["filename_prefix"] = output_prefix
    
    queue_prompt(workflow)
    print(f"Upscaling queued: {image_filename} with {upscale_model}")


def img2img_transform(
    image_filename,
    prompt,
    negative_prompt="blurry, low quality",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    denoise=0.7,
    steps=20,
    cfg=7.0,
    seed=42
):
    """
    Transform an image using img2img with text guidance
    
    Args:
        image_filename: Input image filename
        prompt: What you want the image to look like
        negative_prompt: What to avoid
        checkpoint: Model checkpoint
        denoise: How much to change (0.0-1.0, higher = more change)
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
    """
    workflow = copy.deepcopy(img2img_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = image_filename
    workflow["4"]["inputs"]["text"] = prompt
    workflow["5"]["inputs"]["text"] = negative_prompt
    workflow["6"]["inputs"]["seed"] = seed
    workflow["6"]["inputs"]["steps"] = steps
    workflow["6"]["inputs"]["cfg"] = cfg
    workflow["6"]["inputs"]["denoise"] = denoise
    
    queue_prompt(workflow)
    print(f"Img2img queued: {image_filename}")


def hires_fix_generation(
    prompt,
    negative_prompt="blurry, low quality",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    base_resolution=512,
    upscale_model="RealESRGAN_x4plus.pth",
    first_pass_steps=20,
    second_pass_steps=15,
    second_pass_denoise=0.4,
    cfg=7.0,
    seed=42
):
    """
    High-resolution fix: Generate at low res, upscale, then refine
    
    Args:
        prompt: Text description
        negative_prompt: What to avoid
        checkpoint: Model checkpoint
        base_resolution: Initial generation resolution
        upscale_model: Model for upscaling
        first_pass_steps: Steps for initial generation
        second_pass_steps: Steps for refinement
        second_pass_denoise: Denoise for refinement (lower = preserve more)
        cfg: CFG scale
        seed: Random seed
    """
    workflow = copy.deepcopy(hires_fix_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["text"] = prompt
    workflow["3"]["inputs"]["text"] = negative_prompt
    workflow["4"]["inputs"]["width"] = base_resolution
    workflow["4"]["inputs"]["height"] = base_resolution
    workflow["5"]["inputs"]["seed"] = seed
    workflow["5"]["inputs"]["steps"] = first_pass_steps
    workflow["5"]["inputs"]["cfg"] = cfg
    workflow["7"]["inputs"]["model_name"] = upscale_model
    workflow["10"]["inputs"]["steps"] = second_pass_steps
    workflow["10"]["inputs"]["cfg"] = cfg
    workflow["10"]["inputs"]["denoise"] = second_pass_denoise
    
    queue_prompt(workflow)
    print(f"Hires-fix generation queued: {prompt[:50]}...")


def upscale_and_enhance(
    image_filename,
    enhancement_prompt="high quality, detailed, sharp",
    negative_prompt="blurry, low quality, artifacts",
    upscale_model="RealESRGAN_x4plus.pth",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    denoise=0.3,
    steps=15,
    cfg=7.0,
    seed=42
):
    """
    Upscale an image then enhance it with AI
    
    Args:
        image_filename: Input image filename
        enhancement_prompt: Qualities to enhance
        negative_prompt: What to avoid
        upscale_model: Upscaling model
        checkpoint: Model checkpoint for enhancement
        denoise: Enhancement strength (0.0-1.0)
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
    """
    workflow = copy.deepcopy(upscale_and_enhance_workflow)
    
    workflow["1"]["inputs"]["image"] = image_filename
    workflow["2"]["inputs"]["model_name"] = upscale_model
    workflow["4"]["inputs"]["ckpt_name"] = checkpoint
    workflow["6"]["inputs"]["text"] = enhancement_prompt
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["8"]["inputs"]["seed"] = seed
    workflow["8"]["inputs"]["steps"] = steps
    workflow["8"]["inputs"]["cfg"] = cfg
    workflow["8"]["inputs"]["denoise"] = denoise
    
    queue_prompt(workflow)
    print(f"Upscale and enhance queued: {image_filename}")


if __name__ == "__main__":
    # Example 1: Simple upscaling
    upscale_image(
        image_filename="photo.png",
        upscale_model="RealESRGAN_x4plus.pth"
    )
    
    # Example 2: Image-to-Image transformation
    # img2img_transform(
    #     image_filename="sketch.png",
    #     prompt="a professional oil painting, vibrant colors, detailed",
    #     denoise=0.75,
    #     steps=25,
    #     seed=12345
    # )
    
    # Example 3: High-resolution fix
    # hires_fix_generation(
    #     prompt="a detailed portrait of a person, professional photography, sharp focus",
    #     base_resolution=512,
    #     first_pass_steps=20,
    #     second_pass_steps=15,
    #     second_pass_denoise=0.4,
    #     seed=67890
    # )
    
    # Example 4: Upscale and enhance
    # upscale_and_enhance(
    #     image_filename="old_photo.png",
    #     enhancement_prompt="restored vintage photo, clear details, professional quality",
    #     denoise=0.35,
    #     seed=11111
    # )
