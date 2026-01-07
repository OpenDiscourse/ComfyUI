"""
Video Generation Examples - Text-to-Video and Image-to-Video

This example demonstrates various video generation workflows:
1. Text-to-Video with Hunyuan Video
2. Image-to-Video with Hunyuan Video 1.5
3. Text-to-Video with LTX-Video
4. Image-to-Video with LTX-Video

Requirements:
- A video model checkpoint (e.g., HunyuanVideo, LTXVideo, Mochi, etc.)
- For image-to-video: An input image
"""

import json
import copy
from urllib import request

server_address = "127.0.0.1:8188"

# CLIP text encoder has a maximum token limit
CLIP_TOKEN_LIMIT = 77


def queue_prompt(prompt):
    """Send a prompt to the ComfyUI server"""
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"http://{server_address}/prompt", data=data)
    return request.urlopen(req)


# Text-to-Video workflow with Hunyuan Video
hunyuan_text_to_video_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"
        }
    },
    "2": {
        "class_type": "CLIPTextEncodeHunyuanVideo",
        "inputs": {
            "clip": ["1", 1],
            "llm": "A cinematic shot of a person walking down a street",
            "clip_l": "person walking, street, cinematic"
        }
    },
    "3": {
        "class_type": "CLIPTextEncodeHunyuanVideo",
        "inputs": {
            "clip": ["1", 1],
            "llm": "",
            "clip_l": "blurry, low quality"
        }
    },
    "4": {
        "class_type": "EmptyHunyuanLatentVideo",
        "inputs": {
            "width": 848,
            "height": 480,
            "length": 25,
            "batch_size": 1
        }
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 1.0,
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
    "7": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "frame_rate": 24,
            "loop_count": 0,
            "filename_prefix": "hunyuan_video",
            "format": "video/h264-mp4",
            "images": ["6", 0]
        }
    }
}


# Image-to-Video workflow with Hunyuan Video 1.5
hunyuan_image_to_video_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "hunyuan_video_1.5_fp8_e4m3fn.safetensors"
        }
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "start_frame.png"
        }
    },
    "3": {
        "class_type": "CLIPVisionLoader",
        "inputs": {
            "clip_name": "sigclip_vision_patch14_384.safetensors"
        }
    },
    "4": {
        "class_type": "CLIPVisionEncode",
        "inputs": {
            "clip_vision": ["3", 0],
            "image": ["2", 0]
        }
    },
    "5": {
        "class_type": "TextEncodeHunyuanVideo_ImageToVideo",
        "inputs": {
            "clip": ["1", 1],
            "llm": "The person starts walking forward",
            "clip_l": "walking, movement"
        }
    },
    "6": {
        "class_type": "TextEncodeHunyuanVideo_ImageToVideo",
        "inputs": {
            "clip": ["1", 1],
            "llm": "",
            "clip_l": "static, blurry"
        }
    },
    "7": {
        "class_type": "HunyuanVideo15ImageToVideo",
        "inputs": {
            "positive": ["5", 0],
            "negative": ["6", 0],
            "vae": ["1", 2],
            "width": 848,
            "height": 480,
            "length": 49,
            "batch_size": 1,
            "clip_vision_output": ["4", 0],
            "start_image": ["2", 0]
        }
    },
    "8": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 25,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["7", 0],
            "negative": ["7", 1],
            "latent_image": ["7", 2]
        }
    },
    "9": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["8", 0],
            "vae": ["1", 2]
        }
    },
    "10": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "frame_rate": 24,
            "loop_count": 0,
            "filename_prefix": "hunyuan_i2v",
            "format": "video/h264-mp4",
            "images": ["9", 0]
        }
    }
}


# LTX-Video Text-to-Video workflow
ltxv_text_to_video_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "ltx-video-2b-v0.9.safetensors"
        }
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "A drone shot of a sunset over mountains",
            "clip": ["1", 1]
        }
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, static, low quality",
            "clip": ["1", 1]
        }
    },
    "4": {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": 768,
            "height": 512,
            "length": 97,
            "batch_size": 1
        }
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 25,
            "cfg": 3.0,
            "sampler_name": "euler",
            "scheduler": "simple",
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
    "7": {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "frame_rate": 25,
            "loop_count": 0,
            "filename_prefix": "ltxv_video",
            "format": "video/h264-mp4",
            "images": ["6", 0]
        }
    }
}


def generate_text_to_video_hunyuan(
    prompt,
    negative_prompt="blurry, low quality",
    width=848,
    height=480,
    length=25,
    steps=20,
    cfg=1.0,
    seed=42,
    checkpoint="hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors"
):
    """
    Generate a video from text using Hunyuan Video
    
    Args:
        prompt: Text description of the video
        negative_prompt: What to avoid
        width: Video width (must be divisible by 16)
        height: Video height (must be divisible by 16)
        length: Number of frames (recommended: 25-81)
        steps: Sampling steps
        cfg: CFG scale (Hunyuan Video works well with 1.0)
        seed: Random seed
        checkpoint: Model checkpoint filename
    """
    workflow = copy.deepcopy(hunyuan_text_to_video_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["llm"] = prompt
    workflow["2"]["inputs"]["clip_l"] = prompt[:CLIP_TOKEN_LIMIT]  # CLIP has token limit
    workflow["3"]["inputs"]["clip_l"] = negative_prompt
    workflow["4"]["inputs"]["width"] = width
    workflow["4"]["inputs"]["height"] = height
    workflow["4"]["inputs"]["length"] = length
    workflow["5"]["inputs"]["seed"] = seed
    workflow["5"]["inputs"]["steps"] = steps
    workflow["5"]["inputs"]["cfg"] = cfg
    
    queue_prompt(workflow)
    print(f"Hunyuan text-to-video queued: {prompt[:50]}...")


def generate_image_to_video_hunyuan(
    image_filename,
    prompt,
    negative_prompt="static, blurry",
    width=848,
    height=480,
    length=49,
    steps=25,
    cfg=1.0,
    seed=42,
    checkpoint="hunyuan_video_1.5_fp8_e4m3fn.safetensors",
    clip_vision="sigclip_vision_patch14_384.safetensors"
):
    """
    Generate a video from an image using Hunyuan Video 1.5
    
    Args:
        image_filename: Starting frame image in input folder
        prompt: Text description of the motion/animation
        negative_prompt: What to avoid
        width: Video width
        height: Video height
        length: Number of frames
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
        checkpoint: Model checkpoint filename
        clip_vision: CLIP Vision model for image encoding
    """
    workflow = copy.deepcopy(hunyuan_image_to_video_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = image_filename
    workflow["3"]["inputs"]["clip_name"] = clip_vision
    workflow["5"]["inputs"]["llm"] = prompt
    workflow["5"]["inputs"]["clip_l"] = prompt[:CLIP_TOKEN_LIMIT]
    workflow["6"]["inputs"]["clip_l"] = negative_prompt
    workflow["7"]["inputs"]["width"] = width
    workflow["7"]["inputs"]["height"] = height
    workflow["7"]["inputs"]["length"] = length
    workflow["8"]["inputs"]["seed"] = seed
    workflow["8"]["inputs"]["steps"] = steps
    workflow["8"]["inputs"]["cfg"] = cfg
    
    queue_prompt(workflow)
    print(f"Hunyuan image-to-video queued: {image_filename}")


def generate_text_to_video_ltxv(
    prompt,
    negative_prompt="blurry, static, low quality",
    width=768,
    height=512,
    length=97,
    steps=25,
    cfg=3.0,
    seed=42,
    checkpoint="ltx-video-2b-v0.9.safetensors"
):
    """
    Generate a video from text using LTX-Video
    
    Args:
        prompt: Text description of the video
        negative_prompt: What to avoid
        width: Video width (must be divisible by 32)
        height: Video height (must be divisible by 32)
        length: Number of frames (recommended: 97-161)
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
        checkpoint: Model checkpoint filename
    """
    workflow = copy.deepcopy(ltxv_text_to_video_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["text"] = prompt
    workflow["3"]["inputs"]["text"] = negative_prompt
    workflow["4"]["inputs"]["width"] = width
    workflow["4"]["inputs"]["height"] = height
    workflow["4"]["inputs"]["length"] = length
    workflow["5"]["inputs"]["seed"] = seed
    workflow["5"]["inputs"]["steps"] = steps
    workflow["5"]["inputs"]["cfg"] = cfg
    
    queue_prompt(workflow)
    print(f"LTX-Video text-to-video queued: {prompt[:50]}...")


if __name__ == "__main__":
    # Example 1: Text-to-Video with Hunyuan
    generate_text_to_video_hunyuan(
        prompt="A cinematic shot of a person walking through a beautiful forest, sunlight filtering through trees",
        width=848,
        height=480,
        length=25,
        steps=20,
        seed=12345
    )
    
    # Example 2: Image-to-Video with Hunyuan
    # generate_image_to_video_hunyuan(
    #     image_filename="forest_start.png",
    #     prompt="The camera slowly moves forward through the forest",
    #     length=49,
    #     steps=25,
    #     seed=67890
    # )
    
    # Example 3: Text-to-Video with LTX-Video
    # generate_text_to_video_ltxv(
    #     prompt="A drone flying over a sunset landscape with mountains",
    #     width=768,
    #     height=512,
    #     length=97,
    #     steps=25,
    #     seed=11111
    # )
