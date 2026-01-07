"""
Face and Character Examples - Face swapping, IP-Adapter, and character consistency

This example demonstrates:
1. Face swapping using InstantID or similar techniques
2. IP-Adapter for style/character reference
3. Character-consistent generation
4. Face enhancement workflows

Note: Face swapping typically requires custom nodes like InstantID, ReActor, or similar.
This example shows the conceptual workflow structure.

Requirements:
- Base checkpoint model
- IP-Adapter models (for character reference)
- CLIP Vision models
- Face detection/recognition models (depending on custom nodes)
"""

import json
from urllib import request

server_address = "127.0.0.1:8188"


def queue_prompt(prompt):
    """Send a prompt to the ComfyUI server"""
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"http://{server_address}/prompt", data=data)
    return request.urlopen(req)


# IP-Adapter workflow for character/style reference
ipadapter_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "reference_character.png"
        }
    },
    "3": {
        "class_type": "CLIPVisionLoader",
        "inputs": {
            "clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
        }
    },
    "4": {
        "class_type": "IPAdapterModelLoader",
        "inputs": {
            "ipadapter_file": "ip-adapter_sd15.safetensors"
        }
    },
    "5": {
        "class_type": "IPAdapterApply",
        "inputs": {
            "weight": 0.8,
            "model": ["1", 0],
            "ipadapter": ["4", 0],
            "clip_vision": ["3", 0],
            "image": ["2", 0]
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a person standing in a garden, professional photo",
            "clip": ["1", 1]
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality",
            "clip": ["1", 1]
        }
    },
    "8": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 512,
            "height": 768,
            "batch_size": 1
        }
    },
    "9": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["5", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["8", 0]
        }
    },
    "10": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["9", 0],
            "vae": ["1", 2]
        }
    },
    "11": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ipadapter_character",
            "images": ["10", 0]
        }
    }
}


# Multi-character scene with IP-Adapter
multi_character_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    # Character 1 reference
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "character1.png"
        }
    },
    "3": {
        "class_type": "CLIPVisionLoader",
        "inputs": {
            "clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
        }
    },
    "4": {
        "class_type": "IPAdapterModelLoader",
        "inputs": {
            "ipadapter_file": "ip-adapter_sd15.safetensors"
        }
    },
    "5": {
        "class_type": "IPAdapterApply",
        "inputs": {
            "weight": 0.7,
            "model": ["1", 0],
            "ipadapter": ["4", 0],
            "clip_vision": ["3", 0],
            "image": ["2", 0]
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "two people having a conversation, outdoor setting, natural lighting",
            "clip": ["1", 1]
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, distorted faces",
            "clip": ["1", 1]
        }
    },
    "8": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 768,
            "height": 512,
            "batch_size": 1
        }
    },
    "9": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 25,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["5", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["8", 0]
        }
    },
    "10": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["9", 0],
            "vae": ["1", 2]
        }
    },
    "11": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "multi_character",
            "images": ["10", 0]
        }
    }
}


# Face-focused workflow with FaceDetailer (conceptual)
face_detailer_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "portrait.png"
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
            "text": "professional portrait, detailed face, sharp eyes, natural skin texture",
            "clip": ["1", 1]
        }
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, distorted face",
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
            "denoise": 0.4,
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
    # Note: FaceDetailer is typically from custom nodes
    # This is a simplified version showing the concept
    "8": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "face_enhanced",
            "images": ["7", 0]
        }
    }
}


def generate_with_character_reference(
    reference_image,
    prompt,
    negative_prompt="blurry, low quality",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    ipadapter_model="ip-adapter_sd15.safetensors",
    clip_vision="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    weight=0.8,
    width=512,
    height=768,
    steps=20,
    cfg=7.0,
    seed=42
):
    """
    Generate images with consistent character using IP-Adapter
    
    Args:
        reference_image: Reference image of the character
        prompt: Scene/pose description
        negative_prompt: What to avoid
        checkpoint: Base model checkpoint
        ipadapter_model: IP-Adapter model file
        clip_vision: CLIP Vision model
        weight: IP-Adapter influence (0.0-1.0)
        width: Output width
        height: Output height
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
    """
    workflow = json.loads(json.dumps(ipadapter_workflow))
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = reference_image
    workflow["3"]["inputs"]["clip_name"] = clip_vision
    workflow["4"]["inputs"]["ipadapter_file"] = ipadapter_model
    workflow["5"]["inputs"]["weight"] = weight
    workflow["6"]["inputs"]["text"] = prompt
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["8"]["inputs"]["width"] = width
    workflow["8"]["inputs"]["height"] = height
    workflow["9"]["inputs"]["seed"] = seed
    workflow["9"]["inputs"]["steps"] = steps
    workflow["9"]["inputs"]["cfg"] = cfg
    
    queue_prompt(workflow)
    print(f"Character reference generation queued: {reference_image}")


def enhance_face_details(
    image_filename,
    enhancement_prompt="professional portrait, detailed face, sharp eyes",
    negative_prompt="blurry, low quality, distorted face",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    denoise=0.4,
    steps=20,
    cfg=7.0,
    seed=42
):
    """
    Enhance face details in an existing portrait
    
    Args:
        image_filename: Input portrait image
        enhancement_prompt: Face enhancement description
        negative_prompt: What to avoid
        checkpoint: Base model checkpoint
        denoise: Enhancement strength (0.0-1.0)
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
    """
    workflow = json.loads(json.dumps(face_detailer_workflow))
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = image_filename
    workflow["4"]["inputs"]["text"] = enhancement_prompt
    workflow["5"]["inputs"]["text"] = negative_prompt
    workflow["6"]["inputs"]["seed"] = seed
    workflow["6"]["inputs"]["steps"] = steps
    workflow["6"]["inputs"]["cfg"] = cfg
    workflow["6"]["inputs"]["denoise"] = denoise
    
    queue_prompt(workflow)
    print(f"Face enhancement queued: {image_filename}")


def generate_multi_character_scene(
    character_reference,
    prompt,
    negative_prompt="blurry, low quality, distorted faces",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    ipadapter_model="ip-adapter_sd15.safetensors",
    clip_vision="CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
    weight=0.7,
    width=768,
    height=512,
    steps=25,
    cfg=7.0,
    seed=42
):
    """
    Generate a scene with multiple characters using reference
    
    Args:
        character_reference: Reference image for character consistency
        prompt: Scene description
        negative_prompt: What to avoid
        checkpoint: Base model checkpoint
        ipadapter_model: IP-Adapter model file
        clip_vision: CLIP Vision model
        weight: IP-Adapter influence
        width: Output width
        height: Output height
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
    """
    workflow = json.loads(json.dumps(multi_character_workflow))
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = character_reference
    workflow["3"]["inputs"]["clip_name"] = clip_vision
    workflow["4"]["inputs"]["ipadapter_file"] = ipadapter_model
    workflow["5"]["inputs"]["weight"] = weight
    workflow["6"]["inputs"]["text"] = prompt
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["8"]["inputs"]["width"] = width
    workflow["8"]["inputs"]["height"] = height
    workflow["9"]["inputs"]["seed"] = seed
    workflow["9"]["inputs"]["steps"] = steps
    workflow["9"]["inputs"]["cfg"] = cfg
    
    queue_prompt(workflow)
    print(f"Multi-character scene queued: {character_reference}")


if __name__ == "__main__":
    # Example 1: Generate with character reference
    generate_with_character_reference(
        reference_image="character_ref.png",
        prompt="the character walking in a park, sunny day, casual clothes",
        weight=0.8,
        steps=25,
        seed=12345
    )
    
    # Example 2: Enhance face details
    # enhance_face_details(
    #     image_filename="portrait.png",
    #     enhancement_prompt="high quality portrait, detailed facial features, professional",
    #     denoise=0.35,
    #     steps=20,
    #     seed=67890
    # )
    
    # Example 3: Multi-character scene
    # generate_multi_character_scene(
    #     character_reference="main_character.png",
    #     prompt="two people having a friendly conversation in a cafe, warm lighting",
    #     weight=0.7,
    #     steps=25,
    #     seed=11111
    # )

    # Note: For actual face swapping, you would typically use custom nodes like:
    # - ReActor (formerly roop)
    # - InstantID
    # - FaceSwap nodes
    # These require additional models and have their own specific workflows
    print("\nNote: For face swapping functionality, install custom nodes like:")
    print("  - ReActor: https://github.com/Gourieff/comfyui-reactor-node")
    print("  - InstantID: https://github.com/cubiq/ComfyUI_InstantID")
