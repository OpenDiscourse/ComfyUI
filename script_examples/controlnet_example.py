"""
ControlNet Example - Using pose/depth control with people and positions

This example demonstrates how to use ControlNet for:
1. Pose-guided generation (using OpenPose to control human poses)
2. Depth-guided generation (using depth maps to control scene structure)
3. Canny edge control for precise positioning

Requirements:
- A checkpoint model (SD 1.5 or SDXL)
- ControlNet models (openpose, depth, canny, etc.)
- An input image or preprocessed control image
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


# OpenPose ControlNet workflow for controlling human poses
openpose_controlnet_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "person_pose.png"
        }
    },
    "3": {
        "class_type": "ControlNetLoader",
        "inputs": {
            "control_net_name": "control_v11p_sd15_openpose.pth"
        }
    },
    "4": {
        "class_type": "DWPreprocessor",
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
            "image": ["2", 0]
        }
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a professional dancer in elegant pose, studio lighting, high quality",
            "clip": ["1", 1]
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality, distorted limbs, bad anatomy",
            "clip": ["1", 1]
        }
    },
    "7": {
        "class_type": "ControlNetApply",
        "inputs": {
            "strength": 1.0,
            "conditioning": ["5", 0],
            "control_net": ["3", 0],
            "image": ["4", 0]
        }
    },
    "8": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 512,
            "height": 512,
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
            "model": ["1", 0],
            "positive": ["7", 0],
            "negative": ["6", 0],
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
            "filename_prefix": "controlnet_pose",
            "images": ["10", 0]
        }
    }
}


# Depth ControlNet workflow for scene structure control
depth_controlnet_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "scene_reference.png"
        }
    },
    "3": {
        "class_type": "ControlNetLoader",
        "inputs": {
            "control_net_name": "control_v11f1p_sd15_depth.pth"
        }
    },
    "4": {
        "class_type": "MiDaS-DepthMapPreprocessor",
        "inputs": {
            "a": 6.283185307179586,
            "bg_threshold": 0.1,
            "resolution": 512,
            "image": ["2", 0]
        }
    },
    "5": {
        "class_type": "ControlNetApply",
        "inputs": {
            "strength": 0.8,
            "conditioning": ["6", 0],
            "control_net": ["3", 0],
            "image": ["4", 0]
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a beautiful living room with modern furniture, plants, natural lighting",
            "clip": ["1", 1]
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, dark, cluttered",
            "clip": ["1", 1]
        }
    },
    "8": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 512,
            "height": 512,
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
            "model": ["1", 0],
            "positive": ["5", 0],
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
            "filename_prefix": "controlnet_depth",
            "images": ["10", 0]
        }
    }
}


# Multi-ControlNet workflow combining pose and depth
multi_controlnet_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "reference_image.png"
        }
    },
    # Pose ControlNet
    "3": {
        "class_type": "ControlNetLoader",
        "inputs": {
            "control_net_name": "control_v11p_sd15_openpose.pth"
        }
    },
    "4": {
        "class_type": "DWPreprocessor",
        "inputs": {
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": 512,
            "image": ["2", 0]
        }
    },
    # Depth ControlNet
    "5": {
        "class_type": "ControlNetLoader",
        "inputs": {
            "control_net_name": "control_v11f1p_sd15_depth.pth"
        }
    },
    "6": {
        "class_type": "MiDaS-DepthMapPreprocessor",
        "inputs": {
            "a": 6.283185307179586,
            "bg_threshold": 0.1,
            "resolution": 512,
            "image": ["2", 0]
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "a person in an artistic pose, professional photography",
            "clip": ["1", 1]
        }
    },
    "8": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "blurry, low quality",
            "clip": ["1", 1]
        }
    },
    # Apply pose control
    "9": {
        "class_type": "ControlNetApply",
        "inputs": {
            "strength": 1.0,
            "conditioning": ["7", 0],
            "control_net": ["3", 0],
            "image": ["4", 0]
        }
    },
    # Apply depth control
    "10": {
        "class_type": "ControlNetApply",
        "inputs": {
            "strength": 0.6,
            "conditioning": ["9", 0],
            "control_net": ["5", 0],
            "image": ["6", 0]
        }
    },
    "11": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1
        }
    },
    "12": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 25,
            "cfg": 7.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["10", 0],
            "negative": ["8", 0],
            "latent_image": ["11", 0]
        }
    },
    "13": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["12", 0],
            "vae": ["1", 2]
        }
    },
    "14": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "controlnet_multi",
            "images": ["13", 0]
        }
    }
}


def generate_with_pose_control(
    reference_image,
    prompt,
    negative_prompt="blurry, low quality, bad anatomy",
    controlnet_model="control_v11p_sd15_openpose.pth",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    strength=1.0,
    steps=20,
    cfg=7.0,
    seed=42,
    detect_hands=True,
    detect_face=True
):
    """
    Generate an image using pose control from a reference image
    
    Args:
        reference_image: Image with person(s) to extract pose from
        prompt: Text description of desired output
        negative_prompt: What to avoid
        controlnet_model: OpenPose ControlNet model filename
        checkpoint: Base model checkpoint
        strength: ControlNet strength (0.0-1.0)
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
        detect_hands: Whether to detect hand poses
        detect_face: Whether to detect face pose
    """
    workflow = copy.deepcopy(openpose_controlnet_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = reference_image
    workflow["3"]["inputs"]["control_net_name"] = controlnet_model
    workflow["4"]["inputs"]["detect_hand"] = "enable" if detect_hands else "disable"
    workflow["4"]["inputs"]["detect_face"] = "enable" if detect_face else "disable"
    workflow["5"]["inputs"]["text"] = prompt
    workflow["6"]["inputs"]["text"] = negative_prompt
    workflow["7"]["inputs"]["strength"] = strength
    workflow["9"]["inputs"]["seed"] = seed
    workflow["9"]["inputs"]["steps"] = steps
    workflow["9"]["inputs"]["cfg"] = cfg
    
    queue_prompt(workflow)
    print(f"Pose-controlled generation queued: {reference_image}")


def generate_with_depth_control(
    reference_image,
    prompt,
    negative_prompt="blurry, low quality",
    controlnet_model="control_v11f1p_sd15_depth.pth",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    strength=0.8,
    steps=20,
    cfg=7.0,
    seed=42
):
    """
    Generate an image using depth control from a reference image
    
    Args:
        reference_image: Image to extract depth map from
        prompt: Text description of desired output
        negative_prompt: What to avoid
        controlnet_model: Depth ControlNet model filename
        checkpoint: Base model checkpoint
        strength: ControlNet strength (0.0-1.0)
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
    """
    workflow = copy.deepcopy(depth_controlnet_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = reference_image
    workflow["3"]["inputs"]["control_net_name"] = controlnet_model
    workflow["5"]["inputs"]["strength"] = strength
    workflow["6"]["inputs"]["text"] = prompt
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["9"]["inputs"]["seed"] = seed
    workflow["9"]["inputs"]["steps"] = steps
    workflow["9"]["inputs"]["cfg"] = cfg
    
    queue_prompt(workflow)
    print(f"Depth-controlled generation queued: {reference_image}")


def generate_with_multi_controlnet(
    reference_image,
    prompt,
    negative_prompt="blurry, low quality",
    pose_model="control_v11p_sd15_openpose.pth",
    depth_model="control_v11f1p_sd15_depth.pth",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    pose_strength=1.0,
    depth_strength=0.6,
    steps=25,
    cfg=7.5,
    seed=42
):
    """
    Generate an image using multiple ControlNets (pose + depth)
    
    Args:
        reference_image: Image to extract controls from
        prompt: Text description of desired output
        negative_prompt: What to avoid
        pose_model: OpenPose ControlNet model filename
        depth_model: Depth ControlNet model filename
        checkpoint: Base model checkpoint
        pose_strength: Pose ControlNet strength (0.0-1.0)
        depth_strength: Depth ControlNet strength (0.0-1.0)
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
    """
    workflow = copy.deepcopy(multi_controlnet_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = reference_image
    workflow["3"]["inputs"]["control_net_name"] = pose_model
    workflow["5"]["inputs"]["control_net_name"] = depth_model
    workflow["7"]["inputs"]["text"] = prompt
    workflow["8"]["inputs"]["text"] = negative_prompt
    workflow["9"]["inputs"]["strength"] = pose_strength
    workflow["10"]["inputs"]["strength"] = depth_strength
    workflow["12"]["inputs"]["seed"] = seed
    workflow["12"]["inputs"]["steps"] = steps
    workflow["12"]["inputs"]["cfg"] = cfg
    
    queue_prompt(workflow)
    print(f"Multi-ControlNet generation queued: {reference_image}")


if __name__ == "__main__":
    # Example 1: Pose-controlled generation
    generate_with_pose_control(
        reference_image="dancer.png",
        prompt="a ballet dancer in elegant white costume, stage lighting, professional photo",
        strength=1.0,
        steps=25,
        seed=12345
    )
    
    # Example 2: Depth-controlled generation
    # generate_with_depth_control(
    #     reference_image="room.png",
    #     prompt="a cozy living room with fireplace and bookshelves, warm lighting",
    #     strength=0.8,
    #     steps=20,
    #     seed=67890
    # )
    
    # Example 3: Multi-ControlNet (pose + depth)
    # generate_with_multi_controlnet(
    #     reference_image="person_in_scene.png",
    #     prompt="a professional model in fashionable outfit, studio photography",
    #     pose_strength=1.0,
    #     depth_strength=0.6,
    #     steps=25,
    #     seed=11111
    # )
