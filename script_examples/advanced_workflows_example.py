"""
Advanced Workflow Examples - Complex multi-node pipelines

This example demonstrates advanced workflows combining multiple techniques:
1. Style transfer with ControlNet
2. Batch processing multiple images
3. Animation sequences
4. Custom preprocessing pipelines

These are more complex examples that combine techniques from other examples.
"""

import json
import copy
import math
from urllib import request
import time

server_address = "127.0.0.1:8188"

# MiDaS depth preprocessor parameter (2 * pi)
TWO_PI = 2 * math.pi


def queue_prompt(prompt):
    """Send a prompt to the ComfyUI server"""
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"http://{server_address}/prompt", data=data)
    return request.urlopen(req)


# Style transfer with ControlNet and img2img
style_transfer_controlnet_workflow = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    # Load content image
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "content.png"
        }
    },
    # Load style reference
    "3": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "style_reference.png"
        }
    },
    # Extract depth for structure preservation
    "4": {
        "class_type": "ControlNetLoader",
        "inputs": {
            "control_net_name": "control_v11f1p_sd15_depth.pth"
        }
    },
    "5": {
        "class_type": "MiDaS-DepthMapPreprocessor",
        "inputs": {
            "a": TWO_PI,
            "bg_threshold": 0.1,
            "resolution": 512,
            "image": ["2", 0]
        }
    },
    # Encode content image
    "6": {
        "class_type": "VAEEncode",
        "inputs": {
            "pixels": ["2", 0],
            "vae": ["1", 2]
        }
    },
    # Style prompt
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "in the style of impressionist painting, vibrant brushstrokes",
            "clip": ["1", 1]
        }
    },
    "8": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "photograph, realistic, blurry",
            "clip": ["1", 1]
        }
    },
    # Apply ControlNet
    "9": {
        "class_type": "ControlNetApply",
        "inputs": {
            "strength": 0.7,
            "conditioning": ["7", 0],
            "control_net": ["4", 0],
            "image": ["5", 0]
        }
    },
    # Sample with structure control
    "10": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 25,
            "cfg": 7.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.75,
            "model": ["1", 0],
            "positive": ["9", 0],
            "negative": ["8", 0],
            "latent_image": ["6", 0]
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
            "filename_prefix": "style_transfer",
            "images": ["11", 0]
        }
    }
}


# Batch processing workflow template
def create_batch_workflow(image_files, base_workflow):
    """
    Create a batch processing workflow from a base workflow
    
    Args:
        image_files: List of image filenames to process
        base_workflow: Base workflow dictionary to replicate
    
    Returns:
        Batch workflow dictionary
    """
    batch_workflows = []
    
    for idx, image_file in enumerate(image_files):
        workflow = copy.deepcopy(base_workflow)
        # Update image input (adjust node ID as needed)
        if "2" in workflow and "inputs" in workflow["2"]:
            workflow["2"]["inputs"]["image"] = image_file
        # Update output prefix to avoid conflicts
        for node_id, node in workflow.items():
            if node.get("class_type") == "SaveImage":
                node["inputs"]["filename_prefix"] = f"batch_{idx:03d}"
        
        batch_workflows.append(workflow)
    
    return batch_workflows


def style_transfer_with_controlnet(
    content_image,
    style_description,
    checkpoint="v1-5-pruned-emaonly.safetensors",
    controlnet_model="control_v11f1p_sd15_depth.pth",
    structure_strength=0.7,
    style_strength=0.75,
    steps=25,
    cfg=7.5,
    seed=42
):
    """
    Apply style transfer while preserving content structure using ControlNet
    
    Args:
        content_image: Image to stylize
        style_description: Text description of desired style
        checkpoint: Base model checkpoint
        controlnet_model: ControlNet for structure preservation
        structure_strength: How much to preserve structure (0.0-1.0)
        style_strength: Denoise strength (higher = more stylization)
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed
    """
    workflow = copy.deepcopy(style_transfer_controlnet_workflow)
    
    workflow["1"]["inputs"]["ckpt_name"] = checkpoint
    workflow["2"]["inputs"]["image"] = content_image
    workflow["4"]["inputs"]["control_net_name"] = controlnet_model
    workflow["7"]["inputs"]["text"] = style_description
    workflow["9"]["inputs"]["strength"] = structure_strength
    workflow["10"]["inputs"]["seed"] = seed
    workflow["10"]["inputs"]["steps"] = steps
    workflow["10"]["inputs"]["cfg"] = cfg
    workflow["10"]["inputs"]["denoise"] = style_strength
    
    queue_prompt(workflow)
    print(f"Style transfer queued: {content_image}")


def batch_process_images(
    image_files,
    process_function,
    delay_between=2.0,
    **kwargs
):
    """
    Process multiple images with the same settings
    
    Args:
        image_files: List of image filenames
        process_function: Function to call for each image
        delay_between: Seconds to wait between submissions
        **kwargs: Additional arguments to pass to process_function
    """
    print(f"Starting batch processing of {len(image_files)} images...")
    
    for idx, image_file in enumerate(image_files):
        print(f"Processing {idx + 1}/{len(image_files)}: {image_file}")
        process_function(image_filename=image_file, **kwargs)
        
        if idx < len(image_files) - 1:
            time.sleep(delay_between)
    
    print(f"Batch processing complete! {len(image_files)} images queued.")


def create_image_sequence(
    base_prompt,
    num_frames,
    checkpoint="v1-5-pruned-emaonly.safetensors",
    width=512,
    height=512,
    steps=20,
    cfg=7.0,
    base_seed=42,
    transition_type="interpolate"
):
    """
    Generate a sequence of images for animation (simple seed-based transitions)
    
    Args:
        base_prompt: Base text prompt
        num_frames: Number of frames to generate
        checkpoint: Model checkpoint
        width: Image width
        height: Image height
        steps: Sampling steps
        cfg: CFG scale
        base_seed: Starting seed
        transition_type: "interpolate" or "random"
    """
    print(f"Generating image sequence: {num_frames} frames")
    
    for frame in range(num_frames):
        if transition_type == "interpolate":
            seed = base_seed + frame
        else:
            seed = base_seed + (frame * 1000)
        
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"{base_prompt}, frame {frame}",
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
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
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
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"sequence_frame_{frame:04d}",
                    "images": ["6", 0]
                }
            }
        }
        
        queue_prompt(workflow)
        time.sleep(1.0)  # Brief delay between frames
    
    print(f"Image sequence queued: {num_frames} frames")


def multi_resolution_generation(
    prompt,
    negative_prompt="blurry, low quality",
    checkpoint="v1-5-pruned-emaonly.safetensors",
    resolutions=[(512, 512), (768, 768), (1024, 1024)],
    steps=20,
    cfg=7.0,
    seed=42
):
    """
    Generate the same prompt at multiple resolutions
    
    Args:
        prompt: Text description
        negative_prompt: What to avoid
        checkpoint: Model checkpoint
        resolutions: List of (width, height) tuples
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed (same for all)
    """
    print(f"Generating at {len(resolutions)} resolutions")
    
    for width, height in resolutions:
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["1", 1]}
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["1", 1]}
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
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
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"multirez_{width}x{height}",
                    "images": ["6", 0]
                }
            }
        }
        
        queue_prompt(workflow)
        time.sleep(0.5)
    
    print(f"Multi-resolution generation queued")


if __name__ == "__main__":
    # Example 1: Style transfer with structure preservation
    style_transfer_with_controlnet(
        content_image="photo.png",
        style_description="impressionist painting style, vibrant colors, loose brushstrokes",
        structure_strength=0.7,
        style_strength=0.75,
        steps=25,
        seed=12345
    )
    
    # Example 2: Batch process multiple images (commented out)
    # from image_enhancement_example import upscale_image
    # batch_process_images(
    #     image_files=["img1.png", "img2.png", "img3.png"],
    #     process_function=upscale_image,
    #     delay_between=2.0,
    #     upscale_model="RealESRGAN_x4plus.pth"
    # )
    
    # Example 3: Create animation sequence (commented out)
    # create_image_sequence(
    #     base_prompt="a flower slowly blooming",
    #     num_frames=24,
    #     width=512,
    #     height=512,
    #     base_seed=42,
    #     transition_type="interpolate"
    # )
    
    # Example 4: Multi-resolution comparison (commented out)
    # multi_resolution_generation(
    #     prompt="a detailed landscape with mountains and lake",
    #     resolutions=[(512, 512), (768, 768), (1024, 1024)],
    #     seed=42
    # )
