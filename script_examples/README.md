# ComfyUI Script Examples

This directory contains comprehensive examples demonstrating various ComfyUI workflows through Python scripts. These examples show how to programmatically interact with the ComfyUI API to accomplish different tasks.

## Overview

All examples use the ComfyUI API format and can be run against a running ComfyUI server. The examples are designed to be educational and provide starting points for your own workflows.

## Prerequisites

1. A running ComfyUI server (default: `http://127.0.0.1:8188`)
2. Required models downloaded and placed in appropriate folders:
   - Checkpoint models (Stable Diffusion, SDXL, etc.)
   - ControlNet models (for controlnet_example.py)
   - Upscaling models (for image_enhancement_example.py)
   - Video models (for video_generation_example.py)
   - IP-Adapter models (for face_character_example.py)

## Available Examples

### 1. basic_api_example.py
**Basic text-to-image generation**

The simplest example showing how to:
- Queue a prompt to ComfyUI
- Generate an image from text
- Modify parameters like seed and prompt

```python
python basic_api_example.py
```

### 2. websockets_api_example.py
**Text-to-image with WebSocket monitoring**

Shows how to:
- Connect via WebSocket to monitor execution
- Wait for generation completion
- Download generated images programmatically

```python
python websockets_api_example.py
```

### 3. inpainting_example.py
**Image inpainting with masks**

Demonstrates:
- Loading existing images and masks
- VAEEncodeForInpaint for mask-based editing
- Inpainting specific areas of an image
- Customizable inpainting parameters

Features:
- ✅ Load image and mask
- ✅ Control inpainting strength
- ✅ Adjust generation parameters
- ✅ Multiple mask channel options

```python
from inpainting_example import run_inpainting

run_inpainting(
    image_filename="example.png",
    mask_filename="mask.png",
    positive_prompt="a cat sitting on a chair, photorealistic",
    steps=30,
    cfg=7.5
)
```

### 4. video_generation_example.py
**Video generation (text-to-video and image-to-video)**

Comprehensive video examples including:
- **Hunyuan Video**: Text-to-video generation
- **Hunyuan Video 1.5**: Image-to-video with CLIP Vision
- **LTX-Video**: Alternative text-to-video model

Features:
- ✅ Text-to-video generation
- ✅ Image-to-video (first frame conditioning)
- ✅ Configurable resolution and length
- ✅ Multiple video model support

```python
from video_generation_example import generate_text_to_video_hunyuan

generate_text_to_video_hunyuan(
    prompt="A cinematic shot of a person walking through a forest",
    width=848,
    height=480,
    length=25,
    steps=20
)
```

### 5. controlnet_example.py
**ControlNet for pose and depth control**

Shows advanced control techniques:
- **OpenPose**: Control human poses and positions
- **Depth Control**: Maintain scene structure
- **Multi-ControlNet**: Combine multiple control types

Features:
- ✅ Pose-guided generation (people positioning)
- ✅ Depth-guided scene control
- ✅ Automatic preprocessors (OpenPose, MiDaS)
- ✅ Multiple ControlNets simultaneously
- ✅ Adjustable control strength

```python
from controlnet_example import generate_with_pose_control

generate_with_pose_control(
    reference_image="dancer.png",
    prompt="a ballet dancer in elegant costume",
    strength=1.0,
    steps=25
)
```

### 6. image_enhancement_example.py
**Image upscaling and enhancement**

Multiple enhancement workflows:
- **Simple Upscaling**: Using ESRGAN/RealESRGAN models
- **Img2Img**: Transform images with text guidance
- **Hires Fix**: Two-pass high-resolution generation
- **Upscale + Enhance**: Combine upscaling with AI enhancement

Features:
- ✅ 4x upscaling with various models
- ✅ Image-to-image transformation
- ✅ High-resolution generation workflow
- ✅ Combined upscaling and enhancement
- ✅ Adjustable denoise strength

```python
from image_enhancement_example import upscale_image, img2img_transform

# Simple upscaling
upscale_image("photo.png", upscale_model="RealESRGAN_x4plus.pth")

# Transform style
img2img_transform(
    "sketch.png",
    prompt="professional oil painting",
    denoise=0.75
)
```

### 7. face_character_example.py
**Face swapping and character consistency**

Character-focused workflows:
- **IP-Adapter**: Character reference for consistency
- **Face Enhancement**: Improve facial details
- **Multi-Character**: Multiple people in scenes

Features:
- ✅ Character-consistent generation
- ✅ IP-Adapter for style/character transfer
- ✅ Face detail enhancement
- ✅ Multi-character scene generation
- ✅ Reference image guidance

```python
from face_character_example import generate_with_character_reference

generate_with_character_reference(
    reference_image="character_ref.png",
    prompt="the character walking in a park",
    weight=0.8,
    steps=25
)
```

**Note**: Face swapping typically requires custom nodes like ReActor or InstantID.

## Usage Patterns

### Basic Usage
All examples can be run directly:
```bash
python script_examples/<example_name>.py
```

### Function-Based Usage
Import and use specific functions:
```python
from script_examples.inpainting_example import run_inpainting
from script_examples.video_generation_example import generate_text_to_video_hunyuan

# Use the functions with custom parameters
run_inpainting(...)
generate_text_to_video_hunyuan(...)
```

### Modifying Workflows
Each example includes the complete workflow dictionary. You can:
1. Copy the workflow
2. Modify node parameters
3. Add or remove nodes
4. Create custom combinations

## Configuration

### Server Address
Change the server address at the top of each file:
```python
server_address = "127.0.0.1:8188"  # Default
# or
server_address = "192.168.1.100:8188"  # Remote server
```

### Model Files
Update model filenames to match your installation:
```python
checkpoint="v1-5-pruned-emaonly.safetensors"  # SD 1.5
# or
checkpoint="sd_xl_base_1.0.safetensors"  # SDXL
```

## Common Parameters

### Image Generation
- `seed`: Random seed for reproducibility (default: 42)
- `steps`: Number of sampling steps (default: 20-25)
- `cfg`: CFG scale/guidance strength (default: 7.0-8.0)
- `sampler_name`: Sampler algorithm (euler, dpmpp_2m, etc.)
- `scheduler`: Scheduler type (normal, karras, etc.)
- `denoise`: Denoise strength for img2img (0.0-1.0)

### Video Generation
- `width`: Video width (must be divisible by 16 or 32)
- `height`: Video height (must be divisible by 16 or 32)
- `length`: Number of frames (varies by model)
- `frame_rate`: Output video frame rate (default: 24-25)

### ControlNet
- `strength`: Control strength (0.0-1.0)
- `resolution`: Preprocessor resolution (default: 512)
- `detect_hand/detect_face`: Enable/disable detection (OpenPose)

## Troubleshooting

### Connection Issues
```python
# Error: Connection refused
# Solution: Ensure ComfyUI server is running
python main.py  # Start ComfyUI server first
```

### Missing Models
```python
# Error: Model not found
# Solution: Download required models to appropriate folders
# - checkpoints/ for main models
# - controlnet/ for ControlNet models
# - upscale_models/ for upscaling models
# - ipadapter/ for IP-Adapter models
```

### File Not Found
```python
# Error: Image file not found
# Solution: Place images in ComfyUI's input folder
# Default: ComfyUI/input/
```

## Advanced Topics

### Exporting Workflows from UI
To create your own examples:
1. Build a workflow in the ComfyUI web interface
2. Click "File → Export (API Format)"
3. Save the JSON
4. Convert to Python dictionary
5. Add to your script

### Custom Nodes
Some examples mention custom nodes:
- **ReActor**: Face swapping - https://github.com/Gourieff/comfyui-reactor-node
- **InstantID**: Identity preservation - https://github.com/cubiq/ComfyUI_InstantID
- **ControlNet Aux**: Preprocessors - https://github.com/Fannovel16/comfyui_controlnet_aux

Install via ComfyUI Manager or manually clone into `custom_nodes/`

### Batch Processing
Extend examples for batch processing:
```python
images = ["image1.png", "image2.png", "image3.png"]
for image in images:
    upscale_image(image)
```

## Model Requirements by Example

| Example | Required Models |
|---------|----------------|
| basic_api_example | SD checkpoint |
| websockets_api_example | SD checkpoint |
| inpainting_example | SD checkpoint |
| video_generation_example | Video model (Hunyuan/LTX/Mochi) |
| controlnet_example | SD checkpoint + ControlNet models |
| image_enhancement_example | SD checkpoint + Upscale models |
| face_character_example | SD checkpoint + IP-Adapter + CLIP Vision |

## Tips

1. **Start Simple**: Begin with basic_api_example.py
2. **Check Models**: Ensure all required models are downloaded
3. **Monitor Execution**: Use websockets_api_example.py to see progress
4. **Adjust Parameters**: Lower steps/resolution for faster testing
5. **Read Error Messages**: ComfyUI provides detailed error information
6. **Use Appropriate Models**: Match checkpoint and other models (SD1.5 with SD1.5 ControlNet, etc.)

## Resources

- [ComfyUI Examples](https://comfyanonymous.github.io/ComfyUI_examples/): Official workflow examples
- [ComfyUI Docs](https://docs.comfy.org/): Official documentation
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI): Source code and issues
- [Model Downloads](https://huggingface.co/): HuggingFace model repository

## Contributing

When adding new examples:
1. Follow the existing format
2. Include comprehensive docstrings
3. Add configurable parameters
4. Update this README
5. Test with a clean ComfyUI installation

## License

These examples are provided as-is for educational purposes. Refer to the main ComfyUI license for usage terms.
