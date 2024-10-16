# OmniGen: Unified Image Generation



## Overview

We introduce a novel diffusion framework that unifies various image generation tasks within a single model, eliminating the need for task-specific networks or fine-tuning. ([Paper](https://arxiv.org/pdf/2409.11340))


## Results


![overall](imgs/overall.jpg)

## Generate Images

### Diffusers
TODO

### OmniGen
Install:
```bash
git clone https://github.com/staoxiao/OmniGen.git
cd OmniGen
pip install -e .
```

You can use the following code to generate images (more examples please refer to [inference.ipynb](inference.ipynb)):
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/tmp-preview")

# Text to Image
images = pipe(
    prompt="A woman holds a bouquet of flowers and faces the camera", 
    height=1024, 
    width=1024, 
    guidance_scale=5
    )
images[0].save("t2i.png")

# Multi-modal to Image
# In prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
images = pipe(
    prompt="A woman holds a bouquet of flowers and faces the camera. Thw woman is <img><|image_1|></img>.", 
    input_images=["./imgs/test_cases/liuyifei.png"], 
    height=1024, 
    width=1024,
    guidance_scale=3, 
    img_guidance_scale=1.6)
images[0].save("ti2i.png")
```



## Fine-tuning



## Plan

 - [x] Technical Report
 - [ ] Model
 - [ ] Code
 - [ ] Data
