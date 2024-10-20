# OmniGen: Unified Image Generation

<div align="center">

[![HF Demo](https://img.shields.io/badge/HF%20Demo-🤗-lightblue)]()&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2409.11340-b31b1b.svg)](https://arxiv.org/abs/2409.11340)&nbsp;
[![Model](https://img.shields.io/badge/OmniGen-🤗-yellow)]()&nbsp;
[![Project Page](https://img.shields.io/badge/OmniGen-Page-yellow)]()&nbsp;

</div>




## Overview

To handle some complex tasks, image generation models are becoming increasingly sophisticated, leading to more and more cumbersome workflows. Existing image generation models like SD and Flux require loading many additional network modules (such as ControlNet, IP-Adapter, Reference-Net) and extra preprocessing steps (e.g., face detection, pose detection, image cropping) to generate a satisfactory image. This complex workflow is not user-friendly. We believe that future image generation models should be simpler, generating various images directly through instructions, similar to how GPT works in language generation.

Therefore, we propose OmniGen, a model capable of handling various image generation tasks within a single framework. The goal of OmniGen is to complete various image generation tasks without relying on any additional components or image preprocessing steps. OmniGen supports tasks including text-to-image generation, image editing, subject-driven image generation, and classical vision tasks, among others. More capabilities can be found in our examples. We provide inference code so you can explore more unknown functionalities yourself.

Due to current limitations in resources and data, OmniGen still has room for improvement. We will continue to optimize the model. You can also choose to fine-tune the model for specific tasks. Fine-tuning OmniGen is very straightforward. Since OmniGen's architecture natively supports various tasks, you don't need to worry about designing a network structure for a specific task; you just need to prepare the corresponding data, and you can endow OmniGen with new image generation capabilities through fine-tuning. Imagination is no longer limited; you can construct any image generation task, and perhaps we can achieve some very interesting things.

If you have any questions or interesting tasks you want OmniGen to accomplish, feel free to discuss with us: 2906698981@qq.com.



## Update
- 2024-10-21: We release the training code for OmniGen.
- 2024-10-21: We release the first version of OmniGen.



## Results


![overall](imgs/overall.jpg)

## Generate Images

### Using OmniGen
Install:
```bash
git clone https://github.com/staoxiao/OmniGen.git
cd OmniGen
pip install -e .
```


You can see our examples in [inference.ipynb](inference.ipynb) to generate images. 
Here are some simple examples:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/tmp-preview")

# Text to Image
images = pipe(
    prompt="A woman holds a bouquet of flowers and faces the camera", 
    height=1024, 
    width=1024, 
    guidance_scale=3
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
For more details about inference, please refer to [docs/inference.md](docs/inference.md).

### Diffusers
Coming soon.


## Fine-tuning
We provide a train scrip `train.py` to fine-tune OmniGen. Please refer to [docs/finetune.md](docs/finetune.md) for more details.


## citation

```
@article{xiao2024omnigen,
  title={Omnigen: Unified image generation},
  author={Xiao, Shitao and Wang, Yueze and Zhou, Junjie and Yuan, Huaying and Xing, Xingrun and Yan, Ruiran and Wang, Shuting and Huang, Tiejun and Liu, Zheng},
  journal={arXiv preprint arXiv:2409.11340},
  year={2024}
}
```





