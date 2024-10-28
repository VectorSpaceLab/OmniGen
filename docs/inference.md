# Inference with OmniGen

To handle some complex tasks, image generation models are becoming increasingly sophisticated, leading to more and more cumbersome workflows. Existing image generation models like SD and Flux require loading many additional network modules (such as ControlNet, IP-Adapter, Reference-Net) and extra preprocessing steps (e.g., face detection, pose detection, image cropping) to generate a satisfactory image. This complex workflow is not user-friendly. We believe that future image generation models should be simpler, generating various images directly through instructions, similar to how GPT works in language generation.

Therefore, we propose OmniGen, a model capable of handling various image generation tasks within a single framework. The goal of OmniGen is to complete various image generation tasks without relying on any additional components or image preprocessing steps. OmniGen supports tasks including text-to-image generation, image editing, subject-driven image generation, and classical vision tasks, among others. More capabilities can be found in our examples. We provide inference code so you can explore more unknown functionalities yourself.



## Install
```bash
git clone https://github.com/staoxiao/OmniGen.git
cd OmniGen
pip install -e .
```



## Generate Images
You can use the following code to generate images:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

# Text to Image
images = pipe(
    prompt="A curly-haired man in a red shirt is drinking tea.", 
    height=1024, 
    width=1024, 
    guidance_scale=2.5,
    seed=0,
)
images[0].save("example_t2i.png")  # save output PIL Image

# Multi-modal to Image
# In prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
images = pipe(
    prompt="A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
    input_images=["./imgs/test_cases/two_man.jpg"],
    height=1024, 
    width=1024,
    guidance_scale=2.5, 
    img_guidance_scale=1.6,
    max_input_image_size=1024,
    separate_cfg_infer=True, 
    use_kv_cache=True,
    offload_kv_cache=True,
    offload_model=False,
    use_input_image_size_as_output=False,
    seed=0,
)
images[0].save("example_ti2i.png")  # save output PIL image
```

Some important arguments:
- `guidance_scale`: The strength of the guidance. Based on our experience, it is usually best to set it between 2 and 3. The higher the value, the more similar the generated image will be to the prompt. If the image appears oversaturated, please reduce the scale. 
- `height` and `width`: The height and width of the generated image. The default value is 1024x1024. OmniGen support any size, but these number must be divisible by 16.
- `num_inference_steps`: The number of steps to take in the diffusion process. The higher the value, the more detailed the generated image will be.
- `max_input_image_size`: the maximum size of input image, which will be used to crop the input image to the maximum size. A smaller number will result in faster generation speed and lower memory cost.
- `separate_cfg_infer`: Whether to use separate inference process for CFG guidance. If set to True, memory cost will be lower. Default is True.
- `use_kv_cache`: Whether to use key-value cache. Default is True.
- `offload_kv_cache`:  offload the cached key and value to cpu, which can save memory but slow down the generation silightly. Default is True.
- `offload_model`: offload the model to cpu, which can save memory but slow down the generation. Default is False.
- `use_input_image_size_as_output`: whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task. Default is False.
- `seed`: The seed for random number generator.

**More examples please refer to [inference.ipynb](../inference.ipynb)**


#### Input data
OmniGen can accept multi-modal input data. Specifically, you should pass two arguments: `prompt` and `input_images`.
For text to image generation, you can pass a string as `prompt`, or pass a list of strings as `prompt` to generate multiple images.

For multi-modal to image generation, you should pass a string as `prompt`, and a list of image paths as `input_images`. The placeholder in the prompt should be in the format of `<img><|image_*|></img>`.
For example, if you want to generate an image with a person holding a bouquet of flowers, you can pass the following prompt:
```
prompt = "A woman holds a bouquet of flowers and faces the camera. Thw woman is <img><|image_1|></img>."
input_images = ["./imgs/test_cases/liuyifei.png"]
```
The placeholder `<|image_1|>` will be replaced by the image at `input_images[0]`, i.e., `./imgs/test_cases/liuyifei.png`.

If you want to generate multiple images, you can pass a list of prompts and a list of image paths. For example:
```
prompt = ["A woman holds a bouquet of flowers and faces the camera.", "A woman holds a bouquet of flowers and faces the camera. Thw woman is <img><|image_1|></img>."]
input_images = [[], ["./imgs/test_cases/liuyifei.png"]]
```


#### Gradio Demo
We have constructed a online demo in [Huggingface](https://huggingface.co/spaces/Shitao/OmniGen).

For the local gradio demo, you can run with the following command:
```python
python app.py
```


## Tips
- For out of memory or time cost, you can refer to [./docs/inference.md#requiremented-resources](https://github.com/VectorSpaceLab/OmniGen/blob/main/docs/inference.md#requiremented-resources) to select a appropriate setting.
- Oversaturated: If the image appears oversaturated, please reduce the `guidance_scale`.
- Not match the prompt: If the image does not match the prompt, please try to increase the `guidance_scale`.
- Low-quality: More detailed prompt will lead to better results. 
- Animate Style: If the genereate images is in animate style, you can try to add `photo` to the prompt`.
- Edit generated image. If you generate a image by omnigen and then want to edit it, you cannot use the same seed to edit this image. For example, use seed=0 to generate image, and should use seed=1 to edit this image.
- For image editing tasks, we recommend placing the image before the editing instruction. For example, use `<img><|image_1|></img> remove suit`, rather than `remove suit <img><|image_1|></img>`. 
- For image editing task and controlnet task, we recommend to set the height and width of output image as the same
as input image. For example, if you want to edit a 512x512 image, you should set the height and width of output image as 512x512. You also can set the `use_input_image_size_as_output` to automatically set the height and width of output image as the same as input image.


## Requiremented Resources

We are currently experimenting with some techniques to reduce memory usage and improve speed, including `use_kv_cache, offload_kv_cache, separate_cfg_infer, offload_model`, which you can enable in the pipeline. 
The default setting is`use_kv_cache=True, offload_kv_cache=True, separate_cfg_infer=True, offload_model=False`. 


We conducted experiments on the A800 and RTX 3090. The memory requirements and inference times are shown in the table below. You can choose the appropriate settings based on your available resources.


- Different image size. 

Different image size (`max_input_image_size` is the max size of input image, `height` and `width` are the size of output image) with the default inference settings (`use_kv_cache=True,offload_kv_cache=True,separate_cfg_infer=True`)

For A800 GPU:  
| Settings     |  Only Text | Text + Single Image |  Text + Two Images    |
|:-------------|:----------:|:-------------------:|:---------------------:|
| max_input_image_size=1024,height=1024,width=1024 | 9G, 31s   | 12G, 1m6s  | 13G, 1m20s  |
| max_input_image_size=512,height=1024,width=1024 | 9G, 31s   | 10G, 50s  | 10G, 54s |
| max_input_image_size=768,height=768,width=768 | 9G, 16s  | 10G, 32s  | 10G, 37s  |
| max_input_image_size=512,height=512,width=512 | 9G, 7s   | 9G, 14s  | 9G, 15s  |

For RTX 3090 GPU:
| Settings     |  Only Text | Text + Single Image |  Text + Two Images    |
|:-------------|:----------:|:-------------------:|:---------------------:|
| max_input_image_size=1024,height=1024,width=1024 | 9G, 1m17s   | 12G, 2m46s  | 13G, 3m23s  |
| max_input_image_size=512,height=1024,width=1024 | 9G, 1m18s   | 10G, 2m8s  | 10G, 2m18s |
| max_input_image_size=768,height=768,width=768 | 9G, 41s  | 10G, 1m22s  | 10G, 1m38s  |
| max_input_image_size=512,height=512,width=512 | 9G, 19s   | 9G, 36s  | 9G, 43s  |


You can set smaller `max_input_image_size` to reduce memory usage, but note that the generation quality may be lower.
And please set the `height` and `width` the same as the size of input image for image editing task.


- Different inference settings

Default image size: height=1024, width=1024, max_input_image_size=1024

For A800 GPU:  
| Settings     |  Only Text | Text + Single Image |  Text + Two Images    |
|:-------------|:----------:|:-------------------:|:---------------------:|
| use_kv_cache | 18G, 30s   | 36G, 1m  | 48G, 1m13s |
| use_kv_cache,offload_kv_cache | 10G, 30s   | 14G, 1m10s | 17G, 1m30s  |
| use_kv_cache,offload_kv_cache,separate_cfg_infer | 9G, 31s   | 12G, 1m6s  | 13G, 1m20s  |
| use_kv_cache,offload_kv_cache,offload_model | 4G, 55s   | 7G, 1m30s  | 11G, 1m48s  |
| use_kv_cache,offload_kv_cache,separate_cfg_infer,offload_model | 3G, 1m23s   | 5G, 2m19s   | 6G, 2m30s |

For RTX 3090 GPU:
| Settings     |  Only Text | Text + Single Image |  Text + Two Images    |
|:-------------|:----------:|:-------------------:|:---------------------:|
| use_kv_cache | 18G, 1m14s   | OOM  | OOM  |
| use_kv_cache,offload_kv_cache | 10G, 1m17s   | 14G, 3m11s | 17G, 4m3s  |
| use_kv_cache,offload_kv_cache,separate_cfg_infer | 9G, 1m18s   | 12G, 2m46s  | 13G, 3m21s  |
| use_kv_cache,offload_kv_cache,offload_model | 4G,3m1s    | 7G, 4m14s  | 11G, 5m4s  |
| use_kv_cache,offload_kv_cache,separate_cfg_infer,offload_model | 3G, 4m56s   | 5G, 7m49s   | 6G, 8m6s |

Overall, the text-to-image task only requires minimal memory and time cost, but when input images are used, the computational cost increases. You can reduce memory usage by extending the processing time.
