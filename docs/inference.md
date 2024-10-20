# Inference with OmniGen


### Install
```bash
git clone https://github.com/staoxiao/OmniGen.git
cd OmniGen
pip install -e .
```



### Generate Images
You can use the following code to generate images:
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
    guidance_scale=2)
images[0].save("ti2i.png")
```

Some important arguments:
- `guidance_scale`: The strength of the guidance. Based on our experience, it is usually best to set it between 2 and 3. The higher the value, the more similar the generated image will be to the prompt. If the image appears oversaturated, please reduce the scale. 
- `height` and `width`: The height and width of the generated image. The default value is 1024x1024. OmniGen support any size, but these number must dividid by 16.
- `num_inference_steps`: The number of steps to take in the diffusion process. The higher the value, the more detailed the generated image will be.
- `separate_cfg_infer`: Whether to use separate inference process for CFG guidance. If set to True, memory cost will be lower but the generation speed will be slower. Default is False.
- `use_kv_cache`: Whether to use key-value cache. Default is True.
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


### OOM issue

If you encounter OOM issue, you can try to set `separate_cfg_infer=True` and `use_kv_cache=False`. This will reduce the memory usage but increase the generation speed.
You also can reduce the size of the image, e.g., `height=768, width=512`.