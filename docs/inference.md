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
    prompt="A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>."
    input_images=["./imgs/test_cases/two_man.jpg"]
    height=1024, 
    width=1024,
    separate_cfg_infer=False,  # if OOM, you can set separate_cfg_infer=True 
    guidance_scale=2.5, 
    img_guidance_scale=1.6
)
images[0].save("example_ti2i.png")  # save output PIL image
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


#### Gradio Demo
We have constructed a online demo in [Huggingface](https://huggingface.co/spaces/Shitao/OmniGen).

For the local gradio demo, you can run with the following command:
```python
python app.py
```


## Tips
- OOM issue: If you encounter OOM issue, you can try to set `separate_cfg_infer=True`. This will reduce the memory usage but increase the generation latecy. You also can reduce the size of the image, e.g., `height=768, width=512`.
- Oversaturated: If the image appears oversaturated, please reduce the `guidance_scale`.
- Not match the prompt: If the image does not match the prompt, please try to increase the `guidance_scale`.
- Low-quality: More detailed prompt will lead to better results. Besides, larger size of the image (`height` and `width`) will also help.
- Animate Style: If the genereate images is in animate style, you can try to add `photo` to the prompt`.
- Edit generated image. If you generate a image by omnigen and then want to edit it, you cannot use the same seed to edit this image. For example, use seed=0 to generate image, and should use seed=1 to edit this image.
- For image editing tasks, we recommend placing the image before the editing instruction. For example, use `<img><|image_1|></img> remove suit`, rather than `remove suit <img><|image_1|></img>`.