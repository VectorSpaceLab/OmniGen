import gradio as gr
from PIL import Image
import os
import argparse
import random
import spaces
from transformers import BitsAndBytesConfig

from OmniGen import OmniGenPipeline, OmniGen

@spaces.GPU(duration=180)
def generate_image(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, inference_steps, seed, separate_cfg_infer, offload_model,
            use_input_image_size_as_output, max_input_image_size, randomize_seed, save_images):
    input_images = [img1, img2, img3]
    # Delete None
    input_images = [img for img in input_images if img is not None]
    if len(input_images) == 0:
        input_images = None
    
    if randomize_seed:
        seed = random.randint(0, 10000000)

    output = pipe(
        prompt=text,
        input_images=input_images,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        img_guidance_scale=img_guidance_scale,
        num_inference_steps=inference_steps,
        separate_cfg_infer=separate_cfg_infer, 
        use_kv_cache=True,
        offload_kv_cache=True,
        offload_model=offload_model,
        use_input_image_size_as_output=use_input_image_size_as_output,
        seed=seed,
        max_input_image_size=max_input_image_size,
    )
    img = output[0]
    
    if save_images:
        # Save All Generated Images
        from datetime import datetime
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join('outputs', f'{timestamp}.png')
        # Save the image
        img.save(output_path)
    
    return img

def get_example():
    case = [
        [
            "A curly-haired man in a red shirt is drinking tea.",
            None,
            None,
            None,
            1024,
            1024,
            2.5,
            1.6,
            0,
            1024,
            False,
            False,
        ],
        [
            "The woman in <img><|image_1|></img> waves her hand happily in the crowd",
            "./imgs/test_cases/zhang.png",
            None,
            None,
            1024,
            1024,
            2.5,
            1.9,
            128,
            1024,
            False,
            False,
        ],
        [
            "A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
            "./imgs/test_cases/two_man.jpg",
            None,
            None,
            1024,
            1024,
            2.5,
            1.6,
            0,
            1024,
            False,
            False,
        ],
        [
            "Two woman are raising fried chicken legs in a bar. A woman is <img><|image_1|></img>. Another woman is <img><|image_2|></img>.",
            "./imgs/test_cases/mckenna.jpg",
            "./imgs/test_cases/Amanda.jpg",
            None,
            1024,
            1024,
            2.5,
            1.8,
            65,
            1024,
            False,
            False,
        ],
        [
            "A man and a short-haired woman with a wrinkled face are standing in front of a bookshelf in a library. The man is the man in the middle of <img><|image_1|></img>, and the woman is oldest woman in <img><|image_2|></img>",
            "./imgs/test_cases/1.jpg",
            "./imgs/test_cases/2.jpg",
            None,
            1024,
            1024,
            2.5,
            1.6,
            60,
            1024,
            False,
            False,
        ],
        [
            "A man and a woman are sitting at a classroom desk. The man is the man with yellow hair in <img><|image_1|></img>. The woman is the woman on the left of <img><|image_2|></img>",
            "./imgs/test_cases/3.jpg",
            "./imgs/test_cases/4.jpg",
            None,
            1024,
            1024,
            2.5,
            1.8,
            66,
            1024,
            False,
            False,
        ],
        [
            "The flower <img><|image_1|></img> is placed in the vase which is in the middle of <img><|image_2|></img> on a wooden table of a living room",
            "./imgs/test_cases/rose.jpg",
            "./imgs/test_cases/vase.jpg",
            None,
            1024,
            1024,
            2.5,
            1.6,
            66,
            1024,
            False,
            False,
        ],
        [
            "<img><|image_1|><img>\n Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola.",
            "./imgs/demo_cases/t2i_woman_with_book.png",
            None,
            None,
            None,
            None,
            2.5,
            1.6,
            222,
            1024,
            False,
            True,
        ],
        [
            "Detect the skeleton of human in this image: <img><|image_1|></img>.",
            "./imgs/test_cases/control.jpg",
            None,
            None,
            1024,
            1024,
            2.0,
            1.6,
            0,
            1024,
            False,
            True,
        ],
        [
            "Generate a new photo using the following picture and text as conditions: <img><|image_1|><img>\n A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him.",
            "./imgs/demo_cases/skeletal.png",
            None,
            None,
            1024,
            1024,
            2,
            1.6,
            999,
            1024,
            False,
            True,
        ],
        [
            "Following the pose of this image <img><|image_1|><img>, generate a new photo: A young boy is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him.",
            "./imgs/demo_cases/edit.png",
            None,
            None,
            1024,
            1024,
            2.0,
            1.6,
            123,
            1024,
            False,
            True,
        ],
        [
            "Following the depth mapping of this image <img><|image_1|><img>, generate a new photo: A young girl is sitting on a sofa in the library, holding a book. His hair is neatly combed, and a faint smile plays on his lips, with a few freckles scattered across his cheeks. The library is quiet, with rows of shelves filled with books stretching out behind him.",
            "./imgs/demo_cases/edit.png",
            None,
            None,
            1024,
            1024,
            2.0,
            1.6,
            1,
            1024,
            False,
            True,
        ],
        [
            "<img><|image_1|><\/img> What item can be used to see the current time? Please highlight it in blue.",
            "./imgs/test_cases/watch.jpg",
            None,
            None,
            1024,
            1024,
            2.5,
            1.6,
            666,
            1024,
            False,
            True,
        ],
        [
            "According to the following examples, generate an output for the input.\nInput: <img><|image_1|></img>\nOutput: <img><|image_2|></img>\n\nInput: <img><|image_3|></img>\nOutput: ",
            "./imgs/test_cases/icl1.jpg",
            "./imgs/test_cases/icl2.jpg",
            "./imgs/test_cases/icl3.jpg",
            224,
            224,
            2.5,
            1.6,
            1,
            768,
            False,
            False,
        ],
    ]
    return case

def run_for_examples(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, seed, max_input_image_size, randomize_seed, use_input_image_size_as_output, save_images):    
    # 在函数内部设置默认值
    inference_steps = 50
    separate_cfg_infer = True
    offload_model = False
    
    return generate_image(
        text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, 
        inference_steps, seed, separate_cfg_infer, offload_model,
        use_input_image_size_as_output, max_input_image_size, randomize_seed, save_images
    )

description = """
OmniGen is a unified image generation model that you can use to perform various tasks, including, but not limited to, text-to-image generation, subject-driven generation, Identity-Preserving Generation, and image-conditioned generation.
For multi-modal to image generation, you should pass a string as `prompt`, and a list of image paths as `input_images`. The placeholder in the prompt should be in the format of `<img><|image_*|></img>` (for the first image, the placeholder is <img><|image_1|></img>. for the second image, the placeholder is <img><|image_2|></img>).
For example, use an image of a woman to generate a new image:
prompt = "A woman holds a bouquet of flowers and faces the camera. The woman is \<img\>\<|image_1|\>\</img\>."

Tips:
- For image editing task and controlnet task, we recommend setting the height and width of output image as the same as input image. For example, if you want to edit a 512x512 image, you should set the height and width of output image as 512x512. You also can set the `use_input_image_size_as_output` to automatically set the height and width of output image as the same as input image.
- For out-of-memory or time cost, you can set `offload_model=True` or refer to [./docs/inference.md#requiremented-resources](https://github.com/VectorSpaceLab/OmniGen/blob/main/docs/inference.md#requiremented-resources) to select a appropriate setting.
- If inference time is too long when inputting multiple images, please try to reduce the `max_input_image_size`. For more details please refer to [./docs/inference.md#requiremented-resources](https://github.com/VectorSpaceLab/OmniGen/blob/main/docs/inference.md#requiremented-resources).
- Oversaturated: If the image appears oversaturated, please reduce the `guidance_scale`.
- Not matching the prompt: If the image does not match the prompt, please try to increase the `guidance_scale`.
- Low-quality: A more detailed prompt will lead to better results. 
- Animated Style: If you want the generated image to appear less animated, and more realistic, you can try adding `photo` to the prompt.
- Editing generated images: If you generate an image with OmniGen, and then want to edit it, you cannot use the same seed to edit this image. For example, use seed=0 to generate the image, and then use seed=1 to edit this image.
- Image editing: In your prompt, we recommend placing the image before the editing instructions. For example, use `<img><|image_1|></img> remove suit`, rather than `remove suit <img><|image_1|></img>`.

HF Spaces often encounter errors due to quota limitations, so recommend to run it locally.

"""

article = """
---
**Citation** 
<br> 
If you find this repository useful, please consider giving a star ⭐ and a citation
```
@article{xiao2024omnigen,
  title={Omnigen: Unified image generation},
  author={Xiao, Shitao and Wang, Yueze and Zhou, Junjie and Yuan, Huaying and Xing, Xingrun and Yan, Ruiran and Wang, Shuting and Huang, Tiejun and Liu, Zheng},
  journal={arXiv preprint arXiv:2409.11340},
  year={2024}
}
```
**Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out via email.
"""


# Gradio 
with gr.Blocks() as demo:
    gr.Markdown("# OmniGen: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen)")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            # text prompt
            prompt_input = gr.Textbox(
                label="Enter your prompt, use <img><|image_i|></img> to represent i-th input image", placeholder="Type your prompt here..."
            )

            with gr.Row(equal_height=True):
                # input images
                image_input_1 = gr.Image(label="<img><|image_1|></img>", type="filepath")
                image_input_2 = gr.Image(label="<img><|image_2|></img>", type="filepath")
                image_input_3 = gr.Image(label="<img><|image_3|></img>", type="filepath")

            # slider
            height_input = gr.Slider(
                label="Height", minimum=128, maximum=2048, value=1024, step=16
            )
            width_input = gr.Slider(
                label="Width", minimum=128, maximum=2048, value=1024, step=16
            )

            guidance_scale_input = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=5.0, value=2.5, step=0.1
            )

            img_guidance_scale_input = gr.Slider(
                label="img_guidance_scale", minimum=1.0, maximum=2.0, value=1.6, step=0.1
            )

            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=1, maximum=100, value=50, step=1
            )

            seed_input = gr.Slider(
                label="Seed", minimum=0, maximum=2147483647, value=42, step=1
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            max_input_image_size = gr.Slider(
                label="max_input_image_size", minimum=128, maximum=2048, value=1024, step=16
            )

            separate_cfg_infer = gr.Checkbox(
                label="separate_cfg_infer", info="Whether to use separate inference process for different guidance. This will reduce the memory cost.", value=True,
            )
            offload_model = gr.Checkbox(
                label="offload_model", info="Offload model to CPU, which will significantly reduce the memory cost but slow down the generation speed. You can cancel separate_cfg_infer and set offload_model=True. If both separate_cfg_infer and offload_model are True, further reduce the memory, but slowest generation", value=False,
            )
            use_input_image_size_as_output = gr.Checkbox(
                label="use_input_image_size_as_output", info="Automatically adjust the output image size to be same as input image size. For editing and controlnet task, it can make sure the output image has the same size as input image leading to better performance", value=False,
            )

            # generate
            generate_button = gr.Button("Generate Image")
            

        with gr.Column():
            with gr.Column():
                # quantization = gr.Radio(["4bit (NF4)", "8bit", "None (bf16)"], label="bitsandbytes quantization", value="4bit (NF4)")
                # quantization.input(change_quantization, inputs=quantization, trigger_mode="once", concurrency_limit=1)
                # output image
                output_image = gr.Image(label="Output Image")
                save_images = gr.Checkbox(label="Save generated images", value=False)

    # click
    generate_button.click(
        generate_image,
        inputs=[
            prompt_input,
            image_input_1,
            image_input_2,
            image_input_3,
            height_input,
            width_input,
            guidance_scale_input,
            img_guidance_scale_input,
            num_inference_steps,
            seed_input,
            separate_cfg_infer,
            offload_model,
            use_input_image_size_as_output,
            max_input_image_size,
            randomize_seed,
            save_images,
        ],
        outputs=output_image,
    )

    gr.Examples(
        examples=get_example(),
        fn=run_for_examples,
        inputs=[
            prompt_input,
            image_input_1,
            image_input_2,
            image_input_3,
            height_input,
            width_input,
            guidance_scale_input,
            img_guidance_scale_input,
            seed_input,
            max_input_image_size,
            randomize_seed,
            use_input_image_size_as_output,
        ],
        outputs=output_image,
    )

    gr.Markdown(article)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the OmniGen')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app')
    parser.add_argument('-b', '--nbits', choices=['4','8'], help='bitsandbytes quantization n-bits')
    args = parser.parse_args()
    
    quantization_config = None
    model = None
    if args.nbits:
        if args.nbits == '4':
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')
        elif args.nbits == '8':
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        model = OmniGen.from_pretrained("Shitao/OmniGen-v1", quantization_config=quantization_config)

    
    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1", model=model)
    
    # launch
    demo.launch(share=args.share)
