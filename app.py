import gradio as gr
from PIL import Image
import os
import spaces

from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1"
)

@spaces.GPU(duration=180)
# 示例处理函数：生成图像
def generate_image(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, inference_steps, seed):
    input_images = [img1, img2, img3]
    # 去除 None
    input_images = [img for img in input_images if img is not None]
    if len(input_images) == 0:
        input_images = None

    output = pipe(
        prompt=text,
        input_images=input_images,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        img_guidance_scale=1.6,
        num_inference_steps=inference_steps,
        separate_cfg_infer=True, # set False can speed up the inference process
        use_kv_cache=False,
        seed=seed,
    )
    img = output[0]
    return img
# def generate_image(text, img1, img2, img3, height, width, guidance_scale, inference_steps):
#     input_images = []
#     if img1:
#         input_images.append(Image.open(img1))
#     if img2:
#         input_images.append(Image.open(img2))
#     if img3:
#         input_images.append(Image.open(img3))
        
#     return input_images[0] if input_images else None


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
            50,
            0,
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
            50,
            128,
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
            50,
            0,
        ],
        [
            "Two woman are raising fried chicken legs in a bar. A woman is <img><|image_1|></img>. The other woman is <img><|image_2|></img>.",
            "./imgs/test_cases/mckenna.jpg",
            "./imgs/test_cases/Amanda.jpg",
            None,
            1024,
            1024,
            2.5,
            1.8,
            50,
            168,
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
            50,
            60,
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
            50,
            66,
        ],
        [
            "The flower <img><|image_1|><\/img> is placed in the vase which is in the middle of <img><|image_2|><\/img> on a wooden table of a living room",
            "./imgs/test_cases/rose.jpg",
            "./imgs/test_cases/vase.jpg",
            None,
            1024,
            1024,
            2.5,
            1.6,
            50,
            0,
        ],
        [
            "<img><|image_1|><img>\n Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola.",
            "./imgs/demo_cases/t2i_woman_with_book.png",
            None,
            None,
            1024,
            1024,
            2.5,
            1.6,
            50,
            222,
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
            50,
            0,
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
            50,
            42,
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
            50,
            123,
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
            50,
            1,
        ],
        [
            "<img><|image_1|><\/img> What item can be used to see the current time? Please remove it.",
            "./imgs/test_cases/watch.jpg",
            None,
            None,
            1024,
            1024,
            2.5,
            1.6,
            50,
            0,
        ],
        [
            "According to the following examples, generate an output for the input.\nInput: <img><|image_1|></img>\nOutput: <img><|image_2|></img>\n\nInput: <img><|image_3|></img>\nOutput: ",
            "./imgs/test_cases/icl1.jpg",
            "./imgs/test_cases/icl2.jpg",
            "./imgs/test_cases/icl3.jpg",
            1024,
            1024,
            2.5,
            1.6,
            50,
            1,
        ],
    ]
    return case

def run_for_examples(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, inference_steps, seed):    
    return generate_image(text, img1, img2, img3, height, width, guidance_scale, img_guidance_scale, inference_steps, seed)

description = """
OmniGen is a unified image generation model that you can use to perform various tasks, including but not limited to text-to-image generation, subject-driven generation, Identity-Preserving Generation, and image-conditioned generation.

For multi-modal to image generation, you should pass a string as `prompt`, and a list of image paths as `input_images`. The placeholder in the prompt should be in the format of `<img><|image_*|></img>` (for the first image, the placeholder is <img><|image_1|></img>. for the second image, the the placeholder is <img><|image_2|></img>).
For example, use an image of a woman to generate a new image:
prompt = "A woman holds a bouquet of flowers and faces the camera. Thw woman is \<img\>\<|image_1|\>\</img\>."

Tips:
- Oversaturated: If the image appears oversaturated, please reduce the `guidance_scale`.
- Low-quality: More detailed prompt will lead to better results. 
- Animate Style: If the genereate images is in animate style, you can try to add `photo` to the prompt`.
- Edit generated image. If you generate a image by omnigen and then want to edit it, you cannot use the same seed to edit this image. For example, use seed=0 to generate image, and should use seed=1 to edit this image.
- For image editing tasks, we recommend placing the image before the editing instruction. For example, use `<img><|image_1|></img> remove suit`, rather than `remove suit <img><|image_1|></img>`.
"""

# Gradio 接口
with gr.Blocks() as demo:
    gr.Markdown("# OmniGen: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen)")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            # 文本输入框
            prompt_input = gr.Textbox(
                label="Enter your prompt, use <img><|image_i|></img> to represent i-th input image", placeholder="Type your prompt here..."
            )

            with gr.Row(equal_height=True):
                # 图片上传框
                image_input_1 = gr.Image(label="<img><|image_1|></img>", type="filepath")
                image_input_2 = gr.Image(label="<img><|image_2|></img>", type="filepath")
                image_input_3 = gr.Image(label="<img><|image_3|></img>", type="filepath")

            # 高度和宽度滑块
            height_input = gr.Slider(
                label="Height", minimum=256, maximum=2048, value=1024, step=16
            )
            width_input = gr.Slider(
                label="Width", minimum=256, maximum=2048, value=1024, step=16
            )

            # 引导尺度输入
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

            # 生成按钮
            generate_button = gr.Button("Generate Image")

        with gr.Column():
            # 输出图像框
            output_image = gr.Image(label="Output Image")

    # 按钮点击事件
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
            img_guidance_scale,
            num_inference_steps,
            seed_input,
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
            img_guidance_scale,
            num_inference_steps,
            seed_input,
        ],
        outputs=output_image,
    )

# 启动应用
demo.launch()