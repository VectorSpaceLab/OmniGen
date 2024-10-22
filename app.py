import gradio as gr
from PIL import Image
import os
import spaces

from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained(
    "Shitao/OmniGen-v1"
)

@spaces.GPU(duration=120)
# 示例处理函数：生成图像
def generate_image(text, img1, img2, img3, height, width, guidance_scale, inference_steps, seed):
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
        separate_cfg_infer=True,
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
            "A vintage camera placed on the ground, ejecting a swirling cloud of Polaroid-style photographs into the air. The photos, showing landscapes, wildlife, and travel scenes, seem to defy gravity, floating upward in a vortex of motion. The camera emits a glowing, smoky light from within, enhancing the magical, surreal atmosphere. The dark background contrasts with the illuminated photos and camera, creating a dreamlike, nostalgic scene filled with vibrant colors and dynamic movement. Scattered photos are visible on the ground, further contributing to the idea of an explosion of captured memories.",
            None,
            None,
            None,
            1024,
            1024,
            2.5,
            50,
            0,
        ],
        [
            "A woman <img><|image_1|></img> in a wedding dress. Next to her is a black-haired man.",
            "./imgs/test_cases/yifei2.png",
            None,
            None,
            1024,
            1024,
            2.5,
            50,
            0,
        ],
        [
            "A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
            "./imgs/test_cases/two_man.jpg",
            None,
            None,
            1024,
            1024,
            2.5,
            50,
            0,
        ],
        [
            "Two men are celebrating with raised glasses in a restaurant. A man is <img><|image_1|></img>. The other man is <img><|image_2|></img>.",
            "./imgs/test_cases/young_musk.jpg",
            "./imgs/test_cases/young_trump.jpeg",
            None,
            1024,
            1024,
            2.5,
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
            50,
            123,
        ],
        [
            "<img><|image_1|><\/img> What item can be used to see the current time? Please remove it.",
            "./imgs/test_cases/watch.jpg",
            None,
            None,
            1024,
            1024,
            2.5,
            50,
            0,
        ],
        [
            "Three guitars are displayed side by side on a rustic wooden stage, each showcasing its unique character and style. The left guitar is <img><|image_1|><\/img>. The middle guitar is <img><|image_2|><\/img>. The right guitars is <img><|image_3|><\/img>.",
            "./imgs/test_cases/guitar1.png",
            "./imgs/test_cases/guitar1.png",
            "./imgs/test_cases/guitar1.png",
            1024,
            1024,
            2.5,
            50,
            0,
        ],
    ]
    return case

def run_for_examples(text, img1, img2, img3, height, width, guidance_scale, inference_steps, seed):    
    return generate_image(text, img1, img2, img3, height, width, guidance_scale, inference_steps, seed)

description = """
OmniGen is a unified image generation model that you can use to perform various tasks, including but not limited to text-to-image generation, subject-driven generation, Identity-Preserving Generation, and image-conditioned generation.

For multi-modal to image generation, you should pass a string as `prompt`, and a list of image paths as `input_images`. The placeholder in the prompt should be in the format of `<img><|image_*|></img>`.
For example, use a image of a woman to generate a new image:
prompt = "A woman holds a bouquet of flowers and faces the camera. Thw woman is \<img\>\<|image_1|\>\</img\>."
"""

# Gradio 接口
with gr.Blocks() as demo:
    gr.Markdown("# OmniGen: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen)")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            # 文本输入框
            prompt_input = gr.Textbox(
                label="Enter your prompt, use <img><|image_i|></img> tokens for images", placeholder="Type your prompt here..."
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
                label="Guidance Scale", minimum=1.0, maximum=10.0, value=3.0, step=0.1
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
            num_inference_steps,
            seed_input,
        ],
        outputs=output_image,
    )

# 启动应用
demo.launch()