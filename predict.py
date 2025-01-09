# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import sys
from typing import List

from cog import BasePredictor, Input, Path
from PIL import Image

sys.path.insert(0, "OmniGen")
from OmniGen import OmniGenPipeline


MODEL_CACHE = "model_cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/Shitao/OmniGen-v1/{MODEL_CACHE}.tar"
)

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.pipe = OmniGenPipeline.from_pretrained(f"{MODEL_CACHE}/Shitao/OmniGen-v1")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt. For multi-modal to image generation with one or more input images, the placeholder in the prompt should be in the format of <img><|image_*|></img> (for the first image, the placeholder is <|image_1|>, for the second image, the the placeholder is <|image_2|>). Refer to examples for more details",
            default="a photo of an astronaut riding a horse on mars",
        ),
        img1: Path = Input(description="Input image 1. Optional", default=None),
        img2: Path = Input(description="Input image 2. Optional", default=None),
        img3: Path = Input(description="Input image 3. Optional", default=None),
        width: int = Input(
            description="Width of the output image", default=1024, ge=128, le=2048
        ),
        height: int = Input(
            description="Height of the output image", default=1024, ge=128, le=2048
        ),
        inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=50
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale for text prompt",
            ge=1,
            le=5,
            default=2.5,
        ),
        img_guidance_scale: float = Input(
            description="Classifier-free guidance scale for images",
            ge=1,
            le=2,
            default=1.6,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        max_input_image_size: int = Input(
            description="maximum input image size", ge=128, le=2048, default=1024
        ),
        separate_cfg_infer: bool = Input(
            description="Whether to use separate inference process for different guidance. This will reduce the memory cost.",
            default=True,
        ),
        offload_model: bool = Input(
            description="Offload model to CPU, which will significantly reduce the memory cost but slow down the generation speed. You can cancel separate_cfg_infer and set offload_model=True. If both separate_cfg_infer and offload_model are True, further reduce the memory, but slowest generation",
            default=False,
        ),
        use_input_image_size_as_output: bool = Input(
            description="Automatically adjust the output image size to be same as input image size. For editing and controlnet task, it can make sure the output image has the same size as input image leading to better performance",
            default=False,
        ),
        num_images: int = Input(description="The number of images to generate for the given inputs", default=1, ge=1, le=10),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        input_images = [str(img) for img in [img1, img2, img3] if img is not None]
        output = []
        for i in range(num_images):
            output = self.pipe(
                prompt=prompt,
                input_images=None if len(input_images) == 0 else input_images,
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
                num_images=num_images,
            )
            img = output[0]
            out_path = f"/tmp/out_{i}.png"
            img.save(out_path)
            output.append(Path(out_path))
        return output
