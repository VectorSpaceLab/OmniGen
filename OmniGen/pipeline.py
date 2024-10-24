import os
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from PIL import Image
import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from diffusers.models import AutoencoderKL
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from safetensors.torch import load_file

from OmniGen import OmniGen, OmniGenProcessor, OmniGenScheduler


logger = logging.get_logger(__name__) 

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from OmniGen import OmniGenPipeline
        >>> pipe = FluxControlNetPipeline.from_pretrained(
        ...     base_model
        ... )
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=3.0,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""



class OmniGenPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGen,
        processor: OmniGenProcessor,
    ):
        self.vae = vae
        self.model = model
        self.processor = processor

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            logger.info("Don't detect any available devices, using CPU instead")
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()
        self.vae.to(self.device)

    @classmethod
    def from_pretrained(cls, model_name, vae_path: str=None):
        if not os.path.exists(model_name) or (not os.path.exists(os.path.join(model_name, 'model.safetensors')) and model_name == "Shitao/OmniGen-v1"):
            logger.info("Model not found, downloading...")
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'model.pt'])
            logger.info(f"Downloaded model to {model_name}")
        model = OmniGen.from_pretrained(model_name)
        processor = OmniGenProcessor.from_pretrained(model_name)

        if os.path.exists(os.path.join(model_name, "vae")):
            vae = AutoencoderKL.from_pretrained(os.path.join(model_name, "vae"))
        elif vae_path is not None:
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            logger.info(f"No VAE found in {model_name}, downloading stabilityai/sdxl-vae from HF")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

        return cls(vae, model, processor)
    
    def merge_lora(self, lora_path: str):
        model = PeftModel.from_pretrained(self.model, lora_path)
        model.merge_and_unload()

        
        self.model = model
    
    def to(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self.vae.to(device)

    def vae_encode(self, x, dtype):
        if self.vae.config.shift_factor is not None:
            x = self.vae.encode(x).latent_dist.sample()
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = self.vae.encode(x).latent_dist.sample().mul_(self.vae.config.scaling_factor)
        x = x.to(dtype)
        return x
    
    def move_to_device(self, data):
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        separate_cfg_infer: bool = False,
        use_kv_cache: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = None,
        ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation. 
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the 1-th image in list.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. The number must be a multiple of 16.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. The number must be a multiple of 16.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            separate_cfg_infer (`bool`, *optional*, defaults to False):
                Perform inference on images with different guidance separately; this can save memory when generating images of large size at the expense of slower inference.
            use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
        Examples:

        Returns:
            A list with the generated images.
        """
        assert height%16 == 0 and width%16 == 0
        if separate_cfg_infer:
            use_kv_cache = False
            # raise "Currently, don't support both use_kv_cache and separate_cfg_infer"
        if input_images is None:
            use_img_guidance = False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None

        input_data = self.processor(prompt, input_images, height=height, width=width, use_img_cfg=use_img_guidance, separate_cfg_input=separate_cfg_infer)

        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1
        latent_size_h, latent_size_w = height//8, width//8

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
        latents = torch.cat([latents]*(1+num_cfg), 0).to(dtype)

        input_img_latents = []
        if separate_cfg_infer:
            for temp_pixel_values in input_data['input_pixel_values']:
                temp_input_latents = []
                for img in temp_pixel_values:
                    img = self.vae_encode(img.to(self.device), dtype)
                    temp_input_latents.append(img)
                input_img_latents.append(temp_input_latents)
        else:
            for img in input_data['input_pixel_values']:
                img = self.vae_encode(img.to(self.device), dtype)
                input_img_latents.append(img)

        model_kwargs = dict(input_ids=self.move_to_device(input_data['input_ids']), 
            input_img_latents=input_img_latents, 
            input_image_sizes=input_data['input_image_sizes'], 
            attention_mask=self.move_to_device(input_data["attention_mask"]), 
            position_ids=self.move_to_device(input_data["position_ids"]), 
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
            use_kv_cache=use_kv_cache)
        
        if separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg
        self.model.to(dtype)

        scheduler = OmniGenScheduler(num_steps=num_inference_steps)
        samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache)
        samples = samples.chunk((1+num_cfg), dim=0)[0]

        samples = samples.to(torch.float32)
        if self.vae.config.shift_factor is not None:
            samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            samples = samples / self.vae.config.scaling_factor   
        samples = self.vae.decode(samples).sample
        
        output_samples = (samples * 0.5 + 0.5).clamp(0, 1)*255
        output_samples = output_samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        output_images = []
        for i, sample in enumerate(output_samples):  
            output_images.append(Image.fromarray(sample))
        
        return output_images