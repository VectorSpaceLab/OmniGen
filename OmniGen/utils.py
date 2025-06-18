import gc
import logging

from PIL import Image
import torch
import numpy as np

from transformers import BitsAndBytesConfig
from transformers.quantizers import AutoHfQuantizer
from transformers.integrations import  replace_with_bnb_linear, set_module_quantized_tensor_to_device, get_keys_to_not_convert

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)




def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def crop_arr(pil_image, max_image_size):
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % 16) // 2
    crop_y2 = arr.shape[0] % 16 - crop_y1

    crop_x1 = (arr.shape[1] % 16) // 2
    crop_x2 = arr.shape[1] % 16 - crop_x1

    arr = arr[crop_y1:arr.shape[0]-crop_y2, crop_x1:arr.shape[1]-crop_x2]    
    return Image.fromarray(arr)



def vae_encode(vae, x, weight_dtype):
    if x is not None:
        if vae.config.shift_factor is not None:
            x = vae.encode(x).latent_dist.sample()
            x = (x - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            x = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
        x = x.to(weight_dtype)
    return x

def vae_encode_list(vae, x, weight_dtype):
    latents = []
    for img in x:
        img = vae_encode(vae, img, weight_dtype)
        latents.append(img)
    return latents



@torch.no_grad()
def quantize_bnb(meta_model, state_dict:dict, quantization_config:BitsAndBytesConfig, pre_quantized=None, dtype=None):
    if pre_quantized is None:
        if quantization_config.load_in_4bit:
            pre_quantized = any('bitsandbytes__' in k for k in state_dict)
        elif quantization_config.load_in_8bit:
            pre_quantized = any('weight_format' in k for k in state_dict)
    
    if quantization_config.llm_int8_skip_modules is None:
        quantization_config.llm_int8_skip_modules = get_keys_to_not_convert(meta_model.llm) # ['norm']
    
    quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=pre_quantized)        
    
    meta_model.eval()
    meta_model.requires_grad_(False)
    
    model = meta_model

    quantizer.preprocess_model(model, device_map=None,)
    
    # iterate the model keys, otherwise quantized state dict will throws errors
    for param_name in model.state_dict():
        param = state_dict[param_name]
        if not pre_quantized:
            param = param.to(dtype)
        
        if not quantizer.check_quantized_param(model, param, param_name, state_dict):
            set_module_quantized_tensor_to_device(model, param_name, device=0, value=param)
        else:
            quantizer.create_quantized_param(model, param, param_name, target_device=0, state_dict=state_dict)
        
        del state_dict[param_name], param
    
    model = quantizer.postprocess_model(model)
    
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()
    return model