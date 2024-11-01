import json
from time import time
import argparse
import logging
import os
from pathlib import Path
import math

import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedType
from peft import LoraConfig, set_peft_model_state_dict, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from diffusers.models import AutoencoderKL

from OmniGen import OmniGen, OmniGenProcessor
from OmniGen.train_helper import DatasetFromJson, TrainDataCollator
from OmniGen.train_helper import training_losses
from OmniGen.utils import (
    create_logger,
    update_ema,
    requires_grad,
    center_crop_arr,
    crop_arr,
    vae_encode,
    vae_encode_list
)

def main(args):
    # Setup accelerator:
    from accelerate import DistributedDataParallelKwargs as DDPK
    kwargs = DDPK(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.results_dir,
        kwargs_handlers=[kwargs],
        )
    device = accelerator.device
    accelerator.init_trackers("tensorboard_log", config=args.__dict__)

    # Setup an experiment folder:
    checkpoint_dir = f"{args.results_dir}/checkpoints"  # Stores saved model checkpoints
    logger = create_logger(args.results_dir)
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created at {args.results_dir}")
        json.dump(args.__dict__, open(os.path.join(args.results_dir, 'train_args.json'), 'w'))


    # Create model:    
    if not os.path.exists(args.model_name_or_path):
        cache_folder = os.getenv('HF_HUB_CACHE')
        args.model_name_or_path = snapshot_download(repo_id=args.model_name_or_path,
                                        cache_dir=cache_folder,
                                        ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])
        logger.info(f"Downloaded model to {args.model_name_or_path}")
    model = OmniGen.from_pretrained(args.model_name_or_path)
    model.llm.config.use_cache = False
    model.llm.gradient_checkpointing_enable()
    model = model.to(device)

    if args.vae_path is None:
        vae_path = os.path.join(args.model_name_or_path, "vae")
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_pretrained(vae_path).to(device)
        else:
            logger.info("No VAE found in model, downloading stabilityai/sdxl-vae from HF")
            logger.info("If you have VAE in local folder, please specify the path with --vae_path")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(dtype=torch.float32)
    model.to(weight_dtype)

    processor = OmniGenProcessor.from_pretrained(args.model_name_or_path)

    requires_grad(vae, False)
    if args.use_lora:
        if accelerator.distributed_type == DistributedType.FSDP:
            raise NotImplementedError("FSDP does not support LoRA")
        requires_grad(model, False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["qkv_proj", "o_proj"],
        )
        model.llm.enable_input_require_grads()
        model = get_peft_model(model, transformer_lora_config)
        model.to(weight_dtype)
        transformer_lora_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        opt = torch.optim.AdamW(transformer_lora_parameters, lr=args.lr, weight_decay=args.adam_weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    ema = None
    if args.use_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
    

    # Setup data:
    crop_func = crop_arr
    if not args.keep_raw_resolution:
        crop_func = center_crop_arr
    image_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: crop_func(pil_image, args.max_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = DatasetFromJson(json_file=args.json_file, 
    image_path=args.image_path,
    processer=processor,
    image_transform=image_transform,
    max_input_length_limit=args.max_input_length_limit,
    condition_dropout_prob=args.condition_dropout_prob,
    keep_raw_resolution=args.keep_raw_resolution
    )
    collate_fn = TrainDataCollator(pad_token_id=processor.text_tokenizer.eos_token_id, hidden_size=model.llm.config.hidden_size, keep_raw_resolution=args.keep_raw_resolution)

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size_per_device,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
    )
    
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,}")

    num_update_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    
    if ema is not None:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
    

    if ema is not None:
        model, ema = accelerator.prepare(model, ema)
    else:
        model = accelerator.prepare(model)

    opt, loader, lr_scheduler = accelerator.prepare(opt, loader, lr_scheduler)
    
    
    # Variables for monitoring/logging purposes:
    train_steps, log_steps = 0, 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        
        for data in loader:
            with accelerator.accumulate(model):
                with torch.no_grad():
                    output_images = data['output_images']
                    input_pixel_values = data['input_pixel_values']
                    if isinstance(output_images, list):
                        output_images = vae_encode_list(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode_list(vae, input_pixel_values, weight_dtype)
                    else:
                        output_images = vae_encode(vae, output_images, weight_dtype)
                        if input_pixel_values is not None:
                            input_pixel_values = vae_encode(vae, input_pixel_values, weight_dtype)
                   

                model_kwargs = dict(input_ids=data['input_ids'], input_img_latents=input_pixel_values, input_image_sizes=data['input_image_sizes'], attention_mask=data['attention_mask'], position_ids=data['position_ids'], padding_latent=data['padding_images'], past_key_values=None, return_past_key_values=False)
                
                loss_dict = training_losses(model, output_images, model_kwargs)
                loss = loss_dict["loss"].mean()

                running_loss += loss.item()
                accelerator.backward(loss)
                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()

                log_steps += 1
                train_steps += 1

                accelerator.log({"training_loss": loss.item()}, step=train_steps)
                if train_steps % args.gradient_accumulation_steps == 0:
                    if accelerator.sync_gradients and ema is not None: 
                        update_ema(ema, model)
                    
                if train_steps % (args.log_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / args.gradient_accumulation_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / accelerator.num_processes 
                        
                    if accelerator.is_main_process:
                        cur_lr = opt.param_groups[0]["lr"]
                        logger.info(f"(step={int(train_steps/args.gradient_accumulation_steps):07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Epoch: {train_steps/len(loader)}, LR: {cur_lr}")

                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()


            if train_steps % (args.ckpt_every * args.gradient_accumulation_steps) == 0 and train_steps > 0:
                if accelerator.distributed_type == DistributedType.FSDP:
                    state_dict = accelerator.get_state_dict(model)
                    ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None
                else:
                    if not args.use_lora:
                        state_dict = model.module.state_dict()
                        ema_state_dict = accelerator.get_state_dict(ema) if ema is not None else None

                if accelerator.is_main_process:
                    if args.use_lora:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)

                        model.module.save_pretrained(checkpoint_path)
                    else:
                        checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}/"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
                        processor.text_tokenizer.save_pretrained(checkpoint_path)
                        model.llm.config.save_pretrained(checkpoint_path)
                        if ema_state_dict is not None:
                            checkpoint_path = f"{checkpoint_dir}/{int(train_steps/args.gradient_accumulation_steps):07d}_ema"
                            os.makedirs(checkpoint_path, exist_ok=True)
                            torch.save(ema_state_dict, os.path.join(checkpoint_path, "model.pt"))
                            processor.text_tokenizer.save_pretrained(checkpoint_path)
                            model.llm.config.save_pretrained(checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            dist.barrier()
    accelerator.end_training()
    model.eval()  
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model_name_or_path", type=str, default="OmniGen")
    parser.add_argument("--json_file", type=str)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size_per_device", type=int, default=1)
    parser.add_argument("--vae_path", type=str, default=None) 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=20000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_input_length_limit", type=int, default=1024)
    parser.add_argument("--condition_dropout_prob", type=float, default=0.1)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--keep_raw_resolution",
        action="store_true",
        help="multiple_resolutions",
    )
    parser.add_argument("--max_image_size", type=int, default=1344)

    parser.add_argument(
            "--use_lora",
            action="store_true",
        )
    parser.add_argument(
            "--lora_rank",
            type=int, 
            default=8
        )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether or not to use ema.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    ) 
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )


    args = parser.parse_args()
    assert args.max_image_size % 16 == 0, "Image size must be divisible by 16."

    main(args)


