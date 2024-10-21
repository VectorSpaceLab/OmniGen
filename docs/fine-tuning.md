# Fine-tuning OmniGen

Fine-tuning Omnigen can better help you handle specific image generation tasks. For example, by fine-tuning on a person's images, you can generate multiple pictures of that person while maintaining task consistency.

Previous work focused on designing new networks to solve specific tasks. For instance, ControlNet was designed to handle image conditions, and IP-Adapter was designed to maintain ID features. For these models, if you want to perform new tasks, you need to design new network architectures and repeatedly debug them. Adding extra network parameters is time-consuming and labor-intensive, which is not user-friendly. However, with Omnigen, all of this becomes very simple.

Unlike other models, Omnigen has the capability to accept multi-modal conditional inputs and has been pre-trained on various tasks. You can fine-tune this model on any task without needing to design specialized networks like ControlNet or IP-Adapter for a specific task. 
**All you need to do is prepare the data and start training. You can break the limitations of previous models, allowing Omnigen to accomplish a variety of interesting tasks, even those that have never been done before.**


## Installation

```bash
git clone https://github.com/VectorSpaceLab/OmniGen.git
cd OmniGen
pip install -e .
```


## Full fine-tuning

### fine-tuning command

```bash
accelerate launch  \
--num_processes=1 \
--use_fsdp \
--fsdp_offload_params false \
--fsdp_sharding_strategy SHARD_GRAD_OP \
--fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
--fsdp_transformer_layer_cls_to_wrap Phi3DecoderLayer \
--fsdp_state_dict_type FULL_STATE_DICT \
--fsdp_forward_prefetch false \
--fsdp_use_orig_params True \
--fsdp_cpu_ram_efficient_loading false \
--fsdp_sync_module_states True \
train.py \
--model_name_or_path Shitao/OmniGen-v1 \
--json_file ./toy_data/toy_data.jsonl \
--image_path ./toy_data/images \
--batch_size_per_device 1 \
--lr 2e-5 \
--keep_raw_resolution \
--max_image_size 1024 \
--gradient_accumulation_steps 1 \
--ckpt_every 10 \
--epochs 100 \
--log_every 1 \
--results_dir ./results/toy_finetune
```

Some important arguments:
- num_processes: number of GPU to use for training
- model_name_or_path: path to the pretrained model
- json_file: path to the json file containing the training data, e.g., ./toy_data/toy_data.jsonl
- image_path: path to the image folder, e.g., ./toy_data/images
- batch_size_per_device: batch size per device
- lr: learning rate
- keep_raw_resolution: whether to keep the original resolution of the image, if not, all images will be resized to (max_image_size, max_image_size)
- max_image_size: max image size
- gradient_accumulation_steps: number of steps to accumulate gradients
- ckpt_every: number of steps to save checkpoint
- epochs: number of epochs
- log_every: number of steps to log
- results_dir: path to the results folder

The data format of json_file is as follows:
```
{"instruction": str, "input_images": [str, str, ...], "output_images": str}
```
You can see a toy example in `./toy_data/toy_data.jsonl`.

If an OOM(Out of Memory) issue occurs, you can try to decrease the `batch_size_per_device` or `max_image_size`. You can also try to use LoRA instead of full fine-tuning.


### Inference

The checkpoint can be found at `{results_dir}/checkpoints/*`. You can use the following command to load saved checkpoint:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("checkpoint_path") # e.g., ./results/toy_finetune/checkpoints/0000010
```





## LoRA fine-tuning
LoRA fine-tuning is a simple way to fine-tune OmniGen with less GPU memory. To use lora, you should add `--use_lora` and `--lora_rank` to the command.

```bash
accelerate launch  \
--num_processes=1 \
train.py \
--model_name_or_path Shitao/OmniGen-v1 \
--batch_size_per_device 2 \
--condition_dropout_prob 0.01 \
--lr 3e-4 \
--use_lora \
--lora_rank 8 \
--json_file ./toy_data/toy_data.jsonl \
--image_path ./toy_data/images \
--max_input_length_limit 18000 \
--keep_raw_resolution \
--max_image_size 1024 \
--gradient_accumulation_steps 1 \
--ckpt_every 10 \
--epochs 100 \
--log_every 1 \
--results_dir ./results/toy_finetune_lora
```

### Inference

The checkpoint can be found at `{results_dir}/checkpoints/*`. You can use the following command to load checkpoint:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
pipe.merge_lora("checkpoint_path") # e.g., ./results/toy_finetune_lora/checkpoints/0000010
```


## A simple example

Here is an example for learning new concepts: "sks dog". We use five images of one dog from https://huggingface.co/datasets/diffusers/dog-example. The json file is `./toy_data/toy_subject_data.jsonl`, and the images have been saved in `./toy_data/images`.

```bash
accelerate launch  \
--num_processes=1 \
train.py \
--model_name_or_path Shitao/OmniGen-v1 \
--batch_size_per_device 2 \
--condition_dropout_prob 0.01 \
--lr 1e-3 \
--use_lora \
--lora_rank 8 \
--json_file ./toy_data/toy_subject_data.jsonl \
--image_path ./toy_data/images \
--max_input_length_limit 18000 \
--keep_raw_resolution \
--max_image_size 1024 \
--gradient_accumulation_steps 1 \
--ckpt_every 10 \
--epochs 200 \
--log_every 1 \
--results_dir ./results/toy_finetune_lora
```

After training, you can use the following command to generate images:
```python
from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")
pipe.merge_lora("checkpoint_path") # e.g., ./results/toy_finetune_lora/checkpoints/0000010

images = pipe(
    prompt="a photo of sks dog running in the snow", 
    height=1024, 
    width=1024, 
    guidance_scale=3
    )
images[0].save("sks_dog_snow.png")
```
