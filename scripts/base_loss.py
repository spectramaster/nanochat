"""
Loads a checkpoint, and:
- Evaluates the loss on a larger chunk of train/val splits
- Samples from the model

Example run as:
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
"""
import os
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.utils import compute_init, compute_cleanup, print0
from nanochat.data.dataloader import tokenizing_distributed_data_loader
from nanochat.data.tokenizer import get_token_bytes
from nanochat.core.loss_eval import evaluate_bpb
from nanochat.runtime.engine import Engine

# Configuration
device_batch_size = 32
split_tokens = 20*524288  # number of tokens to evaluate per split
model_tag = None # optional model tag for the output directory name
model_step = None # optional model step for the output directory name
config_keys = [
    k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))
]
from nanochat.configs.loader import load_runtime_config

defaults = {k: globals()[k] for k in config_keys}
overrides = load_runtime_config(defaults=defaults)
globals().update(overrides)

# Load the base model and the tokenizer
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=model_step)
sequence_len = meta["model_config"]["sequence_len"] # could be arbitrary really

# Set up the precision we'll run with
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# Evaluate the loss on each split
tokens_per_step = device_batch_size * sequence_len * ddp_world_size
assert split_tokens % tokens_per_step == 0, "split_tokens must be divisible by tokens_per_step"
steps = split_tokens // tokens_per_step
token_bytes = get_token_bytes(device=device)
bpb_results = {}
for split_name in ["train", "val"]:
    loader = tokenizing_distributed_data_loader(
        device_batch_size,
        sequence_len,
        split_name,
        tokenizer_factory=lambda: tokenizer,
    )
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    print0(f"{split_name} bpb: {bpb:.4f}")
    bpb_results[split_name] = bpb

# Master process also samples from the model
samples = []
if ddp_rank == 0:
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        with autocast_ctx:
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        sample_str = tokenizer.decode(sample[0])
        print0(sample_str)
        samples.append(sample_str)

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model loss", data=[
    {
        "train bpb": bpb_results["train"],
        "val bpb": bpb_results["val"],
    },
    {f"sample {i}": sample for i, sample in enumerate(samples)},
])

# Cleanup
compute_cleanup()
