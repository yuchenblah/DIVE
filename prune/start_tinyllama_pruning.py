import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import LlamaTokenizer
from models.modeling_llama import LlamaForCausalLM

from importlib.metadata import version

from lib.prune_mlp_unif import prune_flap_mlp, check_mlp_sparsity

import os
import pandas as pd

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
    )
    
    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.o_proj.bias = nn.Parameter(torch.zeros_like(model.model.layers[i].self_attn.o_proj.bias, device="cuda"))
        model.model.layers[i].mlp.down_proj.bias = nn.Parameter(torch.zeros_like(model.model.layers[i].mlp.down_proj.bias, device="cuda"))
        nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)

    model.seqlen = 256
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="TinyLlama/TinyLlama_v1.1", type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=42, help='Seed for sampling the calibration data.')
    parser.add_argument('--mlp_pruning_ratio', type=float, default=0, help='MLP pruning ratio.')
    parser.add_argument('--expert', type=str, default="Expert0", help='Calibration data for pruning.')
    parser.add_argument('--nsamples', type=int, default=1024, help='Number of calibration samples.')
    parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
    parser.add_argument('--unstr', action="store_true")
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save_model', type=str, required=True, help='Path to save the pruned model.')
    args = parser.parse_args()


    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # Build the model and tokenizer
    print(f"loading LLM {args.model}...")
    model = get_llm(args.model, args.cache_dir).to(device).eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)

    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    print("*" * 30)


    # Prune the model
    print("pruning starts...")
    if args.metrics == 'N/A':
        raise ValueError("For FLAP pruning, the metrics parameter must be chosen from ['IFV', 'WIFV', 'WIFN']. 'N/A' is not a valid choice.")
    prune_flap_mlp(args, model, tokenizer, device)
    print("*" * 30)
    

    # Check the sparsity of the model
    print("sparsity check starts...")
    mlp_sparsity_ratio = check_mlp_sparsity(model)
    print(f"Total Model Parameter: {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B")
    print("*" * 30)
    

    # Save the model
    print("model saving starts...")
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    print("completes!")


if __name__ == '__main__':
    main()
