import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from lib.eval import eval_ppl
from models.modeling_pruned_tinyllama_smoe import PrunedLlamaSMoEForCausalLM


def get_pruned_smoe_llm(model_path, cache_dir="./pruned_smoe_models"):
    model = PrunedLlamaSMoEForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    return model


def test_model_on_seqlen(model_path, tokenizer, seqlens, device):
    model = get_pruned_smoe_llm(model_path)
    model.eval()

    pruned_smoe_model = nn.DataParallel(model).to(device)

    for seqlen in seqlens:
        print(f"Model: {model_path}, SeqLength: {seqlen}")
        perplexities = eval_ppl(pruned_smoe_model, tokenizer, seqlen, device)
        test_datasets = ["WikiText2"]
        for dataset, ppl in zip(test_datasets, perplexities):
            print(f"ppl on {dataset}: {ppl}")
        print("*" * 30)

    pruned_smoe_model = pruned_smoe_model.module if isinstance(pruned_smoe_model, nn.DataParallel) else pruned_smoe_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./checkpoints/tinyllama_8_1_stage1_results",
                        help="Base directory containing model checkpoints")
    parser.add_argument("--tokenizer_path", type=str, default="./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5",
                        help="Tokenizer path")
    parser.add_argument("--startswith", type=str, default=None, help="Optional prefix filter for folder names")
    parser.add_argument("--endswith", type=str, default=None, help="Optional suffix filter for folder names")

    args = parser.parse_args()

    seqlens = [512, 1024, 2048]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for folder_name in os.listdir(args.base_dir):
        if (
            (args.startswith is None or folder_name.startswith(args.startswith)) and
            (args.endswith is None or folder_name.endswith(args.endswith))
        ):
            model_path = os.path.join(args.base_dir, folder_name)
            test_model_on_seqlen(model_path, tokenizer, seqlens, device)


if __name__ == "__main__":
    main()
