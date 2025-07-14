import argparse

from models.modeling_pruned_tinyllama_smoe import establish_pruned_llama_smoe
from trainers.train_router import train_router
from trainers.train_expert import train_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--seed', type=int, default=1234)
    
    parser.add_argument('--instruction', type=str, default="establish", choices=["establish", "train_router", "train_expert"])
    parser.add_argument('--temperature_coefficient', type=float, default=1.0)

    # folder_path: Path of the Folder of Pruned Models
    # - Needs to Match config.num_experts
    parser.add_argument('--folder_path', type=str, default="./pruned_models/tinyllama_0.5")
    # default_config_path: Path of the Default SMoE Config
    # - Used to Replace the Pruning Model Config
    parser.add_argument('--default_config_path', type=str, default="./configurations/tinyllama_smoe/config_8_1_0.5.json")
    # save_model_path: Path of the Established Pruned SMoE Models
    parser.add_argument('--save_model_path', type=str, default="./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5")
    parser.add_argument('--tokenizer_path', type=str, default="./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5")
    
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--lr_scheduler_type', default="cosine", type=str)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--num_train_epochs', default=1, type=int)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--per_device_train_bs', default=8, type=int) 
    parser.add_argument('--per_device_eval_bs', default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=16, type=int)
    parser.add_argument('--evaluation_strategy', default='steps', type=str)
    parser.add_argument('--eval_steps', default=0.05, type=float)
    parser.add_argument('--save_steps', default=0.05, type=float)
    parser.add_argument('--save_total_limit', default=10, type=int)
    parser.add_argument('--logging_steps', default=1, type=float)
    parser.add_argument('--output_dir', default='./checkpoints/tinyllama_8_1_15B_results', type=str)
    parser.add_argument('--logging_dir', default='./logs/tinyllama_8_1_15B_results', type=str)
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--peft_model_path', default='./checkpoints/tinyllama_8_1_15B_results/peft-model', type=str)
    parser.add_argument('--merged_model_path', default='./checkpoints/tinyllama_8_1_15B_results/merged-model', type=str)
    parser.add_argument('--deepspeed', default='./configurations/ds_config_zero2_stage1.json', type=str)
    args = parser.parse_args()


    folder_path = args.folder_path
    default_config_path = args.default_config_path
    save_model_path = args.save_model_path

    if args.instruction == "establish":
        establish_pruned_llama_smoe(folder_path, default_config_path, save_model_path)
    if args.instruction == "train_router":
        train_router(args)
    if args.instruction == "train_expert":
        train_expert(args)

main()