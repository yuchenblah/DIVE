#!/bin/bash

datasets=("Expert0" "Expert1" "Expert2" "Expert3" "Expert4" "Expert5" "Expert6" "Expert7")
pruning_ratios=("0.75")
commands=()

for expert in "${datasets[@]}"; do
    for ratio in "${pruning_ratios[@]}"; do
        command="CUDA_VISIBLE_DEVICES=0 python prune/start_tinyllama_pruning.py --model "TinyLlama/TinyLlama_v1.1" --mlp_pruning_ratio $ratio --expert $expert --nsamples 1024 --metrics WIFV --save_model ./pruned_models/tinyllama_0.75/tinyllama_MLP_${ratio}_${expert}_1024_WIFV"
        commands+=("$command")
    done
done

echo "Number of commands: ${#commands[@]}"

for cmd in "${commands[@]}"; do
    echo "Running command: $cmd"
    eval "$cmd"
done