CUDA_VISIBLE_DEVICES=0 python ./test/ppl.py \
    --base_dir ./checkpoints/tinyllama_8_2_stage0_results \
    --tokenizer_path ./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5 \
    --startswith checkpoint-