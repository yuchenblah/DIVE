CUDA_VISIBLE_DEVICES=0 python main.py \
    --folder_path "./pruned_models/tinyllama_0.5" \
    --default_config_path "./configurations/tinyllama_smoe/config_8_1_0.5.json" \
    --save_model_path "./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5" \
    --instruction "establish"