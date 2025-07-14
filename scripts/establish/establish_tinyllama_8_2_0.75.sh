CUDA_VISIBLE_DEVICES=0 python main.py \
    --folder_path "./pruned_models/tinyllama_0.75" \
    --default_config_path "./configurations/tinyllama_smoe/config_8_2_0.75.json" \
    --save_model_path "./pruned_smoe_models/tinyllama_smoe/tinyllama_8_2_0.75" \
    --instruction "establish"