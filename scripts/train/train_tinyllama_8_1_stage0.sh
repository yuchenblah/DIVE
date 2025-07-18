deepspeed --include=localhost:0,1,2,3 --master_port=29501 main.py \
    --instruction "train_router" \
    --temperature_coefficient 5e-2 \
    --save_model_path "./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5" \
    --learning_rate 1e-4 \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --max_steps 1905 \
    --per_device_train_bs 16 \
    --per_device_eval_bs 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 0.1 \
    --save_steps 0.2 \
    --logging_steps 1 \
    --output_dir './checkpoints/tinyllama_8_1_stage0_results' \
    --logging_dir './checkpoints/tinyllama_8_1_stage0_results/logs' \
    --deepspeed './configurations/ds_config/zero2_stage0.json' \
    --max_length 1024