deepspeed --include=localhost:0,1,2,3 --master_port=29501 main.py \
    --instruction "train_expert" \
    --save_model_path "./checkpoints/tinyllama_8_1_stage0_results/checkpoint-1905" \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --max_steps 9535 \
    --per_device_train_bs 32 \
    --per_device_eval_bs 32 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 0.1 \
    --save_steps 0.1 \
    --logging_steps 1 \
    --output_dir './checkpoints/tinyllama_8_1_stage1_results' \
    --logging_dir './checkpoints/tinyllama_8_1_stage1_results/logs' \
    --peft_model_path './checkpoints/tinyllama_8_1_stage1_results/peft-model' \
    --merged_model_path './checkpoints/tinyllama_8_1_stage1_results/merged-model' \
    --deepspeed './configurations/ds_config/zero2_stage1.json' \
    --max_length 1024