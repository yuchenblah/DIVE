import torch
from transformers import TrainingArguments, Trainer, LlamaTokenizer, default_data_collator

from models.modeling_pruned_tinyllama_smoe import PrunedLlamaSMoEForCausalLM
from lib.traindata_process import traindata_loaders

import pathlib


def get_pruned_smoe_llm(model, instruction="establish", temperature_coefficient=1.0, cache_dir="./pruned_smoe_models"):
    model = PrunedLlamaSMoEForCausalLM.from_pretrained(
        model,
        instruction=instruction,
        temperature_coefficient=temperature_coefficient,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    )
    return model

def get_router_parameters(model):
    parameters = []
    for layer in model.model.layers:
        noisy_topk_router = layer.pruned_smoe.router
        parameters.extend(noisy_topk_router.parameters())
    return parameters

def print_trainable_parameters(model):
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}")


def train_router(args):
    torch.cuda.empty_cache()

    tokenizer = LlamaTokenizer.from_pretrained(args.save_model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = get_pruned_smoe_llm(args.save_model_path, instruction=args.instruction, temperature_coefficient=args.temperature_coefficient)
    model.train()
    model.gradient_checkpointing_enable()


    # --------- LOAD DATASET ---------
    train_dataset, val_dataset = traindata_loaders(tokenizer, max_length=args.max_length, seed=args.seed)

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_bs,
        per_device_eval_batch_size=args.per_device_eval_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_safetensors=False,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        report_to="tensorboard",
        bf16=True,
        deepspeed=args.deepspeed,
    )

    trainable_params = get_router_parameters(model)

    # --------- PARAMETER UNFREEZING ---------
    # Unfreeze router parameters
    for param in model.parameters():
        param.requires_grad = False
    for param in trainable_params:
        param.requires_grad = True

    print_trainable_parameters(model)

    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
