import torch
from transformers import TrainingArguments, Trainer, LlamaTokenizer, default_data_collator

from models.modeling_pruned_tinyllama_smoe import PrunedLlamaSMoEForCausalLM
from lib.traindata_process import traindata_loaders
from peft import LoraConfig, get_peft_model
import copy
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

def get_layernorm_parameters(model):
    parameters = []
    for layer in model.model.layers:
        parameters.extend(layer.input_layernorm.parameters())
        parameters.extend(layer.post_attention_layernorm.parameters())
    parameters.extend(model.model.norm.parameters())
    return parameters

def get_trainable_parameters(model):
    parameters = []
    parameters.extend(get_router_parameters(model))
    parameters.extend(get_layernorm_parameters(model))
    return parameters


class PeftTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=True):
        if output_dir is None:
            output_dir = self.args.output_dir
        self._save(output_dir, _internal_call=_internal_call)

    def _save(self, output_dir, state_dict=None, _internal_call=True):
        merged_dir = output_dir + "-merged-model"

        if hasattr(self.model, "base_model"):
            temp_model = copy.deepcopy(self.model.base_model)
            temp_model.merge_and_unload()
            temp_model.save_pretrained(merged_dir)

        super()._save(output_dir, state_dict=state_dict)


def train_expert(args):
    torch.cuda.empty_cache()

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model = get_pruned_smoe_llm(args.save_model_path, args.instruction, args.temperature_coefficient)
    model.train()
    model.gradient_checkpointing_enable()

    # --------- TRAINING SETTINGS ---------
    # Load dataset
    train_dataset, val_dataset = traindata_loaders(tokenizer, max_length = args.max_length, seed = args.seed)

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
        # save_total_limit=args.save_total_limit,
        save_safetensors=False,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        report_to="tensorboard",
        bf16=True,
        deepspeed=args.deepspeed,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        target_modules=["gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainable_params = get_trainable_parameters(model)
    model = get_peft_model(model, lora_config)

    # --------- PARAMETER UNFREEZING ---------
    # Unfreeze router & layernorm parameters
    for param in trainable_params:
        param.requires_grad = True

    model.print_trainable_parameters()
    model.config.use_cache = False

    # Create trainer and start training
    trainer = PeftTrainer(
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

    peft_model = trainer.model.base_model
    peft_model.merge_and_unload()
    peft_model.save_pretrained(args.merged_model_path)
    tokenizer.save_pretrained(args.merged_model_path)