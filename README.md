<div align="center">
  <h1>DIVE into MoE</h1>
  <p>Official implementation of the ACL 2025 paper <i>DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts</i>.</p>
  <p>
    <a href="#overview">Overview</a> · <a href="#repository-structure">Repository structure</a> · <a href="#setup">Setup</a> · <a href="#data-preparation">Data</a> · <a href="#quick-start">Quick start</a> · <a href="#configuration-notes">Configuration</a> · <a href="#evaluation">Evaluation</a>
  </p>
</div>

## Overview
- DIVE rebuilds a sparse SMoE model by merging multiple pruned dense checkpoints into a shared architecture while preserving diverse expert behaviors.
- The pipeline covers: establishing the SMoE checkpoint from pruned experts, router training, expert tuning with LoRA, and perplexity evaluation.
- This repo ships TinyLlama-based examples (8 experts, 50%/75% sparsity) with DeepSpeed-ready scripts that can be adapted to other backbones via `configurations/`.

## Repository structure
- `main.py` — entrypoint controlling establish / router training / expert training via `--instruction`.
- `configurations/` — SMoE configs (`tinyllama_smoe/*.json`) and DeepSpeed configs (`ds_config/*.json`).
- `scripts/` — runnable shell scripts for establishing models, training stages, and PPL evaluation.
- `lib/` — data processing (`traindata_process.py`), evaluation helpers (`eval.py`), and dataset loaders.
- `prune/` — utilities for creating the pruned dense checkpoints that seed the experts.
- `models/` — SMoE model definitions and loading utilities.

## Setup
1) Python 3.9+ and CUDA 11.8 are recommended.
2) Create an environment:
```
conda create -n dive python=3.9
conda activate dive
```
3) Install PyTorch with the CUDA build that matches your drivers (example for CUDA 11.8):
```
pip install --upgrade "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2" --index-url https://download.pytorch.org/whl/cu118
```
4) Install the remaining dependencies:
```
pip install -r requirements.txt
```
5) (Optional) Verify CUDA visibility:
```
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
PY
```

## Data preparation
- Training uses `SlimPajama-627B` (train/validation splits). Place or symlink the dataset under `./datasets/SlimPajama-627B` so the HuggingFace `datasets` loader can find it locally.
- When you launch training, `lib/traindata_process.traindata_loaders` will:
  - tokenize the dataset (cached to `./datasets/tokenized_slimpajama`),
  - pack sequences to `max_length` (cached to `./datasets/grouped_slimpajama_<max_length>`),
  - reuse caches on subsequent runs.
- To precompute the caches before training:
```
python - <<'PY'
from transformers import AutoTokenizer
from lib.traindata_process import traindata_loaders

tokenizer = AutoTokenizer.from_pretrained("./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5", use_fast=False)
traindata_loaders(tokenizer, max_length=1024, seed=42)
PY
```

## Quick start
The commands below assume you have pruned TinyLlama checkpoints prepared under `./pruned_models/tinyllama_0.5` (one subfolder per expert).

1) Establish the SMoE checkpoint
```
bash scripts/establish/establish_tinyllama_8_1_0.5.sh
```
- Uses `configurations/tinyllama_smoe/config_8_1_0.5.json`.
- Saves the assembled model and tokenizer to `./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5`.

2) Train the router (stage 0)
```
bash scripts/train/train_tinyllama_8_1_stage0.sh
```
- DeepSpeed ZeRO-2 stage0 config: `configurations/ds_config/zero2_stage0.json`.
- Checkpoints saved under `./checkpoints/tinyllama_8_1_stage0_results/`.

3) Fine-tune experts with LoRA (stage 1 and stage 2)
```
bash scripts/train/train_tinyllama_8_1_stage1.sh
bash scripts/train/train_tinyllama_8_1_stage2.sh
```
- Stage 1 starts from the router checkpoint; Stage 2 continues from the merged Stage 1 model.
- Outputs include PEFT adapters and merged full models in the respective `checkpoints/` subdirectories.

4) Evaluate perplexity
```
bash scripts/test/ppl_tinyllama_8_1_stage0.sh
# or evaluate the merged Stage 1 checkpoints:
bash scripts/test/ppl_tinyllama_8_1_stage1.sh
```
- The script tests multiple sequence lengths (512/1024/2048) on WikiText2 by default.

For the 8x2 experts / 75% sparsity variant, use the corresponding scripts:
```
bash scripts/establish/establish_tinyllama_8_2_0.75.sh
bash scripts/train/train_tinyllama_8_2_stage0.sh
bash scripts/train/train_tinyllama_8_2_stage1.sh
bash scripts/train/train_tinyllama_8_2_stage2.sh
bash scripts/test/ppl_tinyllama_8_2_stage0.sh
bash scripts/test/ppl_tinyllama_8_2_stage1.sh
```

## Configuration notes
- `main.py` arguments:
  - `--instruction`: `establish`, `train_router`, or `train_expert`.
  - `--folder_path`: directory containing pruned dense checkpoints (one per expert).
  - `--default_config_path`: SMoE config (for example, `configurations/tinyllama_smoe/config_8_1_0.5.json`) that sets expert count, top-k, and routing settings.
  - `--save_model_path` / `--tokenizer_path` / `--output_dir`: where models, tokenizers, and checkpoints are stored.
  - Training knobs include learning rate, scheduler, warmup, batch sizes, gradient accumulation, `--max_length`, and DeepSpeed config (`--deepspeed`).
- DeepSpeed configs live under `configurations/ds_config/` and can be swapped per hardware budget.
- To adjust architectures, edit or add JSON configs under `configurations/tinyllama_smoe/` and point `--default_config_path` to the new file.

## Evaluation
- For custom checkpoints, run `python test/ppl.py --base_dir <dir> --tokenizer_path <tokenizer> [--startswith ckpt-prefix] [--endswith suffix]`.
- Evaluations default to WikiText2; extend `lib/eval.py` or `lib/evaldata.py` to add datasets.

## Citation
BibTeX will be released with the camera-ready paper. For now, please cite the working title:
```
@inproceedings{dive2025,
  title={DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts},
  booktitle={ACL},
  year={2025}
}
```

## License
This project is released under the Apache 2.0 License (see `LICENSE`).
