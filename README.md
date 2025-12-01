这有什么问题吗：<div align="center">
  <h1>DIVE into MoE</h1>
  <div>
    <a href="#overview">📝 Overview</a> | <a href="#installation">⚙️ Installation Guide</a> | <a href="#quick-start">🚀 Quick Start</a> | <a href="#method">🚅 Method Details</a> | <a href="#evaluation">💎 Evaluation</a> | <a href="#citation">📝 Citation</a>
  </div>
</div>
<h2 id="todo">📦 To be released</h2>

- [ ] method details
- [x] script usage

<h2 id="overview">📝 Overview</h2>

This repository contains the official implementation of our ACL 2025 paper "DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts".

<h2 id="installation">⚙️ Installation</h2>

Step 1: Create a new conda environment:
```
conda create -n dive python=3.9
conda activate dive
```
Step 2: Install relevant packages:
```
conda install pytorch==2.7.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

<h2 id="quick-start">🚀 Quick Start</h2>

The commands below assume you have pruned TinyLlama checkpoints prepared under `./pruned_models/tinyllama_0.5` (one subfolder per expert). Take DIVE 1/8 as an example.

**Step 1: Domain Affinity Mining**

<details>
  <summary>clustering results are as here.</summary>
</details>
**Step 2: Reconstruct the DIVE model**

``` python
bash scripts/establish/establish_tinyllama_8_1_0.5.sh
```

\- Use `configurations/tinyllama_smoe/config_8_1_0.5.json`.

\- Save the assembled model and tokenizer to `./pruned_smoe_models/tinyllama_smoe/tinyllama_8_1_0.5`.

**Step 3: Train the router**

\- Training uses `SlimPajama-627B` (train/validation splits). Place or symlink the dataset under `./datasets/SlimPajama-627B` so the HuggingFace `datasets` loader can find it locally.

\- When you launch training, `lib/traindata_process.traindata_loaders` will:

​	\- tokenize the dataset (cached to `./datasets/tokenized_slimpajama`),

​	\- pack sequences to `max_length` (cached to `./datasets/grouped_slimpajama_<max_length>`),

​	\- reuse caches on subsequent runs.


``` python
bash scripts/train/train_tinyllama_8_1_stage0.sh
```

\- DeepSpeed ZeRO-2 stage0 config: `configurations/ds_config/zero2_stage0.json`.

\- Checkpoints saved under `./checkpoints/tinyllama_8_1_stage0_results/`.

**Step 4: Train the experts with LoRA**

```
bash scripts/train/train_tinyllama_8_1_stage1.sh
bash scripts/train/train_tinyllama_8_1_stage2.sh
```

\- Stage 1 starts from the router checkpoint; Stage 2 continues from the merged Stage 1 model.

\- Outputs include PEFT adapters and merged full models in the respective `checkpoints/` subdirectories.

**Step 5: Evaluate perplexity**

```
bash scripts/test/ppl_tinyllama_8_1_stage0.sh
bash scripts/test/ppl_tinyllama_8_1_stage1.sh
```

\- The script tests multiple sequence lengths (512 / 1024 / 2048) on WikiText2 by default.

<details>
  <summary>For DIVE 2/8, use the corresponding scripts:</summary>

  ```Python
  bash scripts/establish/establish_tinyllama_8_2_0.75.sh
  ```

  ```Python
  bash scripts/train/train_tinyllama_8_2_stage0.sh
  bash scripts/train/train_tinyllama_8_2_stage1.sh
  bash scripts/train/train_tinyllama_8_2_stage2.sh
  ```

```
bash scripts/test/ppl_tinyllama_8_2_stage0.sh
bash scripts/test/ppl_tinyllama_8_2_stage1.sh
```

</details>

<h2 id="method">🚅 Method Details</h2>

<h2 id="evaluation">💎 Evaluation</h2>

<h2 id="citation">📝 Citation</h2>

```
@article{feng2025dive,
  title={DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts},
  author={Feng, Yuchen and Shen, Bowen and Gu, Naibin and Zhao, Jiaxuan and Fu, Peng and Lin, Zheng and Wang, Weiping},
  journal={arXiv preprint arXiv:2506.09351},
  year={2025}
}
```