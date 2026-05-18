# LaMP Personalize Pipeline

This repository contains OPPU-based personalization experiments for LaMP tasks, with an added CTTA (continual test-time adaptation) workflow for detecting preference drift and updating user LoRA adapters during streaming profile inference.

The repository keeps code, prompts, and experiment utilities under version control. Local datasets, checkpoints, model outputs, and result artifacts are intentionally ignored by Git.

## Repository Layout

```text
.
├── OPPU_original.py                     # Original OPPU implementation
├── OPPU_CTTA.py                         # Monolithic CTTA implementation
├── OPPU_CTTA_refactored.py              # Refactored CTTA entrypoint backed by ctta/
├── task_LoRA.py                         # Base task LoRA training
├── gen_profile.py                       # Profile generation utility
├── run_custom_drift_ctta_experiments.py # Batch runner for base vs CTTA drift experiments
├── run_lora_drift_ctta_experiments.py   # Convenience runner for LoRA drift detector experiments
├── run_ctta_evaluations.py              # Evaluation and drift-log summarization
├── analyze_lamp_preference_drift.py     # Preference-drift analysis and custom drift data generation
├── analyze_user_profile_query_alignment.py
├── plot_comparison_table.py
├── dataset_pipline.py                   # LaMP drift dataset generation helper
├── ctta/                                # Refactored CTTA package
│   ├── config.py                        # CLI arguments and naming helpers
│   ├── data.py                          # Data loading and profile split logic
│   ├── drift.py                         # Semantic / LoRA / hybrid drift detection
│   ├── engine.py                        # CTTA engine composition
│   ├── inference.py                     # Inference routines
│   ├── modeling.py                      # Base model and LoRA loading
│   ├── outputs.py                       # Prediction and drift log writers
│   ├── pipeline.py                      # Per-user streaming pipeline
│   └── training.py                      # Warmup and triggered adaptation
├── eval/                                # LaMP metric implementations
├── prompt/                              # Prompt templates
├── tool/data_p13n/                      # Data sampling/profile/embedding utilities
├── data/                                # Local datasets, ignored by Git
├── ckpt/                                # Local LoRA checkpoints, ignored by Git
├── output/                              # Prediction artifacts, ignored by Git
└── result/                              # Summaries and plots, ignored by Git
```

## Setup

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The scripts assume local Hugging Face model paths by default, for example:

```text
/home/xuyifan/model/meta-llama/llama2_7b_hf
```

Pass `--model_name` and, when using semantic drift detection, `--semantic_model_path` / `--semantic_device` to match your machine.

## Supported Tasks

Use `--task_name` with one of:

```text
citation
movie_tagging
news_categorize
news_headline
product_rating
scholarly_title
tweet_paraphrase
```

The corresponding LaMP IDs are `LaMP_1`, `LaMP_2M`, `LaMP_2N`, `LaMP_4`, `LaMP_3`, `LaMP_5`, and `LaMP_7`.

## Data

`data/` is not committed. Place local task data under:

```text
data/{task_name}/
```

Common files used by the scripts include:

```text
data/{task_name}/all_user.json
data/{task_name}/all_user_golds.json
data/{task_name}/user_top_100_history.json
data/{task_name}/user_top_100_history_label.json
data/{task_name}/custom_drifts/drift_0.1-0.2.json
data/{task_name}/custom_drifts/drift_0.2-0.3.json
...
```

`OPPU_CTTA.py` and `OPPU_CTTA_refactored.py` default to `./data/{task_name}/user_top_100_history.json` unless `--train_data_path` is provided.

## Basic Workflow

### 1. Train a Base Task LoRA

```bash
CUDA_VISIBLE_DEVICES=0 python task_LoRA.py \
  --task_name movie_tagging \
  --model_name /path/to/llama2_7b_hf \
  --k 0 \
  --max_epoch 3
```

With profile prompts:

```bash
CUDA_VISIBLE_DEVICES=0 python task_LoRA.py \
  --task_name movie_tagging \
  --model_name /path/to/llama2_7b_hf \
  --k 0 \
  --add_profile
```

### 2. Run CTTA for One Dataset

Prefer the refactored entrypoint for new experiments:

```bash
CUDA_VISIBLE_DEVICES=0 python OPPU_CTTA_refactored.py \
  --task_name movie_tagging \
  --model_name /path/to/llama2_7b_hf \
  --task_lora ./ckpt/movie_tagging/k0-movie_tagging-llama2_7b_hf-task_LoRA_ckpt \
  --train_data_path ./data/movie_tagging/custom_drifts/drift_0.1-0.2.json \
  --drift_tag drift_0.1-0.2 \
  --adaptation_mode ctta \
  --output_tag ctta \
  --ctta_threshold 0.2 \
  --ctta_window_size 8
```

For a fixed-adapter baseline on the same stream:

```bash
CUDA_VISIBLE_DEVICES=1 python OPPU_CTTA_refactored.py \
  --task_name movie_tagging \
  --model_name /path/to/llama2_7b_hf \
  --task_lora ./ckpt/movie_tagging/k0-movie_tagging-llama2_7b_hf-task_LoRA_ckpt \
  --train_data_path ./data/movie_tagging/custom_drifts/drift_0.1-0.2.json \
  --drift_tag drift_0.1-0.2 \
  --adaptation_mode base \
  --output_tag base
```

Outputs are written to:

```text
output/{task_name}/{drift_tag}/
```

Typical artifacts:

```text
*-heldout_profile.json
*-heldout_profile-summary.json
*-query.json
*-drift_log.json
```

## Drift Detectors

CTTA supports three detector modes:

```text
semantic  # semantic window comparison over profile history
lora      # parameter-space LoRA probe movement
hybrid    # trigger when either detector fires
```

Example with LoRA drift detection:

```bash
CUDA_VISIBLE_DEVICES=0 python OPPU_CTTA_refactored.py \
  --task_name movie_tagging \
  --model_name /path/to/llama2_7b_hf \
  --train_data_path ./data/movie_tagging/custom_drifts/drift_0.3-0.4.json \
  --drift_tag drift_0.3-0.4 \
  --drift_detector lora \
  --lora_drift_threshold 0.05
```

## Batch Experiments

Run base and CTTA over multiple custom drift bins:

```bash
python run_custom_drift_ctta_experiments.py \
  --ctta_script OPPU_CTTA_refactored.py \
  --task_name movie_tagging \
  --dataset_dir ./data/movie_tagging/custom_drifts \
  --dataset_prefix drift \
  --drift_ranges 0.1-0.2 0.2-0.3 0.3-0.4 0.4-0.5 0.5-0.6 \
  --model_name /path/to/llama2_7b_hf \
  --task_lora ./ckpt/movie_tagging/k0-movie_tagging-llama2_7b_hf-task_LoRA_ckpt \
  --ctta_cuda_visible_devices 0 \
  --base_cuda_visible_devices 1 \
  --skip_existing
```

For LoRA-detector-only CTTA runs, use the convenience wrapper:

```bash
python run_lora_drift_ctta_experiments.py \
  --task_name movie_tagging \
  --dataset_dir ./data/movie_tagging/custom_drifts \
  --model_name /path/to/llama2_7b_hf \
  --task_lora ./ckpt/movie_tagging/k0-movie_tagging-llama2_7b_hf-task_LoRA_ckpt \
  --lora_drift_threshold 0.05 \
  --skip_existing
```

Batch summaries and plots are written under:

```text
result/custom_drift_ctta/{task_name}/{profile_split_ratio}/
```

## Evaluation

Evaluate an individual CTTA run:

```bash
python run_ctta_evaluations.py \
  --task_name movie_tagging \
  --train_data_path ./data/movie_tagging/custom_drifts/drift_0.1-0.2.json \
  --drift_tag drift_0.1-0.2 \
  --model_name llama2_7b_hf \
  --output_tag ctta \
  --heldout_preds_json ./output/movie_tagging/drift_0.1-0.2/output-OPPU-k0-movie_tagging-llama2_7b_hf-ctta-heldout_profile.json \
  --query_preds_json ./output/movie_tagging/drift_0.1-0.2/output-OPPU-k0-movie_tagging-llama2_7b_hf-ctta-query.json \
  --drift_log_json ./output/movie_tagging/drift_0.1-0.2/output-OPPU-k0-movie_tagging-llama2_7b_hf-ctta-drift_log.json
```

The evaluation summary includes heldout-profile metrics, final-query metrics, and drift statistics such as trigger counts, update counts, average drift score, and LoRA / semantic trigger breakdowns.

## Analysis Utilities

Generate or inspect custom preference drift splits:

```bash
python analyze_lamp_preference_drift.py \
  --task_name movie_tagging \
  --input_path ./data/movie_tagging/all_user.json \
  --write_custom_drifts \
  --custom_drift_dir ./data/movie_tagging/custom_drifts
```

Check profile/query alignment:

```bash
python analyze_user_profile_query_alignment.py \
  --dataset-dir ./data/movie_tagging \
  --top-n 5
```

Create comparison tables or plots from saved summaries:

```bash
python plot_comparison_table.py --csv_path ./result/custom_drift_ctta/movie_tagging/0.5/summary.csv
```

## Version-Control Notes

The following paths are ignored and should stay local:

```text
data/
output/
ckpt/
result/
__pycache__/
```

Commit source code, prompts, scripts, and documentation only. Large datasets, model checkpoints, and generated experiment artifacts should be stored outside Git or uploaded to a separate artifact store.

## Citation

```bibtex
@article{tan2024democratizing,
  title={Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning},
  author={Tan, Zhaoxuan and Zeng, Qingkai and Tian, Yijun and Liu, Zheyuan and Yin, Bing and Jiang, Meng},
  journal={arXiv preprint arXiv:2402.04401},
  year={2024}
}
```
