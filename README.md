## Installation ##
Please install the dependencies via conda, using the following command:

```bash
pip install -r requirements.txt
```

## Experiment ##
```task_name``` can be selected from ```[citation, movie_tagging, news_categorize, news_headline, product_rating, scholarly_title, tweet_paraphrase]```. Here, we take ```movie_tagging``` as an example.

### AntiOPPU
#### 1. Base LLM Task Adaption

```bash
CUDA_VISIBLE_DEVICES=0,1 python Antitask_LoRA.py --k 0 --task_name movie_tagging
```

#### 2. Train One PEFT Per User
```bash
CUDA_VISIBLE_DEVICES=0,1 python AntiOPPU.py --k 0 --task_name movie_tagging --task_lora 25
```

<!-- ### OPPU
#### 1. Base LLM Task Adaption

```bash
CUDA_VISIBLE_DEVICES=0,1 python task_LoRA.py --k 0 --task_name movie_tagging
```

#### 2. Train One PEFT Per User
```bash
CUDA_VISIBLE_DEVICES=0,1 python OPPU.py --k 0 --task_name movie_tagging --task_lora ./ckpt/movie_tagging/k0-movie_tagging-Llama-2-7b-hf-task_LoRA_ckpt
```

### OPPU + RAG

#### 1. Base LLM Task Adaption

```bash
CUDA_VISIBLE_DEVICES=0 python task_LoRA.py --k 1 --task_name movie_tagging
```

#### 2. Train One PEFT Per User
```bash
CUDA_VISIBLE_DEVICES=0 python OPPU.py --k 1 --task_name movie_tagging --task_lora ./ckpt/movie_tagging/k1-movie_tagging-Llama-2-7b-hf-task_LoRA_ckpt
```
----

### OPPU + PAG
#### 1. Base LLM Task Adaption

```bash
CUDA_VISIBLE_DEVICES=0 python task_LoRA.py --k 1 --task_name movie_tagging --add_profile
```

#### 2. Train One PEFT Per User
```bash
CUDA_VISIBLE_DEVICES=0 python OPPU.py --k 1 --task_name movie_tagging --task_lora ./ckpt/movie_tagging/k1-movie_tagging-Llama-2-7b-hf-profile-task_LoRA_ckpt --add_profile
```

## Evaluation ##
```TASK_ID``` is the corresponding ID selected from ```["LaMP_1", "LaMP_2N", "LaMP_2M", "LaMP_3", "LaMP_4", "LaMP_5", "LaMP_7"]```

```bash
python ./eval/eval_task.py \
    --golds_json {PATH_TO_LABEL_JSON_FILE} \
    --preds_json {PATH_TO_PREDICTION_JSON_FILE} \
    --task_name {TASK_ID} \
    --output_file {RESULT_JSON_PATH}
``` -->

## Citation ##
```bibtex
@article{tan2024democratizing,
  title={Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning},
  author={Tan, Zhaoxuan and Zeng, Qingkai and Tian, Yijun and Liu, Zheyuan and Yin, Bing and Jiang, Meng},
  journal={arXiv preprint arXiv:2402.04401},
  year={2024}
}
```
