import argparse
import json
import os
import re
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import (
    extract_citation_title,
    extract_movie,
    extract_news_cat,
    extract_news_headline,
    extract_option,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
    get_first_k_tokens,
    name2taskid,
    print_trainable_parameters,
    split_batch,
)


parser = argparse.ArgumentParser(description="OPPU with CTTA-triggered user LoRA adaptation")
parser.add_argument("--model_name", type=str, default="/home/xuyifan/model/meta-llama/llama2_7b_hf")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--infer_batch_size", type=int, default=None, help="Batch size for non-streaming query inference; defaults to --batch_size")
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--cut_off", type=int, default=512)
parser.add_argument("--max_epoch", type=int, default=2)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--task_name", type=str, default="movie_tagging")
parser.add_argument("--add_profile", action="store_true")
parser.add_argument("--task_lora", type=str, default="./ckpt/movie_tagging/k0-movie_tagging-llama2_7b_hf-profile-task_LoRA_ckpt")
parser.add_argument("--access_token", type=str, default=None)
parser.add_argument("--device_map", type=str, default="auto", help="HF device_map for multi-GPU loading, e.g. auto/balanced/balanced_low_0")
parser.add_argument("--load_in_8bit", action="store_true", help="Backward-compatible alias for quantized loading; uses 4-bit QLoRA internally")
parser.add_argument("--load_in_4bit", action="store_true", help="Load the base model with bitsandbytes 4-bit quantization")
parser.add_argument("--llm_int8_threshold", type=float, default=6.0, help="Legacy argument kept for CLI compatibility; ignored for 4-bit loading")
parser.add_argument("--train_data_path", type=str, default=None, help="Default: ./data/{task_name}/user_top_100_history.json")
parser.add_argument("--drift_tag", type=str, default=None, help="Dataset tag used in output naming. Defaults to the stem of --train_data_path.")
parser.add_argument("--output_tag", type=str, default="ctta")
parser.add_argument("--profile_split_ratio", type=float, default=0.5, help="Visible prefix ratio used for initial training")
parser.add_argument(
    "--profile_split_mode",
    type=str,
    default="metadata",
    choices=["metadata", "ratio", "count"],
    help="Use dataset profile_split_point metadata, ratio, or fixed count to split profile",
)
parser.add_argument("--profile_split_count", type=int, default=None, help="Visible prefix count when profile_split_mode=count")
parser.add_argument("--memory_size", type=int, default=32, help="Only keep the most recent N history items as visible memory; <=0 means keep all")
parser.add_argument(
    "--adaptation_mode",
    type=str,
    default="ctta",
    choices=["ctta", "base"],
    help="ctta updates the user LoRA after drift triggers; base keeps the warmup user LoRA fixed during streaming",
)
parser.add_argument(
    "--semantic_model_path",
    type=str,
    default="/home/xuyifan/model/Huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    help="Local sentence-transformer checkpoint used for semantic drift detection",
)
parser.add_argument(
    "--semantic_device",
    type=str,
    default="cuda:1",
    help="Device used by the MiniLM semantic encoder, e.g. cpu/cuda:0/cuda:1",
)
parser.add_argument("--drift_tag_weight", type=float, default=3.0, help="Weight for tag/category style semantic fields")
parser.add_argument("--drift_text_weight", type=float, default=1.0, help="Weight for description/text style semantic fields")
parser.add_argument("--verbose_predictions", action="store_true", help="Print every profile/query prediction during inference")

# CTTA parameters
parser.add_argument("--ctta_threshold", type=float, default=0.2, help="Trigger LoRA update when drift score >= threshold")
parser.add_argument("--ctta_window_size", type=int, default=8, help="Sliding window size for preference drift detection")
parser.add_argument("--ctta_init_size", type=int, default=12, help="Warmup history size for initial user LoRA")
parser.add_argument("--ctta_update_min_examples", type=int, default=4, help="Minimum newly arrived examples before another adaptation")
parser.add_argument("--ctta_max_update_size", type=int, default=16, help="Use at most the latest N examples for one triggered adaptation")
parser.add_argument(
    "--ctta_anti_forgetting",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use replay, LoRA anchoring, and LwF distillation during triggered CTTA updates",
)
parser.add_argument("--ctta_replay_size", type=int, default=8, help="Number of older visible examples replayed during each CTTA update")
parser.add_argument(
    "--ctta_replay_strategy",
    type=str,
    default="semantic_diverse",
    choices=["semantic_diverse", "recent", "none"],
    help="How to choose replay examples from older visible history",
)
parser.add_argument("--ctta_anchor_lambda", type=float, default=0.05, help="LoRA parameter anchoring strength for CTTA updates")
parser.add_argument("--ctta_lwf_lambda", type=float, default=0.2, help="LwF distillation strength for CTTA updates")
parser.add_argument("--ctta_distill_temperature", type=float, default=2.0, help="Temperature for CTTA LwF distillation")
parser.add_argument("--save_user_ckpt", action="store_true")

args = parser.parse_args()
quantized_loading = args.load_in_8bit or args.load_in_4bit

model_name = args.model_name
task_name = args.task_name
batch_size = args.batch_size
infer_batch_size = args.infer_batch_size or args.batch_size
k = args.k
cutoff_len = args.cut_off
max_epoch = args.max_epoch
add_eos_token = False


TASK_TO_EXTRACTOR = {
    "movie_tagging": extract_movie,
    "news_categorize": extract_news_cat,
    "news_headline": extract_news_headline,
    "product_rating": extract_product_review,
    "scholarly_title": extract_scholarly_title,
    "tweet_paraphrase": extract_tweet_paraphrasing,
}


TASK_DRIFT_CONFIG = {
    "movie_tagging": {
        "semantic_fields": [
            {"field": "tag", "prefix": "tag", "weight_arg": "drift_tag_weight"},
            {"field": "description", "prefix": "description", "weight_arg": "drift_text_weight"},
        ]
    },
    "citation": {
        "semantic_fields": [
            {"field": "citation", "prefix": "citation", "weight_arg": "drift_tag_weight"},
            {"field": "title", "prefix": "title", "weight_arg": "drift_text_weight"},
        ]
    },
    "news_categorize": {
        "semantic_fields": [
            {"field": "category", "prefix": "category", "weight_arg": "drift_tag_weight"},
            {"field": "text", "prefix": "text", "weight_arg": "drift_text_weight"},
        ]
    },
    "news_headline": {
        "semantic_fields": [
            {"field": "title", "prefix": "title", "weight_arg": "drift_tag_weight"},
            {"field": "text", "prefix": "text", "weight_arg": "drift_text_weight"},
        ]
    },
    "product_rating": {
        "semantic_fields": [
            {"field": "score", "prefix": "score", "weight_arg": "drift_tag_weight"},
            {"field": "text", "prefix": "review", "weight_arg": "drift_text_weight"},
        ]
    },
    "scholarly_title": {
        "semantic_fields": [
            {"field": "title", "prefix": "title", "weight_arg": "drift_tag_weight"},
            {"field": "abstract", "prefix": "abstract", "weight_arg": "drift_text_weight"},
        ]
    },
    "tweet_paraphrase": {
        "semantic_fields": [
            {"field": "text", "prefix": "text", "weight_arg": "drift_text_weight"},
        ]
    },
}


TASK_LABEL_FIELD = {
    "movie_tagging": "tag",
    "citation": "citation",
    "news_categorize": "category",
    "news_headline": "title",
    "product_rating": "score",
    "scholarly_title": "title",
    "tweet_paraphrase": "text",
}


DISCRETE_LABELS = {
    "movie_tagging": [
        "sci-fi",
        "based on a book",
        "comedy",
        "action",
        "twist ending",
        "dystopia",
        "dark comedy",
        "classic",
        "psychology",
        "fantasy",
        "romance",
        "thought-provoking",
        "social commentary",
        "violence",
        "true story",
    ],
    "news_categorize": [
        "travel",
        "education",
        "parents",
        "style & beauty",
        "entertainment",
        "food & drink",
        "science & technology",
        "business",
        "sports",
        "healthy living",
        "women",
        "politics",
        "crime",
        "culture & arts",
        "religion",
    ],
    "product_rating": ["1", "2", "3", "4", "5"],
}


def get_train_data_path():
    if args.train_data_path:
        return args.train_data_path
    return f"./data/{task_name}/user_top_100_history.json"


def infer_drift_tag(train_data_path):
    stem = os.path.splitext(os.path.basename(train_data_path))[0]
    return stem[len("drift_train_"):] if stem.startswith("drift_train_") else stem


def resolve_drift_tag(train_data_path):
    if args.drift_tag:
        return args.drift_tag
    return infer_drift_tag(train_data_path)


def build_ctta_run_name():
    output_tag = args.output_tag
    if args.adaptation_mode == "base" and output_tag == "ctta":
        output_tag = "base"
    run_name = f"output-OPPU-k{k}-{task_name}-{model_name.split('/')[-1]}-{output_tag}"
    if args.add_profile:
        run_name += "-profile"
    return run_name


def build_dataset_output_dirname(tag):
    normalized_tag = str(tag).strip()
    if normalized_tag.startswith("drift_"):
        return normalized_tag
    return f"drift_{normalized_tag}"


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_memory_slice(entries):
    if args.memory_size is None or args.memory_size <= 0:
        return entries
    return entries[-args.memory_size:]


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point["full_prompt"]
    tokenized_full_prompt = tokenize(full_prompt)

    user_prompt = data_point["prompt"]
    tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt


def get_bm25_class():
    from rank_bm25 import BM25Okapi

    return BM25Okapi


def build_train_data_for_entries(profile_entries, profile_prefix=None, start_idx=0, end_idx=None):
    train_data = []
    if end_idx is None:
        end_idx = len(profile_entries)

    for idx in range(start_idx, end_idx):
        q = {key: get_first_k_tokens(value, 768) for key, value in profile_entries[idx].items()}

        prompt = prompt_template[args.task_name]["OPPU_input"].format(**q)
        full_prompt = prompt_template[args.task_name]["OPPU_full"].format(**q)

        if k > 0 and idx != 0 and format_flag:
            visible_history_list = []
            memory_history = get_memory_slice(profile_entries[:idx])
            for history_item in memory_history:
                truncated_history = {
                    key: get_first_k_tokens(value, 768)
                    for key, value in history_item.items()
                }
                visible_history_list.append(truncated_history)

            history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in visible_history_list]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = get_bm25_class()(tokenized_corpus)

            tokenized_query = prompt_template[args.task_name]["retrieval_query"].format(**q).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
            history_string = "".join(retrieved_history)
            prompt = history_string + "\n" + prompt
            full_prompt = history_string + "\n" + full_prompt

        if args.add_profile and format_flag and profile_prefix:
            prompt = profile_prefix + "\n" + prompt
            full_prompt = profile_prefix + "\n" + full_prompt

        train_data.append({"prompt": prompt, "full_prompt": full_prompt})

    return train_data


def build_train_data_for_indices(profile_entries, indices, profile_prefix=None):
    train_data = []
    for idx in indices:
        train_data.extend(
            build_train_data_for_entries(
                profile_entries,
                profile_prefix=profile_prefix,
                start_idx=idx,
                end_idx=idx + 1,
            )
        )
    return train_data


def build_train_data_for_range(user_data, profile_prefix=None, start_idx=0, end_idx=None):
    return build_train_data_for_entries(
        user_data["profile"],
        profile_prefix=profile_prefix,
        start_idx=start_idx,
        end_idx=end_idx,
    )


def build_tokenized_train_dataset(train_data, shuffle=False):
    train_dataset = Dataset.from_list(train_data)
    tokenized_dataset = train_dataset.map(
        generate_and_tokenize_prompt,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
    )
    if shuffle:
        tokenized_dataset = tokenized_dataset.shuffle(seed=42)
    return tokenized_dataset


def cast_norm_modules_to_float32(model):
    for name, module in model.named_modules():
        if "norm" in name:
            module.to(torch.float32)


def snapshot_trainable_params(model):
    return {
        name: param.detach().clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


class CTTAContinualTrainer(transformers.Trainer):
    def __init__(
        self,
        old_params=None,
        anchor_lambda=0.0,
        lwf_lambda=0.0,
        distill_temperature=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.old_params = old_params
        self.anchor_lambda = anchor_lambda
        self.lwf_lambda = lwf_lambda
        self.distill_temperature = distill_temperature
        self._old_param_device_cache = {}

    def _old_param_on_device(self, name, param):
        cache_key = (name, str(param.device), param.dtype)
        old_param = self._old_param_device_cache.get(cache_key)
        if old_param is None or old_param.shape != param.shape:
            old_param = self.old_params[name].to(device=param.device, dtype=param.dtype, non_blocking=True)
            self._old_param_device_cache[cache_key] = old_param
        return old_param

    def _compute_teacher_logits(self, model, inputs):
        cached_params = {}
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.old_params:
                        cached_params[name] = param.detach().clone()
                        param.copy_(self._old_param_on_device(name, param))
                return model(**inputs).logits.detach()
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in cached_params:
                        param.copy_(cached_params[name])
            if was_training:
                model.train()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.old_params and self.anchor_lambda > 0:
            anchor_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
            anchor_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.old_params:
                    old_param = self._old_param_on_device(name, param)
                    anchor_term = torch.sum((param - old_param) ** 2).to(device=loss.device, dtype=loss.dtype)
                    anchor_loss = anchor_loss + anchor_term
                    anchor_count += param.numel()
            if anchor_count > 0:
                loss = loss + self.anchor_lambda * (anchor_loss / anchor_count)

        if self.old_params and self.lwf_lambda > 0:
            teacher_logits = self._compute_teacher_logits(model, inputs)
            temperature = self.distill_temperature
            student_log_probs = F.log_softmax(outputs.logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

            labels = inputs.get("labels")
            if labels is not None:
                mask = labels.ne(-100)
            else:
                mask = inputs.get("attention_mask")

            if mask is not None and mask.sum() > 0:
                distill_loss = (token_kl * mask.to(token_kl.dtype)).sum() / mask.sum()
            else:
                distill_loss = token_kl.mean()
            loss = loss + self.lwf_lambda * distill_loss * (temperature ** 2)

        return (loss, outputs) if return_outputs else loss


def normalize_text(text):
    text = str(text).strip().lower()
    return " ".join(text.split())


semantic_encoder = None
semantic_embedding_cache = {}
semantic_entry_embedding_cache = {}


def resolve_semantic_device():
    requested_device = str(args.semantic_device).strip()
    if not requested_device:
        return "cpu"

    if requested_device == "cpu":
        return requested_device

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[semantic_encoder] CUDA unavailable, fallback to cpu instead of {requested_device}")
            return "cpu"

        if ":" in requested_device:
            try:
                device_index = int(requested_device.split(":", 1)[1])
            except ValueError:
                raise ValueError(f"Invalid --semantic_device value: {requested_device}")
            if device_index >= torch.cuda.device_count():
                print(
                    f"[semantic_encoder] Requested {requested_device} but only {torch.cuda.device_count()} CUDA devices are visible; fallback to cpu"
                )
                return "cpu"

    return requested_device


def get_semantic_encoder():
    global semantic_encoder
    if semantic_encoder is None:
        from sentence_transformers import SentenceTransformer

        semantic_encoder = SentenceTransformer(
            args.semantic_model_path,
            device=resolve_semantic_device(),
        )
        semantic_encoder.eval()
    return semantic_encoder


def l2_normalize(vector):
    norm = float(torch.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def get_text_embedding(text):
    normalized_text = normalize_text(text)
    if not normalized_text:
        return None

    if normalized_text in semantic_embedding_cache:
        return semantic_embedding_cache[normalized_text]

    encoder = get_semantic_encoder()
    embedding = encoder.encode(
        normalized_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embedding = embedding.detach().cpu().to(torch.float32)
    semantic_embedding_cache[normalized_text] = embedding
    return embedding


def build_entry_embedding_cache_key(entry):
    config = TASK_DRIFT_CONFIG[task_name]
    key_parts = [task_name]
    for field_config in config["semantic_fields"]:
        normalized_value = normalize_text(entry.get(field_config["field"], ""))
        weight = float(getattr(args, field_config["weight_arg"]))
        key_parts.append(
            (
                field_config["field"],
                field_config["prefix"],
                weight,
                normalized_value,
            )
        )
    return tuple(key_parts)


def get_entry_semantic_embedding(entry):
    cache_key = build_entry_embedding_cache_key(entry)
    if cache_key in semantic_entry_embedding_cache:
        return semantic_entry_embedding_cache[cache_key]

    config = TASK_DRIFT_CONFIG[task_name]
    weighted_embeddings = []
    total_weight = 0.0

    for field_config in config["semantic_fields"]:
        value = entry.get(field_config["field"])
        if value is None:
            continue

        normalized_value = normalize_text(value)
        if not normalized_value:
            continue

        embedding = get_text_embedding(f"{field_config['prefix']}: {normalized_value}")
        if embedding is None:
            continue

        weight = float(getattr(args, field_config["weight_arg"]))
        if weight <= 0:
            continue

        weighted_embeddings.append(embedding * weight)
        total_weight += weight

    if not weighted_embeddings or total_weight <= 0:
        semantic_entry_embedding_cache[cache_key] = None
        return None

    merged_embedding = torch.stack(weighted_embeddings, dim=0).sum(dim=0) / total_weight
    merged_embedding = l2_normalize(merged_embedding)
    semantic_entry_embedding_cache[cache_key] = merged_embedding
    return merged_embedding


def build_window_embedding(entries):
    entry_embeddings = []
    for entry in entries:
        embedding = get_entry_semantic_embedding(entry)
        if embedding is not None:
            entry_embeddings.append(embedding)

    if not entry_embeddings:
        return None

    window_embedding = torch.stack(entry_embeddings, dim=0).mean(dim=0)
    return l2_normalize(window_embedding)


def build_window_embedding_from_entry_embeddings(entry_embeddings):
    valid_embeddings = [embedding for embedding in entry_embeddings if embedding is not None]
    if not valid_embeddings:
        return None

    window_embedding = torch.stack(valid_embeddings, dim=0).mean(dim=0)
    return l2_normalize(window_embedding)


def embedding_cosine_distance(embedding_a, embedding_b):
    if embedding_a is None or embedding_b is None:
        return 0.0

    cosine_sim = float(torch.dot(embedding_a, embedding_b))
    cosine_sim = max(min(cosine_sim, 1.0), -1.0)
    return 1.0 - cosine_sim


def compute_preference_drift(profile_entries, current_end, window_size):
    if current_end < 2 * window_size:
        return None

    prev_window = profile_entries[current_end - 2 * window_size: current_end - window_size]
    curr_window = profile_entries[current_end - window_size: current_end]
    prev_embedding = build_window_embedding(prev_window)
    curr_embedding = build_window_embedding(curr_window)
    return embedding_cosine_distance(prev_embedding, curr_embedding)


def compute_preference_drift_from_embeddings(entry_embeddings, current_end, window_size):
    if current_end < 2 * window_size:
        return None

    prev_embeddings = entry_embeddings[current_end - 2 * window_size: current_end - window_size]
    curr_embeddings = entry_embeddings[current_end - window_size: current_end]
    prev_embedding = build_window_embedding_from_entry_embeddings(prev_embeddings)
    curr_embedding = build_window_embedding_from_entry_embeddings(curr_embeddings)
    return embedding_cosine_distance(prev_embedding, curr_embedding)


def select_replay_indices(profile_entries, start_idx):
    if (
        not args.ctta_anti_forgetting
        or args.ctta_replay_strategy == "none"
        or args.ctta_replay_size <= 0
        or start_idx <= 0
    ):
        return []

    pool_end = min(start_idx, len(profile_entries))
    replay_size = min(args.ctta_replay_size, pool_end)
    if replay_size <= 0:
        return []

    if args.ctta_replay_strategy == "recent":
        return list(range(pool_end - replay_size, pool_end))

    indexed_embeddings = []
    for idx in range(pool_end):
        embedding = get_entry_semantic_embedding(profile_entries[idx])
        if embedding is not None:
            indexed_embeddings.append((idx, embedding))

    if len(indexed_embeddings) <= replay_size:
        return [idx for idx, _ in indexed_embeddings] or list(range(pool_end - replay_size, pool_end))

    embeddings = torch.stack([embedding for _, embedding in indexed_embeddings], dim=0)
    centroid = l2_normalize(embeddings.mean(dim=0))
    first_pos = int(torch.argmax(torch.matmul(embeddings, centroid)).item())
    selected_positions = [first_pos]
    remaining_positions = set(range(len(indexed_embeddings)))
    remaining_positions.remove(first_pos)

    while remaining_positions and len(selected_positions) < replay_size:
        selected_embeddings = embeddings[selected_positions]
        best_pos = None
        best_distance = None
        for pos in remaining_positions:
            similarities = torch.matmul(selected_embeddings, embeddings[pos])
            min_distance = float((1.0 - similarities).min())
            if best_distance is None or min_distance > best_distance:
                best_distance = min_distance
                best_pos = pos
        selected_positions.append(best_pos)
        remaining_positions.remove(best_pos)

    return sorted(indexed_embeddings[pos][0] for pos in selected_positions)


def resolve_visible_prefix_length(profile_len, user_data=None):
    if profile_len <= 1:
        return profile_len

    if args.profile_split_mode == "metadata":
        visible_len = None
        if user_data is not None and user_data.get("profile_split_point") is not None:
            visible_len = int(user_data["profile_split_point"])
        if visible_len is None:
            visible_len = int(profile_len * args.profile_split_ratio)
    elif args.profile_split_mode == "count":
        if args.profile_split_count is None:
            raise ValueError("--profile_split_count must be set when --profile_split_mode=count")
        visible_len = args.profile_split_count
    else:
        visible_len = int(profile_len * args.profile_split_ratio)

    visible_len = max(1, visible_len)
    visible_len = min(profile_len - 1, visible_len)
    return visible_len


def train_on_segment(model, user_data, profile_prefix, start_idx, end_idx, training_args):
    if end_idx <= start_idx:
        return False

    train_data = build_train_data_for_range(
        user_data,
        profile_prefix=profile_prefix,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    if not train_data:
        return False

    train_dataset = build_tokenized_train_dataset(train_data)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    cast_norm_modules_to_float32(trainer.model)

    model.config.use_cache = False
    trainer.train()
    del trainer
    del train_dataset
    cleanup_memory()
    return True


def train_on_visible_entries(
    model,
    visible_entries,
    profile_prefix,
    start_idx,
    end_idx,
    training_args,
    return_metadata=False,
):
    if end_idx <= start_idx:
        return (False, {}) if return_metadata else False

    train_data = build_train_data_for_entries(
        visible_entries,
        profile_prefix=profile_prefix,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    replay_indices = select_replay_indices(visible_entries, start_idx)
    replay_data = build_train_data_for_indices(
        visible_entries,
        replay_indices,
        profile_prefix=profile_prefix,
    )
    train_data = replay_data + train_data
    if not train_data:
        return (False, {}) if return_metadata else False

    train_dataset = build_tokenized_train_dataset(train_data, shuffle=True)
    old_params = None
    if args.ctta_anti_forgetting and start_idx > 0:
        old_params = snapshot_trainable_params(model)

    trainer = CTTAContinualTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        old_params=old_params,
        anchor_lambda=args.ctta_anchor_lambda if old_params else 0.0,
        lwf_lambda=args.ctta_lwf_lambda if old_params else 0.0,
        distill_temperature=args.ctta_distill_temperature,
    )

    cast_norm_modules_to_float32(trainer.model)

    model.config.use_cache = False
    trainer.train()
    del trainer
    del train_dataset
    cleanup_memory()
    metadata = {
        "anti_forgetting": bool(old_params),
        "replay_size": len(replay_indices),
        "replay_indices": replay_indices,
        "anchor_lambda": args.ctta_anchor_lambda if old_params else 0.0,
        "lwf_lambda": args.ctta_lwf_lambda if old_params else 0.0,
        "distill_temperature": args.ctta_distill_temperature,
    }
    return (True, metadata) if return_metadata else True


def extract_article_from_profile_entry(entry):
    if task_name == "citation":
        return entry.get("title")
    for key in ("description", "text", "abstract"):
        value = entry.get(key)
        if value is not None:
            return value
    return None


def build_profile_eval_prompt(entry, revealed_entries, profile_prefix=None):
    q = {key: get_first_k_tokens(value, 768) for key, value in entry.items()}
    prompt = prompt_template[args.task_name]["OPPU_input"].format(**q)

    if k > 0 and revealed_entries and format_flag:
        visible_history_list = []
        memory_history = get_memory_slice(revealed_entries)
        for history_item in memory_history:
            truncated_history = {
                key: get_first_k_tokens(value, 368)
                for key, value in history_item.items()
            }
            visible_history_list.append(truncated_history)

        history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in visible_history_list]
        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = get_bm25_class()(tokenized_corpus)

        tokenized_query = prompt_template[args.task_name]["retrieval_query"].format(**q).split(" ")
        retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=min(args.k, len(history_list)))
        prompt = "".join(retrieved_history) + "\n" + prompt

    if args.add_profile and format_flag and profile_prefix:
        prompt = profile_prefix + "\n" + prompt

    return prompt


def build_history_list(visible_entries, token_limit):
    history_entries = []
    for item in get_memory_slice(visible_entries):
        history_entries.append(
            {key: get_first_k_tokens(value, token_limit) for key, value in item.items()}
        )
    return [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in history_entries]


def get_model_device_type(model):
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return "cpu"
    return getattr(device, "type", str(device).split(":", 1)[0])


def inference_autocast_context(model):
    if get_model_device_type(model) == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def build_generation_kwargs(max_new_tokens=None):
    generation_kwargs = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if task_name in DISCRETE_LABELS:
        generation_kwargs.update(
            {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens or 4,
                "use_cache": True,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            }
        )
    else:
        generation_kwargs.update(
            {
                "do_sample": True,
                "top_k": 10,
                "temperature": args.temperature,
                "top_p": 0.9,
                "max_new_tokens": max_new_tokens or 64,
                "use_cache": True,
            }
        )
    return generation_kwargs


def decode_generated_suffix(outputs, input_length):
    generated_tokens = outputs[:, input_length:]
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def generate_prediction(model, prompt):
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cutoff_len,
        return_token_type_ids=False,
    )
    inputs = inputs.to(model.device)
    input_length = inputs["input_ids"].shape[1]
    generation_kwargs = build_generation_kwargs()

    with torch.inference_mode():
        with inference_autocast_context(model):
            outputs = model.generate(**inputs, **generation_kwargs)

    result = decode_generated_suffix(outputs, input_length)[0].strip()
    del inputs
    del outputs
    return result


def canonicalize_discrete_prediction(text):
    normalized = normalize_prediction_text(text)
    if task_name not in DISCRETE_LABELS:
        return normalized

    labels = [normalize_text(label) for label in DISCRETE_LABELS[task_name]]
    label_set = set(labels)
    if normalized in label_set:
        return normalized

    for label in sorted(labels, key=len, reverse=True):
        if label in normalized:
            return label

    compact = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    for label in labels:
        label_compact = re.sub(r"[^a-z0-9]+", " ", label).strip()
        if compact == label_compact or label_compact in compact:
            return label

    return normalized


def normalize_prediction_text(text):
    text = normalize_text(text)
    text = text.split("\n")[0].strip()
    return text


def supports_profile_stream_evaluation():
    return task_name != "tweet_paraphrase"


def evaluate_profile_stream(model, visible_entries, hidden_entries, profile_prefix=None, evaluate_predictions=True):
    label_field = TASK_LABEL_FIELD[task_name]
    predictions = []
    revealed_entries = [dict(item) for item in get_memory_slice(visible_entries)]
    revealed_embeddings = []
    if args.adaptation_mode == "ctta":
        revealed_embeddings = [get_entry_semantic_embedding(item) for item in revealed_entries]
    steps_since_last_adapt = 0
    events = []
    trained_segments = []

    for hidden_idx, entry in enumerate(hidden_entries):
        if evaluate_predictions:
            test_prompt = build_profile_eval_prompt(
                entry,
                revealed_entries,
                profile_prefix=profile_prefix,
            )

            raw_output = generate_prediction(model, test_prompt)
            pred_output = canonicalize_discrete_prediction(raw_output)
            gold_output = normalize_prediction_text(entry.get(label_field, ""))
            is_correct = pred_output == gold_output

            predictions.append(
                {
                    "id": entry.get("id", f"heldout_{hidden_idx}"),
                    "output": raw_output,
                    "prediction_normalized": pred_output,
                    "gold": entry.get(label_field, ""),
                    "gold_normalized": gold_output,
                    "correct": is_correct,
                    "stream_index": hidden_idx,
                }
            )
            if args.verbose_predictions:
                print(f"[heldout_profile] pred={pred_output} gold={gold_output} correct={is_correct}")

        revealed_entries.append(dict(entry))
        revealed_entries = get_memory_slice(revealed_entries)

        if args.adaptation_mode == "base":
            continue

        revealed_embeddings.append(get_entry_semantic_embedding(entry))
        revealed_embeddings = get_memory_slice(revealed_embeddings)
        steps_since_last_adapt += 1
        current_visible_len = len(revealed_embeddings)
        drift_score = compute_preference_drift_from_embeddings(
            revealed_embeddings,
            current_visible_len,
            args.ctta_window_size,
        )
        if drift_score is not None and drift_score >= args.ctta_threshold and steps_since_last_adapt >= args.ctta_update_min_examples:
            segment_start = max(0, current_visible_len - args.ctta_max_update_size)
            event = {
                "trigger_end": current_visible_len,
                "drift_score": round(float(drift_score), 6),
                "segment_start": segment_start,
                "segment_end": current_visible_len,
                "segment_size": current_visible_len - segment_start,
                "update_applied": args.adaptation_mode == "ctta",
            }

            did_train, update_metadata = train_on_visible_entries(
                model,
                revealed_entries,
                profile_prefix,
                start_idx=segment_start,
                end_idx=current_visible_len,
                training_args=training_arguments,
                return_metadata=True,
            )
            if did_train:
                event.update(update_metadata)
                events.append(event)
                trained_segments.append(
                    {
                        "start": segment_start,
                        "end": current_visible_len,
                        "reason": "drift",
                    }
                )
                steps_since_last_adapt = 0
                model.gradient_checkpointing_disable()
                model.eval()
                model.config.use_cache = True

    return predictions, events, trained_segments


def run_inference_for_query_field(model, user_data, query_field, profile_prefix=None):
    if k > 0:
        visible_history_list = []
        for item in user_data["profile"]:
            visible_history_list.append(
                {key: get_first_k_tokens(value, 368) for key, value in item.items()}
            )

        history_list = [prompt_template[args.task_name]["retrieval_history"].format(**p) for p in visible_history_list]
        tokenized_corpus = [doc.split(" ") for doc in history_list]
        bm25 = get_bm25_class()(tokenized_corpus)
    else:
        history_list = None
        bm25 = None

    test_question_list = []
    question_id_list = []

    for q in user_data.get(query_field, []):
        if args.task_name == "citation":
            test_question = q["input"]
            test_article = extract_citation_title(test_question)
            option1 = extract_option(test_question, 1)
            option2 = extract_option(test_question, 2)
            test_prompt = prompt_template[args.task_name]["prompt"].format(test_article, option1, option2)
        else:
            test_question = q["input"]
            test_article = extract_article(test_question)
            test_prompt = prompt_template[args.task_name]["prompt"].format(test_article)

        if k > 0:
            tokenized_query = prompt_template[args.task_name]["retrieval_query_wokey"].format(test_article).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=args.k)
            history_string = "".join(retrieved_history)
            test_prompt = history_string + "\n" + test_prompt

        if args.add_profile and profile_prefix:
            test_prompt = profile_prefix + "\n" + test_prompt

        test_question_list.append(test_prompt)
        question_id_list.append(q["id"])

    out_list = []
    query_batches = split_batch(test_question_list, infer_batch_size)
    query_max_new_tokens = 4 if task_name in DISCRETE_LABELS else 200
    generation_kwargs = build_generation_kwargs(max_new_tokens=query_max_new_tokens)
    for batch in tqdm(query_batches, total=len(query_batches), leave=False):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cutoff_len,
            return_token_type_ids=False,
        )
        inputs = inputs.to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            with inference_autocast_context(model):
                outputs = model.generate(**inputs, **generation_kwargs)

        out_sentence = decode_generated_suffix(outputs, input_length)
        out_list += out_sentence
        del inputs
        del outputs

    predictions = []
    for idx, decoded in enumerate(out_list):
        output = decoded.strip()
        predictions.append({"id": question_id_list[idx], "output": output})
        if args.verbose_predictions:
            print(f"[{query_field}] {output}")

    return predictions


train_data_path = get_train_data_path()
drift_tag = resolve_drift_tag(train_data_path)

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=args.access_token)
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "[PAD]"
tokenizer.pad_token_id = tokenizer.eos_token_id

quantization_config = None
if quantized_loading:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=False,
    device_map=args.device_map,
    trust_remote_code=True,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)

base_model.config.use_cache = False
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id
base_model.config.bos_token_id = tokenizer.bos_token_id

base_model.gradient_checkpointing_enable()
if getattr(base_model, "is_loaded_in_4bit", False) or getattr(base_model, "is_loaded_in_8bit", False):
    base_model = prepare_model_for_kbit_training(base_model)
base_model.enable_input_require_grads()

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def create_user_lora_model(base_model):
    model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(model)
    model.gradient_checkpointing_enable()
    return model


def unload_user_lora_model(model):
    peft_base = getattr(model, "base_model", None)
    if peft_base is not None and hasattr(peft_base, "unload"):
        return peft_base.unload()
    if hasattr(model, "unload"):
        return model.unload()
    return model


training_arguments = transformers.TrainingArguments(
    output_dir="output/",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    num_train_epochs=max_epoch,
    save_steps=10**9,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=1e-2,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    group_by_length=False,
    lr_scheduler_type="linear",
    report_to="none",
)

with open(train_data_path, "r") as f:
    test_data = json.load(f)

with open("./prompt/prompt.json", "r") as f:
    prompt_template = json.load(f)

if args.add_profile:
    with open(f"./data/{task_name}/profile_user_100.json", "r") as f:
        test_profile = json.load(f)

format_flag = task_name != "tweet_paraphrase"
extract_article = TASK_TO_EXTRACTOR.get(task_name)

# Match the original OPPU setup for every mode: first absorb the task LoRA into
# the shared task-specific base, then create a fresh user LoRA per user.
task_lora_model = PeftModel.from_pretrained(model=base_model, model_id=args.task_lora, is_trainable=False)
base_model = task_lora_model.merge_and_unload()
print_trainable_parameters(task_lora_model)
del task_lora_model
cleanup_memory()
task_adapter_mode = "merge_task_adapter_then_add_user_adapter"

pred_all_profile = []
pred_all_query = []
ctta_logs = []

for user_idx in tqdm(range(len(test_data))):
    user_data = test_data[user_idx]
    profile_entries = user_data["profile"]
    profile_prefix = test_profile[user_idx]["output"] if args.add_profile else None

    model = create_user_lora_model(base_model)

    profile_len = len(profile_entries)
    visible_prefix_len = resolve_visible_prefix_length(profile_len, user_data=user_data)
    visible_entries = profile_entries[:visible_prefix_len]
    hidden_entries = profile_entries[visible_prefix_len:]
    warmup_end = min(len(visible_entries), max(args.ctta_init_size, args.ctta_window_size))
    user_log = {
        "user_index": user_idx,
        "user_id": user_data.get("user_id"),
        "profile_length": profile_len,
        "visible_prefix_len": visible_prefix_len,
        "hidden_suffix_len": len(hidden_entries),
        "memory_size": args.memory_size,
        "warmup_end": warmup_end,
        "adaptation_mode": args.adaptation_mode,
        "drift_threshold": args.ctta_threshold,
        "window_size": args.ctta_window_size,
        "anti_forgetting": args.ctta_anti_forgetting,
        "replay_size": args.ctta_replay_size,
        "replay_strategy": args.ctta_replay_strategy,
        "anchor_lambda": args.ctta_anchor_lambda,
        "lwf_lambda": args.ctta_lwf_lambda,
        "events": [],
    }

    trained_segments = []
    if warmup_end > 0:
        did_train = train_on_visible_entries(
            model,
            visible_entries,
            profile_prefix,
            start_idx=0,
            end_idx=warmup_end,
            training_args=training_arguments,
        )
        if did_train:
            trained_segments.append({"start": 0, "end": warmup_end, "reason": "warmup"})

    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True

    profile_predictions, drift_events, stream_segments = evaluate_profile_stream(
        model,
        visible_entries=visible_entries,
        hidden_entries=hidden_entries,
        profile_prefix=profile_prefix,
        evaluate_predictions=supports_profile_stream_evaluation(),
    )
    pred_all_profile.extend(profile_predictions)

    user_log["events"].extend(drift_events)
    user_log["trained_segments"] = trained_segments + stream_segments
    user_log["num_profile_predictions"] = len(profile_predictions)
    user_log["num_correct"] = sum(1 for item in profile_predictions if item["correct"])
    ctta_logs.append(user_log)

    if "query" in user_data:
        query_predictions = run_inference_for_query_field(model, user_data, "query", profile_prefix=profile_prefix)
        pred_all_query.extend(query_predictions)

    if args.save_user_ckpt:
        ckpt_dir = os.path.join(".", "ckpt", args.task_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_output_name = os.path.join(
            ckpt_dir,
            f"k{k}-{task_name}-{model_name.split('/')[-1]}-{args.output_tag}-user_{user_idx}",
        )
        model.save_pretrained(ckpt_output_name)

    base_model = unload_user_lora_model(model)
    del model
    cleanup_memory()


output_dir = os.path.join("./output", args.task_name, build_dataset_output_dirname(drift_tag))
os.makedirs(output_dir, exist_ok=True)

output_name = build_ctta_run_name()


def dump_prediction_file(path, predictions):
    payload = {
        "task": name2taskid[task_name],
        "golds": predictions,
        "model": model_name,
        "train_data_path": os.path.abspath(train_data_path),
        "drift_tag": drift_tag,
        "semantic_model_path": args.semantic_model_path,
        "semantic_device": resolve_semantic_device(),
        "drift_tag_weight": args.drift_tag_weight,
        "drift_text_weight": args.drift_text_weight,
        "adaptation_mode": args.adaptation_mode,
        "ctta_threshold": args.ctta_threshold,
        "ctta_window_size": args.ctta_window_size,
        "ctta_init_size": args.ctta_init_size,
        "ctta_update_min_examples": args.ctta_update_min_examples,
        "ctta_max_update_size": args.ctta_max_update_size,
        "ctta_anti_forgetting": args.ctta_anti_forgetting,
        "ctta_replay_size": args.ctta_replay_size,
        "ctta_replay_strategy": args.ctta_replay_strategy,
        "ctta_anchor_lambda": args.ctta_anchor_lambda,
        "ctta_lwf_lambda": args.ctta_lwf_lambda,
        "ctta_distill_temperature": args.ctta_distill_temperature,
        "profile_split_mode": args.profile_split_mode,
        "profile_split_ratio": args.profile_split_ratio,
        "profile_split_count": args.profile_split_count,
        "memory_size": args.memory_size,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
        "quantized_loading": quantized_loading,
        "task_adapter_mode": task_adapter_mode,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


if pred_all_profile:
    dump_prediction_file(os.path.join(output_dir, f"{output_name}-heldout_profile.json"), pred_all_profile)

if pred_all_query:
    dump_prediction_file(os.path.join(output_dir, f"{output_name}-query.json"), pred_all_query)

if pred_all_profile:
    summary = {
        "num_examples": len(pred_all_profile),
        "num_correct": sum(1 for item in pred_all_profile if item["correct"]),
    }
    summary["accuracy"] = (summary["num_correct"] / summary["num_examples"]) if summary["num_examples"] else 0.0
    summary.update(
        {
            "task_name": task_name,
            "train_data_path": os.path.abspath(train_data_path),
            "drift_tag": drift_tag,
            "semantic_model_path": args.semantic_model_path,
            "semantic_device": resolve_semantic_device(),
            "drift_tag_weight": args.drift_tag_weight,
            "drift_text_weight": args.drift_text_weight,
            "adaptation_mode": args.adaptation_mode,
            "ctta_threshold": args.ctta_threshold,
            "ctta_window_size": args.ctta_window_size,
            "ctta_anti_forgetting": args.ctta_anti_forgetting,
            "ctta_replay_size": args.ctta_replay_size,
            "ctta_replay_strategy": args.ctta_replay_strategy,
            "ctta_anchor_lambda": args.ctta_anchor_lambda,
            "ctta_lwf_lambda": args.ctta_lwf_lambda,
            "ctta_distill_temperature": args.ctta_distill_temperature,
            "profile_split_mode": args.profile_split_mode,
            "profile_split_ratio": args.profile_split_ratio,
            "profile_split_count": args.profile_split_count,
            "memory_size": args.memory_size,
            "load_in_8bit": args.load_in_8bit,
            "load_in_4bit": args.load_in_4bit,
            "quantized_loading": quantized_loading,
            "task_adapter_mode": task_adapter_mode,
            "num_query_predictions": len(pred_all_query),
        }
    )
    with open(os.path.join(output_dir, f"{output_name}-heldout_profile-summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

log_path = os.path.join(output_dir, f"{output_name}-drift_log.json")
with open(log_path, "w") as f:
    json.dump(ctta_logs, f, indent=4)
