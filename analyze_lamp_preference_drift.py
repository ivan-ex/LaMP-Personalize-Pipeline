import argparse
import json
import math
import os
import re

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/model/Huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
)
DEFAULT_TASK_NAME = "movie_tagging"

TASK_ALIASES = {
    "movie": "movie_tagging",
    "movie_tagging": "movie_tagging",
    "2m": "movie_tagging",
    "lamp_2m": "movie_tagging",
    "tweet": "tweet_paraphrase",
    "tweets": "tweet_paraphrase",
    "tweet_paraphrase": "tweet_paraphrase",
    "tweet_paraphrasing": "tweet_paraphrase",
    "7": "tweet_paraphrase",
    "lamp_7": "tweet_paraphrase",
}

TASK_CONFIGS = {
    "movie_tagging": {
        "data_subdir": "movie_tagging",
        "input_filename": "all_user.json",
        "text_key": "description",
        "label_key": "tag",
        "profile_num": 20,
        "text_weight": 1.0,
        "gold_weight": 3.0,
        "style_weight": 0.0,
        "style_mode": None,
        "window_size": 8,
        "step_size": 4,
        "comparison_mode": "cross_window",
        "min_window_gap": 2,
        "metric": "cosine_distance_between_weighted_text_label_embeddings",
    },
    "tweet_paraphrase": {
        "data_subdir": "tweet_paraphrase",
        "input_filename": "all_user.json",
        "text_key": "text",
        "label_key": None,
        "profile_num": 16,
        "text_weight": 0.5,
        "gold_weight": 0.0,
        "style_weight": 1.5,
        "style_mode": "tweet",
        "window_size": 8,
        "step_size": 4,
        "comparison_mode": "cross_window",
        "min_window_gap": 0,
        "metric": "cosine_distance_between_tweet_text_and_style_embeddings",
    },
}


def default_input_path(task_name):
    config = TASK_CONFIGS[task_name]
    return os.path.join(REPO_ROOT, "data", config["data_subdir"], config["input_filename"])


def default_output_path(task_name):
    return os.path.join(REPO_ROOT, "data", TASK_CONFIGS[task_name]["data_subdir"], "analysis", "preference_drift_analysis.json")


def default_custom_drift_dir(task_name):
    return os.path.join(REPO_ROOT, "data", TASK_CONFIGS[task_name]["data_subdir"], "custom_drifts")


DEFAULT_INPUT_PATH = default_input_path(DEFAULT_TASK_NAME)
DEFAULT_OUTPUT_PATH = default_output_path(DEFAULT_TASK_NAME)
DEFAULT_CUSTOM_DRIFT_DIR = default_custom_drift_dir(DEFAULT_TASK_NAME)


def load_users(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_task_name(task_name):
    if task_name is None:
        return None
    key = str(task_name).strip().lower().replace("-", "_")
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    raise ValueError(f"Unsupported task_name: {task_name}. Supported tasks: {sorted(TASK_CONFIGS)}")


def infer_task_name_from_path(input_path):
    if not input_path:
        return None
    normalized_parts = set(os.path.normpath(input_path).split(os.sep))
    if "tweet" in normalized_parts or "tweet_paraphrase" in normalized_parts:
        return "tweet_paraphrase"
    if "movie_tagging" in normalized_parts:
        return "movie_tagging"
    return None


def normalize_optional_key(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "na", "n/a"}:
        return None
    return text


def is_auto_value(value):
    return value is None or str(value).strip().lower() in {"auto", "default"}


def apply_task_defaults(args):
    task_name = normalize_task_name(args.task_name) or infer_task_name_from_path(args.input_path) or DEFAULT_TASK_NAME
    config = TASK_CONFIGS[task_name]
    args.task_name = task_name

    if args.input_path is None:
        args.input_path = default_input_path(task_name)
    if args.output_path is None:
        args.output_path = default_output_path(task_name)
    if args.custom_drift_dir is None:
        args.custom_drift_dir = default_custom_drift_dir(task_name)

    if is_auto_value(args.text_key):
        args.text_key = config["text_key"]
    else:
        args.text_key = normalize_optional_key(args.text_key)
    if is_auto_value(args.label_key):
        args.label_key = config["label_key"]
    else:
        args.label_key = normalize_optional_key(args.label_key)

    for name in (
        "profile_num",
        "text_weight",
        "gold_weight",
        "style_weight",
        "window_size",
        "step_size",
        "comparison_mode",
        "min_window_gap",
    ):
        if getattr(args, name) is None:
            setattr(args, name, config[name])

    args.style_mode = config["style_mode"] if config["style_mode"] and args.style_weight > 0 else None
    if args.style_mode:
        args.metric = config["metric"]
    elif args.label_key is None or args.gold_weight <= 0:
        args.metric = "cosine_distance_between_weighted_text_embeddings"
    else:
        args.metric = TASK_CONFIGS["movie_tagging"]["metric"]
    return args


def resolve_hf_model_path(model_root):
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")
    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            revision = f.read().strip()
        snapshot_path = os.path.join(snapshots_dir, revision)
        if os.path.isdir(snapshot_path):
            return snapshot_path
    return model_root


class SemanticEmbedder:
    def __init__(self, model_path, device="cpu", batch_size=64, max_length=512):
        resolved_path = resolve_hf_model_path(model_path)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(resolved_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts):
        if not texts:
            hidden_size = getattr(self.model.config, "hidden_size", 384)
            return np.zeros((0, hidden_size), dtype=np.float32)

        all_embeddings = []
        with torch.no_grad():
            for start_idx in range(0, len(texts), self.batch_size):
                batch = texts[start_idx:start_idx + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = torch.sum(token_embeddings * attention_mask, dim=1)
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = summed / counts
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeddings, axis=0)


def cosine_distance(vec_a, vec_b):
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 1e-12:
        return 0.0
    cosine_sim = float(np.dot(vec_a, vec_b) / denom)
    cosine_sim = max(min(cosine_sim, 1.0), -1.0)
    return 1.0 - cosine_sim


def centroid(embeddings):
    if len(embeddings) == 0:
        return None
    center = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(center)
    if norm <= 1e-12:
        return center
    return center / norm


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
HTML_ENTITY_RE = re.compile(r"&[a-z]+;", re.IGNORECASE)
EMOTICON_RE = re.compile(r"(?:(?:[:;=xX8][-']?[)(DPp/\\])|<3|\^\^)")
LAUGHTER_RE = re.compile(r"\b(?:ha(?:ha)+|he(?:he)+|lol+|lmao+|rofl+)\b", re.IGNORECASE)
SHORT_FORM_RE = re.compile(
    r"\b(?:omg|wtf|idk|imo|imho|btw|thx|pls|plz|bc|cuz|u|ur|ya|yall|"
    r"gonna|wanna|gotta|kinda|sorta|tho|tmrw|2day|gr8|xoxo|msg|dm)\b",
    re.IGNORECASE,
)
FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|im|me|my|mine|we|we're|our|ours)\b", re.IGNORECASE)
SECOND_PERSON_RE = re.compile(r"\b(?:you|u|ur|your|yours|ya|yall)\b", re.IGNORECASE)
REPEATED_LETTER_RE = re.compile(r"([A-Za-z])\1{2,}")
REPEATED_PUNCT_RE = re.compile(r"[!?.,]{2,}")


def squash_count(value, scale):
    return math.tanh(float(value) / float(scale))


def safe_ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def l2_normalize_np(vector):
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return vector
    return vector / norm


def tweet_style_features(text):
    text = "" if text is None else str(text)
    tokens = re.findall(r"[A-Za-z0-9_']+", text)
    word_count = len(tokens)
    char_count = len(text)
    letters = [char for char in text if char.isalpha()]
    uppercase_letters = [char for char in letters if char.isupper()]
    uppercase_tokens = [token for token in tokens if len(token) > 1 and token.isupper()]
    lowercase_tokens = [token for token in tokens if len(token) > 1 and token.islower()]
    avg_word_len = safe_ratio(sum(len(token) for token in tokens), word_count)

    features = [
        squash_count(char_count, 280.0),
        squash_count(word_count, 40.0),
        squash_count(avg_word_len, 8.0),
        safe_ratio(len(uppercase_letters), len(letters)),
        safe_ratio(len(uppercase_tokens), word_count),
        safe_ratio(len(lowercase_tokens), word_count),
        squash_count(len(MENTION_RE.findall(text)), 5.0),
        squash_count(len(HASHTAG_RE.findall(text)), 4.0),
        squash_count(len(URL_RE.findall(text)), 2.0),
        squash_count(text.count("!"), 6.0),
        squash_count(text.count("?"), 4.0),
        squash_count(text.count("."), 8.0),
        squash_count(text.count(","), 6.0),
        squash_count(text.count("'"), 8.0),
        squash_count(text.count('"'), 4.0),
        squash_count(text.count("/"), 4.0),
        squash_count(text.count("&"), 4.0),
        squash_count(text.count("(") + text.count(")"), 4.0),
        squash_count(len(REPEATED_PUNCT_RE.findall(text)), 4.0),
        squash_count(len(REPEATED_LETTER_RE.findall(text)), 4.0),
        squash_count(len(EMOTICON_RE.findall(text)), 4.0),
        squash_count(len(LAUGHTER_RE.findall(text)), 4.0),
        squash_count(len(SHORT_FORM_RE.findall(text)), 6.0),
        squash_count(len(HTML_ENTITY_RE.findall(text)), 4.0),
        squash_count(sum(1 for char in text if ord(char) > 127), 4.0),
        squash_count(sum(1 for char in text if char.isdigit()), 8.0),
        squash_count(len(FIRST_PERSON_RE.findall(text)), 8.0),
        squash_count(len(SECOND_PERSON_RE.findall(text)), 8.0),
        1.0 if text[:1].islower() else 0.0,
        1.0 if text.endswith(("!", "?", "...")) else 0.0,
    ]
    return l2_normalize_np(np.asarray(features, dtype=np.float32))


def representative_examples(texts, embeddings, center, top_k=3):
    if center is None or len(texts) == 0:
        return []
    scored = []
    for idx, emb in enumerate(embeddings):
        scored.append((cosine_distance(emb, center), idx))
    scored.sort(key=lambda x: x[0])
    results = []
    for _, idx in scored[:top_k]:
        text = " ".join(str(texts[idx]).split())
        results.append({
            "index": idx,
            "text": text[:220],
        })
    return results


def extract_text(item, text_key):
    if text_key is None:
        return None
    value = item.get(text_key)
    if value is None:
        return None
    value = " ".join(str(value).split())
    return value or None


def build_user_semantic_profile(
    user,
    text_key,
    label_key,
    embedder,
    text_weight,
    gold_weight,
    style_mode=None,
    style_weight=0.0,
):
    profile_items = []
    rendered_texts = []
    for item in user.get("profile", []):
        text = extract_text(item, text_key)
        label = extract_text(item, label_key)
        if text is None and label is None:
            continue
        profile_items.append({
            "text": text,
            "label": label,
        })
        if text and label:
            rendered_texts.append(f"[gold] {label} [text] {text}")
        else:
            rendered_texts.append(text or label or "")

    hidden_size = getattr(embedder.model.config, "hidden_size", 384)
    use_style = style_mode == "tweet" and style_weight > 0
    style_dim = len(tweet_style_features("")) if use_style else 0
    text_indices = []
    text_values = []
    label_indices = []
    label_values = []
    for idx, item in enumerate(profile_items):
        if item["text"] is not None and text_weight > 0:
            text_indices.append(idx)
            text_values.append(item["text"])
        if item["label"] is not None and gold_weight > 0:
            label_indices.append(idx)
            label_values.append(item["label"])

    text_embedding_map = {
        idx: embedding
        for idx, embedding in zip(text_indices, embedder.encode(text_values))
    }
    label_embedding_map = {
        idx: embedding
        for idx, embedding in zip(label_indices, embedder.encode(label_values))
    }

    embeddings = []
    for idx, item in enumerate(profile_items):
        parts = []
        weights = []
        if item["text"] is not None and text_weight > 0:
            text_embedding = text_embedding_map[idx]
            parts.append(text_embedding)
            weights.append(text_weight)
        if item["label"] is not None and gold_weight > 0:
            label_embedding = label_embedding_map[idx]
            parts.append(label_embedding)
            weights.append(gold_weight)

        if not parts:
            semantic_embedding = np.zeros(hidden_size, dtype=np.float32)
            semantic_weight = 0.0
        else:
            semantic_embedding = np.zeros(hidden_size, dtype=np.float32)
            semantic_weight = float(sum(weights))
            for part, weight in zip(parts, weights):
                semantic_embedding += (weight / semantic_weight) * part

            semantic_embedding = l2_normalize_np(semantic_embedding).astype(np.float32)

        if use_style:
            style_embedding = tweet_style_features(item["text"] or item["label"] or "")
            combined = np.concatenate(
                [
                    semantic_embedding * semantic_weight,
                    style_embedding * float(style_weight),
                ],
                axis=0,
            )
            combined = l2_normalize_np(combined).astype(np.float32)
            embeddings.append(combined)
            continue

        embeddings.append(semantic_embedding.astype(np.float32))

    embedding_dim = hidden_size + style_dim if use_style else hidden_size
    embeddings = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, embedding_dim), dtype=np.float32)

    return {
        "user_id": user.get("user_id"),
        "profile_length": len(rendered_texts),
        "texts": rendered_texts,
        "embeddings": embeddings,
    }


def generate_windows(n_items, window_size, step_size):
    windows = []
    for start_idx in range(0, n_items - window_size + 1, step_size):
        windows.append((start_idx, start_idx + window_size))
    return windows


def generate_adjacent_window_pairs(n_items, window_size, step_size):
    pairs = []
    if n_items < 2 * window_size:
        return pairs

    for current_end in range(2 * window_size, n_items + 1, step_size):
        pairs.append({
            "current_end": current_end,
            "window_a": (current_end - 2 * window_size, current_end - window_size),
            "window_b": (current_end - window_size, current_end),
            "gap": 0,
        })
    return pairs


def generate_cross_window_pairs(n_items, window_size, step_size, min_window_gap):
    pairs = []
    windows = generate_windows(n_items, window_size, step_size)
    for left_idx in range(len(windows)):
        for right_idx in range(left_idx + 1, len(windows)):
            left_start, left_end = windows[left_idx]
            right_start, right_end = windows[right_idx]
            gap = right_start - left_end
            if gap < min_window_gap:
                continue
            pairs.append({
                "current_end": right_end,
                "window_a": (left_start, left_end),
                "window_b": (right_start, right_end),
                "gap": gap,
            })
    return pairs


def serialize_window(start_idx, end_idx, semantic_user, center):
    return {
        "start": start_idx,
        "end": end_idx,
        "examples": representative_examples(
            semantic_user["texts"][start_idx:end_idx],
            semantic_user["embeddings"][start_idx:end_idx],
            center,
        ),
    }


def analyze_sliding(semantic_user, window_size, step_size, comparison_mode, min_window_gap):
    n_items = len(semantic_user["texts"])
    if comparison_mode == "cross_window":
        window_pairs = generate_cross_window_pairs(n_items, window_size, step_size, min_window_gap)
    elif comparison_mode == "adjacent":
        window_pairs = generate_adjacent_window_pairs(n_items, window_size, step_size)
    else:
        raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")

    if not window_pairs:
        return {
            "user_id": semantic_user["user_id"],
            "profile_length": n_items,
            "window_size": window_size,
            "step_size": step_size,
            "comparison_mode": comparison_mode,
            "min_window_gap": min_window_gap if comparison_mode == "cross_window" else None,
            "num_window_pairs": 0,
            "max_semantic_cosine_distance": None,
            "drift_detected": False,
            "best_pair": None,
            "window_pair_scores": [],
        }

    best_distance = -1.0
    best_pair = None
    window_pair_scores = []
    for pair in window_pairs:
        left_start, left_end = pair["window_a"]
        right_start, right_end = pair["window_b"]
        left_center = centroid(semantic_user["embeddings"][left_start:left_end])
        right_center = centroid(semantic_user["embeddings"][right_start:right_end])
        if left_center is None or right_center is None:
            continue

        distance = cosine_distance(left_center, right_center)
        score_item = {
            "current_end": pair["current_end"],
            "window_a": {"start": left_start, "end": left_end},
            "window_b": {"start": right_start, "end": right_end},
            "gap": pair["gap"],
            "semantic_cosine_distance": round(float(distance), 6),
        }
        window_pair_scores.append(score_item)
        if distance > best_distance:
            best_distance = distance
            best_pair = {
                "current_end": pair["current_end"],
                "gap": pair["gap"],
                "window_a": serialize_window(left_start, left_end, semantic_user, left_center),
                "window_b": serialize_window(right_start, right_end, semantic_user, right_center),
            }

    return {
        "user_id": semantic_user["user_id"],
        "profile_length": n_items,
        "window_size": window_size,
        "step_size": step_size,
        "comparison_mode": comparison_mode,
        "min_window_gap": min_window_gap if comparison_mode == "cross_window" else None,
        "num_window_pairs": len(window_pairs),
        "num_scored_window_pairs": len(window_pair_scores),
        "max_semantic_cosine_distance": best_distance if best_pair is not None else None,
        "drift_detected": False,
        "best_pair": best_pair,
        "window_pair_scores": window_pair_scores,
    }


def summarize_sliding(results, threshold):
    values = [item["max_semantic_cosine_distance"] for item in results if item["max_semantic_cosine_distance"] is not None]
    detected_count = sum(
        1 for item in results if item["max_semantic_cosine_distance"] is not None and item["max_semantic_cosine_distance"] >= threshold
    )
    return {
        "num_users": len(results),
        "mean_max_semantic_cosine_distance": float(np.mean(values)) if values else 0.0,
        "max_max_semantic_cosine_distance": float(np.max(values)) if values else 0.0,
        "num_above_threshold": detected_count,
        "threshold": threshold,
    }


def build_distribution_stats(results, step):
    values = sorted(
        item["max_semantic_cosine_distance"]
        for item in results
        if item["max_semantic_cosine_distance"] is not None
    )
    if not values:
        return {
            "step": step,
            "bin_counts": [],
            "threshold_counts": [],
        }

    upper = max(1.0, float(np.ceil(max(values) / step) * step))
    edges = np.arange(0.0, upper + step, step)
    hist, bin_edges = np.histogram(values, bins=edges)

    bin_counts = []
    for idx, count in enumerate(hist):
        bin_counts.append({
            "start": round(float(bin_edges[idx]), 4),
            "end": round(float(bin_edges[idx + 1]), 4),
            "count": int(count),
        })

    threshold_counts = []
    thresholds = np.arange(0.0, upper + 1e-9, step)
    for threshold in thresholds:
        count = sum(value >= threshold for value in values)
        threshold_counts.append({
            "threshold": round(float(threshold), 4),
            "count": int(count),
        })

    return {
        "step": step,
        "bin_counts": bin_counts,
        "threshold_counts": threshold_counts,
    }


def decimal_places(step):
    text = f"{step:.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def build_user_drift_groups(results, group_step):
    groups = {}
    precision = max(1, decimal_places(group_step))
    total_users = len(results)

    for item in results:
        value = item["max_semantic_cosine_distance"]
        if value is None:
            groups.setdefault("insufficient_windows", {
                "range": "insufficient_windows",
                "start": None,
                "end": None,
                "count": 0,
                "users": [],
            })
            groups["insufficient_windows"]["users"].append({
                "user_id": item["user_id"],
                "profile_length": item["profile_length"],
                "drift_score": None,
            })
            continue

        bucket_start = math.floor((float(value) + 1e-12) / group_step) * group_step
        bucket_end = bucket_start + group_step
        label = f"{bucket_start:.{precision}f}-{bucket_end:.{precision}f}"
        groups.setdefault(label, {
            "range": label,
            "start": round(float(bucket_start), precision),
            "end": round(float(bucket_end), precision),
            "count": 0,
            "users": [],
        })
        best_pair = item.get("best_pair") or {}
        groups[label]["users"].append({
            "user_id": item["user_id"],
            "profile_length": item["profile_length"],
            "drift_score": round(float(value), 6),
            "drift_detected": item["drift_detected"],
            "best_current_end": best_pair.get("current_end"),
            "best_gap": best_pair.get("gap"),
        })

    for group in groups.values():
        group["users"].sort(
            key=lambda row: -1.0 if row["drift_score"] is None else row["drift_score"],
            reverse=True,
        )
        group["count"] = len(group["users"])
        group["percentage"] = round((group["count"] / total_users) * 100, 2) if total_users else 0.0

    return sorted(
        groups.values(),
        key=lambda group: (float("inf") if group["start"] is None else group["start"]),
    )


def plot_distribution(distribution_stats, output_path):
    bin_counts = distribution_stats["bin_counts"]
    threshold_counts = distribution_stats["threshold_counts"]

    if not bin_counts:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    precision = max(1, decimal_places(distribution_stats["step"]))
    x_labels = [f"{item['start']:.{precision}f}-{item['end']:.{precision}f}" for item in bin_counts]
    y_counts = [item["count"] for item in bin_counts]
    threshold_x = [item["threshold"] for item in threshold_counts]
    threshold_y = [item["count"] for item in threshold_counts]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].bar(x_labels, y_counts, color="#4C72B0")
    axes[0].set_title("Distribution of User Max Semantic Distance")
    axes[0].set_xlabel("Max Semantic Cosine Distance Range")
    axes[0].set_ylabel("User Count")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(threshold_x, threshold_y, marker="o", color="#DD8452")
    axes[1].set_title("Users Above Threshold")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("User Count")
    axes[1].set_xticks(threshold_x)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_json(data, output_path):
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_custom_drift_datasets(
    users,
    drift_groups,
    output_dir,
    score_min,
    score_max,
    file_prefix,
    write_manifest,
    generation_config,
):
    user_by_id = {str(user.get("user_id")): user for user in users}
    os.makedirs(output_dir, exist_ok=True)

    generated = []
    for group in drift_groups:
        group_start = group.get("start")
        group_end = group.get("end")
        if group_start is None or group_end is None:
            continue
        if group_start < score_min - 1e-12 or group_end > score_max + 1e-12:
            continue
        if group.get("count", 0) <= 0:
            continue

        dataset = []
        missing_user_ids = []
        for row in group["users"]:
            user = user_by_id.get(str(row["user_id"]))
            if user is None:
                missing_user_ids.append(row["user_id"])
                continue
            dataset.append(user)

        range_label = group["range"]
        dataset_path = os.path.join(output_dir, f"{file_prefix}_{range_label}.json")
        save_json(dataset, dataset_path)

        manifest_path = None
        if write_manifest:
            manifest_path = os.path.join(output_dir, f"{file_prefix}_{range_label}-manifest.json")
            save_json(
                {
                    "dataset_path": os.path.abspath(dataset_path),
                    "range": range_label,
                    "start": group_start,
                    "end": group_end,
                    "count": len(dataset),
                    "group_count": group["count"],
                    "percentage": group["percentage"],
                    "missing_user_ids": missing_user_ids,
                    "generation_config": generation_config,
                    "users": group["users"],
                },
                manifest_path,
            )

        generated.append({
            "range": range_label,
            "path": os.path.abspath(dataset_path),
            "manifest_path": os.path.abspath(manifest_path) if manifest_path else None,
            "count": len(dataset),
            "percentage": group["percentage"],
            "missing_user_count": len(missing_user_ids),
        })

    return generated


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic preference drift with sliding windows.")
    parser.add_argument("--task_name", "--task-name", default=None, help="Task name or alias, e.g. movie_tagging or tweet_paraphrase.")
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--text_key", default="auto")
    parser.add_argument("--label_key", default="auto", help="Set to none/null/empty to disable label embeddings.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--profile_num", type=int, default=None)
    parser.add_argument("--text_weight", type=float, default=None)
    parser.add_argument("--gold_weight", type=float, default=None)
    parser.add_argument(
        "--style_weight",
        type=float,
        default=None,
        help="Task-specific stylometric weight. Defaults to 0 for movie_tagging and 1.5 for tweet_paraphrase.",
    )
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument(
        "--comparison_mode",
        choices=["cross_window", "adjacent"],
        default=None,
        help="cross_window keeps the original paired-window max drift logic; adjacent matches CTTA online detection.",
    )
    parser.add_argument(
        "--min_window_gap",
        type=int,
        default=None,
        help="Minimum gap between paired windows in cross_window mode.",
    )
    parser.add_argument("--sliding_threshold", type=float, default=0.20)
    parser.add_argument("--distribution_step", type=float, default=0.10)
    parser.add_argument(
        "--group_step",
        type=float,
        default=None,
        help="Width for drift-score user groups; defaults to --distribution_step.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
    )
    parser.add_argument(
        "--write_custom_drifts",
        "--write-custom-drifts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one custom_drifts dataset per selected drift-score range.",
    )
    parser.add_argument("--custom_drift_dir", "--custom-drift-dir", default=None)
    parser.add_argument(
        "--custom_drift_min_score",
        "--custom-drift-min-score",
        type=float,
        default=0.1,
        help="Lowest drift group start to export into custom_drifts.",
    )
    parser.add_argument(
        "--custom_drift_max_score",
        "--custom-drift-max-score",
        type=float,
        default=0.6,
        help="Highest drift group end to export into custom_drifts.",
    )
    parser.add_argument(
        "--custom_drift_prefix",
        "--custom-drift-prefix",
        default="drift",
        help="Filename prefix for generated custom drift datasets.",
    )
    parser.add_argument(
        "--write_custom_drift_manifests",
        "--write-custom-drift-manifests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write companion manifests with user ids and drift scores.",
    )
    args = parser.parse_args()
    args = apply_task_defaults(args)

    group_step = args.group_step if args.group_step is not None else args.distribution_step
    if args.window_size <= 0:
        raise ValueError("--window_size must be positive")
    if args.step_size <= 0:
        raise ValueError("--step_size must be positive")
    if args.min_window_gap < 0:
        raise ValueError("--min_window_gap must be non-negative")
    if args.text_weight < 0:
        raise ValueError("--text_weight must be non-negative")
    if args.gold_weight < 0:
        raise ValueError("--gold_weight must be non-negative")
    if args.style_weight < 0:
        raise ValueError("--style_weight must be non-negative")
    if args.text_weight == 0 and args.gold_weight == 0 and args.style_weight == 0:
        raise ValueError("At least one of --text_weight, --gold_weight, or --style_weight must be positive")
    if args.distribution_step <= 0:
        raise ValueError("--distribution_step must be positive")
    if group_step <= 0:
        raise ValueError("--group_step must be positive")
    if args.custom_drift_min_score < 0:
        raise ValueError("--custom_drift_min_score must be non-negative")
    if args.custom_drift_max_score <= args.custom_drift_min_score:
        raise ValueError("--custom_drift_max_score must be greater than --custom_drift_min_score")

    users = load_users(args.input_path)
    filtered_users = [user for user in users if len(user.get("profile", [])) >= args.profile_num]

    embedder = SemanticEmbedder(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    semantic_users = [
        build_user_semantic_profile(
            user,
            args.text_key,
            args.label_key,
            embedder,
            args.text_weight,
            args.gold_weight,
            style_mode=args.style_mode,
            style_weight=args.style_weight,
        )
        for user in filtered_users
    ]

    sliding_results = [
        analyze_sliding(
            user,
            args.window_size,
            args.step_size,
            args.comparison_mode,
            args.min_window_gap,
        )
        for user in semantic_users
    ]

    for item in sliding_results:
        value = item["max_semantic_cosine_distance"]
        item["drift_detected"] = value is not None and value >= args.sliding_threshold

    sliding_summary = summarize_sliding(sliding_results, args.sliding_threshold)
    distribution_stats = build_distribution_stats(sliding_results, args.distribution_step)
    drift_groups = build_user_drift_groups(sliding_results, group_step)
    custom_drift_datasets = []
    custom_drift_generation_config = {
        "input_path": os.path.abspath(args.input_path),
        "task_name": args.task_name,
        "score_min": args.custom_drift_min_score,
        "score_max": args.custom_drift_max_score,
        "group_step": group_step,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "comparison_mode": args.comparison_mode,
        "min_window_gap": args.min_window_gap if args.comparison_mode == "cross_window" else None,
        "profile_num": args.profile_num,
        "text_weight": args.text_weight,
        "gold_weight": args.gold_weight,
        "style_mode": args.style_mode,
        "style_weight": args.style_weight,
    }
    if args.write_custom_drifts:
        custom_drift_datasets = write_custom_drift_datasets(
            users=filtered_users,
            drift_groups=drift_groups,
            output_dir=args.custom_drift_dir,
            score_min=args.custom_drift_min_score,
            score_max=args.custom_drift_max_score,
            file_prefix=args.custom_drift_prefix,
            write_manifest=args.write_custom_drift_manifests,
            generation_config=custom_drift_generation_config,
        )
    sliding_top_users = sorted(
        [item for item in sliding_results if item["max_semantic_cosine_distance"] is not None],
        key=lambda x: x["max_semantic_cosine_distance"],
        reverse=True,
    )[:10]

    resolved_model_path = os.path.abspath(resolve_hf_model_path(args.model_path))
    plot_output_path = os.path.splitext(args.output_path)[0] + "_distribution.png"
    plot_distribution(distribution_stats, plot_output_path)

    report = {
        "input_path": os.path.abspath(args.input_path),
        "task_name": args.task_name,
        "text_key": args.text_key,
        "label_key": args.label_key,
        "style_mode": args.style_mode,
        "semantic_model_path": resolved_model_path,
        "metric": args.metric,
        "profile_num": args.profile_num,
        "text_weight": args.text_weight,
        "gold_weight": args.gold_weight,
        "style_weight": args.style_weight,
        "eligible_user_count": len(filtered_users),
        "distribution_step": args.distribution_step,
        "group_step": group_step,
        "drift_group_total_users": len(sliding_results),
        "distribution_stats": distribution_stats,
        "drift_groups": drift_groups,
        "custom_drift_datasets": custom_drift_datasets,
        "distribution_plot_path": os.path.abspath(plot_output_path),
        "sliding_window_analysis": {
            "summary": sliding_summary,
            "top_users": sliding_top_users,
            "window_size": args.window_size,
            "step_size": args.step_size,
            "comparison_mode": args.comparison_mode,
            "min_window_gap": args.min_window_gap if args.comparison_mode == "cross_window" else None,
        },
    }

    save_json(report, args.output_path)

    print(f"Input file: {os.path.abspath(args.input_path)}")
    print(f"Task: {args.task_name}")
    print(f"Semantic model: {resolved_model_path}")
    print(f"Eligible users with profile length >= {args.profile_num}: {len(filtered_users)}")
    print(f"Embedding weights: text={args.text_weight}, gold={args.gold_weight}, style={args.style_weight}")
    if args.style_mode:
        print(f"Style mode: {args.style_mode}")
    print(f"Distribution step: {args.distribution_step}")
    print("")
    print("[Sliding Window Analysis]")
    print(f"Window size: {args.window_size}, step size: {args.step_size}")
    print(f"Comparison mode: {args.comparison_mode}")
    if args.comparison_mode == "cross_window":
        print(f"Minimum window gap: {args.min_window_gap}")
    print(
        "Users above threshold "
        f"{args.sliding_threshold}: {sliding_summary['num_above_threshold']}/{sliding_summary['num_users']}"
    )
    print(f"Mean max semantic cosine distance: {sliding_summary['mean_max_semantic_cosine_distance']:.4f}")
    print(f"Max max semantic cosine distance: {sliding_summary['max_max_semantic_cosine_distance']:.4f}")
    print("")
    print("[Drift Groups]")
    print(f"Total users: {len(sliding_results)}")
    for group in drift_groups:
        print(f"{group['range']}: {group['count']} users ({group['percentage']:.2f}%)")
    if args.write_custom_drifts:
        print("")
        print("[Custom Drift Datasets]")
        print(f"Output dir: {os.path.abspath(args.custom_drift_dir)}")
        print(f"Score range: {args.custom_drift_min_score:.1f}-{args.custom_drift_max_score:.1f}")
        for item in custom_drift_datasets:
            print(f"{os.path.basename(item['path'])}: {item['count']} users ({item['percentage']:.2f}%)")
    print("")
    print(f"Saved report to: {os.path.abspath(args.output_path)}")
    print(f"Saved plot to: {os.path.abspath(plot_output_path)}")


if __name__ == "__main__":
    main()
