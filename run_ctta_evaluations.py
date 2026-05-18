import argparse
import json
import os
import re

from eval.evaluation import LaMPEvaluation, compute_metrics_for_task


TASK_NAME_TO_ID = {
    "citation": "LaMP_1",
    "movie_tagging": "LaMP_2M",
    "news_categorize": "LaMP_2N",
    "news_headline": "LaMP_4",
    "product_rating": "LaMP_3",
    "scholarly_title": "LaMP_5",
    "tweet_paraphrase": "LaMP_7",
}

TASK_ID_TO_NAME = {value: key for key, value in TASK_NAME_TO_ID.items()}
CLASSIFICATION_TASKS = {"LaMP_1", "LaMP_2M", "LaMP_2N"}
REGRESSION_TASKS = {"LaMP_3"}
GENERATION_TASKS = {"LaMP_4", "LaMP_5", "LaMP_7"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def format_float_for_name(value):
    return str(value).replace(".", "p")


def normalize_dataset_tag(tag):
    if tag is None:
        return None
    return str(tag).strip()


def build_result_dataset_dir(tag):
    normalized_tag = normalize_dataset_tag(tag)
    if not normalized_tag:
        return "unknown"
    return normalized_tag


def build_output_dataset_dir(tag):
    normalized_tag = normalize_dataset_tag(tag)
    if not normalized_tag:
        return "unknown"
    if normalized_tag.startswith("drift_"):
        return normalized_tag
    return f"drift_{normalized_tag}"


def build_ctta_run_name(
    task_name,
    model_name,
    output_tag,
    k,
    add_profile=False,
    drift_detector="semantic",
    history_missing_lora_detector=False,
):
    run_name = f"output-OPPU-k{k}-{task_name}-{os.path.basename(model_name.rstrip('/'))}-{output_tag}"
    if drift_detector != "semantic" or history_missing_lora_detector:
        run_name += f"-detector_{drift_detector}"
        if history_missing_lora_detector:
            run_name += "-missing_lora"
    if add_profile:
        run_name += "-profile"
    return run_name


def infer_drift_tag_from_train_data_path(train_data_path):
    if not train_data_path:
        return None
    stem = os.path.splitext(os.path.basename(train_data_path))[0]
    return stem[len("drift_train_"):] if stem.startswith("drift_train_") else stem


def infer_task_name_from_payload(payload):
    task_id = payload.get("task")
    return TASK_ID_TO_NAME.get(task_id)


def infer_model_name_from_payload(payload):
    model_value = payload.get("model")
    if not model_value:
        return None
    return os.path.basename(str(model_value).rstrip("/"))


def infer_drift_tag_from_artifact(path, payload=None):
    if payload:
        if payload.get("drift_tag"):
            return payload["drift_tag"]
        inferred = infer_drift_tag_from_train_data_path(payload.get("train_data_path"))
        if inferred:
            return inferred

    parent_dir = os.path.basename(os.path.dirname(path))
    if parent_dir.startswith("drift_"):
        return parent_dir[len("drift_"):]

    filename = os.path.basename(path)
    match = re.search(
        r"-drift_(.+?)(?:-profile)?-(?:heldout_profile(?:-summary)?|query|drift_log)\.json$",
        filename,
    )
    if match:
        return match.group(1)
    return None


def strip_known_output_suffix(path):
    suffixes = [
        "-heldout_profile.json",
        "-heldout_profile-summary.json",
        "-query.json",
        "-drift_log.json",
    ]
    for suffix in suffixes:
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return None


def build_output_name(args, drift_tag):
    return build_ctta_run_name(
        task_name=args.task_name,
        model_name=args.model_name,
        output_tag=args.output_tag,
        k=args.k,
        add_profile=args.add_profile,
        drift_detector=args.drift_detector,
        history_missing_lora_detector=args.history_missing_lora_detector,
    )


def build_legacy_output_name(args, drift_tag):
    output_name = (
        f"output-OPPU-k{args.k}-{args.task_name}-{os.path.basename(args.model_name.rstrip('/'))}-{args.output_tag}"
        f"-thr{format_float_for_name(args.ctta_threshold)}"
        f"-w{args.ctta_window_size}"
        f"-split{args.profile_split_mode}"
    )
    if args.profile_split_mode == "metadata":
        output_name += "metadata"
    elif args.profile_split_mode == "ratio":
        output_name += format_float_for_name(args.profile_split_ratio)
    else:
        if args.profile_split_count is None:
            raise ValueError("--profile_split_count must be set when --profile_split_mode=count")
        output_name += str(args.profile_split_count)
    output_name += f"-drift_{drift_tag}"
    if args.add_profile:
        output_name += "-profile"
    return output_name


def build_default_artifact_paths(repo_root, args, drift_tag):
    output_name = build_output_name(args, drift_tag)
    legacy_output_name = build_legacy_output_name(args, drift_tag)
    new_output_dir = os.path.join(repo_root, "output", args.task_name, build_output_dataset_dir(drift_tag))
    new_prefix = os.path.join(new_output_dir, output_name)
    old_verbose_prefix = os.path.join(new_output_dir, legacy_output_name)
    legacy_flat_prefix = os.path.join(repo_root, "output", args.task_name, legacy_output_name)

    def pick_path(*candidates):
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    heldout = pick_path(
        new_prefix + "-heldout_profile.json",
        old_verbose_prefix + "-heldout_profile.json",
        legacy_flat_prefix + "-heldout_profile.json",
    )
    query = pick_path(
        new_prefix + "-query.json",
        old_verbose_prefix + "-query.json",
        legacy_flat_prefix + "-query.json",
    )
    drift_log = pick_path(
        new_prefix + "-drift_log.json",
        old_verbose_prefix + "-drift_log.json",
        legacy_flat_prefix + "-drift_log.json",
    )

    resolved_prefix = strip_known_output_suffix(heldout)
    if resolved_prefix is None:
        resolved_prefix = strip_known_output_suffix(query)
    if resolved_prefix is None:
        resolved_prefix = strip_known_output_suffix(drift_log)
    if resolved_prefix is None:
        resolved_prefix = new_prefix

    return {
        "prefix": resolved_prefix,
        "heldout": heldout,
        "query": query,
        "drift_log": drift_log,
    }


def get_metric_family(task_id):
    if task_id in CLASSIFICATION_TASKS:
        return "classification"
    if task_id in REGRESSION_TASKS:
        return "regression"
    if task_id in GENERATION_TASKS:
        return "generation"
    raise ValueError(f"Unsupported task id: {task_id}")


def supports_heldout_profile_eval(task_name):
    return task_name != "tweet_paraphrase"


def get_query_golds_json(task_name):
    candidates = [os.path.join(".", "data", task_name, "all_user_golds.json")]
    if task_name == "tweet_paraphrase":
        candidates.append(os.path.join(".", "data", task_name, "user_more_100_history_label.json"))
    else:
        candidates.append(os.path.join(".", "data", task_name, "user_top_100_history_label.json"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Cannot find a gold file for task={task_name}. Checked: {candidates}"
    )


def evaluate_inline_profile_predictions(preds_json_path):
    payload = load_json(preds_json_path)
    predictions = payload["golds"]
    task_id = payload["task"]
    metric_family = get_metric_family(task_id)

    if metric_family == "generation":
        pred_values = [item.get("output", "") for item in predictions]
        gold_values = [item.get("gold", "") for item in predictions]
        pred_field = "output"
        gold_field = "gold"
    else:
        pred_values = [item.get("prediction_normalized", item.get("output", "")) for item in predictions]
        gold_values = [item.get("gold_normalized", item.get("gold", "")) for item in predictions]
        pred_field = "prediction_normalized"
        gold_field = "gold_normalized"

    metrics = compute_metrics_for_task(task_id, pred_values, gold_values)
    metrics.update(
        {
            "num_predictions": len(predictions),
            "prediction_field": pred_field,
            "gold_field": gold_field,
        }
    )

    if predictions and all("correct" in item for item in predictions):
        metrics["accuracy_from_correct_flag"] = sum(int(item["correct"]) for item in predictions) / len(predictions)

    return metrics, payload


def evaluate_query_predictions(preds_json_path, task_name):
    payload = load_json(preds_json_path)
    golds_json = get_query_golds_json(task_name)
    evaluator = LaMPEvaluation(single_gold_json_file_addr=golds_json)
    metrics = evaluator.evaluate_task(preds_json_path, TASK_NAME_TO_ID[task_name])
    metrics.update(
        {
            "num_predictions": len(payload["golds"]),
            "golds_json": os.path.abspath(golds_json),
        }
    )
    return metrics, payload


def summarize_drift_log(log_json_path):
    logs = load_json(log_json_path)
    total_users = len(logs)
    users_with_events = sum(1 for user in logs if user.get("events"))
    event_counts = [len(user.get("events", [])) for user in logs]
    all_events = [event for user in logs for event in user.get("events", [])]
    applied_events = [
        event
        for event in all_events
        if event.get("update_applied", True)
    ]
    users_with_updates = sum(
        1
        for user in logs
        if any(event.get("update_applied", True) for event in user.get("events", []))
    )
    update_counts = [
        sum(1 for event in user.get("events", []) if event.get("update_applied", True))
        for user in logs
    ]
    drift_scores = [float(event["drift_score"]) for event in all_events if event.get("drift_score") is not None]
    semantic_drift_scores = [
        float(event["semantic_drift_score"])
        for event in all_events
        if event.get("semantic_drift_score") is not None
    ]
    lora_drift_scores = [
        float(event["lora_drift_score"])
        for event in all_events
        if event.get("lora_drift_score") is not None
    ]
    semantic_trigger_count = sum(1 for event in all_events if event.get("semantic_triggered"))
    lora_trigger_count = sum(1 for event in all_events if event.get("lora_triggered"))
    history_missing_lora_event_count = sum(
        1
        for event in all_events
        if event.get("history_missing_lora_detector") and event.get("lora_drift_score") is not None
    )
    segment_sizes = [int(event["segment_size"]) for event in applied_events if event.get("segment_size") is not None]
    hidden_lengths = [int(user.get("hidden_suffix_len", 0)) for user in logs]
    visible_lengths = [int(user.get("visible_prefix_len", 0)) for user in logs]
    warmup_lengths = [int(user.get("warmup_end", 0)) for user in logs]
    total_profile_predictions = sum(int(user.get("num_profile_predictions", 0)) for user in logs)
    total_profile_correct = sum(int(user.get("num_correct", 0)) for user in logs)
    profile_accuracy = None
    if total_profile_predictions:
        profile_accuracy = total_profile_correct / total_profile_predictions

    return {
        "num_users": total_users,
        "users_with_drift_detections": users_with_events,
        "drift_detection_user_rate": (users_with_events / total_users) if total_users else 0.0,
        "total_drift_detections": len(all_events),
        "avg_drift_detections_per_user": (sum(event_counts) / total_users) if total_users else 0.0,
        "max_drift_detections_for_one_user": max(event_counts) if event_counts else 0,
        "users_with_drift_updates": users_with_updates,
        "drift_update_user_rate": (users_with_updates / total_users) if total_users else 0.0,
        "total_drift_updates": len(applied_events),
        "avg_drift_updates_per_user": (sum(update_counts) / total_users) if total_users else 0.0,
        "avg_drift_updates_per_updated_user": (sum(update_counts) / users_with_updates) if users_with_updates else 0.0,
        "max_drift_updates_for_one_user": max(update_counts) if update_counts else 0,
        "avg_drift_score": (sum(drift_scores) / len(drift_scores)) if drift_scores else 0.0,
        "max_drift_score": max(drift_scores) if drift_scores else 0.0,
        "avg_semantic_drift_score": (sum(semantic_drift_scores) / len(semantic_drift_scores)) if semantic_drift_scores else 0.0,
        "max_semantic_drift_score": max(semantic_drift_scores) if semantic_drift_scores else 0.0,
        "avg_lora_drift_score": (sum(lora_drift_scores) / len(lora_drift_scores)) if lora_drift_scores else 0.0,
        "max_lora_drift_score": max(lora_drift_scores) if lora_drift_scores else 0.0,
        "semantic_trigger_count": semantic_trigger_count,
        "lora_trigger_count": lora_trigger_count,
        "history_missing_lora_event_count": history_missing_lora_event_count,
        "avg_update_segment_size": (sum(segment_sizes) / len(segment_sizes)) if segment_sizes else 0.0,
        "avg_hidden_suffix_len": (sum(hidden_lengths) / len(hidden_lengths)) if hidden_lengths else 0.0,
        "avg_visible_prefix_len": (sum(visible_lengths) / len(visible_lengths)) if visible_lengths else 0.0,
        "avg_warmup_len": (sum(warmup_lengths) / len(warmup_lengths)) if warmup_lengths else 0.0,
        "profile_accuracy_from_log": profile_accuracy,
        "num_profile_predictions_from_log": total_profile_predictions,
    }


def resolve_paths(repo_root, args):
    explicit_paths = [
        args.heldout_preds_json,
        args.query_preds_json,
        args.drift_log_json,
    ]
    prefix = None
    for path in explicit_paths:
        if not path:
            continue
        stripped = strip_known_output_suffix(path)
        if stripped:
            prefix = stripped
            break

    if prefix:
        return {
            "prefix": prefix,
            "heldout": args.heldout_preds_json or prefix + "-heldout_profile.json",
            "query": args.query_preds_json or prefix + "-query.json",
            "drift_log": args.drift_log_json or prefix + "-drift_log.json",
        }

    first_explicit_path = next((path for path in explicit_paths if path), None)
    if first_explicit_path:
        prefix = os.path.splitext(first_explicit_path)[0]
        return {
            "prefix": prefix,
            "heldout": args.heldout_preds_json or prefix + "-heldout_profile.json",
            "query": args.query_preds_json or prefix + "-query.json",
            "drift_log": args.drift_log_json or prefix + "-drift_log.json",
        }

    drift_tag = args.drift_tag or infer_drift_tag_from_train_data_path(args.train_data_path)
    if not args.task_name or not drift_tag:
        raise ValueError(
            "Cannot infer CTTA artifact paths. Provide explicit json paths or pass --task_name with --drift_tag/--train_data_path."
        )
    return build_default_artifact_paths(repo_root, args, drift_tag)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CTTA outputs on heldout profile streams and final queries.")
    parser.add_argument("--task_name", default=None, choices=list(TASK_NAME_TO_ID.keys()))
    parser.add_argument("--train_data_path", default=None)
    parser.add_argument("--drift_tag", default=None)
    parser.add_argument("--model_name", default="llama2_7b_hf")
    parser.add_argument("--output_tag", default="ctta", choices=["ctta", "base"])
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--ctta_threshold", type=float, default=0.35)
    parser.add_argument("--ctta_window_size", type=int, default=8)
    parser.add_argument("--drift_detector", default="semantic", choices=["semantic", "lora", "hybrid"])
    parser.add_argument("--history_missing_lora_detector", action="store_true")
    parser.add_argument("--profile_split_mode", default="metadata", choices=["metadata", "ratio", "count"])
    parser.add_argument("--profile_split_ratio", type=float, default=0.5)
    parser.add_argument("--profile_split_count", type=int, default=None)
    parser.add_argument("--add_profile", action="store_true")
    parser.add_argument("--heldout_preds_json", default=None)
    parser.add_argument("--query_preds_json", default=None)
    parser.add_argument("--drift_log_json", default=None)
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(repo_root)

    paths = resolve_paths(repo_root, args)

    heldout_payload = load_json(paths["heldout"]) if os.path.exists(paths["heldout"]) else None
    query_payload = load_json(paths["query"]) if os.path.exists(paths["query"]) else None

    task_name = args.task_name
    if task_name is None:
        for payload in (heldout_payload, query_payload):
            if payload:
                task_name = infer_task_name_from_payload(payload)
                if task_name:
                    break
    if task_name is None:
        raise ValueError("Cannot infer task_name. Please pass --task_name or provide a CTTA prediction file with a task id.")

    drift_tag = normalize_dataset_tag(args.drift_tag)
    if drift_tag is None:
        drift_tag = normalize_dataset_tag(infer_drift_tag_from_train_data_path(args.train_data_path))
    if drift_tag is None:
        for path, payload in ((paths["heldout"], heldout_payload), (paths["query"], query_payload)):
            if os.path.exists(path):
                drift_tag = normalize_dataset_tag(infer_drift_tag_from_artifact(path, payload))
                if drift_tag:
                    break
    if drift_tag is None:
        drift_tag = "unknown"

    preferred_model_name = args.model_name
    for payload in (query_payload, heldout_payload):
        inferred_model_name = infer_model_name_from_payload(payload) if payload else None
        if inferred_model_name:
            preferred_model_name = inferred_model_name
            break

    run_name = build_ctta_run_name(
        task_name=task_name,
        model_name=preferred_model_name,
        output_tag=args.output_tag,
        k=args.k,
        add_profile=args.add_profile,
        drift_detector=args.drift_detector,
        history_missing_lora_detector=args.history_missing_lora_detector,
    )
    result_dataset_dir = build_result_dataset_dir(drift_tag)
    result_dir = args.result_dir or os.path.join(repo_root, "result", "ctta", task_name, result_dataset_dir)
    os.makedirs(result_dir, exist_ok=True)

    summary = {
        "task_name": task_name,
        "task_id": TASK_NAME_TO_ID[task_name],
        "drift_tag": drift_tag,
        "run_name": run_name,
        "adaptation_mode": args.output_tag,
        "artifacts": {key: os.path.abspath(value) for key, value in paths.items()},
        "heldout_profile": None,
        "query": None,
        "drift": None,
    }

    if supports_heldout_profile_eval(task_name) and os.path.exists(paths["heldout"]):
        heldout_metrics, heldout_payload = evaluate_inline_profile_predictions(paths["heldout"])
        summary["heldout_profile"] = heldout_metrics
    elif not supports_heldout_profile_eval(task_name):
        print(f"Info: heldout profile evaluation is not defined for {task_name}; skipped.")
    else:
        print(f"Warning: heldout profile prediction file not found, skipped: {paths['heldout']}")

    if os.path.exists(paths["query"]):
        query_metrics, _ = evaluate_query_predictions(paths["query"], task_name)
        summary["query"] = query_metrics
    else:
        print(f"Warning: query prediction file not found, skipped: {paths['query']}")

    if os.path.exists(paths["drift_log"]):
        drift_metrics = summarize_drift_log(paths["drift_log"])
        summary["drift"] = drift_metrics
    else:
        print(f"Warning: drift log file not found, skipped: {paths['drift_log']}")

    summary_json = args.summary_json or os.path.join(result_dir, f"{run_name}-summary.json")
    ensure_parent_dir(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"CTTA evaluation summary saved to: {summary_json}")


if __name__ == "__main__":
    main()
