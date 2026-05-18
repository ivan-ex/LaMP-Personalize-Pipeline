from evaluation import LaMPEvaluation
import argparse
import json
import warnings
import os
import re

warnings.filterwarnings('ignore')


TASK_NAME_TO_ID = {
    "citation": "LaMP_1",
    "movie_tagging": "LaMP_2M",
    "news_categorize": "LaMP_2N",
    "news_headline": "LaMP_4",
    "product_rating": "LaMP_3",
    "scholarly_title": "LaMP_5",
    "tweet_paraphrase": "LaMP_7",
}


CLASSIFICATION_TASKS = {"LaMP_1", "LaMP_2M", "LaMP_2N"}
REGRESSION_TASKS = {"LaMP_3"}
GENERATION_TASKS = {"LaMP_4", "LaMP_5", "LaMP_7"}

METHOD_SUFFIXES = {
    "base": "",
    "ewc": "-ewc",
    "lwf": "-lwf",
    "ewc-lwf": "-ewc-lwf",
}


def parse_distance_ratio_tag(drift_tag):
    drift_tag = str(drift_tag)
    match = re.fullmatch(r"(near|mid|far)_(0|25|50|75|100)", drift_tag)
    if not match:
        raise ValueError(
            f"Unsupported --drift_tag={drift_tag}. Expected format {{distance}}_{{ratio}}, "
            "where distance is near/mid/far and ratio is one of 0/25/50/75/100."
        )
    return match.group(1), int(match.group(2))


def validate_drift_tag(drift_tag):
    drift_tag = str(drift_tag)
    if re.fullmatch(r"(near|mid|far)_(0|25|50|75|100)", drift_tag):
        return drift_tag
    if re.fullmatch(r"[A-Za-z0-9_.-]+", drift_tag):
        return drift_tag
    raise ValueError(
        f"Unsupported --drift_tag={drift_tag}. Use distance-ratio tags like near_50 or custom tags with letters/numbers/._-."
    )


def get_metric_family(task_name):
    if task_name in CLASSIFICATION_TASKS:
        return "classification"
    if task_name in REGRESSION_TASKS:
        return "regression"
    if task_name in GENERATION_TASKS:
        return "generation"
    raise ValueError(f"Unsupported task name: {task_name}")


def get_expected_metric_keys(task_name):
    metric_family = get_metric_family(task_name)
    if metric_family == "classification":
        return ["accuracy", "f1"]
    if metric_family == "regression":
        return ["MAE", "RMSE"]
    return ["rouge-1", "rouge-L"]


def get_golds_json_candidates(task_data_name):
    candidates = []
    train_outputs = f'./data/{task_data_name}/all_user_golds.json'
    if os.path.exists(train_outputs):
        candidates.append(train_outputs)

    if task_data_name == "tweet_paraphrase":
        candidates.append(f'./data/{task_data_name}/user_more_100_history_label.json')
    else:
        candidates.append(f'./data/{task_data_name}/user_top_100_history_label.json')

    return candidates


def get_golds_json_path(task_data_name, golds_source="auto"):
    candidates = get_golds_json_candidates(task_data_name)
    if golds_source == "auto":
        return candidates[0]
    if golds_source == "train_outputs":
        target = f'./data/{task_data_name}/all_user_golds.json'
        if not os.path.exists(target):
            raise FileNotFoundError(f"Cannot find train_outputs gold file: {target}")
        return target
    if golds_source == "history_labels":
        return candidates[-1]
    raise ValueError(f"Unsupported golds_source: {golds_source}")


def parse_prediction_filename(preds_json_path):
    filename = os.path.basename(preds_json_path)
    if not filename.endswith(".json"):
        raise ValueError(f"Prediction file must be a .json file: {preds_json_path}")

    stem = filename[:-5]
    query_match = re.search(r"-(query(?:_[od])?)$", stem)
    if not query_match:
        raise ValueError(f"Cannot parse query tag from prediction filename: {filename}")
    query_tag = query_match.group(1)
    prefix = stem[:query_match.start()]

    profile_suffix = False
    if prefix.endswith("-profile"):
        profile_suffix = True
        prefix = prefix[:-len("-profile")]

    drift_match = re.search(r"-drift_(.+)$", prefix)
    if not drift_match:
        raise ValueError(f"Cannot parse drift_tag from prediction filename: {filename}")
    drift_tag = drift_match.group(1)
    left = prefix[:drift_match.start()]

    baseline_match = re.fullmatch(r"output-OPPU-baseline-k(\d+)-([A-Za-z0-9_]+)", left)
    if baseline_match:
        return {
            "method_name": "base",
            "k": int(baseline_match.group(1)),
            "task_data_name": baseline_match.group(2),
            "model_name": None,
            "drift_tag": drift_tag,
            "query_tag": query_tag,
            "profile_suffix": profile_suffix,
        }

    method_match = re.fullmatch(
        r"output-OPPU-k(\d+)-([A-Za-z0-9_]+)-(.+?)(?:-(ewc|lwf|ewc-lwf))?$",
        left,
    )
    if not method_match:
        raise ValueError(f"Cannot parse prediction filename: {filename}")

    inferred_method = method_match.group(4) or "base"
    return {
        "method_name": inferred_method,
        "k": int(method_match.group(1)),
        "task_data_name": method_match.group(2),
        "model_name": method_match.group(3),
        "drift_tag": drift_tag,
        "query_tag": query_tag,
        "profile_suffix": profile_suffix,
    }


def build_default_paths(task_data_name, task_name, drift_tag, model_name, k, query_tag, method_name, golds_source):
    validate_drift_tag(drift_tag)
    golds_json = get_golds_json_path(task_data_name, golds_source=golds_source)
    method_suffix = METHOD_SUFFIXES[method_name]
    drift_output_dir = f'./output/{task_data_name}/drift_{drift_tag}'
    if method_name == "base":
        baseline_preds_json = os.path.join(
            drift_output_dir,
            f'output-OPPU-baseline-k{k}-{task_data_name}-drift_{drift_tag}-{query_tag}.json',
        )
        legacy_preds_json = os.path.join(
            drift_output_dir,
            f'output-OPPU-k{k}-{task_data_name}-{model_name}-drift_{drift_tag}-{query_tag}.json',
        )
        legacy_flat_baseline = f'./output/{task_data_name}/output-OPPU-baseline-k{k}-{task_data_name}-drift_{drift_tag}-{query_tag}.json'
        legacy_flat_model = f'./output/{task_data_name}/output-OPPU-k{k}-{task_data_name}-{model_name}-drift_{drift_tag}-{query_tag}.json'
        if os.path.exists(baseline_preds_json):
            preds_json = baseline_preds_json
        elif os.path.exists(legacy_preds_json):
            preds_json = legacy_preds_json
        elif os.path.exists(legacy_flat_baseline):
            preds_json = legacy_flat_baseline
        else:
            preds_json = legacy_flat_model
    else:
        drift_preds_json = os.path.join(
            drift_output_dir,
            f'output-OPPU-k{k}-{task_data_name}-{model_name}{method_suffix}-drift_{drift_tag}-{query_tag}.json',
        )
        legacy_flat_preds_json = f'./output/{task_data_name}/output-OPPU-k{k}-{task_data_name}-{model_name}{method_suffix}-drift_{drift_tag}-{query_tag}.json'
        preds_json = drift_preds_json if os.path.exists(drift_preds_json) else legacy_flat_preds_json
    output_file = f'./result/strict_compare/{task_data_name}/drift_{drift_tag}/{method_name}-{query_tag}.json'
    return golds_json, preds_json, task_name, output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--drift', type=int, default=None, help='Deprecated. Use --drift_tag {distance}_{ratio} only.')
    parser.add_argument('--drift_tag', default=None, help='Dataset tag such as near_50 or drift_top10')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--query_tag', default=None, choices=['query'])
    parser.add_argument('--method_name', default=None, choices=list(METHOD_SUFFIXES.keys()), help='Name used in the evaluation result filename')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument(
        '--task_data_name',
        default=None,
        choices=list(TASK_NAME_TO_ID.keys()),
        help='数据目录名，例如 movie_tagging'
    )
    parser.add_argument(
        '--model_name',
        default=None,
        help='预测文件名中的模型标识'
    )
    parser.add_argument(
        '--preds_json',
        default=None,
        help='Optional explicit prediction json path. If provided, method/query/drift/task can be inferred from filename.'
    )
    parser.add_argument(
        '--golds_source',
        default='auto',
        choices=['auto', 'train_outputs', 'history_labels'],
        help='gold 文件选择策略；auto 优先使用 all_user_golds.json，找不到再回退到历史 label 文件'
    )
    parser.add_argument(
        '--skip_missing_gold_ids',
        action='store_true',
        help='评测时跳过在 gold 文件中找不到 id 的预测项'
    )
    parser.add_argument("--golds_json", default=None)
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--output_file", default=None)

    args, _ = parser.parse_known_args()

    inferred = {}
    if args.preds_json:
        try:
            inferred = parse_prediction_filename(args.preds_json)
        except ValueError:
            # Allow explicit CLI arguments to bypass filename-based inference.
            inferred = {}

    task_data_name = args.task_data_name or inferred.get("task_data_name")
    drift_tag = args.drift_tag or inferred.get("drift_tag")
    query_tag = args.query_tag or inferred.get("query_tag")
    method_name = args.method_name or inferred.get("method_name") or "base"
    model_name = args.model_name or inferred.get("model_name") or 'llama2_7b_hf'
    k_value = args.k if args.k is not None else inferred.get("k", 0)

    if args.golds_json and args.task_name and args.output_file:
        golds_json_default = args.golds_json
        preds_json_default = args.preds_json
        task_name_default = args.task_name
        output_file_default = args.output_file
    else:
        if not task_data_name or not drift_tag or not query_tag:
            raise ValueError(
                "Missing required evaluation metadata. Provide --golds_json, --task_name, and --output_file explicitly, "
                "or pass --preds_json / --task_data_name / --drift_tag / --query_tag so defaults can be inferred."
            )

        validate_drift_tag(drift_tag)
        if re.fullmatch(r"(near|mid|far)_(0|25|50|75|100)", str(drift_tag)):
            distance_level, ratio = parse_distance_ratio_tag(drift_tag)
            if args.drift is not None and args.drift != ratio:
                raise ValueError(
                    f"Inconsistent arguments: --drift={args.drift} but --drift_tag={drift_tag} implies ratio={ratio}."
                )

        default_task_name = TASK_NAME_TO_ID[task_data_name]
        golds_json_default, preds_json_default, task_name_default, output_file_default = build_default_paths(
            task_data_name,
            default_task_name,
            drift_tag,
            model_name,
            k_value,
            query_tag,
            method_name,
            args.golds_source,
        )

        if args.preds_json:
            preds_json_default = args.preds_json
            output_file_default = os.path.join(
                ".",
                "result",
                "strict_compare",
                task_data_name,
                f"drift_{drift_tag}",
                f"{method_name}-{query_tag}.json",
            )

    parser.set_defaults(
        golds_json=golds_json_default,
        preds_json=preds_json_default,
        task_name=task_name_default,
        output_file=output_file_default,
    )

    opts = parser.parse_args()

    expected_metric_keys = get_expected_metric_keys(opts.task_name)
    evaluator = LaMPEvaluation(
        single_gold_json_file_addr=opts.golds_json,
        skip_missing_gold_ids=opts.skip_missing_gold_ids,
    )
    results = evaluator.evaluate_task(opts.preds_json, opts.task_name)
    missing_metric_keys = [key for key in expected_metric_keys if key not in results]
    if missing_metric_keys:
        raise ValueError(
            f"Evaluation result for task {opts.task_name} is missing expected metrics: {missing_metric_keys}. "
            f"Got keys: {sorted(results.keys())}"
        )

    ordered_results = {key: results[key] for key in expected_metric_keys}
    os.makedirs(os.path.dirname(opts.output_file), exist_ok=True)
    with open(opts.output_file, "w") as file:
        json.dump(ordered_results, file)
    print(f"Evaluation result saved to: {opts.output_file}")
