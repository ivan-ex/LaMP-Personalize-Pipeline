import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DRIFT_RANGES = ["0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6"]
DEFAULT_MODES = ["base", "ctta"]
CLASSIFICATION_PLOT_METRICS = ["query_accuracy", "query_f1", "heldout_accuracy", "heldout_f1"]
GENERATION_PLOT_METRICS = ["query_rouge_1", "query_rouge_L"]
DEFAULT_PLOT_METRICS = CLASSIFICATION_PLOT_METRICS
SUMMARY_FIELDNAMES = [
    "drift_range",
    "drift_midpoint",
    "adaptation_mode",
    "drift_detector",
    "history_missing_lora_detector",
    "profile_split_ratio",
    "train_data_path",
    "summary_json",
    "heldout_accuracy",
    "heldout_f1",
    "heldout_rouge_1",
    "heldout_rouge_L",
    "query_accuracy",
    "query_f1",
    "query_rouge_1",
    "query_rouge_L",
    "profile_accuracy_from_log",
    "users_with_drift_detections",
    "drift_detection_user_rate",
    "total_drift_detections",
    "users_with_drift_updates",
    "drift_update_user_rate",
    "total_drift_updates",
    "avg_drift_score",
    "max_drift_score",
    "avg_semantic_drift_score",
    "max_semantic_drift_score",
    "avg_lora_drift_score",
    "max_lora_drift_score",
    "semantic_trigger_count",
    "lora_trigger_count",
    "history_missing_lora_event_count",
]

METHOD_STYLES = {
    "base": {"color": "#DD8452", "marker": "o"},
    "ctta": {"color": "#4C72B0", "marker": "s"},
}

METRIC_LABELS = {
    "heldout_accuracy": "Heldout Profile Accuracy",
    "heldout_f1": "Heldout Profile F1",
    "heldout_rouge_1": "Heldout Profile ROUGE-1",
    "heldout_rouge_L": "Heldout Profile ROUGE-L",
    "query_accuracy": "Query Accuracy",
    "query_f1": "Query F1",
    "query_rouge_1": "Query ROUGE-1",
    "query_rouge_L": "Query ROUGE-L",
    "profile_accuracy_from_log": "Stream Profile Accuracy",
    "drift_detection_user_rate": "Drift Detection User Rate",
    "drift_update_user_rate": "Drift Update User Rate",
    "avg_drift_score": "Average Drift Score",
}


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(cmd, cwd, env, dry_run):
    cuda = env.get("CUDA_VISIBLE_DEVICES")
    prefix = f"CUDA_VISIBLE_DEVICES={cuda} " if cuda is not None else ""
    print("Running:", prefix + quote_cmd(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def cuda_visible_devices_for_mode(args, mode):
    if mode == "ctta":
        return args.ctta_cuda_visible_devices
    if mode == "base":
        return args.base_cuda_visible_devices
    return args.cuda_visible_devices


def build_mode_env(args, mode):
    env = os.environ.copy()
    cuda_visible_devices = cuda_visible_devices_for_mode(args, mode)
    if cuda_visible_devices is not None and str(cuda_visible_devices).lower() == "none":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def normalize_output_dataset_dir(drift_tag):
    return drift_tag if drift_tag.startswith("drift_") else f"drift_{drift_tag}"


def format_ratio_folder(value):
    return format(float(value), ".12g")


def model_leaf(model_name):
    return os.path.basename(str(model_name).rstrip("/"))


def build_run_name(task_name, model_name, mode, k, add_profile, drift_detector="semantic", history_missing_lora_detector=False):
    run_name = f"output-OPPU-k{k}-{task_name}-{model_leaf(model_name)}-{mode}"
    if drift_detector != "semantic" or history_missing_lora_detector:
        run_name += f"-detector_{drift_detector}"
        if history_missing_lora_detector:
            run_name += "-missing_lora"
    if add_profile:
        run_name += "-profile"
    return run_name


def build_paths(repo_root, args, drift_range, mode):
    drift_tag = f"{args.dataset_prefix}_{drift_range}"
    ratio_folder = format_ratio_folder(args.profile_split_ratio)
    train_data_path = os.path.join(args.dataset_dir, f"{drift_tag}.json")
    output_drift_tag = os.path.join(normalize_output_dataset_dir(drift_tag), ratio_folder)
    run_name = build_run_name(
        args.task_name,
        args.model_name,
        mode,
        args.k,
        args.add_profile,
        args.drift_detector,
        args.history_missing_lora_detector,
    )
    output_dir = os.path.join(repo_root, "output", args.task_name, output_drift_tag)
    prefix = os.path.join(output_dir, run_name)
    detector_suffix = ""
    if args.drift_detector != "semantic" or args.history_missing_lora_detector:
        detector_suffix = f"-detector_{args.drift_detector}"
        if args.history_missing_lora_detector:
            detector_suffix += "-missing_lora"
    summary_json = os.path.join(args.result_dir, f"{drift_tag}-{mode}{detector_suffix}-summary.json")
    return {
        "drift_tag": drift_tag,
        "output_drift_tag": output_drift_tag,
        "profile_split_ratio": ratio_folder,
        "train_data_path": train_data_path,
        "heldout": prefix + "-heldout_profile.json",
        "query": prefix + "-query.json",
        "drift_log": prefix + "-drift_log.json",
        "summary_json": summary_json,
    }


def append_optional_value(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def build_ctta_command(args, paths, mode):
    cmd = [
        args.python_bin,
        args.ctta_script,
        "--task_name",
        args.task_name,
        "--train_data_path",
        paths["train_data_path"],
        "--drift_tag",
        paths["output_drift_tag"],
        "--adaptation_mode",
        mode,
        "--output_tag",
        mode,
        "--model_name",
        args.model_name,
        "--k",
        str(args.k),
    ]
    append_optional_value(cmd, "--task_lora", args.task_lora)
    append_optional_value(cmd, "--batch_size", args.batch_size)
    append_optional_value(cmd, "--infer_batch_size", args.infer_batch_size)
    append_optional_value(cmd, "--max_epoch", args.max_epoch)
    append_optional_value(cmd, "--cut_off", args.cut_off)
    append_optional_value(cmd, "--ctta_threshold", args.ctta_threshold)
    append_optional_value(cmd, "--ctta_window_size", args.ctta_window_size)
    append_optional_value(cmd, "--ctta_update_min_examples", args.ctta_update_min_examples)
    if args.drift_detector != "semantic" or args.history_missing_lora_detector:
        append_optional_value(cmd, "--drift_detector", args.drift_detector)
    append_optional_value(cmd, "--lora_drift_threshold", args.lora_drift_threshold)
    append_optional_value(cmd, "--lora_drift_probe_size", args.lora_drift_probe_size)
    append_optional_value(cmd, "--lora_drift_probe_epochs", args.lora_drift_probe_epochs)
    append_optional_value(cmd, "--lora_drift_probe_lr", args.lora_drift_probe_lr)
    append_optional_value(cmd, "--profile_split_ratio", args.profile_split_ratio)
    append_optional_value(cmd, "--semantic_device", args.semantic_device)
    if args.history_missing_lora_detector:
        cmd.append("--history_missing_lora_detector")
    if args.add_profile:
        cmd.append("--add_profile")
    if args.load_in_4bit:
        cmd.append("--load_in_4bit")
    if args.ctta_extra_args:
        cmd.extend(args.ctta_extra_args)
    return cmd


def build_eval_command(args, paths, mode):
    cmd = [
        args.python_bin,
        "run_ctta_evaluations.py",
        "--task_name",
        args.task_name,
        "--train_data_path",
        paths["train_data_path"],
        "--drift_tag",
        paths["drift_tag"],
        "--model_name",
        args.model_name,
        "--output_tag",
        mode,
        "--k",
        str(args.k),
        "--result_dir",
        args.result_dir,
        "--summary_json",
        paths["summary_json"],
        "--heldout_preds_json",
        paths["heldout"],
        "--query_preds_json",
        paths["query"],
        "--drift_log_json",
        paths["drift_log"],
    ]
    append_optional_value(cmd, "--ctta_threshold", args.ctta_threshold)
    append_optional_value(cmd, "--ctta_window_size", args.ctta_window_size)
    if args.drift_detector != "semantic" or args.history_missing_lora_detector:
        append_optional_value(cmd, "--drift_detector", args.drift_detector)
    append_optional_value(cmd, "--profile_split_ratio", args.profile_split_ratio)
    if args.history_missing_lora_detector:
        cmd.append("--history_missing_lora_detector")
    if args.add_profile:
        cmd.append("--add_profile")
    return cmd


def requires_heldout_profile(task_name):
    return task_name != "tweet_paraphrase"


def artifacts_exist(paths, task_name):
    required_keys = ["query", "drift_log"]
    if requires_heldout_profile(task_name):
        required_keys.append("heldout")
    return all(os.path.exists(paths[key]) for key in required_keys)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(summary, section, key):
    section_value = summary.get(section) or {}
    value = section_value.get(key)
    if value is None:
        return None
    return float(value)


def midpoint(drift_range):
    left, right = drift_range.split("-", 1)
    return (float(left) + float(right)) / 2.0


def collect_rows(repo_root, args):
    rows = []
    for drift_range in args.drift_ranges:
        for mode in args.modes:
            paths = build_paths(repo_root, args, drift_range, mode)
            if not os.path.exists(paths["summary_json"]):
                print(f"Warning: summary not found, skipped: {paths['summary_json']}")
                continue

            summary = load_json(paths["summary_json"])
            row = {
                "drift_range": drift_range,
                "drift_midpoint": midpoint(drift_range),
                "adaptation_mode": mode,
                "drift_detector": args.drift_detector,
                "history_missing_lora_detector": args.history_missing_lora_detector,
                "profile_split_ratio": paths["profile_split_ratio"],
                "train_data_path": os.path.abspath(paths["train_data_path"]),
                "summary_json": os.path.abspath(paths["summary_json"]),
                "heldout_accuracy": get_metric(summary, "heldout_profile", "accuracy"),
                "heldout_f1": get_metric(summary, "heldout_profile", "f1"),
                "heldout_rouge_1": get_metric(summary, "heldout_profile", "rouge-1"),
                "heldout_rouge_L": get_metric(summary, "heldout_profile", "rouge-L"),
                "query_accuracy": get_metric(summary, "query", "accuracy"),
                "query_f1": get_metric(summary, "query", "f1"),
                "query_rouge_1": get_metric(summary, "query", "rouge-1"),
                "query_rouge_L": get_metric(summary, "query", "rouge-L"),
                "profile_accuracy_from_log": get_metric(summary, "drift", "profile_accuracy_from_log"),
                "users_with_drift_detections": get_metric(summary, "drift", "users_with_drift_detections"),
                "drift_detection_user_rate": get_metric(summary, "drift", "drift_detection_user_rate"),
                "total_drift_detections": get_metric(summary, "drift", "total_drift_detections"),
                "users_with_drift_updates": get_metric(summary, "drift", "users_with_drift_updates"),
                "drift_update_user_rate": get_metric(summary, "drift", "drift_update_user_rate"),
                "total_drift_updates": get_metric(summary, "drift", "total_drift_updates"),
                "avg_drift_score": get_metric(summary, "drift", "avg_drift_score"),
                "max_drift_score": get_metric(summary, "drift", "max_drift_score"),
                "avg_semantic_drift_score": get_metric(summary, "drift", "avg_semantic_drift_score"),
                "max_semantic_drift_score": get_metric(summary, "drift", "max_semantic_drift_score"),
                "avg_lora_drift_score": get_metric(summary, "drift", "avg_lora_drift_score"),
                "max_lora_drift_score": get_metric(summary, "drift", "max_lora_drift_score"),
                "semantic_trigger_count": get_metric(summary, "drift", "semantic_trigger_count"),
                "lora_trigger_count": get_metric(summary, "drift", "lora_trigger_count"),
                "history_missing_lora_event_count": get_metric(summary, "drift", "history_missing_lora_event_count"),
            }
            rows.append(row)
    return rows


def write_csv(rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def value_or_none(row, metric):
    value = row.get(metric)
    return None if value in (None, "") else float(value)


def plot_rows(rows, drift_ranges, modes, metrics, output_path):
    if not rows:
        print("Warning: no rows to plot.")
        return None

    rows_by_key = {
        (row["drift_range"], row["adaptation_mode"]): row
        for row in rows
    }
    metrics_to_plot = [
        metric
        for metric in metrics
        if any(value_or_none(row, metric) is not None for row in rows)
    ]
    if not metrics_to_plot:
        print("Warning: none of the requested metrics exist in the summaries.")
        return None

    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(11, 4 * len(metrics_to_plot)), squeeze=False)
    x_positions = list(range(len(drift_ranges)))

    for ax, metric in zip(axes[:, 0], metrics_to_plot):
        for mode in modes:
            values = []
            for drift_range in drift_ranges:
                row = rows_by_key.get((drift_range, mode))
                values.append(value_or_none(row, metric) if row else None)
            if all(value is None for value in values):
                continue

            style = METHOD_STYLES.get(mode, {"color": None, "marker": "o"})
            ax.plot(
                x_positions,
                values,
                label=mode,
                linewidth=2,
                marker=style["marker"],
                color=style["color"],
            )

        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(drift_ranges)
        ax.set_xlabel("Drift Score Range")
        ax.set_ylabel("Metric Value")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OPPU_CTTA over custom drift score bins for base/ctta and plot metrics."
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--ctta_script",
        default="OPPU_CTTA.py",
        help="CTTA entrypoint to run. Defaults to the legacy script; use OPPU_CTTA_refactored.py to try the split version.",
    )
    parser.add_argument("--cuda_visible_devices", default=None, help="Fallback CUDA_VISIBLE_DEVICES for modes without a dedicated GPU setting")
    parser.add_argument("--ctta_cuda_visible_devices", default="0", help="CUDA_VISIBLE_DEVICES used by ctta runs")
    parser.add_argument("--base_cuda_visible_devices", default="1", help="CUDA_VISIBLE_DEVICES used by base runs")
    parser.add_argument("--task_name", default="movie_tagging")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--dataset_prefix", default="drift")
    parser.add_argument("--drift_ranges", nargs="+", default=DEFAULT_DRIFT_RANGES)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
    parser.add_argument("--model_name", default="/home/xuyifan/model/meta-llama/llama2_7b_hf")
    parser.add_argument("--task_lora", default=None)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--infer_batch_size", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--cut_off", type=int, default=None)
    parser.add_argument("--ctta_threshold", type=float, default=None)
    parser.add_argument("--ctta_window_size", type=int, default=None)
    parser.add_argument("--ctta_update_min_examples", type=int, default=None)
    parser.add_argument("--drift_detector", default="semantic", choices=["semantic", "lora", "hybrid"])
    parser.add_argument("--history_missing_lora_detector", action="store_true")
    parser.add_argument("--lora_drift_threshold", type=float, default=None)
    parser.add_argument("--lora_drift_probe_size", type=int, default=None)
    parser.add_argument("--lora_drift_probe_epochs", type=float, default=None)
    parser.add_argument("--lora_drift_probe_lr", type=float, default=None)
    parser.add_argument(
        "--profile_split_ratio",
        type=float,
        default=0.5,
        help="Visible profile split ratio; also used as the output/result grouping folder name.",
    )
    parser.add_argument("--semantic_device", default=None)
    parser.add_argument("--add_profile", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--plot_path", default=None)
    parser.add_argument("--plot_metrics", nargs="+", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--only_plot", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--ctta_extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments appended to every OPPU_CTTA.py call. Put this option last.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.dirname(__file__))
    if args.dataset_dir is None:
        args.dataset_dir = os.path.join(".", "data", args.task_name, "custom_drifts")
    if args.result_dir is None:
        args.result_dir = os.path.join(
            repo_root,
            "result",
            "custom_drift_ctta",
            args.task_name,
            format_ratio_folder(args.profile_split_ratio),
        )
    if args.summary_csv is None:
        args.summary_csv = os.path.join(args.result_dir, "summary.csv")
    if args.summary_json is None:
        args.summary_json = os.path.join(args.result_dir, "summary.json")
    if args.plot_path is None:
        args.plot_path = os.path.join(args.result_dir, "summary_plot.png")
    if args.plot_metrics is None:
        args.plot_metrics = GENERATION_PLOT_METRICS if args.task_name == "tweet_paraphrase" else DEFAULT_PLOT_METRICS
    return args


def validate_detector_args(args):
    advanced_detector_requested = (
        args.drift_detector != "semantic"
        or args.history_missing_lora_detector
        or args.lora_drift_threshold is not None
        or args.lora_drift_probe_size is not None
        or args.lora_drift_probe_epochs is not None
        or args.lora_drift_probe_lr is not None
    )
    if advanced_detector_requested and os.path.basename(args.ctta_script) != "OPPU_CTTA_refactored.py":
        raise ValueError(
            "LoRA/history-missing drift options are only supported by OPPU_CTTA_refactored.py. "
            "Pass --ctta_script OPPU_CTTA_refactored.py or remove the detector options."
        )


def run_mode_jobs(args, repo_root, mode):
    env = build_mode_env(args, mode)
    for drift_range in args.drift_ranges:
        paths = build_paths(repo_root, args, drift_range, mode)
        if not os.path.exists(paths["train_data_path"]):
            print(f"Warning: train data not found, skipped drift range {drift_range}: {paths['train_data_path']}")
            continue

        if args.skip_existing and artifacts_exist(paths, args.task_name):
            print(f"Skipping existing run: drift={drift_range}, mode={mode}")
        else:
            run_command(build_ctta_command(args, paths, mode), repo_root, env, args.dry_run)

        if not args.dry_run:
            run_command(build_eval_command(args, paths, mode), repo_root, env, args.dry_run)


def main():
    args = parse_args()
    validate_detector_args(args)
    repo_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(repo_root)
    os.makedirs(args.result_dir, exist_ok=True)

    if not args.only_plot:
        with ThreadPoolExecutor(max_workers=len(args.modes)) as executor:
            futures = {
                executor.submit(run_mode_jobs, args, repo_root, mode): mode
                for mode in args.modes
            }
            for future in as_completed(futures):
                future.result()

    rows = collect_rows(repo_root, args)
    write_csv(rows, args.summary_csv)
    write_json(rows, args.summary_json)
    plot_path = plot_rows(rows, args.drift_ranges, args.modes, args.plot_metrics, args.plot_path)

    print(f"Saved summary CSV to: {os.path.abspath(args.summary_csv)}")
    print(f"Saved summary JSON to: {os.path.abspath(args.summary_json)}")
    if plot_path:
        print(f"Saved plot to: {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    main()
