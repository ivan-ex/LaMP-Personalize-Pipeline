import argparse
import os
import shlex
import subprocess
import sys


DEFAULT_DRIFT_RANGES = ["0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5","0.5-0.6"]


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def format_ratio_folder(value):
    return format(float(value), ".12g")


def build_command(args, repo_root):
    result_dir = args.result_dir
    if result_dir is None:
        result_dir = os.path.join(
            repo_root,
            "result",
            "custom_drift_ctta",
            args.task_name,
            format_ratio_folder(args.profile_split_ratio),
            "detector_lora",
        )

    cmd = [
        args.python_bin,
        "run_custom_drift_ctta_experiments.py",
        "--ctta_script",
        "OPPU_CTTA_refactored.py",
        "--task_name",
        args.task_name,
        "--modes",
        "ctta",
        "--drift_detector",
        "lora",
        "--lora_drift_threshold",
        str(args.lora_drift_threshold),
        "--profile_split_ratio",
        str(args.profile_split_ratio),
        "--result_dir",
        result_dir,
        "--summary_csv",
        os.path.join(result_dir, "summary-detector_lora.csv"),
        "--summary_json",
        os.path.join(result_dir, "summary-detector_lora.json"),
        "--plot_path",
        os.path.join(result_dir, "summary_plot-detector_lora.png"),
        "--drift_ranges",
        *args.drift_ranges,
    ]

    optional_pairs = [
        ("--dataset_dir", args.dataset_dir),
        ("--dataset_prefix", args.dataset_prefix),
        ("--model_name", args.model_name),
        ("--task_lora", args.task_lora),
        ("--k", args.k),
        ("--batch_size", args.batch_size),
        ("--infer_batch_size", args.infer_batch_size),
        ("--max_epoch", args.max_epoch),
        ("--cut_off", args.cut_off),
        ("--ctta_threshold", args.ctta_threshold),
        ("--ctta_window_size", args.ctta_window_size),
        ("--ctta_update_min_examples", args.ctta_update_min_examples),
        ("--lora_drift_probe_size", args.lora_drift_probe_size),
        ("--lora_drift_probe_epochs", args.lora_drift_probe_epochs),
        ("--lora_drift_probe_lr", args.lora_drift_probe_lr),
        ("--semantic_device", args.semantic_device),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            cmd.extend([flag, str(value)])

    cmd.extend(["--ctta_cuda_visible_devices", args.ctta_cuda_visible_devices])
    cmd.extend(["--base_cuda_visible_devices", args.base_cuda_visible_devices])

    if args.add_profile:
        cmd.append("--add_profile")
    if args.load_in_4bit:
        cmd.append("--load_in_4bit")
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.dry_run:
        cmd.append("--dry_run")

    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CTTA experiments with LoRA parameter-space drift detection over selected drift user groups."
    )
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--task_name", default="movie_tagging")
    parser.add_argument("--drift_ranges", nargs="+", default=DEFAULT_DRIFT_RANGES)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--dataset_prefix", default="drift")
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
    parser.add_argument("--lora_drift_threshold", type=float, default=0.05)
    parser.add_argument("--lora_drift_probe_size", type=int, default=None)
    parser.add_argument("--lora_drift_probe_epochs", type=float, default=None)
    parser.add_argument("--lora_drift_probe_lr", type=float, default=None)
    parser.add_argument("--profile_split_ratio", type=float, default=0.5)
    parser.add_argument("--semantic_device", default=None)
    parser.add_argument("--ctta_cuda_visible_devices", default="0")
    parser.add_argument("--base_cuda_visible_devices", default="1")
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--add_profile", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    cmd = build_command(args, repo_root)
    print("Running:", quote_cmd(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
