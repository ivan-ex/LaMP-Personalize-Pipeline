import argparse
import os


def build_parser():
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
    parser.add_argument("--semantic_device", type=str, default="cuda:1", help="Device used by the MiniLM semantic encoder, e.g. cpu/cuda:0/cuda:1")
    parser.add_argument("--drift_tag_weight", type=float, default=3.0, help="Weight for tag/category style semantic fields")
    parser.add_argument("--drift_text_weight", type=float, default=1.0, help="Weight for description/text style semantic fields")
    parser.add_argument("--verbose_predictions", action="store_true", help="Print every profile/query prediction during inference")
    parser.add_argument("--ctta_threshold", type=float, default=0.2, help="Trigger LoRA update when drift score >= threshold")
    parser.add_argument("--ctta_window_size", type=int, default=8, help="Sliding window size for preference drift detection")
    parser.add_argument("--ctta_init_size", type=int, default=12, help="Warmup history size for initial user LoRA")
    parser.add_argument("--ctta_update_min_examples", type=int, default=4, help="Minimum newly arrived examples before another adaptation")
    parser.add_argument("--ctta_max_update_size", type=int, default=16, help="Use at most the latest N examples for one triggered adaptation")
    parser.add_argument(
        "--drift_detector",
        type=str,
        default="semantic",
        choices=["semantic", "lora", "hybrid"],
        help="semantic uses profile embeddings; lora uses parameter-space probe updates; hybrid triggers on either signal",
    )
    parser.add_argument(
        "--history_missing_lora_detector",
        action="store_true",
        help="Fallback to LoRA drift probing when explicit history is insufficient for semantic window comparison",
    )
    parser.add_argument(
        "--lora_drift_threshold",
        type=float,
        default=0.05,
        help="Trigger LoRA drift when relative trainable-parameter movement exceeds this threshold",
    )
    parser.add_argument(
        "--lora_drift_probe_size",
        type=int,
        default=None,
        help="Recent examples used for LoRA drift probing; defaults to --ctta_window_size",
    )
    parser.add_argument(
        "--lora_drift_probe_epochs",
        type=float,
        default=1.0,
        help="Epochs used by the temporary LoRA drift probe",
    )
    parser.add_argument(
        "--lora_drift_probe_lr",
        type=float,
        default=None,
        help="Learning rate for LoRA drift probing; defaults to the CTTA training learning rate",
    )
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
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def get_train_data_path(args):
    if args.train_data_path:
        return args.train_data_path
    return f"./data/{args.task_name}/user_top_100_history.json"


def infer_drift_tag(train_data_path):
    stem = os.path.splitext(os.path.basename(train_data_path))[0]
    return stem[len("drift_train_"):] if stem.startswith("drift_train_") else stem


def resolve_drift_tag(args, train_data_path):
    if args.drift_tag:
        return args.drift_tag
    return infer_drift_tag(train_data_path)


def build_ctta_run_name(args):
    output_tag = args.output_tag
    if args.adaptation_mode == "base" and output_tag == "ctta":
        output_tag = "base"
    run_name = f"output-OPPU-k{args.k}-{args.task_name}-{args.model_name.split('/')[-1]}-{output_tag}"
    if args.drift_detector != "semantic" or args.history_missing_lora_detector:
        run_name += f"-detector_{args.drift_detector}"
        if args.history_missing_lora_detector:
            run_name += "-missing_lora"
    if args.add_profile:
        run_name += "-profile"
    return run_name


def build_dataset_output_dirname(tag):
    normalized_tag = str(tag).strip()
    if normalized_tag.startswith("drift_"):
        return normalized_tag
    return f"drift_{normalized_tag}"
