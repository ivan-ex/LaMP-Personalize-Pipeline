import json
import os

from ctta.config import build_ctta_run_name, build_dataset_output_dirname
from utils import name2taskid


class OutputMixin:
    def output_metadata(self):
        return {
            "task": name2taskid[self.args.task_name],
            "model": self.args.model_name,
            "train_data_path": os.path.abspath(self.train_data_path),
            "drift_tag": self.drift_tag,
            "semantic_model_path": self.args.semantic_model_path,
            "semantic_device": self.resolve_semantic_device(),
            "drift_tag_weight": self.args.drift_tag_weight,
            "drift_text_weight": self.args.drift_text_weight,
            "adaptation_mode": self.args.adaptation_mode,
            "ctta_threshold": self.args.ctta_threshold,
            "ctta_window_size": self.args.ctta_window_size,
            "ctta_init_size": self.args.ctta_init_size,
            "ctta_update_min_examples": self.args.ctta_update_min_examples,
            "ctta_max_update_size": self.args.ctta_max_update_size,
            "drift_detector": self.args.drift_detector,
            "history_missing_lora_detector": self.args.history_missing_lora_detector,
            "lora_drift_threshold": self.args.lora_drift_threshold,
            "lora_drift_probe_size": self.args.lora_drift_probe_size or self.args.ctta_window_size,
            "lora_drift_probe_epochs": self.args.lora_drift_probe_epochs,
            "lora_drift_probe_lr": self.args.lora_drift_probe_lr or self.training_arguments.learning_rate,
            "ctta_anti_forgetting": self.args.ctta_anti_forgetting,
            "ctta_replay_size": self.args.ctta_replay_size,
            "ctta_replay_strategy": self.args.ctta_replay_strategy,
            "ctta_anchor_lambda": self.args.ctta_anchor_lambda,
            "ctta_lwf_lambda": self.args.ctta_lwf_lambda,
            "ctta_distill_temperature": self.args.ctta_distill_temperature,
            "profile_split_mode": self.args.profile_split_mode,
            "profile_split_ratio": self.args.profile_split_ratio,
            "profile_split_count": self.args.profile_split_count,
            "memory_size": self.args.memory_size,
            "load_in_8bit": self.args.load_in_8bit,
            "load_in_4bit": self.args.load_in_4bit,
            "quantized_loading": self.quantized_loading,
            "task_adapter_mode": self.task_adapter_mode,
        }

    def dump_prediction_file(self, path, predictions):
        payload = self.output_metadata()
        payload["golds"] = predictions
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)

    def write_outputs(self, pred_all_profile, pred_all_query, ctta_logs):
        output_dir = os.path.join("./output", self.args.task_name, build_dataset_output_dirname(self.drift_tag))
        os.makedirs(output_dir, exist_ok=True)
        output_name = build_ctta_run_name(self.args)

        if pred_all_profile:
            self.dump_prediction_file(os.path.join(output_dir, f"{output_name}-heldout_profile.json"), pred_all_profile)

        if pred_all_query:
            self.dump_prediction_file(os.path.join(output_dir, f"{output_name}-query.json"), pred_all_query)

        if pred_all_profile:
            summary = {
                "num_examples": len(pred_all_profile),
                "num_correct": sum(1 for item in pred_all_profile if item["correct"]),
            }
            summary["accuracy"] = (summary["num_correct"] / summary["num_examples"]) if summary["num_examples"] else 0.0
            summary.update(
                {
                    "task_name": self.args.task_name,
                    "train_data_path": os.path.abspath(self.train_data_path),
                    "drift_tag": self.drift_tag,
                    "semantic_model_path": self.args.semantic_model_path,
                    "semantic_device": self.resolve_semantic_device(),
                    "drift_tag_weight": self.args.drift_tag_weight,
                    "drift_text_weight": self.args.drift_text_weight,
                    "adaptation_mode": self.args.adaptation_mode,
                    "ctta_threshold": self.args.ctta_threshold,
                    "ctta_window_size": self.args.ctta_window_size,
                    "drift_detector": self.args.drift_detector,
                    "history_missing_lora_detector": self.args.history_missing_lora_detector,
                    "lora_drift_threshold": self.args.lora_drift_threshold,
                    "lora_drift_probe_size": self.args.lora_drift_probe_size or self.args.ctta_window_size,
                    "lora_drift_probe_epochs": self.args.lora_drift_probe_epochs,
                    "lora_drift_probe_lr": self.args.lora_drift_probe_lr or self.training_arguments.learning_rate,
                    "ctta_anti_forgetting": self.args.ctta_anti_forgetting,
                    "ctta_replay_size": self.args.ctta_replay_size,
                    "ctta_replay_strategy": self.args.ctta_replay_strategy,
                    "ctta_anchor_lambda": self.args.ctta_anchor_lambda,
                    "ctta_lwf_lambda": self.args.ctta_lwf_lambda,
                    "ctta_distill_temperature": self.args.ctta_distill_temperature,
                    "profile_split_mode": self.args.profile_split_mode,
                    "profile_split_ratio": self.args.profile_split_ratio,
                    "profile_split_count": self.args.profile_split_count,
                    "memory_size": self.args.memory_size,
                    "load_in_8bit": self.args.load_in_8bit,
                    "load_in_4bit": self.args.load_in_4bit,
                    "quantized_loading": self.quantized_loading,
                    "task_adapter_mode": self.task_adapter_mode,
                    "num_query_predictions": len(pred_all_query),
                }
            )
            with open(os.path.join(output_dir, f"{output_name}-heldout_profile-summary.json"), "w") as f:
                json.dump(summary, f, indent=4)

        log_path = os.path.join(output_dir, f"{output_name}-drift_log.json")
        with open(log_path, "w") as f:
            json.dump(ctta_logs, f, indent=4)
