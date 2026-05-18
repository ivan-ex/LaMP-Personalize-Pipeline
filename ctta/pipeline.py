import os

from tqdm import tqdm


class PipelineMixin:
    def run_user(self, user_idx, user_data):
        profile_entries = user_data["profile"]
        profile_prefix = self.test_profile[user_idx]["output"] if self.args.add_profile else None
        model = self.create_user_lora_model()

        profile_len = len(profile_entries)
        visible_prefix_len = self.resolve_visible_prefix_length(profile_len, user_data=user_data)
        visible_entries = profile_entries[:visible_prefix_len]
        hidden_entries = profile_entries[visible_prefix_len:]
        warmup_end = min(len(visible_entries), max(self.args.ctta_init_size, self.args.ctta_window_size))
        user_log = {
            "user_index": user_idx,
            "user_id": user_data.get("user_id"),
            "profile_length": profile_len,
            "visible_prefix_len": visible_prefix_len,
            "hidden_suffix_len": len(hidden_entries),
            "memory_size": self.args.memory_size,
            "warmup_end": warmup_end,
            "adaptation_mode": self.args.adaptation_mode,
            "drift_threshold": self.args.ctta_threshold,
            "window_size": self.args.ctta_window_size,
            "drift_detector": self.args.drift_detector,
            "history_missing_lora_detector": self.args.history_missing_lora_detector,
            "lora_drift_threshold": self.args.lora_drift_threshold,
            "lora_drift_probe_size": self.args.lora_drift_probe_size or self.args.ctta_window_size,
            "anti_forgetting": self.args.ctta_anti_forgetting,
            "replay_size": self.args.ctta_replay_size,
            "replay_strategy": self.args.ctta_replay_strategy,
            "anchor_lambda": self.args.ctta_anchor_lambda,
            "lwf_lambda": self.args.ctta_lwf_lambda,
            "events": [],
        }

        trained_segments = []
        if warmup_end > 0:
            did_train = self.train_on_visible_entries(
                model,
                visible_entries,
                profile_prefix,
                start_idx=0,
                end_idx=warmup_end,
            )
            if did_train:
                trained_segments.append({"start": 0, "end": warmup_end, "reason": "warmup"})

        model.gradient_checkpointing_disable()
        model.eval()
        model.config.use_cache = True

        profile_predictions, drift_events, stream_segments = self.evaluate_profile_stream(
            model,
            visible_entries=visible_entries,
            hidden_entries=hidden_entries,
            profile_prefix=profile_prefix,
            evaluate_predictions=self.supports_profile_stream_evaluation(),
        )

        user_log["events"].extend(drift_events)
        user_log["trained_segments"] = trained_segments + stream_segments
        user_log["num_profile_predictions"] = len(profile_predictions)
        user_log["num_correct"] = sum(1 for item in profile_predictions if item["correct"])

        query_predictions = []
        if "query" in user_data:
            query_predictions = self.run_inference_for_query_field(
                model,
                user_data,
                "query",
                profile_prefix=profile_prefix,
            )

        if self.args.save_user_ckpt:
            ckpt_dir = os.path.join(".", "ckpt", self.args.task_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_output_name = os.path.join(
                ckpt_dir,
                f"k{self.args.k}-{self.args.task_name}-{self.args.model_name.split('/')[-1]}-{self.args.output_tag}-user_{user_idx}",
            )
            model.save_pretrained(ckpt_output_name)

        self.base_model = self.unload_user_lora_model(model)
        del model
        self.cleanup_memory()
        return profile_predictions, query_predictions, user_log

    def run(self):
        self.load_data()
        self.load_model()

        pred_all_profile = []
        pred_all_query = []
        ctta_logs = []

        for user_idx in tqdm(range(len(self.test_data))):
            profile_predictions, query_predictions, user_log = self.run_user(user_idx, self.test_data[user_idx])
            pred_all_profile.extend(profile_predictions)
            pred_all_query.extend(query_predictions)
            ctta_logs.append(user_log)

        self.write_outputs(pred_all_profile, pred_all_query, ctta_logs)
