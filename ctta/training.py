import transformers

from ctta.common import CTTAContinualTrainer, cast_norm_modules_to_float32, snapshot_trainable_params


class TrainingMixin:
    def train_on_visible_entries(self, model, visible_entries, profile_prefix, start_idx, end_idx, return_metadata=False):
        if end_idx <= start_idx:
            return (False, {}) if return_metadata else False

        train_data = self.build_train_data_for_entries(
            visible_entries,
            profile_prefix=profile_prefix,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        replay_indices = self.select_replay_indices(visible_entries, start_idx)
        replay_data = self.build_train_data_for_indices(
            visible_entries,
            replay_indices,
            profile_prefix=profile_prefix,
        )
        train_data = replay_data + train_data
        if not train_data:
            return (False, {}) if return_metadata else False

        train_dataset = self.build_tokenized_train_dataset(train_data, shuffle=True)
        old_params = None
        if self.args.ctta_anti_forgetting and start_idx > 0:
            old_params = snapshot_trainable_params(model)

        trainer = CTTAContinualTrainer(
            model=model,
            train_dataset=train_dataset,
            args=self.training_arguments,
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
            old_params=old_params,
            anchor_lambda=self.args.ctta_anchor_lambda if old_params else 0.0,
            lwf_lambda=self.args.ctta_lwf_lambda if old_params else 0.0,
            distill_temperature=self.args.ctta_distill_temperature,
        )

        cast_norm_modules_to_float32(trainer.model)
        model.config.use_cache = False
        trainer.train()
        del trainer
        del train_dataset
        self.cleanup_memory()

        metadata = {
            "anti_forgetting": bool(old_params),
            "replay_size": len(replay_indices),
            "replay_indices": replay_indices,
            "anchor_lambda": self.args.ctta_anchor_lambda if old_params else 0.0,
            "lwf_lambda": self.args.ctta_lwf_lambda if old_params else 0.0,
            "distill_temperature": self.args.ctta_distill_temperature,
        }
        return (True, metadata) if return_metadata else True

    def resolve_visible_prefix_length(self, profile_len, user_data=None):
        if profile_len <= 1:
            return profile_len

        if self.args.profile_split_mode == "metadata":
            visible_len = None
            if user_data is not None and user_data.get("profile_split_point") is not None:
                visible_len = int(user_data["profile_split_point"])
            if visible_len is None:
                visible_len = int(profile_len * self.args.profile_split_ratio)
        elif self.args.profile_split_mode == "count":
            if self.args.profile_split_count is None:
                raise ValueError("--profile_split_count must be set when --profile_split_mode=count")
            visible_len = self.args.profile_split_count
        else:
            visible_len = int(profile_len * self.args.profile_split_ratio)

        visible_len = max(1, visible_len)
        visible_len = min(profile_len - 1, visible_len)
        return visible_len

