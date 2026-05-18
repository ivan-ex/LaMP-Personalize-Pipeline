import torch
import transformers

from ctta.common import (
    cast_norm_modules_to_float32,
    compute_trainable_param_distance,
    normalize_text,
    restore_trainable_params,
    snapshot_trainable_params,
)
from ctta.constants import TASK_DRIFT_CONFIG


class DriftMixin:
    def resolve_semantic_device(self):
        requested_device = str(self.args.semantic_device).strip()
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

    def get_semantic_encoder(self):
        if self.semantic_encoder is None:
            from sentence_transformers import SentenceTransformer

            self.semantic_encoder = SentenceTransformer(
                self.args.semantic_model_path,
                device=self.resolve_semantic_device(),
            )
            self.semantic_encoder.eval()
        return self.semantic_encoder

    def l2_normalize(self, vector):
        norm = float(torch.linalg.norm(vector))
        if norm <= 1e-12:
            return vector
        return vector / norm

    def get_text_embedding(self, text):
        normalized_text = normalize_text(text)
        if not normalized_text:
            return None
        if normalized_text in self.semantic_embedding_cache:
            return self.semantic_embedding_cache[normalized_text]

        encoder = self.get_semantic_encoder()
        embedding = encoder.encode(
            normalized_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embedding = embedding.detach().cpu().to(torch.float32)
        self.semantic_embedding_cache[normalized_text] = embedding
        return embedding

    def build_entry_embedding_cache_key(self, entry):
        config = TASK_DRIFT_CONFIG[self.args.task_name]
        key_parts = [self.args.task_name]
        for field_config in config["semantic_fields"]:
            normalized_value = normalize_text(entry.get(field_config["field"], ""))
            weight = float(getattr(self.args, field_config["weight_arg"]))
            key_parts.append((field_config["field"], field_config["prefix"], weight, normalized_value))
        return tuple(key_parts)

    def get_entry_semantic_embedding(self, entry):
        cache_key = self.build_entry_embedding_cache_key(entry)
        if cache_key in self.semantic_entry_embedding_cache:
            return self.semantic_entry_embedding_cache[cache_key]

        config = TASK_DRIFT_CONFIG[self.args.task_name]
        weighted_embeddings = []
        total_weight = 0.0

        for field_config in config["semantic_fields"]:
            value = entry.get(field_config["field"])
            if value is None:
                continue

            normalized_value = normalize_text(value)
            if not normalized_value:
                continue

            embedding = self.get_text_embedding(f"{field_config['prefix']}: {normalized_value}")
            if embedding is None:
                continue

            weight = float(getattr(self.args, field_config["weight_arg"]))
            if weight <= 0:
                continue

            weighted_embeddings.append(embedding * weight)
            total_weight += weight

        if not weighted_embeddings or total_weight <= 0:
            self.semantic_entry_embedding_cache[cache_key] = None
            return None

        merged_embedding = torch.stack(weighted_embeddings, dim=0).sum(dim=0) / total_weight
        merged_embedding = self.l2_normalize(merged_embedding)
        self.semantic_entry_embedding_cache[cache_key] = merged_embedding
        return merged_embedding

    def build_window_embedding_from_entry_embeddings(self, entry_embeddings):
        valid_embeddings = [embedding for embedding in entry_embeddings if embedding is not None]
        if not valid_embeddings:
            return None
        window_embedding = torch.stack(valid_embeddings, dim=0).mean(dim=0)
        return self.l2_normalize(window_embedding)

    def embedding_cosine_distance(self, embedding_a, embedding_b):
        if embedding_a is None or embedding_b is None:
            return 0.0
        cosine_sim = float(torch.dot(embedding_a, embedding_b))
        cosine_sim = max(min(cosine_sim, 1.0), -1.0)
        return 1.0 - cosine_sim

    def compute_preference_drift_from_embeddings(self, entry_embeddings, current_end, window_size):
        if current_end < 2 * window_size:
            return None

        prev_embeddings = entry_embeddings[current_end - 2 * window_size: current_end - window_size]
        curr_embeddings = entry_embeddings[current_end - window_size: current_end]
        prev_embedding = self.build_window_embedding_from_entry_embeddings(prev_embeddings)
        curr_embedding = self.build_window_embedding_from_entry_embeddings(curr_embeddings)
        return self.embedding_cosine_distance(prev_embedding, curr_embedding)

    def semantic_history_is_missing(self, current_visible_len):
        return current_visible_len < 2 * self.args.ctta_window_size

    def drift_detector_uses_semantic(self):
        return self.args.drift_detector in ("semantic", "hybrid")

    def drift_detector_uses_lora(self):
        return self.args.drift_detector in ("lora", "hybrid")

    def should_probe_lora_drift(self, current_visible_len):
        if self.drift_detector_uses_lora():
            return True
        return self.args.history_missing_lora_detector and self.semantic_history_is_missing(current_visible_len)

    def effective_drift_score(self, semantic_score, lora_score):
        if self.args.drift_detector == "semantic" and not self.args.history_missing_lora_detector:
            return semantic_score
        if self.args.drift_detector == "semantic" and self.args.history_missing_lora_detector:
            return semantic_score if semantic_score is not None else lora_score
        if self.args.drift_detector == "lora":
            return lora_score

        normalized_scores = []
        if semantic_score is not None and self.args.ctta_threshold > 0:
            normalized_scores.append(semantic_score / self.args.ctta_threshold)
        if lora_score is not None and self.args.lora_drift_threshold > 0:
            normalized_scores.append(lora_score / self.args.lora_drift_threshold)
        return max(normalized_scores) if normalized_scores else None

    def should_trigger_drift(self, semantic_score, lora_score):
        semantic_triggered = (
            self.drift_detector_uses_semantic()
            and semantic_score is not None
            and semantic_score >= self.args.ctta_threshold
        )
        lora_triggered = (
            (self.drift_detector_uses_lora() or self.args.history_missing_lora_detector)
            and lora_score is not None
            and lora_score >= self.args.lora_drift_threshold
        )
        return semantic_triggered or lora_triggered, semantic_triggered, lora_triggered

    def probe_lora_drift(self, model, visible_entries, profile_prefix, current_visible_len):
        probe_size = self.args.lora_drift_probe_size or self.args.ctta_window_size
        if probe_size <= 0 or current_visible_len <= 0:
            return None

        start_idx = max(0, current_visible_len - probe_size)
        if current_visible_len <= start_idx:
            return None

        train_data = self.build_train_data_for_entries(
            visible_entries,
            profile_prefix=profile_prefix,
            start_idx=start_idx,
            end_idx=current_visible_len,
        )
        if not train_data:
            return None

        old_params = snapshot_trainable_params(model)
        train_dataset = self.build_tokenized_train_dataset(train_data, shuffle=True)
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_dataset,
            args=self.lora_probe_training_arguments,
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )

        try:
            cast_norm_modules_to_float32(trainer.model)
            model.config.use_cache = False
            trainer.train()
            new_params = snapshot_trainable_params(model)
            distance = compute_trainable_param_distance(old_params, new_params)
            distance.update(
                {
                    "probe_start": start_idx,
                    "probe_end": current_visible_len,
                    "probe_size": current_visible_len - start_idx,
                }
            )
            return distance
        finally:
            restore_trainable_params(model, old_params)
            del trainer
            del train_dataset
            self.cleanup_memory()
            model.eval()
            model.config.use_cache = True

    def select_replay_indices(self, profile_entries, start_idx):
        if (
            not self.args.ctta_anti_forgetting
            or self.args.ctta_replay_strategy == "none"
            or self.args.ctta_replay_size <= 0
            or start_idx <= 0
        ):
            return []

        pool_end = min(start_idx, len(profile_entries))
        replay_size = min(self.args.ctta_replay_size, pool_end)
        if replay_size <= 0:
            return []

        if self.args.ctta_replay_strategy == "recent":
            return list(range(pool_end - replay_size, pool_end))

        indexed_embeddings = []
        for idx in range(pool_end):
            embedding = self.get_entry_semantic_embedding(profile_entries[idx])
            if embedding is not None:
                indexed_embeddings.append((idx, embedding))

        if len(indexed_embeddings) <= replay_size:
            return [idx for idx, _ in indexed_embeddings] or list(range(pool_end - replay_size, pool_end))

        embeddings = torch.stack([embedding for _, embedding in indexed_embeddings], dim=0)
        centroid = self.l2_normalize(embeddings.mean(dim=0))
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
