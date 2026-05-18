import re
from contextlib import nullcontext

import torch
from tqdm import tqdm

from ctta.common import normalize_text
from ctta.constants import DISCRETE_LABELS, TASK_LABEL_FIELD
from utils import extract_citation_title, extract_option, get_first_k_tokens, split_batch


class InferenceMixin:
    def build_profile_eval_prompt(self, entry, revealed_entries, profile_prefix=None):
        q = {key: get_first_k_tokens(value, 768) for key, value in entry.items()}
        prompt = self.prompt_template[self.args.task_name]["OPPU_input"].format(**q)

        if self.args.k > 0 and revealed_entries and self.format_flag:
            visible_history_list = []
            memory_history = self.get_memory_slice(revealed_entries)
            for history_item in memory_history:
                visible_history_list.append(
                    {key: get_first_k_tokens(value, 368) for key, value in history_item.items()}
                )

            history_list = [
                self.prompt_template[self.args.task_name]["retrieval_history"].format(**p)
                for p in visible_history_list
            ]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = self.get_bm25_class()(tokenized_corpus)

            tokenized_query = self.prompt_template[self.args.task_name]["retrieval_query"].format(**q).split(" ")
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=min(self.args.k, len(history_list)))
            prompt = "".join(retrieved_history) + "\n" + prompt

        if self.args.add_profile and self.format_flag and profile_prefix:
            prompt = profile_prefix + "\n" + prompt

        return prompt

    def get_model_device_type(self, model):
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                return "cpu"
        return getattr(device, "type", str(device).split(":", 1)[0])

    def inference_autocast_context(self, model):
        if self.get_model_device_type(model) == "cuda":
            return torch.autocast(device_type="cuda")
        return nullcontext()

    def build_generation_kwargs(self, max_new_tokens=None):
        generation_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.args.task_name in DISCRETE_LABELS:
            generation_kwargs.update(
                {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": max_new_tokens or 4,
                    "use_cache": True,
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                }
            )
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "top_k": 10,
                    "temperature": self.args.temperature,
                    "top_p": 0.9,
                    "max_new_tokens": max_new_tokens or 64,
                    "use_cache": True,
                }
            )
        return generation_kwargs

    def decode_generated_suffix(self, outputs, input_length):
        generated_tokens = outputs[:, input_length:]
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def normalize_prediction_text(self, text):
        text = normalize_text(text)
        text = text.split("\n")[0].strip()
        return text

    def canonicalize_discrete_prediction(self, text):
        normalized = self.normalize_prediction_text(text)
        if self.args.task_name not in DISCRETE_LABELS:
            return normalized

        labels = [normalize_text(label) for label in DISCRETE_LABELS[self.args.task_name]]
        label_set = set(labels)
        if normalized in label_set:
            return normalized

        for label in sorted(labels, key=len, reverse=True):
            if label in normalized:
                return label

        compact = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        for label in labels:
            label_compact = re.sub(r"[^a-z0-9]+", " ", label).strip()
            if compact == label_compact or label_compact in compact:
                return label

        return normalized

    def generate_prediction(self, model, prompt):
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.cut_off,
            return_token_type_ids=False,
        )
        inputs = inputs.to(model.device)
        input_length = inputs["input_ids"].shape[1]
        generation_kwargs = self.build_generation_kwargs()

        with torch.inference_mode():
            with self.inference_autocast_context(model):
                outputs = model.generate(**inputs, **generation_kwargs)

        result = self.decode_generated_suffix(outputs, input_length)[0].strip()
        del inputs
        del outputs
        return result

    def supports_profile_stream_evaluation(self):
        return self.args.task_name != "tweet_paraphrase"

    def evaluate_profile_stream(self, model, visible_entries, hidden_entries, profile_prefix=None, evaluate_predictions=True):
        label_field = TASK_LABEL_FIELD[self.args.task_name]
        predictions = []
        revealed_entries = [dict(item) for item in self.get_memory_slice(visible_entries)]
        revealed_embeddings = []
        if self.args.adaptation_mode == "ctta" and self.drift_detector_uses_semantic():
            revealed_embeddings = [self.get_entry_semantic_embedding(item) for item in revealed_entries]
        steps_since_last_adapt = 0
        events = []
        trained_segments = []

        for hidden_idx, entry in enumerate(hidden_entries):
            if evaluate_predictions:
                test_prompt = self.build_profile_eval_prompt(
                    entry,
                    revealed_entries,
                    profile_prefix=profile_prefix,
                )

                raw_output = self.generate_prediction(model, test_prompt)
                pred_output = self.canonicalize_discrete_prediction(raw_output)
                gold_output = self.normalize_prediction_text(entry.get(label_field, ""))
                is_correct = pred_output == gold_output

                predictions.append(
                    {
                        "id": entry.get("id", f"heldout_{hidden_idx}"),
                        "output": raw_output,
                        "prediction_normalized": pred_output,
                        "gold": entry.get(label_field, ""),
                        "gold_normalized": gold_output,
                        "correct": is_correct,
                        "stream_index": hidden_idx,
                    }
                )
                if self.args.verbose_predictions:
                    print(f"[heldout_profile] pred={pred_output} gold={gold_output} correct={is_correct}")

            revealed_entries.append(dict(entry))
            revealed_entries = self.get_memory_slice(revealed_entries)

            if self.args.adaptation_mode == "base":
                continue

            steps_since_last_adapt += 1
            current_visible_len = len(revealed_entries)

            semantic_score = None
            if self.drift_detector_uses_semantic():
                revealed_embeddings.append(self.get_entry_semantic_embedding(entry))
                revealed_embeddings = self.get_memory_slice(revealed_embeddings)
                semantic_score = self.compute_preference_drift_from_embeddings(
                    revealed_embeddings,
                    len(revealed_embeddings),
                    self.args.ctta_window_size,
                )

            lora_metrics = None
            lora_score = None
            if (
                steps_since_last_adapt >= self.args.ctta_update_min_examples
                and self.should_probe_lora_drift(current_visible_len)
            ):
                lora_metrics = self.probe_lora_drift(
                    model,
                    revealed_entries,
                    profile_prefix,
                    current_visible_len,
                )
                if lora_metrics is not None:
                    lora_score = lora_metrics["relative_l2"]

            trigger_drift, semantic_triggered, lora_triggered = self.should_trigger_drift(semantic_score, lora_score)
            if trigger_drift and steps_since_last_adapt >= self.args.ctta_update_min_examples:
                segment_start = max(0, current_visible_len - self.args.ctta_max_update_size)
                drift_score = self.effective_drift_score(semantic_score, lora_score)
                event = {
                    "trigger_end": current_visible_len,
                    "drift_score": round(float(drift_score), 6) if drift_score is not None else None,
                    "drift_detector": self.args.drift_detector,
                    "history_missing_lora_detector": self.args.history_missing_lora_detector,
                    "semantic_history_missing": self.semantic_history_is_missing(current_visible_len),
                    "semantic_drift_score": round(float(semantic_score), 6) if semantic_score is not None else None,
                    "lora_drift_score": round(float(lora_score), 6) if lora_score is not None else None,
                    "semantic_triggered": semantic_triggered,
                    "lora_triggered": lora_triggered,
                    "segment_start": segment_start,
                    "segment_end": current_visible_len,
                    "segment_size": current_visible_len - segment_start,
                    "update_applied": self.args.adaptation_mode == "ctta",
                }
                if lora_metrics is not None:
                    event["lora_probe"] = {
                        "relative_l2": round(float(lora_metrics["relative_l2"]), 6),
                        "l2": round(float(lora_metrics["l2"]), 6),
                        "mean_abs_delta": round(float(lora_metrics["mean_abs_delta"]), 8),
                        "cosine_distance": round(float(lora_metrics["cosine_distance"]), 6),
                        "param_count": int(lora_metrics["param_count"]),
                        "probe_start": int(lora_metrics["probe_start"]),
                        "probe_end": int(lora_metrics["probe_end"]),
                        "probe_size": int(lora_metrics["probe_size"]),
                    }

                did_train, update_metadata = self.train_on_visible_entries(
                    model,
                    revealed_entries,
                    profile_prefix,
                    start_idx=segment_start,
                    end_idx=current_visible_len,
                    return_metadata=True,
                )
                if did_train:
                    event.update(update_metadata)
                    events.append(event)
                    trained_segments.append({"start": segment_start, "end": current_visible_len, "reason": "drift"})
                    steps_since_last_adapt = 0
                    model.gradient_checkpointing_disable()
                    model.eval()
                    model.config.use_cache = True

        return predictions, events, trained_segments

    def run_inference_for_query_field(self, model, user_data, query_field, profile_prefix=None):
        if self.args.k > 0:
            visible_history_list = []
            for item in user_data["profile"]:
                visible_history_list.append(
                    {key: get_first_k_tokens(value, 368) for key, value in item.items()}
                )

            history_list = [
                self.prompt_template[self.args.task_name]["retrieval_history"].format(**p)
                for p in visible_history_list
            ]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = self.get_bm25_class()(tokenized_corpus)
        else:
            history_list = None
            bm25 = None

        test_question_list = []
        question_id_list = []

        for q in user_data.get(query_field, []):
            if self.args.task_name == "citation":
                test_question = q["input"]
                test_article = extract_citation_title(test_question)
                option1 = extract_option(test_question, 1)
                option2 = extract_option(test_question, 2)
                test_prompt = self.prompt_template[self.args.task_name]["prompt"].format(test_article, option1, option2)
            else:
                test_question = q["input"]
                test_article = self.extract_article(test_question)
                test_prompt = self.prompt_template[self.args.task_name]["prompt"].format(test_article)

            if self.args.k > 0:
                tokenized_query = self.prompt_template[self.args.task_name]["retrieval_query_wokey"].format(test_article).split(" ")
                retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=self.args.k)
                history_string = "".join(retrieved_history)
                test_prompt = history_string + "\n" + test_prompt

            if self.args.add_profile and profile_prefix:
                test_prompt = profile_prefix + "\n" + test_prompt

            test_question_list.append(test_prompt)
            question_id_list.append(q["id"])

        out_list = []
        query_batches = split_batch(test_question_list, self.infer_batch_size)
        query_max_new_tokens = 4 if self.args.task_name in DISCRETE_LABELS else 200
        generation_kwargs = self.build_generation_kwargs(max_new_tokens=query_max_new_tokens)
        for batch in tqdm(query_batches, total=len(query_batches), leave=False):
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.cut_off,
                return_token_type_ids=False,
            )
            inputs = inputs.to(model.device)
            input_length = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                with self.inference_autocast_context(model):
                    outputs = model.generate(**inputs, **generation_kwargs)

            out_sentence = self.decode_generated_suffix(outputs, input_length)
            out_list += out_sentence
            del inputs
            del outputs

        predictions = []
        for idx, decoded in enumerate(out_list):
            output = decoded.strip()
            predictions.append({"id": question_id_list[idx], "output": output})
            if self.args.verbose_predictions:
                print(f"[{query_field}] {output}")

        return predictions
